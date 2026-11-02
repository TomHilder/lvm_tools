"""tile.py - Tile classes for LVM data processing."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import astropy.units as u  # type: ignore[import]
import dask.array as da
import numpy as np
from astropy.io import fits  # type: ignore[import]
from astropy.io.fits import FITS_rec, HDUList  # type: ignore[import]
from astropy.units import Unit  # type: ignore[import]
from numpy.typing import NDArray
from xarray import Dataset, concat

from lvm_tools.data.coordinates import get_mjd
from lvm_tools.data.helper import daskify_native, numpyfy_native, summarize_with_units

# Conversions between FWHM and Gaussian sigma
SIGMA_TO_FWHM: float = 2.0 * np.sqrt(2.0 * np.log(2))
"""Ratio of a Gaussian's FWHM to its standard deviation, `2 * sqrt(2 * ln 2)` (about 2.355)."""
FWHM_TO_SIGMA: float = 1.0 / SIGMA_TO_FWHM
"""Ratio of a Gaussian's standard deviation to its FWHM, `1 / SIGMA_TO_FWHM` (about 0.4247)."""

# Physical units for the data
FLUX_UNIT: Unit = u.erg * u.cm**-2 * u.s**-1 * u.angstrom**-1
"""Unit of the flux: erg cm⁻² s⁻¹ Å⁻¹. The inverse variance is in `FLUX_UNIT**-2`."""
SPECTRAL_UNIT: Unit = u.angstrom
"""Unit of the wavelengths and of the LSF width: Angstrom."""
SPATIAL_UNIT: Unit = u.degree
"""Unit of the spaxel coordinates (RA and Dec): degree."""

# Dask chunks for the (fibre, wavelength) cubes: all fibres, 512 pixels in wavelength.
# Chunking along wavelength means selecting a wavelength window only reads those
# chunks; with "auto" a whole cube (~100 MB) is a single chunk.
CHUNKSIZE: tuple[int, int] = (-1, 512)
"""Dask chunks of the (fibre, wavelength) cubes read by `LVMTile.from_file`: all fibres, and
512 pixels in wavelength."""


def get_science_inds(slitmap: FITS_rec) -> NDArray:
    """Indices of the science fibres in a DRP slitmap.

    Args:
        slitmap: The slitmap table (the last HDU of a DRP file).

    Returns:
        The row indices whose `targettype` is `"science"`.
    """
    return np.where(slitmap.field("targettype") == "science")[0]


@dataclass(frozen=True)
class LVMTileMeta:
    """Metadata of one LVM exposure, read from the primary header of its DRP file.

    Attributes:
        filename: Name of the DRP FITS file (without its directory).
        tile_id: LVM tile identifier, from the `OBJECT` header keyword. An `int` when the
            keyword holds a number (either `"...=<id>"` or a bare number), otherwise the
            keyword's string.
        exp_num: Exposure number, from the `EXPOSURE` keyword.
        drp_ver: Version of the Data Reduction Pipeline, from the `DRPVER` keyword.
    """

    filename: str
    tile_id: int | str
    exp_num: int
    drp_ver: str


@dataclass(frozen=True)
class LVMTile:
    """One LVM exposure: its science-fibre spectra and metadata, read lazily from a DRP file.

    Create one with `LVMTile.from_file`. The cubes are Dask arrays, so nothing is read from
    disk until values are computed, and selecting part of the data (for example a wavelength
    window) only reads that part.

    Attributes:
        data: The exposure as an `xarray.Dataset`.

            Data variables, each of shape `(1, n_spaxels, n_wavelengths)` with dimensions
            `(tile, spaxel, wavelength)`, held as Dask arrays:

            - `flux`: flux, in erg cm⁻² s⁻¹ Å⁻¹ (`FLUX_UNIT`).
            - `i_var`: inverse variance of the flux, in `FLUX_UNIT**-2`.
            - `lsf_sigma`: Gaussian standard deviation of the line spread function, in
              Angstrom (converted from the FWHM stored in the file).
            - `mask`: the DRP pixel mask; 1 marks a bad pixel.

            Coordinates (NumPy arrays):

            - `tile`: the exposure number (length 1).
            - `spaxel`: spaxel index, `0 .. n_spaxels - 1`.
            - `wavelength`: wavelength grid, in Angstrom.
            - `mjd`: Modified Julian Date of the exposure midpoint, along `tile`.
            - `ra`, `dec`: right ascension and declination of each spaxel, in degrees,
              shape `(1, n_spaxels)`.
            - `fibre_id`: fibre identifier from the slitmap, shape `(1, n_spaxels)`.
            - `ifu_label`: IFU bundle label (`"Sci1"`, `"Sci2"` or `"Sci3"`).
            - `fibre_status`: fibre status flag from the slitmap; 0 marks a good fibre.
        meta: The exposure's metadata.
    """

    data: Dataset
    meta: LVMTileMeta

    @classmethod
    def from_file(cls, drp_file: Path | str) -> LVMTile:
        """Read an exposure from a DRP FITS file.

        Only the science fibres are kept. The flux, inverse variance, mask and LSF cubes are
        wrapped lazily as Dask arrays (chunked by `CHUNKSIZE`); the wavelength grid and the
        per-fibre coordinates are read immediately.

        Args:
            drp_file: Path to the DRP FITS file.

        Returns:
            The tile.

        Raises:
            FileNotFoundError: If `drp_file` does not exist.
            ValueError: If the primary header lacks the `INTSTART` and `INTEND` keywords.

        Example:
            ```python
            from pathlib import Path
            from lvm_tools import LVMTile

            tile = LVMTile.from_file(Path("/data/lvm-drp-12345-00001.fits"))
            ```
        """
        file = Path(drp_file)
        if not file.exists():
            raise FileNotFoundError("Could not find DRP file.")

        with fits.open(file, memmap=True) as hdul:
            tile_id, exp_num, drp_ver, mjd = cls.get_metadata(hdul)
            (flux, i_var, mask, lsf), (wave, ra, dec, fibre_id, fibre_status, ifu_label) = (
                cls.get_science_data(hdul)
            )

        # Conver the lsf from full width at half maximum (FWHM) to sigma
        lsf *= FWHM_TO_SIGMA

        # Common dimensions for cube data
        pixel_dims = ("tile", "spaxel", "wavelength")
        spaxel_dims = ("tile", "spaxel")

        # Assemble data into xarray Dataset, containing both dask arrays and numpy arrays
        data = Dataset(
            data_vars={
                "flux": (pixel_dims, flux[None, :, :], {"units": str(FLUX_UNIT)}),
                "i_var": (pixel_dims, i_var[None, :, :], {"units": str(FLUX_UNIT**-2)}),
                "lsf_sigma": (pixel_dims, lsf[None, :, :], {"units": str(SPECTRAL_UNIT)}),
                "mask": (pixel_dims, mask[None, :, :]),
            },
            coords={
                # Main dimensions/coordinates
                "tile": ("tile", [exp_num]),
                "spaxel": ("spaxel", np.arange(len(fibre_id))),
                "wavelength": ("wavelength", wave, {"units": str(SPECTRAL_UNIT)}),
                # More coordinates
                "mjd": ("tile", [mjd], {"units": "day"}),
                "ra": (spaxel_dims, ra[None, :], {"units": str(SPATIAL_UNIT)}),
                "dec": (spaxel_dims, dec[None, :], {"units": str(SPATIAL_UNIT)}),
                "fibre_id": (spaxel_dims, fibre_id[None, :]),
                "ifu_label": (spaxel_dims, ifu_label[None, :]),
                "fibre_status": (spaxel_dims, fibre_status[None, :]),
            },
        )

        # Assemble metadata
        meta = LVMTileMeta(
            filename=file.name,
            tile_id=tile_id,
            exp_num=exp_num,
            drp_ver=drp_ver,
        )

        return cls(data=data, meta=meta)

    def __repr__(self) -> str:
        prefix = f"LVMTile ({hex(id(self))}):"
        prefix += f"\n    Filename:        {self.meta.filename}"
        prefix += f"\n    Exposure:        {self.meta.exp_num}"
        prefix += f"\n    DRP version:     {self.meta.drp_ver}"
        prefix += f"\n    Tile ID:         {self.meta.tile_id}"
        return f"{prefix}\n{summarize_with_units(self.data)}"

    @staticmethod
    def get_science_data(drp_hdulist: HDUList) -> tuple[tuple, tuple]:
        """Read the science fibres' cubes and coordinates from an open DRP file.

        Args:
            drp_hdulist: The open DRP file.

        Returns:
            Two tuples. The first holds the lazy Dask cubes `(flux, i_var, mask, lsf)`, each
                of shape `(n_spaxels, n_wavelengths)`, with the LSF still as a FWHM. The
                second holds the NumPy arrays `(wave, ra, dec, fibre_id, fibre_status,
                ifu_label)`, all in native byte order.
        """
        slitmap = drp_hdulist[-1].data
        science_inds = get_science_inds(slitmap)
        # Lazily load cubes
        flux: da.Array = daskify_native(drp_hdulist[1].data, CHUNKSIZE)[science_inds, :]
        i_var: da.Array = daskify_native(drp_hdulist[2].data, CHUNKSIZE)[science_inds, :]
        mask: da.Array = daskify_native(drp_hdulist[3].data, CHUNKSIZE)[science_inds, :]
        lsf: da.Array = daskify_native(drp_hdulist[5].data, CHUNKSIZE)[science_inds, :]
        # Eagerly coordinates
        wave: NDArray = numpyfy_native(drp_hdulist[4].data)
        ra: NDArray = numpyfy_native((slitmap["ra"])[science_inds])
        dec: NDArray = numpyfy_native((slitmap["dec"])[science_inds])
        fibre_id: NDArray = numpyfy_native((slitmap["fiberid"])[science_inds])
        fibre_status: NDArray = numpyfy_native((slitmap["fibstatus"])[science_inds])
        ifu_label: NDArray = numpyfy_native((slitmap["ifulabel"])[science_inds])
        return (flux, i_var, mask, lsf), (wave, ra, dec, fibre_id, fibre_status, ifu_label)

    @staticmethod
    def get_metadata(drp_hdulist: HDUList) -> tuple[int | str, int, str, float]:
        """Read the metadata of an exposure from the primary header of an open DRP file.

        Args:
            drp_hdulist: The open DRP file.

        Returns:
            The four values `(tile_id, exp_num, drp_ver, mjd)`: the tile identifier (an
                `int`, or the `OBJECT` keyword's string if it isn't a number), the exposure
                number, the DRP version, and the MJD of the exposure midpoint.

        Raises:
            ValueError: If the header lacks the `INTSTART` and `INTEND` keywords.
        """
        try:
            tile_id = int(drp_hdulist[0].header["OBJECT"].split("=")[1])
        except IndexError:
            try:
                tile_id = int(drp_hdulist[0].header["OBJECT"])
            except ValueError:
                tile_id = str(drp_hdulist[0].header["OBJECT"])
        exp_num = int(drp_hdulist[0].header["EXPOSURE"])
        drp_ver = str(drp_hdulist[0].header["DRPVER"])
        mjd = float(get_mjd(drp_hdulist[0].header))
        return tile_id, exp_num, drp_ver, mjd


@dataclass(frozen=True)
class LVMTileCollection:
    """Several LVM exposures, combined along the `tile` dimension.

    Create one with `LVMTileCollection.from_tiles`. It can be used wherever an `LVMTile` can
    (see `LVMTileLike`).

    Attributes:
        data: The combined data. It has the same variables and coordinates as `LVMTile.data`,
            with the `tile` dimension as long as the number of exposures; the `tile`
            coordinate holds their exposure numbers.
        meta: The metadata of each exposure, keyed by exposure number.
    """

    data: Dataset
    meta: Mapping[int, LVMTileMeta]

    @classmethod
    def from_tiles(cls, tiles: list[LVMTile]) -> LVMTileCollection:
        """Combine tiles into a collection.

        The tiles' datasets are concatenated along `tile` with `xarray.concat`, which aligns
        the other coordinates (e.g. `spaxel` and `wavelength`) between them.

        Args:
            tiles: The tiles to combine.

        Returns:
            The collection.

        Example:
            ```python
            from lvm_tools import LVMTile, LVMTileCollection

            tiles = [LVMTile.from_file(p) for p in paths]
            collection = LVMTileCollection.from_tiles(tiles)
            print(f"Combined shape: {collection.data['flux'].shape}")
            ```
        """
        # Concatenate tile datasets along the 'tile' dimension
        combined_data = concat([tile.data for tile in tiles], dim="tile")

        # Construct metadata dictionary keyed by exposure number
        meta_dict = {tile.meta.exp_num: tile.meta for tile in tiles}

        return cls(data=combined_data, meta=meta_dict)

    def __repr__(self) -> str:
        prefix = f"LVMTileCollection ({hex(id(self))}):"
        prefix += f"\n    Tiles:            {len(self.meta)}"
        prefix += f"\n    Exposures:        { {meta.exp_num for meta in self.meta.values()} }"
        prefix += f"\n    DRP versions:     { {meta.drp_ver for meta in self.meta.values()} }"
        prefix += f"\n    Tile IDs:         { {meta.tile_id for meta in self.meta.values()} }"
        return f"{prefix}\n{summarize_with_units(self.data)}"


LVMTileLike = LVMTile | LVMTileCollection
"""Either an `LVMTile` or an `LVMTileCollection`. Functions that take one work the same way on a
single exposure and on several."""
