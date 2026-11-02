"""fit_data.py - FitData class for holding data ready to be fitted."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import astropy.units as u
import jax.numpy as jnp
import numpy as np
from astropy.coordinates import SkyCoord, SkyOffsetFrame
from jax.numpy import pi as π
from xarray import DataArray, Dataset

from lvm_tools.physical_properties.barycentric_corr import get_v_barycentric

if TYPE_CHECKING:
    from jaxtyping import Array as JaxArray
    from numpy.typing import ArrayLike
    from spectracles.model.data import SpatialDataLVM

try:
    from spectracles.model.data import SpatialDataLVM
except Exception as e:  # noqa: BLE001 (any failure importing spectracles, e.g. a broken jax-finufft build)
    _import_err = e
    msg = (
        "Could not import SpatialDataLVM from spectracles.model.data. "
        "Install the optional dependency: pip install spectracles"
    )

    class SpatialDataLVM:  # stub preserves the API shape
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(msg) from _import_err

    import warnings

    warnings.warn(msg, ImportWarning)


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"FitData.{old} (coordinates scaled into [-π, π]) is deprecated: use {new}, with a "
        "spectracles Domain for physical length scales.",
        DeprecationWarning,
        stacklevel=3,
    )


def to_π_domain(x) -> ArrayLike:
    """Map values from [0, 1] onto [-π, π]: `x * 2π - π`.

    Args:
        x (ArrayLike): Values in [0, 1].

    Returns:
        The values in [-π, π].
    """
    # return x * 2 * π - π
    return x * 2 * np.pi - np.pi


def from_π_domain(x) -> ArrayLike:
    """Map values from [-π, π] back onto [0, 1]: `(x + π) / 2π`. The inverse of `to_π_domain`.

    Args:
        x (ArrayLike): Values in [-π, π].

    Returns:
        The values in [0, 1].
    """
    return (x + π) / (2 * π)


def to_jax_array(arr: DataArray, dtype=np.float64) -> JaxArray:
    """Convert an array (e.g. an xarray `DataArray` or its values) to a JAX array.

    JAX's warning that a 64-bit dtype was truncated (when JAX's 64-bit mode is off) is
    silenced, so the result is 32-bit unless that mode is on.

    Args:
        arr: The array.
        dtype (numpy.dtype): The dtype to request.

    Returns:
        The JAX array.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        return jnp.array(arr, dtype=dtype)


@dataclass(frozen=True)
class FitData:
    """LVM data ready for model fitting: normalised arrays, positions and indices as JAX arrays.

    Usually made by `FitDataBuilder.build`. The data are held as a flattened xarray dataset
    plus the normalisation functions, and each property computes its JAX array from them
    when accessed.

    The spectral arrays (`flux`, `i_var`, `u_flux`, `lsf_σ`, `mask`) have shape
    `(n_wavelengths, n_spaxels)`, and the per-spaxel arrays (`sky_x`, `sky_y`, the indices,
    `mjd`, `v_bary`) have shape `(n_spaxels,)`, where the spaxels of every tile are counted
    together. Floating-point arrays are 32-bit unless JAX's 64-bit mode is on. Pixels that
    were clipped or filtered out are NaN in the processed data; `mask` marks the others,
    and the spectral properties replace the NaNs with fill values so they can be used in
    a likelihood together with `mask`.

    Sky positions are given as `sky_x` and `sky_y`: offsets in arcsec east and north of
    `sky_centre`, on the tangent plane. `sky_data` packages them for spectracles. The
    members that scale the coordinates into [-π, π] (`α`, `δ`, `αδ_data`, `predict_α`,
    `predict_δ`, `arcsec_to_model_domain`) are deprecated and emit `DeprecationWarning`.

    Attributes:
        processed_data: The clipped, filtered data, flattened to dimensions
            `(wavelength, flat_spaxel)` (see `flatten_tile_coord`).
        normalise_flux: Normalises a flux, `(flux - offset) / scale`.
        predict_flux: Turns a normalised flux back into erg cm⁻² s⁻¹ Å⁻¹.
        normalise_ivar: Normalises an inverse variance (multiplies it by the flux scale
            squared).
        predict_ivar: Turns a normalised inverse variance back into physical units.
        normalise_α: Normalises a right ascension in degrees.
        normalise_δ: Normalises a declination in degrees.
        sky_origin: The (RA, Dec) in degrees to measure sky offsets from, or `None` (the
            default) for the centre of the spaxels' bounding box. Set it with
            `with_sky_origin`.

    The fields are, in order: `processed_data`, `normalise_flux`, `predict_flux`,
    `normalise_ivar`, `predict_ivar`, `normalise_α`, the inverse of `normalise_α`,
    `normalise_δ`, the inverse of `normalise_δ`, and `sky_origin`.

    Example:
        ```python
        fit_data = FitDataBuilder(tiles=tile, config=config).build()
        x, y = fit_data.sky_x, fit_data.sky_y  # arcsec
        flux, i_var, mask = fit_data.flux, fit_data.i_var, fit_data.mask
        ```
    """

    processed_data: Dataset
    normalise_flux: Callable
    predict_flux: Callable
    normalise_ivar: Callable
    predict_ivar: Callable
    normalise_α: Callable
    _predict_α: Callable
    normalise_δ: Callable
    _predict_δ: Callable
    sky_origin: tuple[float, float] | None = None

    @property
    def _flux(self) -> JaxArray:
        return self.normalise_flux(to_jax_array(self.processed_data["flux"].values))

    @property
    def flux(self) -> JaxArray:
        """Normalised flux, shape `(n_wavelengths, n_spaxels)`, with NaNs (excluded pixels)
        set to 0."""
        return jnp.nan_to_num(self._flux)

    @property
    def _i_var(self) -> JaxArray:
        return self.normalise_ivar(to_jax_array(self.processed_data["i_var"].values))

    @property
    def i_var(self) -> JaxArray:
        """Normalised inverse variance of `flux`, shape `(n_wavelengths, n_spaxels)`, with
        NaNs set to 1e-4."""
        return jnp.nan_to_num(self._i_var, nan=1e-4)

    @property
    def _u_flux(self) -> JaxArray:
        return self._i_var**-0.5

    @property
    def u_flux(self) -> JaxArray:
        """Normalised flux uncertainty, `i_var**-0.5` (before its NaNs are filled), shape
        `(n_wavelengths, n_spaxels)`, with NaNs set to 100.

        Infinite values (where the inverse variance is 0) become the largest finite float,
        as in `jax.numpy.nan_to_num`.
        """
        return jnp.nan_to_num(self._u_flux, nan=1e2)

    # Legacy: coordinates scaled into [-π, π] (deprecated, use the sky_* offsets)

    def _α(self) -> JaxArray:
        return to_π_domain(self.normalise_α(to_jax_array(self.processed_data["ra"].values)))

    def _δ(self) -> JaxArray:
        return to_π_domain(self.normalise_δ(to_jax_array(self.processed_data["dec"].values)))

    @property
    def α(self) -> JaxArray:
        """Deprecated: use `sky_x` (arcsec).

        Right ascension of each spaxel, normalised with `normalise_α` and mapped onto
        [-π, π] (see `to_π_domain`), shape `(n_spaxels,)`. Emits `DeprecationWarning`.
        """
        _deprecated("α", "sky_x")
        return self._α()

    @property
    def δ(self) -> JaxArray:
        """Deprecated: use `sky_y` (arcsec).

        Declination of each spaxel, normalised with `normalise_δ` and mapped onto [-π, π]
        (see `to_π_domain`), shape `(n_spaxels,)`. Emits `DeprecationWarning`.
        """
        _deprecated("δ", "sky_y")
        return self._δ()

    def predict_α(self, x: JaxArray) -> JaxArray:
        """Deprecated: use `sky_to_radec`.

        Convert right ascensions in the [-π, π] coordinates of `α` back to degrees. Emits
        `DeprecationWarning`.

        Args:
            x: Values in the coordinates of `α`.

        Returns:
            The right ascensions in degrees.
        """
        _deprecated("predict_α", "sky_to_radec")
        return self._predict_α(from_π_domain(x))

    def predict_δ(self, x: JaxArray) -> JaxArray:
        """Deprecated: use `sky_to_radec`.

        Convert declinations in the [-π, π] coordinates of `δ` back to degrees. Emits
        `DeprecationWarning`.

        Args:
            x: Values in the coordinates of `δ`.

        Returns:
            The declinations in degrees.
        """
        _deprecated("predict_δ", "sky_to_radec")
        return self._predict_δ(from_π_domain(x))

    @property
    def αδ_data(self) -> SpatialDataLVM:
        """Deprecated: use `sky_data` (arcsec).

        The spaxel positions for spectracles as in `sky_data`, but with the coordinates of
        `α` and `δ` (in [-π, π]) instead of arcsec offsets. Emits `DeprecationWarning`.

        Raises:
            ModuleNotFoundError: If spectracles could not be imported.
        """
        _deprecated("αδ_data", "sky_data")
        return SpatialDataLVM(
            x=self._α(),
            y=self._δ(),
            idx=self.spaxel_idx,
            tile_idx=self.tile_idx,
            ifu_idx=self.ifu_idx,
            fib_idx=self.fibre_idx,
        )

    # Sky positions as tangent-plane offsets in arcsec, east and north of a centre

    @property
    def sky_centre(self) -> tuple[float, float]:
        """The (RA, Dec) in degrees that `sky_x` and `sky_y` are measured from.

        This is `sky_origin` if set, otherwise the centre of the spaxels' bounding box in RA
        and Dec. The bounding box is found with RA measured relative to the first spaxel, so
        a footprint that crosses RA = 0 is handled.
        """
        if self.sky_origin is not None:
            return float(self.sky_origin[0]), float(self.sky_origin[1])
        ra = np.asarray(self.processed_data["ra"].values, dtype=float)
        dec = np.asarray(self.processed_data["dec"].values, dtype=float)
        # RA relative to one spaxel, so a footprint across RA = 0 isn't split
        dra = (ra - ra[0] + 180.0) % 360.0 - 180.0
        ra_c = (ra[0] + 0.5 * (dra.min() + dra.max())) % 360.0
        return float(ra_c), float(0.5 * (dec.min() + dec.max()))

    def _sky_frame(self) -> SkyOffsetFrame:
        ra_c, dec_c = self.sky_centre
        return SkyOffsetFrame(origin=SkyCoord(ra=ra_c * u.deg, dec=dec_c * u.deg))

    def _sky_xy(self) -> tuple[np.ndarray, np.ndarray]:
        # float64: positions are stored in float32, which resolves only ~60 mas at RA ~ 270°
        coords = SkyCoord(
            ra=np.asarray(self.processed_data["ra"].values, dtype=float) * u.deg,
            dec=np.asarray(self.processed_data["dec"].values, dtype=float) * u.deg,
        ).transform_to(self._sky_frame())
        return coords.lon.to_value(u.arcsec), coords.lat.to_value(u.arcsec)

    @property
    def sky_x(self) -> JaxArray:
        """Each spaxel's offset east of `sky_centre`, in arcsec, shape `(n_spaxels,)`.

        The offsets are coordinates on the tangent plane at `sky_centre` (astropy's
        `SkyOffsetFrame`), so they include the cos(Dec) factor. They are computed in
        64-bit precision before being converted to a JAX array.
        """
        return jnp.asarray(self._sky_xy()[0])

    @property
    def sky_y(self) -> JaxArray:
        """Each spaxel's offset north of `sky_centre`, in arcsec, shape `(n_spaxels,)`.

        Computed as for `sky_x`, on the tangent plane at `sky_centre`.
        """
        return jnp.asarray(self._sky_xy()[1])

    @property
    def sky_data(self) -> SpatialDataLVM:
        """The spaxel positions for spectracles, as arcsec offsets from `sky_centre`.

        A spectracles `SpatialDataLVM` with `x = sky_x`, `y = sky_y`, and the spaxel, tile,
        IFU and fibre indices (`spaxel_idx`, `tile_idx`, `ifu_idx`, `fibre_idx`). Length
        scales of spectracles fields are then in arcsec. Build the field's region with
        `spectracles.Domain.from_points(d.x, d.y)`. Requires the optional `spectracles`
        dependency (`pip install lvm_tools[spectracles]`).

        Raises:
            ModuleNotFoundError: If spectracles could not be imported.

        Example:
            ```python
            from spectracles import Domain

            spaxels = fit_data.sky_data
            domain = Domain.from_points(spaxels.x, spaxels.y)
            ```
        """
        x, y = self._sky_xy()
        return SpatialDataLVM(
            x=jnp.asarray(x),
            y=jnp.asarray(y),
            idx=self.spaxel_idx,
            tile_idx=self.tile_idx,
            ifu_idx=self.ifu_idx,
            fib_idx=self.fibre_idx,
        )

    def sky_to_radec(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """Convert offsets from `sky_centre` in arcsec to (RA, Dec) in degrees.

        The inverse of `sky_x` and `sky_y`, e.g. to plot a map made on a grid of `sky_x`,
        `sky_y` in RA and Dec.

        Args:
            x (ArrayLike): Offsets east of `sky_centre`, in arcsec.
            y (ArrayLike): Offsets north of `sky_centre`, in arcsec (same shape as `x`).

        Returns:
            The right ascensions and declinations, in degrees (ICRS).
        """
        coords = SkyCoord(
            lon=np.asarray(x, dtype=float) * u.arcsec,
            lat=np.asarray(y, dtype=float) * u.arcsec,
            frame=self._sky_frame(),
        ).icrs
        return coords.ra.to_value(u.deg), coords.dec.to_value(u.deg)

    def with_sky_origin(self, ra: float, dec: float) -> FitData:
        """The same data, with sky offsets measured from (RA, Dec) in degrees.

        Use it e.g. to put several datasets on one frame. The origin is a choice of frame,
        not part of the data, so `FitDataBuilder.hash` doesn't include it.

        Args:
            ra: Right ascension of the origin, in degrees.
            dec: Declination of the origin, in degrees.

        Returns:
            A copy with `sky_origin = (ra, dec)`.
        """
        return replace(self, sky_origin=(float(ra), float(dec)))

    @property
    def λ(self) -> JaxArray:
        """Wavelengths in Angstrom, shape `(n_wavelengths,)`."""
        return to_jax_array(self.processed_data["wavelength"].values)

    @property
    def _lsf_σ(self) -> JaxArray:
        return to_jax_array(self.processed_data["lsf_sigma"].values)

    @property
    def lsf_σ(self) -> JaxArray:
        """Gaussian standard deviation of the line spread function in Angstrom (not
        normalised), shape `(n_wavelengths, n_spaxels)`, with NaNs set to the median of the
        rest."""
        median_lsf_σ = jnp.nanmedian(self._lsf_σ)
        return jnp.nan_to_num(self._lsf_σ, nan=median_lsf_σ)

    @property
    def mjd(self) -> JaxArray:
        """Modified Julian Date of the midpoint of each spaxel's exposure, shape
        `(n_spaxels,)`."""
        return to_jax_array(self.processed_data["mjd"].values)

    @property
    def mask(self) -> JaxArray:
        """Which pixels hold data, shape `(n_wavelengths, n_spaxels)`: `True` where the flux
        is not NaN, i.e. the pixel survived clipping and filtering."""
        return ~jnp.isnan(self._flux)

    @property
    def λ_idx(self) -> JaxArray:
        """Wavelength indices `0 .. n_wavelengths - 1`."""
        return jnp.arange(len(self.λ), dtype=np.int64)

    @property
    def spaxel_idx(self) -> JaxArray:
        """Spaxel indices `0 .. n_spaxels - 1`, counting the spaxels of every tile together."""
        return jnp.arange(self.processed_data["ra"].size, dtype=np.int64)

    @property
    def tile_idx(self) -> JaxArray:
        """Index of each spaxel's tile, shape `(n_spaxels,)`: the tiles' exposure numbers
        replaced by `0 .. n_tiles - 1`, in increasing order of exposure number."""
        tile = to_jax_array(self.processed_data["tile"].values, dtype=np.int64)
        return jnp.unique(tile, return_inverse=True)[1]

    @property
    def ifu_idx(self) -> JaxArray:
        """Index of each spaxel's IFU bundle, shape `(n_spaxels,)`: the IFU labels replaced
        by `0 .. n_ifus - 1`, in sorted order of label."""
        ifu = self.processed_data["ifu_label"].values
        return to_jax_array(np.unique(ifu, return_inverse=True)[1], dtype=np.int64)

    @property
    def fibre_idx(self) -> JaxArray:
        """Fibre identifier of each spaxel, shape `(n_spaxels,)`. These are the slitmap's
        fibre IDs, not re-indexed, so gaps from excluded fibres are kept."""
        fib = self.processed_data["fibre_id"].values
        # CANNOT do unique trick because we need to preserve "gaps"
        return to_jax_array(fib, dtype=np.int64)

    @property
    def v_bary(self) -> JaxArray:
        """Barycentric velocity correction of each spaxel in km/s, shape `(n_spaxels,)`.

        Computed by `get_v_barycentric` for Las Campanas Observatory, at each spaxel's
        position and exposure midpoint.
        """
        return to_jax_array(
            get_v_barycentric(
                mjd=self.mjd,
                α=to_jax_array(self.processed_data["ra"].values),
                δ=to_jax_array(self.processed_data["dec"].values),
                unit="km/s",
            )
        )

    def arcsec_to_model_domain(self, arcsec: float) -> float:
        """Deprecated: with `sky_data`, positions are already in arcsec.

        Convert a length in arcsec into the [-π, π] coordinates of `α`, treating it as a
        length in degrees of right ascension (without a cos(Dec) factor). Emits
        `DeprecationWarning`.

        Args:
            arcsec: The length, in arcsec.

        Returns:
            The length in the coordinates of `α`.
        """
        _deprecated("arcsec_to_model_domain", "sky_data (already in arcsec)")
        deg = arcsec / 3600
        probe = 1.0
        return float(
            to_π_domain(self.normalise_α(probe + deg)) - to_π_domain(self.normalise_α(probe))
        )

    def __repr__(self):
        # TODO: add something here
        raise NotImplementedError
