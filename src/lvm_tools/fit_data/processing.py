"""processing.py - convenience wrappers for filtering and clipping given LVMTileLike and DataConfig."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING

from xarray import Dataset

if TYPE_CHECKING:
    from lvm_tools.config.data_config import DataConfig
from lvm_tools.data.tile import LVMTileLike
from lvm_tools.fit_data.clipping import bounding_square, clip_dataset
from lvm_tools.fit_data.filtering import filter_dataset
from lvm_tools.fit_data.normalisation import calc_normalisation, get_norm_funcs


def clip_data(tile_data: Dataset, config: DataConfig) -> Dataset:
    """Clip tile data to the config's wavelength, RA and Dec ranges (see `clip_dataset`).

    Args:
        tile_data: Tile data, as in `LVMTile.data`.
        config: The config.

    Returns:
        The clipped data.
    """
    return clip_dataset(
        tile_data,
        config.λ_range,
        config.α_range,
        config.δ_range,
    )


def filter_tile_data(tile_data: Dataset, config: DataConfig) -> Dataset:
    """Set the data the config excludes to NaN (see `filter_dataset`).

    Args:
        tile_data: Tile data, as in `LVMTile.data`.
        config: The config.

    Returns:
        The filtered data.
    """
    return filter_dataset(
        tile_data,
        config.nans_strategy,
        config.F_bad_strategy,
        config.F_range,
        config.fibre_status_include,
        config.apply_mask,
    )


def process_tile_data(tiles: LVMTileLike, config: DataConfig) -> Dataset:
    """Clip and then filter tile data with a config.

    NumPy `RuntimeWarning`s (e.g. from medians of all-NaN spaxels) are silenced.

    Args:
        tiles: The data.
        config: The config.

    Returns:
        The processed data, with the same dimensions as `tiles.data` and excluded values set
            to NaN.

    Raises:
        ValueError: If no data are left after clipping, or after filtering.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ds = clip_data(tiles.data, config)
        if 0 in ds["flux"].shape:
            raise ValueError(
                "No data left after clipping to "
                f"λ_range={config.λ_range}, α_range={config.α_range}, δ_range={config.δ_range}. "
                "Check that the ranges overlap the tiles."
            )
        ds = filter_tile_data(ds, config)
        if 0 in ds["flux"].shape:
            raise ValueError("No data left after filtering. Check the DataConfig filter settings.")
        return ds


def get_αδ_ranges(tiles: LVMTileLike) -> tuple[tuple[float, float], tuple[float, float]]:
    """RA and Dec ranges of a square around all the spaxels (see `bounding_square`).

    The square is in degrees of RA and Dec, without a cos(Dec) factor.

    Args:
        tiles: The data.

    Returns:
        The pair `(α_range, δ_range)`, in degrees.
    """
    return bounding_square(
        tiles.data["ra"].values.min(),
        tiles.data["ra"].values.max(),
        tiles.data["dec"].values.min(),
        tiles.data["dec"].values.max(),
    )


def get_normalisations(
    ds: Dataset, config: DataConfig
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Normalisation offsets and scales of processed data, using the config's strategies.

    The flux uses `config.normalise_F_strategy`; RA and Dec each use
    `config.normalise_αδ_strategy`, separately (`DataConfig.from_tiles` then gives both the
    larger scale).

    Args:
        ds: Processed data, e.g. from `process_tile_data`.
        config: The config.

    Returns:
        `((F_offset, F_scale), (α_offset, α_scale), (δ_offset, δ_scale))`.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return (
            calc_normalisation(ds["flux"].values, config.normalise_F_strategy),
            calc_normalisation(ds["ra"].values, config.normalise_αδ_strategy),
            calc_normalisation(ds["dec"].values, config.normalise_αδ_strategy),
        )


def get_normalisation_functions(
    config: DataConfig,
) -> tuple[tuple[Callable, Callable], tuple[Callable, Callable], tuple[Callable, Callable]]:
    """The normalisation functions given by a config's offsets and scales.

    The inverse variance is normalised consistently with the flux: it has offset 0 and
    scale `normalise_F_scale**-2`, so it is multiplied by `normalise_F_scale**2`.

    Args:
        config: The config.

    Returns:
        Eight functions, in the order `FitData` takes them: normalise and denormalise the
            flux, the inverse variance, RA and Dec.
    """
    return (
        *get_norm_funcs(config.normalise_F_offset, config.normalise_F_scale),
        *get_norm_funcs(0.0, config.normalise_F_scale**-2),
        *get_norm_funcs(config.normalise_α_offset, config.normalise_α_scale),
        *get_norm_funcs(config.normalise_δ_offset, config.normalise_δ_scale),
    )


def flatten_tile_coord(ds: Dataset) -> Dataset:
    """Flatten the tile and spaxel dimensions into a single `flat_spaxel` dimension.

    `flat_spaxel` runs over the spaxels of the first tile, then the second, and so on. It is
    placed last, so the cubes have dimensions `(wavelength, flat_spaxel)`. The `tile` and
    `spaxel` values of each flat spaxel are kept as coordinates along it, and per-tile
    coordinates such as `mjd` are repeated for each spaxel.

    Args:
        ds: Data with `tile` and `spaxel` dimensions.

    Returns:
        The flattened data.
    """
    return ds.stack(flat_spaxel=("tile", "spaxel")).reset_index("flat_spaxel")
