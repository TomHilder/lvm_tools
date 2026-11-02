"""filtering.py - data filtering for data preparation."""

import warnings
from typing import Literal

import numpy as np
from xarray import DataArray, Dataset

BAD_FLUX_THRESHOLD = -0.1e-13
"""Default lower bound of acceptable flux (`DataConfig.F_range`), -1e-14 erg cm⁻² s⁻¹ Å⁻¹."""


# None stays inside the Literal: validation checks values against get_args(ExcludeStrategy)
ExcludeStrategy = Literal[None, "pixel", "spaxel", "spaxel_max"]  # noqa: PYI061
"""How to exclude bad data: `None`, `"pixel"`, `"spaxel"` or `"spaxel_max"`. See
`DataConfig.nans_strategy` and `DataConfig.F_bad_strategy` for which each accepts."""
FibreStatus = Literal[0, 1, 2, 3]  # I have no idea what these mean, but they're in the data
"""A fibre status value from the DRP slitmap: 0, 1, 2 or 3."""


def get_where_nan(arr: DataArray) -> DataArray:
    """Where an array is NaN.

    Args:
        arr: The values.

    Returns:
        A boolean array, `True` where the value is NaN.
    """
    return arr.isnull()


def get_where_bad(arr: DataArray, bad_range: tuple[float, float]) -> DataArray:
    """Where pixels are at or below the lower bound of the acceptable range.

    A pixel is flagged when its value is `<= bad_range[0]` (and `< bad_range[1]`). Values
    above `bad_range[1]` and NaN values are not flagged.

    Args:
        arr: The values, e.g. the flux.
        bad_range: The acceptable range `(min, max)`.

    Returns:
        A boolean array of the same shape, `True` where the pixel is bad.
    """
    return ~(arr > bad_range[0]) & (arr < bad_range[1])


def get_where_bad_median(arr: DataArray, bad_range: tuple[float, float]) -> DataArray:
    """Which spaxels have a median outside the acceptable range, or are entirely NaN.

    Args:
        arr: The values, with a `wavelength` dimension, e.g. the flux.
        bad_range: The acceptable range `(min, max)` of the median over wavelength.

    Returns:
        A boolean array without the `wavelength` dimension, `True` where the spaxel is bad.
    """
    where_bad_median_l = arr.median(dim="wavelength") < bad_range[0]
    where_bad_median_u = arr.median(dim="wavelength") > bad_range[1]
    all_nan = arr.isnull().all(dim="wavelength")
    return where_bad_median_l | where_bad_median_u | all_nan


def get_where_bad_max(arr: DataArray, bad_range: tuple[float, float]) -> DataArray:
    """Which spaxels have a maximum outside the acceptable range, or are entirely NaN.

    Args:
        arr: The values, with a `wavelength` dimension, e.g. the flux.
        bad_range: The acceptable range `(min, max)` of the maximum over wavelength.

    Returns:
        A boolean array without the `wavelength` dimension, `True` where the spaxel is bad.
    """
    where_bad_max_l = arr.max(dim="wavelength") < bad_range[0]
    where_bad_max_u = arr.max(dim="wavelength") > bad_range[1]
    all_nan = arr.isnull().all(dim="wavelength")
    return where_bad_max_l | where_bad_max_u | all_nan


def get_where_badfib(fib_stat_arr: DataArray, fibre_status_incl: tuple[FibreStatus]) -> DataArray:
    """Which spaxels have a fibre status that is not included.

    Args:
        fib_stat_arr: The fibre status of each spaxel.
        fibre_status_incl: The status values to keep.

    Returns:
        A boolean array, `True` where the status is not in `fibre_status_incl`.
    """
    return ~fib_stat_arr.isin(fibre_status_incl)


def get_where_mask(arr_mask: DataArray) -> DataArray:
    """Where the DRP mask flags a pixel (mask value 1).

    Args:
        arr_mask: The DRP mask.

    Returns:
        A boolean array, `True` where the pixel is flagged.
    """
    return arr_mask == 1


def combine_wheres(list_where: list[DataArray]) -> DataArray:
    """Combine boolean arrays with a logical OR, broadcasting them against each other.

    Args:
        list_where: The boolean arrays (at least one).

    Returns:
        `True` wherever any of them is `True`.
    """
    combined_where = list_where[0]
    for where in list_where[1:]:
        combined_where = combined_where | where
    return combined_where


def filter_dataset(
    data: Dataset,
    nans_strategy: ExcludeStrategy,
    F_bad_strategy: ExcludeStrategy,
    F_bad_range: tuple[float, float],
    fibre_status_include: tuple[FibreStatus],
    apply_mask: bool,
) -> Dataset:
    """Set bad data to NaN.

    A value is excluded if any of these flag it: the NaN strategy, the bad flux strategy,
    the fibre status, and (if `apply_mask`) the DRP mask. All data variables (flux, inverse
    variance, LSF and mask) are set to NaN where excluded; spaxel-level flags exclude every
    wavelength of the spaxel.

    Args:
        data: Tile data, as in `LVMTile.data`.
        nans_strategy: `"pixel"` (keep other pixels of a spaxel with NaN fluxes) or
            `"spaxel"` (exclude spaxels with any NaN flux).
        F_bad_strategy: `"pixel"`, `"spaxel"` or `"spaxel_max"`; see
            `DataConfig.F_bad_strategy`.
        F_bad_range: The acceptable flux range `(min, max)`.
        fibre_status_include: Fibre status values to keep.
        apply_mask: Whether to exclude the pixels flagged in the DRP mask.

    Returns:
        The data, with excluded values set to NaN.

    Raises:
        ValueError: If `nans_strategy` or `F_bad_strategy` is not one of the values above.
    """
    where_bad = []

    # Nans
    if nans_strategy == "pixel":
        pass  # no action needed
    elif nans_strategy == "spaxel":
        where_bad.append(get_where_nan(data["flux"]).any(dim="wavelength"))
    else:
        raise ValueError(f"Unknown nans strategy: {nans_strategy}")

    # Fluxes
    if F_bad_strategy == "pixel":
        where_bad.append(get_where_bad(data["flux"], F_bad_range))
    elif F_bad_strategy == "spaxel":
        where_bad.append(get_where_bad_median(data["flux"], F_bad_range))
    elif F_bad_strategy == "spaxel_max":
        where_bad.append(get_where_bad_max(data["flux"], F_bad_range))
    else:
        raise ValueError(f"Unknown bad flux strategy: {F_bad_strategy}")

    # Bad fibres
    where_bad.append(get_where_badfib(data["fibre_status"], fibre_status_include))

    # Filter using mask
    if apply_mask:
        where_bad.append(get_where_mask(data["mask"]))

    return data.where(~combine_wheres(where_bad))


def filter_inspector(
    data: Dataset,
    F_bad_range: tuple[float, float],
    fibre_status_include: tuple[FibreStatus],
) -> dict:
    """Count how many pixels and spaxels each filter would exclude, to help choose a config.

    Each entry of the result is a pair `(n_pixels, n_spaxels)`, with `None` where a count
    doesn't apply:

    - `"nans"`: NaN fluxes, and spaxels with any NaN flux.
    - `"bad flux"`: pixels flagged by `get_where_bad` (the `"pixel"` strategy).
    - `"bad flux median"`: spaxels flagged by `get_where_bad_median` (the `"spaxel"`
      strategy).
    - `"fibre status"`: spaxels whose fibre status is not in `fibre_status_include`.
    - `"mask"`: pixels flagged in the DRP mask, and spaxels with any flagged pixel.
    - `"any bad"`: pixels flagged by any of NaN, bad flux, fibre status or mask, and
      spaxels with any pixel flagged by NaN, bad median flux, fibre status or mask.

    Args:
        data: Tile data, as in `LVMTile.data`.
        F_bad_range: The acceptable flux range `(min, max)`.
        fibre_status_include: Fibre status values to keep.

    Returns:
        The counts, keyed by filter name.

    Example:
        ```python
        stats = filter_inspector(tile.data, (-1e-14, np.inf), (0,))
        print(f"NaN pixels: {stats['nans'][0]}")
        print(f"Bad fibre spaxels: {stats['fibre status'][1]}")
        ```
    """
    # TODO: maybe plots instead of printing?

    # ignore warnings about median of all nans
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # nans:
        where_nan = get_where_nan(data["flux"])
        n_nans = int(np.sum(where_nan))
        n_spaxels_nan = int(np.sum(where_nan.any(dim="wavelength")))

        # bad flux (per pix):
        where_Fbad = get_where_bad(data["flux"], F_bad_range)
        n_Fbad = int(np.sum(where_Fbad))

        # bad flux (per spaxel):
        where_Fbad_median = get_where_bad_median(data["flux"], F_bad_range)
        n_Fbad_median = int(np.sum(where_Fbad_median))

        # fibre status:
        where_badfib = get_where_badfib(data["fibre_status"], fibre_status_include)
        n_spaxels_badfib = int(np.sum(where_badfib))

        # mask:
        where_mask = get_where_mask(data["mask"])
        n_mask = int(np.sum(where_mask))
        n_spaxels_mask = int(np.sum(where_mask.any(dim="wavelength")))

        # anything is bad
        where_anybad = where_nan | where_Fbad | where_badfib | where_mask
        n_anybad = int(np.sum(where_anybad))

        where_anybad_spaxel = where_nan | where_Fbad_median | where_badfib | where_mask
        n_spaxels_anybad = int(np.sum(where_anybad_spaxel.any(dim="wavelength")))

    return {
        "nans": (n_nans, n_spaxels_nan),
        "bad flux": (n_Fbad, None),
        "bad flux median": (None, n_Fbad_median),
        "fibre status": (None, n_spaxels_badfib),
        "mask": (n_mask, n_spaxels_mask),
        "any bad": (n_anybad, n_spaxels_anybad),
    }
