"""clipping.py - data clipping for data preparation."""

import warnings

import numpy as np
from xarray import DataArray, Dataset, concat

Range = tuple[float, float]
"""A `(min, max)` range."""
Ranges = tuple[Range, ...]
"""Several `(min, max)` ranges, e.g. `((6500.0, 6600.0), (6700.0, 6800.0))`."""


def bounding_square(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """The square around a rectangle, with a little room to spare.

    The square has the rectangle's centre, and sides 1.01 times the longer side of the
    rectangle (0.5% of padding on each side). It gives RA and Dec ranges of equal width.

    Args:
        x_min: Lower x bound of the rectangle.
        x_max: Upper x bound of the rectangle.
        y_min: Lower y bound of the rectangle.
        y_max: Upper y bound of the rectangle.

    Returns:
        The square's `(x_min, x_max)` and `(y_min, y_max)` ranges.
    """
    t_range = 1.01 * max(x_max - x_min, y_max - y_min)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    x_min_ = x_mid - t_range / 2
    x_max_ = x_mid + t_range / 2
    y_min_ = y_mid - t_range / 2
    y_max_ = y_mid + t_range / 2
    return (x_min_, x_max_), (y_min_, y_max_)


def slice_mask(arr: DataArray, x_min: float, x_max: float) -> DataArray:
    """Where the values of an array lie in `[x_min, x_max]` (bounds included).

    Args:
        arr: The values.
        x_min: Lower bound.
        x_max: Upper bound.

    Returns:
        A boolean array, `True` inside the range.
    """
    return (arr >= x_min) & (arr <= x_max)


def ensure_ranges(ranges: Range | Ranges) -> Ranges:
    """Turn a single range into a tuple of one range; leave several ranges as they are.

    Args:
        ranges: A range `(min, max)`, or several ranges.

    Returns:
        A tuple of ranges.
    """
    # Normalize to always work with tuple of tuples
    if isinstance(ranges[0], (int, float)):
        ranges: Ranges = (ranges,)
    return ranges


def verify_range(r: Range) -> None:
    """Check that a range is a tuple (or list) of floats with `min < max`.

    Args:
        r: The range.

    Raises:
        TypeError: If it is not a tuple or list, or an entry is neither a float nor an int.
        ValueError: If `min >= max`.

    Warns:
        UserWarning: If an entry is an int rather than a float.
    """
    # Error if not a tuple
    if not isinstance(r, (tuple, list)):
        raise TypeError("Range must be a tuple.")
    # Error if entries are not strictly floats
    if not all(isinstance(x, (float, np.floating)) for x in r):
        if any(isinstance(x, int) for x in r):
            warnings.warn(
                "Range entries should be floats, but integers were provided. Implicit conversion will be applied.",
                UserWarning,
            )
        else:
            raise TypeError("Range entries must be floats.")
    if r[0] >= r[1]:
        raise ValueError(f"Invalid range: {r}")


def verify_ranges(ranges: Ranges) -> None:
    """Check several ranges: each must be valid, and they must be in increasing order and
    not overlap.

    Args:
        ranges: The ranges.

    Raises:
        TypeError: If a range is invalid (see `verify_range`) or `ranges` is not a tuple.
        ValueError: If a range has `min >= max`, or the ranges overlap or are out of order.
    """
    # We check not only that each range is ordered, but also that they are ordered overall
    # and non-overlapping
    # Need to make sure (-inf, inf) is allowed as a single range
    previous_upper = -np.inf
    for r in ranges:
        verify_range(r)
        if r[0] <= previous_upper and previous_upper != -np.inf:
            raise ValueError(f"Overlapping or unordered ranges: {ranges}")
        previous_upper = r[1]
    # Error if not a tuple of tuples
    if not isinstance(ranges, tuple):
        raise TypeError("Ranges must be a tuple of tuples.")


def clip_wavelengths(data: Dataset, λ_ranges: Range | Ranges) -> Dataset:
    """Keep only the wavelengths in one or more ranges.

    Each range is selected with its bounds included, and the selections are concatenated
    along the wavelength dimension, so several separate windows (e.g. around different
    emission lines) can be kept.

    Args:
        data: Tile data, with a `wavelength` coordinate in Angstrom.
        λ_ranges: A range `(min, max)` or several ranges, in increasing order and not
            overlapping.

    Returns:
        The data at the selected wavelengths.

    Raises:
        TypeError: If the ranges are malformed (see `verify_ranges`).
        ValueError: If a range has `min >= max`, or the ranges overlap or are out of order.
    """
    λ_ranges = ensure_ranges(λ_ranges)
    verify_ranges(λ_ranges)
    # Assemble slices
    ds_slices = []
    for r in λ_ranges:
        ds_slices.append(data.sel(wavelength=slice(*r)))
    # Concatenate along wavelength axis
    return concat(ds_slices, dim="wavelength")


def clip_dataset(
    data: Dataset,
    λ_range: Range | Ranges,
    α_range: Range,
    δ_range: Range,
) -> Dataset:
    """Clip tile data to wavelength range(s) and an RA/Dec box.

    The wavelengths are clipped with `clip_wavelengths`. Values at spaxels outside the RA/Dec
    box (bounds included) are set to NaN, and spaxels that are outside it in every tile are
    dropped.

    Args:
        data: Tile data, as in `LVMTile.data`.
        λ_range: Wavelength range(s) to keep, in Angstrom.
        α_range: Right ascension range `(min, max)` to keep, in degrees.
        δ_range: Declination range `(min, max)` to keep, in degrees.

    Returns:
        The clipped data.

    Raises:
        TypeError: If a range is malformed (see `verify_range`).
        ValueError: If a range has `min >= max`, or the wavelength ranges overlap or are
            out of order.
    """
    # Clip to wavelength range (simple since wavelength is an indexed coordinate)
    # data = data.sel(wavelength=slice(*λ_range))
    data = clip_wavelengths(data, λ_range)
    # Clip to ra, dec range. Less simple since spaxel is the indexed coordinate
    _ = verify_range(α_range), verify_range(δ_range)
    α_slice = slice_mask(data["ra"], *α_range)
    δ_slice = slice_mask(data["dec"], *δ_range)
    return data.where(α_slice & δ_slice, drop=True)
