"""clipping.py - data clipping for data preparation."""

import numpy as np
from xarray import DataArray, Dataset, concat

Range = tuple[float, float]
Ranges = tuple[Range, ...]


def bounding_square(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    t_range = 1.01 * max(x_max - x_min, y_max - y_min)
    x_mid = (x_min + x_max) / 2
    y_mid = (y_min + y_max) / 2
    x_min_ = x_mid - t_range / 2
    x_max_ = x_mid + t_range / 2
    y_min_ = y_mid - t_range / 2
    y_max_ = y_mid + t_range / 2
    return (x_min_, x_max_), (y_min_, y_max_)


def slice_mask(arr: DataArray, x_min: float, x_max: float) -> DataArray:
    return (arr >= x_min) & (arr <= x_max)


def ensure_ranges(ranges: Range | Ranges) -> Ranges:
    # Normalize to always work with tuple of tuples
    if isinstance(ranges[0], (int, float)):
        ranges: Ranges = (ranges,)
    return ranges


def verify_range(r: Range) -> None:
    # Error if not a tuple
    # if not isinstance(r, tuple):
    # raise TypeError("Range must be a tuple.")
    # Error if entries are not strictly floats
    if not all(isinstance(x, (float, np.floating)) for x in r):
        raise TypeError("Range entries must be floats.")
    if r[0] >= r[1]:
        raise ValueError(f"Invalid range: {r}")


def verify_ranges(ranges: Ranges) -> None:
    # We check not only that each range is ordered, but also that they are ordered overall
    # and non-overlapping
    previous_upper = -np.inf
    for r in ranges:
        verify_range(r)
        if r[0] <= previous_upper:
            raise ValueError(f"Overlapping or unordered ranges: {ranges}")
        previous_upper = r[1]
    # Error if not a tuple of tuples
    if not isinstance(ranges, tuple):
        raise TypeError("Ranges must be a tuple of tuples.")


def clip_wavelengths(data: Dataset, λ_ranges: Range | Ranges) -> Dataset:
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
    # Clip to wavelength range (simple since wavelength is an indexed coordinate)
    # data = data.sel(wavelength=slice(*λ_range))
    data = clip_wavelengths(data, λ_range)
    # Clip to ra, dec range. Less simple since spaxel is the indexed coordinate
    _ = verify_range(α_range), verify_range(δ_range)
    α_slice = slice_mask(data["ra"], *α_range)
    δ_slice = slice_mask(data["dec"], *δ_range)
    return data.where(α_slice & δ_slice, drop=True)
