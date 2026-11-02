"""validation.py - Validation functions for data processing configuration."""

from typing import get_args

import numpy as np

from lvm_tools.fit_data.clipping import Range
from lvm_tools.fit_data.filtering import ExcludeStrategy, FibreStatus
from lvm_tools.fit_data.normalisation import NormaliseStrategy


def validate_range(x_range: Range) -> None:
    """Check that a range is a `(min, max)` pair with `max >= min`.

    Args:
        x_range: The range.

    Raises:
        ValueError: If it doesn't have two values, or if `max < min`.
    """
    # if not isinstance(x_range, tuple):
    # raise TypeError("Data range must be in a tuple.")
    if len(x_range) != 2:
        raise ValueError("Data range must be a tuple with exactly two values (min, max).")
    if x_range[1] < x_range[0]:
        raise ValueError("Requested data range restriction has max < min.")


def validate_excl_strategy(strategy: ExcludeStrategy) -> None:
    """Check that a value is one of the `ExcludeStrategy` options.

    Args:
        strategy: The strategy.

    Raises:
        ValueError: If it is not an option.
    """
    if strategy not in get_args(ExcludeStrategy):
        raise ValueError(f"Unknown exclusion strategy: {strategy}")


def validate_norm_strategy(strategy: NormaliseStrategy) -> None:
    """Check that a value is one of the `NormaliseStrategy` options.

    Args:
        strategy: The strategy.

    Raises:
        ValueError: If it is not an option.
    """
    if strategy not in get_args(NormaliseStrategy):
        raise ValueError(f"Unknown normalisation strategy: {strategy}")


def validate_fib_status_incl(fibre_status_include: tuple[FibreStatus]) -> None:
    """Check that every fibre status is one of the `FibreStatus` values (0, 1, 2 or 3).

    Args:
        fibre_status_include: The fibre status values.

    Raises:
        ValueError: If a value is not a known status.
    """
    # if not isinstance(fibre_status_include, tuple):
    # raise TypeError("fibre_status_include must be a tuple.")
    for fs in fibre_status_include:
        if fs not in get_args(FibreStatus):
            raise ValueError(f"Unknown fibre status: {fs}")


def validate_offset(offset: float) -> None:
    """Check that a normalisation offset is a finite float.

    Args:
        offset: The offset.

    Raises:
        TypeError: If it is not a Python or NumPy float (an `int` is rejected).
        ValueError: If it is NaN or infinite.
    """
    if not isinstance(offset, (float, np.floating)):
        raise TypeError("offset must be float.")
    if not np.isfinite(offset):
        raise ValueError("Bad offset (nan or infty).")


def validate_scale(scale: float) -> None:
    """Check that a normalisation scale is a positive, finite float.

    Args:
        scale: The scale.

    Raises:
        TypeError: If it is not a Python or NumPy float (an `int` is rejected).
        ValueError: If it is NaN, infinite, or not positive.
    """
    if not isinstance(scale, (float, np.floating)):
        raise TypeError("scale must be float.")
    if not np.isfinite(scale):
        raise ValueError("Bad scale (nan or infty).")
    if scale <= 0:
        raise ValueError("Scale is not positive, but it must be.")


def validate_apply_mask(apply_mask: bool) -> None:
    """Check that `apply_mask` is a `bool`.

    Args:
        apply_mask: The value.

    Raises:
        TypeError: If it is not a `bool`.
    """
    if not isinstance(apply_mask, bool):
        raise TypeError("apply_mask must be a boolean.")
