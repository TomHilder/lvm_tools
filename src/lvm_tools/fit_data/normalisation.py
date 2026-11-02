"""normalisation.py - data normalisation for data preparation."""

from collections.abc import Callable
from functools import partial
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

NORM_PADDING = 0.15
"""Padding of the `"padded"` normalisation, as a fraction of the data range on each side."""


# None stays inside the Literal: validation checks values against get_args(NormaliseStrategy)
NormaliseStrategy = Literal[None, "max only", "98 only", "extrema", "1σ", "2σ", "3σ", "padded"]  # noqa: PYI061
"""How to compute a normalisation offset and scale from data; see `calc_normalisation`."""


def calc_normalisation(data: ArrayLike, strategy: NormaliseStrategy) -> tuple[float, float]:
    """The offset and scale that normalise data as `(data - offset) / scale`.

    NaN values are ignored. The strategies give:

    | Strategy     | Offset                    | Scale                                   |
    | ------------ | ------------------------- | --------------------------------------- |
    | `None`       | 0                         | 1                                       |
    | `"max only"` | 0                         | maximum                                 |
    | `"98 only"`  | 0                         | 98th percentile                         |
    | `"extrema"`  | minimum                   | maximum - minimum                       |
    | `"1σ"`       | mean                      | 2 × standard deviation                  |
    | `"2σ"`       | mean                      | 4 × standard deviation                  |
    | `"3σ"`       | mean                      | 6 × standard deviation                  |
    | `"padded"`   | minimum - 0.15 × range    | 1.3 × range                             |

    So `"extrema"` maps the data onto [0, 1], `"Nσ"` maps mean ± Nσ onto an interval of
    width 1 centred on 0, and `"padded"` maps the data onto [0.15/1.3, 1.15/1.3], leaving
    `NORM_PADDING` of the range free on each side.

    Args:
        data: The values.
        strategy: The normalisation strategy.

    Returns:
        The pair `(offset, scale)`.

    Raises:
        ValueError: If the strategy is unknown.
    """
    offset = 0.0
    scale = 1.0
    if strategy is None:
        pass
    elif strategy == "max only":
        scale = np.nanmax(data)
    elif strategy == "98 only":
        scale = np.nanpercentile(data, 98)
    elif strategy == "extrema":
        offset = np.nanmin(data)
        scale = np.nanmax(data) - offset
    elif strategy in ("1σ", "2σ", "3σ"):
        offset = np.nanmean(data)
        scale = 2.0 * int(strategy[0]) * np.nanstd(data)
    elif strategy == "padded":
        data_range = np.nanmax(data) - np.nanmin(data)
        offset = np.nanmin(data) - NORM_PADDING * data_range
        scale = (1 + 2 * NORM_PADDING) * data_range
    else:
        raise ValueError(f"Unknown normalisation strategy: {strategy}")
    return float(offset), float(scale)


def normalise(data: ArrayLike, offset: float, scale: float) -> ArrayLike:
    """Normalise data: `(data - offset) / scale`.

    Args:
        data: The values.
        offset: The offset.
        scale: The scale.

    Returns:
        The normalised values.
    """
    return (data - offset) / scale


def denormalise(data: ArrayLike, offset: float, scale: float) -> ArrayLike:
    """Undo `normalise`: `data * scale + offset`.

    Args:
        data: The normalised values.
        offset: The offset.
        scale: The scale.

    Returns:
        The values on the original scale.
    """
    return data * scale + offset


def get_norm_funcs(offset: float, scale: float) -> tuple[Callable, Callable]:
    """`normalise` and `denormalise` with a fixed offset and scale.

    Args:
        offset: The offset.
        scale: The scale.

    Returns:
        The two functions, `normalise` then `denormalise`, each taking only the data, e.g.
            `norm, denorm = get_norm_funcs(0.0, 2.0)`.
    """
    return tuple(partial(f, offset=offset, scale=scale) for f in (normalise, denormalise))  # type: ignore[return-value]
