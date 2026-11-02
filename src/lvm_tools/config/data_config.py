"""data_config.py - Objects for specifying configuration of data processing before fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from lvm_tools.config.validation import (
    validate_apply_mask,
    validate_excl_strategy,
    validate_fib_status_incl,
    validate_norm_strategy,
    validate_offset,
    validate_range,
    validate_scale,
)
from lvm_tools.data.tile import LVMTileLike
from lvm_tools.fit_data.clipping import Range, Ranges
from lvm_tools.fit_data.filtering import (
    BAD_FLUX_THRESHOLD,
    ExcludeStrategy,
    FibreStatus,
)
from lvm_tools.fit_data.normalisation import NormaliseStrategy
from lvm_tools.fit_data.processing import get_normalisations, get_αδ_ranges, process_tile_data


@dataclass(frozen=True)
class DataConfig:
    """Settings for preparing LVM data for fitting: what to select, what to exclude, and how
    to normalise it.

    A `DataConfig` is immutable. Usually it is made with `DataConfig.from_tiles`, which works
    out the RA/Dec ranges and the normalisation offsets and scales from the data, and then
    passed with the tiles to `FitDataBuilder`.

    The strategies, `F_range`, `fibre_status_include`, `apply_mask` and the normalisation
    offsets and scales are checked when the config is created, raising `ValueError` (or
    `TypeError` for a value of the wrong type). The wavelength, RA and Dec ranges are
    checked when the data are clipped.

    Attributes:
        λ_range: Wavelength range(s) to keep, in Angstrom. A single range `(min, max)`, or
            several ranges `((min1, max1), (min2, max2), ...)` in increasing order and not
            overlapping, whose selections are joined along wavelength. Each range needs
            `min < max`. Default `(-inf, inf)`.
        α_range: Right ascension range `(min, max)` to keep, in degrees. Default
            `(-inf, inf)`.
        δ_range: Declination range `(min, max)` to keep, in degrees. Default `(-inf, inf)`.
        nans_strategy: How to exclude NaN fluxes. `"pixel"` leaves them as individual NaN
            pixels; `"spaxel"` excludes every spaxel with any NaN flux. The other
            `ExcludeStrategy` values pass validation but raise `ValueError` when the data
            are filtered. Default `"pixel"`.
        F_bad_strategy: How to exclude bad fluxes using `F_range`. `"pixel"` excludes each
            pixel whose flux is at or below `F_range[0]` (the upper bound is not applied
            per pixel); `"spaxel"` excludes spaxels whose median flux over wavelength is
            outside `F_range`; `"spaxel_max"` excludes spaxels whose maximum flux is
            outside `F_range`. Both spaxel strategies also exclude spaxels whose fluxes are
            all NaN. `None` passes validation but raises `ValueError` when the data are
            filtered. Default `"spaxel"`.
        F_range: Range `(min, max)` of acceptable flux, in erg cm⁻² s⁻¹ Å⁻¹, used by
            `F_bad_strategy`. Default `(BAD_FLUX_THRESHOLD, inf)`, i.e. `(-1e-14, inf)`.
        fibre_status_include: Fibre status values to keep (each 0, 1, 2 or 3); spaxels
            with any other status are excluded. Default `(0,)`.
        apply_mask: Whether to exclude the pixels flagged in the DRP mask. Default `True`.
        normalise_F_strategy: How `from_tiles` computes the flux normalisation (see
            `NormaliseStrategy` and `calc_normalisation`). Default `"max only"`.
        normalise_F_offset: Offset of the flux normalisation; the normalised flux is
            `(flux - offset) / scale`. Must be a finite float. Default `0.0`.
        normalise_F_scale: Scale of the flux normalisation. Must be a positive, finite
            float. The inverse variance is multiplied by `scale**2`. Default `1.0`.
        normalise_αδ_strategy: How `from_tiles` computes the RA and Dec normalisations
            (same options as `normalise_F_strategy`). Default `"padded"`.
        normalise_α_offset: Offset of the RA normalisation, in degrees. Default `0.0`.
        normalise_α_scale: Scale of the RA normalisation, in degrees. Default `1.0`.
        normalise_δ_offset: Offset of the Dec normalisation, in degrees. Default `0.0`.
        normalise_δ_scale: Scale of the Dec normalisation, in degrees. Default `1.0`.

    Example:
        ```python
        from lvm_tools import DataConfig

        config = DataConfig.from_tiles(tile, λ_range=(6500.0, 6800.0))
        config = DataConfig.from_dict(config.to_dict() | {"apply_mask": False})
        ```
    """

    # Data clipping ranges (aka choose data of interest)
    λ_range: Range | Ranges = (-np.inf, np.inf)
    α_range: Range = (-np.inf, np.inf)
    δ_range: Range = (-np.inf, np.inf)
    # Bad data ranges and strategies (aka exclude bad data)
    nans_strategy: ExcludeStrategy = "pixel"
    F_bad_strategy: ExcludeStrategy = "spaxel"
    F_range: Range = (BAD_FLUX_THRESHOLD, np.inf)
    # Handling of flagged data
    fibre_status_include: tuple[FibreStatus] = (0,)
    apply_mask: bool = True
    # Normalisation
    normalise_F_strategy: NormaliseStrategy = "max only"
    normalise_F_offset: float = 0.0
    normalise_F_scale: float = 1.0
    normalise_αδ_strategy: NormaliseStrategy = "padded"
    normalise_α_offset: float = 0.0
    normalise_α_scale: float = 1.0
    normalise_δ_offset: float = 0.0
    normalise_δ_scale: float = 1.0

    def __post_init__(self) -> None:
        # validate_range(self.λ_range) # NOTE: moved to clipping.py
        # validate_range(self.α_range)
        # validate_range(self.δ_range)
        validate_excl_strategy(self.nans_strategy)
        validate_excl_strategy(self.F_bad_strategy)
        validate_range(self.F_range)
        validate_fib_status_incl(self.fibre_status_include)
        validate_apply_mask(self.apply_mask)
        validate_norm_strategy(self.normalise_F_strategy)
        validate_norm_strategy(self.normalise_αδ_strategy)
        validate_offset(self.normalise_F_offset)
        validate_scale(self.normalise_F_scale)
        validate_norm_strategy(self.normalise_αδ_strategy)
        validate_offset(self.normalise_α_offset)
        validate_scale(self.normalise_α_scale)
        validate_offset(self.normalise_δ_offset)
        validate_scale(self.normalise_δ_scale)

    @staticmethod
    def default() -> DataConfig:
        """A config with every setting at its default.

        Returns:
            The default config, the same as `DataConfig()`.
        """
        return DataConfig()

    @staticmethod
    def from_tiles(
        tiles: LVMTileLike,
        λ_range: Range | Ranges = (-np.inf, np.inf),
        **overrides,
    ) -> DataConfig:
        """A config with the RA/Dec ranges and normalisations computed from the data.

        The steps are:

        1. Set `α_range` and `δ_range` to a square around all the spaxels of `tiles` (see
           `get_αδ_ranges`).
        2. Clip and filter the data with that config, `λ_range` and any `overrides`.
        3. Compute the flux, RA and Dec normalisations of what is left, using the
           normalisation strategies. RA and Dec get the larger of their two scales, so the
           normalised coordinates keep the sky's aspect ratio.
        4. Apply `overrides` again, so they take precedence over the computed values.

        Args:
            tiles: The data to configure for.
            λ_range: Wavelength range(s) to keep, in Angstrom (see the `λ_range` attribute).
                It is not computed from the data.
            **overrides (Any): Any other attributes, by name, e.g. `nans_strategy="spaxel"`.

        Returns:
            The config.

        Raises:
            ValueError: If an override is not an attribute of `DataConfig`, if a setting is
                invalid, or if no data are left after clipping or filtering.
            TypeError: If a setting has the wrong type.

        Example:
            ```python
            config = DataConfig.from_tiles(
                tile,
                λ_range=(6500.0, 6800.0),
                normalise_F_strategy="98 only",
            )
            ```
        """
        # λ_range cannot be set automatically
        α_range, δ_range = get_αδ_ranges(tiles)

        # Instantiate a data config with calc'd + default + overrides
        config_dict = DataConfig(λ_range=λ_range, α_range=α_range, δ_range=δ_range).to_dict()
        config = DataConfig.from_dict(config_dict | overrides)

        # Clip and filter the data
        ds = process_tile_data(tiles, config)

        # Calculate the normalisation parameters
        (
            (normalise_F_offset, normalise_F_scale),
            (normalise_α_offset, normalise_α_scale),
            (normalise_δ_offset, normalise_δ_scale),
        ) = get_normalisations(ds, config)

        # We want a square domain in the α, δ plane
        norm_αδ_scale = max(normalise_α_scale, normalise_δ_scale)

        # Update the config with the calculated values
        norm_overrides = {
            "normalise_F_offset": normalise_F_offset,
            "normalise_F_scale": normalise_F_scale,
            "normalise_α_offset": normalise_α_offset,
            "normalise_α_scale": norm_αδ_scale,
            "normalise_δ_offset": normalise_δ_offset,
            "normalise_δ_scale": norm_αδ_scale,
        }

        # Merge partial config + norm + user overrides, with user overrides taking precedence
        return DataConfig.from_dict(config.to_dict() | norm_overrides | overrides)

    @staticmethod
    def from_dict(config: dict) -> DataConfig:
        """A config from a dictionary of all its attributes, as made by `to_dict`.

        Args:
            config: Every attribute of `DataConfig`, by name.

        Returns:
            The config.

        Raises:
            ValueError: If `config` doesn't have exactly as many entries as `DataConfig` has
                attributes, or if a setting is invalid.
            TypeError: If a key is not an attribute name, or a value has the wrong type.
        """
        if len(config) != len(DataConfig.default().to_dict()):
            raise ValueError("config has the wrong number of entries.")
        return DataConfig(**config)

    def to_dict(self) -> dict:
        """The config as a dictionary of its attributes, by name.

        Returns:
            The attributes, suitable for `from_dict`.
        """
        return asdict(self)

    def __repr__(self) -> str:
        def format_tuple(t):
            """Format tuple with floats to 2 decimal places or scientific notation if very small."""
            formatted = []
            for x in t:
                if isinstance(x, float):
                    if abs(x) < 1e-3 and x != 0:
                        formatted.append(f"{x:.2e}")
                    else:
                        formatted.append(f"{x:.2f}")
                else:
                    formatted.append(str(x))
            return f"({', '.join(formatted)})"

        def format_float(f):
            """Format float to 2 decimal places, or scientific notation if very small."""
            if abs(f) < 1e-3 and f != 0:
                return f"{f:.2e}"
            else:
                return f"{f:.2f}"

        lines = [f"{self.__class__.__name__}("]

        lines = [f"{self.__class__.__name__} ({hex(id(self))}):"]

        pad = 26

        # Data clipping ranges
        lines.append("    Data clipping ranges:")
        lines.append(f"        {'λ_range:':{pad}}{format_tuple(self.λ_range)}")
        lines.append(f"        {'α_range:':{pad}}{format_tuple(self.α_range)}")
        lines.append(f"        {'δ_range:':{pad}}{format_tuple(self.δ_range)}")

        # Bad data handling
        lines.append("    Bad data handling:")
        lines.append(f"        {'nans_strategy:':{pad}}'{self.nans_strategy}'")
        lines.append(f"        {'F_bad_strategy:':{pad}}'{self.F_bad_strategy}'")
        lines.append(f"        {'F_range:':{pad}}{format_tuple(self.F_range)}")

        # Flagged data handling
        lines.append("    Flagged data handling:")
        lines.append(f"        {'fibre_status_include:':{pad}}{self.fibre_status_include}")
        lines.append(f"        {'apply_mask:':{pad}}{self.apply_mask}")

        # Flux normalisation
        lines.append("    Flux normalisation:")
        lines.append(f"        {'normalise_F_strategy:':{pad}}'{self.normalise_F_strategy}'")
        lines.append(
            f"        {'normalise_F_offset:':{pad}}{format_float(self.normalise_F_offset)}"
        )
        lines.append(f"        {'normalise_F_scale:':{pad}}{format_float(self.normalise_F_scale)}")

        # Coordinate normalisation
        lines.append("    Coordinate normalisation:")
        lines.append(f"        {'normalise_αδ_strategy:':{pad}}'{self.normalise_αδ_strategy}'")
        lines.append(
            f"        {'normalise_α_offset:':{pad}}{format_float(self.normalise_α_offset)}"
        )
        lines.append(f"        {'normalise_α_scale:':{pad}}{format_float(self.normalise_α_scale)}")
        lines.append(
            f"        {'normalise_δ_offset:':{pad}}{format_float(self.normalise_δ_offset)}"
        )
        lines.append(f"        {'normalise_δ_scale:':{pad}}{format_float(self.normalise_δ_scale)}")

        return "\n".join(lines)
