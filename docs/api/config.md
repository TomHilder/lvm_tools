# Configuration API

This module provides the configuration system for data processing.

## DataConfig

### `class lvm_tools.DataConfig`

Immutable configuration object specifying all data processing parameters.

All parameters are validated on instantiation. Invalid values raise `ValueError` or `TypeError`.

**Clipping Parameters**

#### `λ_range`

*Type:* `Range \| Ranges`

value  
(-inf, inf)

Wavelength range(s) to include in Angstroms.

Single range: `(min, max)`

Multiple ranges: `((min1, max1), (min2, max2), ...)`

#### `α_range`

*Type:* `Range`

value  
(-inf, inf)

Right ascension range in degrees: `(min, max)`.

#### `δ_range`

*Type:* `Range`

value  
(-inf, inf)

Declination range in degrees: `(min, max)`.

**Filtering Parameters**

#### `nans_strategy`

*Type:* `ExcludeStrategy`

value  
"pixel"

Strategy for handling NaN values:

- `"pixel"`: Exclude individual NaN pixels
- `"spaxel"`: Exclude entire spaxels containing any NaN

#### `F_bad_strategy`

*Type:* `ExcludeStrategy`

value  
"spaxel"

Strategy for identifying bad flux values:

- `"pixel"`: Apply threshold per pixel
- `"spaxel"`: Apply threshold to median flux per spaxel
- `"spaxel_max"`: Apply threshold to maximum flux per spaxel

#### `F_range`

*Type:* `Range`

value  
(-0.1e-13, inf)

Valid flux range: `(min, max)`.

#### `fibre_status_include`

*Type:* `tuple[FibreStatus]`

value  
(0,)

Tuple of fibre status values to include. Status 0 indicates good fibres.

#### `apply_mask`

*Type:* `bool`

value  
True

Whether to apply the DRP quality mask.

**Flux Normalisation Parameters**

#### `normalise_F_strategy`

*Type:* `NormaliseStrategy`

value  
"max only"

Flux normalisation strategy:

- `None`: No normalisation
- `"max only"`: Divide by maximum value
- `"98 only"`: Divide by 98th percentile
- `"extrema"`: Shift to minimum, scale by range
- `"1σ"`, `"2σ"`, `"3σ"`: Centre on the mean, scale by the full width of ±Nσ (2N standard deviations)
- `"padded"`: Shift and scale to the data range, padded by 15% of the range on each side

#### `normalise_F_offset`

*Type:* `float`

value  
0.0

Offset for flux normalisation: `(flux - offset) / scale`.

#### `normalise_F_scale`

*Type:* `float`

value  
1.0

Scale for flux normalisation.

**Coordinate Normalisation Parameters**

#### `normalise_αδ_strategy`

*Type:* `NormaliseStrategy`

value  
"padded"

Coordinate normalisation strategy (same options as flux).

#### `normalise_α_offset`

*Type:* `float`

value  
0.0

Right ascension offset.

#### `normalise_α_scale`

*Type:* `float`

value  
1.0

Right ascension scale.

#### `normalise_δ_offset`

*Type:* `float`

value  
0.0

Declination offset.

#### `normalise_δ_scale`

*Type:* `float`

value  
1.0

Declination scale.

**Class Methods**

#### `classmethod default()`

Create a configuration with all default values.

**Returns:** Default configuration

**Return type:** `DataConfig`

**Example**:

``` python
config = DataConfig.default()
```

#### `classmethod from_tiles(tiles, λ_range=(-inf, inf), \*\*overrides)`

Create a configuration from tile data with automatic parameter calculation.

This method:

1.  Computes spatial ranges from the tile data
2.  Clips and filters data according to defaults
3.  Calculates normalisation parameters from the processed data
4.  Applies any user-specified overrides

- **`tiles`**: Input tile(s)

    *(type: `LVMTileLike`)*

- **`λ_range`**: Wavelength range(s)

    *(type: `Range \| Ranges`)*

- **`overrides`**: Additional parameters to override

**Returns:** Configured instance

**Return type:** `DataConfig`

**Example**:

``` python
config = DataConfig.from_tiles(
    tile,
    λ_range=(6500.0, 6800.0),
    normalise_F_strategy="98 only",
)
```

#### `classmethod from_dict(config)`

Create a configuration from a dictionary.

- **`config`**: Dictionary with all configuration keys

    *(type: `dict`)*

**Returns:** Configuration instance

**Return type:** `DataConfig`

**Raises** `ValueError`: If dictionary has incorrect number of entries

**Instance Methods**

#### `to_dict()`

Convert configuration to a dictionary.

**Returns:** Dictionary representation

**Return type:** `dict`

## Type Definitions

#### `lvm_tools.fit_data.clipping.Range`

Type alias for `tuple[float, float]` representing a (min, max) range.

#### `lvm_tools.fit_data.clipping.Ranges`

Type alias for `tuple[Range, ...]` representing multiple ranges.

#### `lvm_tools.fit_data.filtering.ExcludeStrategy`

Literal type: `None | "pixel" | "spaxel" | "spaxel_max"`.

#### `lvm_tools.fit_data.filtering.FibreStatus`

Literal type: `0 | 1 | 2 | 3`.

#### `lvm_tools.fit_data.normalisation.NormaliseStrategy`

Literal type: `None | "max only" | "98 only" | "extrema" | "1σ" | "2σ" | "3σ" | "padded"`.

## Validation Functions

#### `function lvm_tools.config.validation.validate_range(x_range)`

Validate a range tuple.

- **`x_range`**: Range to validate

    *(type: `Range`)*

**Raises** `ValueError`: If range has incorrect length or max \< min

#### `function lvm_tools.config.validation.validate_excl_strategy(strategy)`

Validate an exclusion strategy.

- **`strategy`**: Strategy to validate

    *(type: `ExcludeStrategy`)*

**Raises** `ValueError`: If strategy is not recognised

#### `function lvm_tools.config.validation.validate_norm_strategy(strategy)`

Validate a normalisation strategy.

- **`strategy`**: Strategy to validate

    *(type: `NormaliseStrategy`)*

**Raises** `ValueError`: If strategy is not recognised
