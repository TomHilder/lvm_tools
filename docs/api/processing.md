# Processing API

This module provides functions for data clipping, filtering, and normalisation.

## Clipping Functions

#### `function lvm_tools.fit_data.clipping.clip_dataset(data, λ_range, α_range, δ_range)`

Clip a dataset to specified ranges.

- **`data`**: Input dataset

    *(type: `xarray.Dataset`)*

- **`λ_range`**: Wavelength range(s)

    *(type: `Range \| Ranges`)*

- **`α_range`**: Right ascension range

    *(type: `Range`)*

- **`δ_range`**: Declination range

    *(type: `Range`)*

**Returns:** Clipped dataset

**Return type:** `xarray.Dataset`

#### `function lvm_tools.fit_data.clipping.clip_wavelengths(data, λ_ranges)`

Clip dataset to specified wavelength ranges.

Supports multiple non-overlapping ranges which are concatenated along the wavelength dimension.

- **`data`**: Input dataset

    *(type: `xarray.Dataset`)*

- **`λ_ranges`**: One or more wavelength ranges

    *(type: `Range \| Ranges`)*

**Returns:** Wavelength-clipped dataset

**Return type:** `xarray.Dataset`

#### `function lvm_tools.fit_data.clipping.bounding_square(x_min, x_max, y_min, y_max)`

Compute a bounding square with 1% padding.

Used to ensure equal scales in RA and Dec for normalisation.

- **`x_min`**: Minimum x value

- **`x_max`**: Maximum x value

- **`y_min`**: Minimum y value

- **`y_max`**: Maximum y value

**Returns:** Tuple of (x_range, y_range)

**Return type:** `tuple[tuple[float, float], tuple[float, float]]`

#### `function lvm_tools.fit_data.clipping.ensure_ranges(ranges)`

Normalise range input to tuple of tuples.

- **`ranges`**: Single range or multiple ranges

    *(type: `Range \| Ranges`)*

**Returns:** Tuple of ranges

**Return type:** `Ranges`

## Filtering Functions

#### `function lvm_tools.fit_data.filtering.filter_dataset(data, nans_strategy, F_bad_strategy, F_bad_range, fibre_status_include, apply_mask)`

Apply all filtering operations to a dataset.

Bad data is set to NaN in the output.

- **`data`**: Input dataset

    *(type: `xarray.Dataset`)*

- **`nans_strategy`**: NaN handling strategy

    *(type: `ExcludeStrategy`)*

- **`F_bad_strategy`**: Bad flux handling strategy

    *(type: `ExcludeStrategy`)*

- **`F_bad_range`**: Valid flux range

    *(type: `tuple[float, float]`)*

- **`fibre_status_include`**: Valid fibre status values

    *(type: `tuple[FibreStatus]`)*

- **`apply_mask`**: Whether to apply DRP mask

    *(type: `bool`)*

**Returns:** Filtered dataset

**Return type:** `xarray.Dataset`

#### `function lvm_tools.fit_data.filtering.get_where_nan(arr)`

Get boolean mask of NaN locations.

- **`arr`**: Input array

    *(type: `xarray.DataArray`)*

**Returns:** Boolean mask (True where NaN)

**Return type:** `xarray.DataArray`

#### `function lvm_tools.fit_data.filtering.get_where_bad(arr, bad_range)`

Get mask of values outside the valid range.

- **`arr`**: Input array

    *(type: `xarray.DataArray`)*

- **`bad_range`**: Valid range (min, max)

    *(type: `tuple[float, float]`)*

**Returns:** Boolean mask (True where bad)

**Return type:** `xarray.DataArray`

#### `function lvm_tools.fit_data.filtering.get_where_bad_median(arr, bad_range)`

Get mask of spaxels with median outside valid range.

- **`arr`**: Input array

    *(type: `xarray.DataArray`)*

- **`bad_range`**: Valid range for median

    *(type: `tuple[float, float]`)*

**Returns:** Boolean mask per spaxel

**Return type:** `xarray.DataArray`

#### `function lvm_tools.fit_data.filtering.get_where_badfib(fib_stat_arr, fibre_status_incl)`

Get mask of spaxels with excluded fibre status.

- **`fib_stat_arr`**: Fibre status values

    *(type: `xarray.DataArray`)*

- **`fibre_status_incl`**: Included status values

    *(type: `tuple[FibreStatus]`)*

**Returns:** Boolean mask (True where excluded)

**Return type:** `xarray.DataArray`

#### `function lvm_tools.fit_data.filtering.filter_inspector(data, F_bad_range, fibre_status_include)`

Generate diagnostic statistics about data quality.

Returns counts of pixels/spaxels affected by each filter.

- **`data`**: Input dataset

    *(type: `xarray.Dataset`)*

- **`F_bad_range`**: Valid flux range

    *(type: `tuple[float, float]`)*

- **`fibre_status_include`**: Valid fibre status values

    *(type: `tuple[FibreStatus]`)*

**Returns:** Dictionary with filter statistics

**Return type:** `dict`

**Example**:

``` python
stats = filter_inspector(data, (-1e-14, np.inf), (0,))
print(f"NaN pixels: {stats['nans'][0]}")
print(f"Bad fibre spaxels: {stats['fibre status'][1]}")
```

## Normalisation Functions

#### `function lvm_tools.fit_data.normalisation.calc_normalisation(data, strategy)`

Calculate normalisation offset and scale.

- **`data`**: Input array

    *(type: `ArrayLike`)*

- **`strategy`**: Normalisation strategy

    *(type: `NormaliseStrategy`)*

**Returns:** Tuple of (offset, scale)

**Return type:** `tuple[float, float]`

Strategies:

- `None`: Returns (0.0, 1.0)
- `"max only"`: Returns (0.0, max(data))
- `"98 only"`: Returns (0.0, percentile(data, 98))
- `"extrema"`: Returns (min(data), max(data) - min(data))
- `"1σ"`: Returns (mean(data), 2 \* std(data))
- `"2σ"`: Returns (mean(data), 4 \* std(data))
- `"3σ"`: Returns (mean(data), 6 \* std(data))
- `"padded"`: Returns (min - 0.15\*range, 1.3\*range)

#### `function lvm_tools.fit_data.normalisation.normalise(data, offset, scale)`

Apply normalisation: `(data - offset) / scale`.

- **`data`**: Input array

    *(type: `ArrayLike`)*

- **`offset`**: Normalisation offset

    *(type: `float`)*

- **`scale`**: Normalisation scale

    *(type: `float`)*

**Returns:** Normalised array

**Return type:** `ArrayLike`

#### `function lvm_tools.fit_data.normalisation.denormalise(data, offset, scale)`

Reverse normalisation: `data * scale + offset`.

- **`data`**: Normalised array

    *(type: `ArrayLike`)*

- **`offset`**: Normalisation offset

    *(type: `float`)*

- **`scale`**: Normalisation scale

    *(type: `float`)*

**Returns:** Original-scale array

**Return type:** `ArrayLike`

#### `function lvm_tools.fit_data.normalisation.get_norm_funcs(offset, scale)`

Create partially-applied normalisation functions.

- **`offset`**: Normalisation offset

    *(type: `float`)*

- **`scale`**: Normalisation scale

    *(type: `float`)*

**Returns:** Generator of (normalise_func, denormalise_func)

**Return type:** `tuple[Callable, Callable]`

## Pipeline Functions

#### `function lvm_tools.fit_data.processing.process_tile_data(tiles, config)`

Apply full processing pipeline to tile data.

- **`tiles`**: Input tile(s)

    *(type: `LVMTileLike`)*

- **`config`**: Processing configuration

    *(type: `DataConfig`)*

**Returns:** Processed dataset

**Return type:** `xarray.Dataset`

#### `function lvm_tools.fit_data.processing.get_αδ_ranges(tiles)`

Compute spatial bounding box from tile data.

- **`tiles`**: Input tile(s)

    *(type: `LVMTileLike`)*

**Returns:** Tuple of (α_range, δ_range)

**Return type:** `tuple[tuple[float, float], tuple[float, float]]`

#### `function lvm_tools.fit_data.processing.get_normalisations(ds, config)`

Calculate normalisation parameters from processed data.

- **`ds`**: Processed dataset

    *(type: `xarray.Dataset`)*

- **`config`**: Processing configuration

    *(type: `DataConfig`)*

**Returns:** Tuple of ((F_offset, F_scale), (α_offset, α_scale), (δ_offset, δ_scale))

**Return type:** `tuple`

#### `function lvm_tools.fit_data.processing.flatten_tile_coord(ds)`

Flatten tile and spaxel dimensions into a single coordinate.

- **`ds`**: Input dataset with (tile, spaxel) dimensions

    *(type: `xarray.Dataset`)*

**Returns:** Dataset with flat_spaxel dimension

**Return type:** `xarray.Dataset`

## Constants

#### `lvm_tools.fit_data.filtering.BAD_FLUX_THRESHOLD`

*Type:* `float`

Default lower threshold for valid flux values.

#### `lvm_tools.fit_data.normalisation.NORM_PADDING`

*Type:* `float`

Padding fraction for "padded" normalisation strategy.
