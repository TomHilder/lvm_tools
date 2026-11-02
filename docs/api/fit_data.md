# Fit Data API

This module provides the `FitData` class and `FitDataBuilder` for constructing model-ready data.

## FitDataBuilder

### `class lvm_tools.FitDataBuilder`

Builder class for constructing `FitData` with reproducibility tracking.

#### `tiles`

*Type:* `LVMTileLike`

Input tile or tile collection.

#### `config`

*Type:* `DataConfig`

Processing configuration.

#### `build()`

Build the fit-ready data.

This method:

1.  Processes tile data according to the configuration
2.  Clips wavelength and spatial ranges
3.  Filters bad data
4.  Flattens tile/spaxel dimensions
5.  Creates normalisation functions

**Returns:** Fit-ready data object

**Return type:** `FitData`

**Example**:

``` python
from lvm_tools import LVMTile, DataConfig, FitDataBuilder

tile = LVMTile.from_file(path)
config = DataConfig.from_tiles(tile, λ_range=(6500.0, 6800.0))

builder = FitDataBuilder(tiles=tile, config=config)
fit_data = builder.build()
```

#### `hash()`

Generate a SHA-256 hash of the processing pipeline.

The hash is computed from:

- Configuration parameters (serialised as sorted JSON)
- Tile metadata (filenames, exposure numbers, DRP versions)

**Returns:** Hexadecimal hash string

**Return type:** `str`

**Example**:

``` python
pipeline_hash = builder.hash()
print(f"Pipeline: {pipeline_hash[:16]}...")
```

## FitData

### `class lvm_tools.fit_data.FitData`

Container for normalised, model-ready spectroscopic data.

All array properties return JAX arrays suitable for GPU-accelerated computation.

**Data Properties**

#### `flux`

*Type:* `jax.Array`

Normalised flux array with NaN values replaced by zero.

Shape: `(n_spaxels, n_wavelengths)`

#### `i_var`

*Type:* `jax.Array`

Normalised inverse variance with NaN values replaced by a small value (1e-4).

Shape: `(n_spaxels, n_wavelengths)`

#### `u_flux`

*Type:* `jax.Array`

Flux uncertainty (inverse square root of variance).

Shape: `(n_spaxels, n_wavelengths)`

#### `λ`

*Type:* `jax.Array`

Wavelength grid in Angstroms.

Shape: `(n_wavelengths,)`

#### `lsf_σ`

*Type:* `jax.Array`

Line spread function sigma with NaN values replaced by median.

Shape: `(n_spaxels, n_wavelengths)`

#### `mask`

*Type:* `jax.Array`

Boolean mask where `True` indicates valid data.

Shape: `(n_spaxels, n_wavelengths)`

**Spatial Properties**

#### `α`

*Type:* `jax.Array`

Normalised right ascension, mapped to the domain [-π, π].

Shape: `(n_spaxels,)`

#### `δ`

*Type:* `jax.Array`

Normalised declination, mapped to the domain [-π, π].

Shape: `(n_spaxels,)`

**Index Properties**

#### `λ_idx`

*Type:* `jax.Array`

Wavelength indices: `[0, 1, 2, ..., n_wavelengths-1]`.

#### `spaxel_idx`

*Type:* `jax.Array`

Spaxel indices: `[0, 1, 2, ..., n_spaxels-1]`.

#### `tile_idx`

*Type:* `jax.Array`

Tile indices, encoded as unique integers per tile.

Shape: `(n_spaxels,)`

#### `ifu_idx`

*Type:* `jax.Array`

IFU bundle indices, encoded as unique integers.

Shape: `(n_spaxels,)`

#### `fibre_idx`

*Type:* `jax.Array`

Fibre identifiers (original values, not re-indexed).

Shape: `(n_spaxels,)`

**Physical Properties**

#### `mjd`

*Type:* `jax.Array`

Modified Julian Date of observations.

Shape: `(n_tiles,)` or `(n_spaxels,)` depending on structure

#### `v_bary`

*Type:* `jax.Array`

Barycentric velocity correction in km/s.

Computed for Las Campanas Observatory using the observation time and pointing coordinates.

Shape: `(n_spaxels,)`

**Coordinate Transform Methods**

#### `predict_α(x)`

Convert normalised right ascension back to physical coordinates.

- **`x`**: Normalised RA values in [-π, π]

    *(type: `jax.Array`)*

**Returns:** Physical RA in degrees

**Return type:** `jax.Array`

#### `predict_δ(x)`

Convert normalised declination back to physical coordinates.

- **`x`**: Normalised Dec values in [-π, π]

    *(type: `jax.Array`)*

**Returns:** Physical Dec in degrees

**Return type:** `jax.Array`

**Sky positions in arcseconds**

#### `sky_origin`

*Type:* `tuple[float, float] \| None`

The (RA, Dec) in degrees to measure sky offsets from, or None for the centre of the spaxels' bounding box. Set it with `with_sky_origin`.

#### `sky_centre`

*Type:* `tuple[float, float]`

The (RA, Dec) in degrees that `sky_x` and `sky_y` are measured from.

#### `sky_x`

*Type:* `jax.Array`

Each spaxel's offset east of `sky_centre` in arcseconds, on the tangent plane.

#### `sky_y`

*Type:* `jax.Array`

Each spaxel's offset north of `sky_centre` in arcseconds, on the tangent plane.

#### `sky_to_radec(x, y)`

Convert offsets in arcseconds (east, north of `sky_centre`) to RA and Dec.

**Returns:** (RA, Dec) in degrees

**Return type:** `tuple[numpy.ndarray, numpy.ndarray]`

#### `with_sky_origin(ra, dec)`

The same data, with sky offsets measured from (RA, Dec) in degrees.

**Return type:** `FitData`

**spectracles Integration**

#### `sky_data`

*Type:* `spectracles.model.data.SpatialDataLVM`

The spaxel positions as arcsecond offsets (`sky_x`, `sky_y`), with their spaxel, tile, IFU and fibre indices, for spectracles models.

Only available when `spectracles` is installed.

**Example**:

``` python
# Requires: pip install lvm-tools[spectracles]
from spectracles import Domain

spaxels = fit_data.sky_data
domain = Domain.from_points(spaxels.x, spaxels.y)
```

#### `αδ_data`

*Type:* `spectracles.model.data.SpatialDataLVM`

The same, with the normalised coordinates in [-π, π] (for spectracles fields without a domain).

Only available when `spectracles` is installed.

**Example**:

``` python
# Requires: pip install lvm-tools[spectracles]
spatial_data = fit_data.αδ_data
```

**Internal Properties**

The following properties provide access to un-masked arrays:

#### `<span id="flux">flux</span>`

*Type:* `jax.Array`

Raw normalised flux (may contain NaN).

#### `<span id="i_var">i_var</span>`

*Type:* `jax.Array`

Raw normalised inverse variance (may contain NaN).

#### `<span id="u_flux">u_flux</span>`

*Type:* `jax.Array`

Raw uncertainty (may contain NaN).

#### `<span id="lsf_σ">lsf_σ</span>`

*Type:* `jax.Array`

Raw LSF sigma (may contain NaN).

## Domain Mapping

The `FitData` class maps normalised coordinates to the domain [-π, π]:

#### `function to_π_domain(x)`

Map values from [0, 1] to [-π, π].

- **`x`**: Values in [0, 1]

**Returns:** Values in [-π, π]

Formula: `x * 2π - π`

#### `function from_π_domain(x)`

Map values from [-π, π] back to [0, 1].

- **`x`**: Values in [-π, π]

**Returns:** Values in [0, 1]

Formula: `(x + π) / (2π)`

## Array Conversion

#### `function to_jax_array(arr, dtype=np.float64)`

Convert an xarray DataArray to a JAX array.

- **`arr`**: Input DataArray

    *(type: `xarray.DataArray`)*

- **`dtype`**: Target data type

    *(type: `numpy.dtype`)*

**Returns:** JAX array

**Return type:** `jax.Array`

## Example Usage

``` python
from pathlib import Path
from lvm_tools import LVMTile, DataConfig, FitDataBuilder

# Load and configure
tile = LVMTile.from_file(Path("/data/observation.fits"))
config = DataConfig.from_tiles(tile, λ_range=(6500.0, 6800.0))

# Build
builder = FitDataBuilder(tiles=tile, config=config)
fit_data = builder.build()

# Use JAX arrays for computation
import jax.numpy as jnp

# Weighted mean spectrum
weights = fit_data.mask.astype(jnp.float32)
mean_flux = jnp.sum(fit_data.flux * weights, axis=0) / jnp.sum(weights, axis=0)

# Physical coordinates from normalised
ra = fit_data.predict_α(fit_data.α)
dec = fit_data.predict_δ(fit_data.δ)

# Barycentric correction
v_bary = fit_data.v_bary
print(f"Mean barycentric velocity: {float(v_bary.mean()):.2f} km/s")
```
