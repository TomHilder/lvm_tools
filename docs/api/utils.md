# Utilities API

This module provides utility functions for physical calculations and spatial operations.

## Barycentric Corrections

#### `function lvm_tools.physical_properties.barycentric_corr.get_v_barycentric(mjd, α, δ, unit="km/s")`

Compute barycentric velocity correction.

Uses the Las Campanas Observatory location and the astropy radial velocity correction framework.

- **`mjd`**: Modified Julian Date(s)

    *(type: `ArrayLike`)*

- **`α`**: Right ascension in degrees

    *(type: `ArrayLike`)*

- **`δ`**: Declination in degrees

    *(type: `ArrayLike`)*

- **`unit`**: Output unit (default "km/s")

    *(type: `str`)*

**Returns:** Barycentric velocity correction

**Return type:** `numpy.ndarray`

**Example**:

``` python
from lvm_tools.physical_properties.barycentric_corr import get_v_barycentric
import numpy as np

# Single observation
v_bary = get_v_barycentric(
    mjd=60000.5,
    α=180.0,
    δ=-30.0,
)
print(f"v_bary = {v_bary:.2f} km/s")

# Multiple observations
v_bary = get_v_barycentric(
    mjd=np.array([60000.5, 60001.5]),
    α=np.array([180.0, 180.1]),
    δ=np.array([-30.0, -30.1]),
)
```

## Spatial Masking

#### `function lvm_tools.utils.mask.mask_near_points(xgrid, ygrid, xpoints, ypoints, threshold=None)`

Generate a boolean mask for grid cells near data points.

Uses a KD-tree for efficient nearest-neighbour lookup.

- **`xgrid`**: Grid coordinates along x-axis (monotonically increasing)

    *(type: `numpy.ndarray`)*

- **`ygrid`**: Grid coordinates along y-axis (monotonically increasing)

    *(type: `numpy.ndarray`)*

- **`xpoints`**: X-coordinates of data points

    *(type: `numpy.ndarray`)*

- **`ypoints`**: Y-coordinates of data points

    *(type: `numpy.ndarray`)*

- **`threshold`**: Maximum distance to be considered "near". If None, uses 1.5 times the maximum of mean grid spacings

    *(type: `float \| None`)*

**Returns:** Boolean mask, shape `(len(ygrid), len(xgrid))`, True means "keep"

**Return type:** `numpy.ndarray`

**Example**:

``` python
import numpy as np
from lvm_tools.utils.mask import mask_near_points

# Create a regular grid
xgrid = np.linspace(0, 10, 100)
ygrid = np.linspace(0, 10, 100)

# Scattered data points
xpoints = np.random.uniform(2, 8, 50)
ypoints = np.random.uniform(2, 8, 50)

# Generate mask
mask = mask_near_points(xgrid, ygrid, xpoints, ypoints)

# Apply to a 2D array
data = np.ones((len(ygrid), len(xgrid)))
masked_data = np.where(mask, data, np.nan)
```

**Use Case**:

This function is useful for creating masks that restrict model evaluation to regions with data coverage, avoiding extrapolation in sparse regions.

## Helper Functions

### Array Conversion

#### `function lvm_tools.data.helper.daskify_native(array, chunks)`

no-index  

Convert an array to a Dask array with native byte order.

The array (typically a FITS memmap) is read lazily: computing a slice reads and byte-swaps only the chunks it touches, not the whole array.

Handles FITS files which may store data in non-native byte order.

- **`array`**: Input array

    *(type: `ArrayLike`)*

- **`chunks`**: Dask chunk specification

    *(type: `str \| int \| tuple`)*

**Returns:** Dask array

**Return type:** `dask.array.Array`

#### `function lvm_tools.data.helper.numpyfy_native(array)`

no-index  

Convert an array to NumPy with native byte order.

- **`array`**: Input array

    *(type: `ArrayLike`)*

**Returns:** NumPy array

**Return type:** `numpy.ndarray`

### Dataset Utilities

#### `function lvm_tools.data.helper.summarize_with_units(ds)`

Generate a formatted string summary of an xarray Dataset.

Includes dimensions, coordinates with units, and data variables with chunk information for Dask arrays.

- **`ds`**: Dataset to summarise

    *(type: `xarray.Dataset`)*

**Returns:** Formatted summary string

**Return type:** `str`

#### `function lvm_tools.data.helper.convert_sci_to_int(arr)`

Convert IFU label strings to integer indices.

- **`arr`**: Array of IFU labels ("Sci1", "Sci2", "Sci3")

    *(type: `ArrayLike`)*

**Returns:** Integer indices (0, 1, 2)

**Return type:** `numpy.ndarray`

### Coordinate Utilities

#### `function lvm_tools.data.coordinates.get_mjd(header)`

no-index  

Extract observation midpoint MJD from FITS header.

Uses INTSTART and INTEND keywords to compute the midpoint.

- **`header`**: FITS primary header

    *(type: `astropy.io.fits.Header`)*

**Returns:** Modified Julian Date

**Return type:** `float`

**Raises** `ValueError`: If time keywords are missing

#### `function lvm_tools.data.coordinates.get_observatory_code(header)`

Extract observatory code from FITS header.

- **`header`**: FITS primary header

    *(type: `astropy.io.fits.Header`)*

**Returns:** Observatory code (e.g., "LCO")

**Return type:** `str`

**Raises** `ValueError`: If observatory is unknown

#### `function lvm_tools.data.coordinates.get_observatory_location(observatory)`

no-index  

Get Earth location for a named observatory.

- **`observatory`**: Observatory code

    *(type: `str`)*

**Returns:** Observatory location

**Return type:** `astropy.coordinates.EarthLocation`

**Raises** `ValueError`: If observatory is unknown

Currently supported: "LCO" (Las Campanas Observatory)
