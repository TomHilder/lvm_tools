# Data Loading API

This module provides classes for loading and managing LVM DRP data.

## LVMTile

### `class lvm_tools.LVMTile`

Container for a single LVM DRP observation.

#### `data`

*Type:* `xarray.Dataset`

The observation data stored as an xarray Dataset with the following structure:

**Data Variables** (Dask arrays):

- `flux`: Flux cube, shape `(1, n_spaxels, n_wavelengths)`, units erg cm⁻² s⁻¹ Å⁻¹
- `i_var`: Inverse variance, same shape as flux
- `lsf_sigma`: Line spread function (sigma), same shape as flux, units Å
- `mask`: DRP quality mask, same shape as flux

**Coordinates**:

- `tile`: Tile index (length 1 for single tile)
- `spaxel`: Spaxel index
- `wavelength`: Wavelength grid in Angstroms
- `mjd`: Modified Julian Date of observation midpoint
- `ra`: Right ascension of each spaxel, shape `(1, n_spaxels)`, units degrees
- `dec`: Declination of each spaxel, shape `(1, n_spaxels)`, units degrees
- `fibre_id`: Fibre identifier
- `ifu_label`: IFU bundle label ("Sci1", "Sci2", "Sci3")
- `fibre_status`: Fibre quality flag (0=good)

#### `meta`

*Type:* `LVMTileMeta`

Metadata for the observation.

#### `classmethod from_file(drp_file)`

Load a tile from a DRP FITS file.

- **`drp_file`**: Path to the FITS file

    *(type: `Path \| str`)*

**Returns:** Loaded tile

**Return type:** `LVMTile`

**Raises** `FileNotFoundError`: If the file does not exist

**Example**:

``` python
from pathlib import Path
from lvm_tools import LVMTile

tile = LVMTile.from_file(Path("/data/lvm-drp-12345-00001.fits"))
```

## LVMTileMeta

### `class lvm_tools.data.tile.LVMTileMeta`

Metadata container for an LVM observation.

#### `filename`

*Type:* `str`

Name of the source FITS file.

#### `tile_id`

*Type:* `int`

LVM tile identifier.

#### `exp_num`

*Type:* `int`

Exposure number within the tile.

#### `drp_ver`

*Type:* `str`

Data Reduction Pipeline version string.

## LVMTileCollection

### `class lvm_tools.LVMTileCollection`

Container for multiple LVM observations combined along the tile dimension.

#### `data`

*Type:* `xarray.Dataset`

Combined observation data. Same structure as `LVMTile.data` but with `tile` dimension length equal to the number of tiles.

#### `meta`

*Type:* `Mapping[int, LVMTileMeta]`

Dictionary mapping exposure number to metadata for each tile.

#### `classmethod from_tiles(tiles)`

Combine multiple tiles into a collection.

- **`tiles`**: List of tiles to combine

    *(type: `list[LVMTile]`)*

**Returns:** Combined collection

**Return type:** `LVMTileCollection`

**Example**:

``` python
from lvm_tools import LVMTile, LVMTileCollection

tiles = [LVMTile.from_file(p) for p in paths]
collection = LVMTileCollection.from_tiles(tiles)

print(f"Combined shape: {collection.data['flux'].shape}")
```

## Type Aliases

#### `lvm_tools.data.tile.LVMTileLike`

Union type for `LVMTile | LVMTileCollection`.

Functions accepting `LVMTileLike` can operate on either single tiles or collections.

## Physical Constants

The following constants are defined in `lvm_tools.data.tile`:

#### `SIGMA_TO_FWHM`

*Type:* `float`

Conversion factor from Gaussian sigma to FWHM: `2 * sqrt(2 * ln(2))`.

#### `FWHM_TO_SIGMA`

*Type:* `float`

Conversion factor from FWHM to Gaussian sigma.

#### `FLUX_UNIT`

*Type:* `astropy.units.Unit`

Physical unit for flux: erg cm⁻² s⁻¹ Å⁻¹.

#### `SPECTRAL_UNIT`

*Type:* `astropy.units.Unit`

Physical unit for wavelength: Angstrom.

#### `SPATIAL_UNIT`

*Type:* `astropy.units.Unit`

Physical unit for coordinates: degree.

## Helper Functions

#### `function lvm_tools.data.helper.daskify_native(array, chunks)`

Convert an array to a Dask array with native byte order.

The array (typically a FITS memmap) is read lazily: computing a slice reads and byte-swaps only the chunks it touches, not the whole array.

- **`array`**: Input array

    *(type: `ArrayLike`)*

- **`chunks`**: Chunk specification for Dask

    *(type: `str \| int \| tuple`)*

**Returns:** Dask array with native byte order

**Return type:** `dask.array.Array`

#### `function lvm_tools.data.helper.numpyfy_native(array)`

Convert an array to NumPy with native byte order.

- **`array`**: Input array

    *(type: `ArrayLike`)*

**Returns:** NumPy array with native byte order

**Return type:** `numpy.ndarray`

## Coordinate Utilities

#### `function lvm_tools.data.coordinates.get_mjd(header)`

Extract the observation midpoint MJD from a FITS header.

- **`header`**: FITS header containing INTSTART and INTEND keywords

    *(type: `astropy.io.fits.Header`)*

**Returns:** Modified Julian Date of observation midpoint

**Return type:** `float`

**Raises** `ValueError`: If time information is not found

#### `function lvm_tools.data.coordinates.get_observatory_location(observatory)`

Get the Earth location for an observatory.

- **`observatory`**: Observatory code (e.g., "LCO")

    *(type: `str`)*

**Returns:** Observatory location

**Return type:** `astropy.coordinates.EarthLocation`

**Raises** `ValueError`: If observatory is unknown
