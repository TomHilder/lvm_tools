# Quick Start Guide

This guide provides a rapid introduction to the core functionality of **lvm_tools**. For more detailed explanations, see the [Core Concepts](concepts.md) page and the [Tutorial: Loading LVM Data](../tutorials/loading_data.md) tutorial.

## Loading Data

The fundamental unit of data in **lvm_tools** is the `LVMTile`, which encapsulates a single LVM DRP observation:

``` python
from pathlib import Path
from lvm_tools import LVMTile

# Load a tile from a DRP FITS file
tile = LVMTile.from_file(Path("/path/to/lvm-drp-0012345-00001.fits"))

# Inspect the tile
print(tile)
```

The tile contains flux, inverse variance, wavelength, and spatial coordinates stored in an xarray Dataset with lazy Dask arrays.

## Combining Multiple Tiles

For analyses spanning multiple observations, combine tiles into a collection:

``` python
from lvm_tools import LVMTile, LVMTileCollection

# Load multiple tiles
tiles = [LVMTile.from_file(Path(f"/path/to/tile_{i}.fits")) for i in range(3)]

# Combine into a collection
collection = LVMTileCollection.from_tiles(tiles)

# Access combined data
print(collection.data.dims)
```

## Configuring Data Processing

The `DataConfig` class specifies how data should be clipped, filtered, and normalised:

``` python
from lvm_tools import DataConfig

# Create a configuration from tile data
config = DataConfig.from_tiles(
    tile,
    λ_range=(6500.0, 6800.0),  # Select H-alpha region
)

# Inspect the configuration
print(config)
```

The `from_tiles` method automatically calculates normalisation parameters based on the data.

## Building Fit-Ready Data

The `FitDataBuilder` combines tiles and configuration to produce data ready for model fitting:

``` python
from lvm_tools import FitDataBuilder

# Build fit data
builder = FitDataBuilder(tiles=tile, config=config)
fit_data = builder.build()

# Access normalised arrays (JAX arrays)
flux = fit_data.flux  # Shape: (n_spaxels, n_wavelengths)
wavelength = fit_data.λ  # Shape: (n_wavelengths,)
alpha = fit_data.α  # Right ascension (normalised)
delta = fit_data.δ  # Declination (normalised)
```

## Reproducibility

Every processing pipeline can be hashed for reproducibility:

``` python
# Generate a unique hash for this configuration
pipeline_hash = builder.hash()
print(f"Pipeline hash: {pipeline_hash[:16]}...")
```

This hash changes if either the input tiles or the configuration changes.

## Complete Example

``` python
from pathlib import Path
from lvm_tools import LVMTile, DataConfig, FitDataBuilder

# Load data
tile = LVMTile.from_file(Path("/path/to/observation.fits"))

# Configure processing
config = DataConfig.from_tiles(
    tile,
    λ_range=(4800.0, 5100.0),  # Select OIII region
    nans_strategy="pixel",  # Exclude individual NaN pixels
    normalise_F_strategy="max only",  # Normalise by maximum flux
)

# Build fit-ready data
fit_data = FitDataBuilder(tile, config).build()

# Use with JAX-based models
print(f"Flux shape: {fit_data.flux.shape}")
print(f"Wavelength range: {fit_data.λ.min():.1f} - {fit_data.λ.max():.1f} Å")
```

## Next Steps

- [Core Concepts](concepts.md): Understand the design philosophy and data structures
- [Tutorial: Loading LVM Data](../tutorials/loading_data.md): Detailed tutorial on data loading
- [Tutorial: Data Configuration](../tutorials/data_configuration.md): Advanced configuration options
- [API Overview](../api/overview.md): Complete API reference
