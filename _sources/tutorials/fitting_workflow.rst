Tutorial: Model Fitting Workflow
================================

This tutorial demonstrates a complete workflow from data loading to model-ready arrays.

Overview
--------

The typical **lvm_tools** workflow consists of:

1. Load raw data from FITS files
2. Configure data processing parameters
3. Build normalised arrays for fitting
4. Access data properties for model construction

This tutorial covers each step in detail.

Step 1: Load Data
-----------------

Begin by loading one or more tiles:

.. code-block:: python

   from pathlib import Path
   from lvm_tools import LVMTile, LVMTileCollection

   # Single tile
   tile = LVMTile.from_file(Path("/path/to/observation.fits"))

   # Multiple tiles (e.g., dithered observations)
   paths = list(Path("/path/to/data/").glob("*.fits"))
   tiles = [LVMTile.from_file(p) for p in sorted(paths)]
   collection = LVMTileCollection.from_tiles(tiles)

Step 2: Configure Processing
----------------------------

Create a ``DataConfig`` that matches your analysis requirements:

.. code-block:: python

   from lvm_tools import DataConfig

   # For emission line analysis
   config = DataConfig.from_tiles(
       collection,
       λ_range=(6500.0, 6800.0),  # H-alpha region
       nans_strategy="pixel",
       F_bad_strategy="spaxel",
       normalise_F_strategy="98 only",
   )

   # Inspect the configuration
   print(config)

Step 3: Build FitData
---------------------

Combine tiles and configuration to produce fit-ready data:

.. code-block:: python

   from lvm_tools import FitDataBuilder

   # Create builder
   builder = FitDataBuilder(tiles=collection, config=config)

   # Build the data (triggers Dask computation)
   fit_data = builder.build()

   # Record the pipeline hash for reproducibility
   pipeline_hash = builder.hash()
   print(f"Pipeline: {pipeline_hash[:16]}...")

Step 4: Access Data Arrays
--------------------------

``FitData`` provides normalised JAX arrays:

Spectral Data
~~~~~~~~~~~~~

.. code-block:: python

   # Normalised flux (NaN replaced with 0)
   flux = fit_data.flux
   print(f"Flux shape: {flux.shape}")

   # Inverse variance (NaN replaced with small value)
   ivar = fit_data.i_var

   # Uncertainty (standard deviation)
   sigma = fit_data.u_flux

   # Wavelength grid
   wavelength = fit_data.λ
   print(f"Wavelength range: {float(wavelength.min()):.1f} - {float(wavelength.max()):.1f} Å")

   # Line spread function
   lsf = fit_data.lsf_σ

Spatial Data
~~~~~~~~~~~~

.. code-block:: python

   # Normalised coordinates (mapped to [-π, π])
   alpha = fit_data.α  # Right ascension
   delta = fit_data.δ  # Declination

   print(f"Number of spaxels: {len(alpha)}")

   # Convert back to physical coordinates
   ra_physical = fit_data.predict_α(alpha)
   dec_physical = fit_data.predict_δ(delta)

Masking
~~~~~~~

.. code-block:: python

   # Boolean mask (True where data is valid)
   mask = fit_data.mask
   print(f"Valid pixels: {float(mask.sum())} / {mask.size}")

   # Apply mask in computations
   masked_flux = flux * mask

Indexing
~~~~~~~~

.. code-block:: python

   # Wavelength indices
   λ_idx = fit_data.λ_idx

   # Spaxel indices
   spaxel_idx = fit_data.spaxel_idx

   # Tile indices (for multi-tile data)
   tile_idx = fit_data.tile_idx

   # IFU bundle indices
   ifu_idx = fit_data.ifu_idx

   # Fibre indices
   fibre_idx = fit_data.fibre_idx

Physical Properties
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Observation time
   mjd = fit_data.mjd
   print(f"MJD: {float(mjd.mean()):.4f}")

   # Barycentric velocity correction
   v_bary = fit_data.v_bary
   print(f"v_bary range: {float(v_bary.min()):.2f} - {float(v_bary.max()):.2f} km/s")

Step 5: Model Fitting
---------------------

Using with JAX Models
~~~~~~~~~~~~~~~~~~~~~

The arrays are ready for JAX-based model fitting:

.. code-block:: python

   import jax.numpy as jnp

   # Example: compute mean spectrum
   mean_spectrum = jnp.nanmean(fit_data.flux, axis=0)

   # Example: compute spatial weighted mean
   weights = fit_data.mask.astype(jnp.float32)
   weighted_mean = jnp.sum(fit_data.flux * weights, axis=0) / jnp.sum(weights, axis=0)

Using with spectracles
~~~~~~~~~~~~~~~~~~~~~~

When ``spectracles`` is installed, access the spatial data structure directly:

.. code-block:: python

   # Get spectracles-compatible spatial data
   spatial_data = fit_data.αδ_data

   # Use with spectracles models
   # from spectracles import SomeSpatialModel
   # model = SomeSpatialModel(spatial_data, ...)

Reproducibility
---------------

The ``FitDataBuilder.hash()`` method generates a unique identifier:

.. code-block:: python

   import json

   # Generate hash
   pipeline_hash = builder.hash()

   # Save configuration for reproducibility
   metadata = {
       "hash": pipeline_hash,
       "config": config.to_dict(),
       "n_tiles": len(collection.meta),
       "exposures": list(collection.meta.keys()),
   }

   with open("pipeline_metadata.json", "w") as f:
       json.dump(metadata, f, indent=2, default=str)

Memory Management
-----------------

For large datasets, manage memory carefully:

.. code-block:: python

   import gc

   # Process in batches
   batch_size = 100
   for i in range(0, len(paths), batch_size):
       batch_paths = paths[i:i+batch_size]
       batch_tiles = [LVMTile.from_file(p) for p in batch_paths]
       batch_collection = LVMTileCollection.from_tiles(batch_tiles)

       # Configure and build
       config = DataConfig.from_tiles(batch_collection, λ_range=(6500.0, 6800.0))
       fit_data = FitDataBuilder(batch_collection, config).build()

       # Process this batch
       # ...

       # Clean up
       del batch_tiles, batch_collection, fit_data
       gc.collect()

Complete Example
----------------

.. code-block:: python

   from pathlib import Path
   import json
   from lvm_tools import LVMTile, LVMTileCollection, DataConfig, FitDataBuilder

   # 1. Load data
   data_dir = Path("/path/to/lvm/data/")
   paths = sorted(data_dir.glob("lvm-drp-*.fits"))
   tiles = [LVMTile.from_file(p) for p in paths]
   collection = LVMTileCollection.from_tiles(tiles)

   print(f"Loaded {len(tiles)} tiles")
   print(collection)

   # 2. Configure processing
   config = DataConfig.from_tiles(
       collection,
       λ_range=(6500.0, 6800.0),
       nans_strategy="pixel",
       F_bad_strategy="spaxel",
       fibre_status_include=(0,),
       apply_mask=True,
       normalise_F_strategy="98 only",
       normalise_αδ_strategy="padded",
   )

   print("\nConfiguration:")
   print(config)

   # 3. Build fit data
   builder = FitDataBuilder(tiles=collection, config=config)
   fit_data = builder.build()

   print(f"\nFit data built successfully")
   print(f"  Flux shape: {fit_data.flux.shape}")
   print(f"  Wavelength points: {len(fit_data.λ)}")
   print(f"  Spaxels: {len(fit_data.α)}")

   # 4. Save metadata
   metadata = {
       "pipeline_hash": builder.hash(),
       "config": config.to_dict(),
       "n_tiles": len(collection.meta),
       "flux_shape": list(fit_data.flux.shape),
   }

   with open("analysis_metadata.json", "w") as f:
       json.dump(metadata, f, indent=2, default=str)

   # 5. Ready for model fitting
   # ... your analysis code here ...

Next Steps
----------

- :doc:`../api/fit_data`: Complete FitData API reference
- :doc:`../api/processing`: Processing utilities
