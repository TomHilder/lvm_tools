API Overview
============

This section provides complete reference documentation for the **lvm_tools** API.

Module Structure
----------------

**lvm_tools** is organised into the following submodules:

.. code-block:: text

   lvm_tools/
   ├── __init__.py          # Public API exports
   ├── config/              # Configuration classes
   │   ├── data_config.py   # DataConfig class
   │   └── validation.py    # Parameter validation
   ├── data/                # Data loading
   │   ├── tile.py          # LVMTile, LVMTileCollection
   │   ├── coordinates.py   # Time/location utilities
   │   └── helper.py        # Array conversion helpers
   ├── fit_data/            # Data processing
   │   ├── fit_data.py      # FitData class
   │   ├── builder.py       # FitDataBuilder class
   │   ├── clipping.py      # Wavelength/spatial clipping
   │   ├── filtering.py     # Bad data filtering
   │   ├── normalisation.py # Normalisation strategies
   │   └── processing.py    # Processing pipelines
   ├── physical_properties/ # Physical calculations
   │   └── barycentric_corr.py  # Barycentric corrections
   └── utils/               # Utilities
       └── mask.py          # Spatial masking

Public API
----------

The following classes and functions are exported from the top-level ``lvm_tools`` namespace:

.. code-block:: python

   from lvm_tools import (
       LVMTile,           # Single observation container
       LVMTileCollection, # Multi-observation container
       DataConfig,        # Processing configuration
       FitDataBuilder,    # Fit data constructor
   )

API Reference Pages
-------------------

.. toctree::
   :maxdepth: 2

   tile
   config
   fit_data
   processing
   utils

Quick Reference
---------------

LVMTile
~~~~~~~

.. code-block:: python

   from lvm_tools import LVMTile

   tile = LVMTile.from_file(path)
   tile.data       # xarray Dataset
   tile.meta       # LVMTileMeta

LVMTileCollection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from lvm_tools import LVMTileCollection

   collection = LVMTileCollection.from_tiles([tile1, tile2])
   collection.data  # Combined Dataset
   collection.meta  # Dict of LVMTileMeta

DataConfig
~~~~~~~~~~

.. code-block:: python

   from lvm_tools import DataConfig

   config = DataConfig.from_tiles(tiles, λ_range=(...))
   config = DataConfig.default()
   config.to_dict()
   DataConfig.from_dict(d)

FitDataBuilder
~~~~~~~~~~~~~~

.. code-block:: python

   from lvm_tools import FitDataBuilder

   builder = FitDataBuilder(tiles, config)
   fit_data = builder.build()
   hash_value = builder.hash()

FitData Properties
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   fit_data.flux      # Normalised flux (JAX array)
   fit_data.i_var     # Inverse variance
   fit_data.u_flux    # Uncertainty
   fit_data.λ         # Wavelength
   fit_data.α         # RA (normalised)
   fit_data.δ         # Dec (normalised)
   fit_data.lsf_σ     # Line spread function
   fit_data.mask      # Valid data mask
   fit_data.mjd       # Modified Julian Date
   fit_data.v_bary    # Barycentric velocity
   fit_data.αδ_data   # spectracles integration
