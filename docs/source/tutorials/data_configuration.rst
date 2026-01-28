Tutorial: Data Configuration
============================

This tutorial covers the ``DataConfig`` class and its options for controlling data processing.

Overview
--------

``DataConfig`` is an immutable dataclass that specifies:

1. **Clipping**: Wavelength and spatial ranges to include
2. **Filtering**: Strategies for handling bad data
3. **Normalisation**: Methods for scaling flux and coordinates

Creating a Configuration
------------------------

Default Configuration
~~~~~~~~~~~~~~~~~~~~~

Create a configuration with default settings:

.. code-block:: python

   from lvm_tools import DataConfig

   config = DataConfig.default()
   print(config)

This uses the full wavelength range with standard filtering and normalisation.

Configuration from Data
~~~~~~~~~~~~~~~~~~~~~~~

The recommended approach is to derive configuration from your data:

.. code-block:: python

   from lvm_tools import LVMTile, DataConfig
   from pathlib import Path

   # Load tile
   tile = LVMTile.from_file(Path("/path/to/data.fits"))

   # Create configuration with automatic parameter calculation
   config = DataConfig.from_tiles(
       tile,
       λ_range=(6500.0, 6800.0),  # H-alpha region
   )

   print(config)

The ``from_tiles`` method:

- Sets ``α_range`` and ``δ_range`` from the tile's spatial extent
- Calculates normalisation offsets and scales from the data
- Applies any user-specified overrides

Wavelength Clipping
-------------------

Single Range
~~~~~~~~~~~~

Select a single wavelength range:

.. code-block:: python

   config = DataConfig.from_tiles(
       tile,
       λ_range=(4800.0, 5100.0),  # OIII region
   )

Multiple Ranges
~~~~~~~~~~~~~~~

Select multiple non-overlapping ranges:

.. code-block:: python

   config = DataConfig.from_tiles(
       tile,
       λ_range=(
           (4800.0, 4900.0),  # H-beta
           (4950.0, 5050.0),  # OIII doublet
       ),
   )

Ranges must be ordered and non-overlapping.

Spatial Clipping
----------------

Restrict to a spatial region:

.. code-block:: python

   config = DataConfig.from_tiles(
       tile,
       λ_range=(6500.0, 6800.0),
       α_range=(180.0, 181.0),  # RA in degrees
       δ_range=(-30.0, -29.0),  # Dec in degrees
   )

By default, ``from_tiles`` computes a bounding square that encompasses all data with 1% padding.

Handling Bad Data
-----------------

NaN Strategy
~~~~~~~~~~~~

Control how NaN values are handled:

.. code-block:: python

   # Exclude individual NaN pixels (default)
   config = DataConfig(
       nans_strategy="pixel",
       # ... other parameters
   )

   # Exclude entire spaxels containing any NaN
   config = DataConfig(
       nans_strategy="spaxel",
       # ... other parameters
   )

Bad Flux Strategy
~~~~~~~~~~~~~~~~~

Control how bad flux values are identified:

.. code-block:: python

   # Per-pixel threshold (strict)
   config = DataConfig(
       F_bad_strategy="pixel",
       F_range=(-1e-14, 1e-10),
       # ...
   )

   # Per-spaxel median threshold (tolerant)
   config = DataConfig(
       F_bad_strategy="spaxel",
       F_range=(-1e-14, 1e-10),
       # ...
   )

   # Per-spaxel maximum threshold
   config = DataConfig(
       F_bad_strategy="spaxel_max",
       F_range=(-1e-14, 1e-10),
       # ...
   )

The default ``F_range`` lower bound is ``-0.1e-13``, designed to exclude unphysical negative fluxes.

Fibre Status Filtering
~~~~~~~~~~~~~~~~~~~~~~

Include only specific fibre status values:

.. code-block:: python

   # Include only status 0 (default - good fibres)
   config = DataConfig(
       fibre_status_include=(0,),
       # ...
   )

   # Include multiple status values
   config = DataConfig(
       fibre_status_include=(0, 1),
       # ...
   )

DRP Mask
~~~~~~~~

Control whether to apply the DRP quality mask:

.. code-block:: python

   # Apply mask (default)
   config = DataConfig(
       apply_mask=True,
       # ...
   )

   # Ignore mask
   config = DataConfig(
       apply_mask=False,
       # ...
   )

Normalisation Strategies
------------------------

Flux Normalisation
~~~~~~~~~~~~~~~~~~

Several strategies are available for normalising flux:

.. code-block:: python

   # No normalisation
   config = DataConfig(normalise_F_strategy=None, ...)

   # Divide by maximum (default)
   config = DataConfig(normalise_F_strategy="max only", ...)

   # Divide by 98th percentile (robust to outliers)
   config = DataConfig(normalise_F_strategy="98 only", ...)

   # Shift to minimum, scale by range
   config = DataConfig(normalise_F_strategy="extrema", ...)

   # Centre on mean, scale by 2σ
   config = DataConfig(normalise_F_strategy="2σ", ...)

   # Shift and scale with padding
   config = DataConfig(normalise_F_strategy="padded", ...)

Coordinate Normalisation
~~~~~~~~~~~~~~~~~~~~~~~~

The same strategies apply to RA/Dec coordinates:

.. code-block:: python

   config = DataConfig(
       normalise_αδ_strategy="padded",  # Default
       # ...
   )

The ``padded`` strategy is recommended for coordinates as it ensures the domain has a small buffer.

Manual Normalisation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Override calculated normalisation parameters:

.. code-block:: python

   config = DataConfig.from_tiles(
       tile,
       λ_range=(6500.0, 6800.0),
       normalise_F_offset=0.0,
       normalise_F_scale=1e-14,
       normalise_α_offset=180.0,
       normalise_α_scale=1.0,
       normalise_δ_offset=-30.0,
       normalise_δ_scale=1.0,
   )

Configuration Serialisation
---------------------------

Convert to/from Dictionary
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # To dictionary
   config_dict = config.to_dict()
   print(config_dict)

   # From dictionary
   restored = DataConfig.from_dict(config_dict)

This enables saving configurations to JSON or YAML.

Configuration Validation
------------------------

All parameters are validated on instantiation:

.. code-block:: python

   # Invalid range (max < min)
   try:
       config = DataConfig(
           λ_range=(6800.0, 6500.0),  # Invalid!
           # ...
       )
   except ValueError as e:
       print(f"Validation error: {e}")

   # Invalid strategy
   try:
       config = DataConfig(
           nans_strategy="invalid_strategy",  # Unknown!
           # ...
       )
   except ValueError as e:
       print(f"Validation error: {e}")

Complete Example
----------------

.. code-block:: python

   from pathlib import Path
   from lvm_tools import LVMTile, DataConfig

   # Load data
   tile = LVMTile.from_file(Path("/path/to/data.fits"))

   # Create a comprehensive configuration
   config = DataConfig.from_tiles(
       tiles=tile,
       # Wavelength selection
       λ_range=(
           (4800.0, 4880.0),  # H-beta
           (4940.0, 5040.0),  # OIII
           (6540.0, 6600.0),  # H-alpha + NII
       ),
       # Filtering
       nans_strategy="pixel",
       F_bad_strategy="spaxel",
       F_range=(-1e-14, np.inf),
       fibre_status_include=(0,),
       apply_mask=True,
       # Normalisation
       normalise_F_strategy="98 only",
       normalise_αδ_strategy="padded",
   )

   print(config)

Best Practices
--------------

1. **Use** ``from_tiles`` when possible to automatically calculate normalisation parameters.
2. **Start with defaults** and adjust parameters incrementally.
3. **Document your configuration** by saving the dictionary representation.
4. **Use the hash** for tracking which configuration produced which results.

Next Steps
----------

- :doc:`fitting_workflow`: Apply configurations in a fitting pipeline
- :doc:`../api/config`: Complete API reference
