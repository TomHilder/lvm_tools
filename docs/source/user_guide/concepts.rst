Core Concepts
=============

This section describes the design philosophy and key abstractions in **lvm_tools**.

Design Philosophy
-----------------

**lvm_tools** is built around three core principles:

1. **Lazy evaluation**: Data is loaded on-demand using Dask, minimising memory footprint for large datasets.
2. **Modularity**: Each processing step (clipping, filtering, normalisation) is independent and composable.
3. **Reproducibility**: Configurations are immutable dataclasses that can be hashed and serialised.

Data Flow
---------

The typical data flow in **lvm_tools** follows this pattern:

.. code-block:: text

   FITS File(s)
        │
        ▼
   LVMTile / LVMTileCollection    ─────────────────────────────────────┐
        │                                                               │
        │  (lazy xarray Dataset with Dask arrays)                      │
        ▼                                                               │
   DataConfig.from_tiles()        ←── λ_range, filtering options       │
        │                                                               │
        │  (calculates normalisation parameters)                       │
        ▼                                                               │
   FitDataBuilder(tiles, config)                                       │
        │                                                               │
        │  .build()                                                    │
        ▼                                                               │
   FitData                        ─────────────────────────────────────┘
        │
        │  (normalised JAX arrays ready for fitting)
        ▼
   Model Fitting (e.g., spectracles)

LVMTile and LVMTileCollection
-----------------------------

``LVMTile`` represents a single LVM observation, containing:

- **Flux cube**: 2D array of shape ``(n_spaxels, n_wavelengths)``
- **Inverse variance**: Corresponding uncertainty estimates
- **Mask**: Data quality flags from the DRP
- **LSF sigma**: Line spread function width at each pixel
- **Coordinates**: Wavelength grid, RA/Dec positions, fibre metadata

Data is stored in an xarray Dataset with Dask-backed arrays for memory efficiency:

.. code-block:: python

   tile = LVMTile.from_file(path)

   # Access the underlying Dataset
   ds = tile.data

   # Data variables (Dask arrays)
   ds["flux"]      # Shape: (1, n_spaxels, n_wavelengths)
   ds["i_var"]     # Inverse variance
   ds["lsf_sigma"] # Line spread function (sigma, not FWHM)
   ds["mask"]      # DRP quality mask

   # Coordinates
   ds["wavelength"]     # Wavelength grid (Angstroms)
   ds["ra"]             # Right ascension (degrees)
   ds["dec"]            # Declination (degrees)
   ds["fibre_status"]   # Fibre quality flags

``LVMTileCollection`` concatenates multiple tiles along the ``tile`` dimension, enabling multi-exposure analyses.

DataConfig
----------

``DataConfig`` is an immutable dataclass specifying all data processing parameters:

Clipping Parameters
~~~~~~~~~~~~~~~~~~~

- ``λ_range``: Wavelength range(s) to include. Can be a single tuple ``(min, max)`` or multiple ranges ``((min1, max1), (min2, max2))``.
- ``α_range``: Right ascension range in degrees.
- ``δ_range``: Declination range in degrees.

Filtering Parameters
~~~~~~~~~~~~~~~~~~~~

- ``nans_strategy``: How to handle NaN values.

  - ``"pixel"``: Exclude individual NaN pixels (default)
  - ``"spaxel"``: Exclude entire spaxels containing any NaN

- ``F_bad_strategy``: How to identify bad flux values.

  - ``"pixel"``: Apply flux threshold per pixel
  - ``"spaxel"``: Apply threshold to median flux per spaxel
  - ``"spaxel_max"``: Apply threshold to maximum flux per spaxel

- ``F_range``: Valid flux range ``(min, max)``.
- ``fibre_status_include``: Tuple of acceptable fibre status values.
- ``apply_mask``: Whether to apply the DRP mask.

Normalisation Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

- ``normalise_F_strategy``: Flux normalisation method.

  - ``None``: No normalisation
  - ``"max only"``: Divide by maximum value
  - ``"98 only"``: Divide by 98th percentile
  - ``"extrema"``: Shift to minimum, scale by range
  - ``"1σ"``, ``"2σ"``, ``"3σ"``: Centre on mean, scale by standard deviation
  - ``"padded"``: Shift and scale with 1% padding

- ``normalise_αδ_strategy``: Coordinate normalisation (same options).

The ``from_tiles`` class method automatically calculates normalisation offsets and scales based on the data.

FitDataBuilder
--------------

``FitDataBuilder`` combines tiles and configuration to produce fit-ready data. It provides:

- **Deterministic processing**: Same inputs always produce the same outputs.
- **Hash generation**: Unique identifier for each processing configuration.

.. code-block:: python

   builder = FitDataBuilder(tiles=tile, config=config)

   # Build the data
   fit_data = builder.build()

   # Generate reproducibility hash
   hash_value = builder.hash()

FitData
-------

``FitData`` provides normalised, JAX-compatible arrays for model fitting:

Properties
~~~~~~~~~~

- ``flux``: Normalised flux array (NaN replaced with 0)
- ``i_var``: Normalised inverse variance (NaN replaced with small value)
- ``u_flux``: Uncertainty (inverse square root of variance)
- ``λ``: Wavelength grid
- ``α``: Normalised right ascension (mapped to [-π, π])
- ``δ``: Normalised declination (mapped to [-π, π])
- ``lsf_σ``: Line spread function sigma
- ``mask``: Boolean mask (True where data is valid)
- ``mjd``: Modified Julian Date
- ``v_bary``: Barycentric velocity correction

Index Properties
~~~~~~~~~~~~~~~~

- ``λ_idx``: Wavelength indices
- ``spaxel_idx``: Spaxel indices
- ``tile_idx``: Tile indices (for multi-tile data)
- ``ifu_idx``: IFU indices
- ``fibre_idx``: Fibre indices

Coordinate Transformations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``predict_α`` and ``predict_δ`` methods convert normalised coordinates back to physical units:

.. code-block:: python

   # Get physical RA from normalised coordinate
   ra_physical = fit_data.predict_α(fit_data.α)

spectracles Integration
~~~~~~~~~~~~~~~~~~~~~~~

When ``spectracles`` is installed, the ``αδ_data`` property returns a ``SpatialDataLVM`` object for use with spectrospatial models.

Physical Constants
------------------

**lvm_tools** defines several physical constants and units:

- ``FLUX_UNIT``: erg cm⁻² s⁻¹ Å⁻¹
- ``SPECTRAL_UNIT``: Angstrom
- ``SPATIAL_UNIT``: degree
- ``SIGMA_TO_FWHM``: 2√(2 ln 2) ≈ 2.355
- ``FWHM_TO_SIGMA``: 1 / SIGMA_TO_FWHM
