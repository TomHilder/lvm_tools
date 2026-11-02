Fit Data API
============

This module provides the ``FitData`` class and ``FitDataBuilder`` for constructing model-ready data.

FitDataBuilder
--------------

.. py:class:: lvm_tools.FitDataBuilder

   Builder class for constructing ``FitData`` with reproducibility tracking.

   .. py:attribute:: tiles
      :type: LVMTileLike

      Input tile or tile collection.

   .. py:attribute:: config
      :type: DataConfig

      Processing configuration.

   .. py:method:: build()

      Build the fit-ready data.

      This method:

      1. Processes tile data according to the configuration
      2. Clips wavelength and spatial ranges
      3. Filters bad data
      4. Flattens tile/spaxel dimensions
      5. Creates normalisation functions

      :returns: Fit-ready data object
      :rtype: FitData

      **Example**:

      .. code-block:: python

         from lvm_tools import LVMTile, DataConfig, FitDataBuilder

         tile = LVMTile.from_file(path)
         config = DataConfig.from_tiles(tile, λ_range=(6500.0, 6800.0))

         builder = FitDataBuilder(tiles=tile, config=config)
         fit_data = builder.build()

   .. py:method:: hash()

      Generate a SHA-256 hash of the processing pipeline.

      The hash is computed from:

      - Configuration parameters (serialised as sorted JSON)
      - Tile metadata (filenames, exposure numbers, DRP versions)

      :returns: Hexadecimal hash string
      :rtype: str

      **Example**:

      .. code-block:: python

         pipeline_hash = builder.hash()
         print(f"Pipeline: {pipeline_hash[:16]}...")

FitData
-------

.. py:class:: lvm_tools.fit_data.FitData

   Container for normalised, model-ready spectroscopic data.

   All array properties return JAX arrays suitable for GPU-accelerated computation.

   **Data Properties**

   .. py:attribute:: flux
      :type: jax.Array

      Normalised flux array with NaN values replaced by zero.

      Shape: ``(n_spaxels, n_wavelengths)``

   .. py:attribute:: i_var
      :type: jax.Array

      Normalised inverse variance with NaN values replaced by a small value (1e-4).

      Shape: ``(n_spaxels, n_wavelengths)``

   .. py:attribute:: u_flux
      :type: jax.Array

      Flux uncertainty (inverse square root of variance).

      Shape: ``(n_spaxels, n_wavelengths)``

   .. py:attribute:: λ
      :type: jax.Array

      Wavelength grid in Angstroms.

      Shape: ``(n_wavelengths,)``

   .. py:attribute:: lsf_σ
      :type: jax.Array

      Line spread function sigma with NaN values replaced by median.

      Shape: ``(n_spaxels, n_wavelengths)``

   .. py:attribute:: mask
      :type: jax.Array

      Boolean mask where ``True`` indicates valid data.

      Shape: ``(n_spaxels, n_wavelengths)``

   **Spatial Properties**

   .. py:attribute:: α
      :type: jax.Array

      Normalised right ascension, mapped to the domain [-π, π].

      Shape: ``(n_spaxels,)``

   .. py:attribute:: δ
      :type: jax.Array

      Normalised declination, mapped to the domain [-π, π].

      Shape: ``(n_spaxels,)``

   **Index Properties**

   .. py:attribute:: λ_idx
      :type: jax.Array

      Wavelength indices: ``[0, 1, 2, ..., n_wavelengths-1]``.

   .. py:attribute:: spaxel_idx
      :type: jax.Array

      Spaxel indices: ``[0, 1, 2, ..., n_spaxels-1]``.

   .. py:attribute:: tile_idx
      :type: jax.Array

      Tile indices, encoded as unique integers per tile.

      Shape: ``(n_spaxels,)``

   .. py:attribute:: ifu_idx
      :type: jax.Array

      IFU bundle indices, encoded as unique integers.

      Shape: ``(n_spaxels,)``

   .. py:attribute:: fibre_idx
      :type: jax.Array

      Fibre identifiers (original values, not re-indexed).

      Shape: ``(n_spaxels,)``

   **Physical Properties**

   .. py:attribute:: mjd
      :type: jax.Array

      Modified Julian Date of observations.

      Shape: ``(n_tiles,)`` or ``(n_spaxels,)`` depending on structure

   .. py:attribute:: v_bary
      :type: jax.Array

      Barycentric velocity correction in km/s.

      Computed for Las Campanas Observatory using the observation time
      and pointing coordinates.

      Shape: ``(n_spaxels,)``

   **Coordinate Transform Methods**

   .. py:method:: predict_α(x)

      Convert normalised right ascension back to physical coordinates.

      :param x: Normalised RA values in [-π, π]
      :type x: jax.Array
      :returns: Physical RA in degrees
      :rtype: jax.Array

   .. py:method:: predict_δ(x)

      Convert normalised declination back to physical coordinates.

      :param x: Normalised Dec values in [-π, π]
      :type x: jax.Array
      :returns: Physical Dec in degrees
      :rtype: jax.Array

   **Sky positions in arcseconds**

   .. py:attribute:: sky_origin
      :type: tuple[float, float] | None

      The (RA, Dec) in degrees to measure sky offsets from, or None for the
      centre of the spaxels' bounding box. Set it with ``with_sky_origin``.

   .. py:attribute:: sky_centre
      :type: tuple[float, float]

      The (RA, Dec) in degrees that ``sky_x`` and ``sky_y`` are measured from.

   .. py:attribute:: sky_x
      :type: jax.Array

      Each spaxel's offset east of ``sky_centre`` in arcseconds, on the tangent plane.

   .. py:attribute:: sky_y
      :type: jax.Array

      Each spaxel's offset north of ``sky_centre`` in arcseconds, on the tangent plane.

   .. py:method:: sky_to_radec(x, y)

      Convert offsets in arcseconds (east, north of ``sky_centre``) to RA and Dec.

      :returns: (RA, Dec) in degrees
      :rtype: tuple[numpy.ndarray, numpy.ndarray]

   .. py:method:: with_sky_origin(ra, dec)

      The same data, with sky offsets measured from (RA, Dec) in degrees.

      :rtype: FitData

   **spectracles Integration**

   .. py:attribute:: sky_data
      :type: spectracles.model.data.SpatialDataLVM

      The spaxel positions as arcsecond offsets (``sky_x``, ``sky_y``), with
      their spaxel, tile, IFU and fibre indices, for spectracles models.

      Only available when ``spectracles`` is installed.

      **Example**:

      .. code-block:: python

         # Requires: pip install lvm-tools[spectracles]
         from spectracles import Domain

         spaxels = fit_data.sky_data
         domain = Domain.from_points(spaxels.x, spaxels.y)

   .. py:attribute:: αδ_data
      :type: spectracles.model.data.SpatialDataLVM

      The same, with the normalised coordinates in [-π, π] (for spectracles
      fields without a domain).

      Only available when ``spectracles`` is installed.

      **Example**:

      .. code-block:: python

         # Requires: pip install lvm-tools[spectracles]
         spatial_data = fit_data.αδ_data

   **Internal Properties**

   The following properties provide access to un-masked arrays:

   .. py:attribute:: _flux
      :type: jax.Array

      Raw normalised flux (may contain NaN).

   .. py:attribute:: _i_var
      :type: jax.Array

      Raw normalised inverse variance (may contain NaN).

   .. py:attribute:: _u_flux
      :type: jax.Array

      Raw uncertainty (may contain NaN).

   .. py:attribute:: _lsf_σ
      :type: jax.Array

      Raw LSF sigma (may contain NaN).

Domain Mapping
--------------

The ``FitData`` class maps normalised coordinates to the domain [-π, π]:

.. py:function:: to_π_domain(x)

   Map values from [0, 1] to [-π, π].

   :param x: Values in [0, 1]
   :returns: Values in [-π, π]

   Formula: ``x * 2π - π``

.. py:function:: from_π_domain(x)

   Map values from [-π, π] back to [0, 1].

   :param x: Values in [-π, π]
   :returns: Values in [0, 1]

   Formula: ``(x + π) / (2π)``

Array Conversion
----------------

.. py:function:: to_jax_array(arr, dtype=np.float64)

   Convert an xarray DataArray to a JAX array.

   :param arr: Input DataArray
   :type arr: xarray.DataArray
   :param dtype: Target data type
   :type dtype: numpy.dtype
   :returns: JAX array
   :rtype: jax.Array

Example Usage
-------------

.. code-block:: python

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
