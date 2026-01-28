Processing API
==============

This module provides functions for data clipping, filtering, and normalisation.

Clipping Functions
------------------

.. py:function:: lvm_tools.fit_data.clipping.clip_dataset(data, λ_range, α_range, δ_range)

   Clip a dataset to specified ranges.

   :param data: Input dataset
   :type data: xarray.Dataset
   :param λ_range: Wavelength range(s)
   :type λ_range: Range | Ranges
   :param α_range: Right ascension range
   :type α_range: Range
   :param δ_range: Declination range
   :type δ_range: Range
   :returns: Clipped dataset
   :rtype: xarray.Dataset

.. py:function:: lvm_tools.fit_data.clipping.clip_wavelengths(data, λ_ranges)

   Clip dataset to specified wavelength ranges.

   Supports multiple non-overlapping ranges which are concatenated along
   the wavelength dimension.

   :param data: Input dataset
   :type data: xarray.Dataset
   :param λ_ranges: One or more wavelength ranges
   :type λ_ranges: Range | Ranges
   :returns: Wavelength-clipped dataset
   :rtype: xarray.Dataset

.. py:function:: lvm_tools.fit_data.clipping.bounding_square(x_min, x_max, y_min, y_max)

   Compute a bounding square with 1% padding.

   Used to ensure equal scales in RA and Dec for normalisation.

   :param x_min: Minimum x value
   :param x_max: Maximum x value
   :param y_min: Minimum y value
   :param y_max: Maximum y value
   :returns: Tuple of (x_range, y_range)
   :rtype: tuple[tuple[float, float], tuple[float, float]]

.. py:function:: lvm_tools.fit_data.clipping.ensure_ranges(ranges)

   Normalise range input to tuple of tuples.

   :param ranges: Single range or multiple ranges
   :type ranges: Range | Ranges
   :returns: Tuple of ranges
   :rtype: Ranges

Filtering Functions
-------------------

.. py:function:: lvm_tools.fit_data.filtering.filter_dataset(data, nans_strategy, F_bad_strategy, F_bad_range, fibre_status_include, apply_mask)

   Apply all filtering operations to a dataset.

   Bad data is set to NaN in the output.

   :param data: Input dataset
   :type data: xarray.Dataset
   :param nans_strategy: NaN handling strategy
   :type nans_strategy: ExcludeStrategy
   :param F_bad_strategy: Bad flux handling strategy
   :type F_bad_strategy: ExcludeStrategy
   :param F_bad_range: Valid flux range
   :type F_bad_range: tuple[float, float]
   :param fibre_status_include: Valid fibre status values
   :type fibre_status_include: tuple[FibreStatus]
   :param apply_mask: Whether to apply DRP mask
   :type apply_mask: bool
   :returns: Filtered dataset
   :rtype: xarray.Dataset

.. py:function:: lvm_tools.fit_data.filtering.get_where_nan(arr)

   Get boolean mask of NaN locations.

   :param arr: Input array
   :type arr: xarray.DataArray
   :returns: Boolean mask (True where NaN)
   :rtype: xarray.DataArray

.. py:function:: lvm_tools.fit_data.filtering.get_where_bad(arr, bad_range)

   Get mask of values outside the valid range.

   :param arr: Input array
   :type arr: xarray.DataArray
   :param bad_range: Valid range (min, max)
   :type bad_range: tuple[float, float]
   :returns: Boolean mask (True where bad)
   :rtype: xarray.DataArray

.. py:function:: lvm_tools.fit_data.filtering.get_where_bad_median(arr, bad_range)

   Get mask of spaxels with median outside valid range.

   :param arr: Input array
   :type arr: xarray.DataArray
   :param bad_range: Valid range for median
   :type bad_range: tuple[float, float]
   :returns: Boolean mask per spaxel
   :rtype: xarray.DataArray

.. py:function:: lvm_tools.fit_data.filtering.get_where_badfib(fib_stat_arr, fibre_status_incl)

   Get mask of spaxels with excluded fibre status.

   :param fib_stat_arr: Fibre status values
   :type fib_stat_arr: xarray.DataArray
   :param fibre_status_incl: Included status values
   :type fibre_status_incl: tuple[FibreStatus]
   :returns: Boolean mask (True where excluded)
   :rtype: xarray.DataArray

.. py:function:: lvm_tools.fit_data.filtering.filter_inspector(data, F_bad_range, fibre_status_include)

   Generate diagnostic statistics about data quality.

   Returns counts of pixels/spaxels affected by each filter.

   :param data: Input dataset
   :type data: xarray.Dataset
   :param F_bad_range: Valid flux range
   :type F_bad_range: tuple[float, float]
   :param fibre_status_include: Valid fibre status values
   :type fibre_status_include: tuple[FibreStatus]
   :returns: Dictionary with filter statistics
   :rtype: dict

   **Example**:

   .. code-block:: python

      stats = filter_inspector(data, (-1e-14, np.inf), (0,))
      print(f"NaN pixels: {stats['nans'][0]}")
      print(f"Bad fibre spaxels: {stats['fibre status'][1]}")

Normalisation Functions
-----------------------

.. py:function:: lvm_tools.fit_data.normalisation.calc_normalisation(data, strategy)

   Calculate normalisation offset and scale.

   :param data: Input array
   :type data: ArrayLike
   :param strategy: Normalisation strategy
   :type strategy: NormaliseStrategy
   :returns: Tuple of (offset, scale)
   :rtype: tuple[float, float]

   Strategies:

   - ``None``: Returns (0.0, 1.0)
   - ``"max only"``: Returns (0.0, max(data))
   - ``"98 only"``: Returns (0.0, percentile(data, 98))
   - ``"extrema"``: Returns (min(data), max(data) - min(data))
   - ``"1σ"``: Returns (mean(data), 2 * std(data))
   - ``"2σ"``: Returns (mean(data), 4 * std(data))
   - ``"3σ"``: Returns (mean(data), 6 * std(data))
   - ``"padded"``: Returns (min - 0.01*range, 1.02*range)

.. py:function:: lvm_tools.fit_data.normalisation.normalise(data, offset, scale)

   Apply normalisation: ``(data - offset) / scale``.

   :param data: Input array
   :type data: ArrayLike
   :param offset: Normalisation offset
   :type offset: float
   :param scale: Normalisation scale
   :type scale: float
   :returns: Normalised array
   :rtype: ArrayLike

.. py:function:: lvm_tools.fit_data.normalisation.denormalise(data, offset, scale)

   Reverse normalisation: ``data * scale + offset``.

   :param data: Normalised array
   :type data: ArrayLike
   :param offset: Normalisation offset
   :type offset: float
   :param scale: Normalisation scale
   :type scale: float
   :returns: Original-scale array
   :rtype: ArrayLike

.. py:function:: lvm_tools.fit_data.normalisation.get_norm_funcs(offset, scale)

   Create partially-applied normalisation functions.

   :param offset: Normalisation offset
   :type offset: float
   :param scale: Normalisation scale
   :type scale: float
   :returns: Generator of (normalise_func, denormalise_func)
   :rtype: tuple[Callable, Callable]

Pipeline Functions
------------------

.. py:function:: lvm_tools.fit_data.processing.process_tile_data(tiles, config)

   Apply full processing pipeline to tile data.

   :param tiles: Input tile(s)
   :type tiles: LVMTileLike
   :param config: Processing configuration
   :type config: DataConfig
   :returns: Processed dataset
   :rtype: xarray.Dataset

.. py:function:: lvm_tools.fit_data.processing.get_αδ_ranges(tiles)

   Compute spatial bounding box from tile data.

   :param tiles: Input tile(s)
   :type tiles: LVMTileLike
   :returns: Tuple of (α_range, δ_range)
   :rtype: tuple[tuple[float, float], tuple[float, float]]

.. py:function:: lvm_tools.fit_data.processing.get_normalisations(ds, config)

   Calculate normalisation parameters from processed data.

   :param ds: Processed dataset
   :type ds: xarray.Dataset
   :param config: Processing configuration
   :type config: DataConfig
   :returns: Tuple of ((F_offset, F_scale), (α_offset, α_scale), (δ_offset, δ_scale))
   :rtype: tuple

.. py:function:: lvm_tools.fit_data.processing.flatten_tile_coord(ds)

   Flatten tile and spaxel dimensions into a single coordinate.

   :param ds: Input dataset with (tile, spaxel) dimensions
   :type ds: xarray.Dataset
   :returns: Dataset with flat_spaxel dimension
   :rtype: xarray.Dataset

Constants
---------

.. py:data:: lvm_tools.fit_data.filtering.BAD_FLUX_THRESHOLD
   :type: float
   :value: -0.1e-13

   Default lower threshold for valid flux values.

.. py:data:: lvm_tools.fit_data.normalisation.NORM_PADDING
   :type: float
   :value: 0.01

   Padding fraction for "padded" normalisation strategy.
