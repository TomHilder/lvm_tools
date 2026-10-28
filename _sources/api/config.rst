Configuration API
=================

This module provides the configuration system for data processing.

DataConfig
----------

.. py:class:: lvm_tools.DataConfig

   Immutable configuration object specifying all data processing parameters.

   All parameters are validated on instantiation. Invalid values raise
   ``ValueError`` or ``TypeError``.

   **Clipping Parameters**

   .. py:attribute:: λ_range
      :type: Range | Ranges
      :value: (-inf, inf)

      Wavelength range(s) to include in Angstroms.

      Single range: ``(min, max)``

      Multiple ranges: ``((min1, max1), (min2, max2), ...)``

   .. py:attribute:: α_range
      :type: Range
      :value: (-inf, inf)

      Right ascension range in degrees: ``(min, max)``.

   .. py:attribute:: δ_range
      :type: Range
      :value: (-inf, inf)

      Declination range in degrees: ``(min, max)``.

   **Filtering Parameters**

   .. py:attribute:: nans_strategy
      :type: ExcludeStrategy
      :value: "pixel"

      Strategy for handling NaN values:

      - ``"pixel"``: Exclude individual NaN pixels
      - ``"spaxel"``: Exclude entire spaxels containing any NaN

   .. py:attribute:: F_bad_strategy
      :type: ExcludeStrategy
      :value: "spaxel"

      Strategy for identifying bad flux values:

      - ``"pixel"``: Apply threshold per pixel
      - ``"spaxel"``: Apply threshold to median flux per spaxel
      - ``"spaxel_max"``: Apply threshold to maximum flux per spaxel

   .. py:attribute:: F_range
      :type: Range
      :value: (-0.1e-13, inf)

      Valid flux range: ``(min, max)``.

   .. py:attribute:: fibre_status_include
      :type: tuple[FibreStatus]
      :value: (0,)

      Tuple of fibre status values to include. Status 0 indicates good fibres.

   .. py:attribute:: apply_mask
      :type: bool
      :value: True

      Whether to apply the DRP quality mask.

   **Flux Normalisation Parameters**

   .. py:attribute:: normalise_F_strategy
      :type: NormaliseStrategy
      :value: "max only"

      Flux normalisation strategy:

      - ``None``: No normalisation
      - ``"max only"``: Divide by maximum value
      - ``"98 only"``: Divide by 98th percentile
      - ``"extrema"``: Shift to minimum, scale by range
      - ``"1σ"``, ``"2σ"``, ``"3σ"``: Centre on mean, scale by N standard deviations
      - ``"padded"``: Shift and scale with 1% padding

   .. py:attribute:: normalise_F_offset
      :type: float
      :value: 0.0

      Offset for flux normalisation: ``(flux - offset) / scale``.

   .. py:attribute:: normalise_F_scale
      :type: float
      :value: 1.0

      Scale for flux normalisation.

   **Coordinate Normalisation Parameters**

   .. py:attribute:: normalise_αδ_strategy
      :type: NormaliseStrategy
      :value: "padded"

      Coordinate normalisation strategy (same options as flux).

   .. py:attribute:: normalise_α_offset
      :type: float
      :value: 0.0

      Right ascension offset.

   .. py:attribute:: normalise_α_scale
      :type: float
      :value: 1.0

      Right ascension scale.

   .. py:attribute:: normalise_δ_offset
      :type: float
      :value: 0.0

      Declination offset.

   .. py:attribute:: normalise_δ_scale
      :type: float
      :value: 1.0

      Declination scale.

   **Class Methods**

   .. py:classmethod:: default()

      Create a configuration with all default values.

      :returns: Default configuration
      :rtype: DataConfig

      **Example**:

      .. code-block:: python

         config = DataConfig.default()

   .. py:classmethod:: from_tiles(tiles, λ_range=(-inf, inf), **overrides)

      Create a configuration from tile data with automatic parameter calculation.

      This method:

      1. Computes spatial ranges from the tile data
      2. Clips and filters data according to defaults
      3. Calculates normalisation parameters from the processed data
      4. Applies any user-specified overrides

      :param tiles: Input tile(s)
      :type tiles: LVMTileLike
      :param λ_range: Wavelength range(s)
      :type λ_range: Range | Ranges
      :param overrides: Additional parameters to override
      :returns: Configured instance
      :rtype: DataConfig

      **Example**:

      .. code-block:: python

         config = DataConfig.from_tiles(
             tile,
             λ_range=(6500.0, 6800.0),
             normalise_F_strategy="98 only",
         )

   .. py:classmethod:: from_dict(config)

      Create a configuration from a dictionary.

      :param config: Dictionary with all configuration keys
      :type config: dict
      :returns: Configuration instance
      :rtype: DataConfig
      :raises ValueError: If dictionary has incorrect number of entries

   **Instance Methods**

   .. py:method:: to_dict()

      Convert configuration to a dictionary.

      :returns: Dictionary representation
      :rtype: dict

Type Definitions
----------------

.. py:data:: lvm_tools.fit_data.clipping.Range

   Type alias for ``tuple[float, float]`` representing a (min, max) range.

.. py:data:: lvm_tools.fit_data.clipping.Ranges

   Type alias for ``tuple[Range, ...]`` representing multiple ranges.

.. py:data:: lvm_tools.fit_data.filtering.ExcludeStrategy

   Literal type: ``None | "pixel" | "spaxel" | "spaxel_max"``.

.. py:data:: lvm_tools.fit_data.filtering.FibreStatus

   Literal type: ``0 | 1 | 2 | 3``.

.. py:data:: lvm_tools.fit_data.normalisation.NormaliseStrategy

   Literal type: ``None | "max only" | "98 only" | "extrema" | "1σ" | "2σ" | "3σ" | "padded"``.

Validation Functions
--------------------

.. py:function:: lvm_tools.config.validation.validate_range(x_range)

   Validate a range tuple.

   :param x_range: Range to validate
   :type x_range: Range
   :raises ValueError: If range has incorrect length or max < min

.. py:function:: lvm_tools.config.validation.validate_excl_strategy(strategy)

   Validate an exclusion strategy.

   :param strategy: Strategy to validate
   :type strategy: ExcludeStrategy
   :raises ValueError: If strategy is not recognised

.. py:function:: lvm_tools.config.validation.validate_norm_strategy(strategy)

   Validate a normalisation strategy.

   :param strategy: Strategy to validate
   :type strategy: NormaliseStrategy
   :raises ValueError: If strategy is not recognised
