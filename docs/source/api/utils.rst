Utilities API
=============

This module provides utility functions for physical calculations and spatial operations.

Barycentric Corrections
-----------------------

.. py:function:: lvm_tools.physical_properties.barycentric_corr.get_v_barycentric(mjd, α, δ, unit="km/s")

   Compute barycentric velocity correction.

   Uses the Las Campanas Observatory location and the astropy
   radial velocity correction framework.

   :param mjd: Modified Julian Date(s)
   :type mjd: ArrayLike
   :param α: Right ascension in degrees
   :type α: ArrayLike
   :param δ: Declination in degrees
   :type δ: ArrayLike
   :param unit: Output unit (default "km/s")
   :type unit: str
   :returns: Barycentric velocity correction
   :rtype: numpy.ndarray

   **Example**:

   .. code-block:: python

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

Spatial Masking
---------------

.. py:function:: lvm_tools.utils.mask.mask_near_points(xgrid, ygrid, xpoints, ypoints, threshold=None)

   Generate a boolean mask for grid cells near data points.

   Uses a KD-tree for efficient nearest-neighbour lookup.

   :param xgrid: Grid coordinates along x-axis (monotonically increasing)
   :type xgrid: numpy.ndarray
   :param ygrid: Grid coordinates along y-axis (monotonically increasing)
   :type ygrid: numpy.ndarray
   :param xpoints: X-coordinates of data points
   :type xpoints: numpy.ndarray
   :param ypoints: Y-coordinates of data points
   :type ypoints: numpy.ndarray
   :param threshold: Maximum distance to be considered "near". If None, uses 1.5 times the maximum of mean grid spacings
   :type threshold: float | None
   :returns: Boolean mask, shape ``(len(ygrid), len(xgrid))``, True means "keep"
   :rtype: numpy.ndarray

   **Example**:

   .. code-block:: python

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

   **Use Case**:

   This function is useful for creating masks that restrict model evaluation
   to regions with data coverage, avoiding extrapolation in sparse regions.

Helper Functions
----------------

Array Conversion
~~~~~~~~~~~~~~~~

.. py:function:: lvm_tools.data.helper.daskify_native(array, chunks)
   :no-index:

   Convert an array to a Dask array with native byte order.

   Handles FITS files which may store data in non-native byte order.

   :param array: Input array
   :type array: ArrayLike
   :param chunks: Dask chunk specification
   :type chunks: str | int | tuple
   :returns: Dask array
   :rtype: dask.array.Array

.. py:function:: lvm_tools.data.helper.numpyfy_native(array)
   :no-index:

   Convert an array to NumPy with native byte order.

   :param array: Input array
   :type array: ArrayLike
   :returns: NumPy array
   :rtype: numpy.ndarray

Dataset Utilities
~~~~~~~~~~~~~~~~~

.. py:function:: lvm_tools.data.helper.summarize_with_units(ds)

   Generate a formatted string summary of an xarray Dataset.

   Includes dimensions, coordinates with units, and data variables
   with chunk information for Dask arrays.

   :param ds: Dataset to summarise
   :type ds: xarray.Dataset
   :returns: Formatted summary string
   :rtype: str

.. py:function:: lvm_tools.data.helper.convert_sci_to_int(arr)

   Convert IFU label strings to integer indices.

   :param arr: Array of IFU labels ("Sci1", "Sci2", "Sci3")
   :type arr: ArrayLike
   :returns: Integer indices (0, 1, 2)
   :rtype: numpy.ndarray

Coordinate Utilities
~~~~~~~~~~~~~~~~~~~~

.. py:function:: lvm_tools.data.coordinates.get_mjd(header)
   :no-index:

   Extract observation midpoint MJD from FITS header.

   Uses INTSTART and INTEND keywords to compute the midpoint.

   :param header: FITS primary header
   :type header: astropy.io.fits.Header
   :returns: Modified Julian Date
   :rtype: float
   :raises ValueError: If time keywords are missing

.. py:function:: lvm_tools.data.coordinates.get_observatory_code(header)

   Extract observatory code from FITS header.

   :param header: FITS primary header
   :type header: astropy.io.fits.Header
   :returns: Observatory code (e.g., "LCO")
   :rtype: str
   :raises ValueError: If observatory is unknown

.. py:function:: lvm_tools.data.coordinates.get_observatory_location(observatory)
   :no-index:

   Get Earth location for a named observatory.

   :param observatory: Observatory code
   :type observatory: str
   :returns: Observatory location
   :rtype: astropy.coordinates.EarthLocation
   :raises ValueError: If observatory is unknown

   Currently supported: "LCO" (Las Campanas Observatory)
