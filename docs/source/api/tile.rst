Data Loading API
================

This module provides classes for loading and managing LVM DRP data.

LVMTile
-------

.. py:class:: lvm_tools.LVMTile

   Container for a single LVM DRP observation.

   .. py:attribute:: data
      :type: xarray.Dataset

      The observation data stored as an xarray Dataset with the following structure:

      **Data Variables** (Dask arrays):

      - ``flux``: Flux cube, shape ``(1, n_spaxels, n_wavelengths)``, units erg cm⁻² s⁻¹ Å⁻¹
      - ``i_var``: Inverse variance, same shape as flux
      - ``lsf_sigma``: Line spread function (sigma), same shape as flux, units Å
      - ``mask``: DRP quality mask, same shape as flux

      **Coordinates**:

      - ``tile``: Tile index (length 1 for single tile)
      - ``spaxel``: Spaxel index
      - ``wavelength``: Wavelength grid in Angstroms
      - ``mjd``: Modified Julian Date of observation midpoint
      - ``ra``: Right ascension of each spaxel, shape ``(1, n_spaxels)``, units degrees
      - ``dec``: Declination of each spaxel, shape ``(1, n_spaxels)``, units degrees
      - ``fibre_id``: Fibre identifier
      - ``ifu_label``: IFU bundle label ("Sci1", "Sci2", "Sci3")
      - ``fibre_status``: Fibre quality flag (0=good)

   .. py:attribute:: meta
      :type: LVMTileMeta

      Metadata for the observation.

   .. py:classmethod:: from_file(drp_file)

      Load a tile from a DRP FITS file.

      :param drp_file: Path to the FITS file
      :type drp_file: Path | str
      :returns: Loaded tile
      :rtype: LVMTile
      :raises FileNotFoundError: If the file does not exist

      **Example**:

      .. code-block:: python

         from pathlib import Path
         from lvm_tools import LVMTile

         tile = LVMTile.from_file(Path("/data/lvm-drp-12345-00001.fits"))

LVMTileMeta
-----------

.. py:class:: lvm_tools.data.tile.LVMTileMeta

   Metadata container for an LVM observation.

   .. py:attribute:: filename
      :type: str

      Name of the source FITS file.

   .. py:attribute:: tile_id
      :type: int

      LVM tile identifier.

   .. py:attribute:: exp_num
      :type: int

      Exposure number within the tile.

   .. py:attribute:: drp_ver
      :type: str

      Data Reduction Pipeline version string.

LVMTileCollection
-----------------

.. py:class:: lvm_tools.LVMTileCollection

   Container for multiple LVM observations combined along the tile dimension.

   .. py:attribute:: data
      :type: xarray.Dataset

      Combined observation data. Same structure as ``LVMTile.data`` but with
      ``tile`` dimension length equal to the number of tiles.

   .. py:attribute:: meta
      :type: Mapping[int, LVMTileMeta]

      Dictionary mapping exposure number to metadata for each tile.

   .. py:classmethod:: from_tiles(tiles)

      Combine multiple tiles into a collection.

      :param tiles: List of tiles to combine
      :type tiles: list[LVMTile]
      :returns: Combined collection
      :rtype: LVMTileCollection

      **Example**:

      .. code-block:: python

         from lvm_tools import LVMTile, LVMTileCollection

         tiles = [LVMTile.from_file(p) for p in paths]
         collection = LVMTileCollection.from_tiles(tiles)

         print(f"Combined shape: {collection.data['flux'].shape}")

Type Aliases
------------

.. py:data:: lvm_tools.data.tile.LVMTileLike

   Union type for ``LVMTile | LVMTileCollection``.

   Functions accepting ``LVMTileLike`` can operate on either single tiles
   or collections.

Physical Constants
------------------

The following constants are defined in ``lvm_tools.data.tile``:

.. py:data:: SIGMA_TO_FWHM
   :type: float
   :value: 2.355

   Conversion factor from Gaussian sigma to FWHM: ``2 * sqrt(2 * ln(2))``.

.. py:data:: FWHM_TO_SIGMA
   :type: float
   :value: 0.425

   Conversion factor from FWHM to Gaussian sigma.

.. py:data:: FLUX_UNIT
   :type: astropy.units.Unit

   Physical unit for flux: erg cm⁻² s⁻¹ Å⁻¹.

.. py:data:: SPECTRAL_UNIT
   :type: astropy.units.Unit

   Physical unit for wavelength: Angstrom.

.. py:data:: SPATIAL_UNIT
   :type: astropy.units.Unit

   Physical unit for coordinates: degree.

Helper Functions
----------------

.. py:function:: lvm_tools.data.helper.daskify_native(array, chunks)

   Convert an array to a Dask array with native byte order.

   :param array: Input array
   :type array: ArrayLike
   :param chunks: Chunk specification for Dask
   :type chunks: str | int | tuple
   :returns: Dask array with native byte order
   :rtype: dask.array.Array

.. py:function:: lvm_tools.data.helper.numpyfy_native(array)

   Convert an array to NumPy with native byte order.

   :param array: Input array
   :type array: ArrayLike
   :returns: NumPy array with native byte order
   :rtype: numpy.ndarray

Coordinate Utilities
--------------------

.. py:function:: lvm_tools.data.coordinates.get_mjd(header)

   Extract the observation midpoint MJD from a FITS header.

   :param header: FITS header containing INTSTART and INTEND keywords
   :type header: astropy.io.fits.Header
   :returns: Modified Julian Date of observation midpoint
   :rtype: float
   :raises ValueError: If time information is not found

.. py:function:: lvm_tools.data.coordinates.get_observatory_location(observatory)

   Get the Earth location for an observatory.

   :param observatory: Observatory code (e.g., "LCO")
   :type observatory: str
   :returns: Observatory location
   :rtype: astropy.coordinates.EarthLocation
   :raises ValueError: If observatory is unknown
