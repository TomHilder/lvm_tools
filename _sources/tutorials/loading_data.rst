Tutorial: Loading LVM Data
==========================

This tutorial provides a comprehensive guide to loading and inspecting LVM DRP data using **lvm_tools**.

Prerequisites
-------------

Ensure you have **lvm_tools** installed and access to LVM DRP FITS files. The examples assume familiarity with Python and basic astronomical data concepts.

LVM DRP File Structure
----------------------

LVM DRP output files are multi-extension FITS files containing:

- **HDU 0**: Primary header with observation metadata
- **HDU 1**: Flux cube (science fibre spectra)
- **HDU 2**: Inverse variance cube
- **HDU 3**: Data quality mask
- **HDU 4**: Wavelength solution
- **HDU 5**: Line spread function (FWHM)
- **HDU -1**: Slitmap (fibre positions and metadata)

Loading a Single Tile
---------------------

The ``LVMTile`` class handles all FITS parsing:

.. code-block:: python

   from pathlib import Path
   from lvm_tools import LVMTile

   # Define the path to your DRP file
   drp_path = Path("/path/to/lvm-drp-0012345-00001.fits")

   # Load the tile
   tile = LVMTile.from_file(drp_path)

   # Display tile information
   print(tile)

Output:

.. code-block:: text

   LVMTile (0x7f...):
       Filename:        lvm-drp-0012345-00001.fits
       Exposure:        1
       DRP version:     1.0.0
       Tile ID:         12345
       Data size:        156MB
       Dimensions:       tile: 1, spaxel: 1801, wavelength: 7676
       Coordinates:
           wavelength    (wavelength)                 float64  60kB   [Angstrom]
           ra            (tile, spaxel)               float64  14kB   [deg]
           dec           (tile, spaxel)               float64  14kB   [deg]
           ...
       Data:
           flux          (tile, spaxel, wavelength)   float64  105MB  DaskArray [erg / (Angstrom cm2 s)]
           i_var         (tile, spaxel, wavelength)   float64  105MB  DaskArray [1 / erg2 / (Angstrom2 cm4 s2)]
           ...

Accessing Tile Data
-------------------

The underlying data is stored in an xarray Dataset:

.. code-block:: python

   # Access the Dataset
   ds = tile.data

   # List all variables
   print("Data variables:", list(ds.data_vars))
   print("Coordinates:", list(ds.coords))

   # Access specific arrays
   flux = ds["flux"]
   wavelength = ds["wavelength"]

   print(f"Flux shape: {flux.shape}")
   print(f"Wavelength range: {float(wavelength.min()):.1f} - {float(wavelength.max()):.1f} Å")

Lazy Evaluation with Dask
-------------------------

Flux, inverse variance, mask, and LSF arrays are Dask arrays, loaded lazily:

.. code-block:: python

   # Check if data is lazily loaded
   print(f"Flux chunks: {tile.data['flux'].data.chunks}")

   # Compute a subset without loading everything
   subset = tile.data["flux"].isel(spaxel=slice(0, 10))

   # Trigger computation
   subset_values = subset.values  # or subset.compute()
   print(f"Subset shape: {subset_values.shape}")

Coordinates are eagerly loaded as they are typically small.

Accessing Metadata
------------------

Tile metadata is stored in the ``meta`` attribute:

.. code-block:: python

   # Access metadata
   print(f"Filename: {tile.meta.filename}")
   print(f"Tile ID: {tile.meta.tile_id}")
   print(f"Exposure number: {tile.meta.exp_num}")
   print(f"DRP version: {tile.meta.drp_ver}")

Loading Multiple Tiles
----------------------

For multi-exposure analyses, combine tiles into a collection:

.. code-block:: python

   from lvm_tools import LVMTile, LVMTileCollection

   # Load multiple tiles
   paths = [
       Path("/path/to/exposure_1.fits"),
       Path("/path/to/exposure_2.fits"),
       Path("/path/to/exposure_3.fits"),
   ]
   tiles = [LVMTile.from_file(p) for p in paths]

   # Create a collection
   collection = LVMTileCollection.from_tiles(tiles)

   # Inspect the collection
   print(collection)

The collection concatenates data along the ``tile`` dimension:

.. code-block:: python

   # Access combined data
   print(f"Number of tiles: {len(collection.meta)}")
   print(f"Combined flux shape: {collection.data['flux'].shape}")

   # Metadata is accessible by exposure number
   for exp_num, meta in collection.meta.items():
       print(f"Exposure {exp_num}: {meta.filename}")

Working with Coordinates
------------------------

Spatial coordinates (RA, Dec) are stored per spaxel:

.. code-block:: python

   import numpy as np

   # Access coordinates
   ra = tile.data["ra"].values.flatten()
   dec = tile.data["dec"].values.flatten()

   # Compute field centre
   ra_centre = np.mean(ra)
   dec_centre = np.mean(dec)
   print(f"Field centre: ({ra_centre:.4f}, {dec_centre:.4f}) deg")

   # Compute field extent
   print(f"RA range: {ra.min():.4f} - {ra.max():.4f} deg")
   print(f"Dec range: {dec.min():.4f} - {dec.max():.4f} deg")

Fibre Information
-----------------

Fibre-level metadata is available for quality filtering:

.. code-block:: python

   # Fibre status flags
   fibre_status = tile.data["fibre_status"].values.flatten()
   unique_status = np.unique(fibre_status)
   print(f"Fibre status values: {unique_status}")

   # Count by status
   for status in unique_status:
       count = np.sum(fibre_status == status)
       print(f"  Status {status}: {count} fibres")

   # IFU labels
   ifu_labels = tile.data["ifu_label"].values.flatten()
   unique_ifu = np.unique(ifu_labels)
   print(f"IFU bundles: {unique_ifu}")

Wavelength Grid
---------------

The wavelength solution is common to all spaxels:

.. code-block:: python

   # Access wavelength grid
   wavelength = tile.data["wavelength"].values
   print(f"Wavelength points: {len(wavelength)}")
   print(f"Range: {wavelength.min():.1f} - {wavelength.max():.1f} Å")

   # Compute spectral resolution
   delta_lambda = np.diff(wavelength)
   print(f"Pixel size: {delta_lambda.mean():.3f} Å (mean)")

Line Spread Function
--------------------

The LSF is provided as sigma (converted from FWHM in the DRP):

.. code-block:: python

   # LSF sigma array
   lsf_sigma = tile.data["lsf_sigma"]
   print(f"LSF shape: {lsf_sigma.shape}")

   # Median LSF at a given wavelength
   median_lsf = float(lsf_sigma.isel(wavelength=3000).median())
   print(f"Median LSF sigma at λ=3000: {median_lsf:.2f} Å")

Observation Time
----------------

The Modified Julian Date (MJD) is computed from the observation midpoint:

.. code-block:: python

   # Access MJD
   mjd = tile.data["mjd"].values
   print(f"Observation MJD: {mjd[0]:.6f}")

Error Handling
--------------

Handle missing or corrupted files gracefully:

.. code-block:: python

   from pathlib import Path
   from lvm_tools import LVMTile

   path = Path("/path/to/nonexistent.fits")

   try:
       tile = LVMTile.from_file(path)
   except FileNotFoundError:
       print(f"File not found: {path}")

Next Steps
----------

- :doc:`data_configuration`: Learn to configure data processing
- :doc:`fitting_workflow`: Complete model fitting workflow
