Installation
============

This section describes the various methods for installing **lvm_tools** and its dependencies.

Requirements
------------

**lvm_tools** requires Python 3.10 or later. The package has the following core dependencies:

- `astropy <https://www.astropy.org>`_ (>=7.1.0): Astronomical data handling and FITS I/O
- `dask <https://www.dask.org>`_ (>=2025.7.0): Lazy array operations
- `jax <https://jax.readthedocs.io>`_ (>=0.5.3): Accelerated numerical computing
- `numpy <https://numpy.org>`_ (>=2.3.2): Numerical array operations
- `scipy <https://scipy.org>`_ (>=1.16.1): Scientific computing utilities
- `xarray <https://xarray.pydata.org>`_ (>=2025.7.1): Labelled multi-dimensional arrays

Installation from PyPI
----------------------

The simplest installation method is via PyPI using pip:

.. code-block:: bash

   pip install lvm-tools

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

For faster, more reliable dependency resolution, we recommend using `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: bash

   uv add lvm-tools

Installation from Source
------------------------

For development or to access the latest features, install from source:

.. code-block:: bash

   git clone git@github.com:TomHilder/lvm_tools.git
   cd lvm_tools
   pip install -e .

This creates an editable installation, allowing changes to the source code to take effect immediately.

Optional Dependencies
---------------------

spectracles Integration
~~~~~~~~~~~~~~~~~~~~~~~

For use with the `spectracles <https://github.com/TomHilder/spectracles>`_ spectrospatial modelling package:

.. code-block:: bash

   pip install lvm-tools[spectracles]

This installs the ``spectracles`` package and enables the ``αδ_data`` property on ``FitData`` objects.

Verifying Installation
----------------------

To verify the installation, open a Python interpreter and import the package:

.. code-block:: python

   import lvm_tools
   print(lvm_tools.__all__)

This should output:

.. code-block:: python

   ['LVMTile', 'LVMTileCollection', 'DataConfig', 'FitDataBuilder']

Troubleshooting
---------------

JAX Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter issues with JAX installation, particularly on systems with NVIDIA GPUs, consult the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_ for platform-specific instructions.

FITS File Access
~~~~~~~~~~~~~~~~

**lvm_tools** requires access to LVM DRP output files. These files are typically available through the SDSS Science Archive Server (SAS). Ensure you have appropriate access credentials and network connectivity.
