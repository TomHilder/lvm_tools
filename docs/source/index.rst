lvm_tools Documentation
========================

**lvm_tools** is a lightweight, modular wrapper for the `Sloan Digital Sky Survey (SDSS) <https://sdss.org/>`_ Local Volume Mapper (LVM) Data Reduction Pipeline (DRP) products. The package provides efficient, lazy data loading via `Dask <https://www.dask.org>`_, making it suitable for large-scale spectroscopic analysis and model fitting.

The package is designed for use with spectrospatial models via `spectracles <https://github.com/TomHilder/spectracles>`_, though its modular architecture supports a variety of downstream applications.

Key Features
------------

- **Lazy data loading**: Read LVM DRP FITS files using Dask for memory-efficient processing of large datasets
- **Modular data configuration**: Flexible configuration system for wavelength clipping, spatial filtering, and data normalisation
- **Reproducible pipelines**: Hash-based tracking of data processing configurations for reproducibility
- **JAX integration**: Seamless conversion to JAX arrays for GPU-accelerated model fitting
- **xarray-based storage**: Data stored in xarray Datasets for intuitive, labelled array manipulation

Installation
------------

The recommended installation method is via PyPI:

.. code-block:: bash

   pip install lvm-tools

Alternatively, using `uv <https://docs.astral.sh/uv/>`_ (recommended):

.. code-block:: bash

   uv add lvm-tools

For development, clone and install from source:

.. code-block:: bash

   git clone git@github.com:TomHilder/lvm_tools.git
   cd lvm_tools
   pip install -e .

Quick Start
-----------

.. code-block:: python

   from pathlib import Path
   from lvm_tools import LVMTile, DataConfig, FitDataBuilder

   # Load a single tile from a DRP file
   tile = LVMTile.from_file(Path("path/to/lvm-drp-file.fits"))

   # Create a configuration for data processing
   config = DataConfig.from_tiles(
       tile,
       λ_range=(6500.0, 6800.0),  # Wavelength range in Angstroms
   )

   # Build fit-ready data
   fit_data = FitDataBuilder(tile, config).build()

   # Access normalised arrays for model fitting
   flux = fit_data.flux        # Normalised flux (JAX array)
   wavelength = fit_data.λ     # Wavelength grid (JAX array)

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user_guide/installation
   user_guide/quickstart
   user_guide/concepts

.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/loading_data
   tutorials/exploring_data
   tutorials/data_configuration
   tutorials/fitting_workflow

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/overview

.. toctree::
   :maxdepth: 1
   :caption: Development

   changelog
   contributing

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Citation
--------

If you use **lvm_tools** in your research, please cite:

.. code-block:: bibtex

   @software{lvm_tools,
     author = {Hilder, Tom},
     title = {lvm\_tools: A modular wrapper for LVM DRP data},
     url = {https://github.com/TomHilder/lvm_tools},
     year = {2025}
   }

License
-------

This project is licensed under the MIT License. See the `LICENSE <https://github.com/TomHilder/lvm_tools/blob/main/LICENSE>`_ file for details.

Contact
-------

For questions or issues, please `open an issue <https://github.com/TomHilder/lvm_tools/issues>`_ on GitHub or contact the author directly at Thomas.Hilder@monash.edu.
