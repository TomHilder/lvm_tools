Changelog
=========

All notable changes to **lvm_tools** are documented in this file.

The format follows `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

Added
~~~~~

- Comprehensive documentation with tutorials and API reference
- GitHub Actions workflow for ReadTheDocs integration
- Working code examples throughout documentation

Changed
~~~~~~~

- LSF values are now converted from FWHM to sigma on load

Fixed
~~~~~

- Native byte order handling for FITS arrays

[0.1.0] - Initial Release
-------------------------

Added
~~~~~

- ``LVMTile`` class for loading single LVM DRP observations
- ``LVMTileCollection`` for combining multiple tiles
- ``DataConfig`` for specifying data processing parameters
- ``FitDataBuilder`` for constructing fit-ready data with reproducibility tracking
- ``FitData`` class with normalised JAX arrays
- Lazy data loading via Dask
- Support for multiple wavelength ranges
- Configurable filtering strategies (NaN, bad flux, fibre status, mask)
- Multiple normalisation strategies
- Barycentric velocity correction
- Integration with spectracles (optional dependency)
