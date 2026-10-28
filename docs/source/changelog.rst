Changelog
=========

All notable changes to **lvm_tools** are documented in this file.

The format follows `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

Unreleased
----------

Changed
~~~~~~~

- Invalid normalisation offsets or scales (NaN, infinite, or a non-positive
  scale) raise ``ValueError`` instead of a bare ``Exception``. Code that caught
  ``Exception`` still catches them.
- Development: ruff is in the dev dependency group, and CI runs
  ``ruff check`` and ``ruff format --check``.
- Documentation moved from ReadTheDocs to GitHub Pages
  (https://tomhilder.github.io/lvm_tools/), deployed with each release.

[0.1.2] - 2026-10-27
--------------------

Fixed
~~~~~

- ``LVMTile.from_file`` no longer reads whole cubes. Each cube is now chunked
  along wavelength and only the chunks in the selected wavelength range are
  read and byte-swapped. For an Hα window over 49 Rosette frames, loading went
  from 7.9 s and 16 GB to 0.5 s and 3.7 GB, with bit-identical output.

[0.1.1] - 2026-09-03
--------------------

Changed
~~~~~~~

- ``NORM_PADDING`` increased from 0.01 to 0.15. The ``"padded"``
  normalisation maps the data onto the periodic domain of the Fourier-basis
  GP, and 1% padding let the field wrap onto itself across the domain edge.
  This changes the output of every ``"padded"`` normalisation, so fitted
  coordinates and ``arcsec_to_model_domain`` results shift.

[0.1.0] - 2026-03-11
--------------------

Initial release.

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
- ``FitData.arcsec_to_model_domain`` to convert a length in arcseconds to the
  normalised model domain
- Documentation with tutorials and an API reference, built with Sphinx on
  ReadTheDocs

Changed
~~~~~~~

- LSF values are converted from FWHM to sigma on load

Fixed
~~~~~

- Native byte order handling for FITS arrays
