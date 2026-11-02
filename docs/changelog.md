# Changelog

All notable changes to **lvm_tools** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Sky positions in arcseconds, for spectracles models with physical length scales: `FitData.sky_x` and `FitData.sky_y` are each spaxel's offset east and north of a centre, on the tangent plane (astropy's `SkyOffsetFrame`, so the cos δ factor is exact across the field), and `FitData.sky_data` is a `SpatialDataLVM` with them.
- `FitData.sky_centre`: the (RA, Dec) the offsets are measured from, by default the centre of the spaxels' bounding box (correct across RA = 0). `FitData.with_sky_origin(ra, dec)` sets it, e.g. to put several datasets on one frame.
- `FitData.sky_to_radec(x, y)`: offsets back to RA and Dec, e.g. for maps on a grid.

- Tests (`tests/`), run in CI on Python 3.10 to 3.14.

### Deprecated

- The coordinates scaled into [-π, π]: `FitData.α`, `FitData.δ`, `FitData.αδ_data`, `FitData.predict_α`, `FitData.predict_δ` and `FitData.arcsec_to_model_domain` warn (`DeprecationWarning`) and will be removed in a future release. Use `sky_x`, `sky_y`, `sky_data` and `sky_to_radec`, with a spectracles `Domain` for physical length scales. They still return the same values.

`DataConfig` and `FitDataBuilder.hash()` are unchanged, so existing hashes are too. The sky origin (`with_sky_origin`) is a choice of frame, not part of the data, and isn't included in the hash.

### Changed

- The documentation is now built with MkDocs Material (as spectracles' is), from Markdown in `docs/`. The content is unchanged.

## [0.1.3] - 2026-10-28

### Fixed

- `FitDataBuilder.hash()` crashed when given a single `LVMTile` (it only worked for collections). Hashes for collections are unchanged.
- Clipping or filtering away all the data now raises a `ValueError` that names the ranges, instead of an obscure `TypeError` from the normalisation.
- Documentation: code examples that failed on real data, stale descriptions of the `"padded"` and `"Nσ"` normalisations, and dependency minimums that didn't match the package.

### Changed

- Invalid normalisation offsets or scales (NaN, infinite, or a non-positive scale) raise `ValueError` instead of a bare `Exception`. Code that caught `Exception` still catches them.
- The `spectracles` extra now installs spectracles on Python 3.11 and later (matching spectracles' own requirement), instead of only on 3.13 and later.
- Development: ruff is in the dev dependency group, and CI runs `ruff check` and `ruff format --check`.
- Documentation moved from ReadTheDocs to GitHub Pages (<https://tomhilder.github.io/lvm_tools/>), deployed with each release.

## [0.1.2] - 2026-10-27

### Fixed

- `LVMTile.from_file` no longer reads whole cubes. Each cube is now chunked along wavelength and only the chunks in the selected wavelength range are read and byte-swapped. For an Hα window over 49 Rosette frames, loading went from 7.9 s and 16 GB to 0.5 s and 3.7 GB, with bit-identical output.

## [0.1.1] - 2026-09-03

### Changed

- `NORM_PADDING` increased from 0.01 to 0.15. The `"padded"` normalisation maps the data onto the periodic domain of the Fourier-basis GP, and 1% padding let the field wrap onto itself across the domain edge. This changes the output of every `"padded"` normalisation, so fitted coordinates and `arcsec_to_model_domain` results shift.

## [0.1.0] - 2026-03-11

Initial release.

### Added

- `LVMTile` class for loading single LVM DRP observations
- `LVMTileCollection` for combining multiple tiles
- `DataConfig` for specifying data processing parameters
- `FitDataBuilder` for constructing fit-ready data with reproducibility tracking
- `FitData` class with normalised JAX arrays
- Lazy data loading via Dask
- Support for multiple wavelength ranges
- Configurable filtering strategies (NaN, bad flux, fibre status, mask)
- Multiple normalisation strategies
- Barycentric velocity correction
- Integration with spectracles (optional dependency)
- `FitData.arcsec_to_model_domain` to convert a length in arcseconds to the normalised model domain
- Documentation with tutorials and an API reference, built with Sphinx on ReadTheDocs

### Changed

- LSF values are converted from FWHM to sigma on load

### Fixed

- Native byte order handling for FITS arrays
