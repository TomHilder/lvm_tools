# Line Spread Function API

The DRP's `LSF` extension gives a Gaussian width per fibre and pixel (`FitData.lsf_σ`). The
real LSF is flat-topped, by an amount that depends on the fibre and the wavelength: at Hα,
fitting a Gaussian to it leaves residuals of about 5% of the peak. `lvm_tools.lsf` reads the
per-fibre survey LSF measured by the LVM sky-subtraction code,
[lvmsky](https://github.com/sdss/lvmsky), and approximates it by a few Gaussians, so that a
model can convolve it with a Gaussian line and integrate over pixels analytically.

```python
import numpy as np
from lvm_tools.lsf import SurveyLSF

lsf = SurveyLSF.load()  # downloads the tables (12 MB) on first use, then reads the cache

d = np.linspace(-2.5, 2.5, 201)  # offset from the line centre, Angstrom
profile = lsf.density(fibre_id=[100, 101], wavelength=6562.8, offset=d)  # (2, 201), 1/Å
mean, sigma = lsf.moments([100, 101], 6562.8)

# For a model: every spaxel's LSF at each line, as 3 Gaussians, with the DRP's widths
mix = fit_data.lsf_mixture([6548.05, 6562.80, 6583.45], lsf=lsf)
mix.weights, mix.offsets, mix.widths  # each (n_spaxels, n_lines, 3), Angstrom
```

A line of intrinsic width σ seen through the mixture is a sum of Gaussians with weights
`mix.weights`, centres shifted by `mix.offsets` and widths √(`mix.widths`² + σ²).

## The survey LSF

For each physical fibre and spectrograph channel (B, R, Z), lvmsky gives a continuous density

L(d, λ) = Σ<sub>kj</sub> c<sub>kj</sub> M<sub>k</sub>(d) B<sub>j</sub>(λ),

where d is the offset from the line centre, M<sub>k</sub> are 11 cubic M-splines (B-splines
of unit area) on uniform knots over [-2.5, 2.5] Å, and B<sub>j</sub> are 4 cubic B-splines in
wavelength per channel. The coefficients are non-negative and sum to one over k, so L
integrates to one at every wavelength. It is the profile before integration over detector
pixels.

It was measured on 555 bright-Moon exposures of DRP 1.3.2: R and Z on sky emission lines, B on
solar absorption lines in moonlight. Each fibre is measured separately and averaged over
observing nights, so the profile is fixed per fibre. Its width changes by a few percent between
calibration epochs, which is why `FitData.lsf_mixture` keeps the shape but takes the width from
the DRP's `LSF` extension of each exposure.

Fibres without a measurement (the standard-star fibres, and two Z fibres flagged by lvmsky) get
a Gaussian, and `measured` is `False` for them.

## Accuracy of the mixtures

Over all fibres, the largest difference between a 3-Gaussian mixture and the survey LSF is
about 1% of the peak (median over fibres; under 2% for every fibre) at Hβ, [O III], Hα,
[S II] and 8500 Å; with 2 Gaussians it is 1–1.7%. The mixture's standard deviation matches the
LSF's to about 1%: the LSF has a faint outer wing that the Gaussians don't follow. Fitting a
mixture takes a few milliseconds per fibre and wavelength; identical profiles are fitted once.

## Data

`SurveyLSF.load()` downloads the tables from the lvmsky commit `LVMSKY_COMMIT` with
`fetch_survey_lsf`, checks them against the SHA-256 sums in lvmsky's manifest, and caches them
in `~/.cache/lvm_tools/survey_lsf/` (or under `$LVM_TOOLS_CACHE`). To use a local copy instead,
pass its directory: `SurveyLSF.load("path/to/lsf")`. The tables are from lvmsky, BSD 3-Clause
License, Copyright (c) 2022, SDSS.

## SurveyLSF

::: lvm_tools.lsf.SurveyLSF

## LSFMixture

::: lvm_tools.lsf.LSFMixture

## FitData.lsf_mixture

::: lvm_tools.fit_data.fit_data.FitData.lsf_mixture
    options:
      show_root_full_path: false

## Downloading

::: lvm_tools.lsf.fetch_survey_lsf

::: lvm_tools.lsf.default_cache_dir

::: lvm_tools.lsf.LVMSKY_COMMIT
