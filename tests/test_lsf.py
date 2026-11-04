"""test_lsf.py - the per-fibre survey LSF and its Gaussian-mixture approximation."""

import csv
import hashlib
import json

import numpy as np
import pytest
import xarray as xr

from lvm_tools.fit_data.fit_data import FitData
from lvm_tools.lsf import (
    CHANNELS,
    DATA_FILES,
    N_OFFSET,
    LSFMixture,
    SurveyLSF,
    _mspline_basis,
    fetch_survey_lsf,
)

KNOTS = {"B": (3600.0, 5786.5), "R": (5787.0, 7453.5), "Z": (7454.0, 9799.999)}
D = np.linspace(-2.5, 2.5, 2001)
DD = D[1] - D[0]


def gaussian_weights(sigma: float) -> np.ndarray:
    """M-spline weights of a centred profile close to a Gaussian of width sigma."""
    centres = -2.5 + (np.arange(N_OFFSET) + 2) * 5 / 14
    w = np.exp(-0.5 * (centres / sigma) ** 2)
    return w / w.sum()


FLAT = np.array([0, 0, 0.02, 0.12, 0.2, 0.32, 0.2, 0.12, 0.02, 0, 0], float)
FLAT /= FLAT.sum()


def write_tables(directory, profiles, unmeasured=()):
    """Survey LSF tables in lvmsky's layout. `profiles[fibre_id]` gives the M-spline weights,
    the same at every wavelength; fibres in `unmeasured` are flagged as not measured."""
    with open(directory / "wavelength_knots.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "knot_index", "wavelength_angstrom"])
        for c, (lo, hi) in KNOTS.items():
            for i, k in enumerate([lo] * 4 + [hi] * 4):
                w.writerow([c, i, k])
    names = [f"coef_{k:02d}_{j:02d}" for k in range(N_OFFSET) for j in range(4)]
    for c in CHANNELS:
        with open(directory / f"survey_lsf_{c}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fiberid", "channel", "quality", *names])
            for fid, a in profiles.items():
                quality = "elevated_chi2" if fid in unmeasured else "measured"
                w.writerow([fid, c, quality, *np.repeat(a, 4)])
    (directory / "coverage.csv").write_text("fiberid,channel\n")
    sums = {n: hashlib.sha256((directory / n).read_bytes()).hexdigest() for n in DATA_FILES}
    (directory / "manifest.json").write_text(json.dumps({"files_sha256": sums}))
    return directory


@pytest.fixture
def tables(tmp_path):
    return write_tables(tmp_path, {3: gaussian_weights(0.6), 7: FLAT, 9: FLAT}, unmeasured=(9,))


@pytest.fixture
def lsf(tables):
    return SurveyLSF.from_directory(tables)


class TestBasis:
    def test_msplines_have_unit_area(self):
        assert np.allclose(_mspline_basis(D).sum(0) * DD, 1.0, atol=1e-6)


class TestSurveyLSF:
    @pytest.mark.parametrize("wavelength", [4000.0, 6562.8, 8500.0])
    def test_density_integrates_to_one(self, lsf, wavelength):
        p = lsf.density([3, 7], wavelength, D)
        assert p.shape == (2, D.size)
        assert np.allclose(p.sum(1) * DD, 1.0, atol=1e-6)

    def test_moments_match_numerical_integration(self, lsf):
        p = lsf.density([3, 7], 6562.8, D)
        mean, sigma = lsf.moments([3, 7], 6562.8)
        m_num = (p * D).sum(1) * DD
        s_num = np.sqrt((p * (D - m_num[:, None]) ** 2).sum(1) * DD)
        assert np.allclose(mean, m_num, atol=1e-6)
        assert np.allclose(sigma, s_num, rtol=1e-5)

    def test_unmeasured_and_unknown_fibres_are_nan(self, lsf):
        assert lsf.is_measured([3, 9, 1000], 6562.8).tolist() == [True, False, False]
        assert np.isnan(lsf.density([9, 1000], 6562.8, D)).all()

    def test_channel(self):
        assert [SurveyLSF.channel(w) for w in (4861.3, 6562.8, 8500.0)] == ["B", "R", "Z"]


class TestMixture:
    def test_reproduces_the_profile(self, lsf):
        mean, _ = lsf.moments([3, 7], 6562.8)
        mix = lsf.mixture([3, 7], 6562.8, n_components=3)
        p = lsf.density([3, 7], 6562.8, D)
        for i in range(2):
            approx = mix.density(D - mean[i])[i]
            assert np.abs(approx - p[i]).max() < 0.02 * p[i].max()

    def test_flat_top_needs_offset_components(self, lsf):
        mix = lsf.mixture([7], 6562.8, n_components=2)
        assert np.ptp(mix.offsets) > 0.3 * mix.sigma[0]

    def test_stretched_to_the_requested_width(self, lsf):
        mix = lsf.mixture([3, 7], 6562.8, n_components=3, sigma=[0.5, 0.7])
        assert mix.sigma == pytest.approx([0.5, 0.7], rel=0.03)
        assert np.abs(mix.mean).max() < 0.02
        assert np.allclose(mix.weights.sum(-1), 1.0)

    def test_unmeasured_fibre_gets_a_gaussian(self, lsf):
        mix = lsf.mixture([9], 6562.8, n_components=3, sigma=0.6)
        assert not mix.measured[0]
        assert mix.weights[0].tolist() == [1.0, 0.0, 0.0]
        assert mix.sigma[0] == pytest.approx(0.6)

    def test_density_shape(self):
        mix = LSFMixture(
            weights=np.full((4, 2), 0.5),
            offsets=np.tile([-0.3, 0.3], (4, 1)),
            widths=np.full((4, 2), 0.5),
            measured=np.ones(4, bool),
        )
        assert mix.density(np.zeros((5, 3))).shape == (4, 5, 3)
        assert np.allclose(mix.density(D).sum(1) * DD, 1.0, atol=1e-6)


class TestFetch:
    def test_downloads_and_checks(self, tables, tmp_path):
        cache = fetch_survey_lsf(tmp_path / "cache", base_url=tables.as_uri())
        assert SurveyLSF.load(cache).is_measured([3], 6562.8)[0]

    def test_rejects_a_corrupted_file(self, tables, tmp_path):
        cache = fetch_survey_lsf(tmp_path / "cache", base_url=tables.as_uri())
        (cache / "survey_lsf_R.csv").write_text("corrupted")
        with pytest.raises(ValueError, match="SHA-256"):
            fetch_survey_lsf(cache, base_url=tables.as_uri())
        assert not (cache / "survey_lsf_R.csv").exists()


class TestFitDataMixture:
    def fit_data(self, sigma_drp):
        wave = np.arange(6540.0, 6600.0, 0.5)
        fibres = np.array([3, 7, 9])
        lsf_sigma = np.tile(sigma_drp, (wave.size, 1))
        lsf_sigma[:4, 0] = np.nan  # pixels excluded by clipping
        ds = xr.Dataset(
            {"lsf_sigma": (("wavelength", "flat_spaxel"), lsf_sigma)},
            coords={
                "wavelength": wave,
                "fibre_id": ("flat_spaxel", fibres),
            },
        )
        identity = lambda x: x
        return FitData(ds, *([identity] * 8))

    def test_shapes_and_widths(self, lsf):
        fd = self.fit_data(np.array([0.55, 0.6, 0.65]))
        mix = fd.lsf_mixture([6562.8, 6583.4], n_components=3, lsf=lsf)
        assert mix.weights.shape == (3, 2, 3)
        assert mix.measured.tolist() == [[True, True], [True, True], [False, False]]
        assert mix.sigma == pytest.approx(np.array([[0.55] * 2, [0.6] * 2, [0.65] * 2]), rel=0.03)
