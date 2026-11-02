"""test_sky.py - FitData's sky offsets in arcsec, and the deprecated [-π, π] coordinates."""

import warnings

import numpy as np
import pytest
import xarray as xr

from lvm_tools.fit_data.fit_data import FitData

RA_C, DEC_C = 270.44, -23.09  # roughly the Trifid


def fit_data(ra, dec, dtype=float) -> FitData:
    n = len(ra)
    ds = xr.Dataset(
        {
            "ra": ("flat_spaxel", np.asarray(ra, dtype=dtype)),
            "dec": ("flat_spaxel", np.asarray(dec, dtype=dtype)),
            "tile": ("flat_spaxel", np.zeros(n, dtype=int)),
            "ifu_label": ("flat_spaxel", np.array(["A"] * n)),
            "fibre_id": ("flat_spaxel", np.arange(n)),
        }
    )
    identity = lambda x: x
    return FitData(ds, *([identity] * 8))


def grid(half_deg=0.2, n=11, ra_c=RA_C, dec_c=DEC_C):
    """A square of points on the sky, half_deg either side of the centre (on the tangent plane)."""
    offsets = np.linspace(-half_deg, half_deg, n)
    dx, dy = np.meshgrid(offsets, offsets)
    dec = dec_c + dy.ravel()
    ra = ra_c + dx.ravel() / np.cos(np.deg2rad(dec))
    return ra, dec


class TestSkyOffsets:
    def test_centre_is_the_middle_of_the_bounding_box(self):
        fd = fit_data([10.0, 10.2, 10.1], [-5.0, -4.8, -4.95])
        assert fd.sky_centre == pytest.approx((10.1, -4.9))

    def test_east_and_north_are_positive(self):
        fd = fit_data(
            [RA_C - 0.01, RA_C + 0.01, RA_C, RA_C], [DEC_C, DEC_C, DEC_C - 0.01, DEC_C + 0.01]
        )
        x, y = np.asarray(fd.sky_x), np.asarray(fd.sky_y)
        assert x[1] > 0 > x[0] and y[3] > 0 > y[2]

    def test_offsets_are_in_arcsec(self):
        # A point 0.01° due north of the centre is 36″ north
        fd = fit_data([RA_C, RA_C, RA_C], [DEC_C - 0.01, DEC_C, DEC_C + 0.01])
        assert np.asarray(fd.sky_y)[2] == pytest.approx(36.0, rel=1e-6)
        assert np.asarray(fd.sky_x)[2] == pytest.approx(0.0, abs=1e-6)

    def test_cos_dec_is_applied(self):
        # Equal steps in RA are shorter on the sky away from the equator
        fd = fit_data([RA_C - 0.01, RA_C, RA_C + 0.01], [DEC_C] * 3)
        expected = 0.01 * 3600 * np.cos(np.deg2rad(DEC_C))
        assert np.asarray(fd.sky_x)[2] == pytest.approx(expected, rel=1e-6)

    def test_round_trip(self):
        ra, dec = grid()
        fd = fit_data(ra, dec)
        ra2, dec2 = fd.sky_to_radec(
            np.asarray(fd.sky_x, dtype=float), np.asarray(fd.sky_y, dtype=float)
        )
        assert np.allclose(ra2, ra, atol=1e-6 / 3600) and np.allclose(dec2, dec, atol=1e-6 / 3600)

    def test_float32_positions_are_transformed_in_float64(self):
        # float32 RA resolves only ~60 mas at 270°; the offsets must not be worse than the input
        ra, dec = grid()
        fd32, fd64 = (
            fit_data(ra, dec, np.float32),
            fit_data(np.float32(ra).astype(float), np.float32(dec).astype(float)),
        )
        assert np.allclose(fd32._sky_xy()[0], fd64._sky_xy()[0], atol=1e-9)

    def test_across_ra_zero(self):
        fd = fit_data([359.9, 359.95, 0.0, 0.05, 0.1], [-0.05, 0.0, 0.0, 0.05, 0.05])
        assert fd.sky_centre[0] == pytest.approx(0.0, abs=1e-9)
        assert np.all(np.diff(np.asarray(fd.sky_x)) > 0)

    def test_with_sky_origin(self):
        ra, dec = grid()
        fd = fit_data(ra, dec)
        moved = fd.with_sky_origin(RA_C + 0.1, DEC_C)
        assert moved.sky_centre == (RA_C + 0.1, DEC_C)
        assert fd.sky_origin is None  # the original is unchanged
        assert np.asarray(moved.sky_x).mean() < np.asarray(fd.sky_x).mean()

    def test_sky_data(self):
        pytest.importorskip("spectracles")
        ra, dec = grid()
        fd = fit_data(ra, dec)
        data = fd.sky_data
        assert np.allclose(data.x, fd.sky_x) and np.allclose(data.y, fd.sky_y)
        assert np.array_equal(data.idx, np.arange(len(ra)))


class TestDeprecated:
    @pytest.mark.parametrize("name", ["α", "δ"])
    def test_properties_warn(self, name):
        fd = fit_data(*grid())
        with pytest.warns(DeprecationWarning, match=f"FitData.{name}"):
            getattr(fd, name)

    def test_methods_warn(self):
        fd = fit_data(*grid())
        for call in (
            lambda: fd.predict_α(0.0),
            lambda: fd.predict_δ(0.0),
            lambda: fd.arcsec_to_model_domain(60.0),
        ):
            with pytest.warns(DeprecationWarning):
                call()

    def test_new_api_doesnt_warn(self):
        fd = fit_data(*grid())
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            _ = (fd.sky_x, fd.sky_y, fd.sky_centre, fd.spaxel_idx, fd.tile_idx)
