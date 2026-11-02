"""test_filtering.py - which pixels are flagged as bad, and which strategies are accepted."""

import numpy as np
import pytest
import xarray as xr

from lvm_tools import DataConfig
from lvm_tools.fit_data.filtering import get_where_bad
from lvm_tools.fit_data.normalisation import get_norm_funcs


class TestWhereBad:
    def flagged(self, values, bad_range):
        return get_where_bad(xr.DataArray(np.array(values, dtype=float)), bad_range).values

    def test_below_the_range(self):
        assert list(self.flagged([-5.0, 0.0, 0.5], (0.0, 1.0))) == [True, True, False]

    def test_above_the_range(self):
        assert list(self.flagged([0.5, 1.0, 5.0], (0.0, 1.0))) == [False, False, True]

    def test_nans_are_left_to_nans_strategy(self):
        assert list(self.flagged([np.nan], (0.0, 1.0))) == [False]

    def test_default_upper_bound_flags_nothing_above(self):
        assert not self.flagged([1e30], (0.0, np.inf)).any()


class TestStrategies:
    @pytest.mark.parametrize("strategy", ["pixel", "spaxel"])
    def test_nans_strategies(self, strategy):
        DataConfig(nans_strategy=strategy)

    @pytest.mark.parametrize("strategy", [None, "spaxel_max", "fibre"])
    def test_unsupported_nans_strategies(self, strategy):
        with pytest.raises(ValueError, match="exclusion strategy"):
            DataConfig(nans_strategy=strategy)

    @pytest.mark.parametrize("strategy", ["pixel", "spaxel", "spaxel_max"])
    def test_bad_flux_strategies(self, strategy):
        DataConfig(F_bad_strategy=strategy)

    @pytest.mark.parametrize("strategy", [None, "fibre"])
    def test_unsupported_bad_flux_strategies(self, strategy):
        with pytest.raises(ValueError, match="exclusion strategy"):
            DataConfig(F_bad_strategy=strategy)


def test_norm_funcs_are_a_reusable_pair():
    funcs = get_norm_funcs(1.0, 2.0)
    normalise, denormalise = funcs
    assert len(funcs) == 2 and denormalise(normalise(5.0)) == 5.0
