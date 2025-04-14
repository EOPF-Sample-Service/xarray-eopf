#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
from collections.abc import Mapping
from typing import Hashable
from unittest import TestCase

import pytest
import xarray as xr

from tests.helpers import make_s2_msi
from xarray_eopf.flatten import flatten_datatree
from xarray_eopf.spatial import rescale_spatial_vars
from xarray_eopf.spatial import get_spline_order, get_agg_method


class RescaleSpatialVarsTest(TestCase):
    ds: xr.Dataset

    @classmethod
    def setUpClass(cls):
        dt = make_s2_msi(r10m_size=48)
        cls.ds = flatten_datatree(dt)

    def test_s2_msi_to_10m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r10m_b02")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 48)
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars)
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 48)

        self.assertEqual(None, rescaled_vars["r10m_b02"].attrs.get("history"))
        self.assertEqual(
            (
                "Up-sampling by factors"
                " r10m_x=0.5 and r10m_y=0.5"
                " using spline interpolation of order 0."
            ),
            rescaled_vars["r20m_b05"].attrs.get("history"),
        )
        self.assertEqual(
            (
                "Up-sampling by factors"
                " r10m_x=0.166667 and r10m_y=0.166667"
                " using spline interpolation of order 0."
            ),
            rescaled_vars["r60m_b01"].attrs.get("history"),
        )

    def test_s2_msi_to_20m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r20m_b05")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 24)

        self.assertEqual(
            "Down-sampling by factors r20m_x=2 and r20m_y=2"
            " using aggregation method 'max'.",
            rescaled_vars["r10m_b02"].attrs.get("history"),
        )
        self.assertEqual(
            None,
            rescaled_vars["r20m_b05"].attrs.get("history"),
        )
        self.assertEqual(
            (
                "Up-sampling by factors r20m_x=0.333333 and r20m_y=0.333333 using "
                "spline interpolation of order 0."
            ),
            rescaled_vars["r60m_b01"].attrs.get("history"),
        )

    def test_s2_msi_to_60m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r60m_b01")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 8)

        self.assertEqual(
            "Down-sampling by factors r60m_x=6 and r60m_y=6"
            " using aggregation method 'max'.",
            rescaled_vars["r10m_b02"].attrs.get("history"),
        )
        self.assertEqual(
            "Down-sampling by factors r60m_x=3 and r60m_y=3"
            " using aggregation method 'max'.",
            rescaled_vars["r20m_b05"].attrs.get("history"),
        )
        self.assertEqual(
            None,
            rescaled_vars["r60m_b01"].attrs.get("history"),
        )

    def assert_rescale_spatial_vars_ok(
        self, rescaled_vars: Mapping[Hashable, xr.DataArray], target_res: int
    ):
        self.assertIsInstance(rescaled_vars, dict)
        self.assertEqual(len(self.ds.data_vars), len(rescaled_vars))
        # Force resampling
        for var_name, var in rescaled_vars.items():
            array = var.values
            self.assertEqual((target_res, target_res), array.shape[-2:])


class UtilsTest(TestCase):
    def test_get_agg_method(self):
        self.assertEqual("min", get_agg_method("x", "min"))
        self.assertEqual("median", get_agg_method("x", {"x": "median"}))
        self.assertEqual("mean", get_agg_method("y", {"x": "median"}))
        self.assertEqual(
            "max", get_agg_method("y", {"x": "median"}, is_categorical=True)
        )
        with pytest.raises(ValueError, match="Unknown aggregation method: mode"):
            get_agg_method("x", "mode")

    def test_spline_order(self):
        self.assertEqual(1, get_spline_order("x", 1))
        self.assertEqual(2, get_spline_order("x", {"x": 2}))
        self.assertEqual(3, get_spline_order("y", {"x": 2}))
        self.assertEqual(0, get_spline_order("y", {"x": 2}, is_categorical=True))
        with pytest.raises(ValueError, match="Unknown spline order: 4"):
            get_spline_order("x", 4)
