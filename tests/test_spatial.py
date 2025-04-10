#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
from collections.abc import Mapping
from typing import Hashable
from unittest import TestCase

import xarray as xr

from tests.helpers import make_s2_msi
from xarray_eopf.flatten import flatten_datatree
from xarray_eopf.spatial import rescale_spatial_vars


class RescaleSpatialVarsTest(TestCase):
    ds: xr.Dataset

    @classmethod
    def setUpClass(cls):
        dt = make_s2_msi(size_r10m=48)
        cls.ds = flatten_datatree(dt)

    def test_s2_msi_to_10m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r10m_b02")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 48)
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars)
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 48)

    def test_s2_msi_to_20m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r20m_b05")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 24)

    def test_s2_msi_to_60m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r60m_b01")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 8)

    def assert_rescale_spatial_vars_ok(
        self, rescaled_vars: Mapping[Hashable, xr.DataArray], target_res: int
    ):
        self.assertIsInstance(rescaled_vars, dict)
        self.assertEqual(len(self.ds.data_vars), len(rescaled_vars))
        # Force resampling
        for var_name, var in rescaled_vars.items():
            array = var.values
            self.assertEqual((target_res, target_res), array.shape[-2:])
