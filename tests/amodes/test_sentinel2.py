#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import numpy as np
import xarray as xr

from xarray_eopf.amode import AnalysisModeRegistry
from xarray_eopf.amodes.sentinel2 import register
from xarray_eopf.amodes.sentinel2 import MSIL1C
from xarray_eopf.amodes.sentinel2 import MSIL2A


class Sentinel2AnalysisModeTest(TestCase):
    def test_register(self):
        registry = AnalysisModeRegistry()
        register(registry)
        self.assertEqual(2, len(list(registry.keys())))


# noinspection PyUnresolvedReferences
class MSITestMixin:
    def test_is_valid_source(self: TestCase):
        pass

    def test_get_applicable_params(self: TestCase):
        self.assertEqual(
            {"resolution": 10, "spline_order": 2},
            self.mode.get_applicable_params(
                resolution=10, spline_order=2, temp_file="."
            ),
        )

    def test_process_metadata(self: TestCase):
        self.assertEqual({}, self.mode.process_metadata(xr.DataTree()))

    def test_assign_grid_mapping(self: TestCase):
        dataset = self.mode.assign_grid_mapping(
            xr.Dataset(
                dict(
                    b01=xr.DataArray(np.zeros((10, 10)), dims=("y", "x")),
                    b02=xr.DataArray(np.zeros((10, 10)), dims=("y", "x")),
                    b03=xr.DataArray(np.zeros((10, 10)), dims=("y", "x")),
                ),
                attrs=dict(horizontal_CRS_code="EPSG:32632"),
            )
        )
        self.assertIn("spatial_ref", dataset)
        self.assertEqual(
            "transverse_mercator", dataset.spatial_ref.attrs.get("grid_mapping_name")
        )
        self.assertEqual("spatial_ref", dataset.b01.attrs.get("grid_mapping"))
        self.assertEqual("spatial_ref", dataset.b02.attrs.get("grid_mapping"))
        self.assertEqual("spatial_ref", dataset.b03.attrs.get("grid_mapping"))


class MSIL1CTest(MSITestMixin, TestCase):
    mode = MSIL1C()

    def test_is_valid_source(self):
        self.assertTrue(self.mode.is_valid_source("S2A_MSIL1C_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source("S2A_MSIL2A_20240201.zarr"))


class MSIL2ATest(MSITestMixin, TestCase):
    mode = MSIL2A()

    def test_is_valid_source(self):
        self.assertTrue(self.mode.is_valid_source("S2A_MSIL2A_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source("S2A_MSIL1C_20240201.zarr"))
