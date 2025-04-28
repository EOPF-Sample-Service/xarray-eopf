#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import fsspec
import numpy as np
import pytest
import xarray as xr
import zarr.storage

from tests.helpers import make_s2_msi_l1c, make_s2_msi_l2a
from xarray_eopf.amode import AnalysisModeRegistry
from xarray_eopf.amodes.sentinel2 import MSIL1C, MSIL2A, register


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
            {
                "spline_orders": {0: ["scl"]},
                "agg_methods": {"center": ["scl"]},
            },
            self.mode.get_applicable_params(),
        )
        self.assertEqual(
            {"resolution": 10, "spline_orders": 2, "agg_methods": {"mode": ["scl"]}},
            self.mode.get_applicable_params(
                resolution=10,
                spline_orders=2,
                agg_methods={"mode": ["scl"]},
                temp_file=".",
            ),
        )

    def test_process_metadata(self: TestCase):
        self.assertEqual({}, self.mode.process_metadata(xr.DataTree()))

    def test_assign_grid_mapping(self: TestCase):
        def make_band():
            return xr.DataArray(
                np.zeros((10, 10)), dims=("y", "x"), attrs={"proj:epsg": 32632}
            )

        dataset = self.mode.assign_grid_mapping(
            xr.Dataset(
                dict(
                    b01=make_band(),
                    b02=make_band(),
                    b03=make_band(),
                ),
            )
        )
        self.assertIn("spatial_ref", dataset)
        self.assertEqual(
            "transverse_mercator", dataset.spatial_ref.attrs.get("grid_mapping_name")
        )
        self.assertEqual("spatial_ref", dataset.b01.attrs.get("grid_mapping"))
        self.assertEqual("spatial_ref", dataset.b02.attrs.get("grid_mapping"))
        self.assertEqual("spatial_ref", dataset.b03.attrs.get("grid_mapping"))

    def test_assign_grid_mapping_fail(self: TestCase):
        def make_band():
            return xr.DataArray(
                np.zeros((10, 10)), dims=("y", "x"), attrs={"proj:epsg": -1}
            )

        dataset = self.mode.assign_grid_mapping(
            xr.Dataset(
                dict(
                    b01=make_band(),
                    b02=make_band(),
                    b03=make_band(),
                ),
            )
        )
        self.assertNotIn("spatial_ref", dataset)
        self.assertEqual(None, dataset.b01.attrs.get("grid_mapping"))
        self.assertEqual(None, dataset.b02.attrs.get("grid_mapping"))
        self.assertEqual(None, dataset.b03.attrs.get("grid_mapping"))

    def assert_transform_datatree_ok(self, original_dt: xr.DataTree):
        dt = self.mode.transform_datatree(original_dt, resolution=10)
        self.assertIs(dt, original_dt)

    def assert_convert_datatree_ok(
        self,
        original_dt: xr.DataTree,
        expected_var_names: list[str],
        expected_size: int,
    ):
        ds = self.mode.convert_datatree(original_dt, resolution=10)
        self.assertIsInstance(ds, xr.Dataset)
        self.assertEqual(
            expected_var_names,
            sorted(ds.data_vars.keys()),
        )
        for var_name, var in ds.data_vars.items():
            self.assertEqual((expected_size, expected_size), var.shape, msg=var_name)

        # noinspection PyTypeChecker
        self.assertEqual(
            ["spatial_ref", "x", "y"],
            sorted(ds.coords.keys()),
        )
        self.assertEqual((expected_size,), ds.x.shape, msg="x")
        self.assertEqual((expected_size,), ds.y.shape, msg="y")

    def assert_convert_datatree_fail(self, original_dt: xr.DataTree):
        with pytest.raises(ValueError, match="No variables selected"):
            self.mode.convert_datatree(original_dt, includes="bibo")

        with pytest.raises(
            ValueError,
            match=(
                "No reference variable found."
                " At least one of the selected variables must have"
                " a native resolution that equals the target resolution."
            ),
        ):
            self.mode.convert_datatree(original_dt, includes=["b11", "b12"])


class MSIL1CTest(MSITestMixin, TestCase):
    mode = MSIL1C()

    def test_is_valid_source_ok(self):
        self.assertTrue(self.mode.is_valid_source("data/S2A_MSIL1C_20240201.zarr"))
        self.assertTrue(
            self.mode.is_valid_source(
                zarr.storage.DirectoryStore("data/S2A_MSIL1C_20240201.zarr")
            )
        )
        fs: fsspec.AbstractFileSystem = fsspec.filesystem("local")
        self.assertTrue(
            self.mode.is_valid_source(
                fs.get_mapper(root="data/S2A_MSIL1C_20240201.zarr")
            )
        )

    def test_is_no_valid_source(self):
        self.assertFalse(self.mode.is_valid_source("data/S2A_MSIL2A_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_transform_datatree(self):
        self.assert_transform_datatree_ok(make_s2_msi_l1c())

    def test_convert_datatree(self):
        self.assert_convert_datatree_ok(
            make_s2_msi_l1c(r10m_size=48),
            expected_var_names=[
                "b01",
                "b02",
                "b03",
                "b04",
                "b05",
                "b06",
                "b07",
                "b08",
                "b09",
                "b10",
                "b11",
                "b12",
                "b8a",
            ],
            expected_size=48,
        )

    def test_convert_datatree_fail(self):
        self.assert_convert_datatree_fail(make_s2_msi_l1c(r10m_size=48))


class MSIL2ATest(MSITestMixin, TestCase):
    mode = MSIL2A()

    def test_is_valid_source(self):
        self.assertTrue(self.mode.is_valid_source("S2A_MSIL2A_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source("S2A_MSIL1C_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_transform_datatree(self):
        self.assert_transform_datatree_ok(make_s2_msi_l2a())

    def test_convert_datatree_ok(self):
        self.assert_convert_datatree_ok(
            make_s2_msi_l2a(r10m_size=48),
            expected_var_names=[
                "b01",
                "b02",
                "b03",
                "b04",
                "b05",
                "b06",
                "b07",
                "b08",
                "b11",
                "b12",
                "b8a",
                "cld",
                "scl",
                "snw",
            ],
            expected_size=48,
        )

    def test_convert_datatree_fail(self):
        self.assert_convert_datatree_fail(make_s2_msi_l2a(r10m_size=48))
