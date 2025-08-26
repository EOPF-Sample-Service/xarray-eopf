#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import fsspec
import numpy as np
import pytest
import xarray as xr
import zarr

from tests.helpers import make_s3_olci_erf
from xarray_eopf.amode import AnalysisModeRegistry
from xarray_eopf.amodes.sentinel3 import Sen3Ol1Efr, register


class Sentinel3AnalysisModeTest(TestCase):
    def test_register(self):
        registry = AnalysisModeRegistry()
        register(registry)
        self.assertEqual(4, len(list(registry.keys())))


# noinspection PyUnresolvedReferences
class Sen3TestMixin:
    def test_get_applicable_params(self: TestCase):
        self.assertEqual(
            {},
            self.mode.get_applicable_params(),
        )
        self.assertEqual(
            {"resolution": 10, "interp_methods": 1, "agg_methods": {"scl": "mode"}},
            self.mode.get_applicable_params(
                resolution=10,
                interp_methods=1,
                agg_methods={"scl": "mode"},
            ),
        )

    def test_process_metadata(self: TestCase):
        self.assertEqual({}, self.mode.process_metadata(xr.DataTree()))
        dt = xr.DataTree()
        dt.attrs["other_metadata"] = {"test_key": "test_val"}
        self.assertEqual({"test_key": "test_val"}, self.mode.process_metadata(dt))

    @staticmethod
    def create_simple_dataset() -> xr.Dataset:
        def make_band():
            return xr.DataArray(
                np.zeros((10, 10)),
                coords=dict(
                    lat=(("y", "x"), np.arange(100).reshape((10, 10))),
                    lon=(("y", "x"), np.arange(100).reshape((10, 10))),
                ),
                dims=("y", "x"),
            )

        return xr.Dataset(dict(data=make_band()))

    def test_assign_grid_mapping(self: TestCase):
        dataset = self.mode.assign_grid_mapping(self.create_simple_dataset())
        self.assertIn("spatial_ref", dataset)
        self.assertEqual(
            "latitude_longitude", dataset.spatial_ref.attrs.get("grid_mapping_name")
        )
        self.assertEqual("spatial_ref", dataset.data.attrs.get("grid_mapping"))

    def test_transform_dataset(self: TestCase):
        dataset = self.mode.transform_dataset(
            self.create_simple_dataset(), interp_method=0
        )
        self.assertIn("spatial_ref", dataset)
        self.assertEqual(
            "latitude_longitude", dataset.spatial_ref.attrs.get("grid_mapping_name")
        )
        self.assertEqual("spatial_ref", dataset.data.attrs.get("grid_mapping"))

    def assert_transform_datatree_ok(self, original_dt: xr.DataTree):
        dt = self.mode.transform_datatree(original_dt)
        self.assertIs(dt, original_dt)

    def assert_convert_datatree_ok(
        self,
        original_dt: xr.DataTree,
        expected_var_names: list[str],
        expected_size: (int, int),
    ):
        ds = self.mode.convert_datatree(
            original_dt, includes=expected_var_names, resolution=0.1
        )
        self.assertIsInstance(ds, xr.Dataset)
        self.assertEqual(
            expected_var_names,
            sorted(ds.data_vars.keys()),
        )
        for var_name, var in ds.data_vars.items():
            self.assertEqual(expected_size, var.shape, msg=var_name)

        # noinspection PyTypeChecker
        self.assertCountEqual(
            ["spatial_ref", "lat", "lon"],
            ds.coords.keys(),
        )
        self.assertEqual((expected_size[0],), ds.lat.shape, msg="lat")
        self.assertEqual((expected_size[1],), ds.lon.shape, msg="lon")

    def assert_convert_datatree_fail(self, original_dt: xr.DataTree):
        with pytest.raises(ValueError, match="No variables selected"):
            self.mode.convert_datatree(original_dt, includes="bibo")


class OLCIERFTest(Sen3TestMixin, TestCase):
    mode = Sen3Ol1Efr()

    def test_is_valid_source_ok(self):
        self.assertTrue(self.mode.is_valid_source("data/S3A_OL_1_EFR_20240201.zarr"))
        self.assertTrue(
            self.mode.is_valid_source(
                zarr.storage.DirectoryStore("data/S3B_OL_1_EFR_20240201.zarr")
            )
        )
        fs: fsspec.AbstractFileSystem = fsspec.filesystem("local")
        self.assertTrue(
            self.mode.is_valid_source(
                fs.get_mapper(root="data/S3B_OL_1_EFR_20240201.zarr")
            )
        )

    def test_is_no_valid_source(self):
        self.assertFalse(self.mode.is_valid_source("data/S3C_OL_1_EFR_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_transform_datatree(self):
        self.assert_transform_datatree_ok(make_s3_olci_erf())

    def test_convert_datatree(self):
        self.assert_convert_datatree_ok(
            make_s3_olci_erf(size=100),
            expected_var_names=[
                "oa01_radiance",
                "oa02_radiance",
                "oa03_radiance",
            ],
            expected_size=(168, 208),
        )

    def test_convert_datatree_fail(self):
        self.assert_convert_datatree_fail(make_s3_olci_erf(size=48))
