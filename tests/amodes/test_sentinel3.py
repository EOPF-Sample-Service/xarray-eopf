#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Sequence
from unittest import TestCase

import fsspec
import numpy as np
import pyproj
import pytest
import xarray as xr
import zarr

from tests.helpers import make_s3_olci_efr, make_s3_slstr_lst, make_s3_slstr_rbt
from xarray_eopf.amode import AnalysisModeRegistry
from xarray_eopf.amodes.sentinel3 import (
    Sen3Ol1Efr,
    Sen3Sl1Rbt,
    Sen3Sl2Lst,
    register,
)
from xarray_eopf.constants import FloatInt


class Sentinel3AnalysisModeTest(TestCase):
    def test_register(self):
        registry = AnalysisModeRegistry()
        register(registry)
        self.assertEqual(5, len(list(registry.keys())))


# noinspection PyUnresolvedReferences
class Sen3TestMixin:
    def test_get_applicable_params(self: TestCase):
        self.assertEqual(
            {
                "resolution": 10,
                "interp_methods": 1,
                "bbox": [1, 3, 4, 5],
                "crs": pyproj.CRS.from_string("EPSG:4326"),
                "agg_methods": {"scl": "mode"},
            },
            self.mode.get_applicable_params(
                resolution=10,
                interp_methods=1,
                bbox=[1, 3, 4, 5],
                crs="EPSG:4326",
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
            self.create_simple_dataset(), {"stac_meta": "test"}, interp_method=0
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
        expected_size: tuple[int, int],
        resolution: FloatInt | tuple[FloatInt, FloatInt] | None = None,
        bbox: Sequence[float | int] | None = None,
    ):
        ds = self.mode.convert_datatree(
            original_dt, includes=expected_var_names, resolution=resolution, bbox=bbox
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

    def assert_convert_datatree_fail_with_include_exclude(
        self, original_dt: xr.DataTree
    ):
        with pytest.raises(ValueError, match="No variables selected"):
            self.mode.convert_datatree(original_dt, includes=".+", excludes=".+")


class OlciEfrTest(Sen3TestMixin, TestCase):
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
        self.assert_transform_datatree_ok(make_s3_olci_efr(size=100))

    def test_convert_datatree(self):
        self.assert_convert_datatree_ok(
            make_s3_olci_efr(size=100),
            expected_var_names=[
                "oa01_radiance",
                "oa02_radiance",
                "oa03_radiance",
            ],
            expected_size=(166, 207),
            resolution=0.1,
        )

    def test_convert_datatree_bbox(self):
        self.assert_convert_datatree_ok(
            make_s3_olci_efr(size=100),
            expected_var_names=[
                "oa01_radiance",
                "oa02_radiance",
                "oa03_radiance",
            ],
            expected_size=(372, 454),
            bbox=[1, 55, 3, 56],
        )

    def test_convert_datatree_default_res(self):
        self.assert_convert_datatree_ok(
            make_s3_olci_efr(size=3000),
            expected_var_names=[
                "oa01_radiance",
                "oa02_radiance",
                "oa03_radiance",
            ],
            expected_size=(6121, 4773),
        )

    def test_convert_datatree_raise_warning(self):
        dt = make_s3_olci_efr(size=3000)
        with self.assertWarns(UserWarning) as cm:
            _ = self.mode.convert_datatree(dt, bbox=[130, 20, 140, 30])
        self.assertIn("Clipping with the specified bounding", str(cm.warning))

    def test_convert_datatree_fail(self):
        self.assert_convert_datatree_fail(make_s3_olci_efr(size=48))

    def test_convert_datatree_fail_include_exclude_overlap(self):
        self.assert_convert_datatree_fail_with_include_exclude(
            make_s3_olci_efr(size=48)
        )

    def test_convert_datatree_sets_other_metadata_as_attrs(self):
        dt = make_s3_olci_efr(size=100)
        dt.attrs["other_metadata"] = {"test_key": "test_val"}
        ds = self.mode.convert_datatree(
            dt,
            includes=["oa01_radiance"],
            resolution=0.1,
        )
        self.assertEqual({"test_key": "test_val"}, ds.attrs)


class SlstrRbtTest(Sen3TestMixin, TestCase):
    mode = Sen3Sl1Rbt()

    def test_is_valid_source_ok(self):
        self.assertTrue(self.mode.is_valid_source("data/S3A_SL_1_RBT_20240201.zarr"))
        self.assertTrue(
            self.mode.is_valid_source(
                zarr.storage.DirectoryStore("data/S3B_SL_1_RBT_20240201.zarr")
            )
        )

    def test_is_no_valid_source(self):
        self.assertFalse(self.mode.is_valid_source("data/S3C_SL_1_RBT_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_transform_datatree(self):
        self.assert_transform_datatree_ok(make_s3_slstr_rbt(size=100))

    def test_convert_datatree(self):
        self.assert_convert_datatree_ok(
            make_s3_slstr_rbt(size=100),
            expected_var_names=[
                "s1_radiance_an",
                "s7_bt_in",
                "s7_bt_io",
            ],
            expected_size=(167, 209),
            resolution=0.1,
        )

    def test_convert_datatree_bbox(self):
        self.assert_convert_datatree_ok(
            make_s3_slstr_rbt(size=1000),
            expected_var_names=[
                "s1_radiance_an",
                "s7_bt_in",
                "s7_bt_io",
            ],
            expected_size=(223, 272),
            bbox=[5, 52, 7, 53],
        )

    def test_convert_datatree_default_res_1000(self):
        self.assert_convert_datatree_ok(
            make_s3_slstr_rbt(size=1000),
            expected_var_names=[
                "s7_bt_in",
                "s7_bt_io",
            ],
            expected_size=(1838, 1434),
        )

    def test_convert_datatree_default_res_500(self):
        self.assert_convert_datatree_ok(
            make_s3_slstr_rbt(size=1000),
            expected_var_names=[
                "s1_radiance_an",
                "s7_bt_in",
                "s7_bt_io",
            ],
            expected_size=(3676, 2867),
        )

    def test_convert_datatree_raise_warning(self):
        dt = make_s3_slstr_rbt(size=1000)
        with self.assertWarns(UserWarning) as cm:
            _ = self.mode.convert_datatree(dt, bbox=[130, 20, 140, 30])
        self.assertIn("Clipping with the specified bounding", str(cm.warning))

    def test_convert_datatree_fail(self):
        self.assert_convert_datatree_fail(make_s3_slstr_rbt(size=48))

    def test_convert_datatree_fail_include_exclude_overlap(self):
        self.assert_convert_datatree_fail_with_include_exclude(
            make_s3_slstr_rbt(size=48)
        )

    def test_get_outer_bbox(self):
        bboxs = np.array([[-2, 10, 8, 20], [2, 12, 13, 25]])
        expected = [-2, 10, 13, 25]
        self.assertEqual(expected, self.mode._get_outer_bbox(bboxs))

        bboxs = np.array([[175, 10, -175, 20], [179, 12, -165, 25]])
        expected = [175, 10, -165, 25]
        self.assertEqual(expected, self.mode._get_outer_bbox(bboxs))


class SlstrLstTest(Sen3TestMixin, TestCase):
    mode = Sen3Sl2Lst()

    def test_is_valid_source_ok(self):
        self.assertTrue(self.mode.is_valid_source("data/S3A_SL_2_LST_20240201.zarr"))
        self.assertTrue(
            self.mode.is_valid_source(
                zarr.storage.DirectoryStore("data/S3B_SL_2_LST_20240201.zarr")
            )
        )

    def test_is_no_valid_source(self):
        self.assertFalse(self.mode.is_valid_source("data/S3C_SL_2_LST_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_transform_datatree(self):
        self.assert_transform_datatree_ok(make_s3_slstr_lst(size=100))

    def test_convert_datatree(self):
        self.assert_convert_datatree_ok(
            make_s3_slstr_lst(size=1000),
            expected_var_names=["lst"],
            expected_size=(1837, 1433),
        )

    def test_convert_datatree_default_res(self):
        self.assert_convert_datatree_ok(
            make_s3_slstr_lst(size=100),
            expected_var_names=["lst"],
            expected_size=(166, 207),
            resolution=0.1,
        )

    def test_convert_datatree_bbox(self):
        self.assert_convert_datatree_ok(
            make_s3_slstr_lst(size=1000),
            expected_var_names=["lst"],
            expected_size=(112, 148),
            bbox=[1, 55, 3, 56],
        )

    def test_convert_datatree_fail(self):
        self.assert_convert_datatree_fail(make_s3_slstr_lst(size=48))

    def test_convert_datatree_fail_include_exclude_overlap(self):
        self.assert_convert_datatree_fail_with_include_exclude(
            make_s3_slstr_lst(size=48)
        )
