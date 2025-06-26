#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr

from integration.helpers import assert_data_arrays_are_chunked
from xarray_eopf.constants import DEFAULT_ENDPOINT_URL
from xarray_eopf.spatial import get_spatial_vars
from xarray_eopf.utils import timeit

s02msil1c_bucket = "e05ab01a9d56408d82ac32d69a5aae2a:202504-s02msil1c"
s02msil2a_bucket = "e05ab01a9d56408d82ac32d69a5aae2a:202504-s02msil2a"
path_prefix = "15/products/cpm_v256"
l1c_filename = "S2B_MSIL1C_20250415T142749_N0511_R139_T25WEV_20250415T180239.zarr"
l2a_filename = "S2B_MSIL2A_20250415T142749_N0511_R139_T25WEV_20250415T181516.zarr"

allowed_open_time = 5  # seconds


class Sentinel2NativeTest(TestCase):
    def test_open_datatree_sen2_l1c_s3(self):
        self._test_open_datatree_sen2_l1c(f"s3://{s02msil1c_bucket}/{path_prefix}")

    def test_open_dataset_sen2_l1c_s3(self):
        self._test_open_dataset_sen2_l1c(f"s3://{s02msil1c_bucket}/{path_prefix}")

    def test_open_dataset_sen2_l1c_s3_subgroups(self):
        self._test_open_dataset_sen2_l1c_subgroup(
            f"s3://{s02msil1c_bucket}/{path_prefix}"
        )

    def test_open_dataset_sen2_l2a_s3_subgroups(self):
        self._test_open_dataset_sen2_l2a_subgroup(
            f"s3://{s02msil2a_bucket}/{path_prefix}"
        )

    def test_open_dataset_sen2_l2a_s3(self):
        self._test_open_dataset_sen2_l2a(f"s3://{s02msil2a_bucket}/{path_prefix}")

    def test_open_datatree_sen2_l1c_https(self):
        self._test_open_datatree_sen2_l1c(
            f"{DEFAULT_ENDPOINT_URL}/{s02msil1c_bucket}/{path_prefix}"
        )

    def test_open_dataset_sen2_l1c_https(self):
        self._test_open_dataset_sen2_l1c(
            f"{DEFAULT_ENDPOINT_URL}/{s02msil1c_bucket}/{path_prefix}"
        )

    def test_open_dataset_sen2_l2a_https(self):
        self._test_open_dataset_sen2_l2a(
            f"{DEFAULT_ENDPOINT_URL}/{s02msil2a_bucket}/{path_prefix}"
        )

    def test_open_dataset_sen2_l1c_https_subgroups(self):
        self._test_open_dataset_sen2_l1c_subgroup(
            f"{DEFAULT_ENDPOINT_URL}/{s02msil1c_bucket}/{path_prefix}"
        )

    def test_open_dataset_sen2_l2a_https_subgroups(self):
        self._test_open_dataset_sen2_l2a_subgroup(
            f"{DEFAULT_ENDPOINT_URL}/{s02msil2a_bucket}/{path_prefix}"
        )

    def _test_open_datatree_sen2_l1c(self, url_prefix: str):
        # noinspection PyTypeChecker
        url = f"{url_prefix}/{l1c_filename}"
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            dt = xr.open_datatree(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(25, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r10m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r10m.ds
        self.assertEqual({"y": 10980, "x": 10980}, ds.sizes)
        spatial_vars = get_spatial_vars(ds)
        self.assertEqual(
            ["b02", "b03", "b04", "b08"], sorted(map(str, spatial_vars.keys()))
        )
        assert_data_arrays_are_chunked(self, spatial_vars, verbose=True)

    def _test_open_datatree_sen2_l2a(self, url_prefix: str):
        url = f"{url_prefix}/{l2a_filename}.zarr"
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            dt = xr.open_datatree(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(36, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r10m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r10m.ds
        self.assertEqual({"y": 10980, "x": 10980}, ds.sizes)
        spatial_vars = get_spatial_vars(ds)
        self.assertEqual(
            ["b02", "b03", "b04", "b08"], sorted(map(str, spatial_vars.keys()))
        )
        assert_data_arrays_are_chunked(self, spatial_vars, verbose=True)

    def _test_open_dataset_sen2_l1c(self, url_prefix: str):
        url = f"{url_prefix}/{l1c_filename}"
        with timeit(url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(62, len(ds.data_vars))
        self.assertIn(
            "measurements_r10m_b02",
            ds.data_vars,
        )
        da = ds.measurements_r10m_b02
        self.assertEqual(
            {"measurements_r10m_y": 10980, "measurements_r10m_x": 10980}, da.sizes
        )
        spatial_vars = get_spatial_vars(ds)
        self.assertEqual(43, len(spatial_vars))
        assert_data_arrays_are_chunked(self, spatial_vars, verbose=True)

    def _test_open_dataset_sen2_l1c_subgroup(self, url_prefix: str):
        url = f"{url_prefix}/{l1c_filename}/measurements/reflectance/r60m"
        with timeit(url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertCountEqual(["b01", "b09", "b10"], ds.data_vars)
        da = ds.b01
        self.assertEqual({"y": 1830, "x": 1830}, da.sizes)
        spatial_vars = get_spatial_vars(ds)
        assert_data_arrays_are_chunked(self, spatial_vars, verbose=True)

    def _test_open_dataset_sen2_l2a_subgroup(self, url_prefix: str):
        url = f"{url_prefix}/{l2a_filename}/measurements/reflectance/r10m"
        with timeit(url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertCountEqual(["b02", "b03", "b04", "b08"], ds.data_vars)
        da = ds.b02
        self.assertEqual({"y": 10980, "x": 10980}, da.sizes)
        spatial_vars = get_spatial_vars(ds)
        assert_data_arrays_are_chunked(self, spatial_vars, verbose=True)

    def _test_open_dataset_sen2_l2a(self, url_prefix: str):
        url = f"{url_prefix}/{l2a_filename}"
        with timeit(url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(86, len(ds.data_vars))
        self.assertIn(
            "measurements_r10m_b02",
            ds.data_vars,
        )
        da = ds.measurements_r10m_b02
        self.assertEqual(
            {"measurements_r10m_y": 10980, "measurements_r10m_x": 10980}, da.sizes
        )
        spatial_vars = get_spatial_vars(ds)
        self.assertEqual(67, len(spatial_vars))
        assert_data_arrays_are_chunked(self, spatial_vars, verbose=True)
