#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr

from integration.helpers import assert_dataset_is_chunked
from xarray_eopf.utils import timeit

allowed_open_time = 5  # seconds


class Sentinel2NativeTest(TestCase):
    def test_open_datatree_sen2_l1c_https(self):
        self._test_open_datatree_sen2_l1c(
            "https://objects.eodc.eu:443/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s02msil1c-eu/13/products/cpm_v262/S2A_MSIL1C_20260313T101741_N0512_"
            "R065_T32TLQ_20260313T153853.zarr"
        )

    def test_open_datatree_sen2_l2a_https(self):
        self._test_open_datatree_sen2_l2a(
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s02msil2a-eu/18/products/cpm_v262/S2A_MSIL2A_20260318T125321_"
            "N0512_R138_T28WDT_20260318T204314.zarr"
        )

    def test_open_dataset_sen2_l1c_https(self):
        self._test_open_dataset_sen2_l1c(
            "https://objects.eodc.eu:443/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s02msil1c-eu/13/products/cpm_v262/S2A_MSIL1C_20260313T101741_N0512_"
            "R065_T32TLQ_20260313T153853.zarr"
        )

    def test_open_dataset_sen2_l2a_https(self):
        self._test_open_dataset_sen2_l2a(
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s02msil2a-eu/18/products/cpm_v262/S2A_MSIL2A_20260318T125321_"
            "N0512_R138_T28WDT_20260318T204314.zarr"
        )

    def test_open_dataset_sen2_l1c_https_subgroups(self):
        self._test_open_dataset_sen2_l1c_subgroup(
            "https://objects.eodc.eu:443/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s02msil1c-eu/13/products/cpm_v262/S2A_MSIL1C_20260313T101741_N0512_"
            "R065_T32TLQ_20260313T153853.zarr"
        )

    def test_open_dataset_sen2_l2a_https_subgroups(self):
        self._test_open_dataset_sen2_l2a_subgroup(
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s02msil2a-eu/18/products/cpm_v262/S2A_MSIL2A_20260318T125321_"
            "N0512_R138_T28WDT_20260318T204314.zarr"
        )

    def _test_open_datatree_sen2_l1c(self, url: str):
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            dt = xr.open_datatree(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(23, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r10m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r10m.ds
        self.assertEqual({"y": 10980, "x": 10980}, ds.sizes)
        self.assertCountEqual(["b02", "b03", "b04", "b08"], ds.data_vars.keys())
        assert_dataset_is_chunked(self, ds, verbose=True)

    def _test_open_datatree_sen2_l2a(self, url: str):
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            dt = xr.open_datatree(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(32, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r10m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r10m.ds
        self.assertEqual({"y": 10980, "x": 10980}, ds.sizes)
        self.assertCountEqual(["b02", "b03", "b04", "b08"], ds.data_vars.keys())
        assert_dataset_is_chunked(self, ds, verbose=True)

    def _test_open_dataset_sen2_l1c(self, url: str):
        with timeit(url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(61, len(ds.data_vars))
        self.assertIn(
            "measurements_r10m_b02",
            ds.data_vars,
        )
        da = ds.measurements_r10m_b02
        self.assertEqual(
            {"measurements_r10m_y": 10980, "measurements_r10m_x": 10980}, da.sizes
        )
        assert_dataset_is_chunked(self, ds, verbose=True)

    def _test_open_dataset_sen2_l2a(self, url: str):
        with timeit(url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(83, len(ds.data_vars))
        self.assertIn(
            "measurements_r10m_b02",
            ds.data_vars,
        )
        da = ds.measurements_r10m_b02
        self.assertEqual(
            {"measurements_r10m_y": 10980, "measurements_r10m_x": 10980}, da.sizes
        )
        assert_dataset_is_chunked(self, ds, verbose=True)

    def _test_open_dataset_sen2_l1c_subgroup(self, url: str):
        url_subgroup = f"{url}/measurements/reflectance/r60m"
        with timeit(url_subgroup) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(
                url_subgroup, engine="eopf-zarr", op_mode="native", chunks={}
            )
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertCountEqual(["b01", "b09", "b10"], ds.data_vars)
        da = ds.b01
        self.assertEqual({"y": 1830, "x": 1830}, da.sizes)
        assert_dataset_is_chunked(self, ds, verbose=True)

    def _test_open_dataset_sen2_l2a_subgroup(self, url: str):
        url_subgroup = f"{url}/measurements/reflectance/r10m"
        with timeit(url_subgroup) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(
                url_subgroup, engine="eopf-zarr", op_mode="native", chunks={}
            )
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertCountEqual(["b02", "b03", "b04", "b08"], ds.data_vars)
        da = ds.b02
        self.assertEqual({"y": 10980, "x": 10980}, da.sizes)
        assert_dataset_is_chunked(self, ds, verbose=True)
