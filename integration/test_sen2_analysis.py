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
show_chunking = False


class Sentinel2AnalysisTest(TestCase):
    def test_open_dataset_sen2_l1c_s3(self):
        self._test_open_dataset_sen2_l1c(f"s3://{s02msil1c_bucket}/{path_prefix}")

    def test_open_dataset_sen2_l2a_s3(self):
        self._test_open_dataset_sen2_l2a(f"s3://{s02msil2a_bucket}/{path_prefix}")

    def test_open_dataset_sen2_l1c_https(self):
        self._test_open_dataset_sen2_l1c(
            f"{DEFAULT_ENDPOINT_URL}/{s02msil1c_bucket}/{path_prefix}"
        )

    def test_open_dataset_sen2_l2a_https(self):
        self._test_open_dataset_sen2_l2a(
            f"{DEFAULT_ENDPOINT_URL}/{s02msil2a_bucket}/{path_prefix}"
        )

    def _test_open_dataset_sen2_l1c(self, url_prefix):
        # See https://stac.browser.user.eopf.eodc.eu/collections/sentinel-2-l1c/items/S2B_MSIL1C_20250415T142749_N0511_R139_T25WEV_20250415T180239
        url = f"{url_prefix}/{l1c_filename}"
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(
                url,
                engine="eopf-zarr",
                op_mode="analysis",
                chunks={},
            )
        self.assertTrue(result.time_delta < allowed_open_time)

        self.assertIn("b03", ds)
        self.assertIn("b11", ds)
        self.assertIn("b01", ds)

        spatial_vars = get_spatial_vars(ds.data_vars)
        assert_data_arrays_are_chunked(self, spatial_vars, verbose=show_chunking)
        for var_name in spatial_vars.keys():
            self.assertEqual((10980, 10980), ds[var_name].shape[-2:], msg=var_name)

    def _test_open_dataset_sen2_l2a(self, url_prefix):
        url = f"{url_prefix}/{l2a_filename}"
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(
                url,
                engine="eopf-zarr",
                op_mode="analysis",
                chunks={},
            )
        self.assertTrue(result.time_delta < allowed_open_time)

        self.assertIn("b03", ds)
        self.assertIn("b11", ds)
        self.assertIn("b01", ds)
        self.assertIn("scl", ds)
        self.assertIn("cld", ds)
        self.assertIn("snw", ds)

        spatial_vars = get_spatial_vars(ds.data_vars)
        assert_data_arrays_are_chunked(self, spatial_vars, verbose=show_chunking)
        for var_name in spatial_vars.keys():
            self.assertEqual((10980, 10980), ds[var_name].shape[-2:], msg=var_name)

    def test_production(self):
        url = (
            "https://objectstore.eodc.eu:2222/"
            "e05ab01a9d56408d82ac32d69a5aae2a:202504-s02msil2a/15/products/"
            "cpm_v256/"
            "S2B_MSIL2A_20250415T142749_N0511_R139_T25WEU_20250415T181516.zarr"
        )
        ds = xr.open_dataset(url, engine="eopf-zarr")
        self.assertIsInstance(ds, xr.Dataset)
