#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr

from integration.helpers import assert_dataset_is_chunked
from xarray_eopf.utils import timeit

allowed_open_time = 1000  # seconds
show_chunking = False


class Sentinel2AnalysisTest(TestCase):
    def test_open_dataset_sen2_l1c(self):
        self._test_open_dataset_sen2_l1c(
            "https://objects.eodc.eu:443/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s02msil1c-eu/13/products/cpm_v262/S2A_MSIL1C_20260313T101741_N0512_"
            "R065_T32TLQ_20260313T153853.zarr"
        )

    def test_open_dataset_sen2_l2a(self):
        self._test_open_dataset_sen2_l2a(
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s02msil2a-eu/18/products/cpm_v262/S2A_MSIL2A_20260318T125321_"
            "N0512_R138_T28WDT_20260318T204314.zarr"
        )

    def _test_open_dataset_sen2_l1c(self, url):
        # See https://stac.browser.user.eopf.eodc.eu/collections/sentinel-2-l1c/items/S2B_MSIL1C_20250415T142749_N0511_R139_T25WEV_20250415T180239
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

        assert_dataset_is_chunked(self, ds, verbose=show_chunking)
        for var_name in ds.data_vars:
            self.assertEqual((10980, 10980), ds[var_name].shape[-2:], msg=var_name)

    def _test_open_dataset_sen2_l2a(self, url):
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

        assert_dataset_is_chunked(self, ds, verbose=show_chunking)
        for var_name in ds.data_vars:
            self.assertEqual((10980, 10980), ds[var_name].shape[-2:], msg=var_name)
