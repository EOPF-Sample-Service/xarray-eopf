#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Sequence
from unittest import TestCase

import xarray as xr

from integration.helpers import assert_dataset_is_chunked
from xarray_eopf.utils import timeit

allowed_open_time = 1000  # seconds
show_chunking = False

ol1efr_url = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-s03olcefr-eu/19/"
    "products/cpm_v262/S3A_OL_1_EFR____20260319T094323_20260319T094623_20260319T1"
    "14002_0180_137_193_2160_PS1_O_NR_004.zarr"
)
ol1err_url = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-s03olcerr-eu/"
    "18/products/cpm_v262/S3A_OL_1_ERR____20260318T114146_20260318T122543_202603"
    "18T134744_2637_137_180______PS1_O_NR_004.zarr"
)
ol2lfr_url = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-s03olclfr-eu/19/"
    "products/cpm_v262/S3B_OL_2_LFR____20260319T090741_20260319T091041_20260319T1121"
    "42_0179_118_050_2340_ESA_O_NR_003.zarr"
)
sl1rbt_url = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-s03slsrbt-eu/19/"
    "products/cpm_v262/S3A_SL_1_RBT____20260319T094623_20260319T094923_20260319T1"
    "15906_0179_137_193_2340_PS1_O_NR_004.zarr"
)
sl2lst_url = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-s03slslst-eu/19/"
    "products/cpm_v262/S3A_SL_2_LST____20260319T094623_20260319T094923_20260319T1"
    "20128_0179_137_193_2340_PS1_O_NR_004.zarr"
)


class Sentinel3AnalysisTest(TestCase):
    def test_open_dataset_sen3_olci_l1_efr(self):
        expected_vars = ["oa01_radiance", "oa02_radiance", "oa03_radiance"]
        expected_size = (4792, 5158)
        self._test_sen3(ol1efr_url, expected_vars, expected_size)

    def test_open_dataset_sen3_olci_l1_err(self):
        expected_vars = ["oa01_radiance", "oa02_radiance", "oa03_radiance"]
        expected_size = (14492, 10701)
        self._test_sen3(ol1err_url, expected_vars, expected_size)

    def test_open_dataset_sen3_olci_l2_lfr(self):
        expected_vars = ["gifapar", "iwv", "otci"]
        expected_size = (4790, 5126)
        self._test_sen3(ol2lfr_url, expected_vars, expected_size)

    def test_open_dataset_sen3_slstr_l1_rbt(self):
        expected_vars = ["s1_radiance_an", "s7_bt_in", "s7_bt_io"]
        expected_size = (2948, 3343)
        self._test_sen3(sl1rbt_url, expected_vars, expected_size)

    def test_open_dataset_sen3_slstr_l2_lst(self):
        expected_vars = ["lst"]
        expected_size = (1474, 1667)
        self._test_sen3(sl2lst_url, expected_vars, expected_size)

    def _test_sen3(
        self, path: str, expected_vars: Sequence[str], expected_size: tuple[int, int]
    ):
        with timeit("open " + path) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(
                path,
                engine="eopf-zarr",
                chunks={},
            )
        self.assertTrue(result.time_delta < allowed_open_time)

        for expected_var in expected_vars:
            self.assertIn(expected_var, ds)

        assert_dataset_is_chunked(self, ds, verbose=show_chunking)
        for var_name in ds.data_vars:
            self.assertEqual(expected_size, ds[var_name].shape[-2:], msg=var_name)
