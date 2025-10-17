#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Sequence
from unittest import TestCase

import xarray as xr

from integration.helpers import assert_dataset_is_chunked
from xarray_eopf.constants import DEFAULT_ENDPOINT_URL
from xarray_eopf.utils import timeit


allowed_open_time = 1000  # seconds
show_chunking = False


class Sentinel3AnalysisTest(TestCase):
    def test_open_dataset_sen3_olci_l1_efr(self):
        path = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202508-s03olcefr"
            "/19/products/cpm_v256/S3B_OL_1_EFR____20250819T074058_20250819T074358_"
            "20250819T092155_0179_110_106_3420_ESA_O_NR_004.zarr"
        )
        expected_vars = ["oa01_radiance", "oa02_radiance", "oa03_radiance"]
        expected_size = (5269, 5000)
        self._test_sen3(path, expected_vars, expected_size)

    def test_open_dataset_sen3_olci_l1_err(self):
        path = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202508-s03olclfr"
            "/22/products/cpm_v256/S3A_OL_2_LFR____20250822T063703_20250822T064003"
            "_20250822T084148_0179_129_291_1980_PS1_O_NR_003.zarr"
        )
        expected_vars = ["oa01_radiance", "oa02_radiance", "oa03_radiance"]
        expected_size = ()
        self._test_sen3(path, expected_vars, expected_size)

    def test_open_dataset_sen3_olci_l2_lfr(self):
        path = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202508-s03olclfr"
            "/22/products/cpm_v256/S3A_OL_2_LFR____20250822T063703_20250822T064003_"
            "20250822T084148_0179_129_291_1980_PS1_O_NR_003.zarr"
        )
        expected_vars = []
        expected_size = ()
        self._test_sen3(path, expected_vars, expected_size)

    def test_open_dataset_sen3_slstr_l1_rbt(self):
        path = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202510-s03slsrbt"
            "-global/16/products/cpm_v256/S3B_SL_1_RBT____20251016T072510_"
            "20251016T072810_20251016T092049_0179_112_163_2700_ESA_O_NR_004.zarr"
        )
        expected_vars = []
        expected_size = ()
        self._test_sen3(path, expected_vars, expected_size)

    def test_open_dataset_sen3_slstr_l2_lst(self):
        path = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202510-s03slslst"
            "-eu/16/products/cpm_v256/S3B_SL_2_LST____20251016T215803_20251016T220103"
            "_20251017T004323_0179_112_172_0540_ESA_O_NR_004.zarr"
        )
        expected_vars = ["lst"]
        expected_size = ()
        self._test_sen3(path, expected_vars, expected_size)

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
