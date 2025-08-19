#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr

from integration.helpers import assert_dataset_is_chunked
from xarray_eopf.constants import DEFAULT_ENDPOINT_URL
from xarray_eopf.utils import timeit


s03ol1efr_bucket = "e05ab01a9d56408d82ac32d69a5aae2a:202508-s03olcefr"
s03ol1err_bucket = "e05ab01a9d56408d82ac32d69a5aae2a:202508-s03olcerr"
s03ol2lfr_bucket = "e05ab01a9d56408d82ac32d69a5aae2a:202508-s03olclfr"
s03sl1rbt_bucket = "e05ab01a9d56408d82ac32d69a5aae2a:202508-s03slsrbt"
s03sl2lst_bucket = "e05ab01a9d56408d82ac32d69a5aae2a:202508-s03slslst"
path_prefix = "19/products/cpm_v256"
ol1efr_filename = (
    "S3B_OL_1_EFR____20250819T074058_20250819T074358_20250819T092155_"
    "0179_110_106_3420_ESA_O_NR_004.zarr"
)
ol1err_filename = (
    "S3A_OL_1_ERR____20250819T092632_20250819T101045_20250819T113731_"
    "2653_129_250______PS1_O_NR_004.zarr"
)
ol2lfr_filename = (
    "S3A_OL_2_LFR____20250819T093936_20250819T094236_20250819T114257_"
    "0180_129_250_2160_PS1_O_NR_003.zarr"
)
sl1rbt_filename = (
    "S3B_SL_1_RBT____20250819T104457_20250819T104757_20250819T124948_"
    "0180_110_108_2340_ESA_O_NR_004.zarr"
)
sl2lst_filenmae = (
    "S3B_SL_2_LST____20250819T104757_20250819T105057_20250819T130331_"
    "0179_110_108_2520_ESA_O_NR_004.zarr"
)


allowed_open_time = 5  # seconds


class Sentinel2NativeTest(TestCase):
    def test_open_datatree_sen3_ol1efr(self):
        url = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202508-s03olcefr/"
            "19/products/cpm_v256/S3B_OL_1_EFR____20250819T074058_20250819T074358_"
            "20250819T092155_0179_110_106_3420_ESA_O_NR_004.zarr"
        )
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            dt = xr.open_datatree(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time_delta < allowed_open_time)
        self.assertEqual(11, len(dt.groups))
        self.assertIn(
            "/measurements",
            dt.groups,
        )
        ds = dt.measurements
        self.assertEqual({"columns": 4865, "rows": 4091}, ds.sizes)
        self.assertEqual(21, len(ds.data_vars))
        assert_dataset_is_chunked(self, ds, verbose=True)
