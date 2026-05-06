#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from pathlib import Path
from unittest import TestCase

import xarray as xr
import zarr

from integration.helpers import assert_dataset_is_chunked
from xarray_eopf.utils import timeit

allowed_open_time = 1000  # seconds
show_chunking = False


class Sentinel2AnalysisTest(TestCase):
    def test_open_dataset_sen1_grd(self):
        dem_path = Path(__file__).resolve().parent / "test_data" / "dem_small.zarr.zip"
        store = zarr.ZipStore(str(dem_path), mode="r")
        dem = xr.open_zarr(
            store,
            group="dem_small.zarr",
            consolidated=False,  # also important
            chunks={},
        )
        dem = dem.dem

        url = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202603-"
            "s01siwgrh-global/19/products/cpm_v262/S1A_IW_GRDH_1SDV_20260319"
            "T102725_20260319T102758_063695_0801D3_2EC6.zarr"
        )
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(
                url,
                engine="eopf-zarr",
                op_mode="analysis",
                dem=dem,
                chunks={},
            )
        self.assertTrue(result.time_delta < allowed_open_time)

        self.assertIn("vv", ds)
        self.assertIn("vh", ds)

        assert_dataset_is_chunked(self, ds, verbose=show_chunking)
        for var_name in ds.data_vars:
            self.assertEqual((541, 1081), ds[var_name].shape, msg=var_name)
