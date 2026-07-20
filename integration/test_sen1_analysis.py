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


class Sentinel1AnalysisTest(TestCase):
    def test_open_dataset_sen1_grd(self):
        dem_path = Path(__file__).resolve().parent / "test_data" / "dem_grd.zarr.zip"
        store = zarr.ZipStore(str(dem_path), mode="r")
        dem = xr.open_zarr(store, group="dem_small.zarr", consolidated=False, chunks={})
        dem = dem.dem

        url = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202606"
            "-s01siwgrh-global/23/products/cpm_v270/S1D_IW_GRDH_1SDV_20260623"
            "T225558_20260623T225623_003369_005EC5_B4C2.zarr"
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

        self.assertIn("gamma0_vv", ds)
        self.assertIn("gamma0_vh", ds)

        assert_dataset_is_chunked(self, ds, verbose=show_chunking)
        for var_name in ds.data_vars:
            self.assertEqual((541, 1081), ds[var_name].shape, msg=var_name)

    def test_open_dataset_sen1_slc(self):
        dem_path = Path(__file__).resolve().parent / "test_data" / "dem_slc.zarr.zip"
        store = zarr.ZipStore(str(dem_path), mode="r")
        dem = xr.open_zarr(store, group="dem_small.zarr", consolidated=False, chunks={})
        dem = dem.dem

        url = (
            "https://data.eodc.eu/collections/EOPF_ZARR/products/cpm_v270/"
            "S01SIWSLC/2026/05/31/S1D_IW_SLC__1SDV_20260531T171503_"
            "20260531T171530_003031_00539E_C573.zarr"
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

        self.assertIn("gamma0_vv", ds)
        self.assertIn("gamma0_vh", ds)

        assert_dataset_is_chunked(self, ds, verbose=show_chunking)
        for var_name in ds.data_vars:
            self.assertEqual((361, 361), ds[var_name].shape, msg=var_name)

    def test_open_datatree_sen1_onc(self):
        url = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202507-s01siwocn"
            "/31/products/cpm_v256/S1A_IW_OCN__2SDV_20250731T213433_20250731T213458_"
            "060333_077FA7_8163.zarr"
        )
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(
                url,
                engine="eopf-zarr",
                op_mode="analysis",
                chunks={},
            )
        self.assertTrue(result.time_delta < allowed_open_time)

        self.assertIn("wind_direction", ds)
        self.assertIn("wind_speed", ds)
        self.assertIn("inversion_quality", ds)
        self.assertIn("wind_quality", ds)
        self.assertIn("percentage_bright_points", ds)

        assert_dataset_is_chunked(self, ds, verbose=show_chunking)
        for var_name in ds.data_vars:
            self.assertEqual((222, 290), ds[var_name].shape, msg=var_name)
