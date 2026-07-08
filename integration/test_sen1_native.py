#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr


class Sentinel1NativeTest(TestCase):
    def test_open_datatree_sen1_grd(self):
        path = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202606"
            "-s01siwgrh-global/23/products/cpm_v270/S1D_IW_GRDH_1SDV_20260623"
            "T225558_20260623T225623_003369_005EC5_B4C2.zarr"
        )
        # noinspection PyTypeChecker
        dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native")
        self.assertEqual(25, len(dt.groups))
        self.assertIn(
            "/S01SIWGRD_20260623T225558_0025_D019_B4C2_005EC5_VH/measurements",
            dt.groups,
        )
        ds = dt.S01SIWGRD_20260623T225558_0025_D019_B4C2_005EC5_VH.measurements
        self.assertEqual({"azimuth_time": 16802, "ground_range": 25319}, ds.sizes)

    def test_open_datatree_sen1_slc(self):
        path = (
            "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:"
            "sample-data/tutorial_data/cpm_v253/S1A_IW_SLC__1SDV_20240205T051225"
            "_20240205T051253_052419_0656D8_454B.zarr"
        )
        # noinspection PyTypeChecker
        dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native")
        self.assertEqual(953, len(dt.groups))
        self.assertIn(
            "/S01SIWSLC_20240205T051225_0028_A300_454B_0656D8_VH_IW3_459410/measurements",
            dt.groups,
        )
        ds = (
            dt.S01SIWSLC_20240205T051225_0028_A300_454B_0656D8_VH_IW3_459410.measurements
        )
        self.assertEqual({"azimuth_time": 1525, "slant_range_time": 25710}, ds.sizes)

    def test_open_datatree_sen1_onc(self):
        path = (
            "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202507-s01siwocn"
            "/31/products/cpm_v256/S1A_IW_OCN__2SDV_20250731T213433_20250731T213458_"
            "060333_077FA7_8163.zarr"
        )
        # noinspection PyTypeChecker
        dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertEqual(16, len(dt.groups))
        self.assertIn(
            "/owi/S01SIWOCN_20250731T213433_0025_A345_8163_077FA7_VV/measurements",
            dt.groups,
        )
        ds = dt.owi.S01SIWOCN_20250731T213433_0025_A345_8163_077FA7_VV.measurements
        self.assertEqual({"azimuth": 168, "range": 255}, ds.sizes)
