#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr


class Sentinel1Test(TestCase):

    def test_open_datatree_sen2_l1c(self):
        path = (
            "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:"
            "sample-data/tutorial_data/cpm_v253/S2B_MSIL1C_20250113T103309_N0511_"
            "R108_T32TLQ_20250113T122458.zarr"
        )
        dt = xr.open_datatree(path, engine="eopf-zarr")
        self.assertEqual(25, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r60m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r60m
        self.assertEqual({"y": 1830, "x": 1830}, ds.sizes)

    def test_open_datatree_sen2_l2a(self):
        path = (
            "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:"
            "sample-data/tutorial_data/cpm_v253/S2A_MSIL2A_20240101T102431_N0510_"
            "R065_T32TNT_20240101T144052.zarr"
        )
        dt = xr.open_datatree(path, engine="eopf-zarr")
        self.assertEqual(36, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r60m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r60m
        self.assertEqual({"y": 1830, "x": 1830}, ds.sizes)
