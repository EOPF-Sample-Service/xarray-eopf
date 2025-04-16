#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr

# TODO: adjust path to new locations
bucket = "e05ab01a9d56408d82ac32d69a5aae2a:sample-data"
path_prefix = "tutorial_data/cpm_v253"
url_prefix = f"s3://{bucket}/{path_prefix}"


class Sentinel1NativeTest(TestCase):
    def test_open_datatree_sen1_grd(self):
        path = (
            f"{url_prefix}/"
            "S1A_IW_GRDH_1SDV_20240201T164915_20240201T164940_052368_065517_750E.zarr"
        )
        # noinspection PyTypeChecker
        dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native")
        self.assertEqual(33, len(dt.groups))
        self.assertIn(
            "/S01SIWGRD_20240201T164915_0025_A299_750E_065517_VH/measurements",
            dt.groups,
        )
        ds = dt.S01SIWGRD_20240201T164915_0025_A299_750E_065517_VH.measurements
        self.assertEqual({"azimuth_time": 16675, "ground_range": 26456}, ds.sizes)

    def test_open_datatree_sen1_slc(self):
        path = (
            f"{url_prefix}/"
            "S1A_IW_SLC__1SDV_20231119T170635_20231119T170702_051289_063021_178F.zarr"
        )
        # noinspection PyTypeChecker
        dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native")
        self.assertEqual(919, len(dt.groups))
        self.assertIn(
            "/S01SIWSLC_20231119T170635_0027_A293_178F_063021_VH_IW1_249411/measurements",
            dt.groups,
        )
        ds = dt.S01SIWSLC_20231119T170635_0027_A293_178F_063021_VH_IW1_249411.measurements
        self.assertEqual({"azimuth_time": 1501, "slant_range_time": 22694}, ds.sizes)

    def test_open_datatree_sen1_onc(self):
        path = (
            f"{url_prefix}/"
            "S1A_IW_OCN__2SDV_20250224T054940_20250224T055005_058034_072A26_160E.zarr"
        )
        # noinspection PyTypeChecker
        dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertEqual(16, len(dt.groups))
        self.assertIn(
            "/owi/S01SIWOCN_20250224T054940_0025_A332_160E_072A26_VV/measurements",
            dt.groups,
        )
        ds = dt.owi.S01SIWOCN_20250224T054940_0025_A332_160E_072A26_VV.measurements
        self.assertEqual({"azimuth": 166, "range": 264}, ds.sizes)
