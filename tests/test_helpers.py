#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr

from tests.helpers import make_s2_msi_l1c, make_s2_msi_l2a


class HelpersTest(TestCase):
    def test_make_s2_msi_l1c(self):
        dt = make_s2_msi_l1c()
        self.assertIsInstance(dt, xr.DataTree)
        self.assertEqual(["measurements"], sorted(dt.children.keys()))
        self.assertEqual(
            ["reflectance"],
            sorted(dt["measurements"].children.keys()),
        )
        self.assertEqual(
            ["r10m", "r20m", "r60m"],
            sorted(dt["measurements"]["reflectance"].children.keys()),
        )
        self.assertEqual(
            [],
            sorted(dt["measurements"]["reflectance"]["r10m"].children.keys()),
        )
        self.assertEqual(
            ["b02", "b03", "b04", "b08"],
            sorted(dt["measurements"]["reflectance"]["r10m"].ds.keys()),
        )

    def test_make_s2_msi_l2a(self):
        dt = make_s2_msi_l2a()
        self.assertIsInstance(dt, xr.DataTree)
        self.assertEqual(
            ["conditions", "measurements", "quality"], sorted(dt.children.keys())
        )
        self.assertEqual(
            ["reflectance"],
            sorted(dt["measurements"].children.keys()),
        )
        self.assertEqual(
            ["r10m", "r20m", "r60m"],
            sorted(dt["measurements"]["reflectance"].children.keys()),
        )
        self.assertEqual(
            [],
            sorted(dt["measurements"]["reflectance"]["r10m"].children.keys()),
        )
        self.assertEqual(
            ["b02", "b03", "b04", "b08"],
            sorted(dt["measurements"]["reflectance"]["r10m"].ds.keys()),
        )
