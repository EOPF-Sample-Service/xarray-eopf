#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

from xarray_eopf.prodtype import ProductTypeRegistry
from xarray_eopf.prodtypes.sentinel2 import MSIL1C, MSIL2A


class ProductTypeRegistryTest(TestCase):
    def get(self):
        reg = ProductTypeRegistry()
        reg.register(MSIL1C)
        reg.register(MSIL2A)
        return reg

    def test_get(self):
        reg = self.get()
        self.assertIsInstance(reg.get("MSIL1C"), MSIL1C)
        self.assertIsInstance(reg.get("MSIL2A"), MSIL2A)
        self.assertIs(None, reg.get("MSIL2B"))

    def test_keys_and_values(self):
        reg = self.get()
        self.assertEqual(["MSIL1C", "MSIL2A"], list(reg.keys()))
        values = list(reg.values())
        self.assertEqual(2, len(values))
        self.assertIsInstance(values[0], MSIL1C)
        self.assertIsInstance(values[1], MSIL2A)
