#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

from xarray_eopf.amode import AnalysisModeRegistry
from xarray_eopf.amodes.sentinel3 import register


class Sentinel3AnalysisModeTest(TestCase):
    def test_register(self):
        registry = AnalysisModeRegistry()
        register(registry)
        self.assertEqual(0, len(list(registry.keys())))
