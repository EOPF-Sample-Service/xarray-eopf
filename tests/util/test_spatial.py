#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

from tests.helpers import make_s2_msi
from xarray_eopf.util.flatten import flatten_datatree

from xarray_eopf.util.spatial import rescale_spatial_vars


class SpatialTest(TestCase):
    def test_rescale_spatial_vars(self):
        dt = make_s2_msi()
        ds = flatten_datatree(dt)
        rescaled_vars = rescale_spatial_vars(ds.data_vars)
        self.assertIsInstance(rescaled_vars, dict)
        self.assertEqual(len(ds.data_vars), len(rescaled_vars))
