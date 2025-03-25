#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

from tests.helpers import make_s2_msi
from xarray_eopf.flatten import flatten_datatree


class FlattenTest(TestCase):
    def test_flatten_datatree(self):
        datatree = make_s2_msi()
        dataset = flatten_datatree(datatree)

        self.assertEqual(
            [
                "r10m_b02",
                "r10m_b03",
                "r10m_b04",
                "r10m_b08",
                "r20m_b05",
                "r20m_b06",
                "r20m_b07",
                "r20m_b11",
                "r20m_b12",
                "r20m_b8a",
                "r60m_b01",
                "r60m_b09",
                "r60m_b10",
            ],
            sorted(dataset.data_vars.keys()),
        )

        # noinspection PyTypeChecker
        self.assertEqual(
            ["r10m_x", "r10m_y", "r20m_x", "r20m_y", "r60m_x", "r60m_y"],
            sorted(dataset.coords.keys()),
        )
