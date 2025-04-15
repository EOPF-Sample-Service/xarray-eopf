#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

from tests.helpers import make_s2_msi
from xarray_eopf.filter import filter_dataset
from xarray_eopf.flatten import flatten_datatree


class FilterDatasetTest(TestCase):
    def setUp(self):
        datatree = make_s2_msi()
        self.dataset = flatten_datatree(datatree)

    def test_filter_dataset_by_name(self):
        dataset = filter_dataset(self.dataset, variables=["r20m_b06", "r60m_b09"])
        self.assertEqual(
            [
                "r20m_b06",
                "r60m_b09",
            ],
            sorted(dataset.data_vars.keys()),
        )

    def test_filter_dataset_by_prefix(self):
        dataset = filter_dataset(self.dataset, variables="r10m")
        self.assertEqual(
            [
                "r10m_b02",
                "r10m_b03",
                "r10m_b04",
                "r10m_b08",
            ],
            sorted(dataset.data_vars.keys()),
        )

    def test_filter_dataset_by_pattern(self):
        dataset = filter_dataset(self.dataset, variables=["r6.*"])
        self.assertEqual(
            [
                "r60m_b01",
                "r60m_b09",
                "r60m_b10",
            ],
            sorted(dataset.data_vars.keys()),
        )

    def test_filter_dataset_by_none(self):
        dataset = filter_dataset(self.dataset, variables=())
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
