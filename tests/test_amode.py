#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
import fsspec
import zarr.storage
from pathlib import Path

import pytest
from typing import Any, Iterable
from unittest import TestCase

import xarray as xr

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.amodes.sentinel2 import MSIL1C, MSIL2A


class TestMode(AnalysisMode):
    product_type = "TEST"

    def is_valid_source(self, source: Any) -> bool:
        return isinstance(source, str) and "TEST" in source

    def get_applicable_params(self, **kwargs) -> dict[str, any]:
        return {}

    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        return datatree

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        **params,
    ) -> xr.Dataset:
        return datatree.dataset


class AnalysisModeTest(TestCase):
    def setUp(self):
        AnalysisMode.registry.register(TestMode)

    def tearDown(self):
        AnalysisMode.registry.unregister(TestMode)

    def test_guess_ok(self):
        self.assertIsInstance(AnalysisMode.guess("TEST.zarr"), TestMode)
        self.assertIsInstance(AnalysisMode.guess({}, product_type="TEST"), TestMode)
        self.assertIsInstance(
            AnalysisMode.guess("TEST.zarr", product_type="REST"), TestMode
        )

    # noinspection PyMethodMayBeStatic
    def test_guess_fail(self):
        with pytest.raises(
            ValueError, match="Unable to detect analysis mode for input"
        ):
            _mode = AnalysisMode.guess("REST.zarr")

        with pytest.raises(
            ValueError, match="Unable to detect analysis mode for input"
        ):
            _mode = AnalysisMode.guess({}, product_type="REST")

    def test_from_source(self):
        self.assertIsInstance(AnalysisMode.from_source("TEST.zarr"), TestMode)
        self.assertIsNone(AnalysisMode.from_source("REST.zarr"))
        self.assertIsNone(AnalysisMode.from_source({}))

    def test_from_product_type(self):
        self.assertIsInstance(AnalysisMode.from_product_type("TEST"), TestMode)
        self.assertIsNone(AnalysisMode.from_product_type("REST"))

    def test_source_to_path(self):
        # From str
        self.assertEqual("test1.zarr", AnalysisMode._source_to_path("test1.zarr"))

        # From pathlib.Path
        self.assertEqual("test2.zarr", AnalysisMode._source_to_path(Path("test2.zarr")))

        # From fsspec.FSMap
        path = AnalysisMode._source_to_path(
            fsspec.filesystem("local").get_mapper("test3.zarr")
        )
        self.assertIsInstance(path, str)
        self.assertEqual("test3.zarr", Path(path).name)

        # From zarr.storage.DirectoryStore
        path = AnalysisMode._source_to_path(zarr.storage.DirectoryStore("test4.zarr"))
        self.assertIsInstance(path, str)
        self.assertEqual("test4.zarr", Path(path).name)

        # From dict
        self.assertEqual(None, AnalysisMode._source_to_path({"path": "test5.zarr"}))


class AnalysisModeRegistryTest(TestCase):
    # noinspection PyMethodMayBeStatic
    def get(self):
        reg = AnalysisModeRegistry()
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

    def test_register_unregister(self):
        reg = self.get()
        reg.register(TestMode)
        self.assertIsInstance(reg.get("TEST"), TestMode)
        reg.unregister(TestMode)
        self.assertIsNone(reg.get("TEST"))
