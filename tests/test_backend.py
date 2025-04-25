#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from typing import Any
from unittest import TestCase

import fsspec
import pytest
import xarray as xr

from tests.helpers import make_s2_msi, make_s2_msi_l2a
from xarray_eopf.backend import EopfBackend


class EopfBackendTest(TestCase):
    def test_is_installed(self):
        engines = xr.backends.list_engines()
        self.assertIn("eopf-zarr", engines)
        self.assertIsInstance(engines["eopf-zarr"], EopfBackend)

    # noinspection PyTypeChecker,PyMethodMayBeStatic
    def test_mode_is_validated(self):
        with pytest.raises(
            ValueError,
            match="mode argument must be 'analysis' or 'native', was 'convenience'",
        ):
            xr.open_datatree(
                "memory://S02MSIL1C.zarr", engine="eopf-zarr", op_mode="convenience"
            )
        with pytest.raises(
            ValueError,
            match="op_mode argument must be 'analysis' or 'native', was 'sensor'",
        ):
            xr.open_dataset(
                "memory://S02MSIL1C.zarr", engine="eopf-zarr", op_mode="sensor"
            )

    def test_guess_can_open(self):
        backend = EopfBackend()
        self.assertFalse(backend.guess_can_open("data/test.zarr"))
        # noinspection PyTypeChecker
        self.assertFalse(backend.guess_can_open({}))


class NativeModeTest(TestCase):
    @classmethod
    def setUpClass(cls):
        original_dt = make_s2_msi()
        original_dt.to_zarr("memory://S02MSIL1C.zarr", mode="w")

    def test_open_datatree(self):
        # noinspection PyTypeChecker
        data_tree = xr.open_datatree(
            "memory://S02MSIL1C.zarr", engine="eopf-zarr", op_mode="native"
        )
        self.assertIn("r10m", data_tree)
        self.assertIn("r20m", data_tree)
        self.assertIn("r60m", data_tree)

    def test_open_datatree_subgroup(self):
        # noinspection PyTypeChecker
        data_tree = xr.open_datatree(
            "memory://S02MSIL1C.zarr/r10m", engine="eopf-zarr", op_mode="native"
        )
        self.assertEqual(
            ["b02", "b03", "b04", "b08"], sorted(data_tree.data_vars.keys())
        )
        # noinspection PyTypeChecker
        self.assertEqual(["x", "y"], sorted(data_tree.coords.keys()))

    def test_open_dataset(self):
        # noinspection PyTypeChecker
        dataset = xr.open_dataset(
            "memory://S02MSIL1C.zarr", engine="eopf-zarr", op_mode="native"
        )
        self.assertIn("r60m_b01", dataset)
        self.assertIn("r10m_b02", dataset)
        self.assertIn("r20m_b05", dataset)
        self.assertIn("r10m_x", dataset)
        self.assertIn("r10m_y", dataset)
        self.assertIn("r20m_x", dataset)
        self.assertIn("r20m_y", dataset)
        self.assertIn("r60m_x", dataset)
        self.assertIn("r60m_y", dataset)

    def test_open_dataset_subgroup(self):
        # noinspection PyTypeChecker
        dataset = xr.open_dataset(
            "memory://S02MSIL1C.zarr/r20m", engine="eopf-zarr", op_mode="native"
        )
        self.assertEqual(
            ["b05", "b06", "b07", "b11", "b12", "b8a"], sorted(dataset.data_vars.keys())
        )
        # noinspection PyTypeChecker
        self.assertEqual(["x", "y"], sorted(dataset.coords.keys()))


class AnalysisModeTest(TestCase):
    path = "memory://S2A_MSIL2A_X.zarr"

    @classmethod
    def setUpClass(cls):
        original_dt = make_s2_msi_l2a()
        original_dt.to_zarr(cls.path, mode="w")

    def test_open_dataset_ok(self):
        # noinspection PyTypeChecker
        dataset = xr.open_dataset(self.path, engine="eopf-zarr", op_mode="analysis")
        self.assert_dataset_ok(dataset)

        fs: fsspec.AbstractFileSystem = fsspec.filesystem("memory")
        store = fs.get_mapper(root=self.path)
        # noinspection PyTypeChecker
        dataset = xr.open_dataset(store, engine="eopf-zarr", op_mode="analysis")
        self.assert_dataset_ok(dataset)

    # noinspection PyMethodMayBeStatic
    def test_open_dataset_fail(self):
        with pytest.raises(FileNotFoundError):
            # noinspection PyTypeChecker
            xr.open_dataset("test.zarr", engine="eopf-zarr", op_mode="analysis")

    def assert_dataset_ok(self, dataset: Any):
        self.assertIsInstance(dataset, xr.Dataset)
        # Note, more detailed analysis is done in `tests/amodes`
        self.assertEqual(
            [
                "b01",
                "b02",
                "b03",
                "b04",
                "b05",
                "b06",
                "b07",
                "b08",
                "b11",
                "b12",
                "b8a",
                "cld",
                "scl",
                "snw",
            ],
            sorted(dataset.data_vars.keys()),
        )
        # noinspection PyTypeChecker
        self.assertEqual(["spatial_ref", "x", "y"], sorted(dataset.coords.keys()))

    def test_open_dataset_subgroup_ok(self):
        # noinspection PyTypeChecker
        dataset = xr.open_dataset(
            self.path + "/measurements/reflectance/r10m", engine="eopf-zarr"
        )
        self.assertIsInstance(dataset, xr.Dataset)
        self.assertEqual(["b02", "b03", "b04", "b08"], sorted(dataset.data_vars.keys()))
        # noinspection PyTypeChecker
        self.assertEqual(["spatial_ref", "x", "y"], sorted(dataset.coords.keys()))

    def test_open_datatree_ok(self):
        # noinspection PyTypeChecker
        dt = xr.open_datatree(self.path, engine="eopf-zarr", op_mode="analysis")
        self.assertIsInstance(dt, xr.DataTree)

        fs: fsspec.AbstractFileSystem = fsspec.filesystem("memory")
        store = fs.get_mapper(root=self.path)
        # noinspection PyTypeChecker
        dt = xr.open_datatree(store, engine="eopf-zarr", op_mode="analysis")
        self.assertIsInstance(dt, xr.DataTree)

    def test_open_datatree_subgroup_ok(self):
        # noinspection PyTypeChecker
        dt = xr.open_datatree(
            self.path + "/measurements/reflectance/r10m", engine="eopf-zarr"
        )
        self.assertIsInstance(dt, xr.DataTree)
