#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import pytest
import xarray as xr

from tests.helpers import make_s2_msi
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

    def test_open_datatree(self):
        original_dt = make_s2_msi()
        original_dt.to_zarr("memory://S02MSIL1C.zarr", mode="w")
        # noinspection PyTypeChecker
        data_tree = xr.open_datatree(
            "memory://S02MSIL1C.zarr", engine="eopf-zarr", op_mode="native"
        )
        self.assertIn("r10m", data_tree)
        self.assertIn("r20m", data_tree)
        self.assertIn("r60m", data_tree)


class EopfBackendNativeTest(TestCase):
    def test_open_dataset(self):
        original_ds = make_s2_msi()
        original_ds.to_zarr("memory://S02MSIL1C.zarr", mode="w")
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

    # TODO: add tests for open_datatree


# TODO: add EopfBackendAnalysisTest
