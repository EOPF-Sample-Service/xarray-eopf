#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import numpy as np
import xarray as xr

from xarray_eopf.backend import EopfBackend


class EopfBackendTest(TestCase):
    def test_is_installed(self):
        engines = xr.backends.list_engines()
        self.assertIn("eopf-zarr", engines)
        self.assertIsInstance(engines["eopf-zarr"], EopfBackend)

    def test_open_datatree(self):
        original_dt = make_s2_msi()
        original_dt.to_zarr("memory://S02MSIL1C.zarr", mode="w")
        data_tree = xr.open_datatree("memory://S02MSIL1C.zarr", engine="eopf-zarr")
        self.assertIn("r10m", data_tree)
        self.assertIn("r20m", data_tree)
        self.assertIn("r60m", data_tree)

    def test_open_dataset(self):
        original_ds = make_s2_msi_r60m()
        original_ds.to_zarr("memory://S02MSIL1C.zarr", mode="w")
        dataset = xr.open_dataset("memory://S02MSIL1C.zarr", engine="eopf-zarr")
        self.assertIn("b01", dataset)
        self.assertIn("b09", dataset)
        self.assertIn("b10", dataset)
        self.assertIn("x", dataset)
        self.assertIn("y", dataset)


def make_s2_msi() -> xr.DataTree:
    return xr.DataTree(
        children={
            "r10m": xr.DataTree(dataset=make_s2_msi_r10m()),
            "r20m": xr.DataTree(dataset=make_s2_msi_r20m()),
            "r60m": xr.DataTree(dataset=make_s2_msi_r60m()),
        }
    )


def make_s2_msi_r10m() -> xr.Dataset:
    return make_s2_msi_rx0m(["b02", "b03", "b04", "b08"], 48, 48)


def make_s2_msi_r20m() -> xr.Dataset:
    return make_s2_msi_rx0m(["b05", "b06", "b07", "b11", "b12", "b8a"], 24, 24)


def make_s2_msi_r60m() -> xr.Dataset:
    return make_s2_msi_rx0m(["b01", "b09", "b10"], 8, 8)


def make_s2_msi_rx0m(bands: list[str], w: int, h: int) -> xr.Dataset:
    x1, x2 = 0.0, 48.0
    dx = 0.5 * (x2 - x1) / w

    y1, y2 = 0.0, 48.0
    dy = 0.5 * (y2 - y1) / h

    return xr.Dataset(
        data_vars={
            band: xr.DataArray(
                np.random.randint(1, 2 << 15, (h, w)),
                dims=("y", "x"),
                attrs={"_FillValue": 0},
            )
            for band in bands
        },
        coords={
            "x": xr.DataArray(np.linspace(x1 + dx, y2 - dx, w), dims="x"),
            "y": xr.DataArray(np.linspace(y1 + dy, y2 - dy, h), dims="y"),
        },
    )
