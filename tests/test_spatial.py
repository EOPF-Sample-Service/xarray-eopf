#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Mapping
from typing import Hashable
from unittest import TestCase

import numpy as np
import xarray as xr

from tests.helpers import make_s2_msi
from xarray_eopf.flatten import flatten_datatree
from xarray_eopf.spatial import rescale_spatial_vars


class RescaleSpatialVarsTest(TestCase):
    ds: xr.Dataset

    @classmethod
    def setUpClass(cls):
        dt = make_s2_msi(r10m_size=48)
        cls.ds = flatten_datatree(dt)

    def test_s2_msi_to_10m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r10m_b02")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 13, 48)
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars)
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 13, 48)

        self.assertEqual(None, rescaled_vars["r10m_b02"].attrs.get("history"))
        self.assertEqual(
            (
                "Upscaling dimensions 'r10m_x' and 'r10m_y'"
                " by scale factor 0.5 using spline interpolation of order 3;\n"
            ),
            rescaled_vars["r20m_b05"].attrs.get("history"),
        )
        self.assertEqual(
            (
                "Upscaling dimensions 'r10m_x' and 'r10m_y'"
                " by scale factor 0.166667 using spline interpolation of order 3;\n"
            ),
            rescaled_vars["r60m_b01"].attrs.get("history"),
        )

    def test_s2_msi_to_20m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r20m_b05")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 13, 24)

        self.assertEqual(
            "Downscaling dimensions 'r20m_x' and 'r20m_y'"
            " by window size 2 using aggregation method 'mean';\n",
            rescaled_vars["r10m_b02"].attrs.get("history"),
        )
        self.assertEqual(
            None,
            rescaled_vars["r20m_b05"].attrs.get("history"),
        )
        self.assertEqual(
            (
                "Upscaling dimensions 'r20m_x' and 'r20m_y'"
                " by scale factor 0.333333 using spline interpolation of order 3;\n"
            ),
            rescaled_vars["r60m_b01"].attrs.get("history"),
        )

    def test_s2_msi_to_60m(self):
        rescaled_vars = rescale_spatial_vars(self.ds.data_vars, ref_var_name="r60m_b01")
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 13, 8)

        self.assertEqual(
            "Downscaling dimensions 'r60m_x' and 'r60m_y'"
            " by window size 6 using aggregation method 'mean';\n",
            rescaled_vars["r10m_b02"].attrs.get("history"),
        )
        self.assertEqual(
            "Downscaling dimensions 'r60m_x' and 'r60m_y'"
            " by window size 3 using aggregation method 'mean';\n",
            rescaled_vars["r20m_b05"].attrs.get("history"),
        )
        self.assertEqual(
            None,
            rescaled_vars["r60m_b01"].attrs.get("history"),
        )

    # noinspection PyMethodMayBeStatic
    def test_downscale_odd(self):
        rescaled_vars = rescale_spatial_vars(
            {
                "a": xr.DataArray(np.zeros((10, 10)), dims=["y", "x"]).chunk(
                    {"y": 5, "x": 5}
                ),
                "b": xr.DataArray(np.zeros((25, 25)), dims=["y", "x"]).chunk(
                    {"y": 5, "x": 5}
                ),
            },
            ref_var_name="a",
        )
        self.assert_rescale_spatial_vars_ok(rescaled_vars, 2, expected_size=10)

        self.assertEqual(
            "Upscaling dimensions 'x' and 'y'"
            " by scale factor 0.833333 using spline interpolation of order 3;\n"
            "Downscaling dimensions 'x' and 'y'"
            " by window size 3 using aggregation method 'mean';\n",
            rescaled_vars["b"].attrs.get("history"),
        )
        self.assertEqual(
            None,
            rescaled_vars["a"].attrs.get("history"),
        )

    def assert_rescale_spatial_vars_ok(
        self,
        rescaled_vars: Mapping[Hashable, xr.DataArray],
        expected_var_count: int,
        expected_size: int,
    ):
        self.assertIsInstance(rescaled_vars, dict)
        self.assertEqual(expected_var_count, len(rescaled_vars))
        # Force resampling
        for var_name, var in rescaled_vars.items():
            array = var.values
            self.assertEqual((expected_size, expected_size), array.shape[-2:])


class DaskTests(TestCase):
    def test_dask_coarsen(self):
        import dask.array as da

        def reduce(
            window: np.ndarray, axis: tuple[int, ...] | None = None
        ) -> np.ndarray:
            if axis is None:
                return np.array([[0]])

            print(f"window={window}, shape={window.shape}, axis={axis}")
            return np.min(window, axis=axis)

        array = da.from_array(np.arange(0, 36).reshape((6, 6)), chunks=(3, 3))
        coarsened_array = da.coarsen(reduce, array, axes={0: 2, 1: 2}, trim_excess=True)
        a: np.ndarray = coarsened_array.compute()

        self.assertEqual((3, 3), a.shape)
        self.assertEqual(
            [
                [0, 2, 4],
                [12, 14, 16],
                [24, 26, 28],
            ],
            a.tolist(),
        )
