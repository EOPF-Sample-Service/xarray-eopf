#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import dask.array as da
import numpy as np

import xarray_eopf.coarsen as xec


def get_input(dtype: np.dtype) -> da.Array:
    return da.array(
        [
            [0, 1, 2, 3, 4, 0],
            [1, 2, 3, 4, 0, 1],
            [2, 3, 4, 0, 1, 2],
            [3, 4, 0, 1, 2, 3],
            [4, 0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 0],
        ],
        dtype=dtype,
    )


class CoarsenTestMixin:
    dtype: np.dtype

    # noinspection PyUnresolvedReferences
    def assert_method_ok(self, reducer, expected: np.ndarray):
        input = get_input(dtype=self.dtype)
        # noinspection PyTypeChecker
        actual = da.coarsen(
            reducer,
            input,
            {0: 3, 1: 3},
        )
        self.assertIsInstance(actual, da.Array)
        if np.issubdtype(self.dtype, np.integer):
            np.testing.assert_array_equal(
                actual.compute(),
                expected,
            )
        else:
            np.testing.assert_array_almost_equal(
                actual.compute(),
                expected,
            )
        self.assertEqual(actual.dtype, expected.dtype)


class CoarsenIntegerArrayTest(CoarsenTestMixin, TestCase):
    dtype = np.int8

    def test_center(self):
        self.assert_method_ok(
            xec.center,
            np.array([[2, 0], [0, 3]], dtype=np.int8),
        )

    def test_count(self):
        self.assert_method_ok(
            np.count_nonzero,
            np.array([[8, 6], [6, 8]], dtype=np.int64),
        )

    def test_first(self):
        self.assert_method_ok(
            xec.first,
            np.array([[0, 3], [3, 1]], dtype=np.int8),
        )

    def test_last(self):
        self.assert_method_ok(
            xec.last,
            np.array([[4, 2], [2, 0]], dtype=np.int8),
        )

    def test_max(self):
        self.assert_method_ok(
            np.nanmax,
            np.array([[4, 4], [4, 4]], dtype=np.int8),
        )

    def test_min(self):
        self.assert_method_ok(
            np.nanmin,
            np.array([[0, 0], [0, 0]], dtype=np.int8),
        )

    def test_mean(self):
        self.assert_method_ok(
            xec.mean,
            np.array([[2, 2], [2, 2]], dtype=np.int8),
        )

    def test_median(self):
        self.assert_method_ok(
            xec.median,
            np.array(
                [[2, 1], [1, 3]],
                dtype=np.int8,
            ),
        )

    def test_mode(self):
        self.assert_method_ok(
            xec.mode,
            np.array([[2, 0], [0, 3]], dtype=np.int8),
        )

    def test_std(self):
        self.assert_method_ok(
            xec.std,
            np.array(
                [[1, 2], [2, 1]],
                dtype=np.int8,
            ),
        )

    def test_sum(self):
        self.assert_method_ok(
            xec.sum,
            np.array(
                [[18, 15], [15, 22]],
                dtype=np.int8,
            ),
        )

    def test_var(self):
        self.assert_method_ok(
            xec.var,
            np.array(
                [[1, 2], [2, 2]],
                dtype=np.int8,
            ),
        )


class CoarsenFloatingArrayTest(CoarsenTestMixin, TestCase):
    dtype = np.float32

    def test_mean(self):
        self.assert_method_ok(
            xec.mean,
            np.array(
                [
                    [2.0, 1.666667],
                    [1.666667, 2.444444],
                ],
                dtype=np.float32,
            ),
        )

    def test_median(self):
        self.assert_method_ok(
            xec.median,
            np.array(
                [[2.0, 1.0], [1.0, 3.0]],
                dtype=np.float32,
            ),
        )

    def test_std(self):
        self.assert_method_ok(
            xec.std,
            np.array(
                [[1.154701, 1.563472], [1.563472, 1.257079]],
                dtype=np.float32,
            ),
        )

    def test_sum(self):
        self.assert_method_ok(
            xec.sum,
            np.array(
                [[18.0, 15.0], [15.0, 22.0]],
                dtype=np.float32,
            ),
        )

    def test_var(self):
        self.assert_method_ok(
            xec.var,
            np.array(
                [[1.333333, 2.444445], [2.444445, 1.580247]],
                dtype=np.float32,
            ),
        )
