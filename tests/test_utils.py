#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import numpy as np
import pytest
import xarray as xr

from tests.helpers import make_s2_msi
from xarray_eopf.utils import (
    NameFilter,
    NameTypeMapping,
    assert_arg_is_instance,
    assert_arg_is_one_of,
    get_data_tree_item,
    timeit,
)


class AssertionTest(TestCase):
    def test_assert_arg_is_one_of(self):
        self.assertIsNone(assert_arg_is_one_of(2, "order", (0, 1, 2, 3)))

        with pytest.raises(
            ValueError, match="order argument must be 0, 1, 2 or 3, was 4"
        ):
            self.assertIsNone(assert_arg_is_one_of(4, "order", (0, 1, 2, 3)))

    def test_assert_arg_is_instance(self):
        self.assertIsNone(assert_arg_is_instance(2, "order", int))
        self.assertIsNone(assert_arg_is_instance(2, "order", (int, float)))

        with pytest.raises(
            TypeError, match="order argument must have type int, was float"
        ):
            self.assertIsNone(assert_arg_is_instance(2.0, "order", int))
        with pytest.raises(
            TypeError, match="order argument must have type int or float, was str"
        ):
            self.assertIsNone(assert_arg_is_instance("4", "order", (int, float)))


class TimeitTest(TestCase):
    def test_assert_arg_is_one_of(self):
        with timeit("test", silent=False) as result:
            pass
        self.assertTrue(result.label == "test")
        self.assertTrue(result.silent is False)
        self.assertTrue(result.start_time > 0)
        self.assertTrue(result.time_delta >= 0)


class GetDataTreeItemTest(TestCase):
    def test_with_pathname(self):
        dt = make_s2_msi()
        self.assertIsInstance(get_data_tree_item(dt, "r10m"), xr.DataTree)
        self.assertIsInstance(get_data_tree_item(dt, "r10m/b02"), xr.DataArray)

    def test_with_path(self):
        dt = make_s2_msi()
        self.assertIsInstance(get_data_tree_item(dt, ("r10m",)), xr.DataTree)
        self.assertIsInstance(get_data_tree_item(dt, ("r10m", "b02")), xr.DataArray)

    def test_not_found(self):
        dt = make_s2_msi()
        self.assertIsNone(get_data_tree_item(dt, "test"))


class NameFilterTest(TestCase):
    def test_accept_name(self):
        f = NameFilter(includes=("ernie", "bert"))
        self.assertTrue(f.accept("ernie"))
        self.assertTrue(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

        f = NameFilter(includes=("ernie", "bert"), excludes="bert")
        self.assertTrue(f.accept("ernie"))
        self.assertFalse(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

    def test_accept_prefix(self):
        f = NameFilter(includes=("er", "be"))
        self.assertTrue(f.accept("ernie"))
        self.assertTrue(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

        f = NameFilter(includes=("er", "be"), excludes="be")
        self.assertTrue(f.accept("ernie"))
        self.assertFalse(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

    def test_accept_pattern(self):
        f = NameFilter(includes="e.*e")
        self.assertTrue(f.accept("ernie"))
        self.assertFalse(f.accept("erno"))
        self.assertFalse(f.accept("bert"))
        self.assertFalse(f.accept("bibo"))

    def test_filter(self):
        f = NameFilter(includes="e.*e")
        self.assertEqual(
            ["ernie", "emmie"], list(f.filter(["bibo", "ernie", "bert", "emmie"]))
        )


class NameTypeMappingTest(TestCase):
    def test_new_with_data_is_none(self):
        ntm = NameTypeMapping.new("spline_orders", 3, None)
        self.assertEqual(3, ntm.get("scl", np.integer))
        self.assertEqual(3, ntm.get("b03", np.floating))
        self.assertEqual(3, ntm.get("b02", np.floating))

    def test_new_with_data_is_default(self):
        ntm = NameTypeMapping.new("spline_orders", 3, 2)
        self.assertEqual(2, ntm.get("scl", np.uint8))
        self.assertEqual(2, ntm.get("b03", np.float32))
        self.assertEqual(2, ntm.get("b02", np.int32))

    def test_new_with_data_is_dict(self):
        ntm = NameTypeMapping.new(
            "spline_orders", 3, {0: ["scl"], 1: [np.floating, "cld"]}
        )
        self.assertEqual(0, ntm.get("scl", np.uint8))
        self.assertEqual(1, ntm.get("b03", np.float32))
        self.assertEqual(1, ntm.get("cld", np.int16))
        self.assertEqual(3, ntm.get("b06", np.int32))

    # noinspection PyMethodMayBeStatic
    def test_new_fail(self):
        with pytest.raises(
            TypeError,
            match=r"spline_orders\[1\]: name or dtype expected, but got value 3",
        ):
            NameTypeMapping.new("spline_orders", 3, {1: [np.floating, "cld", 3]})

        with pytest.raises(
            TypeError, match="agg_methods: keys must be of type str, but got key 1"
        ):
            # noinspection PyTypeChecker
            NameTypeMapping.new("agg_methods", "´mean", {1: [np.floating, "cld"]})
