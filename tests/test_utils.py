#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
from typing import Literal
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from tests.helpers import make_s2_msi
from xarray_eopf.utils import (
    NameFilter,
    find_relative_bbox,
    assert_arg_has_length,
    assert_arg_is_instance,
    assert_arg_is_one_of,
    build_footprint_uv_mapping,
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
        self.assertIsNone(assert_arg_is_instance("a", "order", Literal["a", "b"]))

        with pytest.raises(
            TypeError, match="order argument must have type int, was float"
        ):
            self.assertIsNone(assert_arg_is_instance(2.0, "order", int))
        with pytest.raises(
            TypeError, match="order argument must have type int or float, was str"
        ):
            self.assertIsNone(assert_arg_is_instance("4", "order", (int, float)))
        with pytest.raises(TypeError) as exc:
            self.assertIsNone(assert_arg_is_instance("c", "order", Literal["a", "b"]))
        self.assertEqual(
            str(exc.value), "order argument must be one of ('a', 'b'), was 'c'"
        )

    def test_assert_arg_has_length(self):
        self.assertIsNone(assert_arg_has_length([1, 2, 3], "test_arg", 3))
        with pytest.raises(
            ValueError, match="test_arg argument must have length 3, but has 2."
        ):
            self.assertIsNone(assert_arg_has_length([1, 2], "test_arg", 3))
        with pytest.raises(
            TypeError, match="test_arg argument must be a sequence with length "
        ):
            self.assertIsNone(assert_arg_has_length(123, "test_arg", 3))


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


class BuildFootprintUvMappingTest(TestCase):
    def test_accepts_closed_ring_points(self):
        open_ring = np.array(
            [[10.0, 50.0], [12.0, 50.0], [12.0, 52.0], [10.0, 52.0]],
            dtype=float,
        )
        closed_ring = np.vstack([open_ring, open_ring[0]])

        open_xy, open_uv = build_footprint_uv_mapping(open_ring)
        closed_xy, closed_uv = build_footprint_uv_mapping(closed_ring)

        self.assertTrue(np.allclose(open_xy, closed_xy))
        self.assertTrue(np.allclose(open_uv, closed_uv))

    def test_find_relative_bbox_uses_southern_utm_epsg(self):
        stac_meta = {
            "geometry": {
                "coordinates": [
                    [
                        [10.0, -11.0],
                        [11.0, -11.0],
                        [11.0, -10.0],
                        [10.0, -10.0],
                        [10.0, -11.0],
                    ]
                ]
            },
            "properties": {"sat:orbit_state": "descending"},
        }
        bbox = [10.2, -10.8, 10.8, -10.2]

        with patch("xarray_eopf.utils.pyproj.Transformer.from_crs") as from_crs:
            transformer = from_crs.return_value
            transformer.transform.return_value = (
                np.array([0.0, 1.0, 1.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 1.0, 1.0, 0.0]),
            )
            transformer.transform_bounds.return_value = (0.2, 0.2, 0.8, 0.8)

            rel_bbox = find_relative_bbox(stac_meta, bbox)

        _, utm_epsg = from_crs.call_args.args[:2]
        self.assertEqual("EPSG:32732", utm_epsg)
        self.assertEqual(4, len(rel_bbox))
