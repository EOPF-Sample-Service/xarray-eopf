#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
import re
import time
from collections.abc import Collection, Iterable
from typing import Any, Type, TypeAlias, TypeVar

import pystac
import numpy as np
from shapely.geometry import shape
import xarray as xr

T = TypeVar("T")


class timeit:
    """A context manager used to measure time it takes
    to execute its with-block.
    The result is available as `time_delta` attribute.

    Args:
        label: A text label
        silent: Whether to suppress printing the result
    """

    def __init__(self, label: str | None = None, silent: bool = False):
        self.label = label
        self.silent = silent
        self.start_time: float | None = None
        self.time_delta: float | None = None

    def __enter__(self) -> "timeit":
        self.start_time = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_delta = time.process_time() - self.start_time
        if not self.silent:
            print(f"{self.label or 'code block'} took {self.time_delta:.3f} seconds")


def assert_arg_is_instance(value: Any, name: str, data_type: Type | tuple[Type, ...]):
    """Check if the `value` of the argument `name` has the given `data_type`.
    If not, raise `TypeError`.
    """
    if not isinstance(value, data_type):
        if isinstance(data_type, tuple):
            data_type_name = _text_items_to_text(t.__name__ for t in data_type)
        else:
            data_type_name = data_type.__name__
        actual_type_name = type(value).__name__
        raise TypeError(
            f"{name} argument must have type {data_type_name}, was {actual_type_name}"
        )


def assert_arg_has_length(value: Any, name: str, length: int):
    """
    Ensure that argument `name` has a length equal to `length`.
    Raises:
        TypeError: if `value` has no length.
        ValueError: if `value` length is not equal to `length`.
    """
    try:
        actual_length = len(value)
    except TypeError:
        raise TypeError(
            f"{name} argument must be a sequence with length {length}, "
            f"but got object of type {type(value).__name__!r} with no length."
        )

    if actual_length != length:
        raise ValueError(
            f"{name} argument must have length {length}, but has {actual_length}."
        )


def assert_arg_is_one_of(value: Any, name: str, collection: Collection):
    """Check if the `value` of the argument `name` is one of the items in `collection`.
    If not, raise `ValueError`.
    """
    if value not in collection:
        items_text = _text_items_to_text(map(repr, collection))
        raise ValueError(f"{name} argument must be {items_text}, was {value!r}")


def _text_items_to_text(items: Iterable[str]) -> str:
    items = tuple(items)
    assert len(items) >= 2
    return f"{', '.join(items[:-1])} or {items[-1]}"


def get_data_tree_item(
    datatree: xr.DataTree, group_path: str | Iterable[str]
) -> xr.DataTree | xr.DataArray | None:
    """Get a group in a data tree given by its group path.

    Args:
        datatree: The data tree object
        group_path: An iterable of group names or a string that
            uses slashes as group name separators

    Returns:
        The group of type `xr.DataTree` or `None` if it cannot be found
    """
    if isinstance(group_path, str):
        group_path = group_path.split("/")
    group = datatree
    for group_name in group_path:
        if group_name:
            if group_name not in group:
                return None
            group = group[group_name]
    return group


Matcher: TypeAlias = Any


class NameFilter:
    def __init__(
        self,
        includes: str | Iterable[str] | None,
        excludes: str | Iterable[str] | None = None,
    ):
        self.includes = NameFilter._norm_patterns(includes)
        self.excludes = NameFilter._norm_patterns(excludes)

    def filter(self, names: Iterable[str]) -> Iterable[str]:
        return filter(self.accept, names)

    def accept(self, var_name: str) -> bool:
        accepted = True
        if self.includes:
            accepted = False
            for p, m in self.includes:
                if var_name == p or var_name.startswith(p) or m.match(var_name):
                    accepted = True
                    break
        if accepted:
            for p, m in self.excludes:
                if var_name == p or var_name.startswith(p) or m.match(var_name):
                    accepted = False
                    break
        return accepted

    @staticmethod
    def _norm_patterns(
        patterns: str | Iterable[str] | None,
    ) -> list[tuple[str, Matcher]]:
        patterns = (patterns,) if isinstance(patterns, str) else (patterns or ())
        return [(p, re.compile(p)) for p in patterns if p]


def _segment_points(points: np.ndarray, i0: int, i1: int) -> np.ndarray:
    """Extract points from a closed boundary ring between two indices.

    Args:
        points: Boundary coordinates in ring order.
        i0: Start index.
        i1: End index.

    Returns:
        Coordinate segment including both start and end points. If `i0 > i1`,
        the segment wraps across the ring end.
    """
    if i0 <= i1:
        return points[i0 : i1 + 1]
    return np.vstack([points[i0:], points[: i1 + 1]])


def build_footprint_uv_mapping(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Create geometry control points and normalized image coordinates.

    Args:
        points: Boundary coordinates in ring order.

    Returns:
        A tuple `(control_xy, control_uv)` where `control_xy` are boundary
        coordinates and `control_uv` are corresponding normalized image
        coordinates.
    """
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    lon = points[:, 0]
    lat = points[:, 1]

    idx_ll = int(np.argmin(lat + lon))
    idx_ur = int(np.argmax(lat + lon))
    idx_ul = int(np.argmax(lat - lon))
    idx_lr = int(np.argmin(lat - lon))

    idx_to_uv = {
        idx_ll: np.array([0.0, 1.0]),
        idx_lr: np.array([1.0, 1.0]),
        idx_ul: np.array([0.0, 0.0]),
        idx_ur: np.array([1.0, 0.0]),
    }
    ordered_idx = sorted(idx_to_uv.keys())

    control_xy = []
    control_uv = []
    for k, start in enumerate(ordered_idx):
        end = ordered_idx[(k + 1) % len(ordered_idx)]
        seg_xy = _segment_points(points, start, end)
        uv0 = idx_to_uv[start]
        uv1 = idx_to_uv[end]
        t = np.linspace(0.0, 1.0, len(seg_xy))
        seg_uv = (1 - t)[:, None] * uv0 + t[:, None] * uv1
        control_xy.append(seg_xy[:-1])
        control_uv.append(seg_uv[:-1])

    return np.vstack(control_xy), np.vstack(control_uv)
