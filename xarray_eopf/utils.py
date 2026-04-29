#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
import re
import time
from collections.abc import Collection, Iterable
from typing import Any, Literal, Sequence, Type, TypeAlias, TypeVar

import numpy as np
import pyproj
import xarray as xr
from scipy.interpolate import RBFInterpolator

from .constants import _CRS_WGS84

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


def build_footprint_uv_mapping(
    points: np.ndarray, orbit_state: Literal["ascending", "descending"] = "descending"
) -> tuple[np.ndarray, np.ndarray]:
    """Create geometry control points and normalized image coordinates.

    Args:
        points: Boundary coordinates in ring order.
        orbit_state: Orbit direction, either "ascending" or "descending".

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

    control_xy = np.array(
        [
            [lon[idx_ll], lat[idx_ll]],
            [lon[idx_lr], lat[idx_lr]],
            [lon[idx_ul], lat[idx_ul]],
            [lon[idx_ur], lat[idx_ur]],
        ]
    )

    if orbit_state == "descending":
        control_uv = np.array([[0.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0]])
    else:
        control_uv = np.array([[1.0, 0.0], [0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    return control_xy, control_uv


def _find_relative_bbox(
    stac_meta: dict, bbox: Sequence[float | int]
) -> Sequence[float]:
    """
    Calculates the relative bounding box coordinates in image reference space based
    on geographic bounding box and satellite metadata.

    The function processes a geographic bounding box (`bbox`) by converting its
    coordinates from WGS84 to UTM projection based on the center of the input
    geospatial points. Using the STAC metadata (`stac_meta`), it builds a
    mapping between ground control points and image coordinates using Radial Basis
    Function interpolation. The final result is the corresponding bounding box
    coordinates in the image reference space.

    Args:
        stac_meta: A dictionary containing metadata for the satellite image
            with necessary geographic and orbital details. Includes geometry
            coordinates and satellite orbit state.
        bbox: A sequence of four elements representing the
            geographic bounding box in WGS84 format (west, south, east, north).

    Returns:
        A sequence of four elements representing the bounding box
        coordinates (min_u, min_v, max_u, max_v) in the image reference space.
    """
    points = np.array(stac_meta["geometry"]["coordinates"][0])
    orbit_state = stac_meta["properties"]["sat:orbit_state"]

    # convert to utm
    center = np.mean(points, axis=0)
    utm_zone = int(np.floor((center[0] + 180) / 6) + 1)
    if center[1] >= 0:
        utm_epsg = f"EPSG:326{utm_zone}"
    else:
        utm_epsg = f"EPSG:327{utm_zone}"
    transformer = pyproj.Transformer.from_crs(_CRS_WGS84, utm_epsg, always_xy=True)
    utm_points = transformer.transform(points[:, 0], points[:, 1])
    utm_points = np.stack(utm_points).transpose()
    utm_bbox = transformer.transform_bounds(*bbox, densify_pts=21)

    control_xy, control_uv = build_footprint_uv_mapping(utm_points, orbit_state)
    u_model = RBFInterpolator(control_xy, control_uv[:, 0], kernel="thin_plate_spline")
    v_model = RBFInterpolator(control_xy, control_uv[:, 1], kernel="thin_plate_spline")
    corners = np.array(
        [
            [utm_bbox[0], utm_bbox[1]],
            [utm_bbox[2], utm_bbox[1]],
            [utm_bbox[0], utm_bbox[3]],
            [utm_bbox[2], utm_bbox[3]],
        ]
    )
    us = u_model(corners)
    vs = v_model(corners)

    return np.min(us), np.min(vs), np.max(us), np.max(vs)
