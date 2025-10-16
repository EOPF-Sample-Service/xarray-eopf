#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

import dask.array as da
import numpy as np
import pyproj.crs
from scipy.interpolate import griddata
import xarray as xr
from xcube_resampling.constants import AggMethods, InterpMethod
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.rectify import rectify_dataset
from xcube_resampling.utils import resolution_meters_to_degrees

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.constants import FloatInt, MEAN_EARTH_RADIUS
from xarray_eopf.source import get_source_path
from xarray_eopf.utils import (
    NameFilter,
    assert_arg_is_instance,
)

_CRS = "EPSG:4326"
_CHUNKSIZE = (1024, 1024)


class Sen3(AnalysisMode, ABC):

    # Default resolution in meter for subclasses to override
    default_resolution: int | None = None

    def is_valid_source(self, source: Any) -> bool:
        root_path = get_source_path(source)
        return (
            (
                f"S3A_{self.product_type}_" in root_path
                or f"S3B_{self.product_type}_" in root_path
            )
            if root_path
            else False
        )

    def get_applicable_params(self, **kwargs) -> dict[str, any]:
        params = {}

        resolution = kwargs.get("resolution")
        if resolution is not None:
            assert_arg_is_instance(resolution, "resolution", (int, float))
            params.update(resolution=resolution)

        interp_methods = kwargs.get("interp_methods")
        if interp_methods is not None:
            assert_arg_is_instance(interp_methods, "interp_methods", (str, int, dict))
            params.update(interp_methods=interp_methods)

        agg_methods = kwargs.get("agg_methods")
        if agg_methods is not None:
            assert_arg_is_instance(agg_methods, "agg_methods", (str, dict))
            params.update(agg_methods=agg_methods)

        return params

    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        warnings.warn(
            "Analysis mode not implemented for given source, return data tree as-is."
        )
        return datatree

    def transform_dataset(self, dataset: xr.Dataset, **params) -> xr.Dataset:
        return self.assign_grid_mapping(dataset)

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: FloatInt | tuple[FloatInt, FloatInt] | None = None,
        interp_methods: InterpMethod | None = None,
        agg_methods: AggMethods | None = None,
    ) -> xr.Dataset:
        # filter dataset by variable names
        name_filter = NameFilter(includes=includes, excludes=excludes)
        dataset = datatree.measurements.to_dataset()
        variable_names = [k for k in dataset.data_vars if name_filter.accept(str(k))]
        if not variable_names:
            raise ValueError("No variables selected")
        dataset = dataset[variable_names]
        # remove coordinates except for latitude and longitude
        coords = []
        for coord in dataset.coords:
            if coord not in ["latitude", "longitude"]:
                coords.append(coord)
        dataset = dataset.drop_vars(coords)

        # reproject dataset to regular grid
        source_gm = GridMapping.from_dataset(dataset)
        if resolution is None:
            center_lat = (source_gm.xy_bbox[1] + source_gm.xy_bbox[3]) / 2
            resolution = resolution_meters_to_degrees(
                self.default_resolution, center_lat
            )
        target_gm = GridMapping.regular_from_bbox(
            bbox=source_gm.xy_bbox,
            xy_res=resolution,
            crs=source_gm.crs,
            tile_size=_CHUNKSIZE,
        )

        rectified_dataset = rectify_dataset(
            dataset,
            source_gm=source_gm,
            target_gm=target_gm,
            interp_methods=interp_methods,
            agg_methods=agg_methods,
        )
        rectified_dataset.attrs = self.process_metadata(datatree)
        return rectified_dataset

    # noinspection PyMethodMayBeStatic
    def process_metadata(self, datatree: xr.DataTree | xr.Dataset):
        # TODO: process metadata and try adhering to CF conventions
        other_metadata = datatree.attrs.get("other_metadata", {})
        return other_metadata

    # noinspection PyMethodMayBeStatic
    def assign_grid_mapping(self, dataset: xr.Dataset) -> xr.Dataset:
        crs = pyproj.CRS.from_epsg(4326)
        dataset = dataset.assign_coords(
            dict(spatial_ref=xr.DataArray(0, attrs=crs.to_cf()))
        )
        for var_name in dataset.data_vars:
            dataset[var_name].attrs["grid_mapping"] = "spatial_ref"

        return dataset

    @staticmethod
    def orthorectify_geolocation(datatree: xr.DataTree) -> xr.DataTree:
        return datatree


class Sen3Ol1Err(Sen3):
    product_type = "OL_1_ERR"
    default_resolution = 1200


class Sen3Ol1Efr(Sen3):
    product_type = "OL_1_EFR"
    default_resolution = 300


# Broken data in: https://stac.browser.user.eopf.eodc.eu/collections/sentinel-3-olci-l2-lrr?.language=en
# class Sen3Ol2Lrr(Sen3):
#     product_type = "OL_2_LRR"
#     default_resolution = 1200


class Sen3Ol2Lfr(Sen3):
    product_type = "OL_2_LFR"
    default_resolution = 300


class Sen3Sl2Lst(Sen3):
    product_type = "SL_2_LST"
    default_resolution = 1000


class Sen3Sl1Rbt(Sen3):
    product_type = "SL_1_RBT"
    default_resolution = None

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: FloatInt | tuple[FloatInt, FloatInt] | None = None,
        interp_methods: InterpMethod | None = None,
        agg_methods: AggMethods | None = None,
    ) -> xr.Dataset:
        # filter dataset by variable names
        name_filter = NameFilter(includes=includes, excludes=excludes)
        dataset_map = {}
        for sub_group in datatree.measurements.children.keys():
            dataset = datatree.measurements[sub_group].to_dataset()
            variable_names = [
                k for k in dataset.data_vars if name_filter.accept(str(k))
            ]
            if variable_names:
                dataset_sel = dataset[variable_names]
                # remove coordinates except for latitude and longitude
                coords = []
                for coord in dataset_sel.coords:
                    if coord not in ["latitude", "longitude"]:
                        coords.append(coord)
                dataset_sel = dataset_sel.drop_vars(coords)
                dataset_map[sub_group] = (
                    dataset_sel,
                    GridMapping.from_dataset(dataset_sel),
                )
        if not dataset_map:
            raise ValueError("No variables selected")

        # get outer bounding box
        bboxs = np.array([gm.xy_bbox for (_, gm) in dataset_map.values()])
        bbox = self._get_outer_bbox(bboxs)

        # get resolution if not given
        if resolution is None:
            subgroups_res_1000 = ["fnadir", "foblique", "inadir", "ioblique"]
            if all(key in subgroups_res_1000 for key in dataset_map.keys()):
                resolution = 1000
            else:
                resolution = 500
            center_lat = (bbox[1] + bbox[3]) / 2
            resolution = resolution_meters_to_degrees(resolution, center_lat)
        target_gm = GridMapping.regular_from_bbox(
            bbox=bbox,
            xy_res=resolution,
            crs=_CRS,
            tile_size=_CHUNKSIZE,
        )

        # rectify each group and combine them into one dataset
        final_dataset = None
        for source_ds, source_gm in dataset_map.values():
            rectified_dataset = rectify_dataset(
                source_ds,
                source_gm=source_gm,
                target_gm=target_gm,
                interp_methods=interp_methods,
                agg_methods=agg_methods,
            )
            if final_dataset is None:
                final_dataset = rectified_dataset
            else:
                final_dataset.update(rectified_dataset)
        final_dataset.attrs = self.process_metadata(datatree)
        return final_dataset

    @staticmethod
    def _get_outer_bbox(bboxs: np.ndarray) -> list[FloatInt]:
        if any(bboxs[:, 0] > bboxs[:, 2]):
            # crossing anti-meridian
            bbox = [
                np.min(bboxs[:, 0][bboxs[:, 0] > 0]).item(),
                np.min(bboxs[:, 1]).item(),
                np.max(bboxs[:, 2][bboxs[:, 2] < 0]).item(),
                np.max(bboxs[:, 3]).item(),
            ]
        else:
            bbox = [
                np.min(bboxs[:, 0]).item(),
                np.min(bboxs[:, 1]).item(),
                np.max(bboxs[:, 2]).item(),
                np.max(bboxs[:, 3]).item(),
            ]
        return bbox


def register(registry: AnalysisModeRegistry):
    registry.register(Sen3Ol1Err)
    registry.register(Sen3Ol1Efr)
    registry.register(Sen3Ol2Lfr)
    # registry.register(Sen3Ol2Lrr)
    registry.register(Sen3Sl1Rbt)
    registry.register(Sen3Sl2Lst)


def orthorectify_geolocation(
    dataset: xr.Dataset,
    lat: xr.DataArray,
    lon: xr.DataArray,
    elev: xr.DataArray,
    va_zenith: xr.DataArray,
    va_azimuth: xr.DataArray,
    interp_method: str = "linear",
) -> xr.Dataset:
    """Apply terrain-induced parallax correction to satellite geolocation coordinates.

    This function adjusts latitude and longitude values based on viewing geometry
    and surface elevation, accounting for the apparent displacement of ground
    targets caused by non-zero terrain height when observed at an oblique angle.
    The correction assumes a spherical Earth and a locally planar surface.

    Args:
        lat: Latitude coordinates in degrees (geodetic).
        lon: Longitude coordinates in degrees (geodetic).
        elev: Surface height above reference ellipsoid/sphere (meters).
        va_zenith: Viewing zenith angle in degrees. Must be defined per pixel.
        va_azimuth: Viewing azimuth angle in degrees. Convention of Sentinel-3 is
            clockwise from North.

    Returns:
        Dataset containing the displacement in latitude and longitude

    Notes:
        - Assumes a mean spherical Earth radius of 6,370,997 meters.
        - No correction is applied for atmospheric refraction or ellipsoidal geometry.
        - Results may be inaccurate near the poles where `cos(latitude) → 0`.
    """
    # Convert everything to rad
    phi_true = da.deg2rad(lat.data)
    theta_v = da.deg2rad(va_zenith.data)
    phi_v = da.deg2rad(va_azimuth.data)

    # Horizontal displacement
    t = elev.data * da.tan(theta_v)
    delta_phi = t * da.cos(phi_v) / MEAN_EARTH_RADIUS
    delta_lam = t * da.sin(phi_v) / (MEAN_EARTH_RADIUS * da.cos(phi_true))

    # convert back to degree
    lat_diff = da.rad2deg(delta_phi)
    lon_diff = da.rad2deg(delta_lam)

    # interpolate and correct the latitude and longitude of the dataset
    def _interpolate(displacement, lat_source, lon_source, lat_target, lon_target):
        points = np.stack([lat_source.ravel(), lon_source.ravel()], axis=-1)
        return griddata(
            points, displacement.ravel(), (lat_target, lon_target), method=interp_method
        )

    lat_diff_interp = xr.apply_ufunc(
        _interpolate,
        lat_diff,
        lat.data,
        lon.data,
        dataset.latitude.data,
        dataset.longitude.data,
        input_core_dims=[[], [], [], [], []],
        output_core_dims=[],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[lat_diff.dtype],
    )
    lon_diff_interp = xr.apply_ufunc(
        _interpolate,
        lon_diff,
        lat.data,
        lon.data,
        dataset.latitude.data,
        dataset.longitude.data,
        input_core_dims=[[], [], [], [], []],
        output_core_dims=[],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[lat_diff.dtype],
    )

    return dataset.assign_coords(
        dict(
            latitude=dataset.latitude - lat_diff_interp,
            longitude=dataset.longitude - lon_diff_interp,
        )
    )
