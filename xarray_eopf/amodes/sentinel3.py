#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import warnings
from abc import ABC
from collections.abc import Iterable
from typing import Any

import numpy as np
import pyproj.crs
from scipy.interpolate import griddata
import xarray as xr
from xcube_resampling.constants import SpatialAggMethods, SpatialInterpMethods
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
_CHUNKSIZE = (2048, 2048)


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
        interp_methods: SpatialInterpMethods | None = None,
        agg_methods: SpatialAggMethods | None = None,
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

        # orthorectify geolocation for elevation and viewing geometry
        dataset = self._apply_orthorectification(dataset, datatree)

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

    def _apply_orthorectification(
        self, dataset: xr.Dataset, datatree: xr.DataTree
    ) -> xr.Dataset:
        """Placeholder method to be overwritten by product-specific subclasses
        handling SLSTR datasets.
        """
        return dataset


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

    def _apply_orthorectification(
        self, dataset: xr.Dataset, datatree: xr.DataTree
    ) -> xr.Dataset:
        elevation = datatree.conditions.auxiliary.elevation.persist()
        valid_cols = ~elevation.isnull().any(dim="rows").values
        dataset = dataset.isel(columns=valid_cols)
        elevation = elevation.isel(columns=valid_cols)
        return orthorectify_geolocation(
            dataset,
            elevation,
            datatree.conditions.meteorology.latitude,
            datatree.conditions.meteorology.longitude,
            datatree.conditions.geometry.sat_zenith_tn,
            datatree.conditions.geometry.sat_azimuth_tn,
        )


class Sen3Sl1Rbt(Sen3):
    product_type = "SL_1_RBT"
    default_resolution = None

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: FloatInt | tuple[FloatInt, FloatInt] | None = None,
        interp_methods: SpatialInterpMethods | None = None,
        agg_methods: SpatialAggMethods | None = None,
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
                # orthorectify dataset
                dataset = self._apply_orthorectification(dataset, datatree)
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

    def _apply_orthorectification(
        self, dataset: xr.Dataset, datatree: xr.DataTree
    ) -> xr.Dataset:
        elevation = dataset.elevation.persist()
        valid_cols = ~elevation.isnull().any(dim="rows").values
        dataset = dataset.isel(columns=valid_cols)
        elevation = elevation.isel(columns=valid_cols)
        if any(var.endswith("o") for var in dataset.data_vars.keys()):
            sat_zenith = datatree.conditions.geometry_to.sat_zenith_to
            sat_azimuth = datatree.conditions.geometry_to.sat_azimuth_to
        else:
            sat_zenith = datatree.conditions.geometry_tn.sat_zenith_tn
            sat_azimuth = datatree.conditions.geometry_tn.sat_azimuth_tn

        return orthorectify_geolocation(
            dataset,
            elevation,
            datatree.conditions.meteorology.latitude,
            datatree.conditions.meteorology.longitude,
            sat_zenith,
            sat_azimuth,
        )


def register(registry: AnalysisModeRegistry):
    registry.register(Sen3Ol1Err)
    registry.register(Sen3Ol1Efr)
    registry.register(Sen3Ol2Lfr)
    # registry.register(Sen3Ol2Lrr)
    registry.register(Sen3Sl1Rbt)
    registry.register(Sen3Sl2Lst)


def orthorectify_geolocation(
    dataset: xr.Dataset,
    elev: xr.DataArray,
    lat: xr.DataArray,
    lon: xr.DataArray,
    sat_zenith: xr.DataArray,
    sat_azimuth: xr.DataArray,
) -> xr.Dataset:
    """
    Apply terrain-induced parallax correction to satellite geolocation coordinates.

    Args:
        dataset: Dataset containing geolocation coordinates to be corrected. Must
            include `latitude` and `longitude` coordinates.
        elev: Surface elevation in meters above the reference ellipsoid or sphere.
        lat: Latitude values defining the source grid for satellite angle variables.
        lon: Longitude values defining the source grid for satellite angle variables.
        sat_zenith: Viewing zenith angle in degrees.
        sat_azimuth: Viewing azimuth angle in degrees. Sentinel-3 convention is
            clockwise from North.

    Returns:
        A new dataset with corrected `latitude` and `longitude` coordinates.

    Notes:
    This function adjusts latitude and longitude coordinates in the input dataset to
    compensate for horizontal displacement effects caused by viewing elevated terrain
    from an oblique angle. The correction accounts for local surface height and
    satellite viewing geometry, estimating the apparent pixel shift under the
    assumption of a spherical Earth.

    Satellite zenith and azimuth angles are first interpolated from their native
    grid to the geolocation grid of the dataset using `scipy.interpolate.griddata`.
    Displacements are computed in radians and then applied to produce corrected
    latitude and longitude coordinates.

    The following assumptions are made:

        - Assumes a spherical Earth with a fixed radius of 6,370,997 meters.
        - Atmospheric refraction and ellipsoidal geometry effects are not considered.
        - Accuracy may degrade near the poles where `cos(latitude) → 0`.
    """
    # load coordinates of dataset
    ds_lat = dataset.latitude.values
    ds_lon = dataset.longitude.values

    # interpolate satellite zenith and azimuth angle
    def _interpolate(
        angle: np.ndarray,
        lat_source: np.ndarray,
        lon_source: np.ndarray,
        lat_target: np.ndarray,
        lon_target: np.ndarray,
    ) -> np.ndarray:
        pts_source = np.stack([lat_source.ravel(), lon_source.ravel()], axis=-1)
        pts_target = np.stack([lat_target.ravel(), lon_target.ravel()], axis=-1)
        angle_interp = np.asarray(
            griddata(pts_source, angle.ravel(), pts_target, method="linear")
        )

        # Identify NaNs (outside convex hull)
        mask = np.isnan(angle_interp)
        if np.any(mask):
            # Second pass: nearest fill for NaNs only
            angle_interp[mask] = griddata(
                pts_source, angle.ravel(), pts_target[mask], method="nearest"
            )

        return angle_interp.reshape(lat_target.shape)

    sat_zenith_interp = _interpolate(
        sat_zenith.values, lat.values, lon.values, ds_lat, ds_lon
    )
    sat_azimuth_interp = _interpolate(
        sat_azimuth.values, lat.values, lon.values, ds_lat, ds_lon
    )

    # Convert everything to rad
    phi_true = np.deg2rad(ds_lat)
    theta_v = np.deg2rad(sat_zenith_interp)
    phi_v = np.deg2rad(sat_azimuth_interp)

    # Horizontal displacement
    t = elev.values * np.tan(theta_v)
    delta_phi = t * np.cos(phi_v) / MEAN_EARTH_RADIUS
    delta_lam = t * np.sin(phi_v) / (MEAN_EARTH_RADIUS * np.cos(phi_true))

    # convert back to degree
    lat_diff = np.rad2deg(delta_phi)
    lon_diff = np.rad2deg(delta_lam)

    return dataset.assign_coords(
        dict(
            latitude=(dataset.latitude.dims, ds_lat - lat_diff),
            longitude=(dataset.latitude.dims, ds_lon - lon_diff),
        )
    )
