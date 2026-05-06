#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import warnings
from abc import ABC
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pyproj.crs
import xarray as xr
from scipy.interpolate import griddata
from xcube_resampling.constants import SpatialAggMethods, SpatialInterpMethods
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.rectify import rectify_dataset
from xcube_resampling.utils import (
    clip_dataset_by_bbox,
    reproject_bbox,
    resolution_meters_to_degrees,
)

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.constants import CRS_WGS84, MEAN_EARTH_RADIUS, FloatInt
from xarray_eopf.source import get_source_path
from xarray_eopf.utils import (
    NameFilter,
    assert_arg_has_length,
    assert_arg_is_instance,
    find_relative_bbox,
)

_CHUNKSIZE = (2048, 2048)
_CUTOUT_BUFFER_OLCI = 50
_CUTOUT_BUFFER_SLSTR = 20


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

    def get_applicable_params(self, **kwargs) -> dict[str, Any]:
        params = {}

        resolution = kwargs.get("resolution")
        if resolution is not None:
            assert_arg_is_instance(resolution, "resolution", (int, float))
            params.update(resolution=resolution)

        bbox = kwargs.get("bbox")
        if bbox is not None:
            assert_arg_is_instance(bbox, "bbox", (Sequence,))
            assert_arg_has_length(bbox, "bbox", 4)
            params.update(bbox=bbox)

        crs = kwargs.get("crs")
        if crs is not None:
            if isinstance(crs, str):
                crs = pyproj.CRS.from_string(crs)
            assert_arg_is_instance(crs, "crs", (pyproj.CRS,))
            params.update(crs=crs)

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
            "Analysis mode not implemented for given source, return data tree as-is.",
            UserWarning,
        )
        return datatree

    def transform_dataset(
        self, dataset: xr.Dataset, stac_meta: dict, **params
    ) -> xr.Dataset:
        return self.assign_grid_mapping(dataset)

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: FloatInt | tuple[FloatInt, FloatInt] | None = None,
        bbox: Sequence[float | int] | None = None,
        crs: pyproj.CRS | None = None,
        interp_methods: SpatialInterpMethods | None = None,
        agg_methods: SpatialAggMethods | None = None,
    ) -> xr.Dataset:
        if crs is None:
            crs = CRS_WGS84

        # filter dataset by variable names
        name_filter = NameFilter(includes=includes, excludes=excludes)
        dataset = datatree.measurements.to_dataset()
        variable_names = [k for k in dataset.data_vars if name_filter.accept(str(k))]
        if not variable_names:
            raise ValueError("No variables selected")
        dataset = dataset[variable_names]
        dataset = self._add_elevation(dataset, datatree)

        # remove coordinates except for latitude and longitude
        coords = []
        for coord in dataset.coords:
            if coord not in ["latitude", "longitude"]:
                coords.append(coord)
        dataset = dataset.drop_vars(coords)

        # clip by bounding box
        bbox_idx = None
        if bbox:
            bbox_wgs84 = reproject_bbox(bbox, crs, "EPSG:4326")
            stac_meta = datatree.attrs["stac_discovery"]
            rel_bbox = find_relative_bbox(stac_meta, bbox_wgs84)
            buffer = (
                _CUTOUT_BUFFER_SLSTR
                if self.product_type == "SL_2_LST"
                else _CUTOUT_BUFFER_OLCI
            )
            dataset, bbox_idx = _clip_dataset_relative_bbox(
                rel_bbox, dataset, buffer=buffer
            )
            if any(size <= 1 for size in dataset.sizes.values()):
                warnings.warn(
                    "Clipping with the specified bounding box "
                    "resulted in a dataset too small to compute a valid grid "
                    "mapping. Returning clipped dataset.",
                    UserWarning,
                )
                return dataset

        dataset["latitude"] = dataset["latitude"].persist()
        dataset["longitude"] = dataset["longitude"].persist()

        # orthorectify geolocation for elevation and viewing geometry
        dataset = self._apply_orthorectification(dataset, datatree, bbox_idx=bbox_idx)
        dataset = dataset[variable_names]

        # reproject dataset to regular grid
        source_gm = GridMapping.from_dataset(dataset)
        if bbox is None:
            bbox = source_gm.xy_bbox
        if resolution is None:
            resolution = self.default_resolution
            if crs.is_geographic:
                center_lat = (source_gm.xy_bbox[1] + source_gm.xy_bbox[3]) / 2
                resolution = resolution_meters_to_degrees(resolution, center_lat)

        target_gm = GridMapping.regular_from_bbox(
            bbox=bbox, xy_res=resolution, crs=crs, tile_size=_CHUNKSIZE
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
    def assign_grid_mapping(self, dataset: xr.Dataset) -> xr.Dataset:
        crs = pyproj.CRS.from_epsg(4326)
        dataset = dataset.assign_coords(
            dict(spatial_ref=xr.DataArray(0, attrs=crs.to_cf()))
        )
        for var_name in dataset.data_vars:
            dataset[var_name].attrs["grid_mapping"] = "spatial_ref"

        return dataset

    # noinspection PyMethodMayBeStatic
    def process_metadata(self, datatree: xr.DataTree) -> dict:
        other_metadata = datatree.attrs.get("other_metadata", {})
        return other_metadata

    def _apply_orthorectification(
        self,
        dataset: xr.Dataset,
        datatree: xr.DataTree,
        bbox_idx: tuple[int, int, int, int] = None,
    ) -> xr.Dataset:
        """Placeholder method to be overwritten by product-specific subclasses
        handling SLSTR datasets.
        """
        return dataset

    def _add_elevation(self, dataset: xr.Dataset, datatree: xr.DataTree) -> xr.Dataset:
        """Placeholder method to be overwritten by product-specific subclasses
        handling SLSTR Level-2 LST product.
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
        self,
        dataset: xr.Dataset,
        datatree: xr.DataTree,
        bbox_idx: tuple[int, int, int, int] = None,
    ) -> xr.Dataset:
        angles = datatree.conditions.geometry.to_dataset()
        angles = angles[["sat_zenith_tn", "sat_azimuth_tn"]]
        angles = angles.rename(
            dict(sat_zenith_tn="sat_zenith", sat_azimuth_tn="sat_azimuth")
        )
        angles = angles.assign_coords(
            dict(
                latitude=datatree.conditions.meteorology.latitude,
                longitude=datatree.conditions.meteorology.longitude,
            )
        )
        if bbox_idx:
            # The angles dataset has coarser sampling along the longitude axis,
            # while the latitude dimension matches the resolution of the input dataset.
            # Subset only along the latitude dimension to align with the target region.
            angles = angles.isel(rows=slice(bbox_idx[1], bbox_idx[3]))

        return orthorectify_geolocation(dataset, angles)

    def _add_elevation(self, dataset: xr.Dataset, datatree: xr.DataTree) -> xr.Dataset:
        """Placeholder method to be overwritten by product-specific subclasses
        handling SLSTR Level-2 LST product.
        """
        dataset["elevation"] = datatree.conditions.auxiliary.elevation
        return dataset


class Sen3Sl1Rbt(Sen3):
    product_type = "SL_1_RBT"
    default_resolution = None

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: FloatInt | tuple[FloatInt, FloatInt] | None = None,
        bbox: Sequence[float | int] | None = None,
        crs: pyproj.CRS | None = None,
        interp_methods: SpatialInterpMethods | None = None,
        agg_methods: SpatialAggMethods | None = None,
    ) -> xr.Dataset:
        if crs is None:
            crs = CRS_WGS84

        # filter dataset by variable names
        name_filter = NameFilter(includes=includes, excludes=excludes)
        dataset_map = {}
        for sub_group in datatree.measurements.children.keys():
            dataset = datatree.measurements[sub_group].to_dataset()
            variable_names = [
                k for k in dataset.data_vars if name_filter.accept(str(k))
            ]
            if variable_names:
                if "elevation" not in variable_names:
                    variable_names += ["elevation"]
                dataset = dataset[variable_names]
                # remove coordinates except for latitude and longitude
                coords = []
                for coord in dataset.coords:
                    if coord not in ["latitude", "longitude"]:
                        coords.append(coord)
                dataset = dataset.drop_vars(coords)
                dataset["latitude"] = dataset["latitude"].persist()
                dataset["longitude"] = dataset["longitude"].persist()
                # orthorectify dataset
                dataset = self._apply_orthorectification(dataset, datatree)
                if includes is None or "elevation" not in includes:
                    dataset = dataset.drop_vars("elevation")
                # clip dataset by bbox
                if bbox:
                    bbox_wgs84 = reproject_bbox(bbox, crs, "EPSG:4326")
                    # Clip the dataset using the lat/lon grid rather than the STAC
                    # geometry. The STAC footprint represents the nadir view only and
                    # does not account for spatial extent in oblique acquisition.
                    dataset = clip_dataset_by_bbox(
                        dataset, bbox_wgs84, ("longitude", "latitude")
                    )
                    if any(size <= 1 for size in dataset.sizes.values()):
                        warnings.warn(
                            "Clipping with the specified bounding box "
                            "resulted in a dataset too small to compute a valid grid "
                            "mapping. Returning clipped dataset.",
                            UserWarning,
                        )
                        return dataset

                dataset_map[sub_group] = (dataset, GridMapping.from_dataset(dataset))
        if not dataset_map:
            raise ValueError("No variables selected")

        # get target grid mapping
        if bbox is None:
            bboxs = np.array([gm.xy_bbox for (_, gm) in dataset_map.values()])
            bbox = self._get_outer_bbox(bboxs)
        if resolution is None:
            subgroups_res_1000 = ["fnadir", "foblique", "inadir", "ioblique"]
            if all(key in subgroups_res_1000 for key in dataset_map.keys()):
                resolution = 1000
            else:
                resolution = 500
            if crs.is_geographic:
                center_lat = (bbox[1] + bbox[3]) / 2
                resolution = resolution_meters_to_degrees(resolution, center_lat)
        target_gm = GridMapping.regular_from_bbox(
            bbox=bbox, xy_res=resolution, crs=crs, tile_size=_CHUNKSIZE
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
        self,
        dataset: xr.Dataset,
        datatree: xr.DataTree,
        bbox_idx: tuple[int, int, int, int] = None,
    ) -> xr.Dataset:
        if any(str(var).endswith("o") for var in dataset.data_vars.keys()):
            angles = datatree.conditions.geometry_to.to_dataset()
            angles = angles[["sat_zenith_to", "sat_azimuth_to"]]
            angles = angles.rename(
                dict(sat_zenith_to="sat_zenith", sat_azimuth_to="sat_azimuth")
            )
        else:
            angles = datatree.conditions.geometry_tn.to_dataset()
            angles = angles[["sat_zenith_tn", "sat_azimuth_tn"]]
            angles = angles.rename(
                dict(sat_zenith_tn="sat_zenith", sat_azimuth_tn="sat_azimuth")
            )
        angles = angles.assign_coords(
            dict(
                latitude=datatree.conditions.meteorology.latitude,
                longitude=datatree.conditions.meteorology.longitude,
            )
        )
        return orthorectify_geolocation(dataset, angles)


def register(registry: AnalysisModeRegistry):
    registry.register(Sen3Ol1Err)
    registry.register(Sen3Ol1Efr)
    registry.register(Sen3Ol2Lfr)
    # registry.register(Sen3Ol2Lrr)
    registry.register(Sen3Sl1Rbt)
    registry.register(Sen3Sl2Lst)


def orthorectify_geolocation(dataset: xr.Dataset, angles: xr.Dataset) -> xr.Dataset:
    """
    Apply terrain-induced parallax correction to satellite geolocation coordinates.

    Args:
        dataset: Dataset containing geolocation coordinates to be corrected and
            surface elevation in meters above the reference ellipsoid or sphere. Must
            include `latitude` and `longitude` coordinates and `elevation` variable.
        angles: Dataset containing satellite viewing geometry angles. Must include the
            variables `sat_zenith`, `sat_azimuth` and the corresponding coordinates
            ``latitude` and `longitude`.

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

    # load coordinates of dataset
    ds_lat = dataset.latitude.values
    ds_lon = dataset.longitude.values
    # load coordinates of angles
    angles_lat = angles.latitude.values
    angles_lon = angles.longitude.values

    sat_zenith_interp = _interpolate(
        angles.sat_zenith.values, angles_lat, angles_lon, ds_lat, ds_lon
    )
    sat_azimuth_interp = _interpolate(
        angles.sat_azimuth.values, angles_lat, angles_lon, ds_lat, ds_lon
    )

    # Convert everything to rad
    phi_true = np.deg2rad(ds_lat)
    theta_v = np.deg2rad(sat_zenith_interp)
    phi_v = np.deg2rad(sat_azimuth_interp)

    # Horizontal displacement
    t = dataset.elevation.fillna(0).values * np.tan(theta_v)
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


def _clip_dataset_relative_bbox(
    rel_bbox: Sequence[float], ds: xr.Dataset, buffer: int | tuple[int, int] = 50
) -> tuple[xr.Dataset, tuple[int, int, int, int]]:
    if isinstance(buffer, int):
        buffer = (buffer, buffer)

    w, h = ds.sizes["columns"] - 1, ds.sizes["rows"] - 1
    col_min = int(np.clip((rel_bbox[0] * w) - buffer[0], 0, w))
    row_min = int(np.clip((rel_bbox[1] * h) - buffer[1], 0, h))
    col_max = int(np.clip((rel_bbox[2] * w) + buffer[0], 0, w))
    row_max = int(np.clip((rel_bbox[3] * h) + buffer[1], 0, h))

    ds_sub = ds.isel(rows=slice(row_min, row_max), columns=slice(col_min, col_max))

    return ds_sub, (col_min, row_min, col_max, row_max)
