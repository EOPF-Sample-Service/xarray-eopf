#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import atexit
import functools
import os
import re
import uuid
import warnings
from abc import ABC
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from typing import Any, Callable, Literal

import dask.array as da
import flox.xarray
import fsspec
import numpy as np
import pyproj
import pystac_client
import rioxarray
import xarray as xr
from xcube_resampling import resample_in_space
from xcube_resampling.constants import SpatialAggMethods, SpatialInterpMethods
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.rectify import rectify_dataset

# noinspection PyProtectedMember
from xcube_resampling.utils import (
    SourceTileIndexing,
    _reorganize_tiled_array,
    reproject_bbox,
    transform_resolution,
)

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.source import get_source_path
from xarray_eopf.utils import NameFilter, assert_arg_has_length, assert_arg_is_instance

_SPEED_OF_LIGHT = 299_792_458.0
_S_TO_NS = 10**9
_ONE_SECOND = np.timedelta64(_S_TO_NS, "ns")
_CRS_ECEF = pyproj.CRS.from_string("EPSG:4978")
_CRS_WGS84 = pyproj.CRS.from_string("EPSG:4326")
_DEM_CHUNKSIZE = dict(lat=1800, lon=1800)
_CHUNKSIZE = (2048, 2048)


@dataclass(frozen=True)
class GridParams:
    """RTC grid parameters."""

    gr0: float
    gr0_scale: float
    d_gr: float
    d_gr_scale: float
    az0: np.datetime64
    az0_scale: np.datetime64
    d_az: float
    d_az_scale: float
    spacing_az: float
    spacing_az_scale: float

    def __iter__(self):
        return (f.name for f in fields(self))

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __contains__(self, key):
        return key in {f.name for f in fields(self)}


class Sen1(AnalysisMode, ABC):

    def is_valid_source(self, source: Any) -> bool:
        root_path = get_source_path(source)
        pattern = re.compile(rf"S1[A-D]_[A-Z]{{2}}_{self.product_type}_[^/]+$")
        return bool(pattern.search(root_path)) if root_path else False

    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        warnings.warn(
            "Analysis mode not implemented for given source, "
            "returning data tree as-is."
        )
        return datatree

    def transform_dataset(
        self, dataset: xr.Dataset, stac_meta: dict, **params
    ) -> xr.Dataset:
        # ToDo: what should be added when opening a subgroup in analysis mode?
        return dataset

    def process_metadata(self, datatree: xr.DataTree) -> dict:
        return datatree.attrs


class Sen1GRD(Sen1):
    product_type = "GRDH"
    cache_fs: fsspec.AbstractFileSystem | None = None
    cache_uri: str | None = None
    _cleanup_registered: bool = False

    def get_applicable_params(self, **kwargs) -> dict[str, Any]:
        params = {}

        resolution = kwargs.get("resolution")
        if resolution is not None:
            assert_arg_is_instance(resolution, "resolution", (float, int))
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
            assert_arg_is_instance(
                interp_methods, "interp_methods", Literal["nearest", "bilinear"]
            )
            params.update(interp_methods=interp_methods)

        dem = kwargs.get("dem")
        if dem is not None:
            assert_arg_is_instance(dem, "dem", xr.DataArray)
            params.update(dem=dem)

        footprint_scale_factor = kwargs.get("footprint_scale_factor")
        if footprint_scale_factor is not None:
            assert_arg_is_instance(
                footprint_scale_factor,
                "footprint_scale_factor",
                tuple,
            )
            assert_arg_has_length(footprint_scale_factor, "footprint_scale_factor", 2)
            if not all(isinstance(v, (float, int)) for v in footprint_scale_factor):
                raise TypeError(
                    "footprint_scale_factor argument must contain exactly two "
                    "float or int values."
                )
            params.update(footprint_scale_factor=footprint_scale_factor)

        apply_rtc = kwargs.get("apply_rtc")
        if apply_rtc is not None:
            assert_arg_is_instance(apply_rtc, "apply_rtc", bool)
            params.update(apply_rtc=apply_rtc)

        cache_uri = kwargs.get("cache_uri")
        if cache_uri is not None:
            assert_arg_is_instance(cache_uri, "cache_uri", str)
            params.update(cache_uri=cache_uri)

        return params

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: float = None,
        bbox: Sequence[float | int] | None = None,
        crs: pyproj.CRS | None = None,
        interp_methods: Literal["nearest", "bilinear"] = "bilinear",
        footprint_scale_factor: tuple[float, float] = (3.0, 3.0),
        dem: xr.DataArray | None = None,
        apply_rtc: bool = True,
        cache_uri: str | None = None,
    ) -> xr.Dataset:

        if cache_uri is None:
            self.cache_fs = fsspec.filesystem("file")
            self.cache_uri = f"tmp_{uuid.uuid4().hex}"
        else:
            cache_uri = cache_uri.rstrip("/")
            self.cache_fs, _ = fsspec.url_to_fs(cache_uri)
            self.cache_uri = cache_uri
        if not getattr(self, "_cleanup_registered", False):
            atexit.register(self._cleanup)
            self._cleanup_registered = True

        # get dem data array
        if dem is None:
            if bbox is None:
                bbox = datatree.attrs["stac_discovery"]["bbox"]
                bbox = [
                    min(bbox[0], bbox[2]),
                    min(bbox[1], bbox[3]),
                    max(bbox[0], bbox[2]),
                    max(bbox[1], bbox[3]),
                ]
            dem = get_dem(bbox, resolution=resolution, crs=crs)

        # load measurement data
        grd = None
        group = ""
        for mode in ["VV", "VH", "HV", "HH"]:
            children = [x for x in datatree.children if mode in x]
            if children:
                group = children[0]
                if grd is None:
                    grd = datatree[group].measurements.to_dataset()
                    grd = grd.rename({"grd": mode.lower()})
                else:
                    grd[mode.lower()] = datatree[group].measurements.to_dataset().grd

        # filter dataset by variable names
        name_filter = NameFilter(includes=includes, excludes=excludes)
        variable_names = [k for k in grd.data_vars if name_filter.accept(str(k))]
        if not variable_names:
            raise ValueError("No valid variable names found in dataset")
        grd = grd[variable_names]

        # get calibration LUT data
        lut = datatree[group].quality.calibration.beta_nought
        lut_interp = lut.interp(ground_range=grd.ground_range).chunk(
            dict(ground_range=2048)
        )
        lut_interp = lut_interp.interp(azimuth_time=grd.azimuth_time).chunk(
            dict(azimuth_time=2048)
        )
        grd = (grd / lut_interp) ** 2
        rename_dict = {name: f"beta0_{name}" for name in variable_names}
        grd = grd.rename(rename_dict)
        for var in grd.data_vars:
            grd[var].attrs.update(
                long_name="beta nought backscatter coefficient",
                units="1",
            )

        orbit = datatree[f"{group}/conditions/orbit"].to_dataset()
        sat_position = orbit["position"].compute()

        gcp = datatree[f"{group}/conditions/gcp"].to_dataset()
        time_slr_gcp = gcp["slant_range_time_gcp"]

        grid_params = self._get_grid_parameters(datatree, footprint_scale_factor)

        try:
            return self._terrain_correct(
                grd,
                time_slr_gcp,
                sat_position,
                dem,
                grid_params,
                apply_rtc=apply_rtc,
                interp_method=interp_methods,
            )

        except Exception as _:
            self._cleanup()
            raise

    def _terrain_correct(
        self,
        data: xr.Dataset,
        time_slr_gcp: xr.DataArray,
        sat_position: xr.DataArray,
        dem: xr.DataArray,
        grid_params: GridParams | None = None,
        apply_rtc: bool = True,
        interp_method: Literal["nearest", "bilinear"] = "nearest",
    ) -> xr.Dataset:
        """Apply terrain correction to SAR data.

        Args:
            data: Input SAR dataset.
            time_slr_gcp: GCP slant-range times.
            sat_position: Satellite positions over time.
            dem: DEM for terrain correction.
            apply_rtc: Whether to apply radiometric terrain correction.
            grid_params: Grid parameters for RTC.
            interp_method: Interpolation method.

        Returns:
            Terrain-corrected dataset.

        Raises:
            ValueError: If RTC is enabled without grid parameters.
        """
        gm_dem = GridMapping.from_dataset(dem.to_dataset(name="dem"))
        src_loc = get_source_location(
            dem,
            time_slr_gcp,
            sat_position,
            grid_params,
            gm_dem,
            apply_rtc,
        )
        store = fsspec.get_mapper(f"{self.cache_uri}/src_location.zarr")
        src_loc.to_zarr(store)

        src_loc = xr.open_zarr(store)
        geocoded = geocode_data(data, src_loc, grid_params, interp_method)

        if apply_rtc:
            if interp_method == "bilinear":
                weights_fn = gamma_weights_bilinear
            else:  # interp_method == "nearest"
                weights_fn = gamma_weights_nearest
            gamma_weights = apply_gamma_weights(src_loc, weights_fn, grid_params)
            geocoded /= gamma_weights
            rename_dict = {
                name: str(name).replace("beta0", "gamma0")
                for name in geocoded.data_vars
            }
            geocoded = geocoded.rename(rename_dict)
            for var in geocoded.data_vars:
                geocoded[var].attrs.update(
                    long_name="gamma nought backscatter coefficient",
                    units="1",
                )

        geocoded = assign_grid_mapping(geocoded)
        return geocoded

    def _cleanup(self):
        if not self.cache_uri:
            return

        fs, path = fsspec.url_to_fs(self.cache_uri)
        if fs.exists(path):
            fs.rm(path, recursive=True)

    @staticmethod
    def _get_grid_parameters(
        datatree: xr.DataTree,
        footprint_scale_factor: tuple[float, float],
    ) -> GridParams:
        """Build grid parameters for RTC from Sentinel-1 metadata.

        Args:
            datatree: Source data tree.
            footprint_scale_factor: Scaling for SAR footprint spacing.

        Returns:
            Grid parameters for terrain correction.
        """

        group_vh = [x for x in datatree.children if "VH" in x][0]
        attrs = datatree[f"{group_vh}"].attrs["other_metadata"]["image_annotation"][
            "image_information"
        ]

        az_scale, gr_scale = footprint_scale_factor
        gr0 = 0.0
        d_gr = attrs["range_pixel_spacing"]
        az0 = np.datetime64(attrs["product_first_line_utc_time"])
        d_az = attrs["azimuth_time_interval"]
        spacing_az = attrs["azimuth_pixel_spacing"]

        return GridParams(
            gr0=gr0,
            d_gr=d_gr,
            d_gr_scale=d_gr * gr_scale,
            gr0_scale=(gr0 - (0.5 * d_gr) + (0.5 * d_gr * gr_scale)),
            az0=az0,
            d_az=d_az,
            spacing_az=spacing_az,
            d_az_scale=d_az * az_scale,
            az0_scale=(az0 + (-(0.5 * d_az) + (0.5 * d_az * az_scale)) * _ONE_SECOND),
            spacing_az_scale=spacing_az * az_scale,
        )


class Sen1OCN(Sen1):
    product_type = "OCN"

    def get_applicable_params(self, **kwargs) -> dict[str, Any]:
        params = {}

        resolution = kwargs.get("resolution")
        if resolution is not None:
            assert_arg_is_instance(resolution, "resolution", (float, int))
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
            assert_arg_is_instance(
                interp_methods, "interp_methods", Literal["nearest", "bilinear"]
            )
            params.update(interp_methods=interp_methods)

        agg_methods = kwargs.get("agg_methods")
        if agg_methods is not None:
            assert_arg_is_instance(agg_methods, "agg_methods", (str, dict))
            params.update(agg_methods=agg_methods)

        return params

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: float = None,
        bbox: Sequence[float | int] | None = None,
        crs: pyproj.CRS | None = None,
        interp_methods: SpatialInterpMethods | None = None,
        agg_methods: SpatialAggMethods | None = None,
    ) -> xr.Dataset:
        # load measurement data
        assert (
            len(datatree.owi.children) == 1
        ), "Expected one child in OCN OWI sub data tree"
        sub_dt = next(iter(datatree.owi.children.values()))
        dataset = sub_dt.measurements.to_dataset()
        dataset.update(sub_dt.quality.to_dataset().drop_vars("calibration_constant"))

        # correct attributes and encoding
        def _apply_valid_range(array, *, dtype=None, fill_value=None):
            if dtype is not None:
                array = array.astype(dtype)

            if fill_value is not None:
                array.encoding["_FillValue"] = fill_value

            eopf_attrs = array.attrs["_eopf_attrs"]
            array.attrs.update(
                valid_min=eopf_attrs["valid_min"],
                valid_max=eopf_attrs["valid_max"],
            )

            return array

        dataset["inversion_quality"] = _apply_valid_range(
            dataset.inversion_quality,
            dtype="uint8",
            fill_value=255,
        )
        dataset["wind_quality"] = _apply_valid_range(
            dataset.wind_quality,
            dtype="uint8",
            fill_value=255,
        )
        dataset["percentage_bright_points"] = _apply_valid_range(
            dataset.percentage_bright_points,
        )

        # filter dataset by variable names
        name_filter = NameFilter(includes=includes, excludes=excludes)
        variable_names = [k for k in dataset.data_vars if name_filter.accept(str(k))]
        if not variable_names:
            raise ValueError("No valid variable names found in dataset")
        dataset = dataset[variable_names]

        # reproject dataset to regular grid
        source_gm = GridMapping.from_dataset(dataset)
        if bbox is None:
            if crs:
                bbox = reproject_bbox(source_gm.xy_bbox, source_gm.crs, crs)
            else:
                bbox = source_gm.xy_bbox
        if resolution is None:
            if crs and not crs.is_geographic:
                center_lat = (
                    (source_gm.xy_bbox[0] + source_gm.xy_bbox[2]) / 2,
                    (source_gm.xy_bbox[1] + source_gm.xy_bbox[3]) / 2,
                )
                resolution = transform_resolution(
                    center_lat, source_gm.xy_res, source_gm.crs, crs
                )
            else:
                resolution = source_gm.xy_res
        if crs is None:
            crs = source_gm.crs
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


def register(registry: AnalysisModeRegistry):
    """Register Sentinel-1 analysis modes."""
    registry.register(Sen1GRD)
    registry.register(Sen1OCN)


def get_dem(
    bbox: Sequence[float | int],
    resolution: float | None = None,
    crs: pyproj.CRS | None = None,
):
    """Fetch and prepare a DEM for the given area of interest.

    Args:
        bbox: Spatial bounding box.
        resolution: Target resolution for resampling.
        crs: Target coordinate reference system.

    Returns:
        DEM data array.

    Raises:
        ValueError: If required credentials are missing or resolution is invalid.
    """
    # check that environment variables are set
    missing = [
        name
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
        if not os.environ.get(name)
    ]
    if missing:
        raise ValueError(
            f"Missing AWS credentials for DEM download: {missing}."
            "Set these environment variables for CDSE DEM access "
            "(https://documentation.dataspace.copernicus.eu/APIs/S3.html#generate-secrets) "
            "or provide a DEM directly."
        )
    os.environ.update(
        {
            "AWS_S3_ENDPOINT": "eodata.dataspace.copernicus.eu",
            "AWS_VIRTUAL_HOSTING": "FALSE",
        }
    )

    # make opening parameters applicable
    if crs is not None and not crs.is_geographic:
        bbox_wgs84 = reproject_bbox(bbox, crs, _CRS_WGS84)
    else:
        bbox_wgs84 = bbox

    # get STAC items
    client = pystac_client.Client.open("https://stac.dataspace.copernicus.eu/v1")
    search = client.search(
        collections=["cop-dem-glo-30-dged-cog"],
        bbox=list(bbox_wgs84),
    )
    items = list(search.items())

    # open tiles and combine to one DataArray
    das = []
    for item in items:
        das.append(rioxarray.open_rasterio(item.assets["data"].href, chunks={}))
    dem = xr.combine_by_coords(das, join="outer", fill_value=0.0).sel(band=1, drop=True)
    dem = dem.rename({"x": "lon", "y": "lat"})

    if crs is None and resolution is None:
        dem = dem.sel(
            lat=slice(bbox_wgs84[3], bbox_wgs84[1]),
            lon=slice(bbox_wgs84[0], bbox_wgs84[2]),
        ).chunk(_DEM_CHUNKSIZE)
    else:
        if resolution is None:
            raise ValueError("Resolution must be provided if CRS is not None.")
        if crs is None:
            crs = _CRS_WGS84
        target_gm = GridMapping.regular_from_bbox(
            bbox,
            resolution,
            crs,
            tile_size=(_DEM_CHUNKSIZE["lat"], _DEM_CHUNKSIZE["lon"]),
        )
        dem = resample_in_space(dem.to_dataset(name="dem"), target_gm=target_gm).dem

    return dem


def convert_dem_to_ecef(dem: xr.DataArray, gm_dem_params: dict) -> xr.DataArray:
    """Convert a DEM from its native CRS to ECEF coordinates.

    Args:
        dem: DEM data array.
        gm_dem_params: GridMapping metadata of the DEM data array.

    Returns:
        DEM expressed in ECEF axes.
    """

    x_dim, y_dim = gm_dem_params["xy_var_names"]
    xx, yy = np.meshgrid(dem[x_dim].values, dem[y_dim].values, indexing="xy")

    transformer = pyproj.Transformer.from_crs(
        gm_dem_params["crs"], _CRS_ECEF, always_xy=True
    )
    x, y, z = transformer.transform(xx, yy, dem.values)

    return xr.DataArray(
        np.stack([x, y, z], axis=0),
        dims=("axis", y_dim, x_dim),
        coords={
            y_dim: dem[y_dim].data,
            x_dim: dem[x_dim].data,
            "axis": ["x", "y", "z"],
        },
    )


def az_to_orbit(time_az: xr.DataArray, epoch: np.datetime64) -> xr.DataArray:
    """Convert azimuth time to orbit time coordinates.

    Args:
        time_az: Azimuth time coordinate.
        epoch: Reference epoch.

    Returns:
        Orbit time coordinate.
    """
    return (time_az - epoch) / np.timedelta64(_S_TO_NS, "ns")


def orbit_to_az(time_orb: xr.DataArray, epoch: np.datetime64) -> xr.DataArray:
    """Convert orbit time coordinates to azimuth time.

    Args:
        time_orb: Orbit time coordinate.
        epoch: Reference epoch.

    Returns:
        Azimuth time coordinate.
    """
    return time_orb * np.timedelta64(_S_TO_NS, "ns") + epoch


def fit_position(pos: xr.DataArray, time_dim="azimuth_time", deg=5) -> xr.DataArray:
    """Fit a polynomial position model along the time axis.

    Args:
        pos: Satellite position array.
        time_dim: Name of the time dimension.
        deg: Polynomial degree.

    Returns:
        Polynomial coefficients with an epoch attribute.
    """
    time = pos.coords[time_dim]
    epoch = time.values[0] + (time.values[-1] - time.values[0]) / 2

    time_orbit = az_to_orbit(time, epoch)

    pos = pos.assign_coords({time_dim: time_orbit})
    coeff = pos.polyfit(dim=time_dim, deg=deg).polyfit_coefficients

    coeff.attrs["epoch"] = epoch
    return coeff


def poly_derivative(coeff: xr.DataArray) -> xr.DataArray:
    """Compute the derivative coefficients of a polynomial fit.

    Args:
        coeff: Polynomial coefficients.

    Returns:
        Polynomial coefficients for the derivative.
    """
    out = coeff.isel(degree=slice(1, None)).copy()
    for deg in coeff.degree.values[:-1]:
        out.loc[{"degree": deg - 1}] = coeff.sel(degree=deg) * deg
    return out


def zero_doppler(
    dem_ecef: xr.DataArray,
    pos_coeff: xr.DataArray,
    vel_coeff: xr.DataArray,
    time_orbit: xr.DataArray,
    dim: str = "axis",
) -> tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray]]:
    """Evaluate the zero-Doppler equation and its payload.

    Args:
        dem_ecef: DEM in ECEF coordinates.
        pos_coeff: Position polynomial coefficients.
        vel_coeff: Velocity polynomial coefficients.
        time_orbit: Orbit time coordinate.
        dim: Axis dimension name.

    Returns:
        Zero-Doppler function value and payload (distance, velocity).
    """
    sat = xr.polyval(time_orbit, pos_coeff)
    dist = dem_ecef - sat
    vel = xr.polyval(time_orbit, vel_coeff)

    func = (dist * vel).sum(dim)
    return func, (dist, vel)


def zero_doppler_prime(
    vel_coeff: xr.DataArray,
    time_orbit: xr.DataArray,
    payload: tuple[xr.DataArray, xr.DataArray],
    dim: str = "axis",
) -> xr.DataArray:
    """Compute the derivative of the zero-Doppler function.

    Args:
        vel_coeff: Velocity polynomial coefficients.
        time_orbit: Orbit time coordinate.
        payload: Payload from zero-Doppler evaluation.
        dim: Axis dimension name.

    Returns:
        Derivative of the zero-Doppler function.
    """
    dist, vel = payload
    accel = xr.polyval(time_orbit, poly_derivative(vel_coeff))

    fprime = (dist * accel - vel**2).sum(dim)
    return fprime


def secant(
    func: Callable[[xr.DataArray], tuple[xr.DataArray, Any]],
    t0: xr.DataArray,
    t1: xr.DataArray,
    tol_f: float = 1.0,
    tol_t: float = 1e-6,
    maxiter: int = 10,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, int, Any]:
    """Solve for a root using the secant method.

    Args:
        func: Function returning value and payload.
        t0: Initial time guess.
        t1: Second time guess.
        tol_f: Function tolerance.
        tol_t: Time tolerance.
        maxiter: Maximum number of iterations.

    Returns:
        Updated time, previous time, function value, iteration count, payload.
    """
    f0, payload = func(t0)

    f1, k = None, None
    for k in range(maxiter):
        f1, payload = func(t1)

        if not np.any(np.abs(f1) > tol_f):
            break

        dt = t1 - t0
        if not np.any(np.abs(dt) > tol_t):
            break

        q = f1 - f0

        t0, t1 = t1, t1 - np.where(q != 0, f1 / q, 0) * dt
        f0 = f1

    return t1, t0, f1, k, payload


def newton(
    func: Callable[[xr.DataArray], tuple[xr.DataArray, Any]],
    func_p: Callable[[xr.DataArray, Any], xr.DataArray],
    t: xr.DataArray,
    tol_f: float = 1.0,
    tol_t: float = 1e-6,
    maxiter: int = 10,
) -> tuple[xr.DataArray, xr.DataArray, int, Any]:
    """Solve for a root using Newton's method.

    Args:
        func: Function returning value and payload.
        func_p: Derivative function.
        t: Initial time guess.
        tol_f: Function tolerance.
        tol_t: Time tolerance.
        maxiter: Maximum number of iterations.

    Returns:
        Updated time, function value, iteration count, payload.
    """
    f, k, payload = None, None, None
    for k in range(maxiter):
        f, payload = func(t)

        if not np.any(np.abs(f) > tol_f):
            break

        fp = func_p(t, payload)
        dt = f / fp

        if not np.any(np.abs(dt) > tol_t):
            break

        t = t - dt

    return t, f, k, payload


def backward_geocode(
    dem: xr.DataArray,
    pos_coeff: xr.DataArray = None,
    vel_coeff: xr.DataArray = None,
    gr_coeff: xr.DataArray = None,
    grid_params: GridParams = None,
    apply_rtc: bool = True,
    gm_dem_params: dict = None,
    method="newton",
    tol=1.0,
    speed=7500.0,
    maxiter=10,
    t_shift=-0.1,
) -> xr.Dataset:
    """Compute orbit time and vectors for a DEM using inverse geocoding.

    Args:
        dem: Digital elevation model.
        pos_coeff: Position polynomial coefficients.
        vel_coeff: Velocity polynomial coefficients.
        gr_coeff: Ground-range polynomial coefficients.
        grid_params: Grid parameters for RTC.
        apply_rtc: Whether to compute RTC gamma area.
        gm_dem_params: DEM grid metadata for ECEF conversion.
        method: Root-finding method.
        tol: Function tolerance.
        speed: Nominal platform speed for tolerance scaling.
        maxiter: Maximum number of iterations.
        t_shift: Time shift for the secant method.

    Returns:
        A dataset containging the optimized ground_range and azimuth time
        for each target pixel and optinal the gamma area needed for RTC.

    Raises:
        ValueError: If the method is not supported.
    """
    assert pos_coeff is not None
    assert vel_coeff is not None
    assert gr_coeff is not None
    assert grid_params is not None
    assert gm_dem_params is not None

    dem_ecef = convert_dem_to_ecef(dem, gm_dem_params)

    f = functools.partial(zero_doppler, dem_ecef, pos_coeff, vel_coeff)

    t0 = xr.zeros_like(dem_ecef.sel(axis="x", drop=True), dtype="float64")
    t1 = t0 + t_shift

    if method == "secant":
        time_orbit, _, _, _, payload = secant(
            f, t1, t0, tol_f=tol * speed, maxiter=maxiter
        )
    elif method == "newton":
        fp = functools.partial(zero_doppler_prime, vel_coeff)
        time_orbit, _, _, payload = newton(
            f, fp, t0, tol_f=tol * speed, maxiter=maxiter
        )
    else:
        raise ValueError("method needs to be either 'secant' or 'newton'")

    dist, _ = payload

    # apply bistatic correction
    slant_range = np.sqrt((dist**2).sum("axis"))
    time_orbit += slant_range / _SPEED_OF_LIGHT

    # recalculate slant range
    sat = xr.polyval(time_orbit, pos_coeff)
    dist = dem_ecef - sat
    slant_range = np.sqrt((dist**2).sum("axis"))
    time_slr = 2 * slant_range / _SPEED_OF_LIGHT

    # convert to ground range
    azimuth_time = orbit_to_az(time_orbit, pos_coeff.attrs["epoch"])
    ground_range = get_ground_range(gr_coeff, azimuth_time, time_slr)
    out = xr.Dataset({"azimuth_time": azimuth_time, "ground_range": ground_range})
    if apply_rtc:
        out["gamma_area"] = compute_gamma_area(
            dem_ecef, gm_dem_params, dist / slant_range
        )
    return out


def compute_dem_area(dem_ecef: xr.DataArray, gm_dem_params: dict) -> xr.DataArray:
    """Compute per-pixel surface area on the DEM in ECEF coordinates.

    Args:
        dem_ecef: DEM in ECEF coordinates.
        gm_dem_params: GridMapping metadata of the DEM data array.

    Returns:
        Area vectors per DEM pixel.
    """
    x_dim, y_dim = gm_dem_params["xy_var_names"]
    x = dem_ecef[x_dim]
    y = dem_ecef[y_dim]

    # construct corner coordinates
    x_corner = np.concatenate(
        [
            [x[0] + (x[0] - x[1]) / 2],
            ((x[:-1].data + x[1:].data) / 2),
            [x[-1] + (x[-1] - x[-2]) / 2],
        ]
    )

    y_corner = np.concatenate(
        [
            [y[0] + (y[0] - y[1]) / 2],
            ((y[:-1].data + y[1:].data) / 2),
            [y[-1] + (y[-1] - y[-2]) / 2],
        ]
    )

    # interpolate DEM to pixel corners
    xyz_c = dem_ecef.interp(
        {x_dim: x_corner, y_dim: y_corner},
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # compute edge vectors
    dx = xyz_c.diff(x_dim)
    dy = xyz_c.diff(y_dim)

    # align shapes for two triangles
    dx1 = dx.isel({y_dim: slice(1, None)})
    dy1 = dy.isel({x_dim: slice(1, None)})
    dx2 = dx.isel({y_dim: slice(None, -1)})
    dy2 = dy.isel({x_dim: slice(None, -1)})

    # restore original coords
    dx1 = dx1.assign_coords(dem_ecef.coords)
    dy1 = dy1.assign_coords(dem_ecef.coords)
    dx2 = dx2.assign_coords(dem_ecef.coords)
    dy2 = dy2.assign_coords(dem_ecef.coords)

    # compute triangle areas
    cross1 = xr.cross(dx1, dy1, dim="axis") / 2
    cross2 = xr.cross(dx2, dy2, dim="axis") / 2

    # ensure outward normal direction
    sign1 = np.sign(xr.dot(cross1, dem_ecef, dim="axis"))
    sign2 = np.sign(xr.dot(cross2, dem_ecef, dim="axis"))

    return cross1 * sign1 + cross2 * sign2


def compute_gamma_area(
    dem_ecef: xr.DataArray,
    gm_dem_params: dict,
    direction: xr.DataArray,
) -> xr.DataArray:
    """Compute gamma area by projecting DEM areas onto look direction.

    Args:
        dem_ecef: DEM in ECEF coordinates.
        gm_dem_params: GridMapping metadata of the DEM data array.
        direction: Look direction vectors.

    Returns:
        Gamma area for each DEM pixel.
    """
    area = compute_dem_area(dem_ecef, gm_dem_params)
    gamma_area = xr.dot(area, -direction, dim="axis")
    return gamma_area.where(gamma_area > 0, 0)


def sum_weights(
    weights: xr.DataArray,
    az_idx: xr.DataArray,
    gr_idx: xr.DataArray,
) -> xr.DataArray:
    """Accumulate weights into the SAR image grid.

    Args:
        weights: Weights to accumulate.
        az_idx: Azimuth indices.
        gr_idx: Ground-range indices.

    Returns:
        Accumulated weights on the SAR grid.
    """
    reduced = flox.xarray.xarray_reduce(
        weights,
        gr_idx,
        az_idx,
        func="sum",
        method="map-reduce",
    )

    return reduced.interp(
        gr_idx=gr_idx,
        az_idx=az_idx,
        method="nearest",
    ).drop_vars(("az_idx", "gr_idx"))


def gamma_weights_bilinear(src_loc: xr.Dataset) -> xr.DataArray:
    """Compute bilinear gamma weights for the acquisition grid.

    Args:
        src_loc: Source location dataset with indices and gamma area.

    Returns:
        Gamma weights on the SAR grid.
    """
    az_idx = src_loc.az_idx
    gr_idx = src_loc.gr_idx

    az0 = np.floor(az_idx).astype(np.intp)
    az1 = np.ceil(az_idx).astype(np.intp)
    gr0 = np.floor(gr_idx).astype(np.intp)
    gr1 = np.ceil(gr_idx).astype(np.intp)

    w00 = abs((az1 - az_idx) * (gr1 - gr_idx))
    w01 = abs((az1 - az_idx) * (gr0 - gr_idx))
    w10 = abs((az0 - az_idx) * (gr1 - gr_idx))
    w11 = abs((az0 - az_idx) * (gr0 - gr_idx))

    gamma = src_loc.gamma_area
    return (
        sum_weights(gamma * w00, az0, gr0)
        + sum_weights(gamma * w01, az0, gr1)
        + sum_weights(gamma * w10, az1, gr0)
        + sum_weights(gamma * w11, az1, gr1)
    )


def gamma_weights_nearest(src_loc: xr.Dataset) -> xr.DataArray:
    """Compute nearest-neighbor gamma weights for the acquisition grid.

    Args:
        src_loc: Source location dataset with indices and gamma area.

    Returns:
        Gamma weights on the SAR grid.
    """

    az_idx = np.round(src_loc.az_idx).astype(np.intp)
    gr_idx = np.round(src_loc.gr_idx).astype(np.intp)
    return sum_weights(src_loc.gamma_area, az_idx, gr_idx)


def apply_gamma_weights(
    src_loc: xr.Dataset,
    func: Callable[..., xr.DataArray],
    params: GridParams,
) -> xr.DataArray:
    """Apply gamma weighting block-wise.

    Args:
        src_loc: Source location dataset with geometry.
        func: Weighting function.
        params: Grid parameters for index conversion.

    Returns:
        Gamma-corrected area per pixel.
    """
    src_loc["az_idx"] = (
        (src_loc.azimuth_time - params.az0_scale) / _ONE_SECOND / params.d_az_scale
    )
    src_loc["gr_idx"] = (src_loc.ground_range - params.gr0_scale) / params.d_gr_scale

    template = src_loc.gamma_area * 0
    area = xr.map_blocks(func, src_loc, template=template)

    return area / (params.d_gr_scale * params.spacing_az_scale)


def fit_ground_range(time_slr_gcp: xr.DataArray, deg: int = 8) -> xr.DataArray:
    """Fit ground-range polynomials from GCP slant-range times.

    Args:
        time_slr_gcp: GCP slant-range times.
        deg: Polynomial degree.

    Returns:
        Polynomial coefficients per azimuth line.
    """
    # normalization for stability
    mean = time_slr_gcp.mean().values
    std = time_slr_gcp.std().values
    x_gcp = (time_slr_gcp - mean) / std

    # polynomial fit per azimuth line
    coeff = []
    for i, time in enumerate(x_gcp["azimuth_time"].data):
        coeff.append(np.polyfit(x_gcp[i, :], x_gcp["ground_range"], deg=deg))
    return xr.DataArray(
        coeff,
        coords=dict(azimuth_time=x_gcp["azimuth_time"], degree=np.arange(deg, -1, -1)),
        dims=("azimuth_time", "degree"),
        attrs=dict(mean=mean, std=std),
    )


def get_ground_range(
    coeff: xr.DataArray, time_az: xr.DataArray, time_slr: xr.DataArray
) -> xr.DataArray:
    coeff_interp = coeff.interp(azimuth_time=time_az).drop_vars("azimuth_time")
    x_tgt = (time_slr - coeff.attrs["mean"]) / coeff.attrs["std"]
    return (coeff_interp * x_tgt**coeff.degree).sum("degree")


def geocode_data(
    data: xr.Dataset,
    src_loc: xr.Dataset,
    grid_params: GridParams,
    interp_method: Literal["nearest", "bilinear"],
) -> xr.Dataset:
    """Geocode data from SAR grid to map coordinates.

    Args:
        data: Input dataset on the SAR grid.
        src_loc: Source location dataset with target coordinates.
        grid_params: Grid parameters for index conversion.
        interp_method: Interpolation method.

    Returns:
        Geocoded dataset.
    """
    az_idx = (src_loc.azimuth_time - grid_params.az0) / _ONE_SECOND / grid_params.d_az
    gr_idx = (src_loc.ground_range - grid_params.gr0) / grid_params.d_gr
    scr_indexing = _compute_indexing(data, az_idx, gr_idx)
    temp_ij_bboxes = scr_indexing.ij_bboxes.copy()
    temp_ij_bboxes[[1, 3]] -= scr_indexing.pad_width[0][0]
    temp_ij_bboxes[[0, 2]] -= scr_indexing.pad_width[1][0]
    tile_size = tuple(chunk[0] for chunk in gr_idx.chunks)
    for j in range(temp_ij_bboxes.shape[1]):
        for i in range(temp_ij_bboxes.shape[2]):
            i_min = tile_size[1] * i
            i_max = tile_size[1] * (i + 1)
            j_min = tile_size[0] * j
            j_max = tile_size[0] * (j + 1)
            gr_idx[j_min:j_max, i_min:i_max] -= temp_ij_bboxes[0, j, i]
            az_idx[j_min:j_max, i_min:i_max] -= temp_ij_bboxes[1, j, i]

    target_ds = xr.Dataset(coords=az_idx.coords)
    for var_name, data_array in data.items():
        tiled = _reorganize_tiled_array(data_array.data, scr_indexing, np.nan)
        resampled = da.map_blocks(
            _sample_array_at_indices,
            tiled,
            gr_idx.data,
            az_idx.data,
            interp_method=interp_method,
            dtype=data_array.dtype,
            chunks=gr_idx.data.chunks,
        )
        target_ds[var_name] = (az_idx.dims, resampled)

    return target_ds


def get_source_location(
    dem: xr.DataArray,
    time_slr_gcp: xr.DataArray,
    sat_position: xr.DataArray,
    grid_params: GridParams,
    gm_dem_grid: GridMapping,
    apply_rtc: bool,
) -> xr.Dataset:

    # get polynomial coefficients to convert from slant range to ground range
    gr_coeff = fit_ground_range(time_slr_gcp)

    # get polynomial coefficient to convert from azimuth time to position and velocity
    pos_coeff = fit_position(sat_position)
    vel_coeff = poly_derivative(pos_coeff)

    data_array = xr.zeros_like(dem, dtype="float32")
    azimuth_data = xr.zeros_like(dem, dtype="datetime64[ns]")
    template = xr.Dataset(
        {"azimuth_time": azimuth_data, "ground_range": data_array},
    )
    if apply_rtc:
        template["gamma_area"] = data_array
    if "spatial_ref" in template:
        template = template.drop_vars("spatial_ref")
    gm_dem_params = {
        "crs": gm_dem_grid.crs.to_wkt(),
        "xy_var_names": gm_dem_grid.xy_var_names,
    }
    out = xr.map_blocks(
        backward_geocode,
        dem,
        kwargs={
            "pos_coeff": pos_coeff,
            "vel_coeff": vel_coeff,
            "gr_coeff": gr_coeff,
            "grid_params": grid_params,
            "apply_rtc": apply_rtc,
            "gm_dem_params": gm_dem_params,
        },
        template=template,
    )
    out.coords["spatial_ref"] = xr.DataArray(0, attrs=gm_dem_grid.crs.to_cf())
    return out


def assign_grid_mapping(dataset: xr.Dataset) -> xr.Dataset:
    for var_name, data_var in dataset.data_vars.items():
        dataset[var_name].attrs["grid_mapping"] = "spatial_ref"
    return dataset


# INTERPOLATION -> xcube-resampling?
def _xy_bbox_block(x_coords: np.ndarray, y_coords: np.ndarray):
    x_edges = np.concatenate([x_coords[:, 0], x_coords[:, -1]])
    y_edges = np.concatenate([y_coords[0, :], y_coords[-1, :]])
    bbox = np.array(
        [
            np.floor(x_edges.min()),
            np.floor(y_edges.min()),
            np.ceil(x_edges.max()),
            np.ceil(y_edges.max()),
        ],
        dtype=np.int32,
    )
    return bbox[:, None, None]


def _compute_indexing(
    data: xr.Dataset,
    az_ix: xr.DataArray,
    gr_idx: xr.DataArray,
) -> SourceTileIndexing:

    src_ij_bboxes = da.map_blocks(
        _xy_bbox_block,
        gr_idx.data,
        az_ix.data,
        dtype=gr_idx.dtype,
        chunks=(4, 1, 1),
    )
    src_ij_bboxes = src_ij_bboxes.compute()

    # Extend bounding box indices to match the largest bounding box.
    # This ensures uniform chunk sizes, which are required for da.map_blocks.
    i_diff = src_ij_bboxes[2] - src_ij_bboxes[0]
    j_diff = src_ij_bboxes[3] - src_ij_bboxes[1]
    i_diff_max = np.nanmax(i_diff) + 1
    j_diff_max = np.nanmax(j_diff) + 1
    i_half = (i_diff_max - i_diff) // 2
    j_half = (j_diff_max - j_diff) // 2
    src_ij_bboxes[0] -= i_half
    src_ij_bboxes[2] = src_ij_bboxes[0] + i_diff_max
    src_ij_bboxes[1] -= j_half
    src_ij_bboxes[3] = src_ij_bboxes[1] + j_diff_max

    # assign padding if needed
    i_min = np.nanmin(src_ij_bboxes[0])
    i_max = np.nanmax(src_ij_bboxes[2])
    j_min = np.nanmin(src_ij_bboxes[[1, 3]])
    j_max = np.nanmax(src_ij_bboxes[[1, 3]])
    pad_width = (
        (-min(0, int(j_min)), max(0, int(j_max - data.sizes["azimuth_time"]))),
        (-min(0, int(i_min)), max(0, int(i_max - data.sizes["ground_range"]))),
    )
    src_ij_bboxes[[1, 3]] += pad_width[0][0]
    src_ij_bboxes[[0, 2]] += pad_width[1][0]

    tile_size = (int(j_diff_max), int(i_diff_max))
    size = (
        int(j_diff_max * src_ij_bboxes.shape[1]),
        int(i_diff_max * src_ij_bboxes.shape[2]),
    )

    return SourceTileIndexing(
        ij_bboxes=src_ij_bboxes,
        pad_width=pad_width,
        output_size=size,
        tile_size=tile_size,
    )


def _sample_array_at_indices(
    data: np.ndarray,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    interp_method: Literal["nearest", "bilinear"] | None = None,
) -> np.ndarray:
    """Sample a 3D array at fractional indices (y_idx, x_idx)."""
    if interp_method == "nearest":
        x_i = np.ceil(x_idx - 0.5).astype(np.intp)
        y_i = np.ceil(y_idx - 0.5).astype(np.intp)
        return data[y_i, x_i]

    x_floor = np.floor(x_idx).astype(np.intp)
    y_floor = np.floor(y_idx).astype(np.intp)
    x_ceil = np.ceil(x_idx).astype(np.intp)
    y_ceil = np.ceil(y_idx).astype(np.intp)

    dx = x_idx - x_floor
    dy = y_idx - y_floor

    v00 = data[y_floor, x_floor]
    v01 = data[y_floor, x_ceil]
    v10 = data[y_ceil, x_floor]
    v11 = data[y_ceil, x_ceil]

    if interp_method == "bilinear":
        u0 = v00 + dx * (v01 - v00)
        u1 = v10 + dx * (v11 - v10)
        return u0 + dy * (u1 - u0)

    raise NotImplementedError(
        f"interp_methods must be one of 'nearest', 'bilinear', "
        f"was '{interp_method}'."
    )
