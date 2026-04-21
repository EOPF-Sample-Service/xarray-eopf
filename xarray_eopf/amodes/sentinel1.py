#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.


import functools
import os
import re
import warnings
from abc import ABC
from collections.abc import Iterable, Sequence
from typing import Any, Callable, Literal

import dask.array as da
import flox.xarray
import numpy as np
import pyproj
import pystac_client
import rioxarray
import xarray as xr
from xcube_resampling import resample_in_space
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.utils import reproject_bbox

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.source import get_source_path
from xarray_eopf.utils import assert_arg_has_length, assert_arg_is_instance
from xarray_eopf.utils import NameFilter

_SPEED_OF_LIGHT = 299_792_458.0
_S_TO_NS = 10**9
_ONE_SECOND = np.timedelta64(_S_TO_NS, "ns")
_CRS_ECEF = pyproj.CRS.from_string("EPSG:4978")
_CRS_WGS84 = pyproj.CRS.from_string("EPSG:4326")
_DEM_CHUNKSIZE = dict(lat=3600, lon=3600)


class Sen1(AnalysisMode, ABC):

    def is_valid_source(self, source: Any) -> bool:
        root_path = get_source_path(source)
        pattern = re.compile(rf"S1[A-D]_[A-Z]{{2}}_{self.product_type}_[^/]+$")
        return bool(pattern.search(root_path))

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
                (tuple[float | int, float | int]),
            )
            params.update(footprint_scale_factor=footprint_scale_factor)

        apply_rtc = kwargs.get("apply_rtc")
        if apply_rtc is not None:
            assert_arg_is_instance(apply_rtc, "apply_rtc", bool)
            params.update(apply_rtc=apply_rtc)

        return params

    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        warnings.warn(
            "Analysis mode not implemented for given source, return data tree as-is."
        )
        return datatree

    def transform_dataset(
        self, dataset: xr.Dataset, stac_meta: dict, **params
    ) -> xr.Dataset:
        # ToDo: what should be added when opening a subgroup in analysis mode?
        return dataset

    def process_metadata(self, datatree: xr.DataTree) -> dict:
        other_metadata = datatree.attrs.get("other_metadata", {})
        return other_metadata


class Sen1GRD(Sen1):
    product_type = "GRDH"

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: float = None,
        bbox: Sequence[float | int] | None = None,
        crs: pyproj.CRS | None = None,
        interp_methods: Literal["nearest", "bilinear"] = "nearest",
        footprint_scale_factor: tuple[float, float] = (3.0, 3.0),
        dem: xr.Dataset | None = None,
        apply_rtc: bool = True,
    ) -> xr.Dataset:

        # ToDo filter variable names
        # ToDo allow for different polarization combinations

        if dem is None:
            if bbox is None:
                bbox = datatree.attrs["stac_discovery"]["bbox"]
            dem = get_dem(bbox, resolution=resolution, crs=crs)

        group_vh = [x for x in datatree.children if "VH" in x][0]
        grd = datatree[group_vh].measurements.to_dataset().rename({"grd": "vh"})
        group_vv = [x for x in datatree.children if "VV" in x][0]
        grd["vv"] = datatree[group_vv].measurements.to_dataset().grd
        beta_lut = datatree[group_vh].quality.calibration.to_dataset()["beta_nought"]
        beta_lut_interp = beta_lut.interp(ground_range=grd.ground_range).chunk(
            dict(ground_range=2048)
        )
        beta_lut_interp = beta_lut_interp.interp(azimuth_time=grd.azimuth_time).chunk(
            dict(azimuth_time=2048)
        )
        beta_nought = (grd / beta_lut_interp) ** 2
        beta_nought.assign_attrs(long_name="beta nought", units="m2 m-2")

        orbit = datatree[f"{group_vh}/conditions/orbit"].to_dataset()
        sat_position = orbit["position"]

        gcp = datatree[f"{group_vh}/conditions/gcp"].to_dataset()
        time_slr_gcp = gcp["slant_range_time_gcp"]

        grid_params = self._get_grid_parameters(datatree, footprint_scale_factor)

        return terrain_correct(
            beta_nought,
            time_slr_gcp,
            sat_position,
            dem,
            grid_params=grid_params,
            apply_rtc=apply_rtc,
            interp_method=interp_methods,
        )

    @staticmethod
    def _get_grid_parameters(
        dt: xr.DataTree,
        footprint_scale_factor: tuple[float, float],
    ) -> dict[str, Any]:
        """Build grid parameters for RTC from Sentinel-1 metadata.

        Args:
            dt: Source data tree.
            footprint_scale_factor: Scaling for SAR footprint spacing.

        Returns:
            Dictionary of grid parameters for terrain correction.
        """

        group_VH = [x for x in dt.children if "VH" in x][0]
        attrs = dt[f"{group_VH}"].attrs["other_metadata"]["image_annotation"][
            "image_information"
        ]

        slant_range_spacing_m = attrs["range_pixel_spacing"] * footprint_scale_factor[1]
        slant_range_time_interval_s = slant_range_spacing_m * 2 / _SPEED_OF_LIGHT

        grid_parameters: dict[str, Any] = {
            "slr0": attrs["slant_range_time"],
            "d_slr": slant_range_time_interval_s,
            "spacing_slr": slant_range_spacing_m,
            "az0": np.datetime64(attrs["product_first_line_utc_time"]),
            "d_az": attrs["azimuth_time_interval"] * footprint_scale_factor[0],
            "spacing_az": attrs["azimuth_pixel_spacing"] * footprint_scale_factor[0],
        }
        return grid_parameters


def register(registry: AnalysisModeRegistry):
    """Register Sentinel-1 analysis modes."""
    registry.register(Sen1GRD)


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
    STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
    client = pystac_client.Client.open(STAC_URL)
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
    dem = dem.rename("dem")

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
        dem = resample_in_space(dem.to_dataset(), target_gm=target_gm).dem

    return dem


def convert_dem_to_ecef(dem: xr.DataArray) -> xr.DataArray:
    """Convert a DEM from its native CRS to ECEF coordinates.

    Args:
        dem: DEM data array.

    Returns:
        DEM expressed in ECEF axes.
    """
    gm_dem = GridMapping.from_dataset(dem.to_dataset())
    x_dim, y_dim = gm_dem.xy_var_names
    transformer = pyproj.Transformer.from_crs(gm_dem.crs, _CRS_ECEF, always_xy=True)

    def _transform(
        block_xx: np.ndarray, block_yy: np.ndarray, block_dem: np.ndarray
    ) -> np.ndarray:
        x, y, z = transformer.transform(block_xx, block_yy, block_dem)
        return np.stack([x, y, z], axis=0)

    xx, yy = da.meshgrid(
        da.from_array(dem[x_dim].values, chunks=dem.data.chunks[1][0]),
        da.from_array(dem[y_dim].values, chunks=dem.data.chunks[0][0]),
        indexing="xy",
    )

    xyz_transformed = da.map_blocks(
        _transform,
        xx,
        yy,
        dem.data,
        dtype=np.float32,
        chunks=(3, *dem.data.chunks),
    )

    return xr.DataArray(
        xyz_transformed,
        dims=("axis", "lat", "lon"),
        coords={
            "lat": dem[y_dim].data,
            "lon": dem[x_dim].data,
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
    dem_ecef: xr.DataArray,
    pos_coeff: xr.DataArray,
    vel_coeff: xr.DataArray,
    method="newton",
    tol=1.0,
    speed=7500.0,
    maxiter=10,
    t_shift=-0.1,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Compute orbit time and vectors for a DEM using inverse geocoding.

    Args:
        dem_ecef: DEM in ECEF coordinates.
        pos_coeff: Position polynomial coefficients.
        vel_coeff: Velocity polynomial coefficients.
        method: Root-finding method.
        tol: Function tolerance.
        speed: Nominal platform speed for tolerance scaling.
        maxiter: Maximum number of iterations.
        t_shift: Time shift for the secant method.

    Returns:
        Orbit time, distance vector, and velocity vector.

    Raises:
        ValueError: If the method is not supported.
    """
    f = functools.partial(zero_doppler, dem_ecef, pos_coeff, vel_coeff)

    t0 = xr.zeros_like(dem_ecef.sel(axis="x"), dtype="float64")
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

    dist, vel = payload
    return time_orbit, dist, vel


def simulate_acquisition(
    dem_ecef: xr.DataArray,
    pos_coeff: xr.DataArray,
    vel_coeff: xr.DataArray,
    apply_rtc: bool = True,
) -> xr.Dataset:
    """Simulate SAR acquisition geometry for a DEM.

    Args:
        dem_ecef: DEM in ECEF coordinates.
        pos_coeff: Position polynomial coefficients.
        vel_coeff: Velocity polynomial coefficients.
        apply_rtc: Whether to compute gamma area.

    Returns:
        Dataset with simulated acquisition variables.
    """
    time_orbit, dist, vel = backward_geocode(dem_ecef, pos_coeff, vel_coeff)

    slant_range = np.sqrt((dist**2).sum("axis"))
    time_slr = 2 * slant_range / _SPEED_OF_LIGHT

    out = xr.Dataset(
        {
            "azimuth_time": orbit_to_az(time_orbit, pos_coeff.attrs["epoch"]),
            "distance": dist,
            "velocity": vel.transpose(*dist.dims),
            "slant_range_time": time_slr,
        }
    )

    if apply_rtc:
        out["gamma_area"] = compute_gamma_area(dem_ecef, dist / slant_range)

    return out


def compute_dem_area(dem_ecef: xr.DataArray) -> xr.DataArray:
    """Compute per-pixel surface area on the DEM in ECEF coordinates.

    Args:
        dem_ecef: DEM in ECEF coordinates.

    Returns:
        Area vectors per DEM pixel.
    """
    # construct corner coordinates
    lon = dem_ecef.lon
    lat = dem_ecef.lat
    lon_c = np.concatenate(
        [
            [lon[0] + (lon[0] - lon[1]) / 2],
            ((lon[:-1].data + lon[1:].data) / 2),
            [lon[-1] + (lon[-1] - lon[-2]) / 2],
        ]
    )

    lat_c = np.concatenate(
        [
            [lat[0] + (lat[0] - lat[1]) / 2],
            ((lat[:-1].data + lat[1:].data) / 2),
            [lat[-1] + (lat[-1] - lat[-2]) / 2],
        ]
    )

    # interpolate DEM to pixel corners
    chunksizes = {key: val[0] for key, val in dem_ecef.chunksizes.items()}
    xyz_c = dem_ecef.interp(lon=lon_c).chunk(dict(lon=chunksizes["lon"]))
    xyz_c = xyz_c.interp(lat=lat_c).chunk(chunksizes)

    # compute edge vectors
    dx = xyz_c.diff("lon")
    dy = xyz_c.diff("lat")

    # align shapes for two triangles
    dx1 = dx.isel(lat=slice(1, None))
    dy1 = dy.isel(lon=slice(1, None))
    dx2 = dx.isel(lat=slice(None, -1))
    dy2 = dy.isel(lon=slice(None, -1))

    # restore original coords
    dx1 = dx1.assign_coords(dem_ecef.coords).chunk(chunksizes)
    dy1 = dy1.assign_coords(dem_ecef.coords).chunk(chunksizes)
    dx2 = dx2.assign_coords(dem_ecef.coords).chunk(chunksizes)
    dy2 = dy2.assign_coords(dem_ecef.coords).chunk(chunksizes)

    # compute triangle areas
    cross1 = xr.cross(dx1, dy1, dim="axis") / 2
    cross2 = xr.cross(dx2, dy2, dim="axis") / 2

    # ensure outward normal direction
    sign1 = np.sign(xr.dot(cross1, dem_ecef, dim="axis"))
    sign2 = np.sign(xr.dot(cross2, dem_ecef, dim="axis"))

    return cross1 * sign1 + cross2 * sign2


def compute_gamma_area(dem_ecef: xr.DataArray, direction: xr.DataArray) -> xr.DataArray:
    """Compute gamma area by projecting DEM areas onto look direction.

    Args:
        dem_ecef: DEM in ECEF coordinates.
        direction: Look direction vectors.

    Returns:
        Gamma area for each DEM pixel.
    """
    area = compute_dem_area(dem_ecef)
    gamma_area = xr.dot(area, -direction, dim="axis")
    return gamma_area.where(gamma_area > 0, 0)


def sum_weights(
    weights: xr.DataArray,
    az_idx: xr.DataArray,
    slr_idx: xr.DataArray,
) -> xr.DataArray:
    """Accumulate weights into the SAR image grid.

    Args:
        weights: Weights to accumulate.
        az_idx: Azimuth indices.
        slr_idx: Slant-range indices.

    Returns:
        Accumulated weights on the SAR grid.
    """
    reduced = flox.xarray.xarray_reduce(
        weights,
        slr_idx,
        az_idx,
        func="sum",
        method="map-reduce",
    )

    return reduced.interp(
        slr_idx=slr_idx,
        az_idx=az_idx,
        method="nearest",
    ).drop_vars(("az_idx", "slr_idx"))


def gamma_weights_bilinear(acq: xr.Dataset) -> xr.DataArray:
    """Compute bilinear gamma weights for the acquisition grid.

    Args:
        acq: Acquisition dataset with indices and gamma area.

    Returns:
        Gamma weights on the SAR grid.
    """
    az_idx = acq.az_idx
    slr_idx = acq.slr_idx

    az0 = np.floor(az_idx).astype(np.intp)
    az1 = np.ceil(az_idx).astype(np.intp)
    slr0 = np.floor(slr_idx).astype(np.intp)
    slr1 = np.ceil(slr_idx).astype(np.intp)

    w00 = abs((az1 - az_idx) * (slr1 - slr_idx))
    w01 = abs((az1 - az_idx) * (slr0 - slr_idx))
    w10 = abs((az0 - az_idx) * (slr1 - slr_idx))
    w11 = abs((az0 - az_idx) * (slr0 - slr_idx))

    gamma = acq.gamma_area
    return (
        sum_weights(gamma * w00, az0, slr0)
        + sum_weights(gamma * w01, az0, slr1)
        + sum_weights(gamma * w10, az1, slr0)
        + sum_weights(gamma * w11, az1, slr1)
    )


def gamma_weights_nearest(acq: xr.Dataset) -> xr.DataArray:
    """Compute nearest-neighbor gamma weights for the acquisition grid.

    Args:
        acq: Acquisition dataset with indices and gamma area.

    Returns:
        Gamma weights on the SAR grid.
    """

    az_idx = np.round(acq.az_idx).astype(np.intp)
    slr_idx = np.round(acq.slr_idx).astype(np.intp)
    return sum_weights(acq.gamma_area, az_idx, slr_idx)


def apply_gamma_weights(
    acq: xr.Dataset,
    func: Callable[..., xr.DataArray],
    params: dict,
) -> xr.DataArray:
    """Apply gamma weighting block-wise.

    Args:
        acq: Acquisition dataset with geometry.
        func: Weighting function.
        params: Grid parameters for index conversion.

    Returns:
        Gamma-corrected area per pixel.
    """
    acq["az_idx"] = (acq.azimuth_time - params["az0"]) / _ONE_SECOND / params["d_az"]
    acq["slr_idx"] = (acq.slant_range_time - params["slr0"]) / params["d_slr"]

    template = acq.gamma_area * 0

    area = xr.map_blocks(func, acq, template=template)
    return area / (params["spacing_slr"] * params["spacing_az"])


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
    for i, time in enumerate(x_gcp.azimuth_time.values):
        coeff.append(np.polyfit(x_gcp[i, :], x_gcp.ground_range, deg=deg))
    return xr.DataArray(
        coeff,
        coords=dict(azimuth_time=x_gcp.azimuth_time, degree=np.arange(deg, -1, -1)),
        dims=("azimuth_time", "degree"),
        attrs=dict(mean=mean, std=std),
    )


def geocode_data(
    data: xr.Dataset,
    time_az: xr.DataArray,
    time_slr: xr.DataArray,
    time_slr_gcp: xr.DataArray,
    interp_method: Literal["nearest", "bilinear"] = "nearest",
) -> xr.Dataset:
    """Geocode data from SAR grid to map coordinates.

    Args:
        data: Input dataset on the SAR grid.
        time_az: Target azimuth times.
        time_slr: Target slant-range times.
        time_slr_gcp: GCP slant-range times.
        interp_method: Interpolation method.

    Returns:
        Geocoded dataset.
    """

    coeff = fit_ground_range(time_slr_gcp)

    def _interp_block(block):
        coeff_interp = coeff.interp(azimuth_time=block.time_az)
        x_tgt = (block.time_slr - coeff.attrs["mean"]) / coeff.attrs["std"]
        ground_range = (coeff_interp * x_tgt**coeff.degree).sum("degree")
        return data.interp(
            azimuth_time=block.time_az,
            ground_range=ground_range,
            method=interp_method,
        )

    # Build template with new coordinates
    chunksizes = {}
    for val in [time_az, time_slr]:
        for dim in val.dims:
            chunksizes[dim] = val.chunksizes[dim]
    coeff_interp = coeff.interp(azimuth_time=time_az)
    x_tgt = (time_slr - coeff.attrs["mean"]) / coeff.attrs["std"]
    ground_range = (coeff_interp * x_tgt**coeff.degree).sum("degree")
    template = data.interp(
        azimuth_time=time_az,
        ground_range=ground_range,
    ).chunk(chunksizes)

    target_coords = xr.Dataset({"time_az": time_az, "time_slr": time_slr})
    return xr.map_blocks(_interp_block, target_coords, template=template)


def terrain_correct(
    data: xr.Dataset,
    time_slr_gcp: xr.DataArray,
    sat_position: xr.DataArray,
    dem: xr.DataArray,
    apply_rtc: bool = True,
    grid_params: dict = None,
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

    dem_ecef = convert_dem_to_ecef(dem)

    polyfit_pos = fit_position(sat_position)
    polyfit_vel = poly_derivative(polyfit_pos)

    acquisition = simulate_acquisition(
        dem_ecef, polyfit_pos, polyfit_vel, apply_rtc=apply_rtc
    )

    geocoded = geocode_data(
        data,
        acquisition.azimuth_time,
        acquisition.slant_range_time,
        time_slr_gcp,
        interp_method,
    )

    if apply_rtc:
        if grid_params is None:
            raise ValueError("grid parameters required for RTC")

        if interp_method == "bilinear":
            weights_fn = gamma_weights_bilinear
        else:  # interp_method == "nearest"
            weights_fn = gamma_weights_nearest
        beta_sim = apply_gamma_weights(acquisition, weights_fn, grid_params)
        geocoded = geocoded / beta_sim

    return geocoded
