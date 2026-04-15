from typing import Callable, Any, Literal
import xarray as xr
import numpy as np
import dask.array as da
import pyproj
import functools
import flox.xarray

SPEED_OF_LIGHT = 299_792_458.0
S_TO_NS = 10**9
ONE_SECOND = np.timedelta64(S_TO_NS, "ns")
CRS_ECEF = "EPSG:4978"
CRS_WGS84 = "EPSG:4326"


def convert_dem_to_ecef(dem: xr.Dataset) -> xr.DataArray:
    transformer = pyproj.Transformer.from_crs(CRS_WGS84, CRS_ECEF, always_xy=True)

    def _transform(lon: np.ndarray, lat: np.ndarray, h: np.ndarray) -> np.ndarray:
        x, y, z = transformer.transform(lon, lat, h)
        return np.stack([x, y, z], axis=0)

    lon, lat = da.meshgrid(
        da.from_array(dem.lon.values, chunks=dem.dem.data.chunks[1][0]),
        da.from_array(dem.lat.values, chunks=dem.dem.data.chunks[0][0]),
        indexing="xy",
    )

    xyz_transformed = da.map_blocks(
        _transform,
        lon,
        lat,
        dem.dem.data,
        dtype=np.float32,
        chunks=(3, *dem.dem.data.chunks),
    )

    return xr.DataArray(
        xyz_transformed,
        dims=("axis", "lat", "lon"),
        coords={"lat": dem.lat, "lon": dem.lon, "axis": ["x", "y", "z"]},
    )


def az_to_orbit(time_az: xr.DataArray, epoch: np.datetime64) -> xr.DataArray:
    return (time_az - epoch) / np.timedelta64(S_TO_NS, "ns")


def orbit_to_az(time_orb: xr.DataArray, epoch: np.datetime64) -> xr.DataArray:
    return time_orb * np.timedelta64(S_TO_NS, "ns") + epoch


def fit_position(pos: xr.DataArray, time_dim="azimuth_time", deg=5) -> xr.DataArray:
    time = pos.coords[time_dim]
    epoch = time.values[0] + (time.values[-1] - time.values[0]) / 2

    time_orbit = az_to_orbit(time, epoch)

    pos = pos.assign_coords({time_dim: time_orbit})
    coeff = pos.polyfit(dim=time_dim, deg=deg).polyfit_coefficients

    coeff.attrs["epoch"] = epoch
    return coeff


def poly_derivative(coeff: xr.DataArray) -> xr.DataArray:
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
    """implementation modified from https://en.wikipedia.org/wiki/Secant_method"""
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
    time_orbit, dist, vel = backward_geocode(dem_ecef, pos_coeff, vel_coeff)

    slant_range = np.sqrt((dist**2).sum("axis"))
    time_slr = 2 * slant_range / SPEED_OF_LIGHT

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
    print("compute_dem_area")
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
    area = compute_dem_area(dem_ecef)
    gamma_area = xr.dot(area, -direction, dim="axis")
    return gamma_area.where(gamma_area > 0, 0)


def sum_weights(
    weights: xr.DataArray,
    az_idx: xr.DataArray,
    slr_idx: xr.DataArray,
) -> xr.DataArray:
    """
    Accumulate weights into SAR image grid using (azimuth, range) indices.
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
    """
    Bilinear gamma weighting.
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
    """
    Nearest-neighbor gamma weighting.
    """

    az_idx = np.round(acq.az_idx).astype(np.intp)
    slr_idx = np.round(acq.slr_idx).astype(np.intp)
    return sum_weights(acq.gamma_area, az_idx, slr_idx)


def apply_gamma_weights(
    acq: xr.Dataset,
    func: Callable[..., xr.DataArray],
    params: dict,
) -> xr.DataArray:
    """
    Apply gamma weighting block-wise.
    """
    acq["az_idx"] = (acq.azimuth_time - params["az0"]) / ONE_SECOND / params["d_az"]
    acq["slr_idx"] = (acq.slant_range_time - params["slr0"]) / params["d_slr"]

    template = acq.gamma_area * 0

    area = xr.map_blocks(func, acq, template=template)
    return area / (params["spacing_slr"] * params["spacing_az"])


def _slr_time_to_gr(
    time_az: xr.DataArray,
    time_slr: xr.DataArray,
    time_slr_gcp: xr.DataArray,
    deg: int = 8,
) -> xr.DataArray:

    # normalization for stability
    mean = time_slr_gcp.mean("ground_range")
    std = time_slr_gcp.std("ground_range")
    x_gcp = (time_slr_gcp - mean) / std

    # polynomial fit per azimuth line
    coeff = xr.apply_ufunc(
        lambda x, y: np.polyfit(x, y, deg),
        x_gcp,
        time_slr_gcp.ground_range,
        input_core_dims=[["ground_range"], ["ground_range"]],
        output_core_dims=[["degree"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"degree": deg + 1}},
    ).assign_coords(degree=np.arange(deg, -1, -1))

    # align to target azimuth time
    coeff = coeff.interp(azimuth_time=time_az)

    # normalization
    mean_t = mean.interp(azimuth_time=time_az)
    std_t = std.interp(azimuth_time=time_az)
    x_tgt = (time_slr - mean_t) / std_t

    # evaluate polynomial
    return (coeff * x_tgt**coeff.degree).sum("degree")


def terrain_correct(
    data: xr.Dataset,
    time_slr_gcp: xr.DataArray,
    sat_position: xr.DataArray,
    dem: xr.Dataset,
    apply_rtc: bool = True,
    grid_params: dict = None,
    interp_method: Literal["nearest", "bilinear"] = "nearest",
) -> xr.Dataset:

    dem_ecef = convert_dem_to_ecef(dem)

    polyfit_pos = fit_position(sat_position)
    polyfit_vel = poly_derivative(polyfit_pos)

    acquisition = simulate_acquisition(
        dem_ecef, polyfit_pos, polyfit_vel, apply_rtc=apply_rtc
    )

    ground_range = _slr_time_to_gr(
        acquisition.azimuth_time,
        acquisition.slant_range_time,
        time_slr_gcp,
    )
    geocoded = data.interp(
        azimuth_time=acquisition.azimuth_time,
        ground_range=ground_range,
        method=interp_method,
    )

    if apply_rtc:
        if grid_params is None:
            raise ValueError("grid parameters required for RTC")

        if interp_method == "bilinear":
            weights_fn = gamma_weights_bilinear
        elif interp_method == "nearest":
            weights_fn = gamma_weights_nearest
        else:
            raise ValueError(
                "'interp_method' needs to be either 'bilinear' or 'nearest'"
            )
        beta_sim = apply_gamma_weights(acquisition, weights_fn, grid_params)
        geocoded = geocoded / beta_sim

    return geocoded
