import xarray as xr
import matplotlib.pyplot as plt
from typing import Container, Any, Callable, TypeVar
import pyproj
import numpy as np
import numpy.typing as npt
import dask.array as da
import functools
import itertools
import math
from unittest import mock
import flox.xarray

ArrayLike = TypeVar("ArrayLike", bound=npt.ArrayLike)
FloatArrayLike = TypeVar("FloatArrayLike", bound=npt.ArrayLike)

S_TO_NS = 10**9
SPEED_OF_LIGHT = 299_792_458.0  # m / s
ONE_SECOND = np.timedelta64(10**9, "ns")


def convert_dem_to_ecef(dem: xr.Dataset) -> xr.DataArray:
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

    def transform_block(xx: np.ndarray, yy: np.ndarray, hh: np.ndarray) -> np.ndarray:
        x, y, z = transformer.transform(xx, yy, hh)
        return np.stack([x, y, z], axis=0)

    lon, lat = da.meshgrid(
        da.from_array(dem.lon.values, chunks=dem.dem.data.chunks[1][0]),
        da.from_array(dem.lat.values, chunks=dem.dem.data.chunks[0][0]),
        indexing="xy",
    )

    xyz_transformed = da.map_blocks(
        transform_block,
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


def azimuth_time_to_orbit_time(
    azimuth_time: xr.DataArray, epoch: np.datetime64
) -> xr.DataArray:
    orbit_time = (azimuth_time - epoch) / np.timedelta64(S_TO_NS, "ns")
    return orbit_time.rename("orbit_time")


def orbit_time_to_azimuth_time(
    orbit_time: xr.DataArray, epoch: np.datetime64
) -> xr.DataArray:
    azimuth_time = orbit_time * np.timedelta64(S_TO_NS, "ns") + epoch
    return azimuth_time.rename("azimuth_time")


def fit_position(
    sat_pos: xr.DataArray, time_dim: str = "azimuth_time", deg: int = 5
) -> xr.DataArray:
    time = sat_pos.coords[time_dim]
    epoch = time.values[0] + (time.values[-1] - time.values[0]) / 2
    interval = (time.values[0], time.values[-1])

    orbit_time = azimuth_time_to_orbit_time(time, epoch)
    sat_pos = sat_pos.assign_coords({time_dim: orbit_time})
    results = sat_pos.polyfit(dim=time_dim, deg=deg)
    coeff = results.polyfit_coefficients
    coeff.attrs["epoch"] = epoch
    coeff.attrs["interval"] = interval

    return coeff.compute()


def make_simulate_acquisition_template(
    template_raster: xr.DataArray,
    correct_radiometry: str | None = None,
) -> xr.Dataset:
    acquisition_template = xr.Dataset(
        data_vars={
            "slant_range_time": template_raster,
            "azimuth_time": template_raster.astype("datetime64[ns]"),
        }
    )
    if correct_radiometry is not None:
        acquisition_template["gamma_area"] = template_raster

    return acquisition_template


def get_poly_derivative(coefficients: xr.DataArray) -> xr.DataArray:
    # TODO: raise if "degree" coord is not decreasing
    derivative_coefficients = coefficients.isel(degree=slice(1, None)).copy()
    for degree in coefficients.coords["degree"].values[:-1]:
        derivative_coefficients.loc[{"degree": degree - 1}] = (
            coefficients.loc[{"degree": degree}] * degree
        )
    return derivative_coefficients


def zero_doppler_plane_distance_velocity(
    dem_ecef: xr.DataArray,
    polyfit_pos: xr.DataArray,
    polyfit_vel: xr.DataArray,
    orbit_time: xr.DataArray,
    dim: str = "axis",
) -> tuple[xr.DataArray, tuple[xr.DataArray, xr.DataArray]]:
    sat_pos = xr.polyval(orbit_time, polyfit_pos)
    dem_dist = dem_ecef - sat_pos
    sat_vel = xr.polyval(orbit_time, polyfit_vel)
    plane_distance_velocity = (dem_dist * sat_vel).sum(dim, skipna=False)
    return plane_distance_velocity, (dem_dist, sat_vel)


def zero_doppler_plane_distance_velocity_prime(
    polyfit_vel: xr.DataArray,
    orbit_time: xr.DataArray,
    payload: tuple[xr.DataArray, xr.DataArray],
    dim: str = "axis",
) -> xr.DataArray:
    dem_dist, sat_vel = payload

    acceleration = xr.polyval(orbit_time, get_poly_derivative(polyfit_vel))
    plane_distance_velocity_prime = (dem_dist * acceleration - sat_vel**2).sum(dim)
    return plane_distance_velocity_prime


def secant_method(
    ufunc: Callable[[ArrayLike], tuple[FloatArrayLike, Any]],
    t_prev: ArrayLike,
    t_curr: ArrayLike,
    diff_ufunc: float = 1.0,
    diff_t: Any = 1e-6,
    maxiter: int = 10,
) -> tuple[ArrayLike, ArrayLike, FloatArrayLike, int, Any]:
    """Return the root of ufunc calculated using the secant method."""
    # implementation modified from https://en.wikipedia.org/wiki/Secant_method
    f_prev, _ = ufunc(t_prev)

    # strong convergence, all points below one of the two thresholds
    for k in range(maxiter):
        f_curr, payload_curr = ufunc(t_curr)

        # print(f"{f_curr / 7500}")

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        t_diff = t_curr - t_prev  # type: ignore

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

        q = f_curr - f_prev  # type: ignore

        # NOTE: in same cases f_curr * t_diff overflows datetime64[ns] before the division by q
        t_prev, t_curr = t_curr, t_curr - np.where(q != 0, f_curr / q, 0) * t_diff  # type: ignore
        f_prev = f_curr

    return t_curr, t_prev, f_curr, k, payload_curr


def newton_raphson_method(
    ufunc: Callable[[ArrayLike], tuple[FloatArrayLike, Any]],
    ufunc_prime: Callable[[ArrayLike, Any], FloatArrayLike],
    t_curr: ArrayLike,
    diff_ufunc: float = 1.0,
    diff_t: Any = 1e-6,
    maxiter: int = 10,
) -> tuple[ArrayLike, FloatArrayLike, int, Any]:
    """Return the root of ufunc calculated using the Newton method."""
    # implementation based on https://en.wikipedia.org/wiki/Newton%27s_method
    # strong convergence, all points below one of the two thresholds
    for k in range(maxiter):
        f_curr, payload_curr = ufunc(t_curr)

        # print(f"{f_curr / 7500}")

        # the `not np.any` construct let us accept `np.nan` as good values
        if not np.any((np.abs(f_curr) > diff_ufunc)):
            break

        fp_curr = ufunc_prime(t_curr, payload_curr)

        t_diff = f_curr / fp_curr  # type: ignore

        # the `not np.any` construct let us accept `np.nat` as good values
        if not np.any(np.abs(t_diff) > diff_t):
            break

        t_curr = t_curr - t_diff  # type: ignore

    return t_curr, f_curr, k, payload_curr


def backward_geocode_simple(
    dem_ecef: xr.DataArray,
    polyfit_pos: xr.DataArray,
    polyfit_vel: xr.DataArray,
    zero_doppler_distance: float = 1.0,
    satellite_speed: float = 7_500.0,
    method: str = "secant",
    orbit_time_prev_shift: float = -0.1,
    maxiter: int = 10,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    diff_ufunc = zero_doppler_distance * satellite_speed

    zero_doppler = functools.partial(
        zero_doppler_plane_distance_velocity, dem_ecef, polyfit_pos, polyfit_vel
    )

    orbit_time_guess = xr.full_like(dem_ecef.sel(axis="x"), 0, dtype="float64")

    if method == "secant":
        orbit_time_guess_prev = orbit_time_guess + orbit_time_prev_shift
        orbit_time, _, _, k, (dem_distance, satellite_velocity) = secant_method(
            zero_doppler,
            orbit_time_guess_prev,
            orbit_time_guess,
            diff_ufunc,
            maxiter=maxiter,
        )
    elif method in {"newton", "newton_raphson"}:
        zero_doppler_prime = functools.partial(
            zero_doppler_plane_distance_velocity_prime, polyfit_vel
        )
        orbit_time, _, k, (dem_distance, satellite_velocity) = newton_raphson_method(
            zero_doppler,
            zero_doppler_prime,
            orbit_time_guess,
            diff_ufunc,
            maxiter=maxiter,
        )
    # print(f"iterations: {k}")
    return orbit_time, dem_distance, satellite_velocity


def backward_geocode(
    dem_ecef: xr.DataArray,
    polyfit_pos: xr.DataArray,
    polyfit_vel: xr.DataArray,
    zero_doppler_distance: float = 1.0,
    satellite_speed: float = 7_500.0,
    method: str = "newton",
    maxiter: int = 10,
    orbit_time_prev_shift: float = -0.1,
) -> xr.Dataset:

    orbit_time, dem_distance, satellite_velocity = backward_geocode_simple(
        dem_ecef,
        polyfit_pos,
        polyfit_vel,
        zero_doppler_distance,
        satellite_speed,
        method,
        orbit_time_prev_shift=orbit_time_prev_shift,
        maxiter=maxiter,
    )

    acquisition = xr.Dataset(
        data_vars={
            "azimuth_time": orbit_time_to_azimuth_time(
                orbit_time, polyfit_pos.attrs["epoch"]
            ),
            "dem_distance": dem_distance,
            "satellite_velocity": satellite_velocity.transpose(*dem_distance.dims),
        }
    )
    return acquisition


def compute_dem_oriented_area(dem_ecef: xr.DataArray) -> xr.DataArray:
    lon_corners = np.concatenate(
        [
            [dem_ecef.lon[0] + (dem_ecef.lon[0] - dem_ecef.lon[1]) / 2],
            ((dem_ecef.lon.shift(lon=-1) + dem_ecef.lon) / 2)[:-1].data,
            [dem_ecef.lon[-1] + (dem_ecef.lon[-1] - dem_ecef.lon[-2]) / 2],
        ]
    )
    lat_corners = np.concatenate(
        [
            [dem_ecef.lat[0] + (dem_ecef.lat[0] - dem_ecef.lat[1]) / 2],
            ((dem_ecef.lat.shift(lat=-1) + dem_ecef.lat) / 2)[:-1].data,
            [dem_ecef.lat[-1] + (dem_ecef.lat[-1] - dem_ecef.lat[-2]) / 2],
        ]
    )

    dem_ecef_corners = dem_ecef.interp(
        {"lon": lon_corners, "lat": lat_corners},
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    dx = dem_ecef_corners.diff("lon", 1)
    dy = dem_ecef_corners.diff("lat", 1)

    dx1 = dx.isel(lat=slice(1, None)).assign_coords(dem_ecef.coords)
    dy1 = dy.isel(lon=slice(1, None)).assign_coords(dem_ecef.coords)
    dx2 = dx.isel(lat=slice(None, -1)).assign_coords(dem_ecef.coords)
    dy2 = dy.isel(lon=slice(None, -1)).assign_coords(dem_ecef.coords)

    cross_1 = xr.cross(dx1, dy1, dim="axis") / 2
    sign_1 = np.sign(
        xr.dot(cross_1, dem_ecef, dim="axis")
    )  # ensure direction out of DEM

    cross_2 = xr.cross(dx2, dy2, dim="axis") / 2
    sign_2 = np.sign(
        xr.dot(cross_2, dem_ecef, dim="axis")
    )  # ensure direction out of DEM
    dem_oriented_area: xr.DataArray = cross_1 * sign_1 + cross_2 * sign_2

    return dem_oriented_area.rename("dem_oriented_area")


def compute_gamma_area(
    dem_ecef: xr.DataArray,
    dem_direction: xr.DataArray,
) -> xr.DataArray:
    dem_oriented_area = compute_dem_oriented_area(dem_ecef)
    gamma_area: xr.DataArray = xr.dot(dem_oriented_area, -dem_direction, dim="axis")
    gamma_area = gamma_area.where(gamma_area > 0, 0)
    return gamma_area


def simulate_acquisition(
    dem_ecef: xr.DataArray,
    polyfit_pos: xr.DataArray,
    polyfit_vel: xr.DataArray,
    include_variables: Container[str] = (),
) -> xr.Dataset:
    """Compute the image coordinates of the DEM given the satellite orbit."""
    acquisition = backward_geocode(dem_ecef, polyfit_pos, polyfit_vel)

    slant_range = (acquisition.dem_distance**2).sum(dim="axis") ** 0.5
    slant_range_time = 2.0 / SPEED_OF_LIGHT * slant_range

    acquisition["slant_range_time"] = slant_range_time

    if include_variables and "gamma_area" in include_variables:
        gamma_area = compute_gamma_area(
            dem_ecef, acquisition.dem_distance / slant_range
        )
        acquisition["gamma_area"] = gamma_area

    for data_var_name in acquisition.data_vars:
        if include_variables and data_var_name not in include_variables:
            acquisition = acquisition.drop_vars(data_var_name)  # type: ignore

    # drop coordinates that are not associated with any data variable
    for coord_name in acquisition.coords:
        if all(coord_name not in dv.coords for dv in acquisition.data_vars.values()):
            acquisition = acquisition.drop_vars(coord_name)  # type: ignore

    return acquisition


def map_simulate_acquisition(
    dem_ecef: xr.DataArray,
    polyfit_pos: xr.DataArray,
    polyfit_vel: xr.DataArray,
    correct_radiometry: str | None = None,
) -> xr.Dataset:
    template_raster = dem_ecef.isel(axis=0).drop_vars(["axis"]) * 0.0
    acquisition_template = make_simulate_acquisition_template(
        template_raster, correct_radiometry
    )
    acquisition = xr.map_blocks(
        simulate_acquisition,
        dem_ecef,
        kwargs={
            "polyfit_pos": polyfit_pos,
            "polyfit_vel": polyfit_vel,
            "include_variables": list(acquisition_template.data_vars),
        },
        template=acquisition_template,
    )
    return acquisition


def compute_product(
    slices: list[list[slice]], dims_name: list[str]
) -> list[dict[str, slice]]:
    product: list[dict[str, slice]] = []

    for slices_ in itertools.product(*slices):
        product.append({})
        for dim, sl in zip(dims_name, slices_):
            product[-1][dim] = sl
    return product


def compute_chunks_1d(
    dim_size: int,
    chunks: int = 2048,
    bound: int = 128,
) -> tuple[list[slice], list[slice], list[slice]]:
    ext_slices = []
    ext_slices_bound = []
    int_slices = []

    # -bound is needed to avoid to incorporate the last chunk, if smaller of bound in the previous chunk
    if dim_size > bound:
        number_of_chunks = int(math.ceil((dim_size - bound) / chunks))
    else:
        number_of_chunks = 1
    for n in range(number_of_chunks):
        l_int = n * chunks
        if n * chunks - bound > 0:
            l_ext = n * chunks - bound
        else:
            l_ext = 0
        l_bound = l_int - l_ext

        if (n + 1) * chunks + bound < dim_size:
            r_ext = (n + 1) * chunks + bound
            r_int = (n + 1) * chunks
            r_bound = chunks + l_bound
        else:
            r_ext = dim_size
            r_int = dim_size
            r_bound = r_ext - l_ext

        ext_slices.append(slice(l_ext, r_ext))
        ext_slices_bound.append(slice(l_bound, r_bound))
        int_slices.append(slice(l_int, r_int))
    return ext_slices, ext_slices_bound, int_slices


def compute_chunks(
    dims: dict[str, int] = {},
    chunks: int = 2048,
    bound: int = 128,
) -> tuple[list[dict[str, slice]], list[dict[str, slice]], list[dict[str, slice]]]:
    ext_slices_ = []
    ext_slices_bound_ = []
    int_slices_ = []
    for dim_size in dims.values():
        ec, ecb, ic = compute_chunks_1d(dim_size, chunks=chunks, bound=bound)
        ext_slices_.append(ec)
        ext_slices_bound_.append(ecb)
        int_slices_.append(ic)

    ext_slices = compute_product(ext_slices_, list(dims))
    ext_slices_bound = compute_product(ext_slices_bound_, list(dims))
    int_slices = compute_product(int_slices_, list(dims))
    return ext_slices, ext_slices_bound, int_slices


def map_ovelap(
    function: Callable[..., xr.DataArray],
    obj: xr.Dataset | xr.DataArray,
    chunks: int = 2048,
    bound: int = 128,
    kwargs: dict[Any, Any] = {},
    template: xr.DataArray | None = None,
) -> xr.DataArray:
    dims = {}
    for d in obj.dims:
        dims[str(d)] = len(obj[d])

    if isinstance(obj, xr.Dataset):
        if template is None:
            raise ValueError(
                "template argument is mandatory if obj is type of xr.Dataset"
            )
    elif isinstance(obj, xr.DataArray):
        if template is None:
            template = obj

    ext_chunks, ext_chunks_bounds, int_chunks = compute_chunks(
        dims, chunks, bound
    )  # type ignore

    try:
        from dask.array import empty_like
    except ModuleNotFoundError:
        from numpy import empty_like  # type: ignore

    out = xr.DataArray(empty_like(template.data), dims=template.dims)  # type: ignore
    out.coords.update(obj.coords)
    for ext_chunk, ext_chunk_bounds, int_chunk in zip(
        ext_chunks, ext_chunks_bounds, int_chunks
    ):
        out_chunk = function(obj.isel(ext_chunk), **kwargs)
        out[int_chunk] = out_chunk.isel(ext_chunk_bounds)
    return out


def sum_weights(
    initial_weights: xr.DataArray,
    azimuth_index: xr.DataArray,
    slant_range_index: xr.DataArray,
) -> xr.DataArray:
    geocoded = initial_weights.assign_coords(
        slant_range_index=slant_range_index, azimuth_index=azimuth_index
    )

    flat_sum: xr.DataArray = flox.xarray.xarray_reduce(
        geocoded,
        geocoded.slant_range_index,
        geocoded.azimuth_index,
        func="sum",
        method="map-reduce",
    )

    weights_sum = flat_sum.interp(
        slant_range_index=slant_range_index,
        azimuth_index=azimuth_index,
        method="nearest",
    )

    return weights_sum


def gamma_weights_bilinear(
    dem_coords: xr.Dataset,
    slant_range_time0: float,
    azimuth_time0: np.datetime64,
    slant_range_time_interval_s: float,
    azimuth_time_interval_s: float,
    slant_range_spacing_m: float = 1.0,
    azimuth_spacing_m: float = 1.0,
) -> xr.DataArray:
    # compute dem image coordinates
    azimuth_index = ((dem_coords.azimuth_time - azimuth_time0) / ONE_SECOND) / (
        azimuth_time_interval_s
    )

    slant_range_index = (dem_coords.slant_range_time - slant_range_time0) / (
        slant_range_time_interval_s
    )

    slant_range_index_0 = np.floor(slant_range_index).astype(int).compute()
    slant_range_index_1 = np.ceil(slant_range_index).astype(int).compute()
    azimuth_index_0 = np.floor(azimuth_index).astype(int).compute()
    azimuth_index_1 = np.ceil(azimuth_index).astype(int).compute()

    w_00 = abs(
        (azimuth_index_1 - azimuth_index) * (slant_range_index_1 - slant_range_index)
    )
    tot_area_00 = sum_weights(
        dem_coords["gamma_area"] * w_00,
        azimuth_index=azimuth_index_0,
        slant_range_index=slant_range_index_0,
    )

    w_01 = abs(
        (azimuth_index_1 - azimuth_index) * (slant_range_index_0 - slant_range_index)
    )
    tot_area_01 = sum_weights(
        dem_coords["gamma_area"] * w_01,
        azimuth_index=azimuth_index_0,
        slant_range_index=slant_range_index_1,
    )

    w_10 = abs(
        (azimuth_index_0 - azimuth_index) * (slant_range_index_1 - slant_range_index)
    )
    tot_area_10 = sum_weights(
        dem_coords["gamma_area"] * w_10,
        azimuth_index=azimuth_index_1,
        slant_range_index=slant_range_index_0,
    )

    w_11 = abs(
        (azimuth_index_0 - azimuth_index) * (slant_range_index_0 - slant_range_index)
    )
    tot_area_11 = sum_weights(
        dem_coords["gamma_area"] * w_11,
        azimuth_index=azimuth_index_1,
        slant_range_index=slant_range_index_1,
    )

    tot_area = tot_area_00 + tot_area_01 + tot_area_10 + tot_area_11

    normalized_area = tot_area / (azimuth_spacing_m * slant_range_spacing_m)
    return normalized_area


def gamma_weights_nearest(
    dem_coords: xr.Dataset,
    slant_range_time0: float,
    azimuth_time0: np.datetime64,
    slant_range_time_interval_s: float,
    azimuth_time_interval_s: float,
    slant_range_spacing_m: float = 1.0,
    azimuth_spacing_m: float = 1.0,
) -> xr.DataArray:
    # compute dem image coordinates
    azimuth_index = np.round(
        (dem_coords.azimuth_time - azimuth_time0) / ONE_SECOND / azimuth_time_interval_s
    ).astype(int)

    slant_range_index = np.round(
        (dem_coords.slant_range_time - slant_range_time0) / slant_range_time_interval_s
    ).astype(int)

    tot_area = sum_weights(
        dem_coords["gamma_area"],
        azimuth_index=azimuth_index,
        slant_range_index=slant_range_index,
    )

    normalized_area = tot_area / (azimuth_spacing_m * slant_range_spacing_m)
    return normalized_area


def azimuth_slant_range_grid(
    dt: xr.DataTree,
    grouping_area_factor: tuple[float, float] = (3.0, 3.0),
) -> dict[str, Any]:
    # ToDo SLC

    group_VH = [x for x in dt.children if "VH" in x][0]
    attrs = dt[f"{group_VH}"].attrs["other_metadata"]["image_annotation"][
        "image_information"
    ]

    slant_range_spacing_m = attrs["range_pixel_spacing"] * grouping_area_factor[1]
    slant_range_time_interval_s = (
        slant_range_spacing_m * 2 / SPEED_OF_LIGHT  # ignore type
    )

    grid_parameters: dict[str, Any] = {
        "slant_range_time0": attrs["slant_range_time"],
        "slant_range_time_interval_s": slant_range_time_interval_s,
        "slant_range_spacing_m": slant_range_spacing_m,
        "azimuth_time0": np.datetime64(attrs["product_first_line_utc_time"]),
        "azimuth_time_interval_s": attrs["azimuth_time_interval"]
        * grouping_area_factor[0],
        "azimuth_spacing_m": attrs["azimuth_pixel_spacing"] * grouping_area_factor[0],
    }
    return grid_parameters


def slant_range_time_to_ground_range(
    azimuth_time: xr.DataArray,
    slant_range_time: xr.DataArray,
    slant_range_time_gcp: xr.DataArray,
    deg: int = 3,
) -> xr.DataArray:

    # normalize for numerical stability
    srt_mean = slant_range_time_gcp.mean(dim="ground_range")
    srt_std = slant_range_time_gcp.std(dim="ground_range")

    x = (slant_range_time_gcp - srt_mean) / srt_std

    def _polyfit(x, y):
        return np.polyfit(x, y, deg)

    coeff = xr.apply_ufunc(
        _polyfit,
        x,
        slant_range_time_gcp.ground_range,
        input_core_dims=[["ground_range"], ["ground_range"]],
        output_core_dims=[["degree"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {"degree": deg + 1}},
    )

    coeff = coeff.assign_coords(degree=np.arange(deg, -1, -1))

    # interpolate coefficients in azimuth
    coeff_interp = coeff.interp(azimuth_time=azimuth_time)

    # normalize target values
    srt_mean_i = srt_mean.interp(azimuth_time=azimuth_time)
    srt_std_i = srt_std.interp(azimuth_time=azimuth_time)

    x_target = (slant_range_time - srt_mean_i) / srt_std_i

    ground_range = (coeff_interp * x_target**coeff_interp.degree).sum("degree")

    return ground_range


def do_terrain_correction(
    dt: xr.DataTree,
    dem: xr.Dataset,
    correct_radiometry: str | None = None,
    interp_method: xr.core.types.InterpOptions = "nearest",
    radiometry_chunks: int = 2048,
    radiometry_bound: int = 128,
) -> tuple[xr.DataArray, xr.DataArray | None]:

    dem_ecef = convert_dem_to_ecef(dem)
    template_raster = dem_ecef.isel(axis=0).drop_vars("axis") * 0.0

    group_VH = [x for x in dt.children if "VH" in x][0]
    orbit = dt[f"{group_VH}/conditions/orbit"].to_dataset()
    polyfit_pos = fit_position(orbit["position"])
    polyfit_vel = get_poly_derivative(polyfit_pos)

    acquisition = map_simulate_acquisition(
        dem_ecef,
        polyfit_pos,
        polyfit_vel,
        correct_radiometry=correct_radiometry,
    )

    simulated_beta_nought = None
    if correct_radiometry is not None:

        grid_parameters = azimuth_slant_range_grid(dt)

        if correct_radiometry == "gamma_bilinear":
            gamma_weights = gamma_weights_bilinear
        elif correct_radiometry == "gamma_nearest":
            gamma_weights = gamma_weights_nearest

        simulated_beta_nought = map_ovelap(
            obj=acquisition,
            function=gamma_weights,
            chunks=radiometry_chunks,
            bound=radiometry_bound,
            kwargs=grid_parameters,
            template=template_raster,
        )
        simulated_beta_nought.attrs["long_name"] = "terrain-simulated beta nought"

    group_VH = [x for x in dt.children if "VH" in x][0]
    grd = dt[group_VH].measurements.to_dataset().rename({"grd": "vh"})
    group_VV = [x for x in dt.children if "VV" in x][0]
    grd["vv"] = dt[group_VV].measurements.to_dataset().grd
    beta_lut = dt[group_VH].quality.calibration.to_dataset()["beta_nought"]
    beta_lut_interp = beta_lut.interp(ground_range=grd.ground_range).chunk(
        dict(ground_range=2048)
    )
    beta_lut_interp = beta_lut_interp.interp(azimuth_time=grd.azimuth_time).chunk(
        dict(azimuth_time=2048)
    )
    beta_nought = (grd / beta_lut_interp) ** 2
    beta_nought.assign_attrs(long_name="beta nought", units="m2 m-2")

    gcp = dt[f"{group_VH}/conditions/gcp"].to_dataset()
    ground_range = slant_range_time_to_ground_range(
        acquisition.azimuth_time,
        acquisition.slant_range_time,
        gcp.slant_range_time_gcp,
    )

    geocoded = beta_nought.interp(
        azimuth_time=acquisition.azimuth_time,
        ground_range=ground_range,
        method=interp_method,
    )

    if correct_radiometry is not None:
        assert simulated_beta_nought is not None
        geocoded = geocoded / simulated_beta_nought
        geocoded.attrs["long_name"] = "terrain-corrected gamma nought"

    return geocoded
