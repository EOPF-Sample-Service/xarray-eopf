#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from typing import Any

import dask.array as da
import numpy as np
import xarray as xr


def make_s3_olci_erf(size: int = 48) -> xr.DataTree:
    return create_datatree(
        {
            "measurements": make_s3_olci_erf_meas(size),
        },
    )


def make_s3_olci_erf_meas(size: int) -> xr.Dataset:
    bands = [f"oa{i:02}_radiance" for i in range(1, 22)]
    return xr.Dataset(
        data_vars={
            band: xr.DataArray(
                da.random.random((size, size)).astype("float32") * 65534,
                dims=("rows", "columns"),
                attrs={
                    "long_name": "TOA radiance for OLCI acquisition band oa01",
                    "short_name": "oa01_radiance",
                    "standard_name": "toa_upwelling_spectral_radiance",
                    "units": "mW.m-2.sr-1.nm-1",
                    "valid_max": 65534,
                    "valid_min": 0,
                },
            ).chunk(columns=max(size // 10, 4), rows=max(size // 10, 4))
            for band in bands
        },
        coords=make_coords(size, size),
    )


def make_coords(w: int, h: int) -> dict[str, xr.DataArray]:
    lat = da.linspace(50, 60, h, chunks=max(h // 10, 4))
    lon = da.linspace(-5, 5, w, chunks=max(w // 10, 4))
    lon_grid, lat_grid = da.meshgrid(lon, lat)
    lon_grid /= da.cos(da.radians(lat_grid))
    lon_grid += 10

    # skew due to earth curvature
    skew = 0.2
    lon_grid += skew * (lat_grid - lat[0]) / (lat[-1] - lat[0])

    # rotate image
    rotation_deg = -25
    theta = np.radians(rotation_deg)
    lat0 = np.mean(lat)
    lon0 = np.mean(lon)
    x = lon_grid - lon0
    y = lat_grid - lat0
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    lon_final = x_rot + lon0
    lat_final = y_rot + lat0

    return {
        "latitude": xr.DataArray(lat_final, dims=("rows", "columns")),
        "longitude": xr.DataArray(lon_final, dims=("rows", "columns")),
        "time_stamps": xr.DataArray(
            np.arange(h).astype("datetime64[ns]"), dims=("rows")
        ),
    }


def create_datatree(
    datasets: dict[str, xr.Dataset], attrs: dict[str, Any] | None = None
) -> xr.DataTree:
    root_group = xr.DataTree(dataset=xr.Dataset(attrs=attrs or {}))
    for group_path, dataset in datasets.items():
        path_names = group_path.split("/")
        last_group = root_group
        for group_name in path_names[:-1]:
            if group_name:
                if group_name not in last_group:
                    last_group[group_name] = xr.DataTree(name=group_name)
                last_group = last_group[group_name]
        group_name = path_names[-1]
        last_group[group_name] = xr.DataTree(name=group_name, dataset=dataset)
    return root_group
