#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from typing import Any
from collections.abc import Sequence

import dask.array as da
import numpy as np
import xarray as xr


def make_s3_olci_efr(size: int = 48) -> xr.DataTree:

    return create_datatree(
        {
            "measurements": make_s3_meas(
                size, bands=[f"oa{i:02}_radiance" for i in range(1, 22)]
            ),
        },
    )


def make_s3_slstr_lst(size: int = 48) -> xr.DataTree:
    return create_datatree(
        {
            "conditions/auxiliary": make_s3_meas(size, bands=["elevation"]),
            "conditions/meteorology": make_s3_meas((size, size // 10), bands=["s2m"]),
            "conditions/geometry": make_s3_meas(
                (size, size // 10), bands=["sat_azimuth_tn", "sat_zenith_tn"]
            ),
            "measurements": make_s3_meas(size, bands=["lst"]),
        },
    )


def make_s3_slstr_rbt(size: int = 48) -> xr.DataTree:
    return create_datatree(
        {
            "conditions/geometry_tn": make_s3_meas(
                (size // 2, size // 20), bands=["sat_azimuth_tn", "sat_zenith_tn"]
            ),
            "conditions/geometry_to": make_s3_meas(
                (size // 2, size // 20), bands=["sat_azimuth_to", "sat_zenith_to"]
            ),
            "conditions/meteorology": make_s3_meas(
                (size // 2, size // 20), bands=["s2m"]
            ),
            "measurements/anadir": make_s3_meas(
                size, bands=[f"s{i}_radiance_an" for i in range(1, 7)] + ["elevation"]
            ),
            "measurements/inadir": make_s3_meas(
                size // 2, bands=[f"s{i}_bt_in" for i in range(7, 10)] + ["elevation"]
            ),
            "measurements/ioblique": make_s3_meas(
                (int(size // 2 * 0.7), int(size // 2 * 0.5)),
                bands=[f"s{i}_bt_io" for i in range(7, 10)] + ["elevation"],
                oblique_view=True,
            ),
        },
    )


def make_s3_meas(
    size: int | tuple[int, int], bands: Sequence[str], oblique_view=False
) -> xr.Dataset:
    if not isinstance(size, tuple):
        size = (size, size)

    return xr.Dataset(
        data_vars={
            band: xr.DataArray(
                da.ones((size[0], size[1])).astype("float32") * 1000,
                dims=("rows", "columns"),
                attrs={
                    "long_name": "long name",
                    "short_name": "short name",
                    "standard_name": "standard name",
                    "units": "some units",
                },
            ).chunk(columns=max(size[0] // 4, 4), rows=max(size[1] // 4, 4))
            for band in bands
        },
        coords=make_coords(size[1], size[0], oblique_view=oblique_view),
    )


def make_coords(w: int, h: int, oblique_view=False) -> dict[str, xr.DataArray]:
    if oblique_view:
        lat = da.linspace(51, 58, h, chunks=max(h // 10, 4))
        lon = da.linspace(-4, 1, w, chunks=max(w // 10, 4))
    else:
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
