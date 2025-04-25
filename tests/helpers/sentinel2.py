#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from typing import Any

import numpy as np
import xarray as xr


def make_s2_msi(r10m_size: int = 48) -> xr.DataTree:
    return create_datatree(
        {
            "r10m": make_s2_msi_r10m(r10m_size),
            "r20m": make_s2_msi_l1c_r20m(r10m_size),
            "r60m": make_s2_msi_l1c_r60m(r10m_size),
        }
    )


def make_s2_msi_l1c(r10m_size: int = 48) -> xr.DataTree:
    return create_datatree(
        {
            "measurements/reflectance/r10m": make_s2_msi_l1c_r10m(r10m_size),
            "measurements/reflectance/r20m": make_s2_msi_l1c_r20m(r10m_size),
            "measurements/reflectance/r60m": make_s2_msi_l1c_r60m(r10m_size),
        },
        attrs={
            "other_metadata": {
                "horizontal_CRS_code": "EPSG:32632",
            }
        },
    )


def make_s2_msi_l2a(r10m_size: int = 48) -> xr.DataTree:
    return create_datatree(
        {
            "conditions/mask/l2a_classification/r20m": make_s2_msi_l2a_scl_r20m(
                r10m_size
            ),
            "conditions/mask/l2a_classification/r60m": make_s2_msi_l2a_scl_r60m(
                r10m_size
            ),
            "measurements/reflectance/r10m": make_s2_msi_l2a_r10m(r10m_size),
            "measurements/reflectance/r20m": make_s2_msi_l2a_r20m(r10m_size),
            "measurements/reflectance/r60m": make_s2_msi_l2a_r60m(r10m_size),
            "quality/probability/r20m": make_s2_msi_l2a_probs_r20m(r10m_size),
        },
        attrs={
            "other_metadata": {
                "horizontal_CRS_code": "EPSG:32632",
            }
        },
    )


def make_s2_msi_l1c_r10m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_r10m(r10m_size)


def make_s2_msi_l1c_r20m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_rx0m(["b05", "b06", "b07", "b11", "b12", "b8a"], r10m_size // 2)


def make_s2_msi_l1c_r60m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_rx0m(["b01", "b09", "b10"], r10m_size // 6)


def make_s2_msi_l2a_r10m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_r10m(r10m_size)


def make_s2_msi_l2a_r20m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_rx0m(
        ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b11", "b12", "b8a"],
        r10m_size // 2,
    )


def make_s2_msi_l2a_r60m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_rx0m(
        ["b01", "b02", "b03", "b04", "b05", "b06", "b07", "b11", "b12", "b8a"],
        r10m_size // 6,
    )


def make_s2_msi_l2a_scl_r20m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_scl(r10m_size // 2)


def make_s2_msi_l2a_scl_r60m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_scl(r10m_size // 6)


def make_s2_msi_scl(size: int) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "scl": xr.DataArray(
                np.random.randint(0, 1 << 8, (size, size), dtype="uint8"),
                dims=("y", "x"),
                attrs={"proj:epsg": 32625},
            ).chunk(x=4, y=4)
        },
        coords=make_coords(size, size),
    )


def make_s2_msi_l2a_probs_r20m(r10m_size: int):
    size = r10m_size // 2
    return xr.Dataset(
        data_vars={
            "cld": xr.DataArray(
                np.random.randint(0, 1 << 8, (size, size), dtype="uint8"),
                dims=("y", "x"),
                attrs={"proj:epsg": 32625},
            ).chunk(x=4, y=4),
            "snw": xr.DataArray(
                np.random.randint(0, 1 << 8, (size, size), dtype="uint8"),
                dims=("y", "x"),
                attrs={"proj:epsg": 32625},
            ).chunk(x=4, y=4),
        },
        coords=make_coords(size, size),
    )


def make_s2_msi_r10m(r10m_size: int) -> xr.Dataset:
    return make_s2_msi_rx0m(["b02", "b03", "b04", "b08"], r10m_size)


def make_s2_msi_rx0m(bands: list[str], size: int) -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            band: xr.DataArray(
                np.random.randint(0, 1 << 16, (size, size), dtype="uint16"),
                dims=("y", "x"),
                attrs={
                    "_FillValue": 0,
                    "scale_factor": 0.0001,
                    "units": "digital_counts",
                    "valid_max": 65535,
                    "valid_min": 1,
                    "proj:epsg": 32625,
                },
            ).chunk(x=4, y=4)
            for band in bands
        },
        coords=make_coords(size, size),
    )


def make_coords(w: int, h: int) -> dict[str, xr.DataArray]:
    x1, x2 = 0.0, 10.0 * 10000
    dx = 0.5 * (x2 - x1) / w

    y1, y2 = 0.0, 10.0 * 10000
    dy = 0.5 * (y2 - y1) / h

    return {
        "x": xr.DataArray(np.linspace(x1 + dx, y2 - dx, w), dims="x"),
        "y": xr.DataArray(np.linspace(y1 + dy, y2 - dy, h), dims="y"),
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
