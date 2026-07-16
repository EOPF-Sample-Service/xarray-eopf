#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.


import numpy as np
import xarray as xr


def _make_grd_group(azimuth_time: np.ndarray, ground_range: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {
            "grd": xr.DataArray(
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                dims=("azimuth_time", "ground_range"),
                coords={"azimuth_time": azimuth_time, "ground_range": ground_range},
            )
        }
    )


def make_s1_grd_datatree() -> xr.DataTree:
    azimuth_time = np.array(
        ["2024-01-01T00:00:00", "2024-01-01T00:00:01"], dtype="datetime64[ns]"
    )
    ground_range = np.array([0.0, 10.0])
    axis = np.array(["x", "y", "z"])
    polarizations = ("VV", "VH")

    beta_nought = xr.Dataset(
        {
            "beta_nought": xr.DataArray(
                np.ones((2, 2), dtype="float32"),
                dims=("azimuth_time", "ground_range"),
                coords={"azimuth_time": azimuth_time, "ground_range": ground_range},
            )
        }
    )
    orbit = xr.Dataset(
        {
            "position": xr.DataArray(
                np.ones((2, 3), dtype="float32"),
                dims=("azimuth_time", "axis"),
                coords={"azimuth_time": azimuth_time, "axis": axis},
            )
        }
    )
    gcp = xr.Dataset(
        {
            "slant_range_time_gcp": xr.DataArray(
                np.ones((2, 2), dtype="float32"),
                dims=("azimuth_time", "ground_range"),
                coords={"azimuth_time": azimuth_time, "ground_range": ground_range},
            )
        }
    )

    dt_nodes = {}
    for pol in polarizations:
        group_name = f"S1A_IW_GRDH_TEST_{pol}"
        dt_nodes.update(
            {
                f"{group_name}/measurements": _make_grd_group(
                    azimuth_time, ground_range
                ),
                f"{group_name}/quality/calibration": beta_nought.copy(deep=True),
                f"{group_name}/conditions/orbit": orbit.copy(deep=True),
                f"{group_name}/conditions/gcp": gcp.copy(deep=True),
            }
        )

    dt = xr.DataTree.from_dict(dt_nodes)

    dt.attrs["stac_discovery"] = {"bbox": [0.0, 50.0, 1.0, 51.0]}
    dt["S1A_IW_GRDH_TEST_VH"].attrs["other_metadata"] = {
        "image_annotation": {
            "image_information": {
                "range_pixel_spacing": 10.0,
                "slant_range_time": 1.0e-4,
                "product_first_line_utc_time": "2024-01-01T00:00:00",
                "azimuth_time_interval": 0.5,
                "azimuth_pixel_spacing": 20.0,
            }
        }
    }
    return dt


def _make_slc_group(
    azimuth_time: np.ndarray,
    slant_range_time: np.ndarray,
    scale: float,
) -> xr.Dataset:
    values = np.arange(
        1, len(azimuth_time) * len(slant_range_time) + 1, dtype="float32"
    ).reshape(len(azimuth_time), len(slant_range_time))
    return xr.Dataset(
        {
            "slc": xr.DataArray(
                values * scale,
                dims=("azimuth_time", "slant_range_time"),
                coords={
                    "azimuth_time": azimuth_time,
                    "slant_range_time": slant_range_time,
                    "line": ("azimuth_time", np.arange(len(azimuth_time))),
                    "pixel": ("slant_range_time", np.arange(len(slant_range_time))),
                },
            )
        }
    )


def make_s1_slc_datatree() -> xr.DataTree:
    azimuth_time_by_swath = {
        "IW1": np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:00:01",
                "2024-01-01T00:00:02",
            ],
            dtype="datetime64[ns]",
        ),
        "IW2": np.array(
            [
                "2024-01-01T00:00:00",
                "2024-01-01T00:00:01",
                "2024-01-01T00:00:02",
            ],
            dtype="datetime64[ns]",
        ),
    }
    slant_range_time_by_swath = {
        "IW1": np.array([0.0, 1.0, 2.0, 3.0], dtype="float32"),
        "IW2": np.array([1.0, 2.0, 3.0, 4.0], dtype="float32"),
    }
    axis = np.array(["x", "y", "z"])
    polarizations = ("VV", "VH")
    swaths = ("IW1", "IW2")

    beta_nought = xr.Dataset(
        {
            "beta_nought": xr.DataArray(
                np.ones((3, 4), dtype="float32"),
                dims=("azimuth_time", "slant_range_time"),
                coords={
                    "azimuth_time": azimuth_time_by_swath["IW1"],
                    "slant_range_time": slant_range_time_by_swath["IW1"],
                },
            )
        }
    )
    orbit = xr.Dataset(
        {
            "position": xr.DataArray(
                np.ones((3, 3), dtype="float32"),
                dims=("azimuth_time", "axis"),
                coords={
                    "azimuth_time": azimuth_time_by_swath["IW1"],
                    "axis": axis,
                },
            )
        }
    )
    burst_info = xr.Dataset(
        {
            "first_valid_sample": xr.DataArray(
                np.array([1, 1, -1], dtype="int32"),
                dims=("azimuth_time",),
                coords={"azimuth_time": azimuth_time_by_swath["IW1"]},
            ),
            "last_valid_sample": xr.DataArray(
                np.array([3, 3, -1], dtype="int32"),
                dims=("azimuth_time",),
                coords={"azimuth_time": azimuth_time_by_swath["IW1"]},
            ),
        }
    )

    dt_nodes = {}
    for mode_i, pol in enumerate(polarizations):
        for swath_i, swath in enumerate(swaths):
            group_name = f"S1A_IW_SLC_TEST_{pol}_{swath}_0"
            dt_nodes.update(
                {
                    f"{group_name}/measurements": _make_slc_group(
                        azimuth_time_by_swath[swath],
                        slant_range_time_by_swath[swath],
                        scale=float(mode_i + swath_i + 1),
                    ),
                    f"{group_name}/quality/calibration": beta_nought.assign_coords(
                        azimuth_time=azimuth_time_by_swath[swath],
                        slant_range_time=slant_range_time_by_swath[swath],
                    ).copy(deep=True),
                    f"{group_name}/conditions/orbit": orbit.assign_coords(
                        azimuth_time=azimuth_time_by_swath[swath]
                    ).copy(deep=True),
                    f"{group_name}/conditions/burst_info": burst_info.assign_coords(
                        azimuth_time=azimuth_time_by_swath[swath]
                    ).copy(deep=True),
                }
            )

    dt = xr.DataTree.from_dict(dt_nodes)
    dt.attrs["stac_discovery"] = {"bbox": [0.0, 50.0, 1.0, 51.0]}
    dt["S1A_IW_SLC_TEST_VH_IW1_0"].attrs["other_metadata"] = {
        "image_annotation": {
            "image_information": {
                "range_pixel_spacing": 10.0,
                "slant_range_time": 1.0e-4,
                "product_first_line_utc_time": "2024-01-01T00:00:00",
                "azimuth_time_interval": 0.5,
                "azimuth_pixel_spacing": 20.0,
            }
        }
    }
    return dt


def _make_owi_measurements(height: int, width: int):
    shape = (height, width)

    return xr.Dataset(
        {
            "wind_speed": xr.DataArray(
                np.ones(shape, dtype="float32"),
                dims=("height", "width"),
            ),
            "wind_direction": xr.DataArray(
                np.ones(shape, dtype="float32"),
                dims=("height", "width"),
            ),
        },
        coords=_make_lat_lon_coords(height, width),
    )


def _make_owi_quality(height: int, width: int):
    shape = (height, width)

    return xr.Dataset(
        {
            "calibration_constant": xr.DataArray(
                np.zeros(shape, dtype="float32"),
                dims=("height", "width"),
            ),
            "inversion_quality": xr.DataArray(
                np.zeros(shape, dtype="float64"),
                dims=("height", "width"),
                attrs={"_eopf_attrs": {"valid_min": 0, "valid_max": 3}},
            ),
            "wind_quality": xr.DataArray(
                np.zeros(shape, dtype="float64"),
                dims=("height", "width"),
                attrs={"_eopf_attrs": {"valid_min": 0, "valid_max": 3}},
            ),
            "percentage_bright_points": xr.DataArray(
                np.ones(shape, dtype="float32"),
                dims=("height", "width"),
                attrs={"_eopf_attrs": {"valid_min": 0, "valid_max": 100}},
            ),
        },
        coords=_make_lat_lon_coords(height, width),
    )


def _make_lat_lon_coords(height: int, width: int):
    y = np.arange(height - 1, -1, -1)
    x = np.arange(width)

    latitude = y[:, None] + x[None, :]
    longitude = -y[:, None] + x[None, :]

    return {
        "latitude": (
            ("height", "width"),
            latitude,
        ),
        "longitude": (
            ("height", "width"),
            longitude,
        ),
    }


def make_s1_ocn_datatree() -> xr.DataTree:
    height = 4
    width = 4

    product = "S1A_IW_OCN"

    return xr.DataTree.from_dict(
        {
            f"/owi/{product}/measurements": _make_owi_measurements(
                height,
                width,
            ),
            f"/owi/{product}/quality": _make_owi_quality(
                height,
                width,
            ),
        }
    )
