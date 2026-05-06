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
