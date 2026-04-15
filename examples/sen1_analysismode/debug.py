import datetime

import numpy as np
from typing import Any
import xarray as xr
import matplotlib.pyplot as plt
from sen1 import terrain_correct, SPEED_OF_LIGHT
from dask.distributed import Client, LocalCluster


def azimuth_slant_range_grid(
    dt: xr.DataTree,
    grouping_area_factor: tuple[float, float] = (3.0, 3.0),
) -> dict[str, Any]:

    group_VH = [x for x in dt.children if "VH" in x][0]
    attrs = dt[f"{group_VH}"].attrs["other_metadata"]["image_annotation"][
        "image_information"
    ]

    slant_range_spacing_m = attrs["range_pixel_spacing"] * grouping_area_factor[1]
    slant_range_time_interval_s = (
        slant_range_spacing_m * 2 / SPEED_OF_LIGHT  # ignore type
    )

    grid_parameters: dict[str, Any] = {
        "slr0": attrs["slant_range_time"],
        "d_slr": slant_range_time_interval_s,
        "spacing_slr": slant_range_spacing_m,
        "az0": np.datetime64(attrs["product_first_line_utc_time"]),
        "d_az": attrs["azimuth_time_interval"] * grouping_area_factor[0],
        "spacing_az": attrs["azimuth_pixel_spacing"] * grouping_area_factor[0],
    }
    return grid_parameters


if __name__ == "__main__":
    # cluster = LocalCluster(
    #     n_workers=4,
    #     threads_per_worker=1,
    #     memory_limit="8GB",
    #     dashboard_address="8787",
    # )
    # client = Client(cluster)
    # print(client.dashboard_link)

    # open DEM
    dem = xr.open_dataset("data/dem.zarr", chunks={})
    dem = dem.assign_coords({"spatial_ref": dem.spatial_ref})
    dem = dem[["dem"]]
    dem = dem.sel(lat=slice(37, 36.000001), lon=slice(22, 22.999999))
    dem = dem.chunk(dict(lat=1800, lon=1800))
    print(dem)

    # open Zarr Sen1 GRD Sample as DataTree
    path = (
        "data/S1C_IW_GRDH_1SDV_20260331T043117_20260331T043142_007003_00E2E3_7002.zarr"
    )
    dt = xr.open_datatree(path, engine="zarr", chunks={})

    # radiometric calibration
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

    orbit = dt[f"{group_VH}/conditions/orbit"].to_dataset()
    sat_position = orbit["position"]

    gcp = dt[f"{group_VH}/conditions/gcp"].to_dataset()
    time_slr_gcp = gcp["slant_range_time_gcp"]

    grid_params = azimuth_slant_range_grid(dt)

    print(datetime.datetime.now())
    rtc = terrain_correct(
        beta_nought, time_slr_gcp, sat_position, dem, grid_params=grid_params
    )
    print(rtc)
    print(datetime.datetime.now())
    rtc.vv.plot(vmin=0, vmax=0.4)
    print(datetime.datetime.now())
    plt.show()

    # gtc = do_terrain_correction(dt, dem)
    # rtc = do_terrain_correction(dt, dem, correct_radiometry="gamma_nearest")
    #
    # fig, ax = plt.subplots(1, 2)
    # gtc.vv.plot(ax=ax[0], vmin=0, vmax=1)
    # rtc.vv.plot(ax=ax[1], vmin=0, vmax=1)
    # plt.show()
