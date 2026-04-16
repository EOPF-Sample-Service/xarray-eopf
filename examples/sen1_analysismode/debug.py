import datetime

import xarray as xr
import matplotlib.pyplot as plt
from sen1 import apply_analysis

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
    dem = dem.dem
    dem = dem.sel(lat=slice(37, 36.000001), lon=slice(22, 22.999999))
    dem = dem.chunk(dict(lat=1800, lon=1800))
    print(dem)

    # open Zarr Sen1 GRD Sample as DataTree
    path = (
        "data/S1C_IW_GRDH_1SDV_20260331T043117_20260331T043142_007003_00E2E3_7002.zarr"
    )
    dt = xr.open_datatree(path, engine="zarr", chunks={})

    print(datetime.datetime.now())
    rtc_11 = apply_analysis(dt, dem, footprint_scale_factor=(1, 1))
    print(datetime.datetime.now())
    print(rtc_11)
    rtc_33 = apply_analysis(dt, dem, footprint_scale_factor=(3, 3))
    print(datetime.datetime.now())
    print(rtc_33)
    fig, ax = plt.subplots(1, 2)
    rtc_11.vv.plot(ax=ax[0], vmin=0, vmax=0.2)
    rtc_33.vv.plot(ax=ax[1], vmin=0, vmax=0.2)
    print(datetime.datetime.now())
    plt.show()

    # rtc.vv.plot(vmin=0, vmax=0.4)
    # print(datetime.datetime.now())
    # plt.show()

    # gtc = do_terrain_correction(dt, dem)
    # rtc = do_terrain_correction(dt, dem, correct_radiometry="gamma_nearest")
    #
    # fig, ax = plt.subplots(1, 2)
    # gtc.vv.plot(ax=ax[0], vmin=0, vmax=1)
    # rtc.vv.plot(ax=ax[1], vmin=0, vmax=1)
    # plt.show()
