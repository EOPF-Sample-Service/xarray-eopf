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
    dem = dem.sel(lat=slice(36.5, 36.4), lon=slice(22.42, 22.52))
    # dem = dem.sel(lat=slice(37, 36.000001), lon=slice(22, 22.999999))
    # dem = dem.chunk(dict(lat=1800, lon=1800))
    print(dem)

    # open Zarr Sen1 GRD Sample as DataTree
    path = (
        "data/S1C_IW_GRDH_1SDV_20260331T043117_20260331T043142_007003_00E2E3_7002.zarr"
    )
    dt = xr.open_datatree(path, engine="zarr", chunks={})
    group_VV = [x for x in dt.children if "VV" in x][0]
    grd = dt[group_VV].measurements.to_dataset().rename({"grd": "vv"})
    grd.vv[::20, ::20].plot(vmax=300)
    plt.title("Raw data")
    plt.show()

    rtc = apply_analysis(dt, dem)
    gtc = apply_analysis(dt, dem, apply_rtc=False)

    fig, ax = plt.subplots(1, 2)
    gtc.vv.plot(ax=ax[0], vmin=0, vmax=0.4)
    ax[0].set_title("Geographic terrain correction")
    rtc.vv.plot(ax=ax[1], vmin=0, vmax=0.4)
    ax[1].set_title("Radiometric terrain correction")
    plt.show()
