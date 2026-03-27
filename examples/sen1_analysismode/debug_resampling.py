import numpy as np
import xarray as xr
from xcube.core.store import new_data_store
from xcube_resampling import affine_transform_dataset
from rectify_new_sortin import rectify_dataset
from xcube_resampling.gridmapping import GridMapping
from datetime import datetime
import dask
import matplotlib.pyplot as plt

from dask.distributed import Client, LocalCluster


def main():
    # cluster = LocalCluster(
    #     n_workers=4,
    #     threads_per_worker=4,
    #     memory_limit="4GB",
    #     dashboard_address="8787",
    # )
    #
    # client = Client(cluster)
    # print(datetime.now(), client.dashboard_link)

    path = (
        "data/S1A_IW_GRDH_1SDV_20251209T165439_20251209T165508_062241_07CAD4_3B2A.zarr"
    )

    dt = xr.open_datatree(path, engine="zarr", chunks={})
    group_vh = [x for x in dt.children if "VH" in x][0]
    grd = dt[group_vh].measurements.to_dataset().rename({"grd": "vh"})
    group_vv = [x for x in dt.children if "VV" in x][0]
    grd["vv"] = dt[group_vv].measurements.to_dataset().grd
    print(datetime.now(), grd)
    grd = grd.assign_coords(
        azimuth_time=(grd.azimuth_time - grd.azimuth_time[0]) / np.timedelta64(1, "s")
    )
    grd = grd.assign_coords(ground_range=grd.ground_range / grd.sizes["ground_range"])
    gm = GridMapping.from_coords(
        grd.ground_range,
        grd.azimuth_time,
        "EPSG:3857",
        tile_size=(grd.vv.data.chunksize[1], grd.vv.data.chunksize[0]),
    )

    print(datetime.now(), "Start lat/lon prep")
    gcps = dt[group_vh].conditions.gcp.to_dataset()
    gcps = gcps[["latitude", "longitude"]]
    gcps = gcps.assign_coords(
        azimuth_time=(gcps.azimuth_time - gcps.azimuth_time[0]) / np.timedelta64(1, "s")
    )
    gcps = gcps.assign_coords(
        ground_range=gcps.ground_range / grd.sizes["ground_range"]
    )
    dummy_ds = gcps.copy()
    dummy_ds = dummy_ds.assign_coords(
        dict(
            azimuth_time=np.linspace(
                gcps.azimuth_time[0].item(),
                gcps.azimuth_time[-1].item(),
                gcps.sizes["azimuth_time"],
            ),
            ground_range=np.linspace(
                gcps.ground_range[0].item(),
                gcps.ground_range[-1].item(),
                gcps.sizes["ground_range"],
            ),
        )
    )
    gcps = gcps.interp_like(dummy_ds)
    gcps_gm = GridMapping.from_coords(gcps.ground_range, gcps.azimuth_time, "EPSG:3857")
    gcps_interp = affine_transform_dataset(
        gcps,
        target_gm=gm,
        source_gm=gcps_gm,
        interp_methods=1,
    )

    grd = grd.assign_coords(
        dict(
            latitude=gcps_interp.latitude,
            longitude=gcps_interp.longitude,
        )
    )

    grd = grd.isel(azimuth_time=slice(0, 2048), ground_range=slice(0, 4096))
    grd = grd.where(grd != 65535, np.nan)
    grd.vv.data /= 1000
    grd.vh.data /= 1000

    print(datetime.now(), "Start rectification")
    dask.config.set(scheduler="synchronous")
    grd_rect = rectify_dataset(grd, interp_methods="bilinear")
    grd_rect = grd_rect.drop_vars(["azimuth_time", "ground_range", "line", "pixel"])
    print(datetime.now(), "End rectification")
    print(grd_rect)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    grd.vv.plot(ax=ax[0], vmax=0.3)
    grd_rect.vv.plot(ax=ax[1], vmax=0.3)
    plt.show()
    # print(datetime.now(), "Start writing")
    # store = new_data_store("file", root="data")
    # store.write_data(
    #     grd_rect.isel(lat=slice(0, 4096), lon=slice(0, 8192)),
    #     "sen1_rectified_sub.zarr",
    #     replace=True,
    # )
    # print(datetime.now(), "Finish writing")


if __name__ == "__main__":
    main()
