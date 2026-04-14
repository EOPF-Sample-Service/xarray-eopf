import xarray as xr
import matplotlib.pyplot as plt
from sen1 import do_terrain_correction

if __name__ == "__main__":
    # open DEM
    dem = xr.open_dataset("data/dem.zarr", chunks={})
    dem = dem.assign_coords({"spatial_ref": dem.spatial_ref})
    dem = dem[["dem"]]
    dem = dem.sel(lat=slice(37, 36.0001), lon=slice(22, 22.9999))

    # open Zarr Sen1 GRD Sample as DataTree
    path = (
        "data/S1C_IW_GRDH_1SDV_20260331T043117_20260331T043142_007003_00E2E3_7002.zarr"
    )
    dt = xr.open_datatree(path, engine="zarr", chunks={})

    rtc = do_terrain_correction(dt, dem, correct_radiometry="gamma_nearest")
    rtc.vv.plot(vmin=0, vmax=0.4)
    plt.show()

    # gtc = do_terrain_correction(dt, dem)
    # rtc = do_terrain_correction(dt, dem, correct_radiometry="gamma_nearest")
    #
    # fig, ax = plt.subplots(1, 2)
    # gtc.vv.plot(ax=ax[0], vmin=0, vmax=1)
    # rtc.vv.plot(ax=ax[1], vmin=0, vmax=1)
    # plt.show()
