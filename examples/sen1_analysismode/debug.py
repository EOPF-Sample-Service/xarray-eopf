import xarray as xr
from xcube_resampling.gridmapping import GridMapping
from resample import resample_reg2irreg
import matplotlib.pyplot as plt

dem = xr.open_dataset("data/dem.zarr", chunks={})
dem = dem.assign_coords({"spatial_ref": dem.spatial_ref})
dem = dem[["dem"]]

path = "data/S1C_IW_GRDH_1SDV_20260331T043117_20260331T043142_007003_00E2E3_7002.zarr"
dt = xr.open_datatree(path, engine="zarr", chunks={})
group_VH = [x for x in dt.children if "VH" in x][0]
grd = dt[group_VH].measurements.to_dataset().rename({"grd": "vh"})
group_VV = [x for x in dt.children if "VV" in x][0]
grd["vv"] = dt[group_VV].measurements.to_dataset().grd

gcps = dt[group_VH].conditions.gcp.to_dataset()
gcps = gcps[["latitude", "longitude", "incidence_angle"]]
if gcps.longitude[0, 1] < gcps.longitude[0, 0]:
    gcps = gcps.isel(ground_range=slice(None, None, -1))
gcps_interp = gcps.interp(ground_range=grd.ground_range).chunk(dict(ground_range=2048))
gcps_interp = gcps_interp.interp(azimuth_time=grd.azimuth_time).chunk(
    dict(azimuth_time=2048)
)

target_gm = GridMapping.from_coords(
    gcps_interp.longitude, gcps_interp.latitude, "epsg:4326"
)
dem_interp = resample_reg2irreg(dem, target_gm)
dem_interp.dem[::20, ::20].plot(robust=True)
plt.show()
