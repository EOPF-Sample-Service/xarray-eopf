import xarray as xr

path = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202508-s03olcefr/19/"
    "products/cpm_v256/S3B_OL_1_EFR____20250819T074058_20250819T074358_"
    "20250819T092155_0179_110_106_3420_ESA_O_NR_004.zarr"
)

dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native", chunks={})
print(dt)
