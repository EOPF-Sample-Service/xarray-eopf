import xarray as xr

path = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202510-s03s"
    "lsrbt-global/15/products/cpm_v256/S3A_SL_1_RBT____20251015T042906_"
    "20251015T043206_20251015T053822_0179_131_290_0360_PS1_O_NR_004.zarr"
)

ds = xr.open_dataset(
    path, engine="eopf-zarr", chunks={}, variables=["s1_radiance_an", "s7_bt_io"]
)
print(ds)
