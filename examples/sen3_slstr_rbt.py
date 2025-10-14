import xarray as xr

path = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202510-s03slsrbt-global/09/products/cpm_v256/"
    "S3A_SL_1_RBT____20251009T011313_20251009T011613_20251009T030735_0179_131_202_3420_PS1_O_NR_004.zarr"
)

ds = xr.open_dataset(
    path, engine="eopf-zarr", chunks={}, variables=["s1_radiance_an", "s7_bt_io"]
)
print(ds)
