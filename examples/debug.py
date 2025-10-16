import xarray as xr

path = (
    "https://objects.eodc.eu/e05ab01a9d56408d82ac32d69a5aae2a:202510-s03olcefr-"
    "global/16/products/cpm_v256/S3B_OL_1_EFR____20251016T054111_20251016T054411_"
    "20251016T072931_0180_112_162_2520_ESA_O_NR_004.zarr"
)

ds = xr.open_dataset(path, engine="eopf-zarr", chunks={})
print(ds)
