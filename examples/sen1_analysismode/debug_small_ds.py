import xarray as xr
import numpy as np
import pyproj
import dask

from xcube_resampling import rectify_dataset
from xcube_resampling.gridmapping import GridMapping

CRS_WGS84 = pyproj.crs.CRS(4326)


def create_2x2_dataset_with_irregular_coords():
    lon = np.array([[1.0, 6.0], [0.0, 2.0]])
    lat = np.array([[56.0, 53.0], [52.0, 50.0]])
    rad = np.array([[1.0, 2.0], [3.0, 4.0]])
    return xr.Dataset(
        dict(rad=xr.DataArray(rad, dims=("y", "x"))),
        coords=dict(
            lon=xr.DataArray(lon, dims=("y", "x")),
            lat=xr.DataArray(lat, dims=("y", "x")),
        ),
    )


dask.config.set(scheduler="synchronous")
source_ds = create_2x2_dataset_with_irregular_coords()
target_gm = GridMapping.regular(
    size=(13, 13), xy_min=(0, 50), xy_res=0.5, crs=CRS_WGS84
)
target_ds = rectify_dataset(source_ds, target_gm=target_gm, interp_methods=0)
print(target_ds.rad.values)
