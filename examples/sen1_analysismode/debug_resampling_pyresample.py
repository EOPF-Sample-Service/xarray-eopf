import numpy as np
import dask.array as da
from xarray import DataArray
from pyresample.bilinear import XArrayBilinearResampler
from pyresample import geometry
from datetime import datetime

print(datetime.now())

target_def = geometry.AreaDefinition(
    "areaD",
    "Europe (3km, HRV, VTC)",
    "areaD",
    {
        "a": "6378144.0",
        "b": "6356759.0",
        "lat_0": "50.00",
        "lat_ts": "50.00",
        "lon_0": "8.00",
        "proj": "stere",
    },
    8000,
    8000,
    [-1370912.72, -909968.64, 1029087.28, 1490031.36],
)
data = DataArray(
    da.from_array(np.fromfunction(lambda y, x: y * x, (10000, 10000))), dims=("y", "x")
)
lons = da.from_array(np.fromfunction(lambda y, x: 3 + x * 0.1, (10000, 10000)))
lats = da.from_array(np.fromfunction(lambda y, x: 75 - y * 0.1, (10000, 10000)))
source_def = geometry.SwathDefinition(lons=lons, lats=lats)
resampler = XArrayBilinearResampler(source_def, target_def, 30e3)
print(datetime.now())
result = resampler.resample(data)
print(datetime.now())
print(result)
print(datetime.now())
