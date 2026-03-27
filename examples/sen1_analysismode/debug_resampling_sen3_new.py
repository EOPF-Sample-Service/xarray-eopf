import xarray as xr
from datetime import datetime

from xcube_resampling import rectify_dataset
import matplotlib.pyplot as plt

bbox = [15.98, 58.08, 16, 58.1]
source_ds = xr.open_zarr("./data/S3-OLCI-L2A.zarr.zip", consolidated=False)
print(datetime.now())
target_ds = rectify_dataset(source_ds, interp_methods="nearest")
# print(target_ds.rtoa_8.compute())
plt.imshow(target_ds.rtoa_8.values, vmin=0.0, vmax=0.3)
print(datetime.now())
# target_ds.rtoa_8.plot(vmin=0.0, vmax=0.3)
plt.show()
