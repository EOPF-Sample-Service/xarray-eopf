import pystac_client
import xarray as xr

catalog = pystac_client.Client.open("https://stac.core.eopf.eodc.eu")
items = list(
    catalog.search(
        collections=["sentinel-3-olci-l1-efr"],
        bbox=[7.2, 44.5, 7.4, 44.7],
        datetime=["2026-03-13", "2026-03-13"],
    ).items()
)
item = items[0]

ds = xr.open_dataset(
    item.assets["product"].href,
    engine="eopf-zarr",
    resolution=0.01,
    bbox=[5, 40, 15, 48],
    chunks={},
)
print(ds)
