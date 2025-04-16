# EOPF xarray backend

`xarray-eopf` is a Python package that enhances [xarray](https://docs.xarray.dev/en/stable/user-guide/io.html) by a new backend 
named `"eopf-zarr"`. This backend allows for reading the [ESA EOPF data](https://eopf.copernicus.eu/eopf-products-and-adfs/) products
in Zarr format and representing them using analysis ready data models.

## Overview

After installation of the package, you can open EOPF data products using the
usual xarray top-level functions `open_dataset()` that provides
a [`Dataset`](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) object:

```python
dataset = xr.open_dataset(url_or_path, engine="eopf-zarr")
```

and `open_datatree()` that provides a [`DataTree`](https://docs.xarray.dev/en/stable/generated/xarray.DataTree.html) 
object including groups:

```python
datatree = xr.open_datatree(url_or_path, engine="eopf-zarr")
```

## Features 

> **IMPORTANT**  
> The `xarray-eopf` package is in a preliminary development state.
> The following features are only partly or not at all implemented yet.


The backend supports two distinct modes of operation, `"native"` and `"analysis"`:

- `op_mode="analysis"` - using the _analysis mode_, the returned `Dataset` 
  or `DataTree` objects attempt to be user-friendly and analysis-ready to a maximum
  extend. Certain preprocessing steps will be applied depending on the specific 
  Sentinel product towards an analysis-ready data model and content. For example, 
  for Sentinel-2, we will support the selection of specific bands and resampling to a 
  desired spatial resolution. For Sentinel-3 we will allow for rectifying the images 
  in satellite coordinates to a user-provided, common CRS. It can be used as shown
  in the examples below:
```python
dataset = xr.open_dataset(url_or_path, engine="eopf-zarr", op_mode="analysis")
datatree = xr.open_datatree(url_or_path, engine="eopf-zarr", op_mode="analysis")
```

- `op_mode="native"` - using the _native mode_, the returned `Dataset` or `DataTree`
  objects try to serve as a 1:1 representation of the actual Zarr product structure and
  content with either none or minimal preprocessing applied. It can be used as shown
  in the examples below:
```python
dataset = xr.open_dataset(url_or_path, engine="eopf-zarr", op_mode="native")
datatree = xr.open_datatree(url_or_path, engine="eopf-zarr", op_mode="native")
```

More information on the two modes are given in the [User Guide](guide.md)

Data variables will always be represented as chunked Dask arrays for 
efficient out-of core computations and visualisations.

The package has minimal core dependencies: `xarray`, `zarr`, and `dask`.
Packages for accessing remote filesystems are optional, e.g., you will need `s3fs`
if you need to access EOPF data products in S3-compatible remote object storages.

## License

The package is open source and released under the 
[Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0.html) license. :heart:

