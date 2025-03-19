The xarray backend for EOPF data products `"eopf-zarr"` has two modes of operation,
namely _analysis mode_ and _native mode_, which are described in th e following. 

## Analysis Mode

This mode aims at representing the EOPF data products in an analysis-ready and 
convenient form using the `xarray` data models `DataTree` and `Dataset`. 

By default, data products are provided using a common single grid-mapping
for all data variables. That is, spatial up- and downscaling is applied
to selected variables in order to use only a single pair of `x` and `y` 
coordinates in the returned datasets.

### `xr.open_datatree(filename_or_obj, engine="eopf-zarr", mode="analysis", **kwargs)`



### `xr.open_dataset(filename_or_obj, engine="eopf-zarr", mode="analysis", **kwargs)` 

Basically as `xr.open_datatree` but using flattened Zarr groups, e.g., groups 
`r10m`, `r20m`, `r60m` in Sentinel 2 MSI products.
Flattening includes renaming variables and dimensions by prefixing them using the 
concatenated names of nested groups. 

Parameters `**params`:

- `sep: str = "_"`: Separator string used to concatenate groups names.
  Defaults to the underscore character.


## Native Mode

The aim of this mode is to represent EOPF data products without modification 
using the `xarray` data models `DataTree` and `Dataset`. Content and structure 
of the original data products are preserved to a maximum extend.

### `xr.open_datatree(filename_or_obj, engine="eopf-zarr", mode="native", **kwargs)`

Opens a data product as-is including Zarr groups.

### `xr.open_dataset(filename_or_obj, engine="eopf-zarr", mode="native", **kwargs)`

Basically as `xr.open_datatree` but using flattened Zarr groups, e.g., groups 
`r10m`, `r20m`, `r60m` in Sentinel 2 MSI products.
Flattening includes renaming variables and dimensions by prefixing them using the 
concatenated names of nested groups. Group separator character is `_` by default.  

Parameters `**params`:

- `sep: str = "_"`: Separator string used to concatenate groups names.
  Defaults to the underscore character.
