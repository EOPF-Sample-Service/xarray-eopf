## `mode="native"`

The aim of this mode is to represent EOPF data products without modification 
using the `xarray` data models `DataTree` and `Dataset`. Content and structure 
of the original data products are preserved to a maximum extend.

- `xr.open_datatree(..., mode="native")`: No modifications needed; Zarr is represented 
  as an `xr.DataTree` as-is.

- `xr.open_dataset(..., mode="native")`: Basically as `open_datatree` but using flattened
  Zarr groups (e.g., `r10m`, `r20m`, `r60m` in Sentinel 2 MSI products).
  Flattening includes renaming variables and dimensions by prefixing them using the 
  concatenated names of nested groups. Separator character default is `_`.  

## `mode="analysis"`

- `xr.open_datatree(..., mode="analysis")`:

- `xr.open_dataset(..., mode="analysis")`: 
