The xarray backend for EOPF data products `"eopf-zarr"` has two modes of operation,
namely _analysis mode_ and _native mode_, which are described in the following. 

## Analysis Mode

This mode aims at representing the EOPF data products in an analysis-ready and 
convenient form using the `xarray` data models `DataTree` and `Dataset`. 

By default, data products are provided using a common single grid-mapping
for all data variables. That is, spatial up- and downscaling is applied
to selected variables in order to use only a single pair of `x` and `y` 
coordinates in the returned datasets.

### Function `open_datatree()`

Synopsis: 

```python
datatree = xr.open_datatree(
    filename_or_obj, 
    engine="eopf-zarr", 
    op_mode="analysis", 
    **kwargs
)
```

_Not implemented yet._

### Function `open_dataset()`

Synopsis:  

```python
dataset = xr.open_dataset(
    filename_or_obj, 
    engine="eopf-zarr", 
    op_mode="analysis", 
    **kwargs
)
```

Works basically as `open_datatree()` but using flattened Zarr groups, e.g., groups 
`r10m`, `r20m`, `r60m` in Sentinel 2 MSI products.
Flattening includes renaming variables and dimensions by prefixing them using the 
concatenated names of nested groups. 

Parameters `**params`:

- `resolution`: Target resolution for all spatial data variables / bands.
  Must be one of `10`, `20`, or `60`. 
- `spline_order`: Spline order to be used for resampling 
  spatial data variables / bands.
  Must be one of `0` (nearest neighbor), `1` (linear), `2` (bi-linear), or 
  `3` (cubic). 
- `variables`: Variables to include in the dataset. Can be a name or regex pattern 
  or iterable of the latter.
- `product_type_name`:  Product type name, such as `"S2B_MSIL1C"`. 
  Only required if `filename_or_obj` is not a path or URL 
  that refers to a product path adhering to EOPF naming conventions.


## Native Mode

The aim of this mode is to represent EOPF data products without modification 
using the `xarray` data models `DataTree` and `Dataset`. Content and structure 
of the original data products are preserved to a maximum extend.

### Function `open_datatree()`

Synopsis:  

```python
datatree = xr.open_datatree(
    filename_or_obj, 
    engine="eopf-zarr", 
    op_mode="native", 
    **kwargs
)
```

Opens a data product as-is including Zarr groups and returns a data tree object.

### Function `open_dataset()`

Synopsis:  

```python
dataset = xr.open_dataset(
    filename_or_obj, 
    engine="eopf-zarr", 
    op_mode="native", 
    **kwargs
)
```

Returns a "flattened" version of the data tree returned by `xr.open_datatree()` 
in native mode. Groups are removed by turning their contents into individual datasets
and merging them into one. Variables and dimensions are prefixed using their original 
group paths to make them unique in the returned dataset. For example, the variable 
`b02` found in the group `measurements/reflectance/r10m` will be renamed to 
`measurements_reflectance_r10m_b02` using the default underscore group separator.
The separator character is configurable by setting the `group_sep` parameter.

Parameters `**params`:

- `group_sep`: Separator string used to concatenate groups names 
  to create prefixes for unique variable and dimension names.
  Defaults to the underscore character (`"_"`).
