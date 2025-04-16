The xarray backend for EOPF data products `"eopf-zarr"` has two modes of operation,
namely _analysis mode_ (the default) and _native mode_, which are described in 
the following. 

## Analysis Mode

This mode aims at representing the EOPF data products in an analysis-ready and 
convenient form using the `xarray` data models `DataTree` and `Dataset`. 
For this reason, it is the default mode of operation when using the `"eopf-zarr"` 
backend.

The data products provided in this mode use a unified grid mapping 
for all their data variables. This means that selected variables are 
spatially up-scaled or down-scaled as needed, so that the dataset can use a 
single shared pair of `x` and `y` coordinates in the returned datasets.

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

Returns a EOPF data product from Sentinel-1, -2, or -3 in a analysis-ready, convenient 
form. All bands and quality flags are resampled to a unified, user-provided resolution. 

Parameters `**params`:

- `resolution`: Target resolution for all spatial data variables / bands.
  Must be one of `10`, `20`, or `60`. 
- `spline_order`: Spline order to be used for resampling 
  spatial data variables / bands.
  Must be one of `0` (nearest neighbor), `1` (linear), `2` (bi-linear), or 
  `3` (cubic). 
- `variables`: Variables to include in the dataset. Can be a name or regex pattern 
  or iterable of the latter.
- `product_type`:  Product type name, such as `"S2B_MSIL1C"`. 
  Only required if `filename_or_obj` is not a path or URL 
  that refers to a product path adhering to EOPF naming conventions.


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

This function is currently not implemented for the analysis mode
and will raise a `NotImplementedError`.

## Native Mode

The aim of this mode is to represent EOPF data products without modification 
using the `xarray` data models `DataTree` and `Dataset`. Content and structure 
of the original data products are preserved to a maximum extend.

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

The main use case for this function is to allow passing an EOPF data product 
where the type `xr.Dataset` is expected (not `xr.DataTree`) and where the naming of 
dimensions and variables is not an issue.

Parameters `**params`:

- `group_sep`: Separator string used to concatenate groups names 
  to create prefixes for unique variable and dimension names.
  Defaults to the underscore character (`"_"`).


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

This function currently returns the result of calling 
`xr.open_datatree(filename_or_obj, engine="zarr", **kwargs)`.  

