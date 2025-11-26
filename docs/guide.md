The xarray backend for EOPF data products `"eopf-zarr"` has two modes of operation,
namely _analysis mode_ (the default) and _native mode_, which are described in 
the following. 

---

## Analysis Mode

This mode aims at representing the EOPF data products in an analysis-ready and 
convenient form using the `xarray` data models `DataTree` and `Dataset`. 
For this reason, it is the default mode of operation when using the `"eopf-zarr"` 
backend.

The data products provided in this mode use a unified regular grid mapping 
for all their data variables. This means that selected variables are 
spatially resampled as needed, so that the dataset can use a 
single shared pair of 1d `x` and `y` coordinates in the returned datasets.


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

Parameters `**kwargs`:

- `resolution`: Target resolution for all spatial data variables / bands.
- `crs`: Coordinate reference system of the output dataset. If not provided, a
  mission-specific default CRS is used (see the respective mission sections below).
- `bbox`: Bounding box `[west, south, east, north]` used for spatial subsetting;
  coordinates must be in the same CRS as `crs`.
- `interp_methods`: for upsampling / interpolating
  spatial data variables. Can be a single interpolation method for all
  variables or a dictionary mapping variable names or dtypes to
  interpolation method. Supported methods include:

    - `0` (nearest neighbor)
    - `1` (linear / bilinear)
    - `"nearest"`
    - `"triangular"`
    - `"bilinear"`

  The default is `0` for integer arrays (e.g. Sentinel-2 L2A SCL),
  else `1`. For more information view [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/guide/#spatial-resampling-algorithms).
- `agg_methods`: Optional aggregation methods to be used for downsampling
  spatial data variables / bands. Can be a single method for all variables or 
  a dictionary mapping variable names or dtypes to methods. Supported methods include:
    `"center"`, `"count"`, `"first"`, `"last"`, `"max"`, `"mean"`, `"median"`, 
    `"mode"`, `"min"`, `"prod"`, `"std"`, `"sum"`, and `"var"`.
  Defaults to `"center"` for integer arrays (e.g. Sentinel-2 L2A SCL), else `"mean"`.
  For more information view [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/guide/#spatial-resampling-algorithms).
- `variables`: Variables to include in the dataset. Can be a name or regex pattern 
  or iterable of the latter.
- `product_type`: Product type name, such as `"MSIL1C"`. 
  Only required if `filename_or_obj` is not a path or URL that refers to a product 
  path adhering to EOPF naming conventions.

The spatial resampling of datasets is performed using [xcube-resampling](https://xcube-dev.github.io/xcube-resampling/).
Further explanation of the meaning and usage of these parameters for each Sentinel 
mission is provided in [Remarks on Specific Sentinel Missions](#remark-to-specific-sentinel-missions-).

### Remarks on Specific Sentinel Missions

#### Sentinel-1

Support for Sentinel-1 analysis mode will be added in a future release.


#### Sentinel-2

Sentinel-2 provides multi-spectral imagery at different native resolutions:

- **10m**: `b02`, `b03`, `b04`, `b08`  
- **20m**: `b05`, `b06`, `b07`, `b8a`, `b11`, `b12`  
- **60m**: `b01`, `b09`, `b10`  

The analysis mode enables resampling between these different resolutions, bringing 
bands from multiple resolutions onto the same grid using [affine transformation via xcube-resampling](https://xcube-dev.github.io/xcube-resampling/guide/#1-affine-transformation).

**Suported Products:**

- [Sentinel-2 Level-1C](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-2-l1c)
- [Sentinel-2 Level-2A](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-2-l2a)

**Supported Variables**

- **Surface reflectance bands**:
  `b01`, `b02`, `b03`, `b04`, `b05`, `b06`, `b07`, `b08`, `b8a`, `b09`, `b11`, `b12`
- **Classification/Quality layers** (L2A only):
  `cld`, `scl`, `snw`

**Specific Sentinel-2 parameters `**kwargs`:**

- `variables`: Select specific spectral bands using the names listed above in
  *Supported Variables*. Common spectral band names from the [STAC EO extension](https://github.com/stac-extensions/eo?tab=readme-ov-file#common-band-names) are also supported for Sentinel-2 analysis mode.
- `crs`: Coordinate reference system of the output dataset.
  If not specified, the UTM grid of the native data is used.
- `resolution`: Target resolution for all spatial variables/bands.
  Choose 10, 20, or 60 meters to minimize resampling and retain some of the native
  data resolution.

Examples:  
- [Example notebook - open-sen2.ipynb](https://github.com/EOPF-Sample-Service/xarray-eopf/blob/main/examples/open-sen2.ipynb)  
- [Notebook gallery - Introduction to xarray-eopf backend](https://eopf-sample-service.github.io/eopf-sample-notebooks/introduction-xarray-eopf-plugin/)  
- [Webinar 3 - Access EOPF Zarr Products with the New xarray EOPF Backend](https://zarr.eopf.copernicus.eu/webinars/webinar-3-access-eopf-zarr-products-with-the-new-xarray-eopf-backend/)  


#### Sentinel-3

Sentinel-3 products are provided on their **native grid mapping**, where each pixel 
is defined by a latitude/longitude pair, forming a **2D irregular grid**.

The analysis mode applies the [rectification algorithm in xcube-resampling](https://xcube-dev.github.io/xcube-resampling/guide/#3-rectification)
to transform the irregular dataset into a **regular grid** with 1D latitude/longitude
coordinates.

For SLSTR products, a terrain correction is applied during this process. This is
necessary because the original geolocation is corrected only for Earth curvature,
but not for terrain variability caused by topography. See the
[SLSTR product description](https://sentiwiki.copernicus.eu/web/slstr-products)
for details.

For OLCI products, no additional terrain correction is required, as it is already
incorporated in the Level-1 data. See the [OLCI Level-1 product description](https://sentiwiki.copernicus.eu/web/olci-products#OLCIProducts-L1BProducts-ObservationModeS3-OLCI-Products-L1B-OM)
for details.

**Suported Products:**

- [Sentinel-3 OLCI Level-1 EFR](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-3-olci-l1-efr)
- [Sentinel-3 OLCI Level-1 ERR](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-3-olci-l1-err)
- [Sentinel-3 OLCI Level-2 LFR](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-3-olci-l2-lfr)
- [Sentinel-3 SLSTR Level-1 RBT](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-3-slstr-l1-rbt)
- [Sentinel-3 SLSTR Level-2 LST](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-3-slstr-l2-lst)


**Supported Variables:**
- `sentinel-3-olci-l1-efr`:
  `oa01_radiance`, `oa02_radiance`, `oa03_radiance`, `oa04_radiance`, `oa05_radiance`,
  `oa06_radiance`, `oa07_radiance`, `oa08_radiance`, `oa09_radiance`, `oa10_radiance`,
  `oa11_radiance`, `oa12_radiance`, `oa13_radiance`, `oa14_radiance`, `oa15_radiance`,
  `oa16_radiance`, `oa17_radiance`, `oa18_radiance`, `oa19_radiance`, `oa20_radiance`,
  `oa21_radiance`
- `sentinel-3-olci-l2-lfr`:
  `gifapar`, `iwv`, `otci`, `rc681`, `rc865`
- `sentinel-3-slstr-l1-rbt`:
  `s1_radiance_an`, `s2_radiance_an`, `s3_radiance_an`, `s4_radiance_an`,
  `s5_radiance_an`, `s6_radiance_an`, `s1_radiance_ao`, `s2_radiance_ao`,
  `s3_radiance_ao`, `s4_radiance_ao`, `s5_radiance_ao`, `s6_radiance_ao`,
  `s4_radiance_bn`, `s5_radiance_bn`, `s6_radiance_bn`, `s4_radiance_bo`,
  `s5_radiance_bo`, `s6_radiance_bo`, `f1_bt_fn`, `f1_bt_fo`, `f2_bt_in`,
  `f2_bt_io`, `s7_bt_in`, `s8_bt_in`, `s9_bt_in`, `s7_bt_io`, `s8_bt_io`,
  `s9_bt_io`
- `sentinel-3-slstr-l2-lst`:
  `lst`

**Specific Sentinel-2 parameters `**kwargs`:**

- `variables`: Select variables using the names listed above in *Supported Variables*.
- `crs`: Coordinate reference system of the output dataset.
  If not specified, [EPSG:4326](https://epsg.io/4326) is used.
- `resolution`: Target resolution for all spatial variables/bands.
  If not specified, the default is set per product:

    - Sentinel-3 OLCI Level-1 EFR: 300 meter
    - Sentinel-3 OLCI Level-1 ERR: 1200 meter
    - Sentinel-3 OLCI Level-2 LFR: 300 meter
    - Sentinel-3 SLSTR Level-1 RBT: 500 meter (1000 meter if selected variables come from F- or I-stripe)
    - Sentinel-3 SLSTR Level-2 LST: 1000 meter

Example:  
- [Example notebook (open-sen3.ipynb)](https://github.com/EOPF-Sample-Service/xarray-eopf/blob/main/examples/open-sen3.ipynb)  



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

---

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

Parameters `**kwargs`:

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

