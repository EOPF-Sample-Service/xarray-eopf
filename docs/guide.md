The xarray backend for EOPF data products `"eopf-zarr"` has two modes of operation,
namely _analysis mode_ (the default) and _native mode_, which are described in 
the following.

An introductory example notebook is available at : 
- [Docs - Intoduction to the xarray EOPF backend](https://eopf-sample-service.github.io/xarray-eopf/examples/introduction/)
- [Notebook Gallery - Intoduction to the xarray EOPF backend](https://eopf-sample-service.github.io/eopf-sample-notebooks/introduction/)

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

Returns an EOPF data product from Sentinel-1, -2, or -3 in an analysis-ready, convenient 
form. Common parameters are:

- `resolution`: Target resolution for all spatial data variables / bands.
- `crs`: Coordinate reference system of the output dataset. Can be provided as a 
  `str` or a `pyproj.CRS` object. If a string is given, it will be parsed using 
  [`pyproj.crs.CRS.from_string`](https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_string). 
  If not specified, a mission-specific default CRS will be applied (see the respective 
  mission sections below).
- `bbox`: Bounding box `[west, south, east, north]` used for spatial subsetting;
  coordinates must be in the same CRS as `crs`.
- `variables`: Variables to include in the dataset. Can be a name or regex pattern 
  or iterable of the latter.
- `product_type`: Product type name, such as `"MSIL1C"`. 
  Only required if `filename_or_obj` is not a path or URL that refers to a product 
  path adhering to EOPF naming conventions.

Additional parameters specific to each Sentinel mission are described below.

#### Remarks on Specific Sentinel Missions

Processing workflows differ significantly across Sentinel-1 product types. Therefore, 
each product family is documented in its own dedicated section.

##### Sentinel-1 Level-1 GRD

> Note: Support for Sentinel-1 GRD products in analysis mode is 
> currently experimental and undergoing validation. Some conversion parameters 
> are missing in the new EOPF product, which are currently estimated. Newer EOPF
> product verison will include these parameters. 

Sentinel-1 Level-1 GRD data is provided in radar geometry, defined by the coordinates
(`azimuth_time`, `ground_range`). To transform this data into an
**analysis-ready dataset**, the following processing steps are applied:

1. **Radiometric Calibration:** Raw pixel values (DN) are converted into physically
   meaningful backscatter values using the `beta_nought` calibration lookup table (LUT).
2. **Geometric Terrain Correction (GTC):** Using a Digital Elevation Model (DEM),
   the processor performs **inverse geocoding** by solving the zero-Doppler equation
   based on satellite orbit information and terrain elevation. This step maps the
   data from radar geometry to a georeferenced grid.
3. **Radiometric Terrain Correction (RTC):** (Optional) RTC compensates for 
   terrain-induced radiometric distortions such as foreshortening, layover, 
   and slope-dependent brightness variations.

📖 [D. Small, *Flattening Gamma: Radiometric Terrain Correction for SAR Imagery*](https://ieeexplore.ieee.org/document/5752845)

**Supported Products:**

- [Sentinel-1 Level-1 GRD](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-1-l1-grd)

**Supported Variables**

- **Polarization bands**:  
  `vv`, `vh`, `hh`, `hv` *(each GRD product contains only a subset of these bands)*

**Specific Sentinel-1 Level-1 GRD parameters `**kwargs`:**

- `crs`: Coordinate reference system of the output dataset. Can be provided as a 
  `str` or a `pyproj.CRS` object. If a string is given, it will be parsed using 
  [`pyproj.crs.CRS.from_string`](https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_string).
  If not specified, [EPSG:4326](https://epsg.io/4326) is used.
- `resolution`: Target resolution for all spatial variables expressed in the units 
  of the specified `crs`. If not specified, the resolution is derived (in degrees) 
  from the CopDEM (30 m).
- `dem`: Digital Elevation Model (DEM) as a CF-compliant `xarray.DataArray` used for
  terrain correction. If provided, the parameters `crs`, `bbox`, and `resolution` are 
  ignored, and the target grid is derived from the DEM. If not provided, the
  [CopDEM COG (30 m)](https://browser.stac.dataspace.copernicus.eu/collections/cop-dem-glo-30-dged-cog)  
  is automatically retrieved via the CDSE STAC API. This requires
  [CDSE S3 credentials](https://documentation.dataspace.copernicus.eu/APIs/S3.html#generate-secrets).
- `apply_rtc`: Enable or disable radiometric terrain correction (RTC). Default is `True`.
- `interp_methods`: Interpolation method used during GTC and RTC. Supported methods:
  `nearest`, `bilinear`.
- `footprint_scale_factor`: Defines how radar pixels contribute to the output grid.
  Default: `(3.0, 3.0)`, accounting for resolution differences (e.g., ~10 m GRD
  vs. ~30 m DEM). 
- `cache_uri`: Temporary path used to store intermediate results from the
  backward geocoding step in the Sentinel-1 processing workflow. The cache is 
  automatically removed when the Python process exits. If None, a temporary 
  directory with a unique UUID-based name is created.

Examples:  

- [Docs – Sentinel-1 Analysis Mode](https://eopf-sample-service.github.io/xarray-eopf/examples/sentinel_1_analysis/)

##### Sentinel-1 Level-1 SLC

> Note: Support for Sentinel-1 SLC products in analysis mode is
> currently experimental and undergoing validation.

Sentinel-1 Level-1 SLC data is provided in radar geometry, defined by the coordinates
(`azimuth_time`, `slant_range_time`) and organized in bursts and swaths. To transform
this data into an **analysis-ready dataset**, the following processing steps are applied:

1. **Radiometric Calibration:** For each burst complex SLC measurements for are 
   converted into `beta0` backscatter values using the `beta_nought` calibration 
   lookup table (LUT).
2. **Burst and Swath Merging:** Valid burst regions are extracted using burst metadata,
   merged along azimuth time, aligned across swaths, and then merged along slant range
   to produce one continuous acquisition grid per selected polarization.
3. **Geometric Terrain Correction (GTC):** Using a Digital Elevation Model (DEM),
   the processor performs **inverse geocoding** by solving the zero-Doppler equation
   based on satellite orbit information and terrain elevation. This step maps the
   data from radar geometry to a georeferenced grid.
4. **Radiometric Terrain Correction (RTC):** (Optional) RTC compensates for
   terrain-induced radiometric distortions such as foreshortening, layover,
   and slope-dependent brightness variations.

📖 [D. Small, *Flattening Gamma: Radiometric Terrain Correction for SAR Imagery*](https://ieeexplore.ieee.org/document/5752845)

**Supported Products:**

- Sentinel-1 Level-1 SLC

**Supported Variables**

- **Polarization bands**:
  `vv`, `vh`, `hh`, `hv` *(each SLC product contains only a subset of these bands)*

**Specific Sentinel-1 Level-1 SLC parameters `**kwargs`:**

- `crs`: Coordinate reference system of the output dataset. Can be provided as a
  `str` or a `pyproj.CRS` object. If a string is given, it will be parsed using
  [`pyproj.crs.CRS.from_string`](https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_string).
  If not specified, [EPSG:4326](https://epsg.io/4326) is used.
- `resolution`: Target resolution for all spatial variables expressed in the units
  of the specified `crs`. If not specified, the resolution is derived (in degrees)
  from the CopDEM (30 m).
- `dem`: Digital Elevation Model (DEM) as a CF-compliant `xarray.DataArray` used for
  terrain correction. If provided, the parameters `crs`, `bbox`, and `resolution` are
  ignored, and the target grid is derived from the DEM. If not provided, the
  [CopDEM COG (30 m)](https://browser.stac.dataspace.copernicus.eu/collections/cop-dem-glo-30-dged-cog)
  is automatically retrieved via the CDSE STAC API. This requires
  [CDSE S3 credentials](https://documentation.dataspace.copernicus.eu/APIs/S3.html#generate-secrets).
- `apply_rtc`: Enable or disable radiometric terrain correction (RTC). Default is `True`.
- `interp_methods`: Interpolation method used during GTC and RTC. Supported methods:
  `nearest`, `bilinear`.
- `footprint_scale_factor`: Defines how radar pixels contribute to the output grid.
  Default: `(3.0, 15.0)`, reflecting the different scaling used for azimuth and
  slant-range processing of SLC data.
- `cache_uri`: Temporary path used to store intermediate results from the
  backward geocoding step in the Sentinel-1 processing workflow. The cache is
  automatically removed when the Python process exits. If `None`, a temporary
  directory with a unique UUID-based name is created.

Examples:

- [Docs – Sentinel-1 Analysis Mode](https://eopf-sample-service.github.io/xarray-eopf/examples/sentinel_1_analysis/)

##### Sentinel-1 Level-2 OCN

Sentinel-1 Level-2 OCN products are geolocated datasets provided on their 
**native grid**, where each pixel is associated with an individual 
latitude/longitude pair. As a result, the spatial coordinates form a 
**2D irregular grid** rather than a regular latitude/longitude raster.

The analysis mode uses the [rectification algorithm from xcube-resampling](https://xcube-dev.github.io/xcube-resampling/guide/#3-rectification)
to transform the irregular grid into a regular spatial grid with 1D latitude and 
longitude coordinates.

**Supported Products:**

- [Sentinel-1 Level-2 OCN](https://stac.browser.user.eopf.eodc.eu/collections/sentinel-1-l2-ocn)

**Supported Variables**

- **Wind Variables**:
  `wind_speed`, `wind_direction`
- **Auxiliary Variables**:
  `inversion_quality`, `wind_quality`, `percentage_bright_points`

**Specific Sentinel-1 Level-2 OCN parameters `**kwargs`:**

- `crs`: Coordinate reference system of the output dataset. Can be provided as a 
  `str` or a `pyproj.CRS` object. If a string is given, it will be parsed using 
  [`pyproj.crs.CRS.from_string`](https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_string).
  If not specified, [EPSG:4326](https://epsg.io/4326) is used.
- `resolution`: Target resolution for all spatial variables expressed in the units 
  of the specified `crs`. If not specified, the resolution is derived from the data.
- `interp_methods`: for upsampling / interpolating
  spatial data variables. Can be a single interpolation method for all
  variables or a dictionary mapping variable names or dtypes to
  interpolation method (for more information view [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/guide/#spatial-resampling-algorithms)). 
  Supported methods include:

    - `0` (nearest neighbor, default for integer arrays)
    - `1` (linear / bilinear, default for float arrays)
    - `"nearest"`
    - `"triangular"`
    - `"bilinear"`

- `agg_methods`: Optional aggregation methods to be used for downsampling
  spatial data variables / bands. Can be a single method for all variables or 
  a dictionary mapping variable names or dtypes to methods. Supported methods include:
    `"center"`, `"count"`, `"first"`, `"last"`, `"max"`, `"mean"`, `"median"`, 
    `"mode"`, `"min"`, `"prod"`, `"std"`, `"sum"`, and `"var"`.
  Defaults to `"center"` for integer arrays, else `"mean"`.
  For more information view [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/guide/#spatial-resampling-algorithms).

Examples:  

- [Docs – Sentinel-1 Analysis Mode](https://eopf-sample-service.github.io/xarray-eopf/examples/sentinel_1_analysis/)


##### Sentinel-2

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
- `crs`: Coordinate reference system of the output dataset. Can be provided as a 
  `str` or a `pyproj.CRS` object. If a string is given, it will be parsed using 
  [`pyproj.crs.CRS.from_string`](https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_string).
  If not specified, the UTM grid of the native data is used.
- `resolution`: Target resolution for all spatial variables/bands.
  Choose 10, 20, or 60 meters to minimize resampling and retain some of the native
  data resolution.
- `interp_methods`: for upsampling / interpolating
  spatial data variables. Can be a single interpolation method for all
  variables or a dictionary mapping variable names or dtypes to
  interpolation method (for more information view [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/guide/#spatial-resampling-algorithms)). 
  Supported methods include:

    - `0` (nearest neighbor, default for integer arrays)
    - `1` (linear / bilinear, default for float arrays)
    - `"nearest"`
    - `"triangular"`
    - `"bilinear"`

- `agg_methods`: Optional aggregation methods to be used for downsampling
  spatial data variables / bands. Can be a single method for all variables or 
  a dictionary mapping variable names or dtypes to methods. Supported methods include:
    `"center"`, `"count"`, `"first"`, `"last"`, `"max"`, `"mean"`, `"median"`, 
    `"mode"`, `"min"`, `"prod"`, `"std"`, `"sum"`, and `"var"`.
  Defaults to `"center"` for integer arrays (e.g. Sentinel-2 L2A SCL), else `"mean"`.
  For more information view [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/guide/#spatial-resampling-algorithms).

The spatial resampling of datasets is performed using [xcube-resampling](https://xcube-dev.github.io/xcube-resampling/).
Further explanation of the meaning and usage of these parameters for each Sentinel 
mission is provided in [Remarks on Specific Sentinel Missions](#remarks-on-specific-sentinel-missions).

Examples:  

- [Docs - Sentinel-2 Analysis Mode](https://eopf-sample-service.github.io/xarray-eopf/examples/sentinel_2_analysis/)
- [Notebook Gallery - Sentinel-2 Analysis Mode](https://eopf-sample-service.github.io/eopf-sample-notebooks/sentinel-2-analysis/)
- [Webinar - Access EOPF Zarr Products with the new xarray EOPF Backend](https://www.youtube.com/watch?v=AJz2WJNdFbw)

##### Sentinel-3

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

**Specific Sentinel-3 parameters `**kwargs`:**

- `variables`: Select variables using the names listed above in *Supported Variables*.
- `crs`: Coordinate reference system of the output dataset.  Can be provided as a 
  `str` or a `pyproj.CRS` object. If a string is given, it will be parsed using 
  [`pyproj.crs.CRS.from_string`](https://pyproj4.github.io/pyproj/dev/api/crs/crs.html#pyproj.crs.CRS.from_string).
  If not specified, [EPSG:4326](https://epsg.io/4326) is used.
- `resolution`: Target resolution for all spatial variables/bands.
  If not specified, the default is set per product:

    - Sentinel-3 OLCI Level-1 EFR: 300 meter
    - Sentinel-3 OLCI Level-1 ERR: 1200 meter
    - Sentinel-3 OLCI Level-2 LFR: 300 meter
    - Sentinel-3 SLSTR Level-1 RBT: 500 meter (1000 meter if selected variables come from F- or I-stripe)
    - Sentinel-3 SLSTR Level-2 LST: 1000 meter

- `interp_methods`: for upsampling / interpolating
  spatial data variables. Can be a single interpolation method for all
  variables or a dictionary mapping variable names or dtypes to
  interpolation method (for more information view [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/guide/#spatial-resampling-algorithms)). 
  Supported methods include:

    - `0` (nearest neighbor, default for integer arrays)
    - `1` (linear / bilinear, default for float arrays)
    - `"nearest"`
    - `"triangular"`
    - `"bilinear"`

- `agg_methods`: Optional aggregation methods to be used for downsampling
  spatial data variables / bands. Can be a single method for all variables or 
  a dictionary mapping variable names or dtypes to methods. Supported methods include:
    `"center"`, `"count"`, `"first"`, `"last"`, `"max"`, `"mean"`, `"median"`, 
    `"mode"`, `"min"`, `"prod"`, `"std"`, `"sum"`, and `"var"`.
  Defaults to `"center"` for integer arrays, else `"mean"`.
  For more information view [xcube-resampling Documentation](https://xcube-dev.github.io/xcube-resampling/guide/#spatial-resampling-algorithms).

The spatial resampling of datasets is performed using [xcube-resampling](https://xcube-dev.github.io/xcube-resampling/).
Further explanation of the meaning and usage of these parameters for each Sentinel 
mission is provided in [Remarks on Specific Sentinel Missions](#remarks-on-specific-sentinel-missions).

Example:  

- [Docs - Sentinel-3 Analysis Mode](https://eopf-sample-service.github.io/xarray-eopf/examples/sentinel_3_analysis/)
- [Notebook Gallery - Sentinel-3 Analysis Mode](https://eopf-sample-service.github.io/eopf-sample-notebooks/sentinel-3-analysis/)


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


Example:  

- [Docs - Sentinel-1 Native Mode](https://eopf-sample-service.github.io/xarray-eopf/examples/sentinel_1_native/)
- [Docs - Sentinel-2 Native Mode](https://eopf-sample-service.github.io/xarray-eopf/examples/sentinel_2_native/)
- [Docs - Sentinel-3 Native Mode](https://eopf-sample-service.github.io/xarray-eopf/examples/sentinel_3_native/)
- [Notebook Gallery - Sentinel-1 Native Mode](https://eopf-sample-service.github.io/eopf-sample-notebooks/sentinel-1-native/)
- [Notebook Gallery - Sentinel-2 Native Mode](https://eopf-sample-service.github.io/eopf-sample-notebooks/sentinel-2-native/)
- [Notebook Gallery - Sentinel-3 Native Mode](https://eopf-sample-service.github.io/eopf-sample-notebooks/sentinel-3-native/)
