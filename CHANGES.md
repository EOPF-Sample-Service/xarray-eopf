## Changes in 0.2.10 (in development)

- Sentinel-1 GRD analysis mode is now fully lazy, enabling seamless execution on 
  local and distributed Dask clusters.


## Changes in 0.2.9 (from 2026-06-03)

- Added support for Sentinel-1 Level-2 OCN analysis mode.
- Fixed an issue in Sentinel-1 GRD analysis mode that could produce NaN values along 
  the edges of the bounding box.


## Changes in 0.2.8 (from 2026-05-08)

- Fix package discovery in `pyproject.toml` to ensure only `xarray_eopf` 
  (and its subpackages) is included in the PyPI wheel.
- Remove the `coarsen.py` module, as it has been moved to [xcube-resampling](https://github.com/xcube-dev/xcube-resampling) 
  and is no longer used internally.
- Add support for Sentinel-1 Level-1 GRD analysis mode.
- Updated year in the headers.
- Added footprint-based subsetting for Sentinel-3 OLCI and SLSTR LST using STAC 
  metadata, improving performance by avoiding full latitude/longitude grid downloads 
  during subsetting.


## Changes in 0.2.7 (from 2026-03-27)

- Add the CRS information from the STAC metadata stored in the datatree's attributes;
  Temporally fixes the issue https://gitlab.eopf.copernicus.eu/cpm/eopf-cpm/-/issues/932
- Downgraded Zarr dependency to `zarr>=2,<3.0` for now, to be compatible with
  `xcube-eopf`.


## Changes in 0.2.6 (from 2026-03-20)

- Fixed an issue in `xr.open_dataset` (native mode) where selecting variables did not
  drop unused coordinates; these are now removed correctly.
- Corrected the data type of the Sentinel-2 Level-2A SCL data array in analysis mode 
  (now `uint8` instead of `float64`). The underlying issue has been reported to the 
  CPM repository: https://gitlab.eopf.copernicus.eu/cpm/eopf-cpm/-/issues/1044
- Added improved example notebooks to the documentation.
- Fixed issues in integration tests.
- Updated Zarr dependency to require `zarr>=3.0`.


## Changes in 0.2.5 (from 2025-11-26)

* Added subsetting and reprojection in analysis mode via parameters `crs`,
  `resolution`, and `bbox`.


## Changes in 0.2.4 (from 2025-11-17)

* Added support for **common band names** from the [STAC EO extension](https://github.com/stac-extensions/eo?tab=readme-ov-file#common-band-names)
  in **Sentinel-2 analysis mode**.  The `variables` parameter now accepts standard
  spectral names such as `blue`, `green`, `red`, `nir`, and others.
* Fix: CRS information is missing in Sentinel-2 product data variables since
  CPM v2.6.2. CRS is now correctly read from the dataset’s `other_metadata`
  attributes in the datatree.


## Changes in 0.2.3 (from 2025-10-23)

* **Sentinel-3 SLSTR Level-1 RBT products** are now supported in analysis mode. This
  allows data from grids a, b, f, and i — in both nadir and oblique viewing
  geometries — to be represented on a unified grid within a single dataset.
* **Sentinel-3 SLSTR datasets** are now terrain-corrected using the elevation
  information provided within the product itself.


## Changes in 0.2.2 (from 2025-09-24)

* In analysis mode for Sentinel-3 products, coordinates are now filtered so that only
  `"lat"` and `"lon"` remain. Since the data is rectified, non-spatial coordinates loose
  their association with the data after rectification.


## Changes in 0.2.1 (from 2025-09-23)

* Sentinel-3 products in analysis mode are now chunked into (1024, 1024) blocks to 
  align with the input chunk size. Previously, the data was presented as a single 
  spatial chunk.


## Changes in 0.2.0 (from 2025-08-26)

* Spatial resampling is now performed using [xcube-resampling](https://xcube-dev.github.io/xcube-resampling/).  
  As part of this change, the parameter `spline_orders` has been renamed to 
  `interp_methods` for consistency.
* New **Sentinel-3 analysis mode**: performs rectification from the native 
  irregular grid to a regular grid. Supported products include:  
  - OLCI Level-1 EFR/ERR  
  - OLCI Level-2 EFR  
  - SLSTR Level-2 LST


## Changes in 0.1.2 (from 2025-07-01)

* Fixed a bug that prevented access to sub-groups within a Zarr `DataTree` via HTTPS  
  paths (e.g., `https://.../zarr/sub/group`), addressing the issue reported  
  [here](https://github.com/EOPF-Sample-Service/eopf-stac/issues/26#issuecomment-2978483579).

## Changes in 0.1.1 (from 2025-06-11)

### Bug fixes

* Added support for Sentinel-2C observations. The product type can now be correctly
  inferred from the file path in object storage.
* Support for accessing sub-groups within a Zarr DataTree via HTTPS paths,
  e.g. https://...zarr/sub/group.

## Changes in 0.1.0 (from 2025-04-28)

* Added initial analysis mode for Sentinel-2 L1C and L2A products.
  The analysis mode (the default mode) provides the following features: 
  * Open the deeply nested EOPF products as flat `xarray.Dataset` objects.
  * All bands and quality images resampled to a single, user provided 
    resolution, hence, spatial dimensions will be just `x` and `y`.
  * User-specified resampling by passing spline orders for up-scaling
    and aggregation methods for downscaling.
  * Attach CF-compliant spatial referencing of datasets using a shared grid 
    mapping variable `spatial_ref`.
  * Attach other CF-compliant metadata enhancements such as flag values and 
    meanings for pixel quality information, such as the Sentinel-2 
    scene classification (variable `scl`).
* Added notebook examples for accessing Sentinel-1 and Sentinel-2 using the
  `eopf-zarr` engine
* Added CI for unit and integration tests
* Added CodeCov report


## Changes in 0.0.1

* Initial package structure.
* Basic README and documentation.
* License and contribution guidelines.
