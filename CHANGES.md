## Changes in 0.2.3 (under development)

- **Sentinel-3 SLSTR Level-1 RBT products** are now supported in analysis mode. This
  allows data from bands A, B, F, and I — in both nadir and oblique viewing
  geometries — to be represented on a unified grid within a single dataset.
- **Sentinel-3 SLSTR datasets** are now terrain-corrected using the elevation
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
