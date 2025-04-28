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
