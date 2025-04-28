[![Build Status](https://github.com/EOPF-Sample-Service/xarray-eopf/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/EOPF-Sample-Service/xarray-eopf/actions)
[![codecov](https://codecov.io/gh/EOPF-Sample-Service/xarray-eopf/branch/main/graph/badge.svg)](https://codecov.io/gh/EOPF-Sample-Service/xarray-eopf)
[![PyPI Version](https://img.shields.io/pypi/v/xarray-eopf)](https://pypi.org/project/xarray-eopf/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/xarray-eopf/badges/version.svg)](https://anaconda.org/conda-forge/xarray-eopf)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff)
[![License](https://anaconda.org/conda-forge/xarray-eopf/badges/license.svg)](https://anaconda.org/conda-forge/xarray-eopf)

# xarray-eopf

An [xarray](https://docs.xarray.dev/en/stable/user-guide/io.html) backend implementation
for ESA EOPF data products in Zarr format.

## Features

After installing this package, user can specify a new xarray backend 
named `"eopf-zarr"` to open EOPF sample products. The backend has
two modes of operation, default analysis mode and the native mode. 
Both modes allow 

* to open EOPF sample products from the local filesystem or from their
  original object storage using URLs with both `https` or `s3` 
  protocols;
* to open entire products as `xarray.DataTree` or `xarray.Dataset`; 
* to open a subgroup as `xarray.Dataset`. This works with 
  local filesystem or `s3`-URLs.

The default analysis mode has the aim to represent the EOPF data 
products in an analysis-ready and convenient way. It provides the
following features:

* Open the deeply nested EOPF products as flat `xarray.Dataset` objects.
* All bands and quality images resampled to a single, user provided 
  resolution, hence, spatial dimensions will be just `x` and `y`.
* User-specified resampling by passing spline orders for up-scaling
  and aggregation methods for downscaling.
* CF-compliant spatial referencing of datasets using a shared grid 
  mapping variable `spatial_ref`.

The analysis mode is currently implemented Sentinel-2 products only.
Support for Sentinel-1 and Sentinel-3 is coming soon. 

The native mode does not modify any contents or data, instead it basically 
delegates to the built-in `"zarr"` backend.

More information can be found in the 
[package documentation](https://eopf-sample-service.github.io/xarray-eopf). 

## Development

### Setting up a development environment

The recommended Python distribution for development is 
[miniforge](https://conda-forge.org/download/) which includes 
conda, mamba, and their dependencies.

```shell
git clone https://github.com/EOPF-Sample-Service/xarray-eopf.git
cd xarray-eopf
mamba env create
mamba activate eopf-xr
pip install -ve .
```

### Install the library locally and test

```shell
mamba activate eopf-xr
pip install -ve .
pytest
```
By default, this will run all unit and integration tests. To run only the unit test
suite, use:  

```shell
pytest tests
```

To run tests and generate a coverage report, use:

```shell
pytest --cov xarray_eopf --cov-report html tests
```

### Documentation

### Setting up a documentation environment

```shell
mamba activate eopf-xr
pip install .[doc]
```

### Testing documentation changes

```shell
mkdocs serve
```

### Deploying documentation changes

```shell
mkdocs gh-deploy
```
