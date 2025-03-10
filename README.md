[![Build Status](https://github.com/EOPF-Sample-Service/xarray-eopf/actions/workflows/unit-tests.yml/badge.svg?branch=main)](https://github.com/EOPF-Sample-Service/xarray-eopf/actions)
[![codecov](https://codecov.io/gh/EOPF-Sample-Service/xarray-eopf/branch/main/graph/badge.svg)](https://codecov.io/gh/EOPF-Sample-Service/xarray-eopf)
[![PyPI Version](https://img.shields.io/pypi/v/xarray-eopf)](https://pypi.org/project/xarray-eopf/)
[![Anaconda-Server Badge](https://anaconda.org/conda-forge/xarray-eopf/badges/version.svg)](https://anaconda.org/conda-forge/xarray-eopf)
[![License](https://anaconda.org/conda-forge/xarray-eopf/badges/license.svg)](https://anaconda.org/conda-forge/xarray-eopf)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# xarray-eopf

A backend implementation for [xarray](https://docs.xarray.dev/en/stable/user-guide/io.html) 
that allows for analysis-ready reading of ESA EOPF data products from local and remote 
filesystems.


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
pytest tests/unit
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
