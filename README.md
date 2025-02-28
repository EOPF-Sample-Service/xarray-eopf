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
