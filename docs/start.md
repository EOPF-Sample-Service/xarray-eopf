# Getting Started

By installing the `xarray-eopf` package into an existing Python environment
using

```bash
pip install xarray-eopf
```

or

```bash
conda install -c conda-forge xarray-eopf
```

you are ready to go and use the `engine="eopf-zarr"` keyword argument when calling
`open_dataset()` or `open_datatree()`:

```python
import xarray as xr

dataset = xr.open_dataset(url_or_path_to_product, engine="eopf-zarr")
```
