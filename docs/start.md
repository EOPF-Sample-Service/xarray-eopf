# Getting Started

The `xarray-eopf` package can be installed into an existing Python environment
using

```bash
pip install xarray-eopf
```

or

```bash
conda install -c conda-forge xarray-eopf
```

After installation, you are ready to go and use the `engine="eopf-zarr"` keyword
argument when calling `open_dataset()` or `open_datatree()`:

```python
import xarray as xr

dataset = xr.open_dataset(url_or_path, engine="eopf-zarr")
```
