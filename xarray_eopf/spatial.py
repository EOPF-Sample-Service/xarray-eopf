#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Mapping, Hashable

import dask_image.ndinterp as ndinterp
import numba
import numpy as np
import xarray as xr



from xarray_eopf.utils import timeit


_DEBUG = False


def get_spatial_vars(
    variables: Mapping[Hashable, xr.DataArray],
) -> dict[Hashable, xr.DataArray]:
    return {var_name: var for var_name, var in variables.items() if is_spatial_var(var)}


def is_spatial_var(var: xr.DataArray) -> bool:
    return (
        var.ndim >= 2
        and str(var.dims[-2]).endswith("y")
        and str(var.dims[-1]).endswith("x")
    )


def get_ref_var_name(variables: Mapping[Hashable, xr.DataArray]) -> Hashable | None:
    max_size = -1
    ref_var_name = None
    for var_name, var in get_spatial_vars(variables).items():
        y_size, x_size = var.shape[-2:]
        size = y_size * x_size
        if size > max_size:
            max_size = size
            ref_var_name = var_name
    return ref_var_name


def rescale_spatial_vars(
    variables: Mapping[Hashable, xr.DataArray],
    ref_var_name: Hashable | None = None,
    agg_method: str | dict[Hashable, str] | None = None,
    spline_order: int | dict[Hashable, int] | None = None,
) -> Mapping[Hashable, xr.DataArray]:
    spatial_variables = get_spatial_vars(variables)
    ref_var_name = ref_var_name or get_ref_var_name(spatial_variables)
    ref_var = spatial_variables[ref_var_name]
    ref_spatial_shape = ref_var.shape[-2:]
    rescaled_variables = {}
    for var_name, var in spatial_variables.items():
        spatial_shape = var.shape[-2:]
        if spatial_shape != ref_spatial_shape:
            scale_y = spatial_shape[0] / ref_spatial_shape[0]
            scale_x = spatial_shape[1] / ref_spatial_shape[1]
            factors = (var.ndim - 2) * (1,) + (scale_y, scale_x)
            matrix = np.diag(factors)
            is_down_sampling = scale_x < 1
            is_integer = np.issubdtype(var.dtype, np.integer)
            if is_down_sampling:
                method = agg_method.get(var_name) if isinstance(agg_method,  dict) else agg_method
                if not isinstance(method, str):
                    if is_integer:
                        # For integer/categorical data we take the
                        # most frequent value in a block
                        method = "mode"
                    else:
                        # For floating point we block-average
                        method = "mean"
                isx = int(scale_x)
                isy = int(scale_x)
                if isx == scale_x and isy == scale_y:
                    with timeit(f"{var_name} down-sample by index", silent=not _DEBUG):
                        y_name, x_name = var.dims[-2:]
                        coarsened = var.coarsen(**{y_name: isy, x_name: isx})
                        if method == "mode":
                            rescaled_data = coarsened.reduce(fast_int_mode)
                        else:
                            rescaled_data = getattr(coarsened, method)()
                    rescaled_data = rescaled_data.chunk(ref_var.chunks)
                else:
                    # TODO: implement down-sampling for non-integer factors
                    raise NotImplementedError("Down-sampling only implemented for integer factors")
            else:
                order = spline_order.get(var_name) if isinstance(spline_order, dict) else spline_order
                if not isinstance(order, int):
                    if is_integer:
                        order = 0  # nearest
                    else:
                        order = 3  # cubic
                with timeit(f"{var_name} up-sample by affine_transform()", silent=not _DEBUG):
                    rescaled_data = ndinterp.affine_transform(
                        var.data,
                        matrix,
                        order=order,
                        output_chunks=ref_var.chunks,
                        output_shape=var.shape[:-2] + ref_spatial_shape,
                    )
            y_dim, x_dim = var.dims[-2:]
            ref_y_dim, ref_x_dim = ref_var.dims[-2:]
            coords = dict(var.coords)
            coords.pop(x_dim, None)
            coords.pop(y_dim, None)
            coords[ref_x_dim] = ref_var[ref_x_dim]
            coords[ref_y_dim] = ref_var[ref_y_dim]
            rescaled_var = xr.DataArray(
                coords=coords,
                data=rescaled_data,
                dims=var.dims[:-2] + ref_var.dims[-2:],
                name=var.name,
                attrs=var.attrs,
            ).chunk(ref_var.chunks)
            for enc_name in ("chunks", "preferred_chunks"):
                if enc_name in ref_var.encoding:
                    rescaled_var.encoding[enc_name] = ref_var.encoding[enc_name]
            rescaled_variables[var_name] = rescaled_var
    all_variables = {**variables, **rescaled_variables}
    sorted_var_names = sorted(all_variables)
    return {var_name: all_variables[var_name] for var_name in sorted_var_names}


@numba.njit
def fast_int_mode(array: np.ndarray) -> int:
    if array.size == 0:
        return 0
    # Ensure input is integer
    array = array.astype(np.int64)
    min_val = array.min()
    max_val = array.max()
    offset = -min_val  # To shift negative values to positive indices
    histogram = np.zeros(max_val - min_val + 1, dtype=np.int64)
    for val in array:
        histogram[val + offset] += 1
    index = np.argmax(histogram)
    return index - offset
