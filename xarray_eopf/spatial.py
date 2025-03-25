#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Mapping, Hashable

import dask_image.ndinterp as ndinterp
import numpy as np
import xarray as xr

from xarray_eopf.utils import timeit


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
    spline_order: int = 0,
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
            factors = (var.ndim - 2) * (1,) + (1.0 / scale_y, 1.0 / scale_x)
            matrix = np.diag(factors)
            with timeit(f"{var_name} affine_transform", silent=True):
                rescaled_data = ndinterp.affine_transform(
                    var.data,
                    matrix,
                    order=spline_order,
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
    return {**variables, **rescaled_variables}
