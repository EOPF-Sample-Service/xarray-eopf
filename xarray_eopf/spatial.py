#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
from math import ceil

from collections.abc import Hashable, Mapping
from typing import Any, Literal, TypeAlias

import dask_image.ndinterp as ndinterp
import dask.array as da
import numpy as np
import xarray as xr

from xarray_eopf.utils import timeit

_DEBUG = False

AggMethod: TypeAlias = Literal[
    "all",
    "any",
    "count",
    "max",
    "mean",
    "median",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]

AGG_METHODS = (
    "all",
    "any",
    "count",
    "max",
    "mean",
    "median",
    "min",
    "prod",
    "std",
    "sum",
    "var",
)

SplineOrder: TypeAlias = Literal[0, 1, 2, 3]

SPLINE_ORDERS = 0, 1, 2, 3


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
    agg_method: AggMethod | dict[Hashable, AggMethod] | None = None,
    spline_order: SplineOrder | dict[Hashable, SplineOrder] | None = None,
    eps: float = 1e-5,
) -> Mapping[Hashable, xr.DataArray]:
    spatial_variables = get_spatial_vars(variables)
    ref_var_name = ref_var_name or get_ref_var_name(spatial_variables)
    ref_var = spatial_variables[ref_var_name]
    ref_spatial_shape = ref_var.shape[-2:]
    rescaled_variables = {}

    def format_factor(factor):
        return f"{factor:.6f}".rstrip("0").rstrip(".")

    for var_name, var in spatial_variables.items():
        ref_y_dim, ref_x_dim = ref_var.dims[-2:]
        y_dim, x_dim = var.dims[-2:]
        spatial_shape = var.shape[-2:]
        if spatial_shape != ref_spatial_shape:
            is_integer = np.issubdtype(var.dtype, np.integer)
            y_size, x_size = spatial_shape
            target_y_size, target_x_size = ref_spatial_shape
            x_scale = x_size / target_x_size
            y_scale = y_size / target_y_size
            x_window_size = ceil(x_scale)
            y_window_size = ceil(y_scale)
            target_x_size = x_window_size * target_x_size
            target_y_size = y_window_size * target_y_size
            x_scale = x_size / target_x_size
            y_scale = y_size / target_y_size
            history = var.attrs.get("history", "")
            rescaled_data = da.asarray(var.data)
            if abs(x_scale - 1) > eps or abs(y_scale - 1) > eps:
                factors = (var.ndim - 2) * (1,) + (y_scale, x_scale)
                matrix = np.diag(factors)
                order = get_spline_order(
                    var_name, spline_order, is_categorical=is_integer
                )
                with timeit(f"up-sampling {var_name}!r", silent=not _DEBUG):
                    rescaled_data = ndinterp.affine_transform(
                        rescaled_data,
                        matrix,
                        order=order,
                        output_shape=var.shape[:-2] + (target_y_size, target_x_size),
                    )
                x_sf = format_factor(x_scale)
                y_sf = format_factor(y_scale)
                s = f" {x_sf}" if x_sf == y_sf else f"s {x_sf} and {y_sf}"
                history += (
                    f"Upsampling in dimensions {ref_x_dim!r} and {ref_y_dim!r} by"
                    f" scale factor{s} using spline interpolation of order {order};\n"
                )
            if x_window_size > 1 or y_window_size > 1:
                method = get_agg_method(var_name, agg_method, is_categorical=is_integer)
                with timeit(f"down-sampling {var_name!r}", silent=not _DEBUG):
                    reduction = getattr(np, method)
                    rescaled_data = da.coarsen(
                        reduction,
                        rescaled_data,
                        {1: y_window_size, 0: x_window_size},
                    )
                x_ws = x_window_size
                y_ws = y_window_size
                s = f" {x_ws}" if x_ws == y_ws else f"s {x_ws} and {y_ws}"
                history += (
                    f"Downsampling in dimensions {ref_x_dim!r} and {ref_y_dim!r} by"
                    f" window size{s} using aggregation method {method!r};\n"
                )

            rescaled_data = rescaled_data.astype(var.dtype)
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
                attrs={**var.attrs, "history": history},
            ).chunk({ref_x_dim: ref_var.chunks[-1], ref_y_dim: ref_var.chunks[-2]})
            for enc_name in ("chunks", "preferred_chunks"):
                if enc_name in ref_var.encoding:
                    rescaled_var.encoding[enc_name] = ref_var.encoding[enc_name]
            rescaled_variables[var_name] = rescaled_var
    all_variables = {**variables, **rescaled_variables}
    sorted_var_names = sorted(all_variables)
    return {var_name: all_variables[var_name] for var_name in sorted_var_names}


def get_spline_order(
    var_name: Hashable, spline_order: Any, is_categorical: bool = False
) -> SplineOrder:
    order = (
        spline_order.get(var_name) if isinstance(spline_order, dict) else spline_order
    )
    if order is None:
        if is_categorical:
            order = 0  # nearest
        else:
            order = 3  # cubic
    if order not in SPLINE_ORDERS:
        raise ValueError(f"Unknown spline order: {order}")
    return order


def get_agg_method(
    var_name: Hashable, agg_method: Any = None, is_categorical: bool = False
) -> AggMethod:
    method = agg_method.get(var_name) if isinstance(agg_method, dict) else agg_method
    if method is None:
        if is_categorical:
            # TODO: down-sample categorical data
            #  according to Sentinel-2 products, SCL should be down-sampled
            #  using nearest-neighbour. This would be method="center".
            #  In general, the most natural solution is to take the
            #  most frequent value. This would be method="mode".
            #  The only value preserving methods currently available in
            #  xarray are "min" and "max".
            #
            method = "max"
        else:
            # For floating point we block-average
            method = "mean"
    if method not in AGG_METHODS:
        raise ValueError(f"Unknown aggregation method: {method}")
    return method
