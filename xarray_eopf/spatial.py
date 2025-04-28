#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Hashable, Mapping
from math import ceil
from typing import Callable, Literal, TypeAlias

import dask.array as da
import dask_image.ndinterp as ndinterp
import numpy as np
import xarray as xr

import xarray_eopf.coarsen as xec

from .utils import NameTypeMapping, timeit

ALL = slice(None)

_DEBUG = False

AggMethod: TypeAlias = Literal[
    "center",
    "count",
    "first",
    "last",
    "max",
    "mean",
    "median",
    "mode",
    "min",
    "prod",
    "std",
    "sum",
    "var",
]
AggMethods: TypeAlias = AggMethod | dict[AggMethod, list[str | np.dtype]]

AggFunction: TypeAlias = Callable[[np.ndarray, tuple[int, ...] | None], np.ndarray]

AGG_METHODS: dict[AggMethod, AggFunction] = {
    "center": xec.center,
    "count": np.count_nonzero,
    "first": xec.first,
    "last": xec.last,
    "prod": np.nanprod,
    "max": np.nanmax,
    "mean": xec.mean,
    "median": xec.median,
    "min": np.nanmin,
    "mode": xec.mode,
    "std": xec.std,
    "sum": np.nansum,
    "var": xec.var,
}

SplineOrder: TypeAlias = Literal[0, 1, 2, 3]
SplineOrders: TypeAlias = SplineOrder | dict[SplineOrder, list[str | np.dtype]]

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
    spline_orders: SplineOrders | None = None,
    agg_methods: AggMethods | None = None,
    eps: float = 1e-6,
) -> Mapping[Hashable, xr.DataArray]:
    spatial_variables = get_spatial_vars(variables)
    ref_var_name = ref_var_name or get_ref_var_name(spatial_variables)
    ref_var = spatial_variables[ref_var_name]
    ref_spatial_shape = ref_var.shape[-2:]
    spline_orders = NameTypeMapping.new("spline_orders", 3, spline_orders)
    agg_methods = NameTypeMapping.new("agg_methods", "mean", agg_methods)

    rescaled_variables = {}

    def format_factor(factor):
        return f"{factor:.6f}".rstrip("0").rstrip(".")

    for var_name, var in spatial_variables.items():
        ref_y_dim, ref_x_dim = ref_var.dims[-2:]
        y_dim, x_dim = var.dims[-2:]
        spatial_shape = var.shape[-2:]
        if spatial_shape != ref_spatial_shape:
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
                order = spline_orders.get(var_name, var.dtype)
                with timeit(f"Upscaling {var_name}!r", silent=not _DEBUG):
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
                    f"Upscaling dimensions {ref_x_dim!r} and {ref_y_dim!r} by"
                    f" scale factor{s} using spline interpolation of order {order};\n"
                )
            if x_window_size > 1 or y_window_size > 1:
                method = agg_methods.get(var_name, var.dtype)
                with timeit(f"Downscaling {var_name!r}", silent=not _DEBUG):
                    reduction: AggFunction = AGG_METHODS[method]
                    ndim = rescaled_data.ndim
                    # noinspection PyTypeChecker
                    rescaled_data = da.coarsen(
                        reduction,
                        rescaled_data,
                        {ndim - 1: y_window_size, ndim - 2: x_window_size},
                    )
                x_ws = x_window_size
                y_ws = y_window_size
                s = f" {x_ws}" if x_ws == y_ws else f"s {x_ws} and {y_ws}"
                history += (
                    f"Downscaling dimensions {ref_x_dim!r} and {ref_y_dim!r} by"
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
