#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Mapping, Hashable
from typing import Literal, TypeAlias, Any

import dask_image.ndinterp as ndinterp
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
        var_attrs = dict(var.attrs)
        ref_y_dim, ref_x_dim = ref_var.dims[-2:]
        y_dim, x_dim = var.dims[-2:]
        spatial_shape = var.shape[-2:]
        if spatial_shape != ref_spatial_shape:
            y_scale = spatial_shape[0] / ref_spatial_shape[0]
            x_scale = spatial_shape[1] / ref_spatial_shape[1]
            is_down_sampling = x_scale + eps > 1.0
            is_integer = np.issubdtype(var.dtype, np.integer)
            if is_down_sampling:
                method = get_agg_method(var_name, agg_method, is_categorical=is_integer)
                isx = int(x_scale)
                isy = int(y_scale)
                if abs(isx - x_scale) < eps and abs(isy - y_scale) < eps:
                    with timeit(f"down-sampling {var_name!r}", silent=not _DEBUG):
                        coarsened = var.coarsen(**{y_dim: isy, x_dim: isx})
                        rescaled_data = getattr(coarsened, method)()
                    var_attrs["history"] = (
                        f"Down-sampling by factors"
                        f" {ref_x_dim}={isx} and {ref_y_dim}={isy}"
                        f" using aggregation method {method!r}."
                    )
                else:
                    # TODO: implement down-sampling for non-integer factors
                    raise NotImplementedError(
                        "Down-sampling only implemented for integer factors"
                    )
            else:
                factors = (var.ndim - 2) * (1,) + (y_scale, x_scale)
                matrix = np.diag(factors)
                order = get_spline_order(
                    var_name, spline_order, is_categorical=is_integer
                )
                with timeit(f"up-sampling {var_name}!r", silent=not _DEBUG):
                    rescaled_data = ndinterp.affine_transform(
                        var.data,
                        matrix,
                        order=order,
                        output_chunks=ref_var.chunks,
                        output_shape=var.shape[:-2] + ref_spatial_shape,
                    )
                var_attrs["history"] = (
                    f"Up-sampling by factors"
                    f" {ref_x_dim}={format_factor(x_scale)} and"
                    f" {ref_y_dim}={format_factor(y_scale)}"
                    f" using spline interpolation of order {order}."
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
                attrs=var_attrs,
            ).chunk(ref_var.chunks)
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
