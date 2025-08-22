#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import os
from collections.abc import Mapping
from typing import Any, Iterable

import xarray as xr
from xarray.backends import AbstractDataStore, BackendEntrypoint
from xarray.coding.times import CFTimedeltaCoder
from xarray.core.types import ReadBuffer
from xcube_resampling.constants import AggMethods, InterpMethods

from .amode import AnalysisMode
from .amodes import register_analysis_modes
from .constants import (
    OP_MODE_ANALYSIS,
    OP_MODE_NATIVE,
    OP_MODES,
    OpMode,
)
from .filter import filter_dataset
from .flatten import flatten_datatree, flatten_datatree_as_dict
from .source import normalize_source, normalize_source_path
from .utils import assert_arg_is_one_of


class EopfBackend(BackendEntrypoint):
    """Backend for EOPF Data Products using the Zarr format.

    Note, that the `chunks` parameter passed to xarray top level functions
    `xr.open_datatree()` and `xr.open_dataset()` is _not_ passed to
    backend. Instead, xarray uses them to (re)chunk the results
    from calling the backend equivalents, hence, _after_ backend code.
    """

    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        op_mode: OpMode = OP_MODE_ANALYSIS,
        product_type: str | None = None,
        storage_options: Mapping[str, Any] | None = None,
        drop_variables: str | Iterable[str] | None = None,
        decode_timedelta: (
            bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None
        ) = False,
    ) -> xr.DataTree:
        """Backend implementation delegated to by
        [`xarray.open_datatree()`](https://docs.xarray.dev/en/stable/generated/xarray.open_datatree.html).

        Args:
            filename_or_obj: File path, or URL, a path-like string, or
                a Zarr store, or other key to object mapping.
            op_mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            product_type: Optional product type name, such as `"MSIL1C"`.
                Only used if `op_mode="analysis"`; typically not required
                if the filename inherent to `filename_or_obj`
                adheres to EOPF naming conventions.
            storage_options: If `filename_or_obj` is a file path or URL,
                these options specify the source filesystem.
                Will be passed to [`fsspec.filesystem()`](https://filesystem-spec.readthedocs.io/en/latest/usage.html).
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file. See
                [xarray documentation](https://docs.xarray.dev/en/stable/generated/xarray.open_datatree.html).
            decode_timedelta: How to decode time-delta units. See
                [xarray documentation](https://docs.xarray.dev/en/stable/generated/xarray.open_datatree.html).

        Returns:
            A new data-tree instance.
        """
        # Disable attribute expansion for cleaner, more concise rendering in notebooks
        xr.set_options(display_expand_attrs=False)

        assert_arg_is_one_of(op_mode, "op_mode", OP_MODES)

        filename_or_obj, subgroup_path = normalize_source_path(filename_or_obj)
        source = normalize_source(filename_or_obj, storage_options)

        # noinspection PyTypeChecker
        datatree = xr.open_datatree(
            source,
            engine="zarr",
            group=subgroup_path,
            # prefer the chunking from the Zarr metadata
            chunks={},
            # here as it is required for all backends
            drop_variables=drop_variables,
            # here to silence xarray warnings
            decode_timedelta=decode_timedelta,
        )

        _assert_datatree_is_chunked(datatree)

        if op_mode == OP_MODE_NATIVE:
            # native mode, so we return tree as-is
            return datatree
        else:  # op_mode == OP_MODE_ANALYSIS
            # analysis mode
            if subgroup_path:
                # subgroup level, return subtree as-is
                return datatree
            else:
                # product level, so we transform the tree
                analysis_mode = AnalysisMode.guess(
                    filename_or_obj, product_type=product_type
                )
                return analysis_mode.transform_datatree(datatree)

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        op_mode: OpMode = OP_MODE_ANALYSIS,
        # params for op_mode=native/analysis
        storage_options: Mapping[str, Any] | None = None,
        group_sep: str = "_",
        variables: str | Iterable[str] | None = None,
        # params for op_mode=analysis
        product_type: str | None = None,
        resolution: int | float | None = None,
        interp_methods: InterpMethods | None = None,
        agg_methods: AggMethods | None = None,
        # params required by xarray backend interface
        drop_variables: str | Iterable[str] | None = None,
        # params for other reasons
        decode_timedelta: (
            bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None
        ) = False,
    ) -> xr.Dataset:
        """Backend implementation delegated to by
        [`xarray.open_dataset()`](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html).

        Args:
            filename_or_obj: File path, or URL, or path-like string.
            op_mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            product_type: Optional product type name, such as `"MSIL1C"`.
                Only used if `op_mode="analysis"`; typically not required
                if the filename inherent to `filename_or_obj`
                adheres to EOPF naming conventions.
            storage_options: If `filename_or_obj` is a file path or URL,
                these options specify the source filesystem.
                Will be passed to [`fsspec.filesystem()`](https://filesystem-spec.readthedocs.io/en/latest/usage.html).
            group_sep: Separator string used to concatenate groups names
                to create prefixes for unique variable and dimension names.
                Defaults to the underscore character (`"_"`)
            resolution: Target resolution for all spatial
                data variables / bands. For Sentinel-2 products it be one of
                `10`, `20`, or `60`. Only used if `op_mode="analysis"`.
            interp_methods: Optional interpolation method to be used if
                `op_mode="analysis"`, for upsampling / interpolating
                spatial data variables. Can be a single interpolation method for all
                variables or a dictionary mapping variable names or dtypes to
                interpolation method. Supported methods include:

                - `0` (nearest neighbor)
                - `1` (linear / bilinear)
                - `"nearest"`
                - `"triangular"`
                - `"bilinear"`

                The default is `0` for integer arrays (e.g. Sentinel-2 L2A SCL),
                else `1`.
            agg_methods: Optional aggregation methods to be used if
                `op_mode="analysis"`, for downsampling spatial variables.
                Can be a single method for all variables or a dictionary mapping
                variable names or dtypes to methods. Supported methods include:
                    "center", "count", "first", "last", "max", "mean", "median",
                    "mode", "min", "prod", "std", "sum", and "var".
                Defaults to "center" for integer arrays (e.g. Sentinel-2 L2A SCL),
                else "mean".
            variables: Variables to include in the dataset. Can be a name or
                regex pattern or iterable of the latter.
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file. See
                [xarray documentation](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html).
            decode_timedelta: How to decode time-delta units. See
                [xarray documentation](https://docs.xarray.dev/en/stable/generated/xarray.open_dataset.html).

        Returns:
            A new dataset instance.
        """
        # Disable attribute expansion for cleaner, more concise rendering in notebooks
        xr.set_options(display_expand_attrs=False)

        assert_arg_is_one_of(op_mode, "op_mode", OP_MODES)

        datatree = self.open_datatree(
            filename_or_obj,
            op_mode="native",
            storage_options=storage_options,
            # here as it is required for all backends
            drop_variables=drop_variables,
            # here to silence xarray warnings
            decode_timedelta=decode_timedelta,
        )

        _assert_datatree_is_chunked(datatree)

        if op_mode == OP_MODE_NATIVE:
            # native mode
            if datatree.has_data:
                # subgroup level, so we return dataset as-is
                dataset = datatree.to_dataset()
            else:
                # product level, so we flatten the tree
                dataset = flatten_datatree(datatree, sep=group_sep)
            dataset = filter_dataset(dataset, variables)
        else:
            # analysis mode
            analysis_mode = AnalysisMode.guess(
                filename_or_obj, product_type=product_type
            )
            if datatree.has_data:
                # subgroup level, so we transform the dataset
                dataset = datatree.to_dataset()
                dataset = analysis_mode.transform_dataset(dataset)
            else:
                # product level, so we convert the tree into a dataset
                params = analysis_mode.get_applicable_params(
                    resolution=resolution,
                    interp_methods=interp_methods,
                    agg_methods=agg_methods,
                )
                dataset = analysis_mode.convert_datatree(
                    datatree, includes=variables, **params
                )

        return dataset

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
    ) -> bool:
        """Check if the given `filename_or_obj` refers to an object that
        can be opened by this backend.

        The function returns `False` to indicate that this backend should
        only be used when specified by passing `engine="eopf-zarr"`.

        Args:
            filename_or_obj: File path, or URL, or path-like string.

        Returns:
            Always `False`.
        """
        return False


def _assert_datatree_is_chunked(datatree: xr.DataTree):
    for ds_name, ds in flatten_datatree_as_dict(datatree).items():
        _assert_dataset_is_chunked(ds, name=ds_name)


def _assert_dataset_is_chunked(dataset: xr.Dataset, name: str | None = None):
    ds_name = name or "dataset"
    for var_name, var in dataset.data_vars.items():
        assert var.chunks is not None, f"{ds_name}.{var_name}: no chunks"


register_analysis_modes()
