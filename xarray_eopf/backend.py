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
from .store import open_store
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
        protocol: str | None = None,
        storage_options: Mapping[str, Any] | None = None,
        drop_variables: str | Iterable[str] | None = None,
        decode_timedelta: (
            bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None
        ) = False,
    ) -> xr.DataTree:
        """Backend implementation delegated to by
        [`xarray.open_datatree()`](https://docs.xarray.dev/en/stable/generated/xarray.open_datatree.html).

        Args:
            filename_or_obj: File path, or URL, or path-like string.
            op_mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            product_type: Product type name, such as `"S2B_MSIL1C"`.
                Only used if `op_mode="analysis"` and
                only required if `filename_or_obj` is not a path or URL
                that refers to a product path adhering to EOPF naming conventions.
            protocol: If `filename_or_obj` is a file path or URL,
                forces using the filesystem protocol.
                Otherwise, the protocol will be derived from the file path or URL.
                Will be passed to [`fsspec.filesystem()`](https://filesystem-spec.readthedocs.io/en/latest/usage.html).
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

        assert_arg_is_one_of(op_mode, "op_mode", OP_MODES)

        fs_store = open_store(filename_or_obj, protocol, storage_options)

        datatree = xr.open_datatree(
            fs_store,
            engine="zarr",
            # prefer the chunking from the Zarr metadata
            chunks={},
            # here as it is required for all backends
            drop_variables=drop_variables,
            # here to silence xarray warnings
            decode_timedelta=decode_timedelta,
        )

        _assert_datatree_is_chunked(datatree)

        if op_mode == OP_MODE_NATIVE:
            return datatree
        else:  # op_mode == OP_MODE_ANALYSIS
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
        protocol: str | None = None,
        storage_options: Mapping[str, Any] | None = None,
        group_sep: str = "_",
        variables: str | Iterable[str] | None = None,
        # params for op_mode=analysis
        product_type: str | None = None,
        resolution: int | float | None = None,
        spline_order: int | None = None,
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
            product_type: Product type name, such as `"S2B_MSIL1C"`.
                Only used if `op_mode="analysis"` and
                only required if `filename_or_obj` is not a path or URL
                that refers to a product path adhering to EOPF naming conventions.
            protocol: If `filename_or_obj` is a file path or URL,
                forces using the filesystem protocol.
                Otherwise, the protocol will be derived from the file path or URL.
                Will be passed to [`fsspec.filesystem()`](https://filesystem-spec.readthedocs.io/en/latest/usage.html).
            storage_options: If `filename_or_obj` is a file path or URL,
                these options specify the source filesystem.
                Will be passed to [`fsspec.filesystem()`](https://filesystem-spec.readthedocs.io/en/latest/usage.html).
            group_sep: Separator string used to concatenate groups names
                to create prefixes for unique variable and dimension names.
                Defaults to the underscore character (`"_"`)
            resolution: Target resolution for all spatial data variables / bands.
                Must be one of `10`, `20`, or `60`.
                Only used if `op_mode="analysis"`.
            spline_order: Spline order to be used for resampling
                spatial data variables / bands.
                Must be one of `0` (nearest neighbor), `1` (linear),
                `2` (bi-linear), or `3` (cubic).
                Only used if `op_mode="analysis"`
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
        assert_arg_is_one_of(op_mode, "op_mode", OP_MODES)

        datatree = self.open_datatree(
            filename_or_obj,
            op_mode="native",
            protocol=protocol,
            storage_options=storage_options,
            # here as it is required for all backends
            drop_variables=drop_variables,
            # here to silence xarray warnings
            decode_timedelta=decode_timedelta,
        )

        _assert_datatree_is_chunked(datatree)

        if op_mode == OP_MODE_NATIVE:
            dataset = flatten_datatree(datatree, sep=group_sep)
            dataset = filter_dataset(dataset, variables)
        else:  # op_mode == OP_MODE_ANALYSIS
            analysis_mode = AnalysisMode.guess(
                filename_or_obj, product_type=product_type
            )
            params = analysis_mode.get_applicable_params(
                resolution=resolution, spline_order=spline_order
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
