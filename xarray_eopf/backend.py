#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import os
from collections.abc import Mapping
from typing import Any, Iterable

import s3fs
import xarray as xr
from xarray.backends import BackendEntrypoint, AbstractDataStore
from xarray.coding.times import CFTimedeltaCoder
from xarray.core.types import ReadBuffer

from .constants import (
    OpMode,
    OP_MODE_ANALYSIS,
    OP_MODE_NATIVE,
    OP_MODES,
    OPEN_DS_URL,
    OPEN_DT_URL,
    FSSPEC_USAGE_URL,
)
from .filter import filter_dataset
from .flatten import flatten_datatree, flatten_datatree_as_dict
from .prodtype import ProductType
from .store import open_store

from .prodtypes import register_product_types


class EopfBackend(BackendEntrypoint):
    """Backend for EOPF Data Products using the Zarr format."""

    fs_cache: dict[str, s3fs.S3FileSystem] = {}

    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        op_mode: OpMode = OP_MODE_ANALYSIS,
        product_name: str | None = None,
        protocol: str | None = None,
        storage_options: Mapping[str, Any] | None = None,
        drop_variables: str | Iterable[str] | None = None,
        decode_timedelta: (
            bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None
        ) = False,
    ) -> xr.DataTree:
        f"""Backend implementation delegated to by
        [`xarray.open_datatree()`]({OPEN_DT_URL}).
        
        Args:
            filename_or_obj: File path, or URL, or path-like string.
            op_mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            product_name: Product type name, such as `"S2B_MSIL1C"`. 
                Only used if `op_mode="analysis"` and
                only required if `filename_or_obj` is not a path or URL 
                that refers to a product path adhering to EOPF naming conventions.
            protocol: If `filename_or_obj` is a file path or URL, 
                forces using the filesystem protocol.
                Otherwise the protocol will be derived from the file path or URL. 
                Will be passed to [`fsspec.filesystem()`]({FSSPEC_USAGE_URL}).
            storage_options: If `filename_or_obj` is a file path or URL,
                these options specify the source filesystem.
                Will be passed to [`fsspec.filesystem()`]({FSSPEC_USAGE_URL}).
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file. See
                [xarray documentation]({OPEN_DT_URL}).
            decode_timedelta: How to decode time-delta units. See
                [xarray documentation]({OPEN_DT_URL}).

        Returns:
            A new data-tree instance.
        """
        _assert_valid_op_mode(op_mode)

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
            product_type = _guess_product_type(filename_or_obj, product_name)
            # TODO: derive product-type specific params
            params = {}
            product_type.validate_params(params)
            return product_type.transform_datatree(datatree, **params)

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        op_mode: OpMode = OP_MODE_ANALYSIS,
        product_name: str | None = None,
        protocol: str | None = None,
        storage_options: Mapping[str, Any] | None = None,
        group_sep: str = "_",
        variables: str | Iterable[str] | None = None,
        drop_variables: str | Iterable[str] | None = None,
        decode_timedelta: (
            bool | CFTimedeltaCoder | Mapping[str, bool | CFTimedeltaCoder] | None
        ) = False,
    ) -> xr.Dataset:
        f"""Backend implementation delegated to by
        [`xarray.open_dataset()`]({OPEN_DS_URL}).

        Args:
            filename_or_obj: File path, or URL, or path-like string.
            op_mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            product_name: Product type name, such as `"S2B_MSIL1C"`. 
                Only used if `op_mode="analysis"` and
                only required if `filename_or_obj` is not a path or URL 
                that refers to a product path adhering to EOPF naming conventions.
            protocol: If `filename_or_obj` is a file path or URL, 
                forces using the filesystem protocol.
                Otherwise the protocol will be derived from the file path or URL. 
                Will be passed to [`fsspec.filesystem()`]({FSSPEC_USAGE_URL}).
            storage_options: If `filename_or_obj` is a file path or URL,
                these options specify the source filesystem.
                Will be passed to [`fsspec.filesystem()`]({FSSPEC_USAGE_URL}).
            group_sep: Group name separator string.
                Defaults to the underscore character.
            variables: Variable name or regex pattern or iterable of 
                the latter to include in the dataset.
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file. See
                [xarray documentation]({OPEN_DS_URL}).
            decode_timedelta: How to decode time-delta units. See
                [xarray documentation]({OPEN_DS_URL}).

        Returns:
            A new dataset instance.
        """
        _assert_valid_op_mode(op_mode)

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
            return dataset
        else:  # op_mode == OP_MODE_ANALYSIS
            product_type = _guess_product_type(filename_or_obj, product_name)
            # TODO: derive product-type specific params
            params = {}
            product_type.validate_params(params)
            return product_type.convert_datatree(datatree, **params)

    def guess_can_open(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
    ) -> bool:
        """Check if the given `filename_or_obj` refers to an object that
        can be opened by this backend.

        Args:
            filename_or_obj: File path, or URL, or path-like string.

        Returns:
            Currently always `False`.
        """
        return False


def _guess_product_type(filename_or_obj: Any, product_name: str | None) -> ProductType:
    product_type: ProductType | None = None
    if product_name:
        product_type = ProductType.from_name(product_name)
    if product_type is None:
        product_type = ProductType.from_object(filename_or_obj)
    if product_type is None:
        raise ValueError("unable to detect product type")
    return product_type


def _assert_valid_op_mode(op_mode: Any):
    if op_mode not in OP_MODES:
        raise ValueError(
            f"mode argument must be {' or '.join(map(repr, OP_MODES))}, was {op_mode!r}"
        )


def _assert_datatree_is_chunked(datatree: xr.DataTree):
    for ds_name, ds in flatten_datatree_as_dict(datatree).items():
        _assert_dataset_is_chunked(ds, name=ds_name)


def _assert_dataset_is_chunked(dataset: xr.Dataset, name: str | None = None):
    ds_name = name or "dataset"
    for var_name, var in dataset.data_vars.items():
        assert var.chunks is not None, f"{ds_name}.{var_name}: no chunks"
        # chunk_shape = tuple(
        #     (max(*c) if len(c) > 1 else c[0]) for c in var.chunks
        # )
        # assert var.shape != chunk_shape, (
        #     f"{ds_name}.{var_name}: shape equals chunking"
        # )


register_product_types()
