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
from .flatten import flatten_datatree
from .store import open_store


class EopfBackend(BackendEntrypoint):
    """Backend for EOPF Data Products using the Zarr format."""

    fs_cache: dict[str, s3fs.S3FileSystem] = {}

    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        op_mode: OpMode = OP_MODE_ANALYSIS,
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

        # TODO: remove this block once "analysis" mode is supported
        if op_mode != OP_MODE_NATIVE:
            raise ValueError(f"mode {op_mode!r} is not supported yet")

        fs_store = open_store(filename_or_obj, protocol, storage_options)

        data_tree = xr.open_datatree(
            fs_store,
            engine="zarr",
            # preserve the chunking from the Zarr metadata
            chunks="auto",
            # here as it is required for all backends
            drop_variables=drop_variables,
            # here to silence xarray warnings
            decode_timedelta=decode_timedelta,
        )
        return data_tree

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        op_mode: OpMode = OP_MODE_ANALYSIS,
        protocol: str | None = None,
        storage_options: Mapping[str, Any] | None = None,
        group_sep: str = "_",
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
            group_sep: Group name separator string.
                Defaults to the underscore character.
            protocol: If `filename_or_obj` is a file path or URL, 
                forces using the filesystem protocol.
                Otherwise the protocol will be derived from the file path or URL. 
                Will be passed to [`fsspec.filesystem()`]({FSSPEC_USAGE_URL}).
            storage_options: If `filename_or_obj` is a file path or URL,
                these options specify the source filesystem.
                Will be passed to [`fsspec.filesystem()`]({FSSPEC_USAGE_URL}).
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
            op_mode=op_mode,
            protocol=protocol,
            storage_options=storage_options,
            # here as it is required for all backends
            drop_variables=drop_variables,
            # here to silence xarray warnings
            decode_timedelta=decode_timedelta,
        )
        return flatten_datatree(datatree, group_sep)

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


def _assert_valid_op_mode(op_mode: Any):
    if op_mode not in OP_MODES:
        raise ValueError(
            f"mode argument must be {' or '.join(map(repr, OP_MODES))}, was {op_mode!r}"
        )
