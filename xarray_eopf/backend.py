#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import os
from typing import Any, Iterable, Literal, Final, TypeAlias

import xarray as xr
from xarray.backends import BackendEntrypoint, AbstractDataStore
from xarray.core.types import ReadBuffer

MODE_ANALYSIS: Final = "analysis"
MODE_NATIVE: Final = "native"
MODES: Final = MODE_ANALYSIS, MODE_NATIVE

Mode: TypeAlias = Literal["analysis", "native"]


class EopfBackend(BackendEntrypoint):
    """Backend for EOPF Data Products using the Zarr format."""

    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        mode: Mode = "analysis",
        drop_variables: str | Iterable[str] | None = None,
    ) -> xr.DataTree:
        """Backend implementation delegated to by
        [xarray.open_datatree]().
        Args:
            filename_or_obj: File path, or URL, or path-like string.
            mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file.

        Returns:
            A new data-tree instance.
        """
        _assert_valid_mode(mode)
        data_tree = xr.open_datatree(
            filename_or_obj,
            drop_variables=drop_variables,
        )
        return data_tree

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        mode: Mode = "analysis",
        drop_variables: str | Iterable[str] | None = None,
    ) -> xr.Dataset:
        """Backend implementation delegated to by
        [xarray.open_dataset]().

        Args:
            filename_or_obj: File path, or URL, or path-like string.
            mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file.

        Returns:
            A new dataset instance.
        """
        _assert_valid_mode(mode)
        dataset = xr.open_zarr(
            filename_or_obj,
            consolidated=True,
            drop_variables=drop_variables,
        )
        return dataset

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


def _assert_valid_mode(mode: Any):
    if mode not in MODES:
        raise ValueError(
            f"mode argument must be {' or '.join(map(repr, MODES))}, was {mode!r}"
        )
