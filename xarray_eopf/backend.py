#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import os
from typing import Any, Iterable

import xarray as xr
from xarray.backends import BackendEntrypoint, AbstractDataStore
from xarray.core.types import ReadBuffer


class EopfBackend(BackendEntrypoint):
    """Backend for EOPF Data Products using the Zarr format."""

    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ) -> xr.DataTree:
        """Backend implementation delegated to by
        [xarray.open_datatree]().
        Args:
            filename_or_obj: File path, or URL, or path-like string.
            drop_variables: variable name or iterable of variable names
                to drop from the underlying file.

        Returns:
            A new data-tree instance.
        """
        data_tree = xr.open_datatree(
            filename_or_obj,
            drop_variables=drop_variables,
        )
        return data_tree

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        drop_variables: str | Iterable[str] | None = None,
    ) -> xr.Dataset:
        """Backend implementation delegated to by
        [xarray.open_dataset]().

        Args:
            filename_or_obj: File path, or URL, or path-like string.
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file.

        Returns:
            A new dataset instance.
        """
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
