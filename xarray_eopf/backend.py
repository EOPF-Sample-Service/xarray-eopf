#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import os
from typing import Any, Iterable, Literal, Final, TypeAlias

import xarray as xr
from xarray.backends import BackendEntrypoint, AbstractDataStore
from xarray.core.types import ReadBuffer

OP_MODE_ANALYSIS: Final = "analysis"
OP_MODE_NATIVE: Final = "native"
OP_MODES: Final = OP_MODE_ANALYSIS, OP_MODE_NATIVE

OpMode: TypeAlias = Literal["analysis", "native"]

# Keywords arguments passed to dataset.merge(other) when flattening
# data trees.
MERGE_KWARGS: Final = dict(
    # skip comparing and pick variable from `dataset`
    compat="override",
    # use indexes from `dataset` that are the same size
    # as those of `other` in that dimension
    join="override",
    # skip comparing and copy attrs from `dataset` to
    # the result.
    combine_attrs="override",
)


class EopfBackend(BackendEntrypoint):
    """Backend for EOPF Data Products using the Zarr format."""

    def open_datatree(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        op_mode: OpMode = "analysis",
        drop_variables: str | Iterable[str] | None = None,
    ) -> xr.DataTree:
        """Backend implementation delegated to by
        [xarray.open_datatree]().
        Args:
            filename_or_obj: File path, or URL, or path-like string.
            op_mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file.

        Returns:
            A new data-tree instance.
        """
        _assert_valid_op_mode(op_mode)

        # TODO: remove this block once "analysis" mode is supported
        if op_mode != OP_MODE_NATIVE:
            raise ValueError(f"mode {op_mode!r} is not supported yet")

        data_tree = xr.open_datatree(
            filename_or_obj,
            drop_variables=drop_variables,
        )
        return data_tree

    def open_dataset(
        self,
        filename_or_obj: str | os.PathLike[Any] | ReadBuffer | AbstractDataStore,
        *,
        op_mode: OpMode = "analysis",
        group_sep: str = "_",
        drop_variables: str | Iterable[str] | None = None,
    ) -> xr.Dataset:
        """Backend implementation delegated to by
        [xarray.open_dataset]().

        Args:
            filename_or_obj: File path, or URL, or path-like string.
            op_mode: Mode of operation, either "analysis" or "native".
                Defaults to "analysis".
            group_sep: Group name separator string.
                Defaults to the underscore character.
            drop_variables: Variable name or iterable of variable names
                to drop from the underlying file.

        Returns:
            A new dataset instance.
        """
        _assert_valid_op_mode(op_mode)
        datatree = self.open_datatree(
            filename_or_obj,
            op_mode=op_mode,
            drop_variables=drop_variables,
        )
        return _flatten_datatree(datatree, group_sep)

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


def _flatten_datatree(
    datatree: xr.DataTree, group_sep: str, prefix: str = ""
) -> xr.Dataset:

    prefix_ = f"{prefix}{group_sep}"

    dataset = datatree.to_dataset()

    if datatree.is_leaf:
        if prefix != "":
            names = {
                *dataset.sizes.keys(),
                *dataset.coords.keys(),
                *dataset.data_vars.keys(),
            }
            dataset = dataset.rename({name: f"{prefix_}{name}" for name in names})
        return dataset

    for child_name, child_datatree in datatree.children.items():
        child_dataset = _flatten_datatree(
            child_datatree,
            group_sep=group_sep,
            prefix=f"{prefix_}{child_name}" if prefix else f"{child_name}",
        )
        dataset = dataset.merge(child_dataset, **MERGE_KWARGS)

    return dataset


def _assert_valid_op_mode(op_mode: Any):
    if op_mode not in OP_MODES:
        raise ValueError(
            f"mode argument must be {' or '.join(map(repr, OP_MODES))}, was {op_mode!r}"
        )
