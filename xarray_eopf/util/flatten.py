#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import xarray as xr

from xarray_eopf.constants import DS_MERGE_KWARGS
from .prefix import get_unique_short_sequences


def flatten_datatree(datatree: xr.DataTree, group_sep: str = "_") -> xr.Dataset:
    return _flatten_datatree_rec(datatree, group_sep=group_sep, prefix="")


def _flatten_datatree_rec(
    datatree: xr.DataTree, group_sep: str, prefix: str
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

    group_names = set(datatree.children.keys())
    short_group_name_paths = get_unique_short_sequences(
        list(map(_path_for_group_name, group_names))
    )
    group_count = len(datatree.children)
    for group_name, child_datatree in datatree.children.items():
        group_name_path = _path_for_group_name(group_name)
        short_group_name_path = short_group_name_paths[group_name_path]
        short_group_name = _name_for_group_path(short_group_name_path)
        child_dataset = _flatten_datatree_rec(
            child_datatree,
            group_sep,
            (
                (f"{prefix_}{short_group_name}" if prefix else f"{short_group_name}")
                if group_count > 1
                else prefix
            ),
        )
        dataset = dataset.merge(child_dataset, **DS_MERGE_KWARGS)

    return dataset


def _path_for_group_name(group_name: str, sep: str = "_") -> tuple[str, ...]:
    return tuple(group_name.split(sep))


def _name_for_group_path(group_path: tuple[str, ...], sep: str = "_") -> str:
    return sep.join(group_path)
