#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import xarray as xr

from xarray_eopf.constants import DS_MERGE_KWARGS

from .prefix import get_unique_short_sequences


def flatten_datatree(
    datatree: xr.DataTree, prefix: str = "", sep: str = "_"
) -> xr.Dataset:
    """Flatten a given data tree into a single dataset.

    Args:
        datatree: The data tree.
        prefix: Prefix for all variable and dimension names,
            defaults to `""`.
        sep: Group name separator string used
            for variable and dimension names, defaults to `"_"`.

    Returns:
        A single dataset with variable and dimension names
        renamed to be unique.
    """
    prefix_ = f"{prefix}{sep}"

    dataset = datatree.to_dataset()

    if datatree.has_data:
        if prefix != "":
            names = {
                *dataset.sizes.keys(),
                *dataset.coords.keys(),
                *dataset.data_vars.keys(),
            }
            name_mapping = {name: f"{prefix_}{name}" for name in names}
            dataset = dataset.rename(name_mapping)
            # in-place replacement of "preferred_chunks" encoding
            # in dataset variables
            for var_name, var in dataset.variables.items():
                preferred_chunks = var.encoding.get("preferred_chunks")
                if isinstance(preferred_chunks, dict) and preferred_chunks:
                    renamed_chunks = {
                        name_mapping.get(dim, dim): chunk_size
                        for dim, chunk_size in preferred_chunks.items()
                    }
                    var.encoding["preferred_chunks"] = renamed_chunks

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
        child_dataset = flatten_datatree(
            child_datatree,
            prefix=(
                (f"{prefix_}{short_group_name}" if prefix else f"{short_group_name}")
                if group_count > 1
                else prefix
            ),
            sep=sep,
        )
        dataset = dataset.merge(child_dataset, **DS_MERGE_KWARGS)

    return dataset


def _path_for_group_name(group_name: str, sep: str = "_") -> tuple[str, ...]:
    return tuple(group_name.split(sep))


def _name_for_group_path(group_path: tuple[str, ...], sep: str = "_") -> str:
    return sep.join(group_path)


def flatten_datatree_as_dict(
    datatree: xr.DataTree, prefix: str = "", sep: str = "/"
) -> dict[str, xr.Dataset]:
    """Flatten a given data tree into a mapping from keys to datasets.

    Useful for quickly investigating the datasets of a deeply nested
    data tree.

    Args:
        datatree: The data tree.
        prefix: Prefix for all keys, defaults to `""`.
        sep: Group name separator string used in keys, defaults to `"/"`.

    Returns:
        A mapping from keys to datasets. Keys are generated
          by concatenating names of nested groups using
          the separator `sep`.
    """
    datasets = {}
    if datatree.has_data:
        datasets[prefix] = datatree.to_dataset()
    for name, child in datatree.children.items():
        child_prefix = f"{prefix}{sep}{name}" if prefix else name
        datasets.update(flatten_datatree_as_dict(child, prefix=child_prefix, sep=sep))
    return datasets
