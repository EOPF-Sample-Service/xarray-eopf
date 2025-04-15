#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from typing import Iterable

import xarray as xr

from .utils import NameFilter


def filter_dataset(
    dataset: xr.Dataset,
    variables: str | Iterable[str] | None,
) -> xr.Dataset:
    if not variables:
        return dataset
    name_filter = NameFilter(includes=variables)
    names = set(map(str, dataset.variables.keys()))
    # find all dataset variables including respective coordinates
    drop_names = names - set(name_filter.filter(names))
    if drop_names:
        dataset = dataset.drop_vars(drop_names)
    return dataset
