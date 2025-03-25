#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import re
from typing import Iterable

import xarray as xr


def filter_dataset(
    dataset: xr.Dataset, variables: str | Iterable[str] | None
) -> xr.Dataset:
    if not variables:
        return dataset
    var_patterns = (variables,) if isinstance(variables, str) else tuple(variables)
    var_names = [str(var_name) for var_name in dataset.variables.keys()]
    drop_names = []
    for var_pattern in var_patterns:
        var_matcher = re.compile(var_pattern)
        drop_names.extend(
            var_name
            for var_name in var_names
            if not (
                var_name == var_pattern
                or var_name.startswith(var_pattern)
                or var_matcher.match(var_name)
            )
        )
    if drop_names:
        # TODO: also drop unused coordinates + dimensions
        dataset = dataset.drop_vars(drop_names)
    return dataset
