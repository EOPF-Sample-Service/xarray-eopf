#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections.abc import Hashable, Mapping
from unittest import TestCase

import dask.array
import xarray as xr


def assert_data_arrays_are_chunked(
    test_case: TestCase,
    variables: Mapping[Hashable, xr.DataArray],
    verbose: bool = False,
):
    if verbose:
        for k, v in variables.items():
            print(f"{k}: s={v.shape}, cs={v.chunksizes}, a={type(v.data)}")

    for k, v in variables.items():
        test_case.assertIsInstance(
            v.data,
            dask.array.Array,
            msg=f"{k} with shape {v.shape} should use a dask array",
        )

    for k, v in variables.items():
        test_case.assertIsNotNone(
            v.chunks,
            msg=(
                f"{k} with shape {v.shape} should be chunked,"
                f" but chunk sizes are {v.chunksizes}"
            ),
        )
