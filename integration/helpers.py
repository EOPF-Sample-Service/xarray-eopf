#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import dask.array
import xarray as xr


def assert_dataset_is_chunked(
    test_case: TestCase,
    dataset: xr.Dataset,
    verbose: bool = False,
):
    if verbose:
        for k, v in dataset.data_vars.items():
            print(f"{k}: s={v.shape}, cs={v.chunksizes}, a={type(v.data)}")

    for k, v in dataset.data_vars.items():
        if v.ndim > 1:
            test_case.assertIsInstance(
                v.data,
                dask.array.Array,
                msg=f"{k} with shape {v.shape} should use a dask array",
            )

    for k, v in dataset.data_vars.items():
        if v.ndim > 1:
            test_case.assertIsNotNone(
                v.chunks,
                msg=(f"{k} with shape {v.shape} should be chunked."),
            )
