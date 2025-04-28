#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

"""This module comprises reducer functions to be passed to
`dask.array.coarsen()` that either do not exist in numpy, like `mode`,
or have special implementations compared to their numpy equivalents.
"""

import numba as nb
import numpy as np

_ALL = slice(None)

_DOC = """Computes the {property} of the windows in `block`.

Args:
    block: Array block from `dask.array.coarsen()` reshaped into smaller 
        windows to be reduced to size one. For spatial images, its shape will 
        be `(reduced_height, window_size_y, reduced_width, window_size_x)`.
    axis: A tuple providing the indexes of the window dimensions in the shape
        of `block`. For spatial images, this will be `(1, 3)`.

Returns:
    The reduced array containing the {property} of the windows 
    from `block`. For spatial images, its shape will be 
    `(reduced_height, reduced_width)`.
"""


def first(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    if axis is None:
        return block  # edge block, pass through
    index = tuple(0 if i in axis else _ALL for i in range(block.ndim))
    return block[index]


def last(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    if axis is None:
        return block  # edge block, pass through
    index = tuple(-1 if i in axis else _ALL for i in range(block.ndim))
    return block[index]


def center(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    if axis is None:
        return block  # edge block, pass through
    shape = block.shape
    index = tuple(shape[i] // 2 if i in axis else _ALL for i in range(block.ndim))
    return block[index]


def mean(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    return _reduce(np.mean, np.nanmean, block, axis)


def median(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    return _reduce(np.median, np.nanmedian, block, axis)


def std(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    return _reduce(np.std, np.nanstd, block, axis)


def sum(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    return _reduce(np.sum, np.nansum, block, axis)


def var(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    return _reduce(np.var, np.nanvar, block, axis)


def _reduce(
    reducer, nan_reducer, block: np.ndarray, axis: tuple[int, ...] | None = None
) -> np.ndarray:
    if axis is None:
        # edge block, pass through
        return block
    elif np.issubdtype(block.dtype, np.floating):
        # For floating point types use nan-reducer
        return nan_reducer(block, axis)
    else:
        # For integer and boolean types use "normal" reducer
        a = reducer(block, axis)
        if np.issubdtype(a.dtype, np.floating):
            # If result is floating point,
            # round and cast to original type
            return np.rint(a).astype(block.dtype)
        return a


def mode(block: np.ndarray, axis: tuple[int, ...] | None = None) -> np.ndarray:
    if axis is None:
        return block  # edge block, pass through

    # This implementation assumes that `block` contains categorical
    # numbers as it computes most frequent values. It does therefore
    # neither use fuzzy equality comparisons nor checks for NaN should
    # floating point data be passed.

    ndim = len(axis)  # number of dimensions in which to reduce
    block = np.moveaxis(block, axis, range(-ndim, 0))
    flat = block.reshape(-1, np.prod(block.shape[-ndim:]))

    min_val = int(flat.min())
    max_val = int(flat.max())
    mode_range = max_val - min_val + 1

    normalized = (flat - min_val).astype(np.int64)
    mode_indices = _mode_from_normalized(
        normalized, offset=min_val, mode_range=mode_range
    )
    return mode_indices.reshape(block.shape[:-ndim])


@nb.njit()
def _mode_from_normalized(
    flat_block: np.ndarray, offset: int, mode_range: int
) -> np.ndarray:  # pragma: no cover
    size = flat_block.shape[0]
    out = np.empty(size, dtype=np.int64)
    for i in range(size):
        counts = np.zeros(mode_range, dtype=np.int64)
        for val in flat_block[i]:
            counts[val] += 1
        mode_val = 0
        max_count = counts[0]
        for j in range(1, mode_range):
            if counts[j] > max_count:
                max_count = counts[j]
                mode_val = j
        out[i] = mode_val + offset
    return out


first.__doc__ = _DOC.format(property="first value")
last.__doc__ = _DOC.format(property="last value")
center.__doc__ = _DOC.format(property="center value")
mean.__doc__ = _DOC.format(property="mean")
median.__doc__ = _DOC.format(property="median")
mode.__doc__ = _DOC.format(property="mode (most frequent value)")
std.__doc__ = _DOC.format(property="standard deviation")
sum.__doc__ = _DOC.format(property="sum")
var.__doc__ = _DOC.format(property="variance")
