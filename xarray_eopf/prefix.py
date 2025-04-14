#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from collections import defaultdict
from collections.abc import Collection, Hashable, Mapping, Sequence
from typing import Callable, TypeVar

T = TypeVar("T", bound=Hashable)


def get_common_prefix(seq1: Sequence[T], seq2: Sequence[T]) -> Sequence[T]:
    prefix = []
    for item1, item2 in zip(seq1, seq2):
        if item1 == item2:
            prefix.append(item1)
        else:
            break
    if len(prefix) == len(seq1) and len(prefix) == len(seq2):
        return []
    return prefix


def get_common_string_prefix(str1: str, str2: str) -> str:
    return "".join(get_common_prefix(str1, str2))


def get_unique_short_sequences(
    sequences: Collection[Sequence[T]],
) -> Mapping[tuple[T, ...], tuple[T, ...]]:
    return _get_unique_short_sequences(
        [s if isinstance(s, tuple) else tuple(s) for s in sequences],
        get_common_prefix,
        (),
    )


def get_unique_short_strings(strings: Collection[str]) -> Mapping[str, str]:
    # noinspection PyTypeChecker
    return _get_unique_short_sequences(strings, get_common_string_prefix, "")


def _get_unique_short_sequences(
    sequences: Collection[tuple[T, ...]],
    get_prefix: Callable[[T, T], T],
    max_default: T,
) -> Mapping[tuple[T, ...], tuple[T, ...]]:
    mapping = {}
    prefix_groups = defaultdict(list)

    for sequence in sequences:
        for other in sequences:
            if sequence != other:
                prefix = get_prefix(sequence, other)
                prefix_groups[sequence].append(prefix)

    for sequence in sequences:
        max_prefix = max(prefix_groups[sequence], key=len, default=max_default)
        unique_part = sequence[len(max_prefix) :]
        mapping[sequence] = unique_part if unique_part else sequence

    return mapping
