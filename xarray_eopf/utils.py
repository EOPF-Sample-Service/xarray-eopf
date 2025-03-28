#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import time
from collections.abc import Collection, Iterable
from typing import Any, Type


class timeit:
    """A context manager used to measure time it takes
    to execute its with-block.
    The result is available as `time_delta` attribute.

    Args:
        label: A text label
        silent: Whether to suppress printing the result
    """

    def __init__(self, label: str | None = None, silent: bool = False):
        self.label = label
        self.silent = silent
        self.start_time: float | None = None
        self.time_delta: float | None = None

    def __enter__(self) -> "timeit":
        self.start_time = time.process_time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.time_delta = time.process_time() - self.start_time
        if not self.silent:
            print(f"{self.label or 'code block'} took {self.time_delta:.3f} seconds")


def assert_arg_is_instance(value: Any, name: str, data_type: Type | tuple[Type, ...]):
    """Check if the `value` of the argument `name` has the given `data_type`.
    If not, raise `TypeError`.
    """
    if not isinstance(value, data_type):
        if isinstance(data_type, tuple):
            data_type_name = _text_items_to_text(t.__name__ for t in data_type)
        else:
            data_type_name = data_type.__name__
        actual_type_name = type(value).__name__
        raise TypeError(
            f"{name} argument must have type {data_type_name}, was {actual_type_name}"
        )


def assert_arg_is_one_of(value: Any, name: str, collection: Collection):
    """Check if the `value` of the argument `name` is one of the items in `collection`.
    If not, raise `ValueError`.
    """
    if value not in collection:
        items_text = _text_items_to_text(map(repr, collection))
        raise ValueError(f"{name} argument must be {items_text}, was {value!r}")


def _text_items_to_text(items: Iterable[str]) -> str:
    items = tuple(items)
    n = len(items)
    if n == 0:
        return ""
    elif n == 1:
        return f"{items[0]}"
    else:
        return f"{', '.join(items[:-1])} or {items[-1]}"
