#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
import re
import time
from collections.abc import Collection, Iterable
from typing import Any, Generic, Hashable, Type, TypeAlias, TypeVar

import numpy as np
import xarray as xr

T = TypeVar("T")


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
    assert len(items) >= 2
    return f"{', '.join(items[:-1])} or {items[-1]}"


def get_data_tree_item(
    datatree: xr.DataTree, group_path: str | Iterable[str]
) -> xr.DataTree | xr.DataArray | None:
    """Get a group in a data tree given by its group path.

    Args:
        datatree: The data tree object
        group_path: An iterable of group names or a string that
            uses slashes as group name separators

    Returns:
        The group of type `xr.DataTree` or `None` if it cannot be found
    """
    if isinstance(group_path, str):
        group_path = group_path.split("/")
    group = datatree
    for group_name in group_path:
        if group_name:
            if group_name not in group:
                return None
            group = group[group_name]
    return group


Matcher: TypeAlias = Any


class NameFilter:
    def __init__(
        self,
        includes: str | Iterable[str] | None,
        excludes: str | Iterable[str] | None = None,
    ):
        self.includes = NameFilter._norm_patterns(includes)
        self.excludes = NameFilter._norm_patterns(excludes)

    def filter(self, names: Iterable[str]) -> Iterable[str]:
        return filter(self.accept, names)

    def accept(self, var_name: str) -> bool:
        accepted = True
        if self.includes:
            accepted = False
            for p, m in self.includes:
                if var_name == p or var_name.startswith(p) or m.match(var_name):
                    accepted = True
                    break
        if accepted:
            for p, m in self.excludes:
                if var_name == p or var_name.startswith(p) or m.match(var_name):
                    accepted = False
                    break
        return accepted

    @staticmethod
    def _norm_patterns(
        patterns: str | Iterable[str] | None,
    ) -> list[tuple[str, Matcher]]:
        patterns = (patterns,) if isinstance(patterns, str) else (patterns or ())
        return [(p, re.compile(p)) for p in patterns if p]


class NameTypeMapping(Generic[T]):
    @classmethod
    def new(
        cls,
        name: str,
        default: T,
        data: T | dict[T, list[str | np.dtype]] | None,
    ):
        if isinstance(data, dict):
            assert default is not None
            key_type = type(default)
            name_dict = {}
            type_dict = {}
            for value, name_type_list in data.items():
                if not isinstance(value, key_type):
                    raise TypeError(
                        f"{name}: keys must be of type {key_type.__name__},"
                        f" but got key {value}"
                    )
                for name_type in name_type_list:
                    if isinstance(name_type, str):
                        name_dict[name_type] = value
                    elif isinstance(name_type, (type, np.dtype)):
                        type_dict[name_type] = value
                    else:
                        raise TypeError(
                            f"{name}[{value!r}]: name or dtype expected,"
                            f" but got value {name_type!r}"
                        )
            return NameTypeMapping(name_dict, type_dict, default)
        elif data is None:
            return NameTypeMapping({}, {}, default)
        else:
            return NameTypeMapping({}, {}, data)

    def __init__(
        self,
        name_dict: dict[Hashable, T],
        type_dict: dict[np.dtype | type, T],
        default: T,
    ):
        assert name_dict is not None
        assert type_dict is not None
        assert default is not None
        self.name_dict = name_dict
        self.type_dict = type_dict
        self.default = default

    def get(self, name: Hashable, dtype: np.dtype | type) -> T:
        value = self.name_dict.get(name)
        if value is None:
            value = self.type_dict.get(dtype)
            if value is None:
                for k, v in self.type_dict.items():
                    if np.issubdtype(dtype, k):
                        return v
                value = self.default
        return value
