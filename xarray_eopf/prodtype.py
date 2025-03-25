#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from abc import abstractmethod, ABC
from typing import Any, Optional

import xarray as xr


class ProductType(ABC):
    @classmethod
    def from_name(cls, name: str | None) -> Optional["ProductType"]:
        return registry.get(name)

    @classmethod
    def from_path_or_obj(cls, path_or_obj: Any = None):
        for pt in registry.values():
            if pt.is_applicable(path_or_obj):
                return pt
        return None

    @abstractmethod
    def is_applicable(self, path_or_obj: Any) -> bool:
        """Check if the given path or object is applicable or represents
        this product type.
        """

    @abstractmethod
    def validate_params(self, params: dict[str, Any]):
        """Validate given parameters."""

    @abstractmethod
    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        """Transform `datatree` into an prodtypes-ready form."""

    @abstractmethod
    def convert_datatree(self, datatree: xr.DataTree, **params) -> xr.Dataset:
        """Convert `datatree` into an prodtypes-ready dataset form."""


class ProductTypeRegistry:
    def __init__(self):
        self._product_types: dict[str, ProductType] = {}

    def keys(self) -> tuple[str, ...]:
        return tuple(self._product_types.keys())

    def values(self) -> tuple[ProductType, ...]:
        # noinspection PyTypeChecker
        return tuple(self._product_types.values())

    def get(self, name: str) -> Optional["ProductType"]:
        return self._product_types.get(name)

    def register(
        self, name: str, product_type: "ProductType"
    ) -> Optional["ProductType"]:
        self._product_types[name] = product_type


registry = ProductTypeRegistry()
