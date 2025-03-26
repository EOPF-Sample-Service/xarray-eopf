#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from abc import abstractmethod, ABC
from typing import Any, Optional, Type

import xarray as xr


# TODO: add ProductType tests


class ProductType(ABC):
    """Provides product-type specific functionality
    for the EOPF backend's "analysis" mode.
    """

    # Product type name, e.g., "MSIL2A"
    type_name: str

    @classmethod
    def from_name(cls, type_name: str | None) -> Optional["ProductType"]:
        """Get product type from given `type_name`."""
        return registry.get(type_name)

    @classmethod
    def from_object(cls, obj: Any = None):
        """Get product type from given object `obj`
        that was used to open the datatree or dataset.
        It may be a URL, a path, or another source object .
        """
        for pt in registry.values():
            if pt.is_applicable(obj):
                return pt
        return None

    @abstractmethod
    def is_applicable(self, path_or_obj: Any) -> bool:
        """Check if the given path or object is applicable or represents
        this product type.
        """

    @abstractmethod
    def validate_params(self, params: dict[str, Any]):
        """Validate given product-type specific parameters."""

    @abstractmethod
    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        """Transform `datatree` into an analysis-ready form."""

    @abstractmethod
    def convert_datatree(self, datatree: xr.DataTree, **params) -> xr.Dataset:
        """Convert `datatree` into an analysis-ready dataset form."""


# TODO: add ProductTypeRegistry docstrings
# TODO: add ProductTypeRegistry tests


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

    def register(self, cls: Type[ProductType]):
        assert issubclass(cls, ProductType)
        assert isinstance(cls.type_name, str)
        self._product_types[cls.type_name] = cls()


registry = ProductTypeRegistry()
