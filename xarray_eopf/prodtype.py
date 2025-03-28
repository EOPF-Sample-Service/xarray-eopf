#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from abc import abstractmethod, ABC
from collections.abc import Collection
from typing import Any, Optional, Type

import xarray as xr


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
    def from_source(cls, source: Any = None) -> Optional["ProductType"]:
        """Get product type from given object `source`
        that was used or can be used to open the datatree or dataset.
        It may be a URL, a path, or another source object.
        """
        for pt in registry.values():
            if pt.is_valid_source(source):
                return pt
        return None

    @abstractmethod
    def is_valid_source(self, source: Any) -> bool:
        """Check if this product type is applicable to or can be represented
        by the given object `source`.
        """

    @abstractmethod
    def get_applicable_params(self, **kwargs) -> dict[str, any]:
        """Get applicable and validated parameters from keyword arguments `kwargs`.
        The extracted parameters will be passed to `transform_datatree()`
        and `convert_datatree()`.
        """

    @abstractmethod
    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        """Transform `datatree` into an analysis-ready form."""

    @abstractmethod
    def convert_datatree(self, datatree: xr.DataTree, **params) -> xr.Dataset:
        """Convert `datatree` into an analysis-ready dataset form."""


class ProductTypeRegistry:
    """A simple registry for `ProductType` instances."""

    def __init__(self):
        self._product_types: dict[str, ProductType] = {}

    def keys(self) -> tuple[str, ...]:
        """Get registered product type names."""
        return tuple(self._product_types.keys())

    def values(self) -> tuple[ProductType, ...]:
        """Get registered product types."""
        # noinspection PyTypeChecker
        return tuple(self._product_types.values())

    def get(self, type_name: str) -> Optional["ProductType"]:
        """Get a specific product types for given `type_name`."""
        return self._product_types.get(type_name)

    def register(self, cls: Type[ProductType]):
        """Register the product type given as its class `cls`."""
        assert issubclass(cls, ProductType)
        assert isinstance(cls.type_name, str)
        self._product_types[cls.type_name] = cls()


registry = ProductTypeRegistry()
