#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from abc import abstractmethod, ABC
from collections.abc import Iterable
from typing import Any, Optional, Type

import xarray as xr

# TODO: Rename `ProductType` into something that better fits
#  its purpose. E.g., `AnalysisMode`.


class ProductType(ABC):
    """Provides product-type specific properties and behaviour
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
        """Transform `datatree` into an analysis-ready form.
        Called from the backend's `open_datatree()` implementation to transform.
        a given `xr.DataTree` into a `xr.Dataset` object.

        Args:
            datatree: The data tree to be transformed.
            params: Product type specific parameters.
                See `get_applicable_params()`.

        Returns:
            A transformed data tree.
        """

    @abstractmethod
    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        **params,
    ) -> xr.Dataset:
        """Convert `datatree` into an analysis-ready dataset form.
        Called from the backend's `open_dataset()` implementation to convert
        a given `xr.DataTree` into a `xr.Dataset` object.

        Args:
            datatree: The data tree to be transformed.
            includes: Variables to include in the dataset. Can be a name
                or regex pattern or iterable of the latter.
            excludes: Variables to exclude from the dataset. Can be a name
                or regex pattern or iterable of the latter.
            params: Product type specific parameters.
                See `get_applicable_params()`.

        Returns:
            A transformed data tree.
        """


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
