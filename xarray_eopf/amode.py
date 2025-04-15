#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional, Type

import xarray as xr


class AnalysisMode(ABC):
    """Provides product-type specific properties and behaviour
    for the EOPF backend's "analysis" mode of operation.
    """

    # Registry for analysis modes
    registry: "AnalysisModeRegistry"

    # Product type name, e.g., "MSIL2A"
    product_type: str

    @classmethod
    def from_product_type(cls, product_type: str | None) -> Optional["AnalysisMode"]:
        """Get the analysis mode for given `product_type`."""
        return cls.registry.get(product_type)

    @classmethod
    def from_source(cls, source: Any = None) -> Optional["AnalysisMode"]:
        """Get the analysis mode for given object `source`
        that was used or can be used to open the datatree or dataset.
        It may be a URL, a path, or another source object.
        """
        for pt in cls.registry.values():
            if pt.is_valid_source(source):
                return pt
        return None

    @abstractmethod
    def is_valid_source(self, source: Any) -> bool:
        """Check if this analysis mode is applicable to or can be represented
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
        Called from the backend's `open_datatree()` implementation to transform
        the given `xr.DataTree` object.

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


class AnalysisModeRegistry:
    """A simple registry for `AnalysisMode` instances."""

    def __init__(self):
        self._analysis_modes: dict[str, AnalysisMode] = {}

    def keys(self) -> tuple[str, ...]:
        """Get registered analysis mode keys."""
        return tuple(self._analysis_modes.keys())

    def values(self) -> tuple[AnalysisMode, ...]:
        """Get registered analysis modes."""
        # noinspection PyTypeChecker
        return tuple(self._analysis_modes.values())

    def get(self, product_type: str) -> Optional["AnalysisMode"]:
        """Get a specific analysis modes for given `product_type`."""
        return self._analysis_modes.get(product_type)

    def register(self, cls: Type[AnalysisMode]):
        """Register the analysis mode given as its class `cls`."""
        assert issubclass(cls, AnalysisMode)
        assert isinstance(cls.product_type, str)
        self._analysis_modes[cls.product_type] = cls()

    def unregister(self, cls: Type[AnalysisMode]):
        """Unregister the analysis mode given as its class `cls`."""
        assert issubclass(cls, AnalysisMode)
        assert isinstance(cls.product_type, str)
        del self._analysis_modes[cls.product_type]


AnalysisMode.registry = AnalysisModeRegistry()
