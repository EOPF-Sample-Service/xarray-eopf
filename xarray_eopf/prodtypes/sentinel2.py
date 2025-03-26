#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from abc import ABC
from typing import Any

import xarray as xr

from xarray_eopf.prodtype import ProductType, ProductTypeRegistry
from xarray_eopf.spatial import get_spatial_vars, rescale_spatial_vars


# TODO: add MSI tests


class MSI(ProductType, ABC):
    def is_applicable(self, path_or_obj: Any) -> bool:
        if not isinstance(path_or_obj, str):
            return False
        path: str = path_or_obj
        return f"S2A_{self.type_name}_" in path or f"S2B_{self.type_name}_" in path

    def validate_params(self, params: dict[str, Any]):
        if "resolution" in params:
            resolution = params["resolution"]
            if resolution not in (10, 20, 60):
                raise ValueError("resolution must be one of 10, 20, 60 (meters)")

        if "spline_order" in params:
            spline_order = params["spline_order"]
            if spline_order not in (0, 1, 2, 3):
                raise ValueError("spline_order must be in the range 0 to 3")

    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        raise NotImplementedError

    def convert_datatree(
        self, datatree: xr.DataTree, spline_order: int = 0, resolution: int = 10
    ) -> xr.Dataset:
        # TODO: recognize spline_order
        # TODO: recognize resolution

        # Note:
        # - rescaling conditions_geometry_sun_angles
        #   with shape (2, 23, 23) takes 120 seconds!
        # - rescaling conditions_geometry_viewing_incidence_angles
        #   with shape (13, 7, 2, 23, 23) takes 140 seconds!

        r10m_ds = datatree.measurements.reflectance.r10m.ds
        r20m_ds = datatree.measurements.reflectance.r20m.ds.rename(
            {"x": "r20m_x", "y": "r20m_y"}
        )
        r60m_ds = datatree.measurements.reflectance.r60m.ds.rename(
            {"x": "r20m_x", "y": "r20m_y"}
        )

        spatial_vars = get_spatial_vars(
            {**r10m_ds.data_vars, **r20m_ds.data_vars, **r60m_ds.data_vars},
        )

        rescaled_spatial_vars = rescale_spatial_vars(
            spatial_vars, ref_var_name="b02", spline_order=spline_order
        )

        return xr.Dataset(rescaled_spatial_vars, attrs=r10m_ds.attrs)


# TODO: add MSIL1C tests


class MSIL1C(MSI):
    type_name = "MSIL1C"


# TODO: add MSIL2A tests


class MSIL2A(MSI):
    type_name = "MSIL2A"


def register(registry: ProductTypeRegistry):
    registry.register(MSIL1C)
    registry.register(MSIL2A)
