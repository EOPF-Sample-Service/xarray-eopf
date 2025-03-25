from typing import Any, Sequence

import xarray as xr

from xarray_eopf.prodtype import ProductType, ProductTypeRegistry
from xarray_eopf.spatial import get_spatial_vars, rescale_spatial_vars


class MSIL1C(ProductType):
    def is_applicable(self, path_or_obj: Any) -> bool:
        if not isinstance(path_or_obj, str):
            return False
        path: str = path_or_obj
        return "S2A_MSIL1C_" in path or "S2B_MSIL1C_" in path

    def validate_params(self, params: dict[str, Any]) -> Sequence[str]:
        pass

    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        raise NotImplementedError

    def convert_datatree(self, datatree: xr.DataTree, **params) -> xr.Dataset:
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
        rescaled_spatial_vars = rescale_spatial_vars(spatial_vars, ref_var_name="b02")
        return xr.Dataset(rescaled_spatial_vars, attrs=r10m_ds.attrs)


def register_s2_product_types(registry: ProductTypeRegistry):
    registry.register("MSIL1C", MSIL1C())
