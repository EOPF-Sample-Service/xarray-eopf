#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from abc import ABC
from typing import Any

import pyproj.crs
import xarray as xr

from xarray_eopf.prodtype import ProductType, ProductTypeRegistry
from xarray_eopf.spatial import get_spatial_vars, rescale_spatial_vars
from xarray_eopf.utils import assert_arg_is_instance, assert_arg_is_one_of


# TODO: add MSI tests


class MSI(ProductType, ABC):
    def is_valid_source(self, source: Any) -> bool:
        if not isinstance(source, str):
            return False
        path: str = source
        return f"S2A_{self.type_name}_" in path or f"S2B_{self.type_name}_" in path

    def get_applicable_params(self, **kwargs) -> dict[str, any]:
        params = {}

        resolution = kwargs.get("resolution")
        if resolution is not None:
            assert_arg_is_instance(resolution, "resolution", (int, float))
            assert_arg_is_one_of(resolution, "resolution", (10, 20, 60))
            params.update(resolution=int(resolution))

        spline_order = kwargs.get("spline_order")
        if spline_order is not None:
            assert_arg_is_instance(spline_order, "spline_order", int)
            assert_arg_is_one_of(spline_order, "spline_order", (0, 1, 2, 3))
            params.update(spline_order=spline_order)

        return params

    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        raise NotImplementedError

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        resolution: int = 10,
        spline_order: int = 0,
    ) -> xr.Dataset:
        # Important note: rescale_spatial_vars() may take very long
        # for some variables!
        # - "conditions_geometry_sun_angles"
        #   with shape (2, 23, 23) takes 120 seconds
        # - "conditions_geometry_viewing_incidence_angles"
        #   with shape (13, 7, 2, 23, 23) takes 140 seconds

        resolution_name = f"r{resolution}m"
        dataset = datatree.measurements.reflectance[resolution_name].ds

        if self.type_name != "MSIL2A" or resolution == 10:
            variables = dict(dataset.data_vars)
            assert len(variables) > 0
            ref_var_name = tuple(get_spatial_vars(variables))[0]

            r_names = [f"r{r}m" for r in (10, 20, 60) if r != resolution]
            for r_name in r_names:
                r_ds = datatree.measurements.reflectance[r_name].ds.rename(
                    {"x": f"{r_name}_x", "y": f"{r_name}_y"}
                )
                variables.update(
                    {k: v for k, v in r_ds.data_vars.items() if k not in variables}
                )

            rescaled_variables = rescale_spatial_vars(
                variables,
                ref_var_name=ref_var_name,
                spline_order=spline_order,
            )

            dataset = xr.Dataset(
                rescaled_variables, attrs=self.process_metadata(datatree)
            )

        dataset.attrs = self.process_metadata(datatree)
        dataset = self.assign_grid_mapping(dataset)
        return dataset

    # noinspection PyMethodMayBeStatic
    def process_metadata(self, datatree):
        # TODO: process metadata and try adhering to CF conventions
        other_metadata = datatree.attrs.get("other_metadata", {})
        return other_metadata

    # noinspection PyMethodMayBeStatic
    def assign_grid_mapping(self, dataset: xr.Dataset) -> xr.Dataset:
        # TODO: check if this is the "official" way to detect a
        #  Sentinel-2 tile's CRS
        crs_code = dataset.attrs.get("horizontal_CRS_code")
        if crs_code:
            crs = pyproj.crs.CRS.from_string(crs_code)
            spatial_ref = xr.DataArray(0, attrs=crs.to_cf())
            dataset = dataset.assign_coords(spatial_ref=spatial_ref)
            for var_name, var in dataset.data_vars.items():
                if var.ndim >= 2 and var.dims[-2:] == ("y", "x"):
                    var.attrs.update(grid_mapping="spatial_ref")
        return dataset


# TODO: add MSIL1C tests


class MSIL1C(MSI):
    type_name = "MSIL1C"


# TODO: add MSIL2A tests


class MSIL2A(MSI):
    type_name = "MSIL2A"


def register(registry: ProductTypeRegistry):
    registry.register(MSIL1C)
    registry.register(MSIL2A)
