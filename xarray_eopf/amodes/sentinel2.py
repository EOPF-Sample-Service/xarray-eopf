#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.
import warnings
from abc import ABC
from collections.abc import Iterable
from typing import Any, Hashable

import pyproj.crs
import xarray as xr

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.source import get_source_paths
from xarray_eopf.spatial import SplineOrder, get_spatial_vars, rescale_spatial_vars
from xarray_eopf.utils import (
    NameFilter,
    assert_arg_is_instance,
    assert_arg_is_one_of,
    get_data_tree_item,
)

# Resolutions of bands and variables in the order they contribute
# to a dataset (=value) for a target resolution (= key).
#
RESOLUTION_ORDERS = {
    10: (10, 20, 60),
    20: (20, 10, 60),
    60: (60, 20, 10),
}

# Groups in L1C and L2A that contain resolution groups
# (r10m, r20m, r60m) that contain a dataset
#
GROUP_PATHS = (
    ("measurements", "reflectance"),
    ("quality", "probability"),
    ("conditions", "mask", "l2a_classification"),
)

# Extra attributes (= value) that will be added to the
# named variables (= keys)
#
EXTRA_VAR_ATTRS: dict[Hashable, dict[str, Any]] = {
    "scl": {
        "flag_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "flag_meanings": (
            "no_data "
            "sat_or_defect_pixel "
            "topo_casted_shadows "
            "cloud_shadows "
            "vegetation "
            "not_vegetation "
            "water "
            "unclassified "
            "cloud_medium_prob "
            "cloud_high_prob "
            "thin_cirrus "
            "snow_or_ice"
        ),
        "flag_colors": (
            "#000000 #ff0000 #2f2f2f #643200 "
            "#00a000 #ffe65a #0000ff #808080 "
            "#c0c0c0 #ffffff #64c8ff #ff96ff"
        ),
    }
}


class MSI(AnalysisMode, ABC):
    def is_valid_source(self, source: Any) -> bool:
        root_path, _ = get_source_paths(source)
        return (
            (
                f"S2A_{self.product_type}_" in root_path
                or f"S2B_{self.product_type}_" in root_path
            )
            if root_path
            else False
        )

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
        warnings.warn(
            "Analysis mode not implemented for given source, return data tree as-is."
        )
        return datatree

    def transform_dataset(self, dataset: xr.Dataset, **params) -> xr.Dataset:
        warnings.warn(
            "Analysis mode not implemented for given source, return dataset as-is."
        )
        return dataset

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: int = 10,
        spline_order: SplineOrder = 0,
    ) -> xr.Dataset:
        # Important note: rescale_spatial_vars() may take very long
        # for some variables!
        # - "conditions_geometry_sun_angles"
        #   with shape (2, 23, 23) takes 120 seconds
        # - "conditions_geometry_viewing_incidence_angles"
        #   with shape (13, 7, 2, 23, 23) takes 140 seconds

        name_filter = NameFilter(includes=includes, excludes=excludes)
        ref_var_name: str | None = None

        variables: dict[Hashable, xr.DataArray] = {}
        for group_path in GROUP_PATHS:
            group = get_data_tree_item(datatree, group_path)
            if group is None:
                continue
            for res in RESOLUTION_ORDERS[resolution]:
                res_name = f"r{res}m"
                if res_name not in group:
                    continue
                res_group = group[res_name]
                res_ds = res_group.ds
                if res != resolution:
                    res_ds = res_ds.rename({"x": f"{res_name}_x", "y": f"{res_name}_y"})
                spatial_vars = get_spatial_vars(res_ds.data_vars)
                for k, v in spatial_vars.items():
                    if name_filter.accept(str(k)) and (k not in variables):
                        if ref_var_name is None and res == resolution:
                            ref_var_name = str(k)
                        variables[k] = v

        if not variables:
            raise ValueError("No variables selected")
        if ref_var_name is None:
            raise ValueError(
                "No reference variable found. At least one of the selected"
                " variables must have a native resolution that equals"
                " the target resolution."
            )

        rescaled_variables = rescale_spatial_vars(
            variables,
            ref_var_name=ref_var_name,
            spline_order=spline_order,
        )

        # Assign extra variable attributes
        for var_name, var in rescaled_variables.items():
            attrs = EXTRA_VAR_ATTRS.get(var_name)
            if attrs:
                var.attrs.update(attrs)

        dataset = xr.Dataset(rescaled_variables, attrs=self.process_metadata(datatree))
        dataset.attrs = self.process_metadata(datatree)
        dataset = self.assign_grid_mapping(dataset)
        return dataset

    # noinspection PyMethodMayBeStatic
    def process_metadata(self, datatree: xr.DataTree | xr.Dataset):
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


class MSIL1C(MSI):
    product_type = "MSIL1C"


class MSIL2A(MSI):
    product_type = "MSIL2A"


def register(registry: AnalysisModeRegistry):
    registry.register(MSIL1C)
    registry.register(MSIL2A)
