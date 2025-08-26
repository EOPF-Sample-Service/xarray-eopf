#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import warnings
from abc import ABC
from collections.abc import Iterable
from typing import Any, Hashable

import numpy as np
import pyproj.crs
import xarray as xr
from xcube_resampling.affine import affine_transform_dataset
from xcube_resampling.constants import AggMethods, InterpMethods
from xcube_resampling.gridmapping import GridMapping

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.source import get_source_path
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
SEN2_RESOLUTIONS = list(RESOLUTION_ORDERS.keys())
RESOLUTION_CHUNKSIZE = {10: 1830, 20: 915, 60: 305}

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
LONG_NAME_TRANSLATION = {
    "cld": "Cloud probability, based on Sen2Cor processor",
    "scl": "Scene classification data, based on Sen2Cor processor",
    "snw": "Snow probability, based on Sen2Cor processor",
}


class Msi(AnalysisMode, ABC):
    def is_valid_source(self, source: Any) -> bool:
        root_path = get_source_path(source)
        return (
            (
                f"S2A_{self.product_type}_" in root_path
                or f"S2B_{self.product_type}_" in root_path
                or f"S2C_{self.product_type}_" in root_path
            )
            if root_path
            else False
        )

    def get_applicable_params(self, **kwargs) -> dict[str, any]:
        params = {}

        resolution = kwargs.get("resolution")
        if resolution is not None:
            assert_arg_is_one_of(resolution, "resolution", [10, 20, 60])
            params.update(resolution=resolution)

        interp_methods = kwargs.get("interp_methods")
        if interp_methods is not None:
            assert_arg_is_instance(interp_methods, "interp_methods", (str, int, dict))
            params.update(interp_methods=interp_methods)

        agg_methods = kwargs.get("agg_methods")
        if agg_methods is not None:
            assert_arg_is_instance(agg_methods, "agg_methods", (str, dict))
            params.update(agg_methods=agg_methods)

        return params

    def transform_datatree(self, datatree: xr.DataTree, **params) -> xr.DataTree:
        warnings.warn(
            "Analysis mode not implemented for given source, return data tree as-is."
        )
        return datatree

    def transform_dataset(self, dataset: xr.Dataset, **params) -> xr.Dataset:
        return self.assign_grid_mapping(dataset)

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: int = 10,
        interp_methods: InterpMethods | None = None,
        agg_methods: AggMethods | None = None,
    ) -> xr.Dataset:
        # Important note: rescale_spatial_vars() may take very long
        # for some variables!
        # - "conditions_geometry_sun_angles"
        #   with shape (2, 23, 23) takes 120 seconds
        # - "conditions_geometry_viewing_incidence_angles"
        #   with shape (13, 7, 2, 23, 23) takes 140 seconds

        name_filter = NameFilter(includes=includes, excludes=excludes)

        variables: dict[int, dict[Hashable, xr.DataArray]] = {10: {}, 20: {}, 60: {}}
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
                for k, v in res_ds.data_vars.items():
                    if name_filter.accept(str(k)) and not any(
                        k in variables[sen2_res] for sen2_res in SEN2_RESOLUTIONS
                    ):
                        variables[res][k] = v

        if all(len(v) == 0 for v in variables.values()):
            raise ValueError("No variables selected")
        datasets = dict()
        for res, da_mapping in variables.items():
            if da_mapping:
                datasets[res] = self.assign_grid_mapping(xr.Dataset(da_mapping))

        # resample data set via affine transform
        if resolution in SEN2_RESOLUTIONS and resolution in datasets:
            target_gm = GridMapping.from_dataset(datasets[resolution])
        else:
            res, ds = next(iter(datasets.items()))
            resh = res / 2
            bbox = [ds.x[0] - resh, ds.y[-1] - resh, ds.x[-1] + resh, ds.y[0] + resh]
            x_size = np.ceil((bbox[2] - bbox[0]) / resolution)
            y_size = np.ceil(abs(bbox[3] - bbox[1]) / resolution)
            chunk_size = RESOLUTION_CHUNKSIZE[
                min(SEN2_RESOLUTIONS, key=lambda x: abs(x - resolution))
            ]
            target_gm = GridMapping.regular(
                size=(x_size, y_size),
                xy_min=(bbox[0], bbox[1]),
                xy_res=resolution,
                crs=pyproj.CRS.from_wkt(ds.spatial_ref.attrs["crs_wkt"]),
                tile_size=chunk_size,
            )

        rescaled_ds = None
        for res, ds in datasets.items():
            ds = affine_transform_dataset(
                ds,
                target_gm=target_gm,
                interp_methods=interp_methods,
                agg_methods=agg_methods,
            )
            if rescaled_ds is None:
                rescaled_ds = ds
            else:
                rescaled_ds.update(ds)

        # Assign extra variable attributes
        for var_name in rescaled_ds.data_vars:
            attrs = EXTRA_VAR_ATTRS.get(var_name)
            if attrs:
                rescaled_ds[var_name].attrs.update(attrs)
            if var_name in LONG_NAME_TRANSLATION.keys():
                rescaled_ds[var_name].attrs["long_name"] = LONG_NAME_TRANSLATION[
                    var_name
                ]

        rescaled_ds.attrs = self.process_metadata(datatree)
        return rescaled_ds

    # noinspection PyMethodMayBeStatic
    def process_metadata(self, datatree: xr.DataTree | xr.Dataset):
        # TODO: process metadata and try adhering to CF conventions
        other_metadata = datatree.attrs.get("other_metadata", {})
        return other_metadata

    # noinspection PyMethodMayBeStatic
    def assign_grid_mapping(self, dataset: xr.Dataset) -> xr.Dataset:
        code_to_crs: dict[int, pyproj.CRS] = {}
        var_name_to_code: dict[Hashable, int] = {}
        for var_name, var in dataset.data_vars.items():
            code = var.attrs.get("proj:epsg")
            if isinstance(code, int):
                crs = code_to_crs.get(code)
                if crs is None:
                    try:
                        crs = pyproj.CRS.from_epsg(code)
                        code_to_crs[code] = crs
                    except pyproj.exceptions.CRSError:
                        crs = None
                if crs:
                    var_name_to_code[var_name] = code

        if code_to_crs:
            is_single = len(code_to_crs) == 1
            spatial_ref_names: dict[int, Hashable] = {}
            spatial_refs: dict[Hashable, xr.DataArray] = {}
            for i, (code, crs) in enumerate(code_to_crs.items()):
                spatial_ref_name = (
                    "spatial_ref" if is_single else f"spatial_ref_{i + 1}"
                )
                spatial_refs[spatial_ref_name] = xr.DataArray(0, attrs=crs.to_cf())
                spatial_ref_names[code] = spatial_ref_name
            dataset = dataset.assign_coords(spatial_refs)
            for var_name, code in var_name_to_code.items():
                spatial_ref_name = spatial_ref_names[code]
                dataset[var_name].attrs["grid_mapping"] = spatial_ref_name

        return dataset


class MsiL1c(Msi):
    product_type = "MSIL1C"


class MsiL2a(Msi):
    product_type = "MSIL2A"


def register(registry: AnalysisModeRegistry):
    registry.register(MsiL1c)
    registry.register(MsiL2a)
