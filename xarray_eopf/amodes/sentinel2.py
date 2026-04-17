#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import warnings
from abc import ABC
from collections.abc import Iterable, Sequence
from typing import Any, Hashable

import numpy as np
import pyproj.crs
import xarray as xr
from xcube_resampling.constants import SpatialAggMethods, SpatialInterpMethods
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.spatial import resample_in_space
from xcube_resampling.utils import reproject_bbox

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.constants import CONVERSION_FACTOR_DEG_METER
from xarray_eopf.source import get_source_path
from xarray_eopf.utils import (
    NameFilter,
    assert_arg_has_length,
    assert_arg_is_instance,
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
COMMON_BAND_NAMES = {
    "coastal": "b01",
    "blue": "b02",
    "green": "b03",
    "red": "b04",
    "rededge071": "b05",
    "rededge075": "b06",
    "rededge078": "b07",
    "nir": "b08",
    "nir08": "b8a",
    "nir09": "b09",
    "cirrus": "b10",
    "swir16": "b11",
    "swir22": "b12",
}
COMMON_BAND_NAMES_REVERSE = {v: k for k, v in COMMON_BAND_NAMES.items()}


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

    def get_applicable_params(self, **kwargs) -> dict[str, Any]:
        params = {}

        resolution = kwargs.get("resolution")
        if resolution is not None:
            assert_arg_is_instance(resolution, "resolution", (float, int))
            params.update(resolution=resolution)

        bbox = kwargs.get("bbox")
        if bbox is not None:
            assert_arg_is_instance(bbox, "bbox", (Sequence,))
            assert_arg_has_length(bbox, "bbox", 4)
            params.update(bbox=bbox)

        crs = kwargs.get("crs")
        if crs is not None:
            if isinstance(crs, str):
                crs = pyproj.CRS.from_string(crs)
            assert_arg_is_instance(crs, "crs", (pyproj.CRS,))
            params.update(crs=crs)

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

    def transform_dataset(
        self, dataset: xr.Dataset, stac_meta: dict, **params
    ) -> xr.Dataset:
        return self.assign_grid_mapping(dataset, stac_meta)

    def convert_datatree(
        self,
        datatree: xr.DataTree,
        includes: str | Iterable[str] | None = None,
        excludes: str | Iterable[str] | None = None,
        resolution: int = None,
        bbox: Sequence[float | int] | None = None,
        crs: pyproj.CRS | None = None,
        interp_methods: SpatialInterpMethods | None = None,
        agg_methods: SpatialAggMethods | None = None,
    ) -> xr.Dataset:
        # Important note: rescale_spatial_vars() may take very long
        # for some variables!
        # - "conditions_geometry_sun_angles"
        #   with shape (2, 23, 23) takes 120 seconds
        # - "conditions_geometry_viewing_incidence_angles"
        #   with shape (13, 7, 2, 23, 23) takes 140 seconds

        # rename variable names if given as common band names
        use_common_bands = False
        if includes:
            if isinstance(includes, str) and includes in COMMON_BAND_NAMES:
                use_common_bands = True
                includes = COMMON_BAND_NAMES[includes]
            elif any(name in COMMON_BAND_NAMES.keys() for name in includes):
                use_common_bands = True
                includes = [
                    COMMON_BAND_NAMES[name] if name in COMMON_BAND_NAMES else name
                    for name in includes
                ]

        if resolution is None:
            if crs is not None and crs.is_geographic:
                resolution = 10 / CONVERSION_FACTOR_DEG_METER
            else:
                resolution = 10

        name_filter = NameFilter(includes=includes, excludes=excludes)
        native_res = get_native_res(resolution, crs=crs)
        variables: dict[int, dict[Hashable, xr.DataArray]] = {10: {}, 20: {}, 60: {}}
        for group_path in GROUP_PATHS:
            group = get_data_tree_item(datatree, group_path)
            if group is None:
                continue
            for res in RESOLUTION_ORDERS[native_res]:
                res_name = f"r{res}m"
                if res_name not in group:
                    continue
                res_group = group[res_name]
                res_ds = res_group.ds
                for k, v in res_ds.data_vars.items():
                    if name_filter.accept(str(k)) and not any(
                        k in variables[sen2_res] for sen2_res in SEN2_RESOLUTIONS
                    ):
                        if use_common_bands:
                            k_mod = (
                                COMMON_BAND_NAMES_REVERSE[k]
                                if k in COMMON_BAND_NAMES_REVERSE
                                else k
                            )
                            variables[res][k_mod] = v
                        else:
                            variables[res][k] = v

        if all(len(v) == 0 for v in variables.values()):
            raise ValueError("No variables selected")
        datasets = dict()
        for res, da_mapping in variables.items():
            if da_mapping:
                ds = xr.Dataset(da_mapping)
                ds.attrs.update(self.process_metadata(datatree))
                datasets[res] = self.assign_grid_mapping(
                    ds, datatree.attrs.get("stac_discovery")
                )

        # resample dataset
        if resolution in datasets and crs is None and bbox is None:
            target_gm = GridMapping.from_dataset(datasets[resolution])
        else:
            if crs is None:
                ds = next(iter(datasets.values()))
                crs = pyproj.CRS.from_wkt(ds.spatial_ref.attrs["crs_wkt"])
            if bbox is None:
                res, ds = next(iter(datasets.items()))
                source_crs = pyproj.CRS.from_wkt(ds.spatial_ref.attrs["crs_wkt"])
                resh = res / 2
                bbox = [
                    ds.x[0] - resh,
                    ds.y[-1] - resh,
                    ds.x[-1] + resh,
                    ds.y[0] + resh,
                ]
                bbox = reproject_bbox(bbox, source_crs, crs)
            chunk_size = RESOLUTION_CHUNKSIZE[
                min(SEN2_RESOLUTIONS, key=lambda x: abs(x - resolution))
            ]
            target_gm = GridMapping.regular_from_bbox(
                bbox, resolution, crs, tile_size=chunk_size
            )

        rescaled_ds = None
        for res, ds in datasets.items():
            # if scl in ds, present as uint8
            # Note: this is a bug in CPM library. Issue reported at:
            # https://gitlab.eopf.copernicus.eu/cpm/eopf-cpm/-/issues/1044
            if "scl" in ds:
                ds["scl"] = ds["scl"].astype("uint8")
            ds = resample_in_space(
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

        return rescaled_ds

    # noinspection PyMethodMayBeStatic
    def assign_grid_mapping(self, dataset: xr.Dataset, stac_meta: dict) -> xr.Dataset:
        crs = None
        try:
            crs_code = dataset.attrs.get("horizontal_CRS_code", "EPSG:-1")
            epsg_int = int(crs_code.split(":")[1])
            crs = pyproj.CRS.from_epsg(epsg_int)
        except pyproj.exceptions.CRSError:
            pass
        if crs is None:
            try:
                for var_name, var in dataset.data_vars.items():
                    epsg_int = var.attrs.get("proj:epsg")
                    if isinstance(epsg_int, int):
                        crs = pyproj.CRS.from_epsg(epsg_int)
                        break
            except pyproj.exceptions.CRSError:
                pass
        if crs is None:
            try:
                crs_code = stac_meta.get("properties", {}).get("proj:code", "EPSG:-1")
                epsg_int = int(crs_code.split(":")[1])
                crs = pyproj.CRS.from_epsg(epsg_int)
            except pyproj.exceptions.CRSError:
                pass

        if crs:
            dataset = dataset.assign_coords(
                {"spatial_ref": xr.DataArray(0, attrs=crs.to_cf())}
            )
            for var_name, data_var in dataset.data_vars.items():
                if data_var.ndim == 2 and data_var.dims == ("y", "x"):
                    dataset[var_name].attrs["grid_mapping"] = "spatial_ref"

        return dataset

    def process_metadata(self, datatree: xr.DataTree) -> dict:
        other_metadata = datatree.attrs.get("other_metadata", {})
        return other_metadata


class MsiL1c(Msi):
    product_type = "MSIL1C"


class MsiL2a(Msi):
    product_type = "MSIL2A"


def register(registry: AnalysisModeRegistry):
    registry.register(MsiL1c)
    registry.register(MsiL2a)


def get_native_res(resolution: int | float, crs: pyproj.CRS | None = None) -> int:
    """Return the nearest equal or coarser Sentinel-2 spatial resolution.

    Args:
        resolution: Desired spatial resolution
        crs: Coordinate reference system


    Returns:
        Selected native Sentinel-2 spatial resolution
    """
    if crs is not None and crs.is_geographic:
        res_native = resolution * CONVERSION_FACTOR_DEG_METER
    else:
        res_native = resolution
    idx = np.searchsorted(SEN2_RESOLUTIONS, res_native, side="left")
    return int(
        SEN2_RESOLUTIONS[idx] if idx < len(SEN2_RESOLUTIONS) else SEN2_RESOLUTIONS[-1]
    )
