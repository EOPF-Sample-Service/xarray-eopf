#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import warnings
from abc import ABC
from collections.abc import Iterable
from typing import Any

import numpy as np
import pyproj.crs
import xarray as xr
from xcube_resampling.constants import AggMethods, InterpMethod
from xcube_resampling.gridmapping import GridMapping
from xcube_resampling.rectify import rectify_dataset

from xarray_eopf.amode import AnalysisMode, AnalysisModeRegistry
from xarray_eopf.source import get_source_path
from xarray_eopf.utils import (
    NameFilter,
    assert_arg_is_instance,
)


class Sen3(AnalysisMode, ABC):
    def is_valid_source(self, source: Any) -> bool:
        root_path = get_source_path(source)
        return (
            (
                f"S3A_{self.product_type}_" in root_path
                or f"S3B_{self.product_type}_" in root_path
            )
            if root_path
            else False
        )

    def get_applicable_params(self, **kwargs) -> dict[str, any]:
        params = {}

        resolution = kwargs.get("resolution")
        if resolution is not None:
            assert_arg_is_instance(resolution, "resolution", (int, float))
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
        resolution: float | tuple | None = None,
        interp_methods: InterpMethod | None = None,
        agg_methods: AggMethods | None = None,
    ) -> xr.Dataset:
        # filter dataset by variable names
        name_filter = NameFilter(includes=includes, excludes=excludes)
        dataset = datatree.measurements.to_dataset()
        variable_names = [k for k in dataset.data_vars if name_filter.accept(str(k))]
        if not variable_names:
            raise ValueError("No variables selected")
        dataset = dataset[variable_names]
        # remove coordinates except for latitude and longitude
        coords = []
        for coord in dataset.coords:
            if coord not in ["latitude", "longitude"]:
                coords.append(coord)
        dataset = dataset.drop_vars(coords)

        # reproject dataset to regular grid
        source_gm = GridMapping.from_dataset(dataset)
        target_gm = source_gm.to_regular()
        if resolution is not None:
            resolution: int | tuple
            if not isinstance(resolution, tuple):
                resolution = (resolution, resolution)
            bbox = target_gm.xy_bbox
            x_size = np.ceil((bbox[2] - bbox[0]) / resolution[0])
            y_size = np.ceil(abs(bbox[3] - bbox[1]) / resolution[1])
            target_gm = GridMapping.regular(
                size=(x_size, y_size),
                xy_min=(bbox[0], bbox[1]),
                xy_res=resolution,
                crs=target_gm.crs,
                tile_size=target_gm.tile_size,
            )

        rescaled_dataset = rectify_dataset(
            dataset,
            source_gm=source_gm,
            target_gm=target_gm,
            interp_methods=interp_methods,
            agg_methods=agg_methods,
        )
        rescaled_dataset.attrs = self.process_metadata(datatree)
        return rescaled_dataset

    # noinspection PyMethodMayBeStatic
    def process_metadata(self, datatree: xr.DataTree | xr.Dataset):
        # TODO: process metadata and try adhering to CF conventions
        other_metadata = datatree.attrs.get("other_metadata", {})
        return other_metadata

    # noinspection PyMethodMayBeStatic
    def assign_grid_mapping(self, dataset: xr.Dataset) -> xr.Dataset:
        crs = pyproj.CRS.from_epsg(4326)
        dataset = dataset.assign_coords(
            dict(spatial_ref=xr.DataArray(0, attrs=crs.to_cf()))
        )
        for var_name in dataset.data_vars:
            dataset[var_name].attrs["grid_mapping"] = "spatial_ref"

        return dataset


class Sen3Ol1Err(Sen3):
    product_type = "OL_1_ERR"


class Sen3Ol1Efr(Sen3):
    product_type = "OL_1_EFR"


# Broken data in: https://stac.browser.user.eopf.eodc.eu/collections/sentinel-3-olci-l2-lrr?.language=en
# class Sen3Ol2Lrr(Sen3):
#     product_type = "OL_2_LRR"


class Sen3Ol2Lfr(Sen3):
    product_type = "OL_2_LFR"


# complex data tree groups, implementation postponed;
# class Sen3Sl1Rbt(Sen3):
#     product_type = "SL_1_RBT"


class Sen3Sl2Lst(Sen3):
    product_type = "SL_2_LST"


def register(registry: AnalysisModeRegistry):
    registry.register(Sen3Ol1Err)
    registry.register(Sen3Ol1Efr)
    registry.register(Sen3Ol2Lfr)
    # registry.register(Sen3Ol2Lrr)
    # registry.register(Sen3Sl1Rbt)
    registry.register(Sen3Sl2Lst)
