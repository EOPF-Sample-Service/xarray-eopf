#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import dask.array
import xarray as xr

from xarray_eopf.util.spatial import get_spatial_vars, rescale_spatial_vars


class Sentinel2Test(TestCase):
    def test_open_datatree_sen2_l1c(self):
        path = (
            "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:"
            "sample-data/tutorial_data/cpm_v253/S2B_MSIL1C_20250113T103309_N0511_"
            "R108_T32TLQ_20250113T122458.zarr"
        )
        # noinspection PyTypeChecker
        dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native")
        self.assertEqual(25, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r60m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r60m
        self.assertEqual({"y": 1830, "x": 1830}, ds.sizes)

    def test_open_datatree_sen2_l2a(self):
        path = (
            "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:"
            "sample-data/tutorial_data/cpm_v253/S2A_MSIL2A_20240101T102431_N0510_"
            "R065_T32TNT_20240101T144052.zarr"
        )
        # noinspection PyTypeChecker
        dt = xr.open_datatree(path, engine="eopf-zarr", op_mode="native")
        self.assertEqual(36, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r60m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r60m
        self.assertEqual({"y": 1830, "x": 1830}, ds.sizes)

    def test_open_dataset_sen2_l1c(self):
        path = (
            "https://objectstore.eodc.eu:2222/e05ab01a9d56408d82ac32d69a5aae2a:sample-data/tutorial_data/"
            "cpm_v253/S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr"
        )
        # noinspection PyTypeChecker
        ds = xr.open_dataset(path, engine="eopf-zarr", op_mode="native")

        spatial_vars = get_spatial_vars(ds.data_vars)
        self.assertEqual(
            [
                "conditions_geometry_sun_angles",
                "conditions_geometry_viewing_incidence_angles",
                "conditions_mask_detector_footprint_r10m_b02",
                "conditions_mask_detector_footprint_r10m_b03",
                "conditions_mask_detector_footprint_r10m_b04",
                "conditions_mask_detector_footprint_r10m_b08",
                "conditions_mask_detector_footprint_r20m_b05",
                "conditions_mask_detector_footprint_r20m_b06",
                "conditions_mask_detector_footprint_r20m_b07",
                "conditions_mask_detector_footprint_r20m_b11",
                "conditions_mask_detector_footprint_r20m_b12",
                "conditions_mask_detector_footprint_r20m_b8a",
                "conditions_mask_detector_footprint_r60m_b01",
                "conditions_mask_detector_footprint_r60m_b09",
                "conditions_mask_detector_footprint_r60m_b10",
                "conditions_mask_l1c_classification_b00",
                "measurements_r10m_b02",
                "measurements_r10m_b03",
                "measurements_r10m_b04",
                "measurements_r10m_b08",
                "measurements_r20m_b05",
                "measurements_r20m_b06",
                "measurements_r20m_b07",
                "measurements_r20m_b11",
                "measurements_r20m_b12",
                "measurements_r20m_b8a",
                "measurements_r60m_b01",
                "measurements_r60m_b09",
                "measurements_r60m_b10",
                "quality_l1c_quicklook_tci",
                "quality_mask_r10m_b02",
                "quality_mask_r10m_b03",
                "quality_mask_r10m_b04",
                "quality_mask_r10m_b08",
                "quality_mask_r20m_b05",
                "quality_mask_r20m_b06",
                "quality_mask_r20m_b07",
                "quality_mask_r20m_b11",
                "quality_mask_r20m_b12",
                "quality_mask_r20m_b8a",
                "quality_mask_r60m_b01",
                "quality_mask_r60m_b09",
                "quality_mask_r60m_b10",
            ],
            sorted(spatial_vars.keys()),
        )
        none_dask_arrays = {
            k: v
            for k, v in spatial_vars.items()
            if not isinstance(v.data, dask.array.Array)
        }
        self.assertEqual(
            0,
            len(none_dask_arrays),
            msg=(
                f"{len(none_dask_arrays)} spatial variable(s) are not using dask arrays:\n"
                + "\n".join(f"- {k} ({v.shape})" for k, v in none_dask_arrays.items())
            ),
        )
        # rescaled_spatial_vars = rescale_spatial_vars(spatial_vars)
        # for var_name, var in rescaled_spatial_vars.items():
        #    self.assertEqual((10980, 10980), var.shape[-2:])
