#  Copyright (c) 2025 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

from unittest import TestCase

import xarray as xr

from integration.helpers import assert_data_arrays_are_chunked
from xarray_eopf.constants import DEFAULT_ENDPOINT_URL
from xarray_eopf.spatial import get_spatial_vars, rescale_spatial_vars
from xarray_eopf.utils import timeit

bucket = "e05ab01a9d56408d82ac32d69a5aae2a:sample-data"
path_prefix = "tutorial_data/cpm_v253"
s3_prefix = f"s3://{bucket}/{path_prefix}"
https_prefix = f"https://{DEFAULT_ENDPOINT_URL}/{bucket}/{path_prefix}"

allowed_open_time = 5  # seconds


class Sentinel2NativeTest(TestCase):
    def test_open_datatree_sen2_l1c_s3(self):
        self._test_open_datatree_sen2_l1c(s3_prefix)

    def test_open_datatree_sen2_l1c_https(self):
        self._test_open_datatree_sen2_l1c(https_prefix)

    def _test_open_datatree_sen2_l1c(self, url_prefix: str):
        # noinspection PyTypeChecker
        url = (
            f"{url_prefix}/"
            f"S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr"
        )
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            dt = xr.open_datatree(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time < allowed_open_time)
        self.assertEqual(25, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r10m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r10m.ds
        self.assertEqual({"y": 10980, "x": 10980}, ds.sizes)
        spatial_vars = get_spatial_vars(ds)
        self.assertEqual(
            ["b02", "b03", "b04", "b08"], sorted(map(str, spatial_vars.keys()))
        )
        assert_data_arrays_are_chunked(self, spatial_vars)

    def test_open_datatree_sen2_l2a_s3(self):
        self._test_open_datatree_sen2_l2a(s3_prefix)

    def test_open_datatree_sen2_l2a_https(self):
        self._test_open_datatree_sen2_l2a(https_prefix)

    def _test_open_datatree_sen2_l2a(self, url_prefix: str):
        url = (
            f"{url_prefix}/"
            "S2A_MSIL2A_20240101T102431_N0510_R065_T32TNT_20240101T144052.zarr"
        )
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            dt = xr.open_datatree(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time < allowed_open_time)
        self.assertEqual(36, len(dt.groups))
        self.assertIn(
            "/measurements/reflectance/r10m",
            dt.groups,
        )
        ds = dt.measurements.reflectance.r10m.ds
        self.assertEqual({"y": 10980, "x": 10980}, ds.sizes)
        spatial_vars = get_spatial_vars(ds)
        self.assertEqual(
            ["b02", "b03", "b04", "b08"], sorted(map(str, spatial_vars.keys()))
        )
        assert_data_arrays_are_chunked(self, spatial_vars)

    def test_open_dataset_sen2_l1c_s3(self):
        self._test_open_dataset_sen2_l1c(s3_prefix)

    def test_open_dataset_sen2_l1c_https(self):
        self._test_open_dataset_sen2_l1c(https_prefix)

    def _test_open_dataset_sen2_l1c(self, url_prefix: str):
        url = (
            f"{url_prefix}/"
            "S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr"
        )
        with timeit(url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time < allowed_open_time)
        self.assertEqual(62, len(ds.data_vars))
        self.assertIn(
            "measurements_r10m_b02",
            ds.data_vars,
        )
        da = ds.measurements_r10m_b02
        self.assertEqual(
            {"measurements_r10m_y": 10980, "measurements_r10m_x": 10980}, da.sizes
        )
        spatial_vars = get_spatial_vars(ds)
        self.assertEqual(43, len(spatial_vars))
        assert_data_arrays_are_chunked(self, spatial_vars)


class Sentinel2AnalysisTest(TestCase):
    def test_open_dataset_sen2_l1c_s3(self):
        self._test_open_dataset_sen2_l1c(s3_prefix)

    def test_open_dataset_sen2_l1c_https(self):
        self._test_open_dataset_sen2_l1c(https_prefix)

    def _test_open_dataset_sen2_l1c(self, url_prefix):
        url = (
            f"{url_prefix}/"
            "S2B_MSIL1C_20250113T103309_N0511_R108_T32TLQ_20250113T122458.zarr"
        )
        with timeit("open " + url) as result:
            # noinspection PyTypeChecker
            ds = xr.open_dataset(url, engine="eopf-zarr", op_mode="native", chunks={})
        self.assertTrue(result.time < allowed_open_time)

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
            sorted(map(str, spatial_vars.keys())),
        )

        for k, v in spatial_vars.items():
            print(f"{k}: s={v.shape}, c={v.chunks}, data={type(v.data)}")

        assert_data_arrays_are_chunked(self, spatial_vars)

        # rescaling of its shape (2, 23, 23) takes 120 seconds!
        del spatial_vars["conditions_geometry_sun_angles"]
        # rescaling of its shape (13, 7, 2, 23, 23) takes 140 seconds!
        del spatial_vars["conditions_geometry_viewing_incidence_angles"]

        with timeit("rescale_spatial_vars"):
            rescaled_spatial_vars = rescale_spatial_vars(spatial_vars)

        for var_name, var in rescaled_spatial_vars.items():
            self.assertEqual((10980, 10980), var.shape[-2:])
