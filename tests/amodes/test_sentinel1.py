#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import os
import uuid
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import dask.array as da
import numpy as np
import pyproj
import pytest
import xarray as xr
from xcube_resampling.gridmapping import GridMapping

from tests.helpers import (
    make_s1_grd_datatree,
    make_s1_ocn_datatree,
    make_s1_slc_datatree,
)
from xarray_eopf.amode import AnalysisModeRegistry
from xarray_eopf.amodes import sentinel1 as sen1
from xarray_eopf.amodes.sentinel1 import Sen1GRD, Sen1OCN, Sen1SLC, register


class Sentinel1AnalysisModeTest(TestCase):
    def test_register(self):
        registry = AnalysisModeRegistry()
        register(registry)
        self.assertEqual(3, len(list(registry.keys())))
        self.assertIn(Sen1GRD.product_type, registry.keys())
        self.assertIn(Sen1SLC.product_type, registry.keys())
        self.assertIn(Sen1OCN.product_type, registry.keys())


# noinspection PyUnresolvedReferences
class Sen1TestMixin:
    def test_process_metadata(self: TestCase):
        self.assertEqual({}, self.mode.process_metadata(xr.DataTree()))
        dt = xr.DataTree()
        dt.attrs["other_metadata"] = {"test_key": "test_val"}
        self.assertEqual(
            {"other_metadata": {"test_key": "test_val"}}, self.mode.process_metadata(dt)
        )

    def test_transform_datatree(self: TestCase):
        dt = xr.DataTree()
        with self.assertWarns(UserWarning) as cm:
            out = self.mode.transform_datatree(dt)
        self.assertIs(out, dt)
        self.assertIn("Analysis mode not implemented", str(cm.warning))

    def test_transform_dataset(self: TestCase):
        ds = xr.Dataset({"a": xr.DataArray([1, 2], dims=("x",))})
        out = self.mode.transform_dataset(ds, stac_meta={"k": "v"})
        self.assertIs(out, ds)


class Sen1GRDTest(Sen1TestMixin, TestCase):

    def setUp(self):
        self.mode = Sen1GRD()
        self.dem = xr.DataArray(
            np.ones((2, 2), dtype="float32"),
            dims=("lat", "lon"),
            coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        self.dt = make_s1_grd_datatree()
        self.expected_vv = xr.Dataset(
            {"vv": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))}
        )
        self.expected_beta0_vv = xr.Dataset(
            {
                "beta0_vv": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
                )
            },
        )
        self.src_loc = xr.Dataset(
            {
                "azimuth_time": (
                    ("lat", "lon"),
                    np.zeros((2, 2), dtype="datetime64[ns]"),
                ),
                "ground_range": (("lat", "lon"), np.zeros((2, 2))),
                "gamma_area": (("lat", "lon"), np.ones((2, 2))),
            }
        )

    def test_is_valid_source_ok(self):
        self.assertTrue(self.mode.is_valid_source("data/S1A_IW_GRDH_20240201.zarr"))
        self.assertTrue(self.mode.is_valid_source("S1D_SM_GRDH_TEST"))

    def test_is_not_valid_source(self):
        self.assertFalse(self.mode.is_valid_source("data/S1A_IW_SLC_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_get_grid_parameters(self):
        params = sen1._get_grid_parameters(self.dt, (2.0, 3.0))

        self.assertEqual(0.0, params["range0"])
        self.assertEqual(10.0, params["spacing_range"])
        self.assertEqual(20.0, params["spacing_az"])
        self.assertEqual(10.0, params["d_range"])
        self.assertEqual(30.0, params["d_range_scale"])
        self.assertEqual(np.datetime64("2024-01-01T00:00:00"), params["az0"])
        self.assertEqual(0.5, params["d_az"])
        self.assertEqual(30.0, params["spacing_range_scale"])
        self.assertEqual(40.0, params["spacing_az_scale"])
        self.assertEqual(
            np.datetime64("2024-01-01T00:00:00.250000000"), params["az0_scale"]
        )

    def test_get_applicable_params(self: TestCase):

        self.assertEqual({}, self.mode.get_applicable_params())
        self.assertEqual(
            {
                "resolution": 10,
                "bbox": [1, 3, 4, 5],
                "crs": pyproj.CRS.from_string("EPSG:4326"),
                "dem": self.dem,
                "interp_methods": "nearest",
                "footprint_scale_factor": (2.0, 3.0),
                "apply_rtc": False,
                "cache_uri": "file:///tmp/cache",
            },
            self.mode.get_applicable_params(
                resolution=10,
                bbox=[1, 3, 4, 5],
                crs="EPSG:4326",
                dem=self.dem,
                interp_methods="nearest",
                footprint_scale_factor=(2.0, 3.0),
                apply_rtc=False,
                cache_uri="file:///tmp/cache",
            ),
        )
        with pytest.raises(TypeError, match="interp_methods"):
            self.mode.get_applicable_params(interp_methods="cubic")
        with pytest.raises(TypeError, match="footprint_scale_factor"):
            self.mode.get_applicable_params(footprint_scale_factor=(1.0, "x"))
        with pytest.raises(TypeError, match="apply_rtc"):
            self.mode.get_applicable_params(apply_rtc="yes")
        with pytest.raises(TypeError, match="cache_uri"):
            self.mode.get_applicable_params(cache_uri=123)

    def test_convert_datatree(self):
        with patch.object(
            self.mode, "_terrain_correct", return_value=self.expected_vv
        ) as mocked:
            out = self.mode.convert_datatree(self.dt, includes=["vv"], dem=self.dem)

        self.assertIs(out, self.expected_vv)
        args, kwargs = mocked.call_args
        self.assertIs(args[0], self.dt)
        self.assertEqual(["beta0_vv"], list(args[1].data_vars))
        self.assertIs(args[2], self.dem)
        self.assertEqual("bilinear", kwargs["interp_method"])
        self.assertTrue(kwargs["apply_rtc"])
        self.mode._cleanup()

    def test_convert_datatree_updates_footprint_scale_factor(self):
        with patch.object(self.mode, "_terrain_correct", return_value=self.expected_vv):
            self.mode.convert_datatree(
                self.dt,
                includes=["vv"],
                dem=self.dem,
                footprint_scale_factor=(2.0, 4.0),
            )
        self.assertEqual((2.0, 4.0), self.mode.footprint_scale_factor)
        self.mode._cleanup()

    def test_convert_datatree_with_cache_uri_uses_fs(self):
        fs = SimpleNamespace()
        with (
            patch.object(
                sen1.fsspec, "url_to_fs", return_value=(fs, "/cache")
            ) as url_to_fs,
            patch.object(self.mode, "_terrain_correct", return_value=self.expected_vv),
        ):
            _ = self.mode.convert_datatree(
                self.dt, includes=["vv"], dem=self.dem, cache_uri="file:///cache/"
            )
        url_to_fs.assert_called_once_with("file:///cache")
        self.assertEqual("file:///cache", self.mode.cache_uri)
        self.mode._cleanup()

    def test_convert_datatree_uses_get_dem(self):

        with patch.object(sen1, "get_dem", return_value=self.dem) as get_dem_mock:
            with patch.object(
                self.mode, "_terrain_correct", return_value=self.expected_beta0_vv
            ):
                _ = self.mode.convert_datatree(self.dt, includes=["vv"])

        get_dem_mock.assert_called_once()
        args, _ = get_dem_mock.call_args
        self.assertEqual(self.dt.attrs["stac_discovery"]["bbox"], args[0])
        self.mode._cleanup()

    def test_convert_datatree_fail(self):
        with pytest.raises(ValueError, match="No valid variable names"):
            self.mode.convert_datatree(self.dt, includes="bibo", dem=self.dem)
        self.mode._cleanup()

    def test_convert_datatree_cleans_up_on_failure(self):
        with (
            patch.object(
                self.mode, "_terrain_correct", side_effect=RuntimeError("boom")
            ),
            patch.object(self.mode, "_cleanup") as cleanup,
        ):
            with pytest.raises(RuntimeError, match="boom"):
                self.mode.convert_datatree(self.dt, includes=["vv"], dem=self.dem)
        cleanup.assert_called_once()
        self.mode._cleanup()

    def test_convert_datatree_warns_when_processing_full_product(self):
        with (
            self.assertWarns(UserWarning) as cm,
            patch.object(sen1, "get_dem", return_value=self.dem),
            patch.object(self.mode, "_terrain_correct", return_value=self.expected_vv),
        ):
            self.mode.convert_datatree(self.dt, includes=["vv"])
        self.assertIn("No bounding box specified", str(cm.warning))
        self.mode._cleanup()

    def test_terrain_correct_with_rtc_nearest(self):
        with (
            patch.object(sen1, "get_source_location", return_value=self.src_loc),
            patch.object(sen1, "geocode_data", return_value=self.expected_beta0_vv),
            patch.object(
                sen1,
                "apply_gamma_weights",
                return_value=xr.ones_like(self.src_loc.gamma_area),
            ) as gamma_mock,
            patch.object(sen1, "assign_grid_mapping", side_effect=lambda ds: ds),
            patch.object(xr.Dataset, "to_zarr", return_value=None),
            patch.object(sen1.xr, "open_zarr", return_value=self.src_loc),
        ):
            self.mode.cache_uri = f"tmp_{uuid.uuid4().hex}"
            out = self.mode._terrain_correct(
                self.dt,
                self.expected_beta0_vv,
                self.dem,
                apply_rtc=True,
                interp_method="nearest",
            )
        gamma_mock.assert_called_once()
        self.assertIn("gamma0_vv", out.data_vars)
        self.mode._cleanup()

    def test_terrain_correct_with_rtc_bilinear(self):
        with (
            patch.object(sen1, "get_source_location", return_value=self.src_loc),
            patch.object(sen1, "geocode_data", return_value=self.expected_beta0_vv),
            patch.object(
                sen1,
                "apply_gamma_weights",
                return_value=xr.ones_like(self.src_loc.gamma_area),
            ) as gamma_mock,
            patch.object(sen1, "assign_grid_mapping", side_effect=lambda ds: ds),
            patch.object(xr.Dataset, "to_zarr", return_value=None),
            patch.object(sen1.xr, "open_zarr", return_value=self.src_loc),
        ):
            self.mode.cache_uri = f"tmp_{uuid.uuid4().hex}"
            out = self.mode._terrain_correct(
                self.dt,
                self.expected_beta0_vv,
                self.dem,
                apply_rtc=True,
                interp_method="bilinear",
            )
        gamma_mock.assert_called_once()
        self.assertIn("gamma0_vv", out.data_vars)
        self.mode._cleanup()

    def test_terrain_correct_without_rtc(self):
        with (
            patch.object(sen1, "get_source_location", return_value=self.src_loc),
            patch.object(sen1, "geocode_data", return_value=self.expected_beta0_vv),
            patch.object(sen1, "apply_gamma_weights") as gamma_mock,
            patch.object(sen1, "assign_grid_mapping", side_effect=lambda ds: ds),
            patch.object(xr.Dataset, "to_zarr", return_value=None),
            patch.object(sen1.xr, "open_zarr", return_value=self.src_loc),
        ):
            self.mode.cache_uri = f"tmp_{uuid.uuid4().hex}"
            out = self.mode._terrain_correct(
                self.dt,
                self.expected_beta0_vv,
                self.dem,
                apply_rtc=False,
            )
        gamma_mock.assert_not_called()
        self.assertIn("beta0_vv", out.data_vars)
        self.mode._cleanup()

    def test_cleanup_removes_cache(self):
        with patch.object(sen1.fsspec, "url_to_fs") as url_to_fs:
            self.mode.cache_uri = "file:///tmp/fake-cache"
            fs = SimpleNamespace(
                exists=lambda path: True, rm=lambda path, recursive: None
            )
            url_to_fs.return_value = (fs, "/tmp/fake-cache")
            self.mode._cleanup()
            url_to_fs.assert_called_once_with("file:///tmp/fake-cache")

    def test_cleanup_without_cache_uri_is_noop(self):
        self.mode.cache_uri = None
        self.mode._cleanup()


class Sen1SLCTest(Sen1TestMixin, TestCase):

    def setUp(self):
        self.mode = Sen1SLC()
        self.dt = make_s1_slc_datatree()
        self.dem = xr.DataArray(
            np.ones((2, 2), dtype="float32"),
            dims=("lat", "lon"),
            coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        self.expected_beta0 = xr.Dataset(
            {
                "beta0_vv": xr.DataArray(
                    np.ones((2, 2)),
                    dims=("lat", "lon"),
                    coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
                )
            }
        )
        self.src_loc = xr.Dataset(
            {
                "azimuth_time": (
                    ("lat", "lon"),
                    np.zeros((2, 2), dtype="datetime64[ns]"),
                ),
                "slant_range_time": (("lat", "lon"), np.zeros((2, 2))),
                "gamma_area": (("lat", "lon"), np.ones((2, 2))),
            }
        )

    @staticmethod
    def _make_slc_dataset(
        azimuth_time: np.ndarray, slant_range_time: np.ndarray
    ) -> xr.Dataset:
        return xr.Dataset(
            {
                "beta0_vv": xr.DataArray(
                    np.ones((len(azimuth_time), len(slant_range_time))),
                    dims=("azimuth_time", "slant_range_time"),
                    coords={
                        "azimuth_time": azimuth_time,
                        "slant_range_time": slant_range_time,
                    },
                )
            }
        )

    @staticmethod
    def _as_object_array(*datasets: xr.Dataset) -> np.ndarray:
        out = np.empty(len(datasets), dtype=object)
        for idx, dataset in enumerate(datasets):
            out[idx] = dataset
        return out

    def test_is_valid_source_ok(self):
        self.assertTrue(self.mode.is_valid_source("data/S1A_IW_SLC_20240201.zarr"))
        self.assertTrue(self.mode.is_valid_source("S1D_SM_SLC_TEST"))

    def test_is_not_valid_source(self):
        self.assertFalse(self.mode.is_valid_source("data/S1A_IW_GRDH_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_get_groups(self):
        groups = self.mode._get_groups(self.dt)
        self.assertEqual(("mode", "swath", "burst"), groups.dims)
        self.assertEqual((2, 2, 1), groups.shape)
        self.assertEqual("S1A_IW_SLC_TEST_VV_IW1_0", groups.sel(mode="VV").item(0))

    def test_get_grid_parameters(self):
        params = sen1._get_grid_parameters(
            self.dt, (2.0, 3.0), range_coord="slant_range_time"
        )
        self.assertEqual(0.0, params["range0"])
        self.assertEqual(1.0, params["d_range"])
        self.assertEqual(10.0, params["spacing_range"])
        self.assertEqual(3.0, params["d_range_scale"])
        self.assertEqual(30.0, params["spacing_range_scale"])
        self.assertEqual(1.0, params["range0_scale"])

    def test_calibrate_burst_and_extract_valid_region(self):
        burst = self.dt["S1A_IW_SLC_TEST_VV_IW1_0"]
        beta0 = self.mode._calibrate_burst(burst)
        trimmed = self.mode._extract_valid_region(beta0, burst)
        self.assertEqual(("azimuth_time", "slant_range_time"), trimmed.slc.dims)
        self.assertEqual(2, trimmed.sizes["azimuth_time"])
        self.assertEqual(2, trimmed.sizes["slant_range_time"])

    def test_open_data(self):
        out = self.mode._open_data(self.dt, includes=["vv", "vh"])
        self.assertCountEqual(["beta0_vv", "beta0_vh"], out.data_vars)
        self.assertNotIn("line", out.coords)
        self.assertNotIn("pixel", out.coords)
        self.assertEqual({"azimuth_time": 2, "slant_range_time": 3}, out.sizes)

    def test_open_data_accepts_string_include(self):
        out = self.mode._open_data(self.dt, includes="vv")
        self.assertEqual(["beta0_vv"], list(out.data_vars))

    def test_open_data_fails_when_no_variables_match(self):
        with pytest.raises(ValueError, match="No valid variable names found in dataset"):
            self.mode._open_data(self.dt, includes="does_not_exist")

    def test_convert_datatree(self):
        with patch.object(
            self.mode, "_terrain_correct", return_value=self.expected_beta0
        ) as mocked:
            out = self.mode.convert_datatree(self.dt, includes=["vv"], dem=self.dem)
        self.assertIs(out, self.expected_beta0)
        args, kwargs = mocked.call_args
        self.assertIs(args[0], self.dt)
        self.assertEqual(["beta0_vv"], list(args[1].data_vars))
        self.assertIs(args[2], self.dem)
        self.assertEqual("bilinear", kwargs["interp_method"])
        self.mode._cleanup()

    def test_terrain_correct_uses_slant_range_path(self):
        with (
            patch.object(sen1, "get_source_location", return_value=self.src_loc) as src_mock,
            patch.object(sen1, "geocode_data", return_value=self.expected_beta0),
            patch.object(
                sen1,
                "apply_gamma_weights",
                return_value=xr.ones_like(self.src_loc.gamma_area),
            ) as gamma_mock,
            patch.object(sen1, "assign_grid_mapping", side_effect=lambda ds: ds),
            patch.object(xr.Dataset, "to_zarr", return_value=None),
            patch.object(sen1.xr, "open_zarr", return_value=self.src_loc),
        ):
            self.mode.cache_uri = f"tmp_{uuid.uuid4().hex}"
            out = self.mode._terrain_correct(
                self.dt,
                self.expected_beta0,
                self.dem,
                apply_rtc=True,
                interp_method="nearest",
            )
        self.assertIn("gamma0_vv", out.data_vars)
        self.assertIsNone(src_mock.call_args.kwargs["time_slr_gcp"])
        self.assertEqual("slant_range_time", src_mock.call_args.kwargs["range_coord"])
        self.assertEqual("slant_range_time", gamma_mock.call_args.kwargs["range_coord"])
        self.mode._cleanup()

    def test_merge_bursts_warns_on_irregular_spacing(self):
        ds0 = self._make_slc_dataset(
            np.array(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:01",
                    "2024-01-01T00:00:02",
                ],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        ds1 = self._make_slc_dataset(
            np.array(
                [
                    "2024-01-01T00:00:01.800000000",
                    "2024-01-01T00:00:03.800000000",
                    "2024-01-01T00:00:05.800000000",
                ],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        with pytest.warns(UserWarning, match="Azimuth time spacing is not regular"):
            out = self.mode._merge_bursts(self._as_object_array(ds0, ds1))
        self.assertEqual(5, out.sizes["azimuth_time"])

    def test_merge_bursts_raises_without_overlap(self):
        ds0 = self._make_slc_dataset(
            np.array(
                ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        ds1 = self._make_slc_dataset(
            np.array(
                ["2024-01-01T00:00:10", "2024-01-01T00:00:11"],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        with pytest.raises(ValueError, match="No overlap found"):
            self.mode._merge_bursts(self._as_object_array(ds0, ds1))

    def test_align_azimuth_warns_on_irregular_spacing(self):
        ds0 = self._make_slc_dataset(
            np.array(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:01",
                    "2024-01-01T00:00:02",
                    "2024-01-01T00:00:03",
                ],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        ds1 = self._make_slc_dataset(
            np.array(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:01.100000000",
                    "2024-01-01T00:00:02.100000000",
                    "2024-01-01T00:00:03.100000000",
                ],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        with pytest.warns(UserWarning, match="Azimuth spacing is irregular"):
            out = self.mode._align_azimuth(self._as_object_array(ds0, ds1))
        self.assertEqual(4, out[0].sizes["azimuth_time"])
        self.assertTrue(
            np.array_equal(out[0].azimuth_time.values, out[1].azimuth_time.values)
        )

    def test_align_azimuth_warns_on_size_difference(self):
        ds0 = self._make_slc_dataset(
            np.array(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:01",
                    "2024-01-01T00:00:02",
                    "2024-01-01T00:00:03",
                    "2024-01-01T00:00:04",
                ],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        ds1 = self._make_slc_dataset(
            np.array(
                [
                    "2024-01-01T00:00:01",
                    "2024-01-01T00:00:02",
                    "2024-01-01T00:00:03.300000000",
                    "2024-01-01T00:00:04.300000000",
                ],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        with pytest.warns(UserWarning, match="Aligned swaths have different azimuth sizes"):
            out = self.mode._align_azimuth(self._as_object_array(ds0, ds1))
        self.assertEqual(3, out[0].sizes["azimuth_time"])

    def test_merge_swaths_warns_on_irregular_spacing(self):
        ds0 = self._make_slc_dataset(
            np.array(
                ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0, 2.0]),
        )
        ds1 = self._make_slc_dataset(
            np.array(
                ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                dtype="datetime64[ns]",
            ),
            np.array([1.8, 3.8, 5.8]),
        )
        with pytest.warns(UserWarning, match="Slant range spacing is irregular"):
            out = self.mode._merge_swaths([ds0, ds1])
        self.assertEqual(5, out.sizes["slant_range_time"])

    def test_merge_swaths_raises_without_overlap(self):
        ds0 = self._make_slc_dataset(
            np.array(
                ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                dtype="datetime64[ns]",
            ),
            np.array([0.0, 1.0]),
        )
        ds1 = self._make_slc_dataset(
            np.array(
                ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                dtype="datetime64[ns]",
            ),
            np.array([10.0, 11.0]),
        )
        with pytest.raises(ValueError, match="No overlap found"):
            self.mode._merge_swaths([ds0, ds1])


class Sen1OCNTest(Sen1TestMixin, TestCase):
    mode = Sen1OCN()
    dt = make_s1_ocn_datatree()

    def test_is_valid_source_ok(self):
        self.assertTrue(self.mode.is_valid_source("data/S1A_IW_OCN_20240201.zarr"))
        self.assertTrue(self.mode.is_valid_source("S1A_IW_OCN_TEST"))

    def test_is_not_valid_source(self):
        self.assertFalse(self.mode.is_valid_source("data/S1A_IW_SLC_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_get_applicable_params(self: TestCase):
        self.assertEqual({}, self.mode.get_applicable_params())
        self.assertEqual(
            {
                "resolution": 1,
                "bbox": [1, 3, 4, 5],
                "crs": pyproj.CRS.from_string("EPSG:4326"),
                "interp_methods": "nearest",
                "agg_methods": "nearest",
            },
            self.mode.get_applicable_params(
                resolution=1,
                bbox=[1, 3, 4, 5],
                crs="EPSG:4326",
                interp_methods="nearest",
                agg_methods="nearest",
            ),
        )
        with pytest.raises(TypeError):
            self.mode.get_applicable_params(interp_methods="cubic")

    def test_convert_datatree(self):
        # with bbox and resolution
        out = self.mode.convert_datatree(
            self.dt,
            includes=["wind_direction", "wind_speed"],
            resolution=1,
            bbox=[-1, 1, 2, 4],
        )
        self.assertCountEqual(["wind_direction", "wind_speed"], out.keys())
        self.assertEqual({"lat": 3, "lon": 3}, out.sizes)
        self.assertListEqual([-0.5, 0.5, 1.5], out.lon.values.tolist())
        self.assertListEqual([3.5, 2.5, 1.5], out.lat.values.tolist())
        self.assertTrue(np.all(out.wind_direction.values == 1))
        self.assertTrue(np.all(out.wind_speed.values == 1))

        # without bbox and resolution
        out = self.mode.convert_datatree(self.dt)
        self.assertCountEqual(
            [
                "wind_direction",
                "wind_speed",
                "inversion_quality",
                "wind_quality",
                "percentage_bright_points",
            ],
            out.keys(),
        )
        self.assertEqual({"lat": 7, "lon": 7}, out.sizes)
        self.assertListEqual(
            [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0], out.lon.values.tolist()
        )
        self.assertListEqual(
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0], out.lat.values.tolist()
        )

        # projected crs
        out = self.mode.convert_datatree(
            self.dt, crs=pyproj.CRS.from_string("EPSG:32631")
        )
        self.assertCountEqual(
            [
                "wind_direction",
                "wind_speed",
                "inversion_quality",
                "wind_quality",
                "percentage_bright_points",
            ],
            out.keys(),
        )
        self.assertEqual({"y": 8, "x": 8}, out.sizes)

    def test_convert_datatree_fail(self):
        with pytest.raises(ValueError, match="No valid variable names"):
            self.mode.convert_datatree(self.dt, includes="invalid_var")


class Sentinel1FunctionsTest(TestCase):

    def setUp(self):
        self.gm_dem_params = {
            "crs": "EPSG:4326",
            "xy_var_names": ("lat", "lon"),
        }
        self.grid_params = sen1.GridParams(
            range0=0.0,
            range0_scale=0.0,
            d_range=1.0,
            d_range_scale=1.0,
            spacing_range=1.0,
            spacing_range_scale=1.0,
            az0=np.datetime64("2024-01-01T00:00:00"),
            az0_scale=np.datetime64("2024-01-01T00:00:00"),
            d_az=1.0,
            d_az_scale=1.0,
            spacing_az=2.0,
            spacing_az_scale=2.0,
        )
        self.dem = xr.DataArray(
            np.ones((2, 2), dtype="float64"),
            dims=("lat", "lon"),
            coords={
                "lat": [0.0, 0.1],
                "lon": [0.0, 0.1],
                "spatial_ref": xr.DataArray(
                    0, attrs=pyproj.CRS.from_epsg(4326).to_cf()
                ),
            },
        )
        self.dem_ecef = xr.DataArray(
            np.ones((3, 2, 2), dtype="float64"),
            dims=("axis", "lat", "lon"),
            coords={"axis": ["x", "y", "z"], "lat": [0.0, 0.1], "lon": [0.0, 0.1]},
        )
        self.posvel_coeff = xr.DataArray(
            np.zeros((2, 3)),
            dims=("degree", "axis"),
            coords={"degree": [1, 0], "axis": ["x", "y", "z"]},
            attrs={"epoch": np.datetime64("2024-01-01T00:00:00")},
        )
        self.gr_coeff = xr.DataArray(
            np.zeros((2, 9)),
            dims=("azimuth_time", "degree"),
            coords={
                "degree": np.arange(8, -1, -1),
                "azimuth_time": np.array(
                    ["2024-01-01T00:00:00", "2024-01-01T00:00:02"],
                    dtype="datetime64[ns]",
                ),
            },
            attrs=dict(mean=1, std=1),
        )
        self.time_slr = xr.DataArray(
            np.ones((2, 2), dtype="float64"),
            dims=("azimuth_time", "ground_range"),
            coords={"azimuth_time": [0, 1], "ground_range": [0, 1]},
        )
        self.sat_position = xr.DataArray(
            np.ones((2, 3), dtype="float64"),
            dims=("azimuth_time", "axis"),
            coords={"azimuth_time": [0, 1], "axis": ["x", "y", "z"]},
        )

    def test_gridparams_iter(self):
        self.assertEqual(
            [
                "range0",
                "range0_scale",
                "d_range",
                "d_range_scale",
                "spacing_range",
                "spacing_range_scale",
                "az0",
                "az0_scale",
                "d_az",
                "d_az_scale",
                "spacing_az",
                "spacing_az_scale",
            ],
            list(iter(self.grid_params)),
        )

    def test_gridparams_contains(self):
        self.assertIn("range0", self.grid_params)
        self.assertNotIn("gr0", self.grid_params)

    def test_get_dem_requires_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Missing AWS credentials"):
                sen1.get_dem([0, 50, 1, 51])

    def test_get_dem_resolution_required_if_crs_given(self):
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"},
            clear=True,
        ):
            with (
                patch.object(sen1.pystac_client.Client, "open") as client_open,
                patch.object(sen1.rioxarray, "open_rasterio") as open_rasterio,
            ):
                fake_item = SimpleNamespace(assets={"data": SimpleNamespace(href="x")})
                search = SimpleNamespace(items=lambda: [fake_item])
                client_open.return_value = SimpleNamespace(search=lambda **_: search)
                open_rasterio.return_value = xr.DataArray(
                    np.ones((1, 4, 4), dtype="float32"),
                    dims=("band", "y", "x"),
                    coords={"band": [1], "y": [3, 2, 1, 0], "x": [0, 1, 2, 3]},
                )

                with pytest.raises(
                    ValueError, match="Resolution must be provided if CRS is not None"
                ):
                    sen1.get_dem([0, 0, 1, 1], crs=pyproj.CRS.from_epsg(32632))

    def test_get_dem_reprojects_bbox_and_resamples(self):
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"},
            clear=True,
        ):
            with (
                patch.object(sen1.pystac_client.Client, "open") as client_open,
                patch.object(sen1.rioxarray, "open_rasterio") as open_rasterio,
            ):
                fake_item = SimpleNamespace(assets={"data": SimpleNamespace(href="x")})
                search = SimpleNamespace(items=lambda: [fake_item])
                client_open.return_value = SimpleNamespace(search=lambda **_: search)
                open_rasterio.return_value = xr.DataArray(
                    np.ones((1, 4, 4), dtype="float32"),
                    dims=("band", "y", "x"),
                    coords={"band": [1], "y": [3, 2, 1, 0], "x": [0, 1, 2, 3]},
                )

                out = sen1.get_dem(
                    [0, 0, 900, 900], resolution=30.0, crs=pyproj.CRS.from_epsg(32632)
                )
                self.assertIsInstance(out, xr.DataArray)
                self.assertEqual((30, 30), out.values.shape)

    def test_get_dem_bbox_passthrough_and_crop_branch(self):
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"},
            clear=True,
        ):
            with (
                patch.object(sen1.pystac_client.Client, "open") as client_open,
                patch.object(sen1.rioxarray, "open_rasterio") as open_rasterio,
            ):
                fake_item = SimpleNamespace(assets={"data": SimpleNamespace(href="x")})
                search = SimpleNamespace(items=lambda: [fake_item])
                client_open.return_value = SimpleNamespace(search=lambda **_: search)
                open_rasterio.return_value = xr.DataArray(
                    np.ones((1, 4, 4), dtype="float32"),
                    dims=("band", "y", "x"),
                    coords={"band": [1], "y": [3, 2, 1, 0], "x": [0, 1, 2, 3]},
                ).chunk(dict(y=2, x=2))

                out = sen1.get_dem([0, 0, 2, 2])
                self.assertIn("lat", out.dims)
                self.assertIn("lon", out.dims)
                self.assertEqual(3, out.sizes["lat"])
                self.assertEqual(3, out.sizes["lon"])

    def test_get_dem_resolution_with_no_crs_sets_wgs84(self):
        crs = pyproj.CRS.from_epsg(4326)
        with patch.dict(
            os.environ,
            {"AWS_ACCESS_KEY_ID": "k", "AWS_SECRET_ACCESS_KEY": "s"},
            clear=True,
        ):
            with (
                patch.object(sen1.pystac_client.Client, "open") as client_open,
                patch.object(sen1.rioxarray, "open_rasterio") as open_rasterio,
            ):
                fake_item = SimpleNamespace(assets={"data": SimpleNamespace(href="x")})
                search = SimpleNamespace(items=lambda: [fake_item])
                client_open.return_value = SimpleNamespace(search=lambda **_: search)

                open_rasterio.return_value = xr.DataArray(
                    np.ones((1, 4, 4), dtype="float32"),
                    dims=("band", "y", "x"),
                    coords={
                        "band": [1],
                        "y": [3, 2, 1, 0],
                        "x": [0, 1, 2, 3],
                        "spatial_ref": xr.DataArray(0, attrs=crs.to_cf()),
                    },
                )

                out = sen1.get_dem([0, 0, 1, 1], resolution=0.5, crs=None)
                self.assertIsInstance(out, xr.DataArray)
                self.assertEqual((2, 2), out.values.shape)
                self.assertDictEqual(crs.to_cf(), out.spatial_ref.attrs)

    def test_az_orbit_roundtrip(self):
        epoch = np.datetime64("2024-01-01T00:00:00")
        time_az = xr.DataArray(
            np.array(
                ["2024-01-01T00:00:00", "2024-01-01T00:00:02"], dtype="datetime64[ns]"
            ),
            dims=("azimuth_time",),
        )
        t_orbit = sen1.az_to_orbit(time_az, epoch)
        back = sen1.orbit_to_az(t_orbit, epoch)
        np.testing.assert_array_equal(time_az.values, back.values)

    def test_convert_dem_to_ecef(self):
        dem = xr.DataArray(
            da.from_array(np.ones((4, 4), dtype="float32"), chunks=(2, 2)),
            dims=("lat", "lon"),
            coords={"lat": [0, 1, 2, 3], "lon": [0, 1, 2, 3]},
        )
        gm_dem = GridMapping.from_dataset(dem.to_dataset(name="dem"))
        out = sen1.convert_dem_to_ecef(
            dem,
            {"crs": gm_dem.crs.to_wkt(), "xy_var_names": gm_dem.xy_var_names},
        )
        self.assertEqual(("axis", "lat", "lon"), out.dims)
        self.assertEqual(3, out.sizes["axis"])

    def test_fit_position_and_poly_derivative(self):
        time_az = xr.DataArray(
            np.array(
                [
                    "2024-01-01T00:00:00",
                    "2024-01-01T00:00:01",
                    "2024-01-01T00:00:02",
                    "2024-01-01T00:00:03",
                ],
                dtype="datetime64[ns]",
            ),
            dims=("azimuth_time",),
        )
        pos = xr.DataArray(
            np.ones((4, 3), dtype="float64"),
            dims=("azimuth_time", "axis"),
            coords={"azimuth_time": time_az, "axis": ["x", "y", "z"]},
        )
        coeff = sen1.fit_position(pos, deg=3)
        self.assertIn("epoch", coeff.attrs)
        deriv = sen1.poly_derivative(coeff)
        self.assertEqual(coeff.sizes["degree"] - 1, deriv.sizes["degree"])

    def test_get_source_location_and_assign_grid_mapping(self):
        dem = xr.DataArray(
            np.ones((2, 2), dtype="float64"),
            dims=("lat", "lon"),
            coords={"lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        time_slr = xr.DataArray(
            np.ones((2, 2), dtype="float64"),
            dims=("azimuth_time", "ground_range"),
            coords={"azimuth_time": [0, 1], "ground_range": [0, 1]},
        )
        sat_position = xr.DataArray(
            np.ones((2, 3), dtype="float64"),
            dims=("azimuth_time", "axis"),
            coords={"azimuth_time": [0, 1], "axis": ["x", "y", "z"]},
        )
        with (
            patch.object(sen1, "fit_ground_range", return_value=self.gr_coeff),
            patch.object(sen1, "fit_position", return_value=self.posvel_coeff),
            patch.object(sen1, "backward_geocode") as bg,
        ):
            bg.return_value = xr.Dataset(
                {
                    "azimuth_time": xr.DataArray(
                        np.zeros((2, 2), dtype="datetime64[ns]"), dims=("lat", "lon")
                    ),
                    "ground_range": xr.DataArray(np.zeros((2, 2)), dims=("lat", "lon")),
                    "gamma_area": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
                }
            )
            gm_dem = GridMapping.from_dataset(dem.to_dataset(name="dem"))
            out = sen1.get_source_location(
                dem,
                time_slr,
                sat_position,
                self.grid_params,
                gm_dem,
                True,
            )
        self.assertIn("spatial_ref", out.coords)
        out = sen1.assign_grid_mapping(
            xr.Dataset({"a": xr.DataArray([1], dims=("x",))})
        )
        self.assertEqual("spatial_ref", out["a"].attrs["grid_mapping"])

    def test_get_source_location_without_rtc(self):

        gm_dem = GridMapping.from_dataset(self.dem.to_dataset(name="dem"))
        with (
            patch.object(
                sen1,
                "backward_geocode",
                return_value=xr.Dataset(
                    {
                        "azimuth_time": xr.DataArray(
                            np.zeros((2, 2), dtype="datetime64[ns]"),
                            dims=("lat", "lon"),
                        ),
                        "ground_range": xr.DataArray(
                            np.zeros((2, 2)), dims=("lat", "lon")
                        ),
                    }
                ),
            ) as bg,
            patch.object(sen1, "fit_ground_range", return_value=self.gr_coeff),
            patch.object(sen1, "fit_position", return_value=self.posvel_coeff),
        ):
            out = sen1.get_source_location(
                self.dem,
                self.time_slr,
                self.sat_position,
                self.grid_params,
                gm_dem,
                False,
            )
        bg.assert_called_once()
        self.assertNotIn("gamma_area", out.data_vars)

    def test_compute_indexing_and_sample_array_errors(self):
        data = xr.Dataset(
            {
                "a": xr.DataArray(
                    np.arange(9).reshape(3, 3), dims=("azimuth_time", "ground_range")
                )
            }
        )
        az_idx = xr.DataArray(
            da.from_array(np.array([[0.2, 1.2], [0.2, 1.2]]), chunks=(2, 2)),
            dims=("lat", "lon"),
        )
        gr_idx = xr.DataArray(
            da.from_array(np.array([[0.2, 1.2], [0.2, 1.2]]), chunks=(2, 2)),
            dims=("lat", "lon"),
        )
        indexing = sen1._compute_indexing(data, az_idx, gr_idx)
        np.testing.assert_array_equal(
            indexing.ij_bboxes,
            np.array([[[0]], [[0]], [[3]], [[3]]], dtype=np.int32),
        )

        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        nearest = sen1._sample_array_at_indices(
            arr, np.array([[0.6]]), np.array([[1.4]]), "nearest"
        )
        np.testing.assert_array_equal(nearest, np.array([[4.0]]))
        bilinear = sen1._sample_array_at_indices(
            arr, np.array([[0.5]]), np.array([[0.5]]), "bilinear"
        )
        np.testing.assert_allclose(bilinear, np.array([[2.5]]))
        with pytest.raises(NotImplementedError, match="interp_methods"):
            sen1._sample_array_at_indices(
                np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2)), "cubic"
            )

    def test_zero_doppler_and_prime(self):
        time_orbit = xr.DataArray(np.zeros((2, 2)), dims=("lat", "lon"))
        f, payload = sen1.zero_doppler(
            self.dem_ecef, self.posvel_coeff, self.posvel_coeff, time_orbit
        )
        self.assertEqual(("lat", "lon"), f.dims)
        fp = sen1.zero_doppler_prime(self.posvel_coeff, time_orbit, payload)
        self.assertEqual(("lat", "lon"), fp.dims)

    def test_secant_and_newton(self):
        root = xr.DataArray([3.0], dims=("p",))

        def func(t):
            out = t - root
            return out, {"t": t}

        def func_p(_t, _payload):
            return xr.ones_like(root)

        t0 = xr.DataArray([0.0], dims=("p",))
        t1 = xr.DataArray([1.0], dims=("p",))
        secant_t, _, secant_f, _, _ = sen1.secant(func, t0, t1, tol_f=1e-9, maxiter=20)
        newton_t, newton_f, _, _ = sen1.newton(func, func_p, t0, tol_f=1e-9, maxiter=20)
        self.assertTrue(np.allclose(secant_t.values, [3.0]))
        self.assertTrue(np.allclose(newton_t.values, [3.0]))
        self.assertTrue(np.allclose(secant_f.values, [0.0]))
        self.assertTrue(np.allclose(newton_f.values, [0.0]))

    def test_secant_breaks_on_small_dt(self):
        def func(t):
            return xr.ones_like(t) * 10.0, None

        t0 = xr.DataArray([1.0], dims=("p",))
        t1 = xr.DataArray([1.0 + 1e-8], dims=("p",))
        _, _, _, k, _ = sen1.secant(func, t0, t1, tol_f=1e-12, tol_t=1e-6, maxiter=5)
        self.assertEqual(0, k)

    def test_newton_breaks_on_small_dt(self):
        def func(t):
            return xr.ones_like(t), None

        def func_p(t, _payload):
            return xr.ones_like(t) * 1e9

        t0 = xr.DataArray([0.0], dims=("p",))
        _, _, k, _ = sen1.newton(func, func_p, t0, tol_f=1e-12, tol_t=1e-6, maxiter=5)
        self.assertEqual(0, k)

    def test_backward_geocode_invalid_method(self):
        with pytest.raises(ValueError, match="method needs to be either"):
            sen1.backward_geocode(
                self.dem,
                pos_coeff=self.posvel_coeff,
                vel_coeff=self.posvel_coeff,
                gr_coeff=self.gr_coeff,
                gm_dem_params=self.gm_dem_params,
                grid_params=self.grid_params,
                method="x",
            )

    def test_backward_geocode_secant_and_newton_paths(self):
        payload = (
            xr.DataArray(np.ones((3, 2, 2)), dims=("axis", "lat", "lon")),
            xr.DataArray(np.ones((3, 2, 2)), dims=("axis", "lat", "lon")),
        )
        with (
            patch.object(
                sen1,
                "secant",
                return_value=(
                    xr.DataArray([[0.0, 0.0], [0.0, 0.0]], dims=("lat", "lon")),
                    None,
                    None,
                    0,
                    payload,
                ),
            ),
            patch.object(
                sen1,
                "newton",
                return_value=(
                    xr.DataArray([[0.0, 0.0], [0.0, 0.0]], dims=("lat", "lon")),
                    None,
                    0,
                    payload,
                ),
            ),
        ):
            out_secant = sen1.backward_geocode(
                self.dem,
                pos_coeff=self.posvel_coeff,
                vel_coeff=self.posvel_coeff,
                gr_coeff=self.gr_coeff,
                grid_params=self.grid_params,
                gm_dem_params=self.gm_dem_params,
                method="secant",
            )
            out_newton = sen1.backward_geocode(
                self.dem,
                pos_coeff=self.posvel_coeff,
                vel_coeff=self.posvel_coeff,
                gr_coeff=self.gr_coeff,
                grid_params=self.grid_params,
                gm_dem_params=self.gm_dem_params,
                method="newton",
            )
        self.assertEqual(3, len(out_secant))
        self.assertEqual(3, len(out_newton))

    def test_compute_gamma_area_clips_negative(self):
        area = xr.DataArray(
            np.array(
                [
                    [[1.0, -1.0], [1.0, -1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            dims=("axis", "lat", "lon"),
            coords=self.dem_ecef.coords,
        )
        direction = xr.DataArray(
            np.array(
                [
                    [[1.0, -1.0], [1.0, -1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            dims=("axis", "lat", "lon"),
            coords=self.dem_ecef.coords,
        )
        with patch.object(sen1, "compute_dem_area", return_value=area):
            gamma = sen1.compute_gamma_area(
                self.dem_ecef, self.gm_dem_params, direction
            )
        self.assertTrue(np.all(gamma.values >= 0))
        self.assertEqual(0.0, float(gamma.values[0, 1]))

    def test_compute_dem_area(self):
        lat = np.arange(10)
        lon = np.arange(10)
        lon2d, lat2d = np.meshgrid(lon, lat, indexing="xy")
        x = 10.0 + lon2d
        y = 20.0 + lat2d
        z = 30.0 + lon2d + lat2d
        dem_ecef = xr.DataArray(
            np.stack([x, y, z], axis=0).astype("float32"),
            dims=("axis", "lat", "lon"),
            coords={"axis": ["x", "y", "z"], "lat": lat, "lon": lon},
        )
        area = sen1.compute_dem_area(dem_ecef, self.gm_dem_params)
        self.assertEqual(("axis", "lat", "lon"), area.dims)

    def test_sum_weights_and_gamma_weight_helpers(self):
        scr_indices = xr.Dataset(
            {
                "gamma_area": xr.DataArray([[1.0, 2.0]], dims=("lat", "lon")),
                "az_idx": xr.DataArray([[0.2, 0.8]], dims=("lat", "lon")),
                "gr_idx": xr.DataArray([[1.2, 1.8]], dims=("lat", "lon")),
            }
        )
        reduced = xr.DataArray(
            [[3.0]],
            dims=("gr_idx", "az_idx"),
            coords={"gr_idx": [1], "az_idx": [1]},
        )
        with patch.object(sen1.flox.xarray, "xarray_reduce", return_value=reduced):
            summed = sen1.sum_weights(
                scr_indices.gamma_area, scr_indices.az_idx, scr_indices.gr_idx
            )
        self.assertEqual(("lat", "lon"), summed.dims)
        self.assertEqual((1, 2), summed.data.shape)

        with patch.object(
            sen1, "sum_weights", return_value=xr.zeros_like(scr_indices.gamma_area)
        ) as sw:
            _ = sen1.gamma_weights_bilinear(scr_indices)
            self.assertEqual(4, sw.call_count)
        with patch.object(
            sen1, "sum_weights", return_value=xr.zeros_like(scr_indices.gamma_area)
        ) as sw:
            _ = sen1.gamma_weights_nearest(scr_indices)
            sw.assert_called_once()

    def test_apply_gamma_weights(self):
        params = sen1.GridParams(
            range0=0.0,
            range0_scale=0.0,
            d_range=1.0,
            d_range_scale=1.0,
            spacing_range=1.0,
            spacing_range_scale=1.0,
            az0=np.datetime64("2024-01-01T00:00:00"),
            az0_scale=np.datetime64("2024-01-01T00:00:00"),
            d_az=1.0,
            d_az_scale=1.0,
            spacing_az=2.0,
            spacing_az_scale=2.0,
        )
        src_loc = xr.Dataset(
            {
                "azimuth_time": xr.DataArray(
                    np.array(
                        [
                            ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                            ["2024-01-01T00:00:00", "2024-01-01T00:00:01"],
                        ],
                        dtype="datetime64[ns]",
                    ),
                    dims=("lat", "lon"),
                ),
                "ground_range": xr.DataArray(
                    np.array([[0.0, 1.0], [0.0, 1.0]]), dims=("lat", "lon")
                ),
                "gamma_area": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon")),
            }
        )

        def passthrough(ds):
            return ds.gamma_area * 2

        out = sen1.apply_gamma_weights(src_loc, passthrough, params)
        self.assertTrue(np.allclose(out.values, 1.0))

    def test_fit_ground_range(self):
        time_slr_gcp = xr.DataArray(
            np.array(
                [
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                ]
            ),
            dims=("azimuth_time", "ground_range"),
            coords={"azimuth_time": [0, 1], "ground_range": [0, 1, 2]},
        )
        coeff = sen1.fit_ground_range(time_slr_gcp, deg=1)
        self.assertEqual(("azimuth_time", "degree"), coeff.dims)
        self.assertIn("mean", coeff.attrs)
        self.assertIn("std", coeff.attrs)

    def test_geocode_data(self):
        data = xr.Dataset(
            {
                "vv": xr.DataArray(
                    da.ones((2, 3), chunks=(2, 3)),
                    dims=("azimuth_time", "ground_range"),
                )
            },
            coords={
                "azimuth_time": np.array(
                    ["2023-12-31T23:59:50", "2024-01-01T00:00:20"],
                    dtype="datetime64[ns]",
                ),
                "ground_range": [0, 3, 6],
            },
        )
        time_az = xr.DataArray(
            da.from_array(
                np.array(
                    [
                        ["2024-01-01T00:00:00", "2024-01-01T00:00:02"],
                        ["2024-01-01T00:00:04", "2024-01-01T00:00:06"],
                    ],
                    dtype="datetime64[ns]",
                ),
                chunks=(2, 2),
            ),
            dims=("lat", "lon"),
        )
        ground_range = xr.DataArray(
            da.from_array(np.array([[2, 3], [2, 3]]), chunks=(2, 2)),
            dims=("lat", "lon"),
        )
        src_loc = xr.Dataset({"azimuth_time": time_az, "ground_range": ground_range})
        grid_params = sen1.GridParams(
            range0=0.0,
            range0_scale=0.0,
            d_range=3.0,
            d_range_scale=3.0,
            spacing_range=3.0,
            spacing_range_scale=3.0,
            az0=np.datetime64("2023-12-31T23:59:50"),
            az0_scale=np.datetime64("2023-12-31T23:59:50"),
            d_az=20.0,
            d_az_scale=20.0,
            spacing_az=3.0,
            spacing_az_scale=3.0,
        )

        out = sen1.geocode_data(data, src_loc, grid_params, "nearest")
        self.assertIn("vv", out.data_vars)
        np.testing.assert_allclose(out.vv.values, np.ones((2, 2), dtype=float))
