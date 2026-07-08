#  Copyright (c) 2025-2026 by EOPF Sample Service team and contributors
#  Permissions are hereby granted under the terms of the Apache 2.0 License:
#  https://opensource.org/license/apache-2-0.

import os
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import dask.array as da
import numpy as np
import pyproj
import pytest
import xarray as xr
from xcube_resampling.gridmapping import GridMapping

from tests.helpers import make_s1_grd_datatree, make_s1_ocn_datatree
from xarray_eopf.amode import AnalysisModeRegistry
from xarray_eopf.amodes import sentinel1 as sen1
from xarray_eopf.amodes.sentinel1 import Sen1GRD, Sen1OCN, register, GridParams


class Sentinel1AnalysisModeTest(TestCase):
    def test_register(self):
        registry = AnalysisModeRegistry()
        register(registry)
        self.assertEqual(2, len(list(registry.keys())))
        self.assertIn(Sen1GRD.product_type, registry.keys())
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
    mode = Sen1GRD()
    dem = xr.DataArray(np.ones((2, 2), dtype="float32"), dims=("lat", "lon"))
    dt = make_s1_grd_datatree()

    def test_is_valid_source_ok(self):
        self.assertTrue(self.mode.is_valid_source("data/S1A_IW_GRDH_20240201.zarr"))
        self.assertTrue(self.mode.is_valid_source("S1D_SM_GRDH_TEST"))

    def test_is_not_valid_source(self):
        self.assertFalse(self.mode.is_valid_source("data/S1A_IW_SLC_20240201.zarr"))
        self.assertFalse(self.mode.is_valid_source(dict()))

    def test_get_grid_parameters(self):
        params = self.mode._get_grid_parameters(self.dt, (2.0, 3.0))

        self.assertEqual(0.0, params["gr0"])
        self.assertEqual(20.0, params["spacing_az"])
        self.assertEqual(10.0, params["d_gr"])
        self.assertEqual(30.0, params["d_gr_scale"])
        self.assertEqual(np.datetime64("2024-01-01T00:00:00"), params["az0"])
        self.assertEqual(0.5, params["d_az"])
        self.assertEqual(40.0, params["spacing_az_scale"])
        self.assertEqual(
            np.datetime64("2024-01-01T00:00:00.250000000"), params["az0_scale"]
        )

    def test_get_applicable_params(self: TestCase):
        dem = xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))
        self.assertEqual({}, self.mode.get_applicable_params())
        self.assertEqual(
            {
                "resolution": 10,
                "bbox": [1, 3, 4, 5],
                "crs": pyproj.CRS.from_string("EPSG:4326"),
                "dem": dem,
                "interp_methods": "nearest",
                "footprint_scale_factor": (2.0, 3.0),
                "apply_rtc": False,
            },
            self.mode.get_applicable_params(
                resolution=10,
                bbox=[1, 3, 4, 5],
                crs="EPSG:4326",
                dem=dem,
                interp_methods="nearest",
                footprint_scale_factor=(2.0, 3.0),
                apply_rtc=False,
            ),
        )
        with pytest.raises(TypeError, match="interp_methods"):
            self.mode.get_applicable_params(interp_methods="cubic")
        with pytest.raises(TypeError, match="footprint_scale_factor"):
            self.mode.get_applicable_params(footprint_scale_factor=(1.0, "x"))

    def test_convert_datatree(self):
        expected = xr.Dataset(
            {"vv": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))}
        )

        with patch.object(
            self.mode, "_terrain_correct", return_value=expected
        ) as mocked:
            out = self.mode.convert_datatree(self.dt, includes=["vv"], dem=self.dem)

        self.assertIs(out, expected)
        args, kwargs = mocked.call_args
        self.assertEqual(["beta0_vv"], list(args[0].data_vars))
        self.assertIs(args[3], self.dem)
        self.assertEqual("bilinear", kwargs["interp_method"])
        self.assertTrue(kwargs["apply_rtc"])
        self.assertIn("gr0", args[4])

    def test_convert_datatree_uses_get_dem(self):
        expected = xr.Dataset(
            {"vv": xr.DataArray(np.ones((2, 2)), dims=("lat", "lon"))}
        )

        with patch.object(sen1, "get_dem", return_value=self.dem) as get_dem_mock:
            with patch.object(self.mode, "_terrain_correct", return_value=expected):
                _ = self.mode.convert_datatree(self.dt, includes=["vv"])

        get_dem_mock.assert_called_once()
        args, _ = get_dem_mock.call_args
        self.assertEqual(self.dt.attrs["stac_discovery"]["bbox"], args[0])

    def test_convert_datatree_fail(self):
        with pytest.raises(ValueError, match="No valid variable names"):
            self.mode.convert_datatree(self.dt, includes="bibo", dem=self.dem)


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
            gr0=0.0,
            gr0_scale=0.0,
            d_gr=1.0,
            d_gr_scale=1.0,
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
            coords={"lat": [0.0, 0.1], "lon": [0.0, 0.1]},
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

    def test_gridparams_iter(self):
        self.assertEqual(
            [
                "gr0",
                "gr0_scale",
                "d_gr",
                "d_gr_scale",
                "az0",
                "az0_scale",
                "d_az",
                "d_az_scale",
                "spacing_az",
                "spacing_az_scale",
            ],
            list(iter(self.grid_params)),
        )

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

    def test_zero_doppler_and_prime(self):
        dem_ecef = xr.DataArray(
            np.ones((3, 2, 2), dtype="float64"),
            dims=("axis", "lat", "lon"),
            coords={"axis": ["x", "y", "z"], "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        time_orbit = xr.DataArray(np.zeros((2, 2)), dims=("lat", "lon"))
        pos_coeff = xr.DataArray(
            np.zeros((2, 3)),
            dims=("degree", "axis"),
            coords={"degree": [1, 0], "axis": ["x", "y", "z"]},
        )
        vel_coeff = xr.DataArray(
            np.zeros((2, 3)),
            dims=("degree", "axis"),
            coords={"degree": [1, 0], "axis": ["x", "y", "z"]},
        )
        vel_coeff.loc[{"degree": 0, "axis": "x"}] = 1.0
        f, payload = sen1.zero_doppler(dem_ecef, pos_coeff, vel_coeff, time_orbit)
        self.assertEqual(("lat", "lon"), f.dims)
        fp = sen1.zero_doppler_prime(vel_coeff, time_orbit, payload)
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
        dem_ecef = xr.DataArray(
            np.ones((3, 2, 2), dtype="float64"),
            dims=("axis", "lat", "lon"),
            coords={"axis": ["x", "y", "z"], "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
        )
        gm_dem = {
            "crs": GridMapping.from_dataset(
                dem_ecef.to_dataset(name="dem")
            ).crs.to_wkt(),
            "xy_var_names": ("lon", "lat"),
        }
        area = xr.DataArray(
            np.array(
                [
                    [[1.0, -1.0], [1.0, -1.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 0.0]],
                ]
            ),
            dims=("axis", "lat", "lon"),
            coords=dem_ecef.coords,
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
            coords=dem_ecef.coords,
        )
        with patch.object(sen1, "compute_dem_area", return_value=area):
            gamma = sen1.compute_gamma_area(dem_ecef, gm_dem, direction)
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
        acq = xr.Dataset(
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
            summed = sen1.sum_weights(acq.gamma_area, acq.az_idx, acq.gr_idx)
        self.assertEqual(("lat", "lon"), summed.dims)
        self.assertEqual((1, 2), summed.data.shape)

        with patch.object(
            sen1, "sum_weights", return_value=xr.zeros_like(acq.gamma_area)
        ) as sw:
            _ = sen1.gamma_weights_bilinear(acq)
            self.assertEqual(4, sw.call_count)
        with patch.object(
            sen1, "sum_weights", return_value=xr.zeros_like(acq.gamma_area)
        ) as sw:
            _ = sen1.gamma_weights_nearest(acq)
            sw.assert_called_once()

    def test_apply_gamma_weights(self):
        params = sen1.GridParams(
            gr0=0.0,
            gr0_scale=0.0,
            d_gr=1.0,
            d_gr_scale=1.0,
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
            gr0=0.0,
            gr0_scale=0.0,
            d_gr=3.0,
            d_gr_scale=3.0,
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
