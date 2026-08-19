from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from green_roof_scenario.config import ScenarioConfig
from green_roof_scenario.io import read_raster
from green_roof_scenario.l2 import SR_OFFSET, SR_SCALE, build_lst_from_l2, compute_ndvi_albedo_from_l2
from green_roof_scenario.masking import roof_mask_fraction, subset_buildings
from green_roof_scenario.modeling import fit_model, predict_partial, sample_model_inputs


def _write_raster(path: Path, values: np.ndarray, *, transform, nodata=None) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:32632",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(values, 1)


class ConfigTests(unittest.TestCase):
    def test_valid_config_normalizes_paths(self) -> None:
        config = ScenarioConfig(
            l2_folder="scene",
            buildings="buildings.gpkg",
            roof_materials_type="metal",
        )
        self.assertEqual(config.l2_folder, Path("scene"))
        self.assertEqual(config.buildings, Path("buildings.gpkg"))

    def test_config_rejects_invalid_sampling_fraction(self) -> None:
        with self.assertRaisesRegex(ValueError, "sample_frac"):
            ScenarioConfig(
                l2_folder="scene",
                buildings="buildings.gpkg",
                roof_materials_type="metal",
                sample_frac=0,
            )

    def test_config_requires_an_lst_source(self) -> None:
        with self.assertRaisesRegex(ValueError, "Provide lst"):
            ScenarioConfig(
                l2_folder="scene",
                buildings="buildings.gpkg",
                roof_materials_type="metal",
                build_lst=False,
            )


class RasterIoTests(unittest.TestCase):
    def test_read_raster_converts_nodata_to_nan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "integer.tif"
            values = np.array([[1, -9999], [2, 3]], dtype="int16")
            _write_raster(path, values, transform=from_origin(0, 2, 1, 1), nodata=-9999)

            actual, profile = read_raster(path)

            self.assertEqual(actual.dtype, np.dtype("float32"))
            self.assertTrue(np.isnan(actual[0, 1]))
            self.assertEqual(profile["crs"].to_epsg(), 32632)


class LandsatTests(unittest.TestCase):
    def test_indices_are_reprojected_to_template_extent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            transform = from_origin(0, 4, 1, 1)
            base = np.arange(16, dtype="uint16").reshape(4, 4)
            raw_bands = {
                "B2": 8_000 + base,
                "B4": 9_000 + base * 2,
                "B5": 12_000 + base * 3,
                "B6": 11_000 + base * 4,
                "B7": 10_000 + base * 5,
            }
            for band, values in raw_bands.items():
                _write_raster(folder / f"SCENE_SR_{band}.TIF", values, transform=transform)

            template = folder / "template.tif"
            _write_raster(
                template,
                np.zeros((2, 2), dtype="float32"),
                transform=from_origin(1, 3, 1, 1),
                nodata=np.nan,
            )

            ndvi, albedo, ndbi, _ = compute_ndvi_albedo_from_l2(folder, template)

            red = raw_bands["B4"][1:3, 1:3].astype("float32") * SR_SCALE + SR_OFFSET
            nir = raw_bands["B5"][1:3, 1:3].astype("float32") * SR_SCALE + SR_OFFSET
            expected_ndvi = np.clip((nir - red) / (nir + red + 1e-6), -1, 1)
            np.testing.assert_allclose(ndvi, expected_ndvi, rtol=1e-5, atol=1e-5)
            self.assertEqual(albedo.shape, (2, 2))
            self.assertEqual(ndbi.shape, (2, 2))

    def test_lst_builder_applies_qa_cloud_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            transform = from_origin(0, 2, 1, 1)
            _write_raster(
                folder / "SCENE_ST_B10.TIF",
                np.full((2, 2), 43_000, dtype="uint16"),
                transform=transform,
            )
            qa = np.zeros((2, 2), dtype="uint16")
            qa[0, 1] = 1 << 3
            _write_raster(folder / "SCENE_QA_PIXEL.TIF", qa, transform=transform)

            _, lst, _ = build_lst_from_l2(folder, out_path=folder / "lst.tif")

            self.assertTrue(np.isfinite(lst[0, 0]))
            self.assertTrue(np.isnan(lst[0, 1]))


class MaskingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buildings = gpd.GeoDataFrame(
            {"material": ["metal", None]},
            geometry=[box(0, 1, 1, 2), box(1, 0, 2, 1)],
            crs="EPSG:32632",
        )

    def test_keep_null_roof_includes_null_values(self) -> None:
        without_null = subset_buildings(self.buildings, "material", "metal")
        with_null = subset_buildings(
            self.buildings,
            "material",
            "metal",
            keep_null_roof=True,
        )
        self.assertEqual(len(without_null), 1)
        self.assertEqual(len(with_null), 2)

    def test_fraction_mask_uses_requested_grid(self) -> None:
        profile = {
            "height": 2,
            "width": 2,
            "transform": from_origin(0, 2, 1, 1),
        }
        result = roof_mask_fraction(self.buildings.iloc[[0]], profile, supersample=2)
        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(float(result[0, 0]), 1.0)
        self.assertAlmostEqual(float(result[1, 1]), 0.0)


class ModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        grid = np.arange(9, dtype="float32").reshape(3, 3)
        self.ndvi = grid / 10
        self.albedo = 0.1 + grid / 100
        self.ndbi = 0.3 - grid / 20
        self.lst = 25 - 2 * self.ndvi + self.albedo + self.ndbi

    def test_linear_model_and_partial_prediction_skip_non_finite_inputs(self) -> None:
        model, metrics = fit_model(
            self.lst,
            self.ndvi,
            self.albedo,
            self.ndbi,
            frac=1,
            model_type="linear",
        )
        ndvi = self.ndvi.copy()
        ndvi[0, 0] = np.nan
        prediction = predict_partial(
            model,
            ndvi,
            self.albedo,
            np.ones((3, 3), dtype=bool),
            self.ndbi,
        )
        self.assertTrue(np.isnan(prediction[0, 0]))
        self.assertTrue(np.isfinite(prediction[1, 1]))
        self.assertIn("rmse_test", metrics)

    def test_sampling_rejects_mismatched_shapes(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            sample_model_inputs(
                self.lst,
                self.ndvi[:2],
                self.albedo,
                self.ndbi,
                frac=1,
                seed=42,
            )


if __name__ == "__main__":
    unittest.main()
