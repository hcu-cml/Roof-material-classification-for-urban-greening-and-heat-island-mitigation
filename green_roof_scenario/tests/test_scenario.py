from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from green_roof_scenario import ScenarioConfig, run_scenario
from green_roof_scenario.io import read_raster


def _write_band(path: Path, values: np.ndarray, transform) -> None:
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
    ) as dst:
        dst.write(values, 1)


class ScenarioIntegrationTest(unittest.TestCase):
    def test_small_synthetic_scenario_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / "scene"
            scene.mkdir()
            out = root / "out"
            transform = from_origin(500_000, 1_000, 30, 30)
            rows, cols = np.indices((24, 24))

            bands = {
                "SR_B2": 8_000 + cols * 5 + rows * 2,
                "SR_B4": 9_000 + rows * 8,
                "SR_B5": 12_000 + cols * 12 - rows * 3,
                "SR_B6": 11_000 + rows * 10 + cols * 2,
                "SR_B7": 10_000 + cols * 6 + rows,
                "ST_B10": 43_000 + rows * 12 + cols * 5,
                "QA_PIXEL": np.zeros((24, 24), dtype="uint16"),
            }
            for suffix, values in bands.items():
                _write_band(
                    scene / f"SYNTHETIC_{suffix}.TIF",
                    np.asarray(values, dtype="uint16"),
                    transform,
                )

            buildings_path = root / "buildings.gpkg"
            buildings = gpd.GeoDataFrame(
                {"material": ["metal", "tile"]},
                geometry=[
                    box(500_060, 700, 500_300, 940),
                    box(500_420, 340, 500_660, 580),
                ],
                crs="EPSG:32632",
            )
            buildings.to_file(buildings_path, layer="buildings", driver="GPKG")

            outputs = run_scenario(
                ScenarioConfig(
                    l2_folder=scene,
                    buildings=buildings_path,
                    layer="buildings",
                    roof_material_field="material",
                    roof_materials_type="metal",
                    out_dir=out,
                    build_lst=True,
                    model="linear",
                    sample_frac=1,
                    min_sample_spacing=0,
                    supersample=2,
                    target_ndvi=0.6,
                    target_albedo=0.25,
                    target_ndbi=-0.2,
                    write_indices_rasters=True,
                    write_roof_fraction_raster=True,
                )
            )

            delta, _ = read_raster(outputs.delta_raster)
            result_buildings = gpd.read_file(outputs.buildings_layer)
            provenance = (out / "_greening_provenance.txt").read_text(encoding="utf-8")

            self.assertEqual(delta.shape, (20, 20))
            self.assertTrue(np.isfinite(delta).any())
            selected_by_material = dict(
                zip(result_buildings["material"], result_buildings["selected_for_greening"])
            )
            self.assertEqual(selected_by_material, {"metal": True, "tile": False})
            self.assertIn("Target NDBI: -0.2", provenance)
            self.assertTrue(outputs.stats_report and outputs.stats_report.exists())
            self.assertTrue(outputs.roof_fraction_raster and outputs.roof_fraction_raster.exists())


if __name__ == "__main__":
    unittest.main()
