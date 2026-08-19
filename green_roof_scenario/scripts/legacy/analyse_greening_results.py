# -----------------------------------------------------------------------------
# LEGACY / AD-HOC SCRIPT
# Kept for transparency: this is one of the helper scripts used during the
# original city runs of the study. It is not part of the supported CLI.
#
# ALL FILE PATHS AND DEFAULTS BELOW ARE EXAMPLES FROM THOSE RUNS.
# No data ships with this repository: replace every path with a link to your
# own data before running, either by editing the constants/defaults in this
# file or by passing the corresponding command-line arguments.
# -----------------------------------------------------------------------------

"""Summaries for baseline vs. greening LST rasters and building averages."""

from __future__ import annotations

import argparse
from pathlib import Path
import ast

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize greening scenario impacts at city and building scales."
    )
    parser.add_argument(
        "--baseline",
        default=Path("results_flachdach_only/baseline_LST.tif"),
        type=Path,
        help="Baseline LST raster (default: %(default)s).",
    )
    parser.add_argument(
        "--scenario",
        default=Path("results_flachdach_only/scenario_pred_LST.tif"),
        type=Path,
        help="Greening scenario LST raster (default: %(default)s).",
    )
    parser.add_argument(
        "--boundary",
        default=Path("data/hamburg_boundary.gpkg"),
        type=Path,
        help="City boundary vector file (default: %(default)s).",
    )
    parser.add_argument(
        "--buildings",
        default=Path("results_flachdach_only/buildings_greening_impact.gpkg"),
        type=Path,
        help="Building footprints with LST aggregates (default: %(default)s).",
    )
    parser.add_argument(
        "--roof-field",
        default="predicted_roof_materials",
        help="Optional column storing roof types/materials for grouped stats.",
    )
    parser.add_argument(
        "--roof-types",
        nargs="+",
        default=['0', '4'],
        help="Specific roof types to summarize (requires --roof-field).",
    )
    return parser.parse_args()


def _load_boundary(boundary_path: Path, target_crs) -> list[dict]:
    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary layer {boundary_path} is empty.")
    boundary = boundary.to_crs(target_crs)
    shape = boundary.union_all()
    return [shape.__geo_interface__]


def _masked_to_array(data: np.ndarray, nodata: float | None) -> np.ndarray:
    arr = np.ma.filled(data, np.nan).astype("float32")
    if nodata is not None and not np.isnan(nodata):
        arr[arr == nodata] = np.nan
    return np.squeeze(arr, axis=0)


def summarize_city(
    baseline_path: Path,
    scenario_path: Path,
    boundary_path: Path,
) -> float:
    with rasterio.open(baseline_path) as baseline_ds:
        with rasterio.open(scenario_path) as scenario_ds:
            if baseline_ds.crs != scenario_ds.crs:
                raise ValueError("Baseline and scenario rasters must share the same CRS.")
            geoms = _load_boundary(boundary_path, baseline_ds.crs)
            baseline_data, _ = mask(baseline_ds, geoms, crop=True)
            scenario_data, _ = mask(scenario_ds, geoms, crop=True)
            baseline_arr = _masked_to_array(baseline_data, baseline_ds.nodata)
            scenario_arr = _masked_to_array(scenario_data, scenario_ds.nodata)

    valid = np.isfinite(baseline_arr) & np.isfinite(scenario_arr)
    if not np.any(valid):
        raise ValueError("No valid LST pixels found inside the boundary.")
    diff = scenario_arr - baseline_arr
    city_mean = float(diff[valid].mean())
    print(
        f"City-wide mean LST difference (scenario - baseline): {city_mean:.3f} "
        "LST units."
    )
    return city_mean


def summarize_buildings(
    buildings_path: Path,
    roof_field: str | None,
    roof_types: list[str] | None,
) -> None:
    gdf = gpd.read_file(buildings_path)
    required = {"lst_baseline_mean", "lst_scenario_mean", "delta_mean"}
    missing = required - set(gdf.columns)
    if missing:
        raise KeyError(f"Missing required columns in {buildings_path}: {sorted(missing)}")

    gdf = gdf.copy()
    # If roof_field contains lists or stringified lists, keep only the first element
    if roof_field in gdf.columns:
        def _first_roof(value):
            # Already a list
            if isinstance(value, list):
                return value[0] if value else None
            # String representation of a list, e.g. '["0", "1", "4"]'
            if isinstance(value, str) and value.strip().startswith("["):
                try:
                    parsed = ast.literal_eval(value)
                    if isinstance(parsed, list) and parsed:
                        return parsed[0]
                except (SyntaxError, ValueError):
                    pass
            return value

        gdf[roof_field] = gdf[roof_field].apply(_first_roof)

    gdf["lst_diff"] = gdf["lst_scenario_mean"] - gdf["lst_baseline_mean"]

    overall_mean = float(gdf["lst_diff"].dropna().mean())
    print(f"Mean LST difference across all buildings: {overall_mean:.3f}")

    roof_col = roof_field
    if roof_col is None:
        for candidate in [
            "predictedroofmaterials",
            "predictedroofmaterial",
            "roof_type",
            "roofmaterial",
        ]:
            if candidate in gdf.columns:
                roof_col = candidate
                break

    if roof_col is None and roof_types:
        raise ValueError("Provide --roof-field when requesting --roof-types.")
    if roof_col is None or roof_col not in gdf.columns:
        return

    if roof_types:
        for rt in roof_types:
            subset = gdf[gdf[roof_col] == rt]
            if subset.empty:
                print(f"  Roof type '{rt}' not present in the building layer.")
                continue
            mean_val = float(subset["lst_diff"].dropna().mean())
            print(f"  Mean LST difference for roof type '{rt}': {mean_val:.3f}")
    else:
        grouped = gdf.groupby(roof_col)["lst_diff"].mean().sort_values()
        print("Mean LST difference by roof type:")
        for rt, mean_val in grouped.items():
            print(f"  {rt}: {mean_val:.3f}")


def main() -> None:
    args = parse_args()
    summarize_city(args.baseline, args.scenario, args.boundary)
    summarize_buildings(args.buildings, args.roof_field, args.roof_types)


if __name__ == "__main__":
    main()
