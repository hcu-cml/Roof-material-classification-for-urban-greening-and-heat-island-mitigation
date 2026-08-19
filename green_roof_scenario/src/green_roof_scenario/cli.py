"""Command-line interface for the green roof scenario package."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from .config import ScenarioConfig
from .scenario import run_scenario


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the main scenario workflow."""
    parser = argparse.ArgumentParser(description="Greened roof LST scenario (empirical model).")
    parser.add_argument("--lst", default=None, help="Baseline LST raster (°C), aligned to Landsat grid.")
    parser.add_argument(
        "--build_lst",
        action="store_true",
        help="Build the baseline LST raster directly from the provided Landsat L2 folder.",
    )
    parser.add_argument(
        "--lst_unit",
        choices=["celsius", "kelvin"],
        default="celsius",
        help="Unit for any LST raster built from the L2 folder (default Celsius).",
    )
    parser.add_argument(
        "--keep_lst_water",
        action="store_true",
        help="When building LST internally, keep QA water pixels instead of masking them.",
    )
    parser.add_argument("--buildings", required=True, help="Buildings file (GPKG/GeoJSON/shp).")
    parser.add_argument("--layer", default=None, help="Optional layer name inside GPKG.")
    parser.add_argument(
        "--roof_material_field",
        default="predictedrooftypematerial",
        help="Roof material field in the buildings layer.",
    )
    parser.add_argument(
        "--roof_materials_type",
        default=None,
        help="Comma-separated roof material types to convert to green (e.g., 'concrete,bitumen').",
    )
    parser.add_argument(
        "--roof_shape_field",
        default=None,
        help="Roof shape field in the buildings layer (optional).",
    )
    parser.add_argument(
        "--roof_shape_type",
        default=None,
        help="Comma-separated roof shape values to convert to green (e.g., 'flat,low_slope').",
    )
    parser.add_argument(
        "--roof_slope_field",
        default="roof_slope_mean_deg",
        help="Numeric roof slope field in degrees (default: roof_slope_mean_deg).",
    )
    parser.add_argument(
        "--max_roof_slope_deg",
        type=float,
        default=None,
        help="Only green roofs with slope <= this value in degrees (e.g., 15).",
    )
    parser.add_argument("--out_dir", default="results_greening", help="Output folder.")
    parser.add_argument(
        "--boundary",
        default=None,
        help=(
            "Optional boundary vector for clipping. If omitted, the building layer "
            "extent is used as the analysis extent."
        ),
    )
    parser.add_argument(
        "--l2_folder",
        required=True,
        help="Path to Landsat L2 scene folder (used to compute NDVI, Albedo, and NDBI).",
    )
    parser.add_argument(
        "--target_ndvi",
        type=float,
        default=0.4,
        help="Target NDVI for green roofs (default: 0.4).",
    )
    parser.add_argument(
        "--target_albedo",
        type=float,
        default=0.20,
        help="Target broadband albedo for green roofs (default: 0.20).",
    )
    parser.add_argument(
        "--target_ndbi",
        type=float,
        default=-0.15,
        help="Target NDBI for green roofs (default: -0.15). Lower values = less 'built-up'.",
    )
    parser.add_argument(
        "--sample_frac",
        type=float,
        default=0.1,
        help="Random sample fraction of valid pixels for model fitting (default: 0.1).",
    )
    parser.add_argument(
        "--min_sample_spacing",
        type=float,
        default=100.0,
        help="Approximate minimum spacing between training samples in meters (0 disables thinning).",
    )
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--model",
        choices=["linear", "rf"],
        default="rf",
        help="Model type: 'rf' (default) or 'linear' (Linear Regression).",
    )
    parser.add_argument(
        "--supersample",
        type=int,
        default=4,
        help="Supersampling factor to estimate roof fraction per pixel (default: 4).",
    )
    parser.add_argument(
        "--all_touched",
        action="store_true",
        help="Use all_touched=True for coarse roof mask (fast, no fractions).",
    )
    parser.add_argument(
        "--write_pred_baseline",
        action="store_true",
        help="Write model baseline prediction raster to the output folder.",
    )
    parser.add_argument(
        "--keep_null_roof",
        action="store_true",
        help="Keep features with NULL roof field instead of dropping them.",
    )
    parser.add_argument(
        "--write_roof_fraction_raster",
        action="store_true",
        help="Write the per-pixel roof fraction raster (0..1) to the output folder.",
    )
    parser.add_argument(
        "--write_indices_rasters",
        action="store_true",
        help="Write NDVI, albedo, and NDBI rasters aligned to the baseline LST.",
    )
    parser.add_argument(
        "--min_roof_area",
        type=float,
        default=0.0,
        help="Minimum roof area in square meters to consider for greening (default: 0).",
    )
    parser.add_argument(
        "--clip_positive_delta",
        action="store_true",
        help="Set any positive delta values to 0 (cooling-only output).",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default INFO).",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> ScenarioConfig:
    """Parse command-line arguments into a validated scenario configuration."""
    parser = build_parser()
    args = parser.parse_args(argv)
    boundary = args.boundary
    boundary_path = Path(boundary) if boundary and boundary.strip() else None

    has_materials = bool(args.roof_materials_type and args.roof_materials_type.strip())
    has_shapes = bool(args.roof_shape_type and args.roof_shape_type.strip())
    has_slope = args.max_roof_slope_deg is not None
    if not has_materials and not has_shapes and not has_slope:
        parser.error("Provide at least one of --roof_materials_type, --roof_shape_type, or --max_roof_slope_deg.")
    if has_slope and args.max_roof_slope_deg < 0:
        parser.error("--max_roof_slope_deg must be >= 0.")

    try:
        return ScenarioConfig(
            l2_folder=args.l2_folder,
            buildings=args.buildings,
            roof_material_field=args.roof_material_field,
            roof_materials_type=args.roof_materials_type,
            roof_shape_field=args.roof_shape_field,
            roof_shape_type=args.roof_shape_type,
            roof_slope_field=args.roof_slope_field,
            max_roof_slope_deg=args.max_roof_slope_deg,
            boundary=boundary_path,
            out_dir=args.out_dir,
            lst=args.lst,
            build_lst=args.build_lst,
            lst_unit=args.lst_unit,
            keep_lst_water=args.keep_lst_water,
            layer=args.layer,
            target_ndvi=args.target_ndvi,
            target_albedo=args.target_albedo,
            target_ndbi=args.target_ndbi,
            sample_frac=args.sample_frac,
            min_sample_spacing=args.min_sample_spacing,
            random_state=args.random_state,
            model=args.model,
            supersample=args.supersample,
            all_touched=args.all_touched,
            write_pred_baseline=args.write_pred_baseline,
            keep_null_roof=args.keep_null_roof,
            write_roof_fraction_raster=args.write_roof_fraction_raster,
            write_indices_rasters=args.write_indices_rasters,
            log_level=args.log_level,
            min_roof_area=args.min_roof_area,
            clip_positive_delta=args.clip_positive_delta,
        )
    except ValueError as exc:
        parser.error(str(exc))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the green-roof scenario command-line entry point."""
    config = parse_args(argv)
    log_level = getattr(logging, config.log_level)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    run_scenario(config)


if __name__ == "__main__":  # pragma: no cover
    main()
