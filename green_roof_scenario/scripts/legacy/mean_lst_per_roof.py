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

"""
Heat-Island vs Roof-Material Analysis
=====================================

Assign mean Land Surface Temperature (LST) from a Level-2 Landsat-derived raster
to each building polygon, then save an enriched building layer and optional
summary CSV by roof-material.

Example:
    python3 mean_lst_per_roof.py \
        --buildings data/building_footprint.gpkg \
        --raster data/lst_bremen.tif \
        --roof_field predictedroofmaterials \
        --out_dir results \
        --all_touched
"""

import argparse
from pathlib import Path
import warnings

import geopandas as gpd
import rasterio
from rasterstats import zonal_stats
import pandas as pd
import numpy as np


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(
        description=(
            "Compute mean LST per building and export an enriched layer "
            "plus an optional roof-type summary."
        )
    )
    p.add_argument(
        "--buildings",
        required=True,
        help="Path to the building GeoPackage/GeoJSON/Shapefile, etc.",
    )
    p.add_argument(
        "--layer",
        default=None,
        help="Optional layer name inside a GeoPackage (if applicable).",
    )
    p.add_argument(
        "--raster",
        required=True,
        help=(
            "Single-band LST raster (GeoTIFF). Values should be in °C or the "
            "unit you intend to report; the script will not rescale."
        ),
    )
    p.add_argument(
        "--roof_field",
        default="predictedrooftypematerial",
        help="Name of the roof-material field (default: %(default)s).",
    )
    p.add_argument(
        "--out_dir",
        default="results",
        help="Folder for outputs (default: %(default)s).",
    )
    p.add_argument(
        "--all_touched",
        action="store_true",
        help="Use all_touched=True when computing zonal stats (includes edge pixels).",
    )
    p.add_argument(
        "--csv_summary",
        action="store_true",
        help="Also write a CSV summarizing mean LST by roof type.",
    )
    p.add_argument(
        "--keep_null_roof",
        action="store_true",
        help="Keep features with NULL roof field instead of dropping them.",
    )
    return p.parse_args()


essential_cols = ["lst_mean"]


def read_buildings(path: str, layer: str | None) -> gpd.GeoDataFrame:
    print("Reading building layer …")
    if layer:
        gdf = gpd.read_file(path, layer=layer)
    else:
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Building layer is empty.")
    if gdf.geometry.isna().any():
        warnings.warn("Some features have NULL geometry and will be dropped.")
        gdf = gdf[~gdf.geometry.isna()].copy()
    # Attempt to fix invalid geometries (self-intersections)
    if not gdf.is_valid.all():
        warnings.warn("Fixing invalid geometries with buffer(0).")
        gdf["geometry"] = gdf.buffer(0)
        gdf = gdf[gdf.is_valid].copy()
    return gdf


def open_raster_meta(raster_path: str):
    print("Opening LST raster …")
    with rasterio.open(raster_path) as src:
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "nodata": src.nodata,
            "width": src.width,
            "height": src.height,
            "dtype": src.dtypes[0],
        }
    return meta


def ensure_same_crs(gdf: gpd.GeoDataFrame, raster_crs) -> gpd.GeoDataFrame:
    if gdf.crs != raster_crs:
        print("Re-projecting buildings to raster CRS …")
        gdf = gdf.to_crs(raster_crs)
    return gdf


def compute_mean_lst(
    gdf: gpd.GeoDataFrame,
    raster_path: str,
    nodata,
    all_touched: bool,
) -> np.ndarray:
    """Compute mean of raster values within each polygon using raster path.

    Using the raster path lets rasterstats stream from disk (more memory-safe)
    instead of loading the whole array into memory.
    """
    print("Computing mean LST for each building …")
    stats = zonal_stats(
        gdf,
        raster_path,
        stats=["mean"],
        nodata=nodata,
        all_touched=all_touched,
        geojson_out=False,
        raster_out=False,
    )
    return np.array([s["mean"] for s in stats], dtype=float)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gdf = read_buildings(args.buildings, args.layer)
    meta = open_raster_meta(args.raster)

    # Ensure CRS alignment
    gdf = ensure_same_crs(gdf, meta["crs"])

    # Guard: roof field present
    if args.roof_field not in gdf.columns:
        raise KeyError(
            f"Roof field '{args.roof_field}' not found. Available columns: {list(gdf.columns)}"
        )

    # Optionally drop features without roof info
    if not args.keep_null_roof:
        before = len(gdf)
        gdf = gdf.dropna(subset=[args.roof_field]).copy()
        after = len(gdf)
        if after < before:
            print(f"Dropped {before - after} features with NULL '{args.roof_field}'.")

    # Compute zonal means
    lst_means = compute_mean_lst(
        gdf=gdf,
        raster_path=args.raster,
        nodata=meta["nodata"],
        all_touched=args.all_touched,
    )
    gdf["lst_mean"] = lst_means

    # Drop rows where mean is NaN (e.g., all masked)
    valid_mask = ~pd.isna(gdf["lst_mean"]) & np.isfinite(gdf["lst_mean"])
    dropped = int((~valid_mask).sum())
    if dropped > 0:
        print(f"Warning: {dropped} features had no valid LST coverage and were dropped.")
    gdf = gdf.loc[valid_mask].copy()

    # City-wide mean and anomaly
    city_mean = float(gdf["lst_mean"].mean())
    gdf["delta"] = gdf["lst_mean"] - city_mean

    # Save enriched layer
    enriched_path = out_dir / "buildings_with_lst.gpkg"
    gdf.to_file(enriched_path, driver="GPKG")
    print(f"Enriched layer written to {enriched_path}")

    # Optional roof-type summary CSV
    if args.csv_summary:
        summary = (
            gdf.groupby(args.roof_field)["lst_mean"]
            .agg(["count", "mean", "std", "min", "max"])
            .reset_index()
            .sort_values("mean")
        )
        csv_path = out_dir / "roof_type_lst_summary.csv"
        summary.to_csv(csv_path, index=False)
        print(f"Roof-type summary written to {csv_path}")

    # Also emit a tiny metadata text file for provenance
    meta_txt = out_dir / "_lst_join_provenance.txt"
    meta_txt.write_text(
        (
            f"Raster: {Path(args.raster).name}\n"
            f"Raster CRS: {meta['crs']}\n"
            f"Raster nodata: {meta['nodata']}\n"
            f"Buildings: {Path(args.buildings).name}\n"
            f"Layer: {args.layer}\n"
            f"Roof field: {args.roof_field}\n"
            f"All touched: {args.all_touched}\n"
            f"City mean LST: {city_mean:.3f}\n"
        ),
        encoding="utf-8",
    )
    print(f"Provenance written to {meta_txt}")


if __name__ == "__main__":
    main()