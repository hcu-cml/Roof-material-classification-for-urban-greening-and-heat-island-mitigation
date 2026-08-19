"""Compute green-roof NDVI/NDBI/albedo zonal means and aggregate targets."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.errors import WindowError
from rasterio.features import geometry_mask, geometry_window
from shapely.geometry import mapping

from .io import save_raster
from .l2 import build_lst_from_l2, compute_ndvi_albedo_from_l2

DEFAULT_RASTER_NAMES = ("ndvi", "ndbi", "albedo")


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for green-roof parameter estimation."""
    parser = argparse.ArgumentParser(
        description=(
            "Add mean NDVI, NDBI, and albedo to green-roof polygons and report "
            "unweighted and area-weighted averages."
        )
    )
    parser.add_argument("--green-roofs", required=True, type=Path, help="Input green-roof vector file.")
    parser.add_argument("--layer", default=None, help="Optional input layer name for GeoPackage files.")
    parser.add_argument("--ndvi", type=Path, help="NDVI raster path.")
    parser.add_argument("--ndbi", type=Path, help="NDBI raster path.")
    parser.add_argument("--albedo", type=Path, help="Albedo raster path.")
    parser.add_argument(
        "--l2-folder",
        type=Path,
        help=(
            "Optional Landsat Collection 2 Level-2 scene folder. If index rasters "
            "are missing, they are derived from this folder before zonal stats run."
        ),
    )
    parser.add_argument(
        "--indices-out-dir",
        type=Path,
        help=(
            "Folder for derived ndvi.tif, ndbi.tif, albedo.tif, and baseline_LST.tif. "
            "Required with --l2-folder unless all three raster paths are provided."
        ),
    )
    parser.add_argument(
        "--baseline-lst",
        type=Path,
        help="Optional baseline LST/template raster path to use or create when deriving indices from --l2-folder.",
    )
    parser.add_argument(
        "--keep-lst-water",
        action="store_true",
        help="When deriving indices from --l2-folder, keep QA water pixels in the template LST mask.",
    )
    parser.add_argument(
        "--mask-with-lst",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mask derived index rasters to clear valid LST pixels (default: true).",
    )
    parser.add_argument(
        "--overwrite-indices",
        action="store_true",
        help="Rebuild derived index rasters even if existing paths already exist.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional output vector path with mean_ndvi, mean_ndbi, mean_albedo, and roof_area_m2.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        help="Optional CSV path for aggregate averages.",
    )
    parser.add_argument(
        "--area-crs",
        default="auto",
        help=(
            "CRS used for area weights. 'auto' estimates a local UTM CRS; "
            "fallback is EPSG:6933. Use explicit values like EPSG:25832 if needed."
        ),
    )
    parser.add_argument(
        "--all-touched",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include all raster pixels touched by a roof polygon (default: true).",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Drop roofs missing any requested raster mean before writing output and summaries.",
    )
    return parser


def _resolve_raster_paths(
    ndvi: Path | None,
    ndbi: Path | None,
    albedo: Path | None,
    indices_out_dir: Path | None,
) -> dict[str, Path]:
    """Resolve explicit and inferred paths for all required index rasters."""
    explicit = {"ndvi": ndvi, "ndbi": ndbi, "albedo": albedo}
    if all(path is not None for path in explicit.values()):
        return {name: path for name, path in explicit.items() if path is not None}
    if any(path is not None for path in explicit.values()) and indices_out_dir is None:
        missing = [name for name, path in explicit.items() if path is None]
        raise ValueError(
            "Provide all three raster paths or provide --indices-out-dir so missing paths "
            f"can be inferred. Missing: {', '.join(missing)}"
        )
    if indices_out_dir is None:
        raise ValueError("Provide --ndvi/--ndbi/--albedo or provide --indices-out-dir with --l2-folder.")
    defaults = {name: indices_out_dir / f"{name}.tif" for name in DEFAULT_RASTER_NAMES}
    return {name: explicit[name] or defaults[name] for name in DEFAULT_RASTER_NAMES}


def _derive_missing_indices(
    *,
    l2_folder: Path | None,
    rasters: dict[str, Path],
    indices_out_dir: Path | None,
    baseline_lst: Path | None,
    keep_lst_water: bool,
    mask_with_lst: bool,
    overwrite: bool,
) -> None:
    """Derive any missing index rasters from a Landsat Level-2 scene."""
    missing = [path for path in rasters.values() if overwrite or not path.exists()]
    if not missing:
        return
    if l2_folder is None:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Index rasters are missing. Provide existing --ndvi/--ndbi/--albedo paths, "
            f"or add --l2-folder to derive them. Missing: {missing_text}"
        )

    if baseline_lst is None:
        if indices_out_dir is not None:
            baseline_lst = indices_out_dir / "baseline_LST.tif"
        else:
            first_parent = next(iter(rasters.values())).parent
            baseline_lst = first_parent / "baseline_LST.tif"

    baseline_lst, lst, _ = build_lst_from_l2(
        l2_folder,
        out_path=baseline_lst,
        unit="celsius",
        keep_water=keep_lst_water,
    )
    ndvi, albedo, ndbi, profile = compute_ndvi_albedo_from_l2(l2_folder, baseline_lst)

    if mask_with_lst:
        valid = np.isfinite(lst)
        ndvi = ndvi.astype("float32")
        ndbi = ndbi.astype("float32")
        albedo = albedo.astype("float32")
        ndvi[~valid] = np.nan
        ndbi[~valid] = np.nan
        albedo[~valid] = np.nan

    for name, arr in {"ndvi": ndvi, "ndbi": ndbi, "albedo": albedo}.items():
        save_raster(rasters[name], arr.astype("float32"), profile)


def _read_roofs(path: Path, layer: str | None) -> gpd.GeoDataFrame:
    """Read and validate a non-empty green-roof vector layer."""
    gdf = gpd.read_file(path, layer=layer) if layer else gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"Green-roof layer is empty: {path}")
    gdf = gdf[gdf.geometry.notnull()].copy()
    if gdf.empty:
        raise ValueError(f"Green-roof layer has no valid geometries: {path}")
    if gdf.crs is None:
        raise ValueError(f"Green-roof layer has no CRS: {path}")
    return gdf


def _zonal_mean(gdf: gpd.GeoDataFrame, raster_path: Path, all_touched: bool) -> list[float]:
    """Calculate a raster mean for each polygon using windowed reads."""
    if not raster_path.exists():
        raise FileNotFoundError(f"Raster not found: {raster_path}")
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata
        projected = gdf.to_crs(raster_crs) if gdf.crs != raster_crs else gdf
        means: list[float] = []
        for geom in projected.geometry:
            if geom is None or geom.is_empty:
                means.append(float("nan"))
                continue
            try:
                window = geometry_window(src, [mapping(geom)])
            except WindowError:
                means.append(float("nan"))
                continue

            data = src.read(1, window=window, masked=True)
            if data.size == 0:
                means.append(float("nan"))
                continue

            geom_mask = geometry_mask(
                [mapping(geom)],
                out_shape=data.shape,
                transform=src.window_transform(window),
                invert=True,
                all_touched=all_touched,
            )
            arr = np.ma.filled(data, np.nan).astype("float32")
            vals = arr[geom_mask]
            if nodata is not None and not np.isnan(nodata):
                vals = vals[vals != nodata]
            vals = vals[np.isfinite(vals)]
            means.append(float(vals.mean()) if vals.size else float("nan"))
    return means


def _area_weights(gdf: gpd.GeoDataFrame, area_crs: str) -> np.ndarray:
    """Calculate polygon areas in an explicit or automatically selected CRS."""
    if area_crs == "auto":
        try:
            resolved = gdf.estimate_utm_crs()
        except RuntimeError:
            resolved = None
        if resolved is None:
            resolved = "EPSG:6933"
    else:
        resolved = area_crs
    return gdf.to_crs(resolved).geometry.area.to_numpy(dtype=float)


def _finite_pair(values: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return values paired with finite, positive area weights."""
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return values[mask], weights[mask]


def _summaries(gdf: gpd.GeoDataFrame, weights: np.ndarray) -> list[dict[str, float | int | str]]:
    """Summarize valid counts and weighted and unweighted index means."""
    rows: list[dict[str, float | int | str]] = []
    for name in DEFAULT_RASTER_NAMES:
        col = f"mean_{name}"
        vals = gdf[col].to_numpy(dtype=float)
        finite_vals = vals[np.isfinite(vals)]
        weighted_vals, weighted_weights = _finite_pair(vals, weights)
        weighted = float("nan")
        if weighted_weights.size and float(weighted_weights.sum()) > 0:
            weighted = float(np.average(weighted_vals, weights=weighted_weights))
        rows.append(
            {
                "parameter": name,
                "valid_roofs": int(finite_vals.size),
                "unweighted_mean": float(np.mean(finite_vals)) if finite_vals.size else float("nan"),
                "area_weighted_mean": weighted,
            }
        )
    return rows


def _format_float(value: float | int | str) -> str:
    """Format a summary value for stable command-line output."""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.6f}"


def _write_vector(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write a vector layer, selecting the GeoPackage driver when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"driver": "GPKG"} if path.suffix.lower() == ".gpkg" else {}
    gdf.to_file(path, **kwargs)


def _write_summary_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    """Write aggregate green-roof parameter rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["parameter", "valid_roofs", "unweighted_mean", "area_weighted_mean"],
        )
        writer.writeheader()
        writer.writerows(rows)


def compute_green_roof_parameters(
    *,
    green_roofs: Path,
    rasters: dict[str, Path],
    out: Path | None = None,
    summary_csv: Path | None = None,
    layer: str | None = None,
    area_crs: str = "auto",
    all_touched: bool = True,
    drop_missing: bool = False,
) -> tuple[gpd.GeoDataFrame, list[dict[str, float | int | str]]]:
    """Enrich green roofs with raster means and return aggregate summaries."""
    gdf = _read_roofs(green_roofs, layer)
    for name in DEFAULT_RASTER_NAMES:
        gdf[f"mean_{name}"] = _zonal_mean(gdf, rasters[name], all_touched)

    weights = _area_weights(gdf, area_crs)
    gdf["roof_area_m2"] = weights.astype("float64")

    if drop_missing:
        required = [f"mean_{name}" for name in DEFAULT_RASTER_NAMES]
        gdf = gdf.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
        weights = gdf["roof_area_m2"].to_numpy(dtype=float)

    rows = _summaries(gdf, weights)

    if out is not None:
        _write_vector(gdf, out)
    if summary_csv is not None:
        _write_summary_csv(rows, summary_csv)

    return gdf, rows


def main(argv: Sequence[str] | None = None) -> None:
    """Run the green-roof parameter estimation command-line workflow."""
    args = build_parser().parse_args(argv)
    rasters = _resolve_raster_paths(args.ndvi, args.ndbi, args.albedo, args.indices_out_dir)
    _derive_missing_indices(
        l2_folder=args.l2_folder,
        rasters=rasters,
        indices_out_dir=args.indices_out_dir,
        baseline_lst=args.baseline_lst,
        keep_lst_water=args.keep_lst_water,
        mask_with_lst=args.mask_with_lst,
        overwrite=args.overwrite_indices,
    )
    _, rows = compute_green_roof_parameters(
        green_roofs=args.green_roofs,
        rasters=rasters,
        out=args.out,
        summary_csv=args.summary_csv,
        layer=args.layer,
        area_crs=args.area_crs,
        all_touched=args.all_touched,
        drop_missing=args.drop_missing,
    )

    print("Green-roof parameter averages")
    print("parameter,valid_roofs,unweighted_mean,area_weighted_mean")
    for row in rows:
        print(
            ",".join(
                [
                    str(row["parameter"]),
                    str(row["valid_roofs"]),
                    _format_float(row["unweighted_mean"]),
                    _format_float(row["area_weighted_mean"]),
                ]
            )
        )
    if args.out:
        print(f"Wrote enriched roofs to {args.out}")
    if args.summary_csv:
        print(f"Wrote summary CSV to {args.summary_csv}")
    print(
        "Index rasters used: "
        + ", ".join(f"{name}={path}" for name, path in rasters.items())
    )


if __name__ == "__main__":
    main()
