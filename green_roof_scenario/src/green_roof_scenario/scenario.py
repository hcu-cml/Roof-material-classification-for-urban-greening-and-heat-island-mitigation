"""High-level orchestration for the green roof scenario workflow."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
from rasterstats import zonal_stats
from rasterio.io import MemoryFile
from rasterio.mask import mask
from shapely.geometry import box, mapping

from .config import ScenarioConfig
from .io import read_raster, save_raster
from .l2 import build_lst_from_l2, compute_ndvi_albedo_from_l2
from .masking import roof_mask_fraction, subset_buildings
from .modeling import fit_model, predict_model, predict_partial

logger = logging.getLogger(__name__)


@dataclass
class ScenarioOutputs:
    out_dir: Path
    scenario_raster: Path
    delta_raster: Path
    buildings_layer: Path
    stats_report: Path | None = None
    baseline_pred_raster: Path | None = None
    roof_fraction_raster: Path | None = None
    feature_importance: Path | None = None


def _load_boundary(boundary_path: Path, target_crs) -> tuple[list[dict], gpd.GeoDataFrame]:
    """Load, validate, and reproject an analysis boundary."""
    boundary = gpd.read_file(boundary_path)
    if boundary.empty:
        raise ValueError(f"Boundary layer {boundary_path} is empty.")
    boundary = boundary[boundary.geometry.notnull()].copy()
    boundary = boundary[~boundary.geometry.is_empty].copy()
    if boundary.crs is None:
        raise ValueError(f"Boundary layer {boundary_path} has no CRS.")
    boundary = boundary.to_crs(target_crs)
    geom = boundary.geometry.unary_union
    if geom.is_empty:
        raise ValueError(f"Boundary layer {boundary_path} has no geometry after reprojecting to the raster CRS.")
    return [mapping(geom)], boundary


def _building_extent_boundary(buildings: gpd.GeoDataFrame, target_crs) -> tuple[list[dict], gpd.GeoDataFrame]:
    """Create a rectangular analysis boundary from building bounds."""
    if buildings.empty:
        raise ValueError("Cannot derive an extent from an empty building layer.")
    extent = box(*buildings.total_bounds)
    if extent.is_empty:
        raise ValueError("Cannot derive a valid extent from the building layer.")
    boundary = gpd.GeoDataFrame([{"source": "building_extent"}], geometry=[extent], crs=target_crs)
    return [mapping(extent)], boundary


def _geometry_area_m2(buildings: gpd.GeoDataFrame) -> np.ndarray:
    """Return geometry areas in square metres, independent of the input CRS."""
    if buildings.crs is None:
        raise ValueError(
            "Building layer has no CRS; square-metre roof areas cannot be calculated."
        )
    try:
        area_crs = buildings.estimate_utm_crs()
    except (RuntimeError, ValueError):
        area_crs = None
    if area_crs is None:
        area_crs = "EPSG:6933"
    return buildings.to_crs(area_crs).geometry.area.to_numpy(dtype="float64")


def _clip_raster_to_boundary(arr: np.ndarray, profile: dict, geoms: list[dict]) -> tuple[np.ndarray, dict]:
    """Clip an in-memory raster array and update its output profile."""
    tmp_profile = profile.copy()
    tmp_profile.setdefault("driver", "GTiff")
    tmp_profile.update(count=1, dtype="float32", nodata=np.nan)
    with MemoryFile() as memfile:
        with memfile.open(**tmp_profile) as ds:
            ds.write(arr.astype("float32"), 1)
            clipped, out_transform = mask(ds, geoms, crop=True, filled=False)
            out_profile = ds.profile.copy()
            out_profile.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=out_transform,
                nodata=np.nan,
                dtype="float32",
                count=1,
            )
    clipped_arr = np.ma.filled(clipped, np.nan).astype("float32")[0]
    return clipped_arr, out_profile


def run_scenario(config: ScenarioConfig) -> ScenarioOutputs:
    """Execute the complete green-roof intervention scenario workflow."""
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    lst_path = config.lst
    if config.build_lst and lst_path is None:
        lst_path = out_dir / "baseline_LST.tif"

    if config.build_lst:
        lst_path, lst, lst_profile = build_lst_from_l2(
            config.l2_folder,
            out_path=lst_path,
            unit=config.lst_unit,
            keep_water=config.keep_lst_water,
        )
    else:
        if lst_path is None:
            raise ValueError("Provide lst path or enable build_lst to derive it from the Landsat L2 folder.")
        lst, lst_profile = read_raster(lst_path)

    template_profile = lst_profile
    if template_profile.get("crs") is None:
        raise ValueError(f"Baseline LST raster {lst_path} has no CRS.")

    if config.layer:
        buildings = gpd.read_file(config.buildings, layer=config.layer)
    else:
        buildings = gpd.read_file(config.buildings)
    if buildings.empty:
        raise ValueError(f"Building layer {config.buildings} is empty.")
    buildings = buildings[buildings.geometry.notnull()].copy()
    buildings = buildings[~buildings.geometry.is_empty].copy()
    if buildings.empty:
        raise ValueError(f"Building layer {config.buildings} has no valid geometries.")
    if buildings.crs is None:
        raise ValueError(f"Building layer {config.buildings} has no CRS.")
    if buildings.crs != template_profile["crs"]:
        buildings = buildings.to_crs(template_profile["crs"])

    boundary_geoms: list[dict] | None = None
    boundary_gdf: gpd.GeoDataFrame | None = None
    clip_source = "building_extent"
    if config.boundary:
        boundary_geoms, boundary_gdf = _load_boundary(config.boundary, template_profile["crs"])
        clip_source = str(config.boundary)
    else:
        boundary_geoms, boundary_gdf = _building_extent_boundary(buildings, template_profile["crs"])

    if boundary_gdf is not None:
        buildings = gpd.clip(buildings, boundary_gdf)
        buildings = buildings[buildings.geometry.notnull()].copy()
        if buildings.empty:
            raise ValueError("No buildings intersect the analysis extent.")

    bld_green = subset_buildings(
        buildings,
        config.roof_material_field,
        config.roof_materials_type,
        roof_shape_field=config.roof_shape_field,
        roof_shape_type=config.roof_shape_type,
        roof_slope_field=config.roof_slope_field,
        max_roof_slope_deg=config.max_roof_slope_deg,
        keep_null_roof=config.keep_null_roof,
    )
    if config.min_roof_area > 0:
        roof_areas = _geometry_area_m2(bld_green)
        bld_green = bld_green.loc[roof_areas >= config.min_roof_area].copy()

    if bld_green.empty:
        raise ValueError("No buildings match the requested roof types to green.")

    ndvi, albedo, ndbi, _ = compute_ndvi_albedo_from_l2(config.l2_folder, lst_path)

    if boundary_geoms is not None:
        base_profile = template_profile.copy()
        lst, template_profile = _clip_raster_to_boundary(lst, base_profile, boundary_geoms)
        ndvi, _ = _clip_raster_to_boundary(ndvi, base_profile, boundary_geoms)
        albedo, _ = _clip_raster_to_boundary(albedo, base_profile, boundary_geoms)
        ndbi, _ = _clip_raster_to_boundary(ndbi, base_profile, boundary_geoms)
        if config.build_lst:
            save_raster(lst_path, lst, template_profile)

    valid_lsts = np.isfinite(lst)
    ndvi = np.where(valid_lsts, ndvi, np.nan).astype("float32")
    albedo = np.where(valid_lsts, albedo, np.nan).astype("float32")
    ndbi = np.where(valid_lsts, ndbi, np.nan).astype("float32")

    if config.write_indices_rasters:
        save_raster(out_dir / "ndvi.tif", ndvi.astype("float32"), template_profile)
        save_raster(out_dir / "albedo.tif", albedo.astype("float32"), template_profile)
        save_raster(out_dir / "ndbi.tif", ndbi.astype("float32"), template_profile)

    if config.target_ndvi is None:
        valid = np.isfinite(ndvi) & np.isfinite(lst)
        veg = ndvi[valid & (ndvi > 0.3)]
        target_ndvi = float(np.median(veg)) if veg.size else 0.5
    else:
        target_ndvi = float(config.target_ndvi)
    target_albedo = float(config.target_albedo)
    target_ndbi = float(config.target_ndbi)

    transform = template_profile["transform"]
    pixel_size = float(np.sqrt(abs(transform.a * transform.e - transform.b * transform.d)))
    if pixel_size <= 0:
        raise ValueError(
            "The baseline LST raster has an invalid transform with zero pixel area."
        )
    if config.min_sample_spacing > 0:
        block_size = max(1, int(round(config.min_sample_spacing / pixel_size)))
    else:
        block_size = None

    model, metrics = fit_model(
        lst,
        ndvi,
        albedo,
        ndbi,
        frac=config.sample_frac,
        seed=config.random_state,
        model_type=config.model,
        block_size=block_size,
    )
    logger.info(
        "Model fitted (type=%s): R2 train=%.3f test=%.3f, RMSE train=%.3f test=%.3f",
        config.model,
        metrics["r2_train"],
        metrics["r2_test"],
        metrics["rmse_train"],
        metrics["rmse_test"],
    )
    feat_report = out_dir / "model_feature_importance.txt"
    feature_importance_path: Path | None = None
    report_lines = [
        f"Model: {config.model}",
        f"R2 train: {metrics['r2_train']:.6f}",
        f"R2 test: {metrics['r2_test']:.6f}",
        f"RMSE train: {metrics['rmse_train']:.6f}",
        f"RMSE test: {metrics['rmse_test']:.6f}",
        "",
    ]
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
        names = ["ndvi", "albedo", "ndbi"]
        lines = [f"{n}: {v:.4f}" for n, v in zip(names, vals)]
        report_lines.extend(["Random Forest feature importances (sum=1):", *lines])
        feat_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        feature_importance_path = feat_report
        logger.info("Feature importances saved to %s", feat_report)
    elif hasattr(model, "coef_"):
        vals = model.coef_
        names = ["ndvi", "albedo", "ndbi"]
        lines = [f"{n}: {v:.4f}" for n, v in zip(names, vals)]
        report_lines.extend(["Linear model coefficients:", *lines])
        feat_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        feature_importance_path = feat_report
        logger.info("Linear coefficients saved to %s", feat_report)

    baseline_pred = predict_model(model, ndvi, albedo, ndbi)

    baseline_pred_path = None
    if config.write_pred_baseline:
        baseline_pred_path = out_dir / "baseline_pred_LST.tif"
        save_raster(baseline_pred_path, baseline_pred, template_profile)

    roof_frac = roof_mask_fraction(
        bld_green,
        template_profile,
        supersample=config.supersample,
        all_touched=config.all_touched,
    )
    roof_fraction_raster = None
    if config.write_roof_fraction_raster:
        roof_fraction_raster = out_dir / "roof_fraction.tif"
        save_raster(roof_fraction_raster, roof_frac.astype("float32"), template_profile)

    f = np.clip(roof_frac, 0.0, 1.0).astype("float32")
    valid_inputs = np.isfinite(ndvi) & np.isfinite(albedo) & np.isfinite(ndbi)
    pred_mask = (f > 0) & valid_lsts & valid_inputs

    scen_ndvi = (1.0 - f) * ndvi + f * target_ndvi
    scen_albedo = (1.0 - f) * albedo + f * target_albedo
    scen_ndbi = (1.0 - f) * ndbi + f * target_ndbi

    scen_pred = predict_partial(model, scen_ndvi, scen_albedo, pred_mask, scen_ndbi)

    # Subtract the modeled baseline to isolate the intervention and cancel model bias.
    delta = np.zeros(ndvi.shape, dtype="float32")
    mfin = pred_mask & np.isfinite(scen_pred) & np.isfinite(baseline_pred)
    delta[mfin] = scen_pred[mfin] - baseline_pred[mfin]

    if config.clip_positive_delta:
        delta[(delta > 0) & np.isfinite(delta)] = 0.0

    # Apply only the modeled intervention delta to observed LST to avoid model-bias seams.
    scen_pred_filled = lst.copy()
    scen_pred_filled[mfin] = lst[mfin] + delta[mfin]

    # Handle NaNs for output
    delta[~valid_lsts] = np.nan
    scen_pred_filled[~valid_lsts] = np.nan

    scenario_raster = out_dir / "scenario_pred_LST.tif"
    delta_raster = out_dir / "delta_LST.tif"
    save_raster(scenario_raster, scen_pred_filled, template_profile)
    save_raster(delta_raster, delta, template_profile)

    # City-wide raster statistic: mean cooling (baseline LST - scenario LST)
    stats_report_path: Path | None = None
    valid_city_mask = np.isfinite(lst) & np.isfinite(scen_pred_filled)
    city_mean_cooling = float("nan")
    if np.any(valid_city_mask):
        city_mean_cooling = float(np.nanmean(lst[valid_city_mask] - scen_pred_filled[valid_city_mask]))
    else:
        logger.warning("No valid pixels available to compute city-wide mean cooling.")

    baseline_stats = zonal_stats(
        buildings,
        lst,
        affine=template_profile["transform"],
        nodata=np.nan,
        stats=["mean"],
        all_touched=True,
    )
    scenario_stats = zonal_stats(
        buildings,
        scen_pred_filled,
        affine=template_profile["transform"],
        nodata=np.nan,
        stats=["mean"],
        all_touched=True,
    )
    delta_stats = zonal_stats(
        buildings,
        delta,
        affine=template_profile["transform"],
        nodata=np.nan,
        stats=["mean"],
        all_touched=True,
    )
    buildings = buildings.copy()
    buildings["lst_baseline_mean"] = [z["mean"] for z in baseline_stats]
    buildings["lst_scenario_mean"] = [z["mean"] for z in scenario_stats]
    buildings["delta_mean"] = [z["mean"] for z in delta_stats]
    buildings["cooling_mean"] = buildings["lst_baseline_mean"] - buildings["lst_scenario_mean"]
    buildings["roof_area_m2"] = _geometry_area_m2(buildings)
    buildings["selected_for_greening"] = buildings.index.isin(bld_green.index)

    cooling_valid = buildings[np.isfinite(buildings["cooling_mean"])].copy()
    cooling_valid = cooling_valid[np.isfinite(cooling_valid["roof_area_m2"])]

    def _aggregate_stats(df: gpd.GeoDataFrame) -> tuple[float, float, float, str | None]:
        """Calculate mean, area-weighted mean, and maximum building cooling."""
        if df.empty:
            return float("nan"), float("nan"), float("nan"), None
        avg = float(df["cooling_mean"].mean())
        total_area = float(df["roof_area_m2"].sum())
        weighted = float("nan")
        if total_area > 0:
            weighted = float(np.average(df["cooling_mean"], weights=df["roof_area_m2"]))
        max_row = df["cooling_mean"].idxmax()
        max_val = float(df.loc[max_row, "cooling_mean"])
        return avg, weighted, max_val, str(max_row)

    all_avg, all_weighted, all_max, all_max_label = _aggregate_stats(cooling_valid)

    selected = cooling_valid.loc[cooling_valid["selected_for_greening"]].copy()
    sel_avg, sel_weighted, sel_max, sel_max_label = _aggregate_stats(selected)

    buildings_layer = out_dir / "buildings_greening_impact.gpkg"
    buildings.to_file(buildings_layer, driver="GPKG")

    # Write a concise statistics report
    def _fmt(val: float) -> str:
        """Format a finite statistic or return a stable NaN marker."""
        return "nan" if val != val or np.isinf(val) else f"{val:.4f}"

    stats_lines = [
        "Greening statistics",
        "-------------------",
        f"Raster mean (baseline - scenario): {_fmt(city_mean_cooling)} °C",
        f"Buildings in analysis: {len(buildings)}",
        f"Buildings selected for greening: {len(bld_green)}",
        "All buildings (city-wide dilution/impact):",
        f"  Average cooling: {_fmt(all_avg)} °C",
        f"  Area-weighted average cooling: {_fmt(all_weighted)} °C",
        f"  Max cooling: {_fmt(all_max)} °C" + (f" (id={all_max_label})" if all_max_label else ""),
        "Buildings selected for greening (with valid statistics):",
        f"  Average cooling: {_fmt(sel_avg)} °C",
        f"  Area-weighted average cooling: {_fmt(sel_weighted)} °C",
        f"  Max cooling: {_fmt(sel_max)} °C" + (f" (id={sel_max_label})" if sel_max_label else ""),
    ]
    stats_report_path = out_dir / "_greening_statistics.txt"
    stats_report_path.write_text("\n".join(stats_lines), encoding="utf-8")

    logger.info(
        "Cooling stats (°C) | Raster mean (baseline - scenario)=%s | Avg all=%s, weighted all=%s, max all=%s | "
        "Avg selected=%s, weighted selected=%s, max selected=%s",
        _fmt(city_mean_cooling),
        _fmt(all_avg),
        _fmt(all_weighted),
        _fmt(all_max),
        _fmt(sel_avg),
        _fmt(sel_weighted),
        _fmt(sel_max),
    )

    provenance = (
        f"Roof material field/types: {config.roof_material_field} / {config.roof_materials_type}\n"
        f"Roof shape field/types: {config.roof_shape_field} / {config.roof_shape_type}\n"
        f"Roof slope field/max deg: {config.roof_slope_field} / {config.max_roof_slope_deg}\n"
        f"Analysis extent: {clip_source}\n"
        f"Landsat L2 folder: {config.l2_folder}\n"
        f"Target NDVI: {target_ndvi}\n"
        f"Target Albedo: {target_albedo}\n"
        f"Target NDBI: {target_ndbi}\n"
        f"LST source: {'built from L2 folder' if config.build_lst else lst_path}\n"
        f"Supersample: {config.supersample}\n"
        f"Model type: {config.model}\n"
        f"Sample fraction: {config.sample_frac}\n"
        f"Minimum sample spacing: {config.min_sample_spacing}\n"
        f"Random state: {config.random_state}\n"
        f"R2 train/test: {metrics['r2_train']:.6f} / {metrics['r2_test']:.6f}\n"
        f"RMSE train/test: {metrics['rmse_train']:.6f} / {metrics['rmse_test']:.6f}\n"
        f"Buildings analyzed/selected: {len(buildings)} / {len(bld_green)}\n"
    )
    if config.build_lst:
        provenance += f"LST build options: unit={config.lst_unit}, keep_water={config.keep_lst_water}\n"

    provenance_path = out_dir / "_greening_provenance.txt"
    provenance_path.write_text(provenance, encoding="utf-8")

    logger.info("Scenario written to %s", out_dir)

    return ScenarioOutputs(
        out_dir=out_dir,
        scenario_raster=scenario_raster,
        delta_raster=delta_raster,
        buildings_layer=buildings_layer,
        stats_report=stats_report_path,
        baseline_pred_raster=baseline_pred_path,
        roof_fraction_raster=roof_fraction_raster,
        feature_importance=feature_importance_path,
    )
