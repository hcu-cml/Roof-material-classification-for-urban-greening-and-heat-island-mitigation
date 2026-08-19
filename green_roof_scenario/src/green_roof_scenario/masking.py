"""Vector/raster masking helpers."""

from __future__ import annotations

import ast
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import mapping

__all__ = ["roof_mask_fraction", "subset_buildings"]


def roof_mask_fraction(
    buildings_gdf: gpd.GeoDataFrame,
    template_profile: dict,
    *,
    supersample: int = 4,
    all_touched: bool = False,
) -> np.ndarray:
    """Rasterize buildings into per-pixel roof coverage fractions."""
    if supersample < 1:
        raise ValueError("supersample must be >= 1.")
    shapes = [
        mapping(geom)
        for geom in buildings_gdf.geometry.values
        if geom is not None and not geom.is_empty
    ]
    if not shapes:
        raise ValueError("No non-empty building geometries are available for rasterization.")
    height, width = template_profile["height"], template_profile["width"]
    transform = template_profile["transform"]

    if all_touched or supersample <= 1:
        mask = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,
            default_value=1,
            all_touched=all_touched,
        ).astype("float32")
        return mask

    hss, wss = height * supersample, width * supersample
    a, b, c, d, e, f = transform[:6]
    up_transform = rasterio.Affine(
        a / supersample,
        b / supersample,
        c,
        d / supersample,
        e / supersample,
        f,
    )
    up = rasterize(
        shapes,
        out_shape=(hss, wss),
        transform=up_transform,
        fill=0,
        default_value=1,
        all_touched=True,
    ).astype("float32")
    up = up.reshape(height, supersample, width, supersample).mean(axis=(1, 3))
    return up


def subset_buildings(
    gdf: gpd.GeoDataFrame,
    roof_material_field: str | None,
    roof_materials_type: str | None,
    *,
    roof_shape_field: str | None = None,
    roof_shape_type: str | None = None,
    roof_slope_field: str | None = None,
    max_roof_slope_deg: float | None = None,
    keep_null_roof: bool = False,
) -> gpd.GeoDataFrame:
    """Select buildings matching the requested material, shape, and slope filters."""
    material_filter = bool(roof_materials_type and roof_materials_type.strip())
    shape_filter = bool(roof_shape_type and roof_shape_type.strip())
    slope_filter = max_roof_slope_deg is not None

    if not material_filter and not shape_filter and not slope_filter:
        raise ValueError("Provide at least one roof material, shape, or slope filter to select roofs.")

    df = gdf.copy()
    mask = np.ones(len(df), dtype=bool)

    if material_filter:
        if not roof_material_field:
            raise ValueError("roof_material_field must be provided when roof_materials_type is set.")
        if roof_material_field not in df.columns:
            raise KeyError(f"Roof material field '{roof_material_field}' not in buildings.")
        normalized_mat = df[roof_material_field].apply(_normalize_roof_value)
        wanted_mat = {t.strip().lower() for t in roof_materials_type.split(",") if t.strip()}
        mat_mask = normalized_mat.isin(wanted_mat)
        if keep_null_roof:
            mat_mask |= normalized_mat.isna()
        mask &= mat_mask

    if shape_filter:
        if not roof_shape_field:
            raise ValueError("roof_shape_field must be provided when roof_shape_type is set.")
        if roof_shape_field not in df.columns:
            raise KeyError(f"Roof shape field '{roof_shape_field}' not in buildings.")
        normalized_shape = df[roof_shape_field].apply(_normalize_roof_value)
        wanted_shape = {t.strip().lower() for t in roof_shape_type.split(",") if t.strip()}
        shape_mask = normalized_shape.isin(wanted_shape)
        if keep_null_roof:
            shape_mask |= normalized_shape.isna()
        mask &= shape_mask

    if slope_filter:
        if not roof_slope_field:
            raise ValueError("roof_slope_field must be provided when max_roof_slope_deg is set.")
        if roof_slope_field not in df.columns:
            raise KeyError(f"Roof slope field '{roof_slope_field}' not in buildings.")
        normalized_slope = df[roof_slope_field].apply(_normalize_numeric_value)
        slope_mask = normalized_slope.notna() & (normalized_slope <= float(max_roof_slope_deg))
        mask &= slope_mask

    return df.loc[mask].copy()


def _normalize_roof_value(value: object) -> str | None:
    """Normalize scalar or container-like roof attributes for comparison."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return _normalize_roof_value(parsed)
        value = stripped
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
    if pd.isna(value):
        return None
    return str(value).strip().lower()


def _normalize_numeric_value(value: object) -> float | None:
    """Convert scalar or container-like attributes to a finite-style float value."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple, np.ndarray)):
                return _normalize_numeric_value(parsed)
        value = stripped
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.flat[0]
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
