"""Enrich building footprints with roof slope metrics from CityGML LoD2 files."""

from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from xml.etree.ElementTree import Element, iterparse

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BLDG_NS = "{http://www.opengis.net/citygml/building/1.0}"
GML_NS = "{http://www.opengis.net/gml}"
GML_ID_ATTR = "{http://www.opengis.net/gml}id"

DEFAULT_SLOPE_FIELD = "roof_slope_mean_deg"
DEFAULT_LOW_SLOPE_THRESHOLD_DEG = 15.0


@dataclass(frozen=True)
class RoofSlopeStats:
    gml_id: str
    roof_slope_mean_deg: float
    roof_slope_min_deg: float
    roof_slope_max_deg: float
    roof_plane_count: int
    roof_area_2d_m2: float
    roof_area_share_le_15deg: float
    citygml_roof_type: str | None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for CityGML slope enrichment."""
    parser = argparse.ArgumentParser(
        description="Add area-weighted CityGML roof slope metrics to a building vector layer."
    )
    parser.add_argument("--buildings", required=True, type=Path, help="Input building GPKG/GeoJSON/shapefile.")
    parser.add_argument("--citygml-dir", required=True, type=Path, help="Directory containing CityGML .gml tiles.")
    parser.add_argument("--out", required=True, type=Path, help="Output vector file. The input is not modified.")
    parser.add_argument("--layer", default=None, help="Optional input layer name, for example 'buildings'.")
    parser.add_argument(
        "--output-layer",
        default="buildings",
        help="Output layer name when writing a GeoPackage (default: buildings).",
    )
    parser.add_argument("--id-field", default="gml_id", help="Building ID field to join on (default: gml_id).")
    parser.add_argument(
        "--gml-pattern",
        default="*.gml",
        help="Glob pattern for CityGML files inside --citygml-dir (default: *.gml).",
    )
    parser.add_argument(
        "--replace-existing-fields",
        action="store_true",
        help="Replace existing slope enrichment fields in the input layer if present.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Overwrite --out if it already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CityGML slope-enrichment command-line workflow."""
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    enrich_buildings_with_citygml_slopes(
        buildings_path=args.buildings,
        citygml_dir=args.citygml_dir,
        out_path=args.out,
        layer=args.layer,
        output_layer=args.output_layer,
        id_field=args.id_field,
        gml_pattern=args.gml_pattern,
        replace_existing_fields=args.replace_existing_fields,
        overwrite_output=args.overwrite_output,
    )


def enrich_buildings_with_citygml_slopes(
    *,
    buildings_path: Path,
    citygml_dir: Path,
    out_path: Path,
    layer: str | None = None,
    output_layer: str = "buildings",
    id_field: str = "gml_id",
    gml_pattern: str = "*.gml",
    replace_existing_fields: bool = False,
    overwrite_output: bool = False,
) -> Path:
    """Join CityGML roof-slope statistics onto a building vector layer."""
    buildings_path = Path(buildings_path)
    citygml_dir = Path(citygml_dir)
    out_path = Path(out_path)

    if not buildings_path.exists():
        raise FileNotFoundError(f"Building file does not exist: {buildings_path}")
    if not citygml_dir.exists():
        raise FileNotFoundError(f"CityGML directory does not exist: {citygml_dir}")
    if out_path.exists() and not overwrite_output:
        raise FileExistsError(f"Output already exists: {out_path}. Pass --overwrite-output to replace it.")
    logger.info("Reading buildings from %s", buildings_path)
    buildings = gpd.read_file(buildings_path, layer=layer) if layer else gpd.read_file(buildings_path)
    if buildings.empty:
        raise ValueError(f"Building layer {buildings_path} is empty.")
    if id_field not in buildings.columns:
        raise KeyError(f"ID field '{id_field}' not found in buildings.")

    target_ids = _target_ids(buildings[id_field])
    if not target_ids:
        raise ValueError(f"No non-null IDs found in field '{id_field}'.")

    gml_files = sorted(citygml_dir.glob(gml_pattern))
    if not gml_files:
        raise FileNotFoundError(f"No CityGML files matching '{gml_pattern}' found in {citygml_dir}.")

    logger.info("Extracting slopes for %s target IDs from %s CityGML files", len(target_ids), len(gml_files))
    stats = extract_citygml_roof_slope_stats(
        gml_files,
        target_ids=target_ids,
    )
    if not stats:
        raise ValueError("No matching CityGML roof slope records were extracted.")

    stats_df = pd.DataFrame([s.__dict__ for s in stats.values()])
    output = _merge_stats(
        buildings,
        stats_df,
        id_field=id_field,
        replace_existing_fields=replace_existing_fields,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and overwrite_output:
        out_path.unlink()
    logger.info("Writing enriched buildings to %s", out_path)
    if out_path.suffix.lower() == ".gpkg":
        output.to_file(out_path, layer=output_layer, driver="GPKG")
    else:
        output.to_file(out_path)

    matched_count = int(output[DEFAULT_SLOPE_FIELD].notna().sum())
    missing_count = len(output) - matched_count
    logger.info(
        "Done. Enriched %s/%s buildings; %s missing usable CityGML roof slope.",
        matched_count,
        len(output),
        missing_count,
    )
    return out_path


def extract_citygml_roof_slope_stats(
    gml_files: Iterable[Path],
    *,
    target_ids: set[str],
    low_slope_threshold_deg: float = DEFAULT_LOW_SLOPE_THRESHOLD_DEG,
) -> dict[str, RoofSlopeStats]:
    """Extract slope statistics for target building IDs from CityGML files."""
    stats: dict[str, RoofSlopeStats] = {}
    remaining = set(target_ids)
    duplicate_count = 0
    files = list(gml_files)

    for file_index, path in enumerate(files, start=1):
        for _event, elem in iterparse(path, events=("end",)):
            if elem.tag != BLDG_NS + "Building":
                continue
            gml_id = elem.attrib.get(GML_ID_ATTR)
            if gml_id in remaining:
                extracted = _extract_building_slope_stats(
                    elem,
                    gml_id=gml_id,
                    low_slope_threshold_deg=low_slope_threshold_deg,
                )
                if extracted is not None:
                    stats[gml_id] = extracted
                remaining.discard(gml_id)
            elif gml_id in stats:
                duplicate_count += 1
            elem.clear()

        if file_index % 50 == 0 or file_index == len(files):
            logger.info(
                "Processed %s/%s CityGML files; matched %s; remaining %s",
                file_index,
                len(files),
                len(stats),
                len(remaining),
            )
        if not remaining:
            logger.info("All target IDs have been encountered; stopping early at %s/%s files", file_index, len(files))
            break

    if duplicate_count:
        logger.warning("Encountered %s duplicate target building IDs in CityGML; kept first occurrence.", duplicate_count)
    if remaining:
        logger.warning("%s target IDs were not found or had no usable roof surfaces.", len(remaining))
    return stats


def _extract_building_slope_stats(
    building: Element,
    *,
    gml_id: str,
    low_slope_threshold_deg: float,
) -> RoofSlopeStats | None:
    """Summarize usable roof planes for one CityGML building element."""
    roof_type = _child_text(building, BLDG_NS + "roofType")
    slopes: list[float] = []
    areas: list[float] = []

    for roof_surface in building.findall(".//" + BLDG_NS + "RoofSurface"):
        for polygon in roof_surface.findall(".//" + GML_NS + "Polygon"):
            exterior = polygon.find(GML_NS + "exterior")
            if exterior is None:
                continue
            exterior_coords = _ring_coords(exterior)
            if exterior_coords is None:
                continue
            slope = _slope_deg(exterior_coords)
            if slope is None:
                continue
            area = _polygon_area_2d_m2(polygon, exterior_coords)
            if area <= 0:
                continue
            slopes.append(slope)
            areas.append(area)

    if not slopes:
        return None

    slope_arr = np.asarray(slopes, dtype=float)
    area_arr = np.asarray(areas, dtype=float)
    total_area = float(area_arr.sum())
    if total_area <= 0:
        return None

    area_weighted_mean = float(np.average(slope_arr, weights=area_arr))
    low_slope_area = float(area_arr[slope_arr <= low_slope_threshold_deg].sum())
    return RoofSlopeStats(
        gml_id=gml_id,
        roof_slope_mean_deg=area_weighted_mean,
        roof_slope_min_deg=float(slope_arr.min()),
        roof_slope_max_deg=float(slope_arr.max()),
        roof_plane_count=int(len(slope_arr)),
        roof_area_2d_m2=total_area,
        roof_area_share_le_15deg=float(low_slope_area / total_area),
        citygml_roof_type=roof_type,
    )


def _target_ids(values: pd.Series) -> set[str]:
    """Normalize non-null building identifiers into a lookup set."""
    ids = values.dropna().astype(str).str.strip()
    return {value for value in ids if value}


def _merge_stats(
    buildings: gpd.GeoDataFrame,
    stats_df: pd.DataFrame,
    *,
    id_field: str,
    replace_existing_fields: bool,
) -> gpd.GeoDataFrame:
    """Merge extracted slope fields into buildings while handling collisions."""
    enrichment_fields = [
        "roof_slope_mean_deg",
        "roof_slope_min_deg",
        "roof_slope_max_deg",
        "roof_plane_count",
        "roof_area_2d_m2",
        "roof_area_share_le_15deg",
        "citygml_roof_type",
    ]
    existing = [field for field in enrichment_fields if field in buildings.columns]
    if existing and not replace_existing_fields:
        raise ValueError(
            "Input already contains enrichment fields "
            f"{existing}. Pass --replace-existing-fields to replace them."
        )

    base = buildings.drop(columns=existing) if existing else buildings.copy()
    stats_df = stats_df.rename(columns={"gml_id": id_field})
    stats_df[id_field] = stats_df[id_field].astype(str)
    base[id_field] = base[id_field].astype(str)
    merged = base.merge(stats_df[[id_field, *enrichment_fields]], on=id_field, how="left")
    merged["roof_plane_count"] = merged["roof_plane_count"].astype("Int64")
    return merged


def _child_text(elem: Element, tag: str) -> str | None:
    """Return stripped text from a direct child element when present."""
    child = elem.find(tag)
    if child is None or child.text is None:
        return None
    text = child.text.strip()
    return text or None


def _ring_coords(ring_parent: Element) -> np.ndarray | None:
    """Parse a GML ring into an array of three-dimensional coordinates."""
    pos_list = ring_parent.find(".//" + GML_NS + "posList")
    if pos_list is not None and pos_list.text:
        coords = _coords_from_pos_list(pos_list)
        if coords is not None:
            return coords

    pos_elems = ring_parent.findall(".//" + GML_NS + "pos")
    if not pos_elems:
        return None
    rows = []
    for pos in pos_elems:
        vals = np.fromstring(pos.text or "", sep=" ", dtype=float)
        if vals.size >= 3:
            rows.append(vals[:3])
    if len(rows) < 3:
        return None
    return _without_closing_point(np.asarray(rows, dtype=float))


def _coords_from_pos_list(pos_list: Element) -> np.ndarray | None:
    """Parse a GML posList element into valid three-dimensional vertices."""
    vals = np.fromstring(pos_list.text or "", sep=" ", dtype=float)
    if vals.size < 9:
        return None
    dim = _srs_dimension(pos_list, vals.size)
    if dim < 3:
        return None
    vals = vals[: vals.size - (vals.size % dim)]
    if vals.size < dim * 3:
        return None
    coords = vals.reshape((-1, dim))[:, :3]
    if len(coords) < 3:
        return None
    return _without_closing_point(coords)


def _srs_dimension(elem: Element, value_count: int) -> int:
    """Resolve coordinate dimensionality from metadata or value count."""
    raw = elem.attrib.get("srsDimension")
    if raw:
        try:
            dim = int(raw)
        except ValueError:
            dim = 0
        if dim > 0:
            return dim
    if value_count % 3 == 0:
        return 3
    if value_count % 2 == 0:
        return 2
    return 3


def _without_closing_point(coords: np.ndarray) -> np.ndarray | None:
    """Remove a duplicated closing vertex and reject undersized rings."""
    if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
        coords = coords[:-1]
    return coords if len(coords) >= 3 else None


def _polygon_area_2d_m2(polygon: Element, exterior_coords: np.ndarray) -> float:
    """Calculate horizontal polygon area after subtracting interior rings."""
    area = _ring_area_2d_m2(exterior_coords)
    for interior in polygon.findall(GML_NS + "interior"):
        interior_coords = _ring_coords(interior)
        if interior_coords is not None:
            area -= _ring_area_2d_m2(interior_coords)
    return max(float(area), 0.0)


def _ring_area_2d_m2(coords: np.ndarray) -> float:
    """Calculate the absolute horizontal area of a coordinate ring."""
    x = coords[:, 0]
    y = coords[:, 1]
    return abs(0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _slope_deg(coords: np.ndarray) -> float | None:
    """Calculate a polygon plane's slope from horizontal in degrees."""
    normal = np.zeros(3, dtype=float)
    for p, q in zip(coords, np.roll(coords, -1, axis=0)):
        normal[0] += (p[1] - q[1]) * (p[2] + q[2])
        normal[1] += (p[2] - q[2]) * (p[0] + q[0])
        normal[2] += (p[0] - q[0]) * (p[1] + q[1])
    norm = float(np.linalg.norm(normal))
    if norm == 0:
        return None
    return math.degrees(math.acos(min(1.0, abs(float(normal[2])) / norm)))


if __name__ == "__main__":  # pragma: no cover
    main()
