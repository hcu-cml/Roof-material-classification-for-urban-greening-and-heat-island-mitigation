"""Fetch already greened roofs from OpenStreetMap via Nominatim and Overpass."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Sequence

import geopandas as gpd
from shapely.geometry import LineString, MultiPolygon, Polygon, box, shape
from shapely.ops import unary_union

DEFAULT_NOMINATIM_URL = "https://nominatim.openstreetmap.org"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_USER_AGENT = "green-roof-scenario/0.1"
DEFAULT_ROOF_MATERIALS = "grass,green_roof,vegetated,vegetation,plants,sedum"
DEFAULT_GREEN_VALUES = "yes,true,green,extensive,intensive,vegetated"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the OSM green-roof fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch OSM buildings tagged as already greened roofs for a city."
    )
    parser.add_argument(
        "--city",
        help=(
            "City query for Nominatim, e.g. 'Madrid, Spain'. Required unless "
            "--osm-relation-id is provided."
        ),
    )
    parser.add_argument(
        "--osm-relation-id",
        type=int,
        help="OSM administrative relation id to use directly, e.g. Madrid=5326784.",
    )
    parser.add_argument(
        "--buildings",
        type=Path,
        help="Building layer whose extent should be used as the Overpass search bbox.",
    )
    parser.add_argument(
        "--buildings-layer",
        help="Optional layer name for --buildings when reading a GeoPackage.",
    )
    parser.add_argument(
        "--bbox",
        help="Explicit Overpass bbox as south,west,north,east in EPSG:4326.",
    )
    parser.add_argument(
        "--extent-padding",
        type=float,
        default=0.0,
        help="Padding added to --buildings extent before conversion to EPSG:4326, in building CRS units.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output vector path. Defaults to data/helpers/<area>_green_roofs.gpkg.",
    )
    parser.add_argument(
        "--boundary-out",
        type=Path,
        help="Optional path to save the resolved city boundary or building extent bbox as a vector file.",
    )
    parser.add_argument(
        "--target-crs",
        default="EPSG:4326",
        help="CRS for output files. Use a projected CRS if you want meter-based areas.",
    )
    parser.add_argument(
        "--roof-materials",
        default=DEFAULT_ROOF_MATERIALS,
        help="Comma-separated roof:material values treated as green roofs.",
    )
    parser.add_argument(
        "--green-values",
        default=DEFAULT_GREEN_VALUES,
        help="Comma-separated values for boolean green-roof tags such as roof:greening=yes.",
    )
    parser.add_argument(
        "--include-building-parts",
        action="store_true",
        help="Also fetch building:part geometries, not only full building footprints.",
    )
    parser.add_argument(
        "--overpass-url",
        default=DEFAULT_OVERPASS_URL,
        help="Overpass API endpoint.",
    )
    parser.add_argument(
        "--nominatim-url",
        default=DEFAULT_NOMINATIM_URL,
        help="Nominatim base URL.",
    )
    parser.add_argument("--timeout", type=int, default=180, help="HTTP and Overpass timeout in seconds.")
    parser.add_argument("--user-agent", default=DEFAULT_USER_AGENT, help="HTTP User-Agent header.")
    return parser


def _request_json(url: str, *, timeout: int, user_agent: str, data: bytes | None = None) -> Any:
    """Perform an HTTP request and decode its JSON response."""
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": user_agent,
            "Accept": "application/json",
        },
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _slugify(value: str) -> str:
    """Convert a label into a filesystem-friendly lowercase slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "city"


def _csv_regex(values: str) -> str:
    """Convert comma-separated literal values into an anchored regex."""
    parts = [re.escape(v.strip()) for v in values.split(",") if v.strip()]
    if not parts:
        raise ValueError("At least one value is required.")
    return "^(" + "|".join(parts) + ")$"


def _nominatim_feature_from_city(city: str, base_url: str, timeout: int, user_agent: str) -> dict[str, Any]:
    """Resolve a city query to the first Nominatim GeoJSON feature."""
    params = urllib.parse.urlencode(
        {
            "format": "geojson",
            "limit": 1,
            "polygon_geojson": 1,
            "q": city,
        }
    )
    data = _request_json(f"{base_url.rstrip('/')}/search?{params}", timeout=timeout, user_agent=user_agent)
    features = data.get("features") or []
    if not features:
        raise ValueError(f"Nominatim returned no result for city query: {city!r}")
    return features[0]


def _nominatim_feature_from_relation(
    relation_id: int,
    base_url: str,
    timeout: int,
    user_agent: str,
) -> dict[str, Any] | None:
    """Look up an OSM relation as a Nominatim GeoJSON feature."""
    params = urllib.parse.urlencode(
        {
            "format": "geojson",
            "polygon_geojson": 1,
            "osm_ids": f"R{relation_id}",
        }
    )
    data = _request_json(f"{base_url.rstrip('/')}/lookup?{params}", timeout=timeout, user_agent=user_agent)
    features = data.get("features") or []
    return features[0] if features else None


def _resolve_relation(
    city: str | None,
    relation_id: int | None,
    base_url: str,
    timeout: int,
    user_agent: str,
) -> tuple[int, dict[str, Any] | None]:
    """Resolve either a city query or explicit relation ID to an OSM relation."""
    if relation_id is not None:
        feature = _nominatim_feature_from_relation(relation_id, base_url, timeout, user_agent)
        return relation_id, feature

    if not city:
        raise ValueError("Provide either --city or --osm-relation-id.")

    feature = _nominatim_feature_from_city(city, base_url, timeout, user_agent)
    props = feature.get("properties") or {}
    osm_type = props.get("osm_type")
    osm_id = props.get("osm_id")
    if osm_type != "relation" or osm_id is None:
        raise ValueError(
            "Nominatim did not resolve the city to an OSM relation. "
            "Use --osm-relation-id with the city's administrative boundary relation."
        )
    return int(osm_id), feature


def _selector_templates(include_building_parts: bool) -> list[str]:
    """Return Overpass tag selectors for known green-roof conventions."""
    selectors = [
        '["building"]["roof:material"~"{roof_material_regex}",i]',
        '["building"]["roof:greening"~"{green_value_regex}",i]',
        '["building"]["roof:green"~"{green_value_regex}",i]',
        '["building"]["green_roof"~"{green_value_regex}",i]',
        '["building"]["roof:vegetation"~"{green_value_regex}",i]',
    ]
    if include_building_parts:
        selectors.extend(
            [
                '["building:part"]["roof:material"~"{roof_material_regex}",i]',
                '["building:part"]["roof:greening"~"{green_value_regex}",i]',
                '["building:part"]["roof:green"~"{green_value_regex}",i]',
                '["building:part"]["green_roof"~"{green_value_regex}",i]',
                '["building:part"]["roof:vegetation"~"{green_value_regex}",i]',
            ]
        )
    return selectors


def _selector_lines(
    selectors: Iterable[str],
    *,
    locator: str,
    roof_material_regex: str,
    green_value_regex: str,
) -> list[str]:
    """Render Overpass selector lines for ways and relations."""
    body_lines: list[str] = []
    for selector in selectors:
        filled = selector.format(
            roof_material_regex=roof_material_regex,
            green_value_regex=green_value_regex,
        )
        body_lines.append(f"  way{filled}{locator};")
        body_lines.append(f"  relation{filled}{locator};")
    return body_lines


def _overpass_area_query(
    relation_id: int,
    *,
    timeout: int,
    roof_material_regex: str,
    green_value_regex: str,
    include_building_parts: bool,
) -> str:
    """Build an Overpass query constrained to an administrative area."""
    area_id = 3_600_000_000 + relation_id
    body_lines = _selector_lines(
        _selector_templates(include_building_parts),
        locator="(area.searchArea)",
        roof_material_regex=roof_material_regex,
        green_value_regex=green_value_regex,
    )

    body = "\n".join(body_lines)
    return f"""[out:json][timeout:{timeout}];
area({area_id})->.searchArea;
(
{body}
);
out tags geom;
"""


def _overpass_bbox_query(
    bbox_wgs84: tuple[float, float, float, float],
    *,
    timeout: int,
    roof_material_regex: str,
    green_value_regex: str,
    include_building_parts: bool,
) -> str:
    """Build an Overpass query constrained to a WGS84 bounding box."""
    south, west, north, east = bbox_wgs84
    locator = f"({south:.8f},{west:.8f},{north:.8f},{east:.8f})"
    body_lines = _selector_lines(
        _selector_templates(include_building_parts),
        locator=locator,
        roof_material_regex=roof_material_regex,
        green_value_regex=green_value_regex,
    )
    body = "\n".join(body_lines)
    return f"""[out:json][timeout:{timeout}];
(
{body}
);
out tags geom;
"""


def _parse_bbox(value: str) -> tuple[float, float, float, float]:
    """Parse and validate a south,west,north,east bounding box string."""
    parts = [float(part.strip()) for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--bbox must contain four comma-separated numbers: south,west,north,east.")
    south, west, north, east = parts
    if south >= north or west >= east:
        raise ValueError("--bbox must be ordered as south,west,north,east.")
    return south, west, north, east


def _extent_from_buildings(
    buildings: Path,
    layer: str | None,
    padding: float,
) -> tuple[tuple[float, float, float, float], gpd.GeoDataFrame]:
    """Derive a WGS84 search box and source-CRS extent from buildings."""
    gdf = gpd.read_file(buildings, layer=layer) if layer else gpd.read_file(buildings)
    if gdf.empty:
        raise ValueError(f"Building layer is empty: {buildings}")
    if gdf.crs is None:
        raise ValueError(f"Building layer has no CRS: {buildings}")
    minx, miny, maxx, maxy = gdf.total_bounds
    extent_geom = box(minx - padding, miny - padding, maxx + padding, maxy + padding)
    extent = gpd.GeoDataFrame(
        [{"source": str(buildings), "padding": padding}],
        geometry=[extent_geom],
        crs=gdf.crs,
    )
    extent_wgs84 = extent.to_crs("EPSG:4326")
    west, south, east, north = extent_wgs84.total_bounds
    return (float(south), float(west), float(north), float(east)), extent


def _closed_polygon(coords: Iterable[dict[str, float]]) -> Polygon | None:
    """Convert Overpass coordinate mappings into a valid closed polygon."""
    pairs = [(float(pt["lon"]), float(pt["lat"])) for pt in coords if "lon" in pt and "lat" in pt]
    if len(pairs) < 3:
        return None
    if pairs[0] != pairs[-1]:
        pairs.append(pairs[0])
    poly = Polygon(pairs)
    if poly.is_empty:
        return None
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    if isinstance(poly, Polygon):
        return poly
    if isinstance(poly, MultiPolygon):
        return max(poly.geoms, key=lambda geom: geom.area)
    return None


def _relation_geometry(element: dict[str, Any]) -> Polygon | MultiPolygon | None:
    """Assemble polygonal geometry from an Overpass relation's members."""
    outers: list[Polygon] = []
    inners: list[Polygon] = []
    for member in element.get("members", []):
        geom = member.get("geometry")
        if not geom:
            continue
        poly = _closed_polygon(geom)
        if poly is None:
            pairs = [(float(pt["lon"]), float(pt["lat"])) for pt in geom if "lon" in pt and "lat" in pt]
            if len(pairs) < 3:
                continue
            line = LineString(pairs)
            if not line.is_ring:
                continue
            poly = Polygon(line)
        if member.get("role") == "inner":
            inners.append(poly)
        else:
            outers.append(poly)
    if not outers:
        return None
    geom = unary_union(outers)
    if inners:
        geom = geom.difference(unary_union(inners))
    if geom.is_empty:
        return None
    if not geom.is_valid:
        geom = geom.buffer(0)
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    return None


def _element_geometry(element: dict[str, Any]) -> Polygon | MultiPolygon | None:
    """Extract polygonal geometry from an Overpass way or relation."""
    if element.get("type") == "way":
        return _closed_polygon(element.get("geometry") or [])
    if element.get("type") == "relation":
        return _relation_geometry(element)
    return None


def _elements_to_gdf(elements: Sequence[dict[str, Any]]) -> gpd.GeoDataFrame:
    """Convert unique polygonal Overpass elements into a GeoDataFrame."""
    records: list[dict[str, Any]] = []
    geometries: list[Polygon | MultiPolygon] = []
    seen: set[tuple[str, int]] = set()

    for element in elements:
        osm_type = str(element.get("type"))
        osm_id = int(element.get("id"))
        key = (osm_type, osm_id)
        if key in seen:
            continue
        seen.add(key)

        geom = _element_geometry(element)
        if geom is None or geom.is_empty:
            continue

        props = dict(element.get("tags") or {})
        props["osm_type"] = osm_type
        props["osm_id"] = osm_id
        records.append(props)
        geometries.append(geom)

    return gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")


def _fetch_green_roofs_from_query(
    query: str,
    *,
    overpass_url: str,
    timeout: int,
    user_agent: str,
) -> gpd.GeoDataFrame:
    """Execute an Overpass query and return green-roof geometries."""
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    data = _request_json(overpass_url, timeout=timeout, user_agent=user_agent, data=payload)
    return _elements_to_gdf(data.get("elements") or [])


def fetch_green_roofs_for_relation(
    *,
    relation_id: int,
    overpass_url: str = DEFAULT_OVERPASS_URL,
    timeout: int = 180,
    user_agent: str = DEFAULT_USER_AGENT,
    roof_materials: str = DEFAULT_ROOF_MATERIALS,
    green_values: str = DEFAULT_GREEN_VALUES,
    include_building_parts: bool = False,
) -> gpd.GeoDataFrame:
    """Fetch tagged green roofs within an OSM administrative relation."""
    query = _overpass_area_query(
        relation_id,
        timeout=timeout,
        roof_material_regex=_csv_regex(roof_materials),
        green_value_regex=_csv_regex(green_values),
        include_building_parts=include_building_parts,
    )
    return _fetch_green_roofs_from_query(query, overpass_url=overpass_url, timeout=timeout, user_agent=user_agent)


def fetch_green_roofs_for_bbox(
    *,
    bbox_wgs84: tuple[float, float, float, float],
    overpass_url: str = DEFAULT_OVERPASS_URL,
    timeout: int = 180,
    user_agent: str = DEFAULT_USER_AGENT,
    roof_materials: str = DEFAULT_ROOF_MATERIALS,
    green_values: str = DEFAULT_GREEN_VALUES,
    include_building_parts: bool = False,
) -> gpd.GeoDataFrame:
    """Fetch tagged green roofs within a WGS84 bounding box."""
    query = _overpass_bbox_query(
        bbox_wgs84,
        timeout=timeout,
        roof_material_regex=_csv_regex(roof_materials),
        green_value_regex=_csv_regex(green_values),
        include_building_parts=include_building_parts,
    )
    return _fetch_green_roofs_from_query(query, overpass_url=overpass_url, timeout=timeout, user_agent=user_agent)


def _write_vector(gdf: gpd.GeoDataFrame, path: Path) -> None:
    """Write fetched features using a driver inferred from the output path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"driver": "GPKG"} if path.suffix.lower() == ".gpkg" else {}
    gdf.to_file(path, **kwargs)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the OSM green-roof fetch command-line workflow."""
    args = build_parser().parse_args(argv)
    area_args = [bool(args.city), args.osm_relation_id is not None, args.buildings is not None, bool(args.bbox)]
    if sum(area_args) != 1:
        raise ValueError("Provide exactly one search area: --city, --osm-relation-id, --buildings, or --bbox.")

    relation_id: int | None = None
    boundary_feature: dict[str, Any] | None = None
    extent_gdf: gpd.GeoDataFrame | None = None
    bbox_wgs84: tuple[float, float, float, float] | None = None

    if args.buildings:
        bbox_wgs84, extent_gdf = _extent_from_buildings(args.buildings, args.buildings_layer, args.extent_padding)
    elif args.bbox:
        bbox_wgs84 = _parse_bbox(args.bbox)
        south, west, north, east = bbox_wgs84
        extent_gdf = gpd.GeoDataFrame(
            [{"source": "explicit_bbox", "padding": 0.0}],
            geometry=[box(west, south, east, north)],
            crs="EPSG:4326",
        )
    else:
        relation_id, boundary_feature = _resolve_relation(
            args.city,
            args.osm_relation_id,
            args.nominatim_url,
            args.timeout,
            args.user_agent,
        )

    out = args.out
    if out is None:
        if args.buildings:
            slug_source = args.buildings.stem
        elif args.bbox:
            slug_source = "bbox"
        else:
            slug_source = args.city or f"relation_{relation_id}"
        out = Path("data/helpers") / f"{_slugify(slug_source)}_green_roofs.gpkg"

    if bbox_wgs84 is not None:
        roofs = fetch_green_roofs_for_bbox(
            bbox_wgs84=bbox_wgs84,
            overpass_url=args.overpass_url,
            timeout=args.timeout,
            user_agent=args.user_agent,
            roof_materials=args.roof_materials,
            green_values=args.green_values,
            include_building_parts=args.include_building_parts,
        )
        area_label = "bbox " + ",".join(f"{value:.6f}" for value in bbox_wgs84)
    else:
        assert relation_id is not None
        roofs = fetch_green_roofs_for_relation(
            relation_id=relation_id,
            overpass_url=args.overpass_url,
            timeout=args.timeout,
            user_agent=args.user_agent,
            roof_materials=args.roof_materials,
            green_values=args.green_values,
            include_building_parts=args.include_building_parts,
        )
        area_label = f"OSM relation {relation_id}"

    if roofs.empty:
        print(
            f"No green-roof building geometries found for {area_label}.",
            file=sys.stderr,
        )
    else:
        roofs = roofs.to_crs(args.target_crs)
    _write_vector(roofs, out)
    print(f"Wrote {len(roofs)} green-roof features to {out}")

    if args.boundary_out:
        if extent_gdf is not None:
            boundary = extent_gdf.to_crs(args.target_crs)
        else:
            if boundary_feature is None or not boundary_feature.get("geometry"):
                raise ValueError("No boundary geometry was returned by Nominatim for this relation.")
            boundary = gpd.GeoDataFrame(
                [
                    {
                        **(boundary_feature.get("properties") or {}),
                        "osm_relation_id": relation_id,
                    }
                ],
                geometry=[shape(boundary_feature["geometry"])],
                crs="EPSG:4326",
            ).to_crs(args.target_crs)
        _write_vector(boundary, args.boundary_out)
        print(f"Wrote search extent to {args.boundary_out}")


if __name__ == "__main__":
    main()
