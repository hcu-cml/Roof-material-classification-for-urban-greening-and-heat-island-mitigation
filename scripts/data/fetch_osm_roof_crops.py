#!/usr/bin/env python3
"""
Stage 1 (training data): stream georeferenced roof crops for OSM buildings that
already carry `roof:material` and `roof:shape` tags.

For every tagged building inside the AOI the script requests a square crop from
an orthophoto WMS, centred on the building centroid, and writes a JPEG plus a
`.jgw` world file so the crop can be re-loaded as a georeferenced raster.

Data access
-----------
No data source is hard-coded. Supply your own AOI and your own imagery service:

  --wms_url    WMS GetMap endpoint you are licensed to use
  --wms_layer  layer name inside that service
  --bbox       AOI as "minlon minlat maxlon maxlat" (WGS84)
  --city       free-text label used in output file names

Building geometries/tags come from the public Overpass API (default endpoint can
be overridden with --overpass_url).
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from PIL import Image
from pyproj import Transformer, datadir
from shapely.geometry import MultiPolygon, Polygon
from sklearn.cluster import DBSCAN

DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Label sets kept in the study. Buildings tagged with anything else are dropped.
MATERIAL_KEEP = {"concrete", "glass", "metal", "tar_paper", "roof_tiles"}
SHAPE_KEEP = {"flat", "hipped", "gabled", "skillion", "pyramidal"}


# ---- PROJ data dir setup ----------------------------------------------------
def setup_proj_data():
    """Set up the PROJ data directory to avoid CRSError on some installs."""
    possible_paths = [
        Path(sys.prefix) / "share" / "proj",
        Path(sys.prefix) / "Library" / "share" / "proj",
        Path(sys.prefix) / "lib" / "proj",
        Path(os.environ.get("CONDA_PREFIX", sys.prefix)) / "share" / "proj",
    ]

    proj_lib = os.environ.get("PROJ_LIB")
    if proj_lib:
        possible_paths.insert(0, Path(proj_lib))

    for cand in possible_paths:
        if cand.exists() and (cand / "proj.db").exists():
            datadir.set_data_dir(str(cand))
            print(f"PROJ data directory set to: {cand}")
            return True

    try:
        import pyproj

        pyproj.CRS.from_epsg(4326)
        print("PROJ database found via auto-discovery")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not set up PROJ data directory: {e}")
        return False


setup_proj_data()
# ----------------------------------------------------------------------------


def norm(s):
    return str(s).strip().lower().replace("-", "_").replace(" ", "_")


def overpass_fetch_buildings(bbox_lonlat, out_geojson, overpass_url=DEFAULT_OVERPASS_URL):
    """Fetch buildings carrying both tags via Overpass, return GeoDataFrame (WGS84)."""
    minx, miny, maxx, maxy = bbox_lonlat
    s, w, n, e = miny, minx, maxy, maxx
    query = f"""
    [out:json][timeout:120];
    (
      way["building"]["roof:material"]["roof:shape"]({s},{w},{n},{e});
      relation["building"]["roof:material"]["roof:shape"]({s},{w},{n},{e});
    );
    out tags geom;
    """
    r = requests.post(overpass_url, data=query.encode("utf-8"), timeout=180)
    r.raise_for_status()
    data = r.json()
    feats = []
    for el in data.get("elements", []):
        tags = el.get("tags", {})
        mat = norm(tags.get("roof:material", ""))
        shp = norm(tags.get("roof:shape", ""))
        if mat not in MATERIAL_KEEP or shp not in SHAPE_KEEP:
            continue

        geom = None
        if el["type"] == "way":
            pts = el.get("geometry", [])
            coords = [(p["lon"], p["lat"]) for p in pts]
            if len(coords) >= 3:
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                geom = Polygon(coords)
        elif el["type"] == "relation":
            outers = []
            inners = []
            for m in el.get("members", []):
                if "geometry" not in m:
                    continue
                ring = [(p["lon"], p["lat"]) for p in m["geometry"]]
                if len(ring) < 3:
                    continue
                if ring[0] != ring[-1]:
                    ring.append(ring[0])
                poly = Polygon(ring)
                if m.get("role") == "outer":
                    outers.append(poly)
                elif m.get("role") == "inner":
                    inners.append(poly)
            if outers:
                polys = []
                for o in outers:
                    holes = [
                        i.exterior.coords[:]
                        for i in inners
                        if o.buffer(0).contains(i.representative_point())
                    ]
                    holes = [h if h[0] == h[-1] else h + [h[0]] for h in holes]
                    polys.append(Polygon(o.exterior.coords[:], holes))
                geom = MultiPolygon(polys) if len(polys) > 1 else polys[0]

        if geom is not None and geom.is_valid and not geom.is_empty:
            feats.append(
                {
                    "geometry": geom,
                    "material": mat,
                    "shape": shp,
                    "osm_id": el.get("id", ""),
                }
            )

    if not feats:
        gdf = gpd.GeoDataFrame(
            columns=["material", "shape", "geometry", "osm_id"], geometry="geometry"
        )
        gdf.crs = "EPSG:4326"
        return gdf

    gdf = gpd.GeoDataFrame(feats, geometry="geometry")
    gdf.crs = "EPSG:4326"

    Path(out_geojson).parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_geojson, driver="GeoJSON")
    return gdf


def deduplicate_buildings(gdf, min_distance_m=50):
    """
    Remove near-duplicate buildings based on centroid proximity (DBSCAN in
    EPSG:3857). Within each cluster only the largest footprint is kept.
    """
    print(f"Original number of buildings: {len(gdf)}")

    gdf_3857 = gdf.to_crs("EPSG:3857")
    centroids = np.array([(g.centroid.x, g.centroid.y) for g in gdf_3857.geometry])

    clustering = DBSCAN(eps=min_distance_m, min_samples=1).fit(centroids)

    unique_buildings = []
    for cluster_id in np.unique(clustering.labels_):
        cluster_mask = clustering.labels_ == cluster_id
        cluster_buildings = gdf_3857[cluster_mask]

        if len(cluster_buildings) == 1:
            unique_buildings.append(cluster_buildings.index[0])
        else:
            largest_idx = cluster_buildings.geometry.area.idxmax()
            unique_buildings.append(largest_idx)

            dup_count = len(cluster_buildings) - 1
            if dup_count > 0:
                print(
                    f"  Cluster {cluster_id}: keeping largest building, "
                    f"removing {dup_count} duplicates"
                )

    deduplicated_gdf = gdf.loc[unique_buildings].copy()
    print(f"Deduplicated number of buildings: {len(deduplicated_gdf)}")
    print(f"Removed {len(gdf) - len(deduplicated_gdf)} duplicate buildings")

    return deduplicated_gdf


def create_world_file(image_path, x_center, y_center, ground_size_m, pad_frac=0.10, size_px=512):
    """
    Write a .jgw world file (EPSG:3857) for the JPEG crop.

    Line order is the world-file standard:
      A pixel size in x
      D rotation (0 for north-up)
      B rotation (0 for north-up)
      E pixel size in y (negative for north-up)
      C x of the CENTRE of the upper-left pixel
      F y of the CENTRE of the upper-left pixel
    """
    half = 0.5 * ground_size_m * (1.0 + pad_frac)
    pixel_size = (2 * half) / size_px  # metres per pixel

    world_file_content = f"""{pixel_size}
0
0
-{pixel_size}
{x_center - half}
{y_center + half}"""

    world_file_path = image_path.with_suffix(".jgw")
    with open(world_file_path, "w") as f:
        f.write(world_file_content)
    return world_file_path


def wms_crop_3857(
    wms_url,
    wms_layer,
    x_c,
    y_c,
    side_m,
    width_px=512,
    height_px=512,
    pad_frac=0.10,
    sleep=0.25,
):
    """Request one square WMS crop in EPSG:3857 and return it as a PIL image."""
    half = 0.5 * side_m * (1.0 + pad_frac)
    bbox = f"{x_c - half},{y_c - half},{x_c + half},{y_c + half}"
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": wms_layer,
        "STYLES": "",
        "CRS": "EPSG:3857",
        "BBOX": bbox,
        "WIDTH": str(width_px),
        "HEIGHT": str(height_px),
        "FORMAT": "image/jpeg",
    }
    r = requests.get(wms_url, params=params, timeout=30)
    if r.status_code != 200 or "image" not in r.headers.get("Content-Type", ""):
        return None
    time.sleep(sleep)

    img = Image.open(io.BytesIO(r.content)).convert("RGB")

    # Force the exact requested size: the world file assumes it.
    if img.size != (width_px, height_px):
        img = img.resize((width_px, height_px), Image.Resampling.LANCZOS)

    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Data access")[0].strip())
    ap.add_argument("--city", required=True, help="Label used in file names, e.g. 'mycity'.")
    ap.add_argument(
        "--bbox",
        required=True,
        nargs=4,
        type=float,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        help="AOI bounding box in WGS84 degrees.",
    )
    ap.add_argument("--wms_url", required=True, help="Orthophoto WMS GetMap endpoint (your own).")
    ap.add_argument("--wms_layer", required=True, help="WMS layer name (your own).")
    ap.add_argument("--overpass_url", default=DEFAULT_OVERPASS_URL)
    ap.add_argument("--out_images", default="data/crops/images")
    ap.add_argument("--out_meta", default="data/crops/meta/crops_meta.csv")
    ap.add_argument(
        "--buildings_geojson",
        default="",
        help="Cache path for the Overpass result. Default: data/raw/buildings_<city>.geojson",
    )
    ap.add_argument("--ground_size_m", type=float, default=100.0)
    ap.add_argument("--pad_frac", type=float, default=0.10)
    ap.add_argument("--size_px", type=int, default=512, help="Output crop size in pixels.")
    ap.add_argument(
        "--min_distance",
        type=float,
        default=50.0,
        help="Minimum distance between building centroids (m) used for deduplication.",
    )
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--no_georef",
        action="store_true",
        help="Do not write .jgw world files (they are written by default).",
    )
    args = ap.parse_args()

    georef = not args.no_georef
    city = args.city
    bbox = tuple(args.bbox)

    out_images = Path(args.out_images)
    out_images.mkdir(parents=True, exist_ok=True)
    Path(args.out_meta).parent.mkdir(parents=True, exist_ok=True)

    # 1) Fetch (or reuse) tagged buildings
    b_path = (
        Path(args.buildings_geojson)
        if args.buildings_geojson
        else Path("data/raw") / f"buildings_{city}.geojson"
    )
    if b_path.exists():
        print(f"Loading existing buildings from {b_path}")
        b_gdf = gpd.read_file(b_path)
    else:
        print("Fetching buildings from Overpass...")
        b_gdf = overpass_fetch_buildings(bbox, b_path, overpass_url=args.overpass_url)

    if b_gdf.empty:
        print("No buildings with both roof:material and roof:shape in AOI.")
        return

    # 2) Deduplicate
    print("Deduplicating buildings...")
    b_gdf_dedup = deduplicate_buildings(b_gdf, min_distance_m=args.min_distance)

    # 3) Reproject to metres for cropping
    gdf = b_gdf_dedup.to_crs("EPSG:3857")
    if args.limit > 0:
        gdf = gdf.iloc[: args.limit].copy()

    to_wgs84 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    rows = []
    kept = 0
    skipped = 0
    for _, row in gdf.iterrows():
        c = row.geometry.centroid
        x_c, y_c = c.x, c.y
        img = wms_crop_3857(
            args.wms_url,
            args.wms_layer,
            x_c,
            y_c,
            side_m=args.ground_size_m,
            width_px=args.size_px,
            height_px=args.size_px,
            pad_frac=args.pad_frac,
        )
        if img is None:
            skipped += 1
            continue

        stem = f"{city}_{kept:06d}_{row['material']}_{row['shape']}.jpg"
        out_p = out_images / stem
        img.save(out_p, quality=92)

        if georef:
            world_file_path = create_world_file(
                out_p, x_c, y_c, args.ground_size_m, args.pad_frac, size_px=args.size_px
            )
            print(f"Created world file: {world_file_path}")

        lon, lat = to_wgs84.transform(x_c, y_c)
        rows.append(
            {
                "path": str(out_p),
                "world_file": str(out_p.with_suffix(".jgw")) if georef else "",
                "center_lon": lon,
                "center_lat": lat,
                "center_x_3857": x_c,
                "center_y_3857": y_c,
                "material": row["material"],
                "shape": row["shape"],
                "osm_id": row.get("osm_id", ""),
                "pixel_size_m": (2 * 0.5 * args.ground_size_m * (1.0 + args.pad_frac))
                / args.size_px,
            }
        )
        kept += 1
        if kept % 10 == 0:
            print(f"Processed {kept} crops...")

    if rows:
        meta_df = pd.DataFrame(rows)
        meta_df.to_csv(args.out_meta, index=False)
        print(f"Saved {kept} crops, skipped {skipped}. Meta -> {args.out_meta}")

        if georef:
            print(f"Georeferencing: generated {kept} world files (.jgw)")

        print("\nFirst 5 meta entries:")
        print(meta_df.head())
    else:
        print("No crops were generated!")


if __name__ == "__main__":
    main()
