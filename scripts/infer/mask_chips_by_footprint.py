#!/usr/bin/env python3
"""
Stage 8 (inference): for every patch produced by `tile_orthophoto.py`, write one
GeoTIFF chip per building, with all pixels outside that building's footprint set
to zero.

Footprints must be a GeoJSON FeatureCollection in the SAME CRS as the patches
(e.g. EPSG:25832 for German LoD2 data). Link your own footprints and your own
imagery: nothing is hard-coded.

Chips are rejected when the roof mask is smaller than --min_px or when the chip
ends up completely black.
"""

import os
os.environ.setdefault("PROJ_NETWORK", "ON")

import sys, json, argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape, mapping, Polygon, MultiPolygon, box
from shapely.errors import TopologicalError
from shapely.strtree import STRtree


def to_2d(g):
    """
    Drop Z, keep XY. Handles Polygon / MultiPolygon.
    """
    if g.is_empty:
        return g
    if isinstance(g, Polygon):
        ext = [(x, y) for x, y, *rest in g.exterior.coords]
        ints = [
            [(x, y) for x, y, *rest in ring.coords]
            for ring in g.interiors
        ]
        return Polygon(ext, ints)
    if isinstance(g, MultiPolygon):
        return MultiPolygon([to_2d(p) for p in g.geoms])
    return g


def load_footprints(path):
    """
    Load CityGML footprints GeoJSON (same CRS as orthophotos, e.g. EPSG:25832).
    Build 2D geometries + STRtree for fast per-image queries.
    """
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    if gj.get("type") != "FeatureCollection":
        raise SystemExit("--footprints must be a GeoJSON FeatureCollection.")

    feats = gj.get("features", [])
    if not feats:
        raise SystemExit(f"No features in footprints: {path}")

    geoms_2d = []
    indices = []  # index into feats
    for idx, f in enumerate(feats):
        g = shape(f["geometry"])
        if g.is_empty:
            continue
        g2 = to_2d(g)
        if g2.is_empty or g2.area <= 0:
            continue
        geoms_2d.append(g2)
        indices.append(idx)

    if not geoms_2d:
        raise SystemExit("All footprint geometries are empty/invalid.")

    tree = STRtree(geoms_2d)
    return feats, geoms_2d, indices, tree


def mask_one_polygon_fullframe(dataset,
                               full_img,
                               geom_img_crs,
                               out_dir,
                               stem,
                               idx,
                               min_px=200,
                               dilate_px=2):
    """
    Mask the full image with ONE building polygon, write a GeoTIFF chip if:
      - roof mask has >= min_px pixels
      - chip is not all-black

    Output:
      - GeoTIFF (.tif), same width/height/transform/CRS as the source tile
      - dtype uint8, 3 bands, 1–99 percentile stretched (like PNG chips)
    """
    import cv2

    poly = shape(geom_img_crs["geometry"])
    if poly.is_empty:
        return 0

    H, W = dataset.height, dataset.width
    mask_chip = features.rasterize(
        [(mapping(poly), 1)],
        out_shape=(H, W),
        transform=dataset.transform,
        fill=0,
        dtype="uint8",
        all_touched=True
    ).astype(bool)

    if dilate_px and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask_chip = cv2.dilate(mask_chip.astype(np.uint8), k, iterations=1).astype(bool)

    # Reject tiny roofs
    roof_px = int(mask_chip.sum())
    if roof_px < min_px:
        return 0

    # Apply mask to pre-loaded image
    chip = full_img.copy()  # (C, H, W)
    chip[:, ~mask_chip] = 0

    # Reject all-black chips (nodata-only)
    if not chip.any():
        return 0

    # Convert to uint8 3-band with percentile stretch (like PNG pipeline)
    arr = chip.astype(np.float32)
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo:
        return 0
    arr = np.clip((arr - lo) / max(1e-6, (hi - lo)), 0, 1) * 255.0
    arr = arr.astype(np.uint8)  # (C, H, W)

    # Ensure 3 channels
    if arr.shape[0] == 1:
        arr = np.repeat(arr, 3, axis=0)
    elif arr.shape[0] > 3:
        arr = arr[:3, :, :]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"{stem}_b{idx:03d}"
    out_path = out_dir / f"{out_stem}.tif"

    meta = dataset.meta.copy()
    meta.update({
        "count": arr.shape[0],
        "dtype": "uint8",
        "compress": "DEFLATE",
    })

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr)

    return roof_px


def main():
    ap = argparse.ArgumentParser(
        description="Batch: mask CityGML building footprints for all GeoTIFFs in a folder (full-frame, GeoTIFF chips)."
    )
    ap.add_argument("--dir", required=True, help="Folder with .tif/.tiff (recursive).")
    ap.add_argument("--out_dir", required=True, help="Single output folder (flat).")
    ap.add_argument("--footprints", required=True,
                    help="GeoJSON with CityGML building footprints in SAME CRS as GeoTIFFs (e.g. EPSG:25832).")
    ap.add_argument("--min_px", type=int, default=200, help="Min roof mask pixels to keep a chip.")
    ap.add_argument("--dilate_px", type=int, default=2, help="Mask dilation (pixels).")

    args = ap.parse_args()

    tif_files = sorted([p for p in Path(args.dir).rglob("*")
                        if p.suffix.lower() in (".tif", ".tiff")])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not tif_files:
        raise SystemExit("No GeoTIFFs found in --dir.")

    # Load footprints once + build STRtree
    foot_feats, foot_geoms_2d, foot_indices, tree = load_footprints(args.footprints)

    for tif_path in tif_files:
        try:
            with rasterio.open(tif_path) as ds:
                if ds.crs is None:
                    print(f"[SKIP] {tif_path.name}: no CRS.", file=sys.stderr)
                    continue

                # Pre-load the full image ONCE per file (huge speedup)
                full_img = ds.read()  # (C, H, W)

                b = ds.bounds
                img_bbox_imgcrs = box(b.left, b.bottom, b.right, b.top)

                # Query spatial index for candidate buildings overlapping this image
                cand_idx = tree.query(img_bbox_imgcrs)
                if cand_idx is None or getattr(cand_idx, "size", 0) == 0:
                    print(f"[SKIP] {tif_path.name}: no buildings in extent.", file=sys.stderr)
                    continue

                feats_img = []
                for loc_idx in cand_idx:
                    loc_idx = int(loc_idx)
                    base_idx = foot_indices[loc_idx]
                    base_feat = foot_feats[base_idx]
                    g2 = foot_geoms_2d[loc_idx]
                    try:
                        inter = g2.intersection(img_bbox_imgcrs)
                        if inter.is_empty:
                            continue
                        feats_img.append({
                            "type": "Feature",
                            "properties": base_feat.get("properties", {}),
                            "geometry": mapping(inter),
                        })
                    except TopologicalError:
                        continue
                    except Exception:
                        continue

                if not feats_img:
                    print(f"[SKIP] {tif_path.name}: zero overlap after intersection.", file=sys.stderr)
                    continue

                kept = 0
                for i, feat in enumerate(feats_img, start=1):
                    geom = shape(feat["geometry"])
                    parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
                    for j, sub in enumerate(parts, start=1):
                        area_px = mask_one_polygon_fullframe(
                            dataset=ds,
                            full_img=full_img,
                            geom_img_crs={
                                "type": "Feature",
                                "properties": feat.get("properties", {}),
                                "geometry": mapping(sub),
                            },
                            out_dir=out_dir,
                            stem=tif_path.stem,
                            idx=(i * 1000 + j),
                            min_px=args.min_px,
                            dilate_px=args.dilate_px,
                        )
                        if area_px > 0:
                            kept += 1

                if kept == 0:
                    print(f"[SKIP] {tif_path.name}: no chips passed thresholds.", file=sys.stderr)
                else:
                    print(f"[OK] {tif_path.name}: wrote {kept} masked TIFs → {out_dir}", file=sys.stderr)

        except Exception as e:
            print(f"[ERR] {tif_path.name}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
