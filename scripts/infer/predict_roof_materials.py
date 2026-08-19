#!/usr/bin/env python3
"""
Stage 9 (inference): predict multi-material roof composition per building and
write the result back onto the CityGML footprints.

Input
  * Masked GeoTIFF chips (one building per chip, background black) from
    `mask_chips_by_footprint.py`
  * CityGML building footprints as GeoJSON, same CRS as the chips (e.g.
    EPSG:25832) - your own data
  * YOLO-CLS weights (.pt); the class order is read from `model.names`

Per chip
  * Convert TIF to uint8 BGR (percentile stretch computed on non-zero pixels)
  * Derive the roof mask from grayscale > 5
  * Grid-refined classification into a per-pixel class map (whole-roof default
    prediction, confident per-tile overrides, small islands reverted)
  * Optional orientation-aware majority smoothing along the roof main axis
  * Coverage fraction per material, dominant first

Matching
  * Vectorise the roof mask and match it to a footprint by IoU (--min_iou)
  * Per gml_id the chip with the most roof pixels wins

Output
  * GeoJSON with the ORIGINAL footprint geometry plus two properties:
      predicted_roof_materials : list of TARGET_ID values as strings
      material_cov             : matching coverage fractions

Class id spaces (important)
  Two id spaces coexist in this file and they are NOT the same:
    * the YOLO index space, i.e. the order of `model.names` in your weights
    * the output space, defined by TARGET_ID below
  CLASS_PRIORS and CLASS_MIN_COV are looked up with YOLO indices, while the
  values of CLASS_MIN_COV are written as TARGET_ID entries. This mismatch is
  kept as-is because it is the behaviour the published results were produced
  with. If you retrain with a different class order, review these three dicts
  before trusting the thresholds.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import (
    shape,
    mapping,
    Polygon,
    MultiPolygon,
)
from shapely.geometry import box as shp_box
from shapely.strtree import STRtree
from shapely.ops import unary_union
from skimage.morphology import remove_small_objects
import cv2
from ultralytics import YOLO

# Multiplicative priors in the YOLO index space (order of model.names).
# They encode city-scale rarity, e.g. fully glazed roofs are uncommon.
CLASS_PRIORS = {
    0: 1.0,   # concrete
    1: 0.2,   # glass -> strongly penalize
    2: 0.2,   # metal
    3: 1.0,   # roof_tiles
    4: 1.0,   # tar_paper
}
# ---------------------------------------------------------------------
# Constants / mappings
# ---------------------------------------------------------------------
DEFAULT_DST_EPSG = 25832  # CRS tag written into the output GeoJSON

# Output id space: the material codes written to predicted_roof_materials.
# See the "Class id spaces" note in the module docstring.
TARGET_ID = {
    "concrete":   0,
    "metal":      1,
    "glass":      2,
    "roof_tiles": 3,
    "tar_paper":  4,
}

# Extra per-material coverage thresholds (in roof pixel fraction)
# These are applied AFTER mapping to TARGET_ID.
CLASS_MIN_COV = {
    TARGET_ID["concrete"]:   0.02,  # almost always fine
    TARGET_ID["metal"]:      0.06,  # metal must cover >= 6% of roof
    TARGET_ID["glass"]:      0.15,  # glass must cover >= 15% of roof (very strict)
    TARGET_ID["roof_tiles"]: 0.03,
    TARGET_ID["tar_paper"]:  0.03,
}



ALIASES = {
    "asphalt": "tar_paper",
    "bitumen": "tar_paper",
    "tar-paper": "tar_paper",
    "tiles": "roof_tiles",
    "tile": "roof_tiles",
    "clay_tiles": "roof_tiles",
    "slate": "roof_tiles",
}


# ---------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------
def norm_label(name: str) -> str:
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(s, s)


# ---------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------
def to_2d(g):
    """Drop Z, keep XY. Handles Polygon / MultiPolygon."""
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
    Load CityGML footprints GeoJSON (same CRS as orthos, e.g. EPSG:25832).
    Returns:
        feats       : full original features list
        geoms2d     : list of 2D geometries (subset / valid only)
        indices     : index in feats for each item in geoms2d
        tree        : STRtree built from geoms2d
    """
    gj = json.loads(Path(path).read_text(encoding="utf-8"))
    if gj.get("type") != "FeatureCollection":
        raise SystemExit("--footprints must be a GeoJSON FeatureCollection.")

    feats = gj.get("features", [])
    if not feats:
        raise SystemExit(f"No features in footprints: {path}")

    geoms2d = []
    indices = []

    for idx, f in enumerate(feats):
        g = shape(f["geometry"])
        if g.is_empty:
            continue
        g2 = to_2d(g)
        if g2.is_empty or g2.area <= 0:
            continue
        geoms2d.append(g2)
        indices.append(idx)

    if not geoms2d:
        raise SystemExit("All footprint geometries are empty/invalid.")

    tree = STRtree(geoms2d)
    return feats, geoms2d, indices, tree


def build_roof_geom_from_mask(mask01: np.ndarray, transform):
    """
    Convert a binary mask (1 = roof) to a Shapely geometry in the
    chip's CRS using rasterio.features.shapes.
    """
    mask_uint8 = mask01.astype(np.uint8)
    polys = []
    for geom, val in features.shapes(mask_uint8, transform=transform):
        if val != 1:
            continue
        g = shape(geom)
        if not g.is_empty and g.area > 0:
            polys.append(g)

    if not polys:
        return None

    if len(polys) == 1:
        return polys[0]

    return unary_union(polys)


def match_building(roof_geom, geoms2d, indices, tree, min_iou: float = 0.1):
    """
    Match roof_geom (from chip mask) to best CityGML footprint via IoU.
    Returns:
        (feat_index_in_feats, iou_value) or (None, 0.0) if no good match.
    """
    cand_idx = tree.query(roof_geom)
    if cand_idx is None or getattr(cand_idx, "size", 0) == 0:
        return None, 0.0

    best_idx = None
    best_iou = 0.0
    roof_area = roof_geom.area
    if roof_area <= 0:
        return None, 0.0

    for loc_idx in cand_idx:
        loc_idx = int(loc_idx)
        g_build = geoms2d[loc_idx]

        inter = roof_geom.intersection(g_build)
        if inter.is_empty:
            continue
        inter_area = inter.area
        if inter_area <= 0:
            continue

        union_area = roof_area + g_build.area - inter_area
        if union_area <= 0:
            continue

        iou = inter_area / union_area
        if iou > best_iou:
            best_iou = iou
            best_idx = indices[loc_idx]

    if best_idx is None or best_iou < min_iou:
        return None, best_iou

    return best_idx, best_iou


# ---------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------
def to_bgr_uint8_from_tif_maskaware(path: Path) -> np.ndarray:
    """
    Read a masked GeoTIFF and convert to uint8 BGR (H,W,3).

    Important:
      - Percentile stretch is computed ONLY on non-zero pixels to avoid
        black background dominating histogram.
    """
    with rasterio.open(path) as ds:
        arr = ds.read()  # (C,H,W)

    if arr.ndim != 3:
        raise ValueError(f"Expected (C,H,W) array for {path}, got {arr.shape}")

    if arr.dtype != np.uint8:
        arrf = arr.astype(np.float32)
        vals = arrf[arrf > 0]
        if vals.size < 100:
            lo, hi = 0.0, 255.0
        else:
            lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
            if hi <= lo:
                hi = lo + 1.0
        arrf = np.clip((arrf - lo) / max(1e-6, (hi - lo)), 0, 1) * 255.0
        arr = arrf.astype(np.uint8)

    arr = np.moveaxis(arr, 0, 2)  # (H,W,C)

    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]

    # rasterio gives RGB, OpenCV expects BGR → flip channels
    arr = arr[:, :, ::-1]
    return arr


def mask_from_image(img_bgr: np.ndarray, gray_thr: int = 5) -> np.ndarray:
    """Roof mask from grayscale threshold (1 = roof, 0 = background)."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask01 = (gray > gray_thr).astype(np.uint8)
    return mask01


# ---------------------------------------------------------------------
# YOLO-CLS grid refinement (v8-style)
def classify_tta(model, img_bgr, device="cpu", imgsz_list=(160, 224, 288), use_priors=True):
    """
    Simple test-time augmentation over image sizes.
    img_bgr: HxWx3 uint8.

    If use_priors=True, re-weight per-class probabilities
    by CLASS_PRIORS to encode city-scale material priors
    (e.g. glass is rare).
    """
    probs = None
    for s in imgsz_list:
        r = model(img_bgr, imgsz=s, device=device, verbose=False)[0]
        p = r.probs.data.cpu().numpy()
        probs = p if probs is None else probs + p

    probs /= len(imgsz_list)

    if use_priors:
        # apply simple multiplicative prior
        w = np.ones_like(probs, dtype=np.float32)
        for k, alpha in CLASS_PRIORS.items():
            if 0 <= k < w.shape[0]:
                w[k] = float(alpha)
        probs = probs * w
        s = probs.sum()
        if s > 0:
            probs /= s

    k = int(probs.argmax())
    return k, float(probs[k]), probs


def infer_default(model, img, mask01, device="cpu"):
    """Whole-roof default prediction."""
    whole = img.copy()
    whole[mask01 == 0] = 0
    return classify_tta(model, whole, device=device, imgsz_list=(256, 320, 384))


def grid_iter(H, W, tile=192, stride=96):
    """Sliding-window coordinates (x1,y1,x2,y2)."""
    for y in range(0, max(1, H - tile + 1), stride):
        for x in range(0, max(1, W - tile + 1), stride):
            yield x, y, x + tile, y + tile
    if (H - tile) % stride != 0:
        y = H - tile
        for x in range(0, max(1, W - tile + 1), stride):
            yield x, y, x + tile, y + tile
    if (W - tile) % stride != 0:
        x = W - tile
        for y in range(0, max(1, H - tile + 1), stride):
            yield x, y, x + tile, y + tile
    if (H - tile) % stride != 0 and (W - tile) % stride != 0:
        yield W - tile, H - tile, W, H


def build_material_map(model, img, mask01,
                       tile=192, stride=96, pad=32,
                       patch_min_roof_cov=0.55,
                       nondefault_min_conf=0.86,
                       margin_over_default=0.28,
                       min_island_px=200,
                       device="cpu"):
    """
    v8-like material-map logic:
      - default class on whole roof
      - per-patch overrides if confident enough
      - small islands reverted to default
    """
    H, W = mask01.shape
    classes = [model.names[i] for i in sorted(model.names.keys())]

    # 1) default prediction
    def_k, _, def_probs = infer_default(model, img, mask01, device=device)
    class_map = -1 * np.ones((H, W), dtype=np.int32)
    conf_map = np.zeros((H, W), dtype=np.float32)
    class_map[mask01 == 1] = def_k
    conf_map[mask01 == 1] = float(def_probs[def_k])

    # 2) grid overrides
    for x1, y1, x2, y2 in grid_iter(H, W, tile=tile, stride=stride):
        x1p = max(0, x1 - pad)
        y1p = max(0, y1 - pad)
        x2p = min(W, x2 + pad)
        y2p = min(H, y2 + pad)

        patch = img[y1p:y2p, x1p:x2p]
        mpatch = mask01[y1p:y2p, x1p:x2p]
        if mpatch.mean() < patch_min_roof_cov:
            continue

        h, w = patch.shape[:2]
        s = max(h, w)
        sq = np.zeros((s, s, 3), dtype=patch.dtype)
        sq[:h, :w] = patch

        sqm = np.zeros((s, s), dtype=np.uint8)
        sqm[:h, :w] = mpatch
        sq[sqm == 0] = 0

        k, c, probs = classify_tta(
            model, sq, device=device, imgsz_list=(160, 224, 288)
        )
        if k == def_k:
            continue
        c_def = float(probs[def_k])
        if c < nondefault_min_conf or (c - c_def) < margin_over_default:
            continue

        sub = (mask01[y1:y2, x1:x2] == 1)
        if not np.any(sub):
            continue

        region_conf = conf_map[y1:y2, x1:x2]
        update = np.zeros_like(sub, dtype=bool)
        update[sub] = c > region_conf[sub]
        if np.any(update):
            class_map[y1:y2, x1:x2][update] = k
            region_conf[update] = c
            conf_map[y1:y2, x1:x2] = region_conf

    # 3) remove tiny islands of non-default
    K = len(classes)
    for k in range(K):
        if k == def_k:
            continue
        mk = (class_map == k)
        if mk.any():
            kept = remove_small_objects(mk, min_size=min_island_px)
            class_map[(mk) & (~kept)] = def_k

    return class_map, conf_map, def_k, classes


# ---------------------------------------------------------------------
# Orientation-aware smoothing (optional)
# ---------------------------------------------------------------------
def estimate_roof_angle_deg(mask01):
    ys, xs = np.where(mask01 > 0)
    if ys.size < 50:
        return 0.0
    X = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    X -= X.mean(axis=0, keepdims=True)
    cov = X.T @ X / max(1, len(X) - 1)
    _, vecs = np.linalg.eigh(cov)
    v = vecs[:, 1]
    return np.degrees(np.arctan2(v[1], v[0]))


def rotate_img_int(arr, angle_deg, center=None, interp=cv2.INTER_NEAREST, border=-1):
    h, w = arr.shape[:2]
    if center is None:
        center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        arr, M, (w, h),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    ), M


def oriented_majority_smooth(class_map, mask01, kernel_long=45, kernel_short=5):
    """Orientation-aware 1D majority filter along roof main axis."""
    angle = estimate_roof_angle_deg(mask01)
    rot, _ = rotate_img_int(class_map.astype(np.int16), -angle,
                            interp=cv2.INTER_NEAREST, border=-1)
    rot_mask, _ = rotate_img_int(mask01, -angle,
                                 interp=cv2.INTER_NEAREST, border=0)

    if not (rot >= 0).any():
        return class_map

    K = int(rot[rot >= 0].max()) + 1
    kernel = np.ones((kernel_short, kernel_long), np.float32)
    H, W = rot.shape
    scores = np.zeros((K, H, W), np.float32)

    for k_id in range(K):
        scores[k_id] = cv2.filter2D(
            (rot == k_id).astype(np.float32),
            -1,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
        )

    smooth_rot = -1 * np.ones_like(rot, np.int16)
    inside = rot_mask > 0
    if inside.any():
        smooth_rot[inside] = np.argmax(scores[:, inside], axis=0).astype(np.int16)

    smooth, _ = rotate_img_int(smooth_rot, angle,
                               interp=cv2.INTER_NEAREST, border=-1)
    smooth[mask01 == 0] = -1
    return smooth.astype(np.int32)


# ---------------------------------------------------------------------
# Coverage summary
# ---------------------------------------------------------------------
def compute_material_summary(class_map, mask01, classes,
                             min_cov_frac=0.02,
                             max_materials=5):
    """
    Compute coverage fractions per material (relative to roof pixels),
    with:
      - class-specific coverage thresholds (CLASS_MIN_COV in YOLO index space)
      - class priors (CLASS_PRIORS) for ranking only
      - multi-material output (up to max_materials)

    Returns:
        material_ids: list of CityGML IDs as strings, dominant first
        material_cov: list of floats (fractions of roof pixels)
        roof_px:      integer count of roof pixels
    """
    # --- 1) roof mask and total roof pixels ---
    roof_mask = ((class_map != -1) & (mask01 > 0))
    roof_px = int(roof_mask.sum())
    if roof_px == 0:
        return None

    # --- 2) fractional coverage per YOLO class index ---
    frac_by_k = {}
    for k in np.unique(class_map):
        if k < 0:
            continue
        k = int(k)
        px = int((class_map == k).sum())
        if px == 0:
            continue
        frac = px / roof_px
        frac_by_k[k] = frac

    if not frac_by_k:
        return None

    # --- 3) find majority class in YOLO index space ---
    k_major, frac_major = max(frac_by_k.items(), key=lambda kv: kv[1])

    idx_to_name = {i: norm_label(n) for i, n in enumerate(classes)}

    # Base thresholds (you can tweak these two numbers):
    primary_min_cov   = min_cov_frac          # e.g. 0.02
    secondary_min_cov = min_cov_frac          # base for non-major classes
    secondary_rel_to_major = 0.20             # secondary must be >= 20% of majority

    # --- 4) build candidate set: majority + valid secondaries ---
    yolo_candidates = []  # list of (k_idx, frac)

    # Always keep majority class (as long as it maps to a known material)
    yolo_candidates.append((int(k_major), float(frac_major)))

    for k_idx, frac in frac_by_k.items():
        k_idx = int(k_idx)
        if k_idx == k_major:
            continue

        name = idx_to_name.get(k_idx)
        if name not in TARGET_ID:
            continue

        # absolute threshold for this class
        base_thr_abs = secondary_min_cov
        class_thr_abs = max(base_thr_abs, CLASS_MIN_COV.get(k_idx, base_thr_abs))

        # extra strictness for glass (YOLO index 1 by your mapping)
        if name == "glass":
            class_thr_abs *= 1.5  # you can push this to 2.0 if glass still appears too often

        # relative threshold vs majority
        thr_rel = secondary_rel_to_major * frac_major

        # secondary is kept only if it passes both absolute AND relative thresholds
        if frac >= class_thr_abs and frac >= thr_rel:
            yolo_candidates.append((k_idx, frac))

    # If somehow only majority survives and even that is very tiny, keep it anyway:
    if not yolo_candidates:
        name = idx_to_name.get(k_major)
        if name not in TARGET_ID:
            return None
        tid = TARGET_ID[name]
        return [str(tid)], [float(frac_major)], roof_px

    # --- 5) map YOLO indices to CityGML IDs and rank with priors ---
    scored = []  # (eff_score, citygml_id, frac)

    for k_idx, frac in yolo_candidates:
        name = idx_to_name.get(k_idx)
        if name not in TARGET_ID:
            continue
        tid = TARGET_ID[name]

        prior = CLASS_PRIORS.get(k_idx, 1.0)
        eff_score = frac * prior

        scored.append((eff_score, tid, frac))

    if not scored:
        return None

    # Sort by effective score (coverage * prior), dominant first
    scored.sort(key=lambda t: t[0], reverse=True)

    # limit number of materials per roof
    if max_materials and len(scored) > max_materials:
        scored = scored[:max_materials]

    material_ids = [str(tid) for _, tid, _ in scored]
    material_cov = [float(frac) for _, _, frac in scored]

    return material_ids, material_cov, roof_px
                
# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Masked GeoTIFF building chips + YOLO-CLS grid refinement "
                    "→ CityGML GeoJSON with multi-material coverage."
    )

    ap.add_argument("--weights", required=True, help="YOLO-CLS weights (.pt).")
    ap.add_argument("--chips_dir", required=True,
                    help="Folder containing masked GeoTIFF chips (.tif/.tiff).")
    ap.add_argument("--footprints", required=True,
                    help="CityGML building footprints GeoJSON (same CRS, e.g. EPSG:25832).")
    ap.add_argument("--out_geojson", default="citygml_roof_materials_multi.geojson",
                    help="Output GeoJSON with enriched buildings.")
    ap.add_argument("--device", default="mps",
                    help="Torch device (e.g. 'mps', 'cuda', 'cpu').")
    ap.add_argument("--dst_epsg", type=int, default=DEFAULT_DST_EPSG,
                    help="EPSG code tagged on the output GeoJSON (must match your footprints).")

    # roof mask / chip filters
    ap.add_argument("--min_px", type=int, default=200,
                    help="Minimum roof pixels per chip to run classification.")

    # grid-refinement hyperparams
    ap.add_argument("--tile", type=int, default=192)
    ap.add_argument("--stride", type=int, default=96)
    ap.add_argument("--pad", type=int, default=32)
    ap.add_argument("--patch_min_roof_cov", type=float, default=0.55)
    ap.add_argument("--nondefault_min_conf", type=float, default=0.86)
    ap.add_argument("--margin_over_default", type=float, default=0.28)
    ap.add_argument("--min_island_px", type=int, default=200)

    # orientation smoothing
    ap.add_argument("--orient_smooth", action="store_true", default=True)
    ap.add_argument("--kernel_long", type=int, default=45)
    ap.add_argument("--kernel_short", type=int, default=5)

    # coverage settings
    ap.add_argument("--min_cov_frac", type=float, default=0.05)
    ap.add_argument("--max_materials", type=int, default=5)

    # matching
    ap.add_argument("--min_iou", type=float, default=0.1,
                    help="Minimum IoU between chip roof mask and footprint to accept match.")

    args = ap.parse_args()

    chips_root = Path(args.chips_dir)
    if not chips_root.exists():
        raise SystemExit(f"--chips_dir does not exist: {chips_root}")

    chip_paths = sorted(
        p for p in chips_root.rglob("*")
        if p.suffix.lower() in (".tif", ".tiff")
    )
    if not chip_paths:
        raise SystemExit("No GeoTIFF chips found under --chips_dir.")

    # footprints
    foot_feats, foot_geoms2d, foot_indices, tree = load_footprints(args.footprints)

    # YOLO-CLS
    model = YOLO(args.weights)
    device = args.device

    results_by_gml = {}
    n_chips = 0

    for chip_path in chip_paths:
        n_chips += 1
        chip_path = Path(chip_path)

        try:
            img_bgr = to_bgr_uint8_from_tif_maskaware(chip_path)
            h, w = img_bgr.shape[:2]

            # derive roof mask
            mask01 = mask_from_image(img_bgr, gray_thr=5)
            roof_px = int(mask01.sum())
            if roof_px < args.min_px:
                sys.stderr.write(
                    f"[SKIP] {chip_path.name}: roof_px={roof_px} < {args.min_px}\n"
                )
                continue

            # zero-out background just in case
            img_bgr[mask01 == 0] = 0

            # grid-refined material map
            with rasterio.open(chip_path) as ds:
                transform = ds.transform

            class_map, conf_map, def_k, classes = build_material_map(
                model, img_bgr, mask01,
                tile=args.tile,
                stride=args.stride,
                pad=args.pad,
                patch_min_roof_cov=args.patch_min_roof_cov,
                nondefault_min_conf=args.nondefault_min_conf,
                margin_over_default=args.margin_over_default,
                min_island_px=args.min_island_px,
                device=device,
            )

            if args.orient_smooth:
                class_map = oriented_majority_smooth(
                    class_map, mask01,
                    kernel_long=args.kernel_long,
                    kernel_short=args.kernel_short,
                )

            # coverage summary
            summary = compute_material_summary(
                class_map, mask01, classes,
                min_cov_frac=args.min_cov_frac,
                max_materials=args.max_materials,
            )
            if summary is None:
                sys.stderr.write(
                    f"[SKIP] {chip_path.name}: no material summary (probably too small / filtered).\n"
                )
                continue

            material_ids, material_cov, roof_px = summary

            # roof geometry + match to CityGML
            roof_geom = build_roof_geom_from_mask(mask01, transform)
            if roof_geom is None or roof_geom.area <= 0:
                sys.stderr.write(
                    f"[SKIP] {chip_path.name}: empty roof geometry from mask.\n"
                )
                continue

            feat_idx, iou = match_building(
                roof_geom, foot_geoms2d, foot_indices, tree, min_iou=args.min_iou
            )
            if feat_idx is None:
                sys.stderr.write(
                    f"[SKIP] {chip_path.name}: no footprint match (IoU < {args.min_iou}).\n"
                )
                continue

            base_feat = foot_feats[feat_idx]
            props = base_feat.get("properties", {})
            gml_id = props.get("gml_id", str(feat_idx))

            prev = results_by_gml.get(gml_id)
            if (prev is None) or (roof_px > prev["roof_px"]):
                results_by_gml[gml_id] = {
                    "feat_idx": feat_idx,
                    "material_ids": material_ids,
                    "material_cov": material_cov,
                    "roof_px": roof_px,
                }

            # Debug log
            mat_str = ",".join(material_ids)
            cov_str = ",".join(f"{c:.3f}" for c in material_cov)
            sys.stderr.write(
                f"[INFO] chip {n_chips}: {chip_path.name} -> gml_id={gml_id}, "
                f"materials=[{mat_str}], cov=[{cov_str}], roof_px={roof_px}, IoU={iou:.3f}\n"
            )

        except Exception as e:
            sys.stderr.write(f"[ERR] {chip_path.name}: {e}\n")
            continue

    sys.stderr.write(
        f"[INFO] chips processed: {n_chips}, enriched buildings: {len(results_by_gml)}\n"
    )

    if not results_by_gml:
        sys.stderr.write("[WARN] No buildings enriched; nothing written.\n")
        return

    # Build output GeoJSON: ORIGINAL CityGML geometry + new properties
    out_features = []
    for gml_id, info in results_by_gml.items():
        idx = info["feat_idx"]
        base_feat = foot_feats[idx]
        props = dict(base_feat.get("properties", {}))

        props["predicted_roof_materials"] = info["material_ids"]
        props["material_cov"] = info["material_cov"]

        out_features.append({
            "type": "Feature",
            "properties": props,
            "geometry": base_feat["geometry"],  # keep CityGML geometry (with Z if present)
        })

    fc = {
        "type": "FeatureCollection",
        "features": out_features,
        "crs": {
            "type": "name",
            "properties": {
                "name": f"EPSG:{args.dst_epsg}"
            }
        }
    }

    out_path = Path(args.out_geojson)
    out_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
    sys.stderr.write(
        f"[OK] wrote {len(out_features)} enriched buildings → {out_path}\n"
    )


if __name__ == "__main__":
    main()
