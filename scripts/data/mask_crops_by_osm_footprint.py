#!/usr/bin/env python3
"""
Stage 3 (training data): blacken everything outside the OSM building footprint,
so the classifier only ever sees roof pixels.

Inputs are the georeferenced crops from stage 2, the metadata CSV from stage 1
and the OSM footprint GeoJSON. All three are passed as arguments: link your own
data, nothing is hard-coded.

The footprint is matched per crop by OSM id first, and by point-in-polygon on
the crop centre as a fallback.
"""

import argparse, sys, os, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import transform_geom
import geopandas as gpd
from shapely.geometry import shape, Point
from pyproj import Transformer, datadir
from PIL import Image

# -------- PROJ auto-setup (prevents proj_create: no database context) --------
def _setup_proj():
    cands = []
    if os.environ.get("PROJ_LIB"):
        cands.append(Path(os.environ["PROJ_LIB"]))
    cands += [
        Path(os.environ.get("CONDA_PREFIX", sys.prefix))/ "share"/"proj",
        Path(sys.prefix)/"share"/"proj",
        Path(sys.prefix)/"Library"/"share"/"proj",
        Path(sys.prefix)/"lib"/"proj",
    ]
    for p in cands:
        if p and p.exists() and (p/"proj.db").exists():
            datadir.set_data_dir(str(p))
            os.environ["PROJ_LIB"] = str(p)
            break
_setup_proj()
# -----------------------------------------------------------------------------

# ----------------------------- helpers ---------------------------------------
def load_buildings_geojson(path: str):
    try:
        gdf = gpd.read_file(path)
    except Exception:
        gdf = gpd.read_file(path, engine="fiona")
    if "material" not in gdf.columns and "roof:material" in gdf.columns:
        gdf["material"] = gdf["roof:material"]
    if "shape" not in gdf.columns and "roof:shape" in gdf.columns:
        gdf["shape"] = gdf["roof:shape"]
    for c in ["osm_id","@id","id","osm_way_id","osm_rel_id","osm_relation_id"]:
        if c in gdf.columns:
            gdf["__osm_id__"] = gdf[c].astype(str)
            break
    if "__osm_id__" not in gdf.columns:
        gdf["__osm_id__"] = ""
    if gdf.crs is None:
        gdf.set_crs(epsg=4326, inplace=True)
    return gdf

def robust_to_uint8(img):
    if img.dtype == np.uint8:
        return img
    if img.ndim == 2:
        img = img[:, :, None]
    out = np.empty((*img.shape[:2], img.shape[2]), dtype=np.uint8)
    for b in range(img.shape[2]):
        band = img[:, :, b].astype(np.float32)
        ok = np.isfinite(band)
        if not ok.any():
            out[:, :, b] = 0
            continue
        lo, hi = np.percentile(band[ok], [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = band[ok].min(), band[ok].max()
            if hi <= lo:
                out[:, :, b] = 0; continue
        band = (band - lo) / max(1e-6, (hi - lo))
        out[:, :, b] = np.clip(band, 0, 1) * 255.0
    if out.shape[2] == 1: out = np.repeat(out, 3, axis=2)
    if out.shape[2] > 3:  out = out[:, :, :3]
    return out

def read_rgb_uint8(ds):
    if ds.count >= 3:
        arr = ds.read([1,2,3]).transpose(1,2,0)  # HWC
    else:
        arr = ds.read(1)[:, :, None]
    if arr.dtype != np.uint8:
        arr = robust_to_uint8(arr)
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] > 3:
        arr = arr[:, :, :3]
    return arr

def resolve_image_path(meta_path: str, images_root: Path, exts):
    """Return a valid Path to the image. Try CSV path, else search by stem under images_root."""
    p = Path(meta_path)
    if p.exists():
        return p
    stem = p.stem  # ignore extension & directory
    # search exact filename under images_root
    hits = list(images_root.rglob(p.name))
    if hits:
        return hits[0]
    # search by stem with allowed extensions
    for ext in exts:
        for hit in images_root.rglob(stem + ext):
            return hit
    # also try upper/lowercase of stem
    ul = {stem.lower(), stem.upper()}
    for s in ul:
        for ext in exts:
            for hit in images_root.rglob(s + ext):
                return hit
    return None

def polygon_for_row(meta_row, b_gdf: gpd.GeoDataFrame, ds):
    """Prefer OSM ID; else point-in-polygon using center (from meta or computed from raster)."""
    # 1) by osm_id
    osm_id = str(meta_row.get("osm_id", "")).strip()
    if osm_id and "__osm_id__" in b_gdf.columns:
        cand = b_gdf[b_gdf["__osm_id__"] == osm_id]
        if not cand.empty:
            geom = cand.iloc[0].geometry
            geom_json = transform_geom(str(b_gdf.crs), str(ds.crs), geom.__geo_interface__, precision=6)
            return shape(geom_json)

    # 2) spatial join at center
    lon, lat = meta_row.get("center_lon", None), meta_row.get("center_lat", None)
    if pd.isna(lon) or pd.isna(lat):
        # compute center from raster bounds and transform to lon/lat
        left, bottom, right, top = ds.bounds
        cx, cy = (left+right)/2.0, (top+bottom)/2.0
        to4326 = Transformer.from_crs(ds.crs, "EPSG:4326", always_xy=True)
        lon, lat = to4326.transform(cx, cy)

    pt = gpd.GeoDataFrame(geometry=[Point(float(lon), float(lat))], crs="EPSG:4326")
    b = b_gdf if str(b_gdf.crs).upper() in ("EPSG:4326","OGC:CRS84") else b_gdf.to_crs("EPSG:4326")
    try:
        join = gpd.sjoin(pt, b, how="left", predicate="within")
    except Exception:
        join = gpd.sjoin(pt, b, how="left", op="within")
    if join.empty:
        return None
    idx = join.index_right.iloc[0]
    geom = b.loc[idx].geometry
    geom_json = transform_geom(str(b.crs), str(ds.crs), geom.__geo_interface__, precision=6)
    return shape(geom_json)

def rasterize_mask(poly, ds):
    return rasterize([(poly, 1)], out_shape=(ds.height, ds.width), transform=ds.transform, fill=0, dtype="uint8").astype(bool)

def blacken_outside(rgb, mask, alpha_outside=False):
    if alpha_outside:
        a = np.where(mask, 255, 0).astype(np.uint8)
        return np.dstack([rgb, a])
    out = rgb.copy()
    out[~mask] = 0
    return out
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="Root where the real images live (we will search here)")
    ap.add_argument("--meta_csv",   required=True, help="Meta CSV with at least: path, material, shape (and optional osm_id, center_lon, center_lat)")
    ap.add_argument("--buildings_geojson", required=True, help="OSM buildings GeoJSON")
    ap.add_argument("--out_dir",    default="data/masked", help="Output folder")
    ap.add_argument("--out_ext",    default="png", choices=["png","jpg","jpeg","tif","tiff"])
    ap.add_argument("--alpha_outside", action="store_true")
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--exts", default=".tif,.tiff,.png,.jpg,.jpeg", help="Extensions to try when resolving images")
    args = ap.parse_args()

    images_root = Path(args.images_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    exts = tuple(e.strip().lower() for e in args.exts.split(","))

    df = pd.read_csv(args.meta_csv)
    b_gdf = load_buildings_geojson(args.buildings_geojson)

    rows_out = []
    missing_imgs = 0
    missing_poly = 0
    done = 0

    for _, r in df.iterrows():
        resolved = resolve_image_path(str(r["path"]), images_root, exts)
        if resolved is None:
            warnings.warn(f"Image missing (unresolved): {r['path']}")
            missing_imgs += 1
            continue

        with rasterio.open(resolved) as ds:
            rgb = read_rgb_uint8(ds)
            poly = polygon_for_row(r, b_gdf, ds)
            if poly is None or poly.is_empty:
                # if no polygon found, skip this image
                missing_poly += 1
                continue
            mask = rasterize_mask(poly, ds)
            out_img = blacken_outside(rgb, mask, alpha_outside=args.alpha_outside)

        out_name = resolved.stem + f"_masked.{args.out_ext}"
        out_path = out_dir / out_name
        if args.out_ext in ("jpg","jpeg"):
            Image.fromarray(out_img).save(out_path, quality=args.quality)
        elif args.out_ext in ("tif","tiff"):
            # write plain 8-bit GeoTIFF (no CRS) for training simplicity
            profile = {"driver":"GTiff","height":out_img.shape[0],"width":out_img.shape[1],"count":out_img.shape[2],"dtype":"uint8"}
            with rasterio.open(out_path, "w", **profile) as dst:
                for b in range(out_img.shape[2]): dst.write(out_img[:,:,b], b+1)
        else:
            Image.fromarray(out_img).save(out_path)

        rows_out.append({
            "path": str(out_path),
            "material": r.get("material",""),
            "shape": r.get("shape",""),
            "osm_id": r.get("osm_id",""),
        })
        done += 1
        if done % 50 == 0:
            print(f"Processed {done} images…")

    train_csv = out_dir / "masked_training.csv"
    pd.DataFrame(rows_out, columns=["path","material","shape","osm_id"]).to_csv(train_csv, index=False)
    print(f"\nDone. Masked images: {done}  |  Missing images: {missing_imgs}  |  Missing polygons: {missing_poly}")
    print(f"Training CSV -> {train_csv}")

if __name__ == "__main__":
    main()
