#!/usr/bin/env python3
"""
Stage 2 (training data): turn the JPEG crops + `.jgw` world files written by
`fetch_osm_roof_crops.py` into proper GeoTIFFs with an embedded CRS.

Point --in at your own crop folder; nothing is hard-coded.
"""

import argparse
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine

# --- helpers -----------------------------------------------------------------

def read_worldfile(jgw_path: Path):
    """
    Reads a 6-line world file (.jgw/.wld).
    Returns A, D, B, E, C, F in that order (standard world file order).
    """
    vals = [float(x.strip()) for x in jgw_path.read_text().splitlines() if x.strip()]
    if len(vals) != 6:
        raise ValueError(f"{jgw_path} must have 6 numbers, found {len(vals)}")
    A, D, B, E, C, F = vals
    return A, D, B, E, C, F

def worldfile_to_affine(A, D, B, E, C, F) -> Affine:
    """
    Convert world-file parameters to a rasterio Affine transform.

    World files store the coordinate of the CENTER of the upper-left pixel (C,F).
    rasterio Affine expects the mapping from (col,row) to map coordinates as:
       x = a*col + b*row + c
       y = d*col + e*row + f

    For world files:
       a = A, b = B
       d = D, e = E
       c = C - 0.5*A - 0.5*B
       f = F - 0.5*D - 0.5*E
    """
    c = C - 0.5 * A - 0.5 * B
    f = F - 0.5 * D - 0.5 * E
    return Affine(A, B, c, D, E, f)

def find_worldfile(img_path: Path) -> Path | None:
    """
    Look for a matching world file next to the image.
    Common names for JPEG: .jgw or .wld.
    """
    cand = img_path.with_suffix(".jgw")
    if cand.exists():
        return cand
    cand = img_path.with_suffix(".wld")
    if cand.exists():
        return cand
    return None

def pil_read_rgb(img_path: Path) -> np.ndarray:
    """
    Read as 8-bit RGB ndarray (H,W,3).
    """
    im = Image.open(img_path).convert("RGB")
    return np.array(im)

def write_geotiff(arr: np.ndarray, tif_path: Path, transform: Affine, crs: str, overwrite: bool = False):
    """
    Write an 8-bit RGB GeoTIFF with LZW compression and tiling.
    """
    tif_path.parent.mkdir(parents=True, exist_ok=True)
    if tif_path.exists() and not overwrite:
        print(f"[skip] {tif_path} exists. Use --overwrite to replace.", file=sys.stderr)
        return False

    height, width = arr.shape[0], arr.shape[1]
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[2] != 3 or arr.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB array, got shape={arr.shape}, dtype={arr.dtype}")

    with rasterio.open(
        tif_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        compress="lzw",
        tiled=True,
        photometric="RGB",
    ) as dst:
        dst.write(arr[:, :, 0], 1)
        dst.write(arr[:, :, 1], 2)
        dst.write(arr[:, :, 2], 3)
    return True

def convert_one(img_path: Path, out_dir: Path | None, crs: str, overwrite: bool = False) -> Path | None:
    """
    Convert one JPG/PNG + worldfile to GeoTIFF.
    Returns path to written TIFF (or None if skipped).
    """
    wf = find_worldfile(img_path)
    if wf is None:
        print(f"[warn] No world file found for {img_path.name} (expected .jgw or .wld). Skipping.", file=sys.stderr)
        return None

    A, D, B, E, C, F = read_worldfile(wf)
    transform = worldfile_to_affine(A, D, B, E, C, F)

    arr = pil_read_rgb(img_path)

    if out_dir is None:
        tif_path = img_path.with_suffix(".tif")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        tif_path = out_dir / (img_path.stem + ".tif")

    ok = write_geotiff(arr, tif_path, transform, crs, overwrite=overwrite)
    if ok:
        print(f"[ok] Wrote {tif_path}")
        return tif_path
    return None

# --- main --------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Convert image + worldfile (.jgw/.wld) to GeoTIFF with CRS.")
    ap.add_argument("--in", dest="input_path", required=True,
                    help="Input image file OR directory containing images (e.g., .jpg)")
    ap.add_argument("--out_dir", default="", help="Output directory for GeoTIFFs (default: alongside inputs)")
    ap.add_argument("--crs", required=True, help="CRS to embed, e.g., 'EPSG:3857' or 'EPSG:25833'")
    ap.add_argument("--exts", default=".jpg,.jpeg,.png", help="Comma-separated image extensions to scan in folders")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing .tif files")
    args = ap.parse_args()

    input_path = Path(args.input_path)
    out_dir = Path(args.out_dir) if args.out_dir else None
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    if input_path.is_file():
        convert_one(input_path, out_dir, args.crs, overwrite=args.overwrite)
        return

    if input_path.is_dir():
        imgs = []
        for ext in exts:
            imgs.extend(input_path.rglob(f"*{ext}"))
        if not imgs:
            print(f"[info] No images with extensions {sorted(exts)} found under {input_path}")
            return
        written = 0
        for p in sorted(imgs):
            tif = convert_one(p, out_dir, args.crs, overwrite=args.overwrite)
            if tif is not None:
                written += 1
        print(f"[done] Converted {written} file(s).")
        return

    print(f"[error] {input_path} not found or not a file/dir.", file=sys.stderr)

if __name__ == "__main__":
    main()
