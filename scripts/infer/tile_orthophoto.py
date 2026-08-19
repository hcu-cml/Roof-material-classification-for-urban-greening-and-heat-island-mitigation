#!/usr/bin/env python
"""
Stage 7 (inference): mosaic all GeoTIFF tiles of an area of interest and cut the
mosaic into fixed-size patches.

Defaults reproduce the setting used in the paper: 500 x 500 px patches which, at
0.20 m/px, cover 100 x 100 m on the ground.

Bring your own orthophoto: --in_dir must point to a folder of GeoTIFFs that you
downloaded/licensed yourself. No imagery is shipped with this repository.
"""

import argparse
import glob
import os

import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in_dir", required=True, help="Folder with the .tif tiles of your AOI.")
    ap.add_argument("--out_dir", required=True, help="Where the patches are written.")
    ap.add_argument("--patch_size_px", type=int, default=500)
    ap.add_argument("--stride_px", type=int, default=500)
    ap.add_argument(
        "--keep_partial",
        action="store_true",
        help="Also write edge patches smaller than --patch_size_px.",
    )
    ap.add_argument(
        "--expected_res_m",
        type=float,
        default=0.20,
        help="Expected pixel size in metres; only used for a sanity warning.",
    )
    args = ap.parse_args()

    input_dir = args.in_dir
    output_dir = args.out_dir
    patch_size_px = args.patch_size_px
    stride_px = args.stride_px
    keep_partial = args.keep_partial

    os.makedirs(output_dir, exist_ok=True)

    # -- mosaic all tiles in the directory ------------------------------------
    tif_files = sorted(glob.glob(os.path.join(input_dir, "*.tif")))
    print(f"Found {len(tif_files)} tiles in {input_dir}")
    if not tif_files:
        raise SystemExit("No .tif files found - check --in_dir")

    sources = [rasterio.open(f) for f in tif_files]
    mosaic, mosaic_transform = merge(sources)

    meta = sources[0].meta.copy()
    meta.update(
        {
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": mosaic_transform,
        }
    )

    for src in sources:
        src.close()

    height = mosaic.shape[1]
    width = mosaic.shape[2]
    res_x = mosaic_transform.a
    res_y = abs(mosaic_transform.e)

    print(f"Mosaic size : {width} x {height} pixels")
    print(f"Bands       : {mosaic.shape[0]}")
    print(f"Resolution  : {res_x:.4f} x {res_y:.4f} m/pixel")

    if abs(res_x - args.expected_res_m) < 1e-4:
        ground = patch_size_px * res_x
        print(
            f"{args.expected_res_m:.2f} m/pixel confirmed -> "
            f"{patch_size_px}x{patch_size_px} px = {ground:.0f}x{ground:.0f} m patches"
        )
    else:
        print(
            f"WARNING: resolution is {res_x} m/pixel, not {args.expected_res_m} - "
            "patch footprint will differ"
        )

    # -- patch ----------------------------------------------------------------
    x_starts = range(0, width - patch_size_px + 1, stride_px)
    y_starts = range(0, height - patch_size_px + 1, stride_px)

    if keep_partial:
        x_starts = range(0, width, stride_px)
        y_starts = range(0, height, stride_px)

    patch_id = 0
    for y in y_starts:
        for x in x_starts:
            w = min(patch_size_px, width - x)
            h = min(patch_size_px, height - y)

            if not keep_partial and (w < patch_size_px or h < patch_size_px):
                continue

            patch = mosaic[:, y : y + h, x : x + w]

            patch_transform = window_transform(Window(x, y, w, h), mosaic_transform)

            out_meta = meta.copy()
            out_meta.update({"height": h, "width": w, "transform": patch_transform})

            out_path = os.path.join(output_dir, f"patch_{patch_id:06d}_x{x}_y{y}.tif")
            with rasterio.open(out_path, "w", **out_meta) as dst:
                dst.write(patch)

            patch_id += 1

        if y % (stride_px * 10) == 0:
            print(f"  row y={y} done, {patch_id} patches so far...")

    print(f"Done. Wrote {patch_id} patches to: {output_dir}")


if __name__ == "__main__":
    main()
