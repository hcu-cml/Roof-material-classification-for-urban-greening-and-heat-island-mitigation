#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# LEGACY / AD-HOC SCRIPT
# Kept for transparency: this is one of the helper scripts used during the
# original city runs of the study. It is not part of the supported CLI.
#
# ALL FILE PATHS AND DEFAULTS BELOW ARE EXAMPLES FROM THOSE RUNS.
# No data ships with this repository: replace every path with a link to your
# own data before running, either by editing the constants/defaults in this
# file or by passing the corresponding command-line arguments.
# -----------------------------------------------------------------------------

"""
Make LST GeoTIFF from a Landsat 8/9 Collection 2 Level-2 scene folder.

Input:  a folder containing the scene's *_ST_B10.TIF and *_QA_PIXEL.TIF
Output: LST GeoTIFF (Celsius by default), cloud/snow/cirrus/shadow masked.

Usage:
  python lst_lvl2.py data/LC08_L2SP_195023_20250622_20250630_02_T1 --out lst_hamburg.tif --unit celsius
"""

import argparse
import glob
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine

# USGS C2 L2 scaling for Surface Temperature (Kelvin)
ST_SCALE = 0.00341802
ST_OFFSET = 149.0   # add after scaling  (Kelvin)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("folder", help="Path to the Level-2 scene folder")
    p.add_argument("--out", default=None, help="Output GeoTIFF path (default: auto)")
    p.add_argument("--unit", choices=["celsius", "kelvin"], default="celsius",
                   help="Output temperature unit")
    p.add_argument("--keep-water", action="store_true",
                   help="Do not mask water pixels (QA bit 7)")
    return p.parse_args()

def find_band(folder, suffix):
    pats = [
        os.path.join(folder, f"*{suffix}.TIF"),
        os.path.join(folder, f"*{suffix}.tif"),
    ]
    matches = []
    for pat in pats:
        matches.extend(glob.glob(pat))
    return matches[0] if matches else None

def qa_bits(arr, bit):
    """Return boolean array where the given bit is set."""
    return ((arr >> bit) & 1).astype(bool)

def build_clear_mask(qa, keep_water=False):
    """
    QA_PIXEL bits (L8/9 C2):
      0 Fill
      1 Dilated Cloud
      2 Cirrus (high confidence)
      3 Cloud (high confidence)
      4 Cloud Shadow (high confidence)
      5 Snow (high confidence)
      7 Water
    We mark pixels invalid if any of 0,1,2,3,4,5 are set.
    Optionally also exclude water (bit 7).
    """
    invalid = (
        qa_bits(qa, 0) |  # fill
        qa_bits(qa, 1) |  # dilated cloud
        qa_bits(qa, 2) |  # cirrus
        qa_bits(qa, 3) |  # cloud
        qa_bits(qa, 4) |  # cloud shadow
        qa_bits(qa, 5)    # snow
    )
    if not keep_water:
        invalid = invalid | qa_bits(qa, 7)  # water
    return ~invalid

def main():
    args = parse_args()
    st_path  = find_band(args.folder, "_ST_B10")
    qa_path  = find_band(args.folder, "_QA_PIXEL")
    if not st_path:
        raise FileNotFoundError("Could not find *_ST_B10.TIF in folder")
    if not qa_path:
        raise FileNotFoundError("Could not find *_QA_PIXEL.TIF in folder")

    # Read ST as uint16; scale to Kelvin
    with rasterio.open(st_path) as src_st:
        st_raw = src_st.read(1, masked=True)  # mask nodata if present
        profile = src_st.profile.copy()

    # Apply scale and offset to get Kelvin (float32)
    st_kelvin = st_raw.astype("float32") * ST_SCALE + ST_OFFSET

    # Read QA and build a clear mask
    with rasterio.open(qa_path) as src_qa:
        qa = src_qa.read(1)

    clear = build_clear_mask(qa, keep_water=args.keep_water)

    # Combine with any nodata mask from ST
    if np.ma.isMaskedArray(st_raw):
        valid = clear & ~st_raw.mask
    else:
        valid = clear

    # Convert to desired unit
    if args.unit == "celsius":
        out_data = st_kelvin - 273.15
        out_nodata = np.float32(np.nan)
    else:
        out_data = st_kelvin
        out_nodata = np.float32(np.nan)

    # Apply mask
    out_arr = np.array(out_data, dtype="float32")
    out_arr[~valid] = np.nan

    # Prepare profile
    profile.update(
        dtype="float32",
        count=1,
        nodata=np.nan,
    )

    # Default output name
    if args.out is None:
        base = os.path.basename(st_path).replace("_ST_B10.TIF", "")
        suffix = "_LST_C.tif" if args.unit == "celsius" else "_LST_K.tif"
        args.out = os.path.join(args.folder, f"{base}{suffix}")

    # Write
    with rasterio.open(args.out, "w", **profile) as dst:
        dst.write(out_arr, 1)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()