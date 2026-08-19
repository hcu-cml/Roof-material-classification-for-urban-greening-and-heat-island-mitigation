#!/usr/bin/env python3
"""
Stage 4 (training): convert a YOLO *detection* dataset into the folder layout
that Ultralytics classification expects.

  in :  <det_root>/images/<split>/*.tif  +  <det_root>/labels/<split>/*.txt
  out:  <out_root>/<split>/<class_name>/*.tif

Class names are read from the dataset YAML (--yaml). If no YAML is given the
numeric class id is used as the folder name.

NOTE: only the first token of the first line of each label file is read, i.e.
each image is assumed to carry exactly one material label. Multi-object label
files are collapsed to their first entry.

All paths are arguments: point them at your own dataset, nothing is
hard-coded and no data ships with this repository.
"""

import argparse, shutil
from pathlib import Path
import yaml

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--det_root", required=True, help="dataset/material_yolo_bal")
    ap.add_argument("--out_root", required=True, help="dataset_cls/material")
    ap.add_argument("--yaml",      default=None, help="material.yaml (to read class names)")
    args = ap.parse_args()

    det = Path(args.det_root)
    out = Path(args.out_root)
    names = None
    if args.yaml:
        d = yaml.safe_load(Path(args.yaml).read_text())
        # accept either dict {id:name} or list
        if isinstance(d.get("names"), dict):
            names = [v for k,v in sorted(d["names"].items(), key=lambda x:int(x[0]))]
        elif isinstance(d.get("names"), list):
            names = d["names"]

    for split in ["train","val","test"]:
        imgdir = det/"images"/split
        lbldir = det/"labels"/split
        if not imgdir.is_dir() or not lbldir.is_dir():
            print(f"skip {split} (missing dirs)")
            continue
        for lbl in lbldir.glob("*.txt"):
            parts = lbl.read_text().strip().split()
            if len(parts) < 1: continue
            cid = int(parts[0])
            cname = names[cid] if names else str(cid)
            img = imgdir/(lbl.stem + ".tif")
            if not img.exists():
                # try common fallbacks
                for ext in [".tiff",".png",".jpg",".jpeg"]:
                    p = imgdir/(lbl.stem+ext)
                    if p.exists(): img = p; break
            if not img.exists(): 
                continue
            dst = out/split/cname
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img, dst/img.name)
    print("Done ->", out)
if __name__ == "__main__":
    main()
