#!/usr/bin/env python3
"""
Stage 5 (training): balance the classification dataset by duplicating images of
under-represented materials up to --target_per_class.

Only TRAIN is oversampled; VAL and TEST are copied unchanged, so the evaluation
splits keep their natural class distribution.

The MATERIALS list below must match the class folder names produced by
`build_classification_dataset.py`.

All paths are arguments: point them at your own dataset, nothing is
hard-coded and no data ships with this repository.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from collections import defaultdict

MATERIALS = ["concrete", "metal", "glass", "roof_tiles", "tar_paper"]
IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def list_images(folder: Path):
    """Return all image files in a folder (non-recursive)."""
    if not folder.exists():
        return []
    return [p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS and p.is_file()]


def copy_unique(src: Path, dst_dir: Path, suffix: str = "") -> Path:
    """Copy file to dst_dir, appending suffix and avoiding collisions."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    stem = src.stem + suffix
    out = dst_dir / f"{stem}{src.suffix}"
    k = 1
    while out.exists():
        out = dst_dir / f"{stem}__dup{k}{src.suffix}"
        k += 1
    shutil.copy2(src, out)
    return out


def oversample_train_split(in_root: Path, out_root: Path, target_per_class: int, seed: int = 42):
    random.seed(seed)

    in_train = in_root / "train"
    if not in_train.exists():
        raise SystemExit(f"[ERR] {in_train} does not exist. Expected {in_root}/train/<material>.")

    out_train = out_root / "train"
    if out_train.exists():
        shutil.rmtree(out_train)
    out_train.mkdir(parents=True, exist_ok=True)

    # Collect originals per class
    originals = {}
    for mat in MATERIALS:
        src_dir = in_train / mat
        files = list_images(src_dir)
        if not files:
            print(f"[WARN] No images for class '{mat}' in {src_dir}", file=sys.stderr)
        originals[mat] = files

    # Report current counts
    cur_counts = {mat: len(files) for mat, files in originals.items()}
    print("[INFO] Current TRAIN counts:", cur_counts, file=sys.stderr)

    # Copy originals once
    for mat, files in originals.items():
        dst_dir = out_train / mat
        for f in files:
            copy_unique(f, dst_dir)

    # Oversample classes below target
    for mat, files in originals.items():
        cur = len(files)
        if cur == 0:
            print(f"[WARN] Skipping class '{mat}' (no samples).", file=sys.stderr)
            continue

        if cur >= target_per_class:
            print(
                f"[INFO] Class '{mat}' has {cur} >= {target_per_class}, no oversampling.",
                file=sys.stderr,
            )
            continue

        need = target_per_class - cur
        print(
            f"[INFO] Oversampling class '{mat}' from {cur} to {target_per_class} (+{need})",
            file=sys.stderr,
        )

        dst_dir = out_train / mat
        pool = files.copy()
        random.shuffle(pool)

        for i in range(need):
            src = pool[i % len(pool)]
            suffix = f"__os_{i+1}"
            copy_unique(src, dst_dir, suffix=suffix)

    # Final recount
    final_counts = {}
    for mat in MATERIALS:
        dst_dir = out_train / mat
        final_counts[mat] = len(list_images(dst_dir))

    print("[INFO] Balanced TRAIN counts:", final_counts, file=sys.stderr)
    print(f"[OK] Balanced train split written to: {out_train}", file=sys.stderr)


def mirror_split_if_exists(split: str, in_root: Path, out_root: Path):
    """Copy val/test as-is if they exist (classification structure)."""
    src = in_root / split
    dst = out_root / split
    if not src.exists():
        print(f"[WARN] {src} does not exist, skipping {split}.", file=sys.stderr)
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"[INFO] Copied {src} -> {dst}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Oversample a YOLO-CLS style dataset with structure "
            "`root/split/material/` (e.g. dataset_masked/train/concrete). "
            "Only TRAIN is oversampled; VAL/TEST are copied as-is if present."
        )
    )
    ap.add_argument("--in_root", required=True, help="Input root (e.g. dataset_masked)")
    ap.add_argument("--out_root", required=True, help="Output root (e.g. dataset_masked_bal)")
    ap.add_argument(
        "--target_per_class",
        type=int,
        default=6000,
        help="Target number of samples per class in TRAIN (only oversample lower classes).",
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    in_root = Path(args.in_root)
    out_root = Path(args.out_root)

    # 1) Oversample train
    oversample_train_split(in_root, out_root, args.target_per_class, seed=args.seed)

    # 2) Mirror val/test if they exist
    for split in ["val", "test"]:
        mirror_split_if_exists(split, in_root, out_root)

    print(f"\n[DONE] Balanced classification dataset at: {out_root}")


if __name__ == "__main__":
    main()
