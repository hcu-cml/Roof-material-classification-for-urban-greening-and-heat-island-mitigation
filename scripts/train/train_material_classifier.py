#!/usr/bin/env python3
"""
Stage 6 (training): fine-tune a YOLO11 classification model on the balanced roof
material dataset.

The defaults below are the settings used for the released checkpoint. Point
--data at your own dataset root (the folder that contains train/ val/ test/ with
one subfolder per material) and --project at your own output directory.
"""

import argparse

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--data",
        required=True,
        help="Dataset root with train/val/test/<material>/ subfolders (your own data).",
    )
    ap.add_argument("--model", default="yolo11l-cls.pt", help="Base checkpoint to fine-tune.")
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--cache", default="disk", help="Ultralytics cache mode: disk, ram or ''.")
    ap.add_argument("--device", default="0", help="'0', 'cpu', 'mps', ...")
    ap.add_argument("--workers", type=int, default=8)

    # Light augmentation on top of the Ultralytics classification defaults.
    ap.add_argument("--flipud", type=float, default=0.0)
    ap.add_argument("--fliplr", type=float, default=0.5)
    ap.add_argument("--degrees", type=float, default=5.0)
    ap.add_argument("--scale", type=float, default=0.10)
    ap.add_argument("--label_smoothing", type=float, default=0.05)

    ap.add_argument("--project", default="runs/classify", help="Output directory (your own path).")
    ap.add_argument("--name", default="material_cls")
    ap.add_argument("--save_period", type=int, default=5)
    args = ap.parse_args()

    model = YOLO(args.model)

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        cache=args.cache if args.cache else False,
        device=args.device,
        workers=args.workers,
        flipud=args.flipud,
        fliplr=args.fliplr,
        degrees=args.degrees,
        scale=args.scale,
        label_smoothing=args.label_smoothing,
        project=args.project,
        name=args.name,
        save_period=args.save_period,
    )


if __name__ == "__main__":
    main()
