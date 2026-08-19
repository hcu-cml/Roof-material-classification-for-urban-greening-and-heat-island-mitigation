"""Raster IO helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import rasterio

__all__ = ["read_raster", "save_raster"]


def read_raster(path: Path | str) -> Tuple[np.ndarray, dict]:
    """Read one raster band as float32 with nodata represented by NaN."""
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32").filled(np.nan)
        profile = src.profile.copy()
        profile.update(width=src.width, height=src.height, transform=src.transform, crs=src.crs)
    return arr, profile


def save_raster(path: Path | str, arr: np.ndarray, profile: dict) -> None:
    """Write a float32 single-band raster using a template profile."""
    dst_profile = profile.copy()
    dst_profile.update(dtype="float32", count=1, nodata=np.nan)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **dst_profile) as dst:
        dst.write(arr.astype("float32"), 1)
