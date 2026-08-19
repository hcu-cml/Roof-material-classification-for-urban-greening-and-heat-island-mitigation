"""Landsat Level-2 utilities for the green roof scenario package."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

SR_SCALE = 0.0000275
SR_OFFSET = -0.2

ST_SCALE = 0.00341802
ST_OFFSET = 149.0  # Kelvin

__all__ = [
    "SR_SCALE",
    "SR_OFFSET",
    "ST_SCALE",
    "ST_OFFSET",
    "build_lst_from_l2",
    "compute_ndvi_albedo_from_l2",
]


def _qa_bits(arr: np.ndarray, bit: int) -> np.ndarray:
    """Return a boolean mask for one bit in a Landsat QA array."""
    return ((arr >> bit) & 1).astype(bool)


def _build_clear_mask(qa: np.ndarray, keep_water: bool = False) -> np.ndarray:
    """Build a valid-pixel mask from Landsat QA_PIXEL flags."""
    invalid = (
        _qa_bits(qa, 0)
        | _qa_bits(qa, 1)
        | _qa_bits(qa, 2)
        | _qa_bits(qa, 3)
        | _qa_bits(qa, 4)
        | _qa_bits(qa, 5)
    )
    if not keep_water:
        invalid = invalid | _qa_bits(qa, 7)
    return ~invalid


def _find_band(folder: Path | str, suffix: str) -> Optional[str]:
    """Find exactly one Landsat band matching a filename suffix."""
    folder = Path(folder)
    matches = sorted({*folder.glob(f"*{suffix}.TIF"), *folder.glob(f"*{suffix}.tif")})
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise ValueError(f"Multiple Landsat bands match '*{suffix}': {names}")
    return str(matches[0]) if matches else None


def build_lst_from_l2(
    l2_folder: Path | str,
    out_path: Path | None = None,
    *,
    unit: str = "celsius",
    keep_water: bool = False,
) -> Tuple[Path, np.ndarray, dict]:
    """Scale Landsat ST_B10 to LST, apply QA masking, and write it."""
    if unit not in {"celsius", "kelvin"}:
        raise ValueError("unit must be 'celsius' or 'kelvin'.")
    st_path = _find_band(l2_folder, "_ST_B10")
    qa_path = _find_band(l2_folder, "_QA_PIXEL")
    if not st_path:
        raise FileNotFoundError("Could not find *_ST_B10.TIF in the provided L2 folder.")
    if not qa_path:
        raise FileNotFoundError("Could not find *_QA_PIXEL.TIF in the provided L2 folder.")

    st_path = Path(st_path)
    qa_path = Path(qa_path)

    with rasterio.open(st_path) as src_st:
        st_raw = src_st.read(1, masked=True)
        profile = src_st.profile.copy()
        profile.update(width=src_st.width, height=src_st.height, transform=src_st.transform, crs=src_st.crs)

    st_kelvin = st_raw.astype("float32") * ST_SCALE + ST_OFFSET

    with rasterio.open(qa_path) as src_qa:
        if (
            src_qa.crs != profile["crs"]
            or src_qa.transform != profile["transform"]
            or src_qa.width != profile["width"]
            or src_qa.height != profile["height"]
        ):
            raise ValueError(
                "QA_PIXEL and ST_B10 must use the same CRS, transform, width, and height."
            )
        qa = src_qa.read(1)

    clear = _build_clear_mask(qa, keep_water=keep_water)

    if np.ma.isMaskedArray(st_raw):
        valid = clear & ~st_raw.mask
    else:
        valid = clear

    if unit == "celsius":
        out_arr = st_kelvin - 273.15
    else:
        out_arr = st_kelvin

    out_arr = np.array(out_arr, dtype="float32")
    out_arr[~valid] = np.nan

    profile.update(dtype="float32", count=1, nodata=np.nan)

    if out_path is None:
        base = st_path.name
        upper = base.upper()
        suffix_str = "_ST_B10.TIF"
        if upper.endswith(suffix_str):
            base = base[: -len(suffix_str)]
        else:
            base = st_path.stem
        suffix = "_LST_C.tif" if unit == "celsius" else "_LST_K.tif"
        out_path = Path(l2_folder) / f"{base}{suffix}"
    else:
        out_path = Path(out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out_arr, 1)

    return out_path, out_arr, profile


def _read_rescaled_to_template(path: str, template_profile: dict) -> np.ndarray:
    """Reproject and scale a surface-reflectance band to a template grid."""
    shape = (template_profile["height"], template_profile["width"])
    destination = np.full(shape, np.nan, dtype="float32")
    with rasterio.open(path) as src:
        if src.crs is None or template_profile.get("crs") is None:
            raise ValueError("Source bands and the template raster must have a CRS.")
        source = src.read(1, masked=True).astype("float32").filled(np.nan)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return destination * SR_SCALE + SR_OFFSET


def _load_l2_bands_to_template(l2_folder: Path | str, template_path: Path | str):
    """Load required Landsat reflectance bands on the template raster grid."""
    b2 = _find_band(l2_folder, "_SR_B2")
    b4 = _find_band(l2_folder, "_SR_B4")
    b5 = _find_band(l2_folder, "_SR_B5")
    b6 = _find_band(l2_folder, "_SR_B6")
    b7 = _find_band(l2_folder, "_SR_B7")

    if not (b2 and b4 and b5 and b6 and b7):
        raise FileNotFoundError("Could not find all required SR_B2, SR_B4, SR_B5, SR_B6, SR_B7 bands in L2 folder.")

    with rasterio.open(template_path) as tmp:
        profile = tmp.profile.copy()
        profile.update(width=tmp.width, height=tmp.height, transform=tmp.transform, crs=tmp.crs)

    blue = _read_rescaled_to_template(b2, profile)
    red = _read_rescaled_to_template(b4, profile)
    nir = _read_rescaled_to_template(b5, profile)
    swir1 = _read_rescaled_to_template(b6, profile)
    swir2 = _read_rescaled_to_template(b7, profile)

    return blue, red, nir, swir1, swir2, profile


def _compute_indices(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    swir1: np.ndarray,
    swir2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate NDVI, broadband albedo, and NDBI from reflectance arrays."""
    ndvi = (nir - red) / (nir + red + 1e-6)
    ndvi = np.clip(ndvi, -1.0, 1.0)

    albedo = (
        0.356 * blue
        + 0.130 * red
        + 0.373 * nir
        + 0.085 * swir1
        + 0.072 * swir2
        - 0.0018
    )
    albedo = np.clip(albedo, 0.0, 1.0)

    ndbi = (swir1 - nir) / (swir1 + nir + 1e-6)
    ndbi = np.clip(ndbi, -1.0, 1.0)

    return ndvi, albedo, ndbi


def compute_ndvi_albedo_from_l2(l2_folder: Path | str, template_path: Path | str):
    """Derive NDVI, albedo, and NDBI from a Landsat Level-2 scene."""
    blue, red, nir, swir1, swir2, profile = _load_l2_bands_to_template(l2_folder, template_path)
    ndvi, albedo, ndbi = _compute_indices(blue, red, nir, swir1, swir2)
    return ndvi, albedo, ndbi, profile
