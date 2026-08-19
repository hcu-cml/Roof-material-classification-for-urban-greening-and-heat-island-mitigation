"""Configuration dataclasses for running scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["ScenarioConfig"]


@dataclass(slots=True)
class ScenarioConfig:
    l2_folder: Path
    buildings: Path
    roof_material_field: str | None = "predictedrooftypematerial"
    roof_materials_type: str | None = None
    roof_shape_field: str | None = None
    roof_shape_type: str | None = None
    roof_slope_field: str | None = "roof_slope_mean_deg"
    max_roof_slope_deg: float | None = None
    boundary: Path | None = None
    out_dir: Path = Path("results_greening")
    lst: Path | None = None
    build_lst: bool = True
    lst_unit: str = "celsius"
    keep_lst_water: bool = False
    layer: Optional[str] = None
    target_ndvi: Optional[float] = 0.4
    target_albedo: float = 0.20
    target_ndbi: float = -0.15
    sample_frac: float = 0.1
    min_sample_spacing: float = 100.0
    random_state: int = 42
    model: str = "rf"
    supersample: int = 4
    all_touched: bool = False
    write_pred_baseline: bool = False
    keep_null_roof: bool = False
    write_roof_fraction_raster: bool = False
    write_indices_rasters: bool = False
    log_level: str = "INFO"
    min_roof_area: float = 0.0
    clip_positive_delta: bool = False

    def __post_init__(self) -> None:
        """Normalize paths and validate scenario parameter combinations."""
        self.l2_folder = Path(self.l2_folder)
        self.buildings = Path(self.buildings)
        self.out_dir = Path(self.out_dir)
        if self.lst is not None:
            self.lst = Path(self.lst)
        if self.boundary is not None:
            self.boundary = Path(self.boundary)

        has_materials = bool(self.roof_materials_type and self.roof_materials_type.strip())
        has_shapes = bool(self.roof_shape_type and self.roof_shape_type.strip())
        has_slope = self.max_roof_slope_deg is not None
        if not has_materials and not has_shapes and not has_slope:
            raise ValueError(
                "Provide at least one roof material, shape, or maximum-slope filter."
            )
        if not self.build_lst and self.lst is None:
            raise ValueError("Provide lst or set build_lst=True.")
        if self.lst_unit not in {"celsius", "kelvin"}:
            raise ValueError("lst_unit must be 'celsius' or 'kelvin'.")
        if self.model not in {"linear", "rf"}:
            raise ValueError("model must be 'linear' or 'rf'.")
        if not 0 < self.sample_frac <= 1:
            raise ValueError("sample_frac must be greater than 0 and at most 1.")
        if self.min_sample_spacing < 0:
            raise ValueError("min_sample_spacing must be >= 0.")
        if self.supersample < 1:
            raise ValueError("supersample must be >= 1.")
        if self.min_roof_area < 0:
            raise ValueError("min_roof_area must be >= 0.")
        if self.max_roof_slope_deg is not None and self.max_roof_slope_deg < 0:
            raise ValueError("max_roof_slope_deg must be >= 0.")
        if self.target_ndvi is not None and not -1 <= self.target_ndvi <= 1:
            raise ValueError("target_ndvi must be between -1 and 1, or None.")
        if not 0 <= self.target_albedo <= 1:
            raise ValueError("target_albedo must be between 0 and 1.")
        if not -1 <= self.target_ndbi <= 1:
            raise ValueError("target_ndbi must be between -1 and 1.")
