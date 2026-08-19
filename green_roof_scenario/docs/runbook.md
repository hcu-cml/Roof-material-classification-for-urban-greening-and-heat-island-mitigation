# Local Runbook

> **Note for readers of the public repository.** The commands below are the
> ad-hoc commands used for the original city runs of the study. They are kept
> for transparency and reproducibility of the reported numbers.
>
> **No data ships with this repository.** Every `data/...` path here points to a
> local dataset that you have to obtain yourself (Landsat Collection 2 Level-2
> scenes, building layers, CityGML LoD2 files). Replace every path with a link
> to your own data before running anything. The green-roof target values are
> city and acquisition specific and must not be reused for a different city.


Ad hoc commands used for the current city runs. Local inputs live under
`data/raw/` and `data/processed/`; generated rasters and GeoPackages live under
`outputs/scenarios/`. These paths are ignored by Git.

## OSM green-roof fetch notes

Wrote 18 green-roof features to data/processed/helpers/madrid_green_roofs.gpkg

Wrote 2 green-roof features to data/processed/helpers/malaga_green_roofs.gpkg

Wrote 39 green-roof features to data/processed/helpers/paris_green_roofs.gpkg

Wrote 155 green-roof features to data/processed/helpers/hamburg_green_roofs.gpkg


## Paris

green-roof-params \
  --green-roofs data/processed/helpers/paris_green_roofs.gpkg \
  --l2-folder data/raw/landsat/LC09_L2SP_199026_20240810_20240811_02_T1paris \
  --indices-out-dir outputs/scenarios/results_paris \
  --out data/processed/helpers/paris_green_roofs_with_params.gpkg \
  --summary-csv data/processed/helpers/paris_green_roof_parameter_summary.csv

green-roof-scenario \
  --l2_folder data/raw/landsat/LC09_L2SP_199026_20240810_20240811_02_T1paris \
  --buildings data/processed/buildings/paris-cls-finale-finetuned.gpkg \
  --roof_material_field predicted_roof_materials \
  --roof_materials_type "0,4" \
  --out_dir outputs/scenarios/paris \
  --build_lst \
  --model rf \
  --target_ndvi 0.40624622109631436 \
  --target_ndbi -0.05327758769465608 \
  --target_albedo 0.14593065672091077 \
  --clip_positive_delta \
  --write_indices_rasters


## Madrid

green-roof-fetch-osm \
  --buildings data/processed/buildings/madrid-cls-finale.gpkg \
  --buildings-layer buildings \
  --out data/processed/helpers/madrid_green_roofs.gpkg \
  --boundary-out data/processed/helpers/madrid_building_extent.gpkg \
  --target-crs EPSG:25830

green-roof-params \
  --green-roofs data/processed/helpers/madrid_green_roofs.gpkg \
  --l2-folder data/raw/landsat/LC09_L2SP_201032_20240723_20240724_02_T1madrid \
  --indices-out-dir outputs/scenarios/results_madrid \
  --out data/processed/helpers/madrid_green_roofs_with_params.gpkg \
  --summary-csv data/processed/helpers/madrid_green_roof_parameter_summary.csv

green-roof-scenario \
  --l2_folder data/raw/landsat/LC09_L2SP_201032_20240723_20240724_02_T1madrid \
  --buildings data/processed/buildings/madrid-cls-finale.gpkg \
  --roof_material_field predicted_roof_materials \
  --roof_materials_type "0,4" \
  --out_dir outputs/scenarios/madrid \
  --build_lst \
  --model rf \
  --target_ndvi 0.284311 \
  --target_ndbi -0.079927 \
  --target_albedo 0.173726 \
  --clip_positive_delta \
  --write_indices_rasters


## Hamburg

green-roof-fetch-osm \
  --buildings data/processed/buildings/hamburg-cls-finale.gpkg \
  --buildings-layer buildings \
  --out data/processed/helpers/hamburg_green_roofs.gpkg \
  --boundary-out data/processed/helpers/hamburg_building_extent.gpkg \
  --target-crs EPSG:25832

green-roof-params \
  --green-roofs data/processed/helpers/hamburg_green_roofs.gpkg \
  --l2-folder data/raw/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg \
  --indices-out-dir outputs/scenarios/results_hamburg \
  --out data/processed/helpers/hamburg_green_roofs_with_params.gpkg \
  --summary-csv data/processed/helpers/hamburg_green_roof_parameter_summary.csv


### Ohne slope
green-roof-scenario \
  --l2_folder data/raw/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg \
  --buildings data/processed/buildings/hamburg-cls-finale.gpkg \
  --roof_material_field predicted_roof_materials \
  --roof_materials_type "0,4" \
  --out_dir outputs/scenarios/hamburg \
  --build_lst \
  --model rf \
  --target_ndvi 0.5098841067653748 \
  --target_ndbi -0.15313389460307741 \
  --target_albedo 0.1435968737428123 \
  --clip_positive_delta \
  --write_indices_rasters

### Mit slope
green-roof-scenario \
  --l2_folder data/raw/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg \
  --buildings data/processed/buildings/hamburg-cls-finale-with-slopes.gpkg \
  --layer buildings \
  --roof_material_field predicted_roof_materials \
  --roof_materials_type "0,4" \
  --roof_slope_field roof_slope_mean_deg \
  --max_roof_slope_deg 15 \
  --out_dir outputs/scenarios/hamburg_slope \
  --build_lst \
  --model rf \
  --target_ndvi 0.5098841067653748 \
  --target_ndbi -0.15313389460307741 \
  --target_albedo 0.1435968737428123 \
  --clip_positive_delta \
  --write_indices_rasters

