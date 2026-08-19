#!/usr/bin/env bash
# =============================================================================
# End-to-end example run: roof material classification -> green-roof LST scenario
#
#   >>> PUT YOUR OWN DATA LINKS IN THE CONFIGURATION BLOCK BELOW. <<<
#
# Nothing in this repository ships with data. Every variable in the block below
# points to something you have to download, license or produce yourself:
#   - an orthophoto WMS endpoint (or a folder of orthophoto GeoTIFF tiles)
#   - a building footprint layer (CityGML/LoD2 derived, GeoJSON)
#   - a Landsat 8/9 Collection 2 Level-2 scene folder
#   - a trained classification checkpoint (weights/best.pt)
#
# The script stops at the first error and prints every command it runs.
# Comment out the stages you do not need.
# =============================================================================
set -euo pipefail

# ----------------------------------------------------------------------------
# CONFIGURATION - EDIT EVERYTHING IN THIS BLOCK
# ----------------------------------------------------------------------------

# Free-text label used in output file names.
CITY="mycity"

# Coordinate reference system of your imagery AND your footprints. They must
# match; no reprojection happens between them.
EPSG=25832

# --- Part 1, inference inputs -----------------------------------------------
# Folder containing the orthophoto GeoTIFF tiles of your area of interest.
ORTHO_TILES="/PATH/TO/YOUR/ORTHOPHOTO_TILES"

# Building footprints as a GeoJSON FeatureCollection, same CRS as the imagery,
# with a gml_id property used as the building key.
FOOTPRINTS="/PATH/TO/YOUR/FOOTPRINTS.geojson"

# Trained YOLO classification checkpoint (see weights/README.md).
WEIGHTS="weights/best.pt"

# Torch device: cuda | mps | cpu
DEVICE="cpu"

# --- Part 2, scenario inputs ------------------------------------------------
# One Landsat 8/9 Collection 2 Level-2 scene folder (ST_B10, QA_PIXEL,
# SR_B2/B4/B5/B6/B7).
LANDSAT_L2="/PATH/TO/YOUR/LANDSAT_L2_SCENE_FOLDER"

# Green-roof target values. Do NOT copy these from another city: derive them for
# your own scene with `green-roof-params` (see stage 12 below) and paste the
# area_weighted_mean values here.
TARGET_NDVI="REPLACE_ME"
TARGET_NDBI="REPLACE_ME"
TARGET_ALBEDO="REPLACE_ME"

# Material codes selected for greening: 0 = concrete, 4 = tar_paper.
GREENING_MATERIALS="0,4"

# --- Optional, training data acquisition (Part 1 stages 1-3) ----------------
# Only needed if you want to rebuild the training set yourself.
WMS_URL="https://PUT-YOUR-ORTHOPHOTO-WMS-ENDPOINT-HERE"
WMS_LAYER="PUT_YOUR_WMS_LAYER_HERE"
BBOX="MINLON MINLAT MAXLON MAXLAT"   # WGS84 degrees

# ----------------------------------------------------------------------------
# END OF CONFIGURATION
# ----------------------------------------------------------------------------

set -x

WORK="work/${CITY}"
mkdir -p "${WORK}"

# === Part 1 - inference ======================================================

# 7) mosaic the orthophoto tiles and cut fixed-size patches
python scripts/infer/tile_orthophoto.py \
  --in_dir  "${ORTHO_TILES}" \
  --out_dir "${WORK}/patches"

# 8) one footprint-masked GeoTIFF chip per building
python scripts/infer/mask_chips_by_footprint.py \
  --dir        "${WORK}/patches" \
  --out_dir    "${WORK}/chips" \
  --footprints "${FOOTPRINTS}" \
  --min_px     100 \
  --dilate_px  2

# 9) classify and write the enriched footprints
python scripts/infer/predict_roof_materials.py \
  --weights     "${WEIGHTS}" \
  --chips_dir   "${WORK}/chips" \
  --footprints  "${FOOTPRINTS}" \
  --out_geojson "${WORK}/${CITY}_roof_materials.geojson" \
  --device      "${DEVICE}" \
  --dst_epsg    "${EPSG}" \
  --tile 192 --stride 96 --pad 32 \
  --patch_min_roof_cov 0.55 \
  --nondefault_min_conf 0.6 \
  --margin_over_default 0.05 \
  --min_island_px 100 \
  --min_cov_frac 0.02 \
  --max_materials 5 \
  --min_px 100 \
  --min_iou 0.1

# 10) optional: add a scalar dominant_material column
python scripts/infer/add_dominant_material.py \
  --in_geojson  "${WORK}/${CITY}_roof_materials.geojson" \
  --out_geojson "${WORK}/${CITY}_roof_materials_dominant.geojson"

# === Part 2 - green-roof cooling scenario ====================================
# Requires: pip install -e ./green_roof_scenario

# 11) known green roofs inside the same extent, from OpenStreetMap
green-roof-fetch-osm \
  --buildings  "${WORK}/${CITY}_roof_materials_dominant.geojson" \
  --out        "${WORK}/${CITY}_green_roofs.gpkg" \
  --target-crs "EPSG:${EPSG}"

# 12) derive the city-specific NDVI / NDBI / albedo targets from those roofs,
#     then paste the area_weighted_mean values into the config block above
green-roof-params \
  --green-roofs     "${WORK}/${CITY}_green_roofs.gpkg" \
  --l2-folder       "${LANDSAT_L2}" \
  --indices-out-dir "${WORK}/parameter_estimation" \
  --out             "${WORK}/${CITY}_green_roofs_with_params.gpkg" \
  --summary-csv     "${WORK}/${CITY}_green_roof_parameter_summary.csv"

# 13) run the scenario
green-roof-scenario \
  --l2_folder "${LANDSAT_L2}" \
  --buildings "${WORK}/${CITY}_roof_materials_dominant.geojson" \
  --roof_material_field dominant_material \
  --roof_materials_type "${GREENING_MATERIALS}" \
  --out_dir   "${WORK}/scenario" \
  --build_lst \
  --model rf \
  --target_ndvi   "${TARGET_NDVI}" \
  --target_ndbi   "${TARGET_NDBI}" \
  --target_albedo "${TARGET_ALBEDO}" \
  --clip_positive_delta \
  --write_indices_rasters

set +x
echo "Done. Scenario outputs in ${WORK}/scenario"
