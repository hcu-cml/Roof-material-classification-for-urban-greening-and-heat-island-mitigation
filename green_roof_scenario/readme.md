# Green Roof Scenario

Green Roof Scenario is a Python package and command-line workflow for estimating how selected green-roof interventions could change land surface temperature (LST). It derives NDVI, broadband albedo, and NDBI from a Landsat 8/9 Collection 2 Level-2 scene, fits an empirical regression model against observed LST, changes the predictors only over selected roofs, and exports scenario rasters and per-building statistics.

The current package version is **0.2.0**. It supports Python 3.10 through 3.13.

> This is an empirical scenario tool, not a physical urban-climate model. Treat results as comparative estimates for the supplied scene and assumptions. Validate target parameters, input quality, and model diagnostics before using results in planning or research.

## Quick start

The easiest reproducible installation uses [uv](https://docs.astral.sh/uv/) and the committed `uv.lock` file:

```bash
git clone https://github.com/jaenixm/lst_rooftype_material.git
cd lst_rooftype_material
uv sync
uv run green-roof-scenario --help
```

You can also use a standard virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
green-roof-scenario --help
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`.

## Required input data

### Landsat scene

Pass one directory containing a single Landsat 8 or 9 Collection 2 Level-2 scene. The main scenario needs files ending in:

- `_ST_B10.TIF` for surface temperature when `--build_lst` is used;
- `_QA_PIXEL.TIF` for the cloud, shadow, snow, fill, and optional water mask;
- `_SR_B2.TIF`, `_SR_B4.TIF`, `_SR_B5.TIF`, `_SR_B6.TIF`, and `_SR_B7.TIF` for the predictors.

Band matching is case-insensitive for `.tif`/`.TIF`. The command stops if a required band is missing or if multiple files match the same band in one folder.

### Building layer

Provide a GeoPackage, GeoJSON, or shapefile containing building polygons, a valid CRS, and at least one usable selection field:

- roof material, selected with `--roof_material_field` and `--roof_materials_type`;
- roof shape, selected with `--roof_shape_field` and `--roof_shape_type`;
- numeric roof slope in degrees, selected with `--roof_slope_field` and `--max_roof_slope_deg`.

At least one selection filter is required. Multiple filters are combined with logical AND. The optional boundary and all raster/vector inputs must overlap spatially; Landsat surface-reflectance bands are reprojected onto the exact baseline LST grid.

## Run a scenario

The examples below use the Hamburg scenario from this project. The completed run selects dominant roof-material classes `0` and `4` (tar paper and concrete), requires mean roof slope `<= 15°`, uses the locally estimated Hamburg green-roof parameters, and rasterizes eligible roofs with supersampling factor `8`.

This is the command used for the reported Hamburg run. It builds a Celsius LST raster from the Hamburg Landsat scene, fits the default Random Forest model on the building extent, and writes the scenario outputs to `hamburg_slope`:

```bash
uv run green-roof-scenario \
  --l2_folder data/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg \
  --buildings data/buildings/hamburg-cls-finale-with-slopes_dominant.gpkg \
  --roof_material_field dominant_material \
  --roof_materials_type "0,4" \
  --roof_slope_field roof_slope_mean_deg \
  --max_roof_slope_deg 15 \
  --build_lst \
  --out_dir hamburg_slope \
  --target_ndvi 0.5098841067653748 \
  --target_albedo 0.1435968737428123 \
  --target_ndbi -0.15313389460307741 \
  --model rf \
  --supersample 8 \
  --write_indices_rasters \
  --write_pred_baseline
```

If the Hamburg baseline LST raster has already been built, omit `--build_lst` and pass it directly. Use a different output directory so the source raster is not overwritten while it is being read:

```bash
uv run green-roof-scenario \
  --lst hamburg_slope/baseline_LST.tif \
  --l2_folder data/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg \
  --buildings data/buildings/hamburg-cls-finale-with-slopes_dominant.gpkg \
  --roof_material_field dominant_material \
  --roof_materials_type "0,4" \
  --roof_slope_field roof_slope_mean_deg \
  --max_roof_slope_deg 15 \
  --out_dir outputs/scenarios/hamburg_from_existing_lst \
  --target_ndvi 0.5098841067653748 \
  --target_albedo 0.1435968737428123 \
  --target_ndbi -0.15313389460307741 \
  --model rf \
  --supersample 8
```

When `--boundary` is omitted, the rectangular extent of the building layer becomes the analysis extent. This is intentional in the reported Hamburg run. In the current implementation, `--boundary` is applied before spatial sampling and model fitting, so adding it changes the fitted model rather than merely cropping the finished outputs. For comparable Hamburg statistics, run the model without `--boundary` and clip the spatial outputs afterward for figures. The current Hamburg deliverables were post-processed using the `verwaltungsgrenzen__landesgrenze` layer in `data/helpers/hamburg_boundary.gpkg`; values inside that boundary were not resampled or recalculated.

The Hamburg material and slope filters are combined with logical AND:

```bash
--roof_material_field dominant_material \
--roof_materials_type "0,4" \
--roof_slope_field roof_slope_mean_deg \
--max_roof_slope_deg 15
```

Useful controls include:

- `--model rf|linear`: Random Forest is the default; linear regression is a simpler deterministic alternative.
- `--sample_frac 0.1`: fraction of the spatially thinned valid training pool to sample, with a minimum target of 1,000 pixels when available.
- `--min_sample_spacing 100`: approximate training-sample spacing in raster units (normally metres for projected Landsat data); set `0` to disable thinning.
- `--supersample 8`: Hamburg setting used to estimate fractional eligible-roof coverage in each 30 m Landsat pixel by rasterizing an `8 x 8` subgrid.
- `--min_roof_area 25`: minimum selected roof area in square metres.
- `--clip_positive_delta`: replace modeled warming with zero. This is off by default and should only be enabled when a cooling-only product is explicitly required.

Run `uv run green-roof-scenario --help` for the complete option list.

## How the calculation works

1. Landsat ST_B10 is scaled to LST and invalid QA pixels are masked.
2. Surface-reflectance bands are scaled and used to calculate NDVI, albedo, and NDBI on the LST grid.
3. Spatially thinned valid pixels train either a Random Forest or linear regression model.
4. Selected roof polygons are rasterized as fractional pixel coverage.
5. NDVI, albedo, and NDBI are blended toward the requested green-roof targets in proportion to that coverage.
6. Both the baseline and intervention predictors are evaluated by the same fitted model.
7. The intervention effect is calculated as:

   ```text
   delta_LST = modeled_scenario_LST - modeled_baseline_LST
   ```

8. That delta is applied to observed LST to produce a seam-free absolute scenario raster. Negative `delta_LST` values represent cooling.

Subtracting the modeled baseline rather than observed LST in step 7 isolates the predictor intervention and avoids embedding model bias in the delta.

## Outputs

The scenario output directory contains:

| File | Description |
| --- | --- |
| `baseline_LST.tif` | QA-masked LST built from ST_B10 when `--build_lst` is used |
| `scenario_pred_LST.tif` | Observed baseline LST plus the modeled intervention delta |
| `delta_LST.tif` | Temperature change; negative values indicate cooling |
| `buildings_greening_impact.gpkg` | Per-building baseline, scenario, delta, cooling, area, and selection fields |
| `_greening_statistics.txt` | Domain-wide, all-building, and selected-building summaries |
| `_greening_provenance.txt` | Inputs, targets, sampling settings, model metrics, and selection counts |
| `model_feature_importance.txt` | Train/test metrics plus RF feature importances or linear coefficients |
| `baseline_pred_LST.tif` | Optional fitted-model baseline prediction |
| `ndvi.tif`, `albedo.tif`, `ndbi.tif` | Optional aligned predictor rasters |
| `roof_fraction.tif` | Optional roof coverage fraction from 0 to 1 |

## Preparation utilities

### Fetch known green roofs from OpenStreetMap

This command uses Nominatim and Overpass, so it requires internet access and should be used in accordance with those services' usage policies:

```bash
uv run green-roof-fetch-osm \
  --city "Hamburg, Germany" \
  --out data/helpers/hamburg_green_roofs.gpkg \
  --boundary-out data/helpers/hamburg_boundary.gpkg \
  --target-crs EPSG:25832
```

Use the Hamburg classified-building layer's exact extent instead of the city relation:

```bash
uv run green-roof-fetch-osm \
  --buildings data/processed/buildings/hamburg-cls-finale.gpkg \
  --buildings-layer buildings \
  --out data/helpers/hamburg_green_roofs_building_extent.gpkg \
  --target-crs EPSG:25832
```

### Estimate green-roof target parameters

Calculate per-roof and area-weighted NDVI, NDBI, and albedo values for known green roofs:

```bash
uv run green-roof-params \
  --green-roofs data/helpers/hamburg_green_roofs.gpkg \
  --l2-folder data/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg \
  --indices-out-dir outputs/parameter_estimation/hamburg \
  --out data/helpers/hamburg_green_roofs_with_params.gpkg \
  --summary-csv data/helpers/hamburg_green_roof_parameter_summary.csv
```

The `area_weighted_mean` values in the summary CSV can be passed to `--target_ndvi`, `--target_ndbi`, and `--target_albedo`.

### Add CityGML roof slopes

Enrich a building layer with area-weighted LoD2 roof-plane slope metrics:

```bash
uv run green-roof-enrich-slopes \
  --buildings data/processed/buildings/hamburg-cls-finale.gpkg \
  --layer buildings \
  --citygml-dir data/raw/citygml/hamburg_lod2 \
  --out data/processed/buildings/hamburg-cls-finale-with-slopes.gpkg
```

The default scenario slope field is `roof_slope_mean_deg`.

## Python API

```python
from green_roof_scenario import ScenarioConfig, run_scenario

config = ScenarioConfig(
    l2_folder="data/landsat/LC09_L2SP_196023_20250621_20250622_02_T1hamburg",
    buildings="data/buildings/hamburg-cls-finale-with-slopes_dominant.gpkg",
    roof_material_field="dominant_material",
    roof_materials_type="0,4",
    roof_slope_field="roof_slope_mean_deg",
    max_roof_slope_deg=15,
    out_dir="hamburg_slope",
    build_lst=True,
    target_ndvi=0.5098841067653748,
    target_albedo=0.1435968737428123,
    target_ndbi=-0.15313389460307741,
    model="rf",
    supersample=8,
)

outputs = run_scenario(config)
print(outputs.delta_raster)
```

Programmatic configuration validates filter selection, input mode, model name, parameter ranges, sampling values, and rasterization settings before the run starts.

## Tests

The test suite creates temporary synthetic rasters and vectors; it does not require the large local research datasets:

```bash
uv run python -m unittest discover -s tests -v
```

If GIS commands report a `proj.db` layout-version conflict, a Conda installation is leaking PROJ/GDAL paths into another environment. Deactivate Conda, clear those variables, and try again:

```bash
conda deactivate
unset PROJ_DATA PROJ_LIB GDAL_DATA GDAL_DRIVER_PATH
uv run python -m unittest discover -s tests -v
```

## Repository layout

```text
.
├── pyproject.toml
├── uv.lock
├── readme.md
├── docs/
│   ├── methodology/Model_green_roof_effect.pdf
│   └── runbook.md
├── reports/
├── scripts/legacy/
├── src/green_roof_scenario/
│   ├── cli.py
│   ├── config.py
│   ├── l2.py
│   ├── masking.py
│   ├── modeling.py
│   ├── scenario.py
│   ├── green_roof_params.py
│   ├── osm_green_roofs.py
│   └── citygml_slopes.py
└── tests/
```

Local Landsat scenes, building datasets, GeoPackages, rasters, top-level exports, virtual environments, and generated scenario directories are intentionally excluded by `.gitignore`. Keep large or restricted datasets in external storage and document how collaborators can obtain them. The project-specific city commands are retained in [`docs/runbook.md`](docs/runbook.md). Before committing, review the exact scope with:

```bash
git status --short
git diff --stat
```

Do not use `git add -f` on ignored research data unless the repository has an explicit data-publication plan, appropriate permissions, and a large-file strategy.

## Current limitations

- Results describe one satellite acquisition and do not by themselves represent seasonal or air-temperature effects.
- Model performance is scene-dependent; inspect test R²/RMSE and spatial residual behavior.
- Green-roof target values should ideally come from comparable local roofs and the same acquisition.
- Very small roofs relative to the Landsat pixel size have fractional, mixed-pixel effects.
- OSM green-roof tags can be incomplete, and CityGML identifiers must match the building layer for slope enrichment.

## Methodology reference

[`docs/methodology/Model_green_roof_effect.pdf`](docs/methodology/Model_green_roof_effect.pdf) contains the project's detailed scientific background and modeling rationale. The implementation and test metrics remain the authoritative reference for the behavior of version 0.2.0.

## License

No open-source license has been selected yet. Add a `LICENSE` file before presenting this repository as open source or inviting reuse outside the project team.
