# Green roof rerun results for Paris, Madrid, and Hamburg

This report summarizes the local reruns completed on July 10, 2026. These reruns use corrected dominant-material parsing for roof materials. Positive cooling values mean `baseline LST - scenario LST`, in degrees Celsius.

## Landsat scene and weather context

The reruns use the same Landsat scenes and weather context as the earlier comparison report.

| City | Landsat product | Acquisition date and time | Nearest weather hour | Weather at nearest hour | Landsat cloud cover | Sun elevation |
|---|---|---:|---:|---|---:|---:|
| Paris | `LC09_L2SP_199026_20240810_20240811_02_T1` | 2024-08-10 10:40 UTC / 12:40 CEST | 13:00 CEST | 24.4 °C, 55% RH, 0.0 mm precipitation, 42% cloud cover, 4.8 km/h wind | 6.16% | 53.08° |
| Madrid | `LC09_L2SP_201032_20240723_20240724_02_T1` | 2024-07-23 10:55 UTC / 12:55 CEST | 13:00 CEST | 33.5 °C, 22% RH, 0.0 mm precipitation, 0% cloud cover, 5.8 km/h wind | 0.00% | 62.42° |
| Hamburg | `LC09_L2SP_196023_20250621_20250622_02_T1` | 2025-06-21 10:20 UTC / 12:20 CEST | 12:00 CEST | 24.6 °C, 37% RH, 0.0 mm precipitation, 1% cloud cover, 6.1 km/h wind | 0.04% | 57.82° |

## Green-roof reference parameters

The target green-roof parameters are unchanged from the earlier report and were applied as area-weighted means.

| City | Valid reference roofs | Target NDVI | Target NDBI | Target albedo |
|---|---:|---:|---:|---:|
| Paris | 37 | 0.4062 | -0.0533 | 0.1459 |
| Madrid | 18 | 0.2843 | -0.0799 | 0.1737 |
| Hamburg | 155 | 0.5099 | -0.1531 | 0.1436 |

## Prepared local rerun inputs

The three reruns use corrected dominant-material parsing on local prepared inputs. Paris and Madrid use `slope_median_deg <= 11°`. Hamburg uses `roof_slope_mean_deg <= 15°`.

| City | Prepared local input | Analysis input set: features / area | Material candidates: features / area | Eligible in rerun: features / area | Missing slope among analysis set | Notes |
|---|---|---:|---:|---:|---:|---|
| Paris | `data/buildings/paris-cls-finale-finetuned_with_slope_dominant.gpkg` | 106,391 / 4,102.2 ha | 61,630 / 2,237.8 ha | 8,603 / 616.1 ha | 24,496 / 799.1 ha | Local rerun uses numeric `slope_median_deg <= 11`. |
| Madrid | `data/buildings/madrid-cls-finale_with_slope_dominant.gpkg` | 121,006 / 5,647.5 ha | 43,020 / 2,851.9 ha | 4,987 / 755.7 ha | 0 / 0.0 ha | Local rerun uses numeric `slope_median_deg <= 11`. |
| Hamburg | `data/buildings/hamburg-cls-finale-with-slopes_dominant.gpkg` | 194,799 / 6,439.5 ha | 74,358 / 3,915.8 ha | 60,070 / 3,609.3 ha | 582 / 19.4 ha | Uses CityGML-derived `roof_slope_mean_deg` and dominant material parsing. |

## Scenario results

| City / rerun | Features in analysis | Selected candidate roofs | Candidate roof area | Features with cooling > 0.01 °C | Raster mean cooling | Area-weighted cooling, all features | Area-weighted cooling, cooled features | Max feature cooling |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Paris, corrected dominant material + slope <= 11° | 106,391 | 8,603 | 616.1 ha | 18,457 (17.3%) | 0.0106 °C | 0.0254 °C | 0.2064 °C | 2.5856 °C |
| Madrid, corrected dominant material + slope <= 11° | 121,006 | 4,987 | 755.7 ha | 12,394 (10.2%) | 0.0537 °C | 0.2444 °C | 1.1042 °C | 5.4377 °C |
| Hamburg, corrected dominant material + slope <= 15° | 194,799 | 60,070 | 3,609.3 ha | 53,876 (27.7%) | 0.0569 °C | 0.7316 °C | 1.5649 °C | 7.6172 °C |


