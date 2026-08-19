# -----------------------------------------------------------------------------
# LEGACY / AD-HOC SCRIPT
# Kept for transparency: this is one of the helper scripts used during the
# original city runs of the study. It is not part of the supported CLI.
#
# ALL FILE PATHS AND DEFAULTS BELOW ARE EXAMPLES FROM THOSE RUNS.
# No data ships with this repository: replace every path with a link to your
# own data before running, either by editing the constants/defaults in this
# file or by passing the corresponding command-line arguments.
# -----------------------------------------------------------------------------

from pathlib import Path

import geopandas as gpd
import json
from shapely.geometry import shape

# -------------------------------------------------------------------
# USER INPUTS – EDIT THESE
# -------------------------------------------------------------------
GEOJSON_FOLDER = Path(r"data/geojson_hamburg")    # folder with many .geojson files
BUILDINGS_GPKG = Path(r"data/hamburg_osm_building_footprints.gpkg")   # building footprints GPKG
BUILDINGS_LAYER = None  # e.g. "buildings"; use None if there is only one layer
OUTPUT_GPKG = Path(r"data/buildings_enriched_hamburg.gpkg")
OUTPUT_LAYER = "buildings_enriched"
PREDICTION_FIELD = "predicted_roof_materials"  # field name in geojson
# -------------------------------------------------------------------


def load_and_merge_geojson(folder: Path, epsg: int = 25832) -> gpd.GeoDataFrame:
    geojson_files = (
        list(folder.rglob("*.geojson"))
        + list(folder.rglob("*.geojsonl"))
        + list(folder.rglob("*.json"))
    )

    if not geojson_files:
        raise FileNotFoundError(f"No GeoJSON files found in {folder}")

    def _take_first(value):
        """Return first element of a list, or the value itself if not a list."""
        if isinstance(value, list):
            return value[0] if value else None
        return value

    gdfs = []
    for fp in geojson_files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        records = []
        for feat in data.get("features", []):
            props = (feat.get("properties") or {}).copy()

            # Flatten list-valued attributes we care about
            if PREDICTION_FIELD in props:
                props[PREDICTION_FIELD] = _take_first(props[PREDICTION_FIELD])

            if "material_cov" in props:
                props["material_cov"] = _take_first(props["material_cov"])

            geom = shape(feat.get("geometry")) if feat.get("geometry") else None
            if geom is None:
                continue

            props["geometry"] = geom
            records.append(props)

        if not records:
            continue

        gdf = gpd.GeoDataFrame(records, geometry="geometry", crs=f"EPSG:{epsg}")
        gdfs.append(gdf)

    if not gdfs:
        raise ValueError("No valid features found in any GeoJSON file.")

    merged = gpd.pd.concat(gdfs, ignore_index=True)

    # Ensure correct CRS on merged (assign or reproject to EPSG:25832)
    if merged.crs is None:
        merged = merged.set_crs(epsg, allow_override=True)
    elif merged.crs.to_epsg() != epsg:
        merged = merged.to_crs(epsg)

    return merged


def load_buildings(buildings_path: Path, layer: str | None, epsg: int = 25832) -> gpd.GeoDataFrame:
    if layer:
        blds = gpd.read_file(buildings_path, layer=layer)
    else:
        blds = gpd.read_file(buildings_path)

    # Reproject buildings to EPSG:25832 as well
    if blds.crs is None:
        blds = blds.set_crs(epsg, allow_override=True)
    elif blds.crs.to_epsg() != epsg:
        blds = blds.to_crs(epsg)

    # Give buildings a stable ID if they don't have one
    if "bld_id_tmp" not in blds.columns:
        blds = blds.reset_index().rename(columns={"index": "bld_id_tmp"})

    return blds


def join_biggest_overlap(buildings: gpd.GeoDataFrame,
                         roofs: gpd.GeoDataFrame,
                         prediction_field: str) -> gpd.GeoDataFrame:
    """
    2. Create spatial index (implicitly via sjoin).
    3. Join attributes by location (intersection), one-to-one,
       choosing the intersecting roof polygon with the biggest overlap.
    """

    # Make sure both datasets have a spatial index (this will create it if not existing)
    _ = buildings.sindex
    _ = roofs.sindex

    # Prepare roofs with an explicit geometry column we can keep after sjoin
    roofs_for_join = roofs[[prediction_field, "geometry"]].copy()
    roofs_for_join["roof_geom"] = roofs_for_join.geometry

    # Spatial join: get all intersecting roof polygons for each building
    joined = gpd.sjoin(
        buildings,
        roofs_for_join,
        how="left",
        predicate="intersects",
    )

    # Compute intersection area (in CRS units – meters for EPSG:25832)
    joined["intersect_area"] = joined.geometry.intersection(
        joined["roof_geom"]
    ).area

    # For each building, keep the roof with the largest intersection area
    # (note: 'bld_id_tmp' is our building key)
    joined_sorted = joined.sort_values(["bld_id_tmp", "intersect_area"], ascending=[True, False])

    # Drop duplicates per building, keeping first (largest area)
    best = joined_sorted.drop_duplicates(subset="bld_id_tmp", keep="first")

    # Now bring the predicted_roof_materials back to a clean buildings GeoDataFrame
    # We just need building columns + chosen prediction field
    cols_buildings = [c for c in buildings.columns if c != "geometry"]
    best = best[cols_buildings + ["geometry", prediction_field]]

    # Optional: rename field in case you want a cleaner name in buildings
    # best = best.rename(columns={prediction_field: "roof_material"})

    # Keep only the first material if multiple are present
    best[prediction_field] = (
        best[prediction_field]
        .astype(str)
        .str.split(',')
        .str[0]
    )

    return best


def main():
    print("Loading GeoJSON predictions...")
    roofs = load_and_merge_geojson(GEOJSON_FOLDER, epsg=25832)

    print("Loading building footprints...")
    buildings = load_buildings(BUILDINGS_GPKG, BUILDINGS_LAYER, epsg=25832)

    print("Performing spatial join with biggest overlap logic...")
    buildings_enriched = join_biggest_overlap(buildings, roofs, PREDICTION_FIELD)

    print(f"Exporting enriched buildings to {OUTPUT_GPKG} (layer: {OUTPUT_LAYER})...")
    # 4. Export enriched building layer as GPKG
    buildings_enriched.to_file(OUTPUT_GPKG, layer=OUTPUT_LAYER, driver="GPKG")

    print("Done.")


if __name__ == "__main__":
    main()