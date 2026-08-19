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

import geopandas as gpd

# Load the geopackage file
gpkg_path = "data/citygml_roof_materials_hamburg_all_new.gpkg"
gdf = gpd.read_file(gpkg_path)

# Check for duplicates in gml_id column
duplicates = gdf[gdf.duplicated(subset=['gml_id'], keep=False)]

if len(duplicates) > 0:
    print(f"Found {len(duplicates)} duplicate entries:")
    print(duplicates[['gml_id']].sort_values('gml_id'))
else:
    print("No duplicates found in gml_id column")

# Optional: Get count of duplicates
duplicate_counts = gdf['gml_id'].value_counts()
duplicate_counts = duplicate_counts[duplicate_counts > 1]
print(f"\nDuplicate gml_id values and their counts:\n{duplicate_counts}")