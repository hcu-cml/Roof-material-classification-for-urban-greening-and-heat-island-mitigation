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
import pandas as pd
import numpy as np

# Path to your generated GPKG
gpkg_path = "results_greening_demo/buildings_greening_impact.gpkg"

try:
    # 1. Load the buildings layer
    print("Loading buildings layer...")
    gdf = gpd.read_file(gpkg_path)

    # 2. Filter for roofs that were actually changed
    # We look for a negative delta (cooling). We use a small threshold (-0.01) 
    # to filter out floating-point noise or buildings with negligible overlap.
    changed_roofs = gdf[gdf["delta_mean"] < -0.01].copy()
    
    # 3. Calculate Statistics
    num_changed = len(changed_roofs)
    
    if num_changed > 0:
        # Average cooling ONLY for the greened roofs
        avg_decrease = -changed_roofs["delta_mean"].mean()
        
        # Optional: Weighted average by roof area (more accurate for physical impact)
        # This gives more weight to large warehouses/flats than small sheds.
        avg_decrease_weighted = -np.average(
            changed_roofs["delta_mean"], 
            weights=changed_roofs.geometry.area
        )
        
        print(f"--- Results for Changed Roofs Only ---")
        print(f"Number of Greened Buildings: {num_changed}")
        print(f"Average Cooling (Per Building):  {avg_decrease:.4f} °C")
        print(f"Average Cooling (Area Weighted): {avg_decrease_weighted:.4f} °C")
        
        # Additional context: Distribution
        print(f"Max Cooling on a Single Roof:    {-changed_roofs['delta_mean'].min():.4f} °C")
        print(f"Min Cooling on a Single Roof:    {-changed_roofs['delta_mean'].max():.4f} °C")
        
    else:
        print("No buildings found with significant cooling (delta < -0.01).")
        print("Check if the correct 'roof_types' were targeted in the simulation.")

except Exception as e:
    print(f"Error processing file: {e}")