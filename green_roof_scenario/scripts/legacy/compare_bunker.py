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
import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import box
import numpy as np
import os

def calculate_weighted_temperature(raster_path, bunker_geometry):
    """
    Berechnet die gewichtete Durchschnittstemperatur für eine Geometrie,
    indem der Überlappungsgrad jedes Pixels als Gewichtung genutzt wird.
    """
    with rasterio.open(raster_path) as src:
        # 1. Sicherstellen, dass die Geometrie im gleichen CRS ist wie das Raster
        # Hinweis: Wir gehen davon aus, dass 'bunker_geometry' bereits im korrekten CRS ist
        # oder vor dem Aufruf transformiert wurde.
        
        # 2. Ein "Fenster" (Window) lesen, das nur den relevanten Ausschnitt lädt
        # Das spart Speicher und Zeit.
        minx, miny, maxx, maxy = bunker_geometry.bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        
        # Daten und die Transformation für dieses Fenster laden
        data = src.read(1, window=window)
        win_transform = src.window_transform(window)
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        # 3. Durch jeden Pixel im Fenster iterieren
        height, width = data.shape
        for row in range(height):
            for col in range(width):
                val = data[row, col]
                
                # Leere oder ungültige Werte (NoData) überspringen
                if val == src.nodata or np.isnan(val):
                    continue
                
                # 4. Die Geometrie des einzelnen Pixels rekonstruieren
                # Wir erstellen ein Rechteck (Polygon) für diesen Pixel
                # Die Koordinaten ergeben sich aus der Affine-Transformation
                x_left, y_top = win_transform * (col, row)
                x_right, y_bottom = win_transform * (col + 1, row + 1)
                
                # Achtung: Je nach CRS kann y_top < y_bottom sein, daher min/max sortieren
                pixel_poly = box(
                    min(x_left, x_right), 
                    min(y_bottom, y_top), 
                    max(x_left, x_right), 
                    max(y_bottom, y_top)
                )
                
                # 5. Schnittfläche berechnen (Intersection)
                # Das ist der entscheidende Schritt für "partly overlapping pixels"
                intersection = pixel_poly.intersection(bunker_geometry)
                
                if not intersection.is_empty:
                    weight = intersection.area
                    
                    # Optional: Gewichtung relativ zur Pixelgröße (0.0 bis 1.0)
                    # weight_fraction = weight / pixel_poly.area 
                    
                    weighted_sum += val * weight
                    total_weight += weight

        if total_weight == 0:
            return None
        
        return weighted_sum / total_weight

def main():
    # --- KONFIGURATION ---
    # Pfade zu Ihren Dateien anpassen
    path_bunker = "data/bunker_hamburg.gpkg"
    path_raster_2019 = "lst_29062019.tif"
    path_raster_2025 = "results_greening_hamburg_rf_clipped/baseline_LST.tif"
    
    # 1. Bunker Footprint laden
    print("Lade Bunker-Footprint...")
    gdf = gpd.read_file(path_bunker)
    
    # Wir nehmen an, der Bunker ist das erste (oder einzige) Polygon im File
    bunker_poly_original = gdf.geometry.iloc[0]
    
    # Hilfsfunktion, um CRS-Probleme zu vermeiden
    def process_raster(raster_path, poly_gdf):
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
        
        # Reprojezieren des Vektors in das Koordinatensystem des Rasters
        # Das ist extrem wichtig, damit die Pixel an der richtigen Stelle liegen!
        poly_reprojected = poly_gdf.to_crs(raster_crs).geometry.iloc[0]
        
        return calculate_weighted_temperature(raster_path, poly_reprojected)

    # 2. Berechnung 2019
    print(f"Analysiere 2019: {path_raster_2019}")
    temp_2019 = process_raster(path_raster_2019, gdf)
    
    # 3. Berechnung 2025
    print(f"Analysiere 2025: {path_raster_2025}")
    temp_2025 = process_raster(path_raster_2025, gdf)
    
    # 4. Ergebnis
    print("-" * 30)
    if temp_2019 is not None:
        print(f"Ø Temperatur 2019 (Vorher):  {temp_2019:.2f} °C")
    else:
        print("Keine Überlappung oder Daten für 2019 gefunden.")
        
    if temp_2025 is not None:
        print(f"Ø Temperatur 2025 (Nachher): {temp_2025:.2f} °C")
    else:
        print("Keine Überlappung oder Daten für 2025 gefunden.")
        
    if temp_2019 and temp_2025:
        delta = temp_2025 - temp_2019
        print(f"Veränderung: {delta:+.2f} °C")
        if delta < 0:
            print("Erfolg: Der Bunker ist kühler geworden!")
        else:
            print("Hinweis: Der Bunker ist wärmer geworden.")

if __name__ == "__main__":
    main()