import os
import leafmap
import geopandas as gpd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import folium
from folium.plugins import Draw
from shapely.geometry import Polygon
from streamlit_folium import st_folium
from leafmap import WhiteboxTools

# Define a function for processing DEM
def process_dem(aoi_polygon, dem_filename, output_path):
    # Get the current working directory
    cwd = os.getcwd()

    # Set up WhiteboxTools and set file paths for processing
    wbt = WhiteboxTools()
    leaf_path = fr'{cwd}\\data-inputs\\leafmap'
    wbt.set_working_dir(leaf_path)

    # Pull the DEM using leafmap
    leafmap.get_3dep_dem(
        aoi_polygon,
        resolution=10,
        output=f"{output_path}\\{dem_filename}.tif",
        dst_crs="EPSG:4326",
        to_cog=True
    )

    # Smooth the DEM with feature-preserving smoothing
    smoothed_filename = fr"{dem_filename}_smoothed"
    wbt.feature_preserving_smoothing(
        f"{dem_filename}.tif",  # The input DEM filename
        f"{smoothed_filename}.tif",  # The output filename
        filter=3  # Filter size
    )

    # Breach depressions on the smoothed DEM
    conditioned_dem_filename = f"{smoothed_filename}_conditioned"
    wbt.breach_depressions(
        f'{smoothed_filename}.tif',  # Input DEM
        f"{conditioned_dem_filename}.tif"  # Output name
    )

    return f"{conditioned_dem_filename}.tif"

# Define a function for plotting the DEM
def plot_dem(dem_path):
    # Open the DEM file
    with rasterio.open(dem_path) as src:
        dem = src.read(1)  # Read the first band
        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]  # Get spatial extent

    # Mask no-data values
    dem = np.where(dem == src.nodata, np.nan, dem)

    # Plot the DEM
    plt.figure(figsize=(8, 6))
    plt.imshow(dem, cmap="terrain", extent=extent, origin="upper")
    plt.colorbar(label="Elevation (m)")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Digital Elevation Model (DEM)")
    plt.show()

# Define Streamlit app layout and functionality
st.title("Draw a Polygon, Process DEM, and View Results")

# Create the map and draw tool
m = folium.Map(location=[27.6380, -80.3984], zoom_start=12)  # Vero Beach, Florida
Draw(export=False).add_to(m)

# Layout with map and sidebar
c1, c2 = st.columns(2)
with c1:
    output = st_folium(m, width=700, height=500)

with c2:
    # Grab raw coordinate data from drawing
    coords = output.get("last_active_drawing", {}).get("geometry", {}).get("coordinates")

    if coords:
        # Extract exterior ring (first part of outer polygon)
        flat_coords = coords[0]  # assumes Polygon with no holes
        polygon = Polygon(flat_coords)

        # Add button to compute area
        if st.button("Compute Area"):
            try:
                # Process DEM for the drawn polygon
                st.info("Processing DEM...")
                processed_dem_path = process_dem(polygon, "dem_filename", "output_path")

                # Plot the processed DEM (conditioned DEM)
                st.success("DEM Processing Complete!")
                plot_dem(processed_dem_path)

            except Exception as e:
                st.error(f"Error calculating area or processing DEM: {e}")
    else:
        st.write("Draw a polygon on the map to enable area calculation.")
