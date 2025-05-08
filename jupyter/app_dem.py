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
import tempfile
from leafmap import WhiteboxTools

# Define a function for processing DEM
def process_dem(aoi_polygon, dem_filename, output_path):
    # Create a temporary directory to store intermediate files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up WhiteboxTools with the temporary directory as working directory
        wbt = WhiteboxTools()
        wbt.set_working_dir(temp_dir)

        # Pull the DEM using leafmap
        leafmap.get_3dep_dem(
            aoi_polygon,
            resolution=10,
            output=os.path.join(temp_dir, f"{dem_filename}.tif"),
            dst_crs="EPSG:4326",
            to_cog=True
        )

        # Smooth the DEM with feature-preserving smoothing
        smoothed_filename = f"{dem_filename}_smoothed"
        wbt.feature_preserving_smoothing(
            os.path.join(temp_dir, f"{dem_filename}.tif"),  # The input DEM filename
            os.path.join(temp_dir, f"{smoothed_filename}.tif"),  # The output filename
            filter=3  # Filter size
        )

        # Breach depressions on the smoothed DEM
        conditioned_dem_filename = f"{smoothed_filename}_conditioned"
        wbt.breach_depressions(
            os.path.join(temp_dir, f'{smoothed_filename}.tif'),  # Input DEM
            os.path.join(temp_dir, f"{conditioned_dem_filename}.tif")  # Output name
        )

        # Define the final path for the conditioned DEM
        final_dem_path = os.path.join(output_path, f"{conditioned_dem_filename}.tif")
        
        # If the file already exists, remove it to avoid the error
        if os.path.exists(final_dem_path):
            os.remove(final_dem_path)
        
        # Move the conditioned DEM to the output path (permanent location)
        os.rename(os.path.join(temp_dir, f"{conditioned_dem_filename}.tif"), final_dem_path)

        # Return the path of the processed DEM (conditioned DEM)
        return final_dem_path

# Define a function for plotting the DEM
def plot_dem(dem_path):
    st.info("Plotting DEM...")
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
    st.info("Plotting complete!")

# Define Streamlit app layout and functionality
st.title("Draw a Polygon, Process DEM, and View Results")

# Create a permanent output folder (outside of temporary directory)
output_path = "processed_dem_files"
os.makedirs(output_path, exist_ok=True)

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
                processed_dem_path = process_dem(polygon, "dem_filename", output_path)

                # Plot the processed DEM (conditioned DEM)
                st.info("DEM Processing Complete!")
                plot_dem(processed_dem_path)

            except Exception as e:
                st.error(f"Error calculating area or processing DEM: {e}")
    else:
        st.write("Draw a polygon on the map to enable area calculation.")
