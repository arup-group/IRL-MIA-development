import os
import folium
import geopandas as gpd
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import Polygon
import leafmap
import tempfile
import rasterio
import numpy as np
import matplotlib.pyplot as plt

from streamlit_folium import st_folium

# st.set_page_config(layout="wide")
st.title("Draw a Polygon to Calculate Area and Download DEM")

# Centered on Vero Beach, FL
m = folium.Map(location=[27.6386, -80.3973], zoom_start=10)

# Allow only one polygon
draw_options = {
    "polyline": False,
    "rectangle": False,
    "circle": False,
    "circlemarker": False,
    "marker": False,
    "polygon": {
        "allowIntersection": False,
        "showArea": True,
        "drawError": {"color": "#e1e100", "message": "You can't draw intersecting polygons!"},
        "shapeOptions": {"color": "#97009c"},
        "repeatMode": False,
    },
}
Draw(export=False, draw_options=draw_options, edit_options={"edit": False}).add_to(m)

# Map display (top section)
output = st_folium(m, width=700, height=500)

# Coordinates of the drawn polygon (if available)
coords = output.get("last_active_drawing", {}).get("geometry", {}).get("coordinates") if output else None

# Display Area and DEM Download buttons and the results below the map
if coords:
    flat_coords = coords[0]
    polygon = Polygon(flat_coords)
    gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon], crs="EPSG:4326")

    # Area Calculation
    if st.button("Compute Area"):
        try:
            gdf_proj = gdf.to_crs(epsg=3857)
            area_sqm = gdf_proj.area.iloc[0]
            area_acres = area_sqm * 0.000247105
            st.success(f"Area: {area_sqm:,.2f} m² ({area_acres:,.2f} acres)")
        except Exception as e:
            st.error(f"Error calculating area: {e}")

    # DEM Download and Plotting
    if st.button("Download and Plot DEM"):
        with st.spinner('Loading DEM... Please wait.'):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    dem_path = os.path.join(tmpdir, "temp_dem.tif")

                    # Download DEM
                    leafmap.get_3dep_dem(
                        gdf,
                        resolution=10,
                        output=dem_path,
                        dst_crs="EPSG:4326",
                        to_cog=True
                    )

                    # Plot DEM
                    with rasterio.open(dem_path) as src:
                        dem = src.read(1)
                        extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
                        dem = np.where(dem == src.nodata, np.nan, dem)

                    fig, ax = plt.subplots(figsize=(8, 6))
                    im = ax.imshow(dem, cmap="terrain", extent=extent, origin="upper")
                    plt.colorbar(im, ax=ax, label="Elevation (m)")
                    ax.set_title("Digital Elevation Model (DEM)")
                    ax.set_xlabel("Longitude")
                    ax.set_ylabel("Latitude")
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error fetching or plotting DEM: {e}")

else:
    st.info("Draw a polygon on the map to get started.")
