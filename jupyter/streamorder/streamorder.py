make_catchments_b#!/usr/bin/env python
# coding: utf-8

# In[1]:


import folium
import jupyter.streamorder.make_catchments_b as make_catchments_b


# In[ ]:


import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Open the source raster
with rasterio.open("Terrain_Melbourne-2.tif") as src:
    # Open the target raster (the one you want to match)
    with rasterio.open("colorado_sample_dem.tiff") as dst:
        # Calculate the transformation parameters
        transform, width, height = calculate_default_transform(
            src.crs, dst.crs, src.width, src.height, *src.bounds
        )

        # Create an empty array for the reprojected data
        reprojected_data = np.empty((src.count, height, width), dtype=src.dtypes[0])

        # Reproject the data
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst.crs,
            resampling=Resampling.nearest  # Choose a suitable resampling method
        )

        # Save the reprojected raster
        with rasterio.open(
            "reprojected_raster.tif",
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=src.count,
            dtype=src.dtypes[0],
            crs=dst.crs,
            transform=transform,
        ) as dst:
            dst.write(reprojected_data)


# In[30]:


basins, branches = make_catchments_b.generate_catchments('reprojected_raster.tif',acc_thresh=3000,so_filter=4)


# In[31]:


# Visualize output
gdf = basins.copy()
# separate by stream order
ones = gdf[gdf['Order']==1]
twos = gdf[gdf['Order']==2]
threes = gdf[gdf['Order']==3] # skip fours

# Map 'em!
cols = ['Index','Length','Relief','Order','Slope','AreaSqKm','LocalPP_X','LocalPP_Y','Final_Chain_Val','BasinGeo']
cols_branches = ['Index','Length','Relief','Order','Slope','geometry','LocalPP_X','LocalPP_Y','Final_Chain_Val']

m = branches[cols_branches].explore(color='black')
ones[cols].explore(m=m,color='blue',tiles='Stamen Terrain')
twos[cols].explore(m=m,color='purple',tiles='Stamen Terrain')
threes[cols].explore(m=m,color='yellow',tiles='Stamen Terrain')
folium.LayerControl().add_to(m) 
m


# In[28]:


gdf


# In[10]:


import rasterio

# Open the DEM file
with rasterio.open("Terrain_Melbourne-2.tif") as dataset:
    # Get the coordinate reference system (CRS)
    crs = dataset.crs

print(crs)


# In[16]:


import rasterio
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Open the source raster
with rasterio.open("Terrain_Melbourne-2.tif") as src:
    # Open the target raster (the one you want to match)
    with rasterio.open("colorado_sample_dem.tiff") as dst:
        # Calculate the transformation parameters
        transform, width, height = calculate_default_transform(
            src.crs, dst.crs, src.width, src.height, *src.bounds
        )

        # Create an empty array for the reprojected data
        reprojected_data = np.empty((src.count, height, width), dtype=src.dtypes[0])

        # Reproject the data
        reproject(
            source=rasterio.band(src, 1),
            destination=reprojected_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst.crs,
            resampling=Resampling.nearest  # Choose a suitable resampling method
        )

        # Save the reprojected raster
        with rasterio.open(
            "reprojected_raster.tif",
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=src.count,
            dtype=src.dtypes[0],
            crs=dst.crs,
            transform=transform,
        ) as dst:
            dst.write(reprojected_data)


# In[ ]:




