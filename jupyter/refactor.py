
import streamlit as st
import os
import subprocess

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import rasterio
from datetime import datetime
from collections import defaultdict
import geopandas as gpd
from shapely import geometry
# from streamlit_folium import st_folium

# sample inputs
# file = 'Terrain_Grant_Valkaria_ClipNoData_NAD83'
# aggregation = 16
# flow_file_path = 'IRL-Flowlines-Export_NAD83.shp'
# burn_width=4
# burn_value = -3  # Adjust this value as needed
# river_network_min_flow_acc = 1000
# min_total_pond_area = 20

## Define Key Functions

def read_reproject_dem(c_path, file):
    # Import initial 1m resolution DEM (to be downsampled)
    file = file
    dem_path = fr'{c_path}jupyter\\data-inputs\\{file}.tif'

    grid = Grid.from_raster(dem_path)
    grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
    dem = grid.read_raster(dem_path)


    # For the pysheds wrapper, convert to web mercator
    import rasterio
    import numpy as np
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    # Open the source raster
    with rasterio.open(dem_path) as src:
        # Open the target raster (the one you want to match)
        with rasterio.open("streamorder\\colorado_sample_dem.tiff") as dst:
            # Calculate the transformation parameters
            print(src.crs)
            print(dst.crs)
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
                "data-inputs\\temp_reprojected_raster.tif",
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


    dem_path = 'data-inputs\\temp_reprojected_raster.tif'
    grid = Grid.from_raster(dem_path)
    grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
    dem = grid.read_raster(dem_path)

    return dem_path, grid, dem

def initialize_pdf(c_path, file, epsg, units, aggregation, flow_file_path, burn_width, burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds):
    # Create a PDF file to save the plots
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    # Get the current datetime and format it
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H%M")  # Format: 2024-11-22_0230

    pdf_path = f'{c_path}jupyter\\outputs\\Output_{file}_{burn_value}Burn_{min_total_pond_area}MinTotPondArea_{max_num_ponds}MaxNumPonds_{datetime_str}.pdf'
    pdf_pages = PdfPages(pdf_path)
    
    # Add user inputs as a page
    # Create a new figure for the user inputs
    plt.figure(figsize=(8, 6))
    plt.axis('off')  # Turn off the axis

    # Prepare the text to display
    user_inputs_text = f"""
    Report Parameters:

    DEM File Name: {file}
    EPSG Code: {epsg}
    Units of DEM: {units}
    DEM Aggregation Factor: {aggregation}
    Clipped Flowlines Path: {flow_file_path}
    Burn Width: {burn_width}
    Burn Value: {burn_value}
    Minimum Flow Accumulation - Channels: {river_network_min_flow_acc}
    Minimum Total Pond Area per Microwatershed: {min_total_pond_area}
    Max Number of Ponds per Microwatershed: {max_num_ponds}
    """

    # Add the text to the figure
    plt.text(0.1, 0.5, user_inputs_text, fontsize=12, ha='left', va='center', wrap=True)

    # Save the figure to the PDF
    pdf_pages.savefig()
    plt.close()

    return pdf_path, pdf_pages

def aggregate_dem(c_path, dem_path, aggregation):
    # Reduce DEM resolution

    import rasterio
    from rasterio.enums import Resampling

    # PARAMETER: set the upscale factor for downsampling
    aggregation = aggregation
    upscale_factor = 1/aggregation

    with rasterio.open(dem_path) as dataset:
        # Resample data to target shape
        dem = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * upscale_factor),
                int(dataset.width * upscale_factor)
            ),
            resampling=Resampling.bilinear
        )

        # Scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / dem.shape[-1]),
            (dataset.height / dem.shape[-2])
        )

        # Write the downsampled data to a new file
        with rasterio.open(
            fr'{c_path}jupyter\\data-inputs\\{file}_{aggregation}_Agg.tif',
            'w',
            driver='GTiff',
            height=dem.shape[1],
            width=dem.shape[2],
            count=dataset.count,
            dtype=dem.dtype,
            crs=dataset.crs,
            transform=transform
        ) as dst:
            dst.write(dem)

    dem_agg_path = fr'{c_path}jupyter\\data-inputs\\{file}_{aggregation}_Agg.tif'
    grid = Grid.from_raster(dem_agg_path)
    grid_clip = Grid.from_raster(dem_agg_path) # to be clipped by the delineation extent to preserve the original grid
    dem = grid.read_raster(dem_agg_path)

    print("Downsampling complete")

    return dem_agg_path, grid, dem

def confirm_crs(dem_agg_path):
    # LOG: Confirm CRS
    import rasterio
    # Open the raster file
    with rasterio.open(dem_agg_path) as src:
        crs_dem = src.crs
        print(crs_dem)

    # set crs variable to whatever the crs is
    # for now, manually, becuase the rasterio lib isn't returning the CRS correctly for the FLA projection
    crs_dem = epsg
    # crs_dem = str(crs_dem).replace("EPSG:", "")

    return crs_dem

def burn_flowlines(c_path, crs_dem, dem_agg_path, burn_width, burn_value, dem, file, aggregation):
    # Burn the NHD Flowlines into the DEM

    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    import numpy as np

    # Load the flowlines dataset
    # flow_file_path = 'IRL-Flowlines-Export_NAD83.shp'
    flowlines = gpd.read_file(fr'{c_path}jupyter\\data-inputs\\IRL-Flowlines\\{flow_file_path}')
    flowlines.to_crs(epsg=crs_dem, inplace=True)


    burn_width=burn_width
    # PARAMETER: burn width
    flowlines['geometry'] = flowlines.geometry.buffer(burn_width)  # Buffer by half the desired width


    # Ensure the flowlines are in the same CRS as the DEM
    # file = 'Terrain_Grant_Valkaria_ClipNoData_AggMedian16_NAD83'
    # NOTE: replace dem_agg_path with dem_path if the dem isn't being aggregated
    with rasterio.open(dem_agg_path) as src:
        # flowlines = flowlines.to_crs(src.crs)
        transform = src.transform
        out_shape = src.shape


    # Rasterize the flowlines
    flowline_raster = rasterize(
        [(geom, 1) for geom in flowlines.geometry],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype='uint8',
        all_touched=True
    )

    # Read the DEM
    with rasterio.open(dem_agg_path) as src:
        dem = src.read(1)  # Read the first band

    # PARAMETER: Burn the flowlines into the DEM
    burn_value = burn_value  # Adjust this value as needed
    dem_burned = np.where(flowline_raster == 1, dem + burn_value, dem)

    # Save the modified DEM
    with rasterio.open(
        fr'{c_path}jupyter\\data-inputs\\{file}_{aggregation}_Agg_Burned.tif', 
        'w', 
        driver='GTiff', 
        height=dem_burned.shape[0], 
        width=dem_burned.shape[1], 
        count=1, 
        dtype=dem_burned.dtype, 
        crs=src.crs, 
        transform=src.transform
    ) as dst:
        dst.write(dem_burned, 1)

    dem_agg_burn_path = fr'{c_path}jupyter\\data-inputs\\{file}_{aggregation}_Agg_Burned.tif'
    grid = Grid.from_raster(dem_agg_burn_path)
    grid_clip = Grid.from_raster(dem_agg_burn_path) # to be clipped by the delineation extent to preserve the original grid
    dem = grid.read_raster(dem_agg_burn_path)

    print("Flowlines have been burned into the DEM and saved as a new file.")

    return grid, dem, dem_agg_burn_path

def plot_burned_dem(dem, grid, pdf_pages, crs_dem, aggregation, burn_value, burn_width, units):
    # Plot the DEM

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colors
    import seaborn as sns

    # Define the normalization for the elevation between 0 and 15 meters
    norm = colors.Normalize(vmin=0, vmax=15)

    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)

    # Plot the DEM with normalization applied
    plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1)
    #plt.imshow(dem, extent=grid.extent, cmap='terrain', zorder=1)
    plt.colorbar(label='Elevation (m)')
    plt.grid(zorder=0)
    plt.title(f'Digital Elevation Map - {aggregation}x{aggregation} - {burn_value} depth, {burn_width*2} wide Burn', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    # Save plot to pdf
    pdf_pages.savefig(fig)

    return pdf_pages

def delineate_microwatersheds(dem_agg_burn_path, river_network_min_flow_acc):
    # NEW METHOD to delineate the catchments and stream orders 
    import make_catchments

    basins_, branches_ = make_catchments.generate_catchments(dem_agg_burn_path,acc_thresh=river_network_min_flow_acc,so_filter=4, shoreline_clip=True)

    # Visualize output
    mws_with_stream_order = basins_.copy()
    # mws_with_stream_order

    # To use the basins wrapper output (switch the use the gdf)
    microwatersheds_gdf = mws_with_stream_order

    return branches_, microwatersheds_gdf

def plot_microwatersheds(dem, grid, microwatersheds_gdf, branches_, pdf_pages):
    # Create a figure for plotting all catchments
    fig, ax = plt.subplots(figsize=(8,6))
    norm = colors.Normalize(vmin=0, vmax=15)
    plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
    # Set the plot boundaries and aspect ratio
    plt.xlim(grid.bbox[0], grid.bbox[2])
    plt.ylim(grid.bbox[1], grid.bbox[3])
    plt.gca().set_aspect('equal')
    microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
    branches_.plot(ax=ax, aspect=1, color='black')
    # Plot ponds
    # Add title and show the combined plot
    plt.title('Microwatersheds - Delineated from Channel Junction Points')
    plt.show()
    # Save plot to pdf
    pdf_pages.savefig(fig)

    return pdf_pages

def calc_mws_areas(microwatersheds_gdf):
    # Calculate areas - tabulate

    # Add a simple numeric 'Microwatershed_ID' column starting at 1
    microwatersheds_gdf['Microwatershed_ID'] = range(1, len(microwatersheds_gdf) + 1)
    
    # Reproject to a suitable projected CRS (e.g., UTM Zone 17N)
    microwatersheds_gdf_projected = microwatersheds_gdf.to_crs(epsg=26917)
    # Calculate area in square meters
    microwatersheds_gdf_projected['Area_SqMeters'] = microwatersheds_gdf_projected['BasinGeo'].area

    # Convert to acres
    microwatersheds_gdf_projected['Area_Acres'] = microwatersheds_gdf_projected['Area_SqMeters'] / 4046.85642

    # Create a new GeoDataFrame with the original ID and the calculated area
    area_acres_gdf = microwatersheds_gdf_projected[['Microwatershed_ID', 'Area_Acres']]

    # Join the 'Area_Acres' field back to the original GeoDataFrame using the original ID
    microwatersheds_gdf = microwatersheds_gdf.merge(area_acres_gdf, on='Microwatershed_ID')
    
    # # OLD Calculate the area of each polygon in acres (originally in square meters because of the CRS then divide by conversion factor)
    # if units == "Meters":
    #     microwatersheds_gdf['Area_Acres'] = microwatersheds_gdf['geometry'].area/4046.85642
    # elif units == "US Foot":
    #     microwatersheds_gdf['Area_Acres'] = microwatersheds_gdf['geometry'].area/43560

    print(microwatersheds_gdf)

    return microwatersheds_gdf

def overlay_ponds(c_path, microwatersheds_gdf):
    # Pull in ponds dataset and intersect

    import pandas as pd
    import geopandas as gpd
    from shapely import geometry, ops

    # Load ponds data
    # NOTE make sure to read in the shapefile with a CRS aligned with the DEM
    ponds = gpd.read_file(fr'{c_path}jupyter\\data-inputs\\IRL-Ponds-Export\\IRL-Ponds-Export_4269.shp')
    # ponds.to_crs(epsg=crs_dem, inplace=True)

    # NOTE Filter out ponds with an area less than 1 acre
    ponds = ponds[ponds['Area_Acres'] >= 1]

    # Find intersecting ponds
    ponds_intersect = gpd.sjoin(ponds, microwatersheds_gdf, how='inner', predicate='intersects')

    # Count the number of intersecting ponds for each microwatershed
    pond_counts = ponds_intersect.groupby('index_right').size().reset_index(name='Pond_Count')

    # Sum the area of intersecting ponds for each microwatershed
    pond_area_sum = ponds_intersect.groupby('index_right')['Area_Acres_left'].sum().reset_index(name='Total_Pond_Area_Acres')

    # Calculate the average pond area within each microwatershed
    pond_area_avg = ponds_intersect.groupby('index_right')['Area_Acres_left'].mean().reset_index(name='Average_Pond_Area_Acres')

    # Combine pond_counts, pond_area_sum, and pond_area_avg into a single DataFrame
    pond_summary = pond_counts.merge(pond_area_sum, on='index_right').merge(pond_area_avg, on='index_right')

    # Merge the combined summary DataFrame back into the microwatersheds_gdf
    microwatersheds_all_gdf = microwatersheds_gdf.merge(pond_summary, left_index=True, right_on='index_right', how='left')

    # Fill NaN values with 0 (if there are microwatersheds with no intersecting ponds)
    microwatersheds_all_gdf['Pond_Count'] = microwatersheds_all_gdf['Pond_Count'].fillna(0)
    microwatersheds_all_gdf['Total_Pond_Area_Acres'] = microwatersheds_all_gdf['Total_Pond_Area_Acres'].fillna(0)
    microwatersheds_all_gdf['Average_Pond_Area_Acres'] = microwatersheds_all_gdf['Average_Pond_Area_Acres'].fillna(0)

    # Calculate the ratio of total pond area to the area of the microwatershed
    microwatersheds_all_gdf['Pond_Area_Percentage'] = microwatersheds_all_gdf['Total_Pond_Area_Acres'] / microwatersheds_all_gdf['Area_Acres'] *100

    # Calculate volume
    microwatersheds_all_gdf['Pond_Controllable_Volume_Ac-Ft'] = 0.6431378064 + 2.5920596874*microwatersheds_all_gdf['Total_Pond_Area_Acres']

    # Select only the specified columns and order by Pond_Count
    columns_to_display = ['Microwatershed_ID', 'Area_Acres', 'Order', 'Pond_Count', 'Total_Pond_Area_Acres', 'Average_Pond_Area_Acres', 'Pond_Area_Percentage']
    summary_df = microwatersheds_all_gdf[columns_to_display].sort_values(by='Pond_Count', ascending=False)

    # Print the DataFrame
    print(summary_df)

    return summary_df, ponds_intersect, microwatersheds_all_gdf

def plot_pond_overlay(grid, dem, microwatersheds_gdf, ponds_intersect, pdf_pages):
    # Create a figure for plotting all catchments
    fig, ax = plt.subplots(figsize=(8,6))
    norm = colors.Normalize(vmin=0, vmax=15)
    # Set the plot boundaries and aspect ratio
    plt.xlim(grid.bbox[0], grid.bbox[2])
    plt.ylim(grid.bbox[1], grid.bbox[3])
    plt.gca().set_aspect('equal')
    #plot DEM with high transparency
    plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
    microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='black', alpha=0.5)
    # Plot ponds
    ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='blue')

    # Add title and show the combined plot
    plt.title('Microwatersheds - Pond Overlay')
    plt.show()

    # Save plot to pdf
    pdf_pages.savefig(fig)

    return pdf_pages

def pondshed_buffer(ponds_gdf, mws_all_gdf, tolerance=1e-3):
    # Calculates pondshed buffer area and adds 'total pondshed area' to each MWS

    # Calculate volume
    ponds_gdf['Pond_Controllable_Volume_Ac-Ft'] = 0.6431378064 + 2.5920596874 * ponds_gdf['Area_Acres_left']

    # Calculate pondshed area
    ponds_gdf['Pondshed_Area_Ac'] = ponds_gdf['Pond_Controllable_Volume_Ac-Ft'] / (1/3) # assuming the runoff depth is 4 inches, or 1/3 ft

    # Reproject to a coordinate system with meters as units (e.g., EPSG 3857)
    ponds_gdf = ponds_gdf.to_crs(epsg=26917)

    # Calculate the area in acres
    ponds_gdf['Area_Acres_Temp'] = ponds_gdf.geometry.area / 4046.86
    
    buffered_ponds = []
    
    print('Calculating pondshed buffers...')
    for idx, row in ponds_gdf.iterrows():
        pond = row.geometry
        pond_area = row['Area_Acres_left']
        controlled_area = row['Pondshed_Area_Ac']
        buffer_distance = 0.0
        step = 1  # Initial step size for buffer distance
        
        while True:
            buffered_pond = pond.buffer(buffer_distance)
            buffered_area = buffered_pond.area / 4046.86
            
            if abs(buffered_area - controlled_area) < tolerance:
                break
            
            if buffered_area < controlled_area:
                buffer_distance += step
            else:
                buffer_distance -= step
                step /= 2  # Reduce step size for finer adjustment
        
        buffered_ponds.append(buffered_pond)
    
    buffered_ponds_gdf = ponds_gdf.copy()
    buffered_ponds_gdf['geometry'] = buffered_ponds

    print('Calculating pondshed areas...')

    # Dissolve the buffered areas, then sum the area per MWS

    # Step 1: Reproject both datasets to the same CRS (EPSG:26917)
    mws_all_gdf = mws_all_gdf.to_crs(epsg=26917)
    buff = buffered_ponds_gdf

    # Step 2: Dissolve the buffer layer to remove overlaps
    buff_dissolved = buff.dissolve(by='Microwatershed_ID', as_index=False)

    # Step 3: OPTIONAL - Clip the dissolved buffer layer to the microwatershed boundaries
    # buff_clipped = gpd.clip(buff_dissolved, mws_all_gdf)
    buff_clipped = buff_dissolved

    buff_clipped['Clipped_Pondshed_Area_Acres'] = buff_clipped.geometry.area / 4046.86

    # Step 4: Calculate the total area by summing by MWS
    buff_clipped['PondshedAreaSum'] = buff_clipped.groupby('Microwatershed_ID')['Clipped_Pondshed_Area_Acres'].transform('sum')
    pondsheds = buff_clipped.to_crs(epsg=4269)
    mws_all_gdf = mws_all_gdf.merge(pondsheds[['Microwatershed_ID', 'PondshedAreaSum']], on='Microwatershed_ID', how='left')

    # Add columns to MWS layer
    mws_all_gdf['Total_Pondshed_Area_Acres'] = mws_all_gdf['PondshedAreaSum']
    mws_all_gdf['Pondshed_to_Pond_Ratio'] = (mws_all_gdf['Total_Pondshed_Area_Acres'] / mws_all_gdf['Total_Pond_Area_Acres']).round(2)
    mws_all_gdf['Pondshed_to_MWS_Percentage'] = (mws_all_gdf['Total_Pondshed_Area_Acres'] / mws_all_gdf['Area_Acres'] * 100 ).round(2)
    mws_all_gdf = mws_all_gdf.to_crs(epsg=4269)

    print('Pondshed areas complete')

    return mws_all_gdf, pondsheds

def summarize_nutrients(overlay_gdf, microwatersheds_gdf, column_name):
    # calculate areas (just ensure the two datasets are in the same CRS)
    microwatersheds_gdf['RatArea'] = microwatersheds_gdf.area
    overlay_gdf['NutArea'] = overlay_gdf.area

    # Create areas of intersection
    df_is = gpd.overlay(microwatersheds_gdf, overlay_gdf, how='intersection')
    
    # # Store the area size of intersections
    df_is['is_area'] = df_is.area
    
    # # Ratio of intersection / whole area of microwatershed
    df_is['MwsRatio'] = df_is['is_area'] / df_is['RatArea']
    df_is['NutRatio'] = df_is['is_area'] / df_is['NutArea']

    # # Weight by average
    df_is[f'Avg_{column_name}'] = df_is[column_name] * df_is['NutRatio']
    df_is[f'Avg_SUM_Annu_8'] = df_is['SUM_Annu_8'] * df_is['NutRatio']
    
    # Sum over microwatersheds
    # df_weighted_avg = df_is.groupby(['Microwatershed_ID', 'Area_Acres', 'Order', 'Pond_Count', 'Total_Pond_Area_Acres', 'Average_Pond_Area_Acres', 'Pond_Area_Percentage', 'geometry'])[[f'Avg_{column_name}']].sum().reset_index()
    df_weighted_avg = df_is.groupby(['Microwatershed_ID'])[[f'Avg_{column_name}', 'Avg_SUM_Annu_8']].sum()

    microwatersheds_gdf = microwatersheds_gdf.merge(df_weighted_avg, on='Microwatershed_ID', how='left')

    microwatersheds_gdf[f'Control_Vol /_Avg_{column_name}_Ratio'] = microwatersheds_gdf['Total_Pond_Area_Acres'] / microwatersheds_gdf[f'Avg_{column_name}']

    return microwatersheds_gdf

def calculate_impervious_percentage(raster_path, microwatersheds_gdf):
    import rasterio
    import geopandas as gpd
    from rasterio.mask import mask
    import numpy as np
    
    # Open the raster file
    with rasterio.open(raster_path) as src:
        # Initialize a list to store the impervious percentage for each microwatershed
        impervious_percentages = []

        # Iterate over each microwatershed polygon
        for _, microwatershed in microwatersheds_gdf.iterrows():
            # Mask the raster with the current microwatershed polygon
            out_image, out_transform = mask(src, [microwatershed['BasinGeo']], crop=True, nodata=2)
            
            # Convert the masked raster to a numpy array
            out_image = out_image[0]
            
            # Calculate the total number of pixels and the number of impervious pixels (value 1)
            total_pixels = np.count_nonzero(out_image != 2)
            impervious_pixels = np.count_nonzero(out_image == 1)
            
            # Calculate the percentage of impervious area
            impervious_percentage = (impervious_pixels / total_pixels) * 100
            
            # Append the result to the list
            impervious_percentages.append(impervious_percentage)

    # Add the impervious percentages to the geodataframe
    microwatersheds_gdf['Percent_Impervious'] = impervious_percentages

    return microwatersheds_gdf

def urban_area(overlay_gdf, microwatersheds_gdf):
    # Ensure both GeoDataFrames use the same CRS
    
    # # MWS area
    microwatersheds_gdf['MicrowshedArea_Unitless'] = microwatersheds_gdf.area
    # print(microwatersheds_gdf.columns)

    # Create areas of intersection
    df_is2 = gpd.overlay(microwatersheds_gdf, overlay_gdf, how='intersection')
    # print(df_is2.columns)
    
    urban = [
        'Commercial and Services',
        'Institutional',
        'Industrial',
        'Residential Low Density',
        'Residential Medium Density', 
        'Residential High Density', 
        'Transportation',
        'Communications',
        'Utilities'
    ]

    # Filter intersections to only include impervious areas
    df_is2 = df_is2[df_is2['LEVEL2_L_1'].isin(urban)]

    # # Store the area size of intersections
    df_is2['Urban_Area'] = df_is2.area
    # print(df_is2.columns)

    # Sum over microwatersheds
    df_is2 = df_is2.groupby(['Microwatershed_ID', 'MicrowshedArea_Unitless'])[['Urban_Area']].sum().reset_index()

    df_is2['Percent_Urban'] = df_is2['Urban_Area'] / df_is2['MicrowshedArea_Unitless'] * 100

    print(df_is2.head(20))
    microwatersheds_gdf = microwatersheds_gdf.merge(df_is2, on='Microwatershed_ID', how='left')

    return microwatersheds_gdf

def filter_mws_characteristics(microwatersheds_all_gdf, grid, dem, ponds_intersect, pdf_pages, min_total_pond_area, max_num_ponds, pondsheds):
    # Filter MWS characteristics

    # Total Pond Area - likely the most important
    min_total_pond_area = min_total_pond_area
    microwatersheds_filter_gdf = microwatersheds_all_gdf[microwatersheds_all_gdf['Total_Pond_Area_Acres'] >= min_total_pond_area]
    # Pond Count - second of importance (likely won't implement the tech on a high number of ponds)
    max_pond_count = max_num_ponds
    microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Pond_Count'] <= max_pond_count]
    # MWS Area - less important (?) becuase a large area could still have favorable above characteristics
    max_mws_area = 500
    # microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Area_Acres'] <= max_mws_area]

    # Filter out order 3 catchments (the largest order)
    microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Order'] != 3]

    # Print MWS and intersecting ponds
    from matplotlib import colors
    # print(microwatersheds_gdf)
    fig, ax = plt.subplots(figsize=(8,6))
    norm = colors.Normalize(vmin=0, vmax=15)
    # Set the plot boundaries and aspect ratio
    plt.xlim(grid.bbox[0], grid.bbox[2])
    plt.ylim(grid.bbox[1], grid.bbox[3])
    plt.gca().set_aspect('equal')
    cmap = plt.get_cmap('tab20')
    #plot DEM with high transparency
    plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
    microwatersheds_filter_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
    pondsheds.plot(ax=ax, aspect=1, alpha=0.25, color='blue')
    ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='blue')

    # Add labels to the microwatersheds using 'BasinGeo' as the geometry column
    for idx, row in microwatersheds_filter_gdf.iterrows():
        plt.annotate(text=row['Microwatershed_ID'], xy=(row['BasinGeo'].centroid.x, row['BasinGeo'].centroid.y),
                    horizontalalignment='center', verticalalignment='center', fontsize=6, color='black',fontweight='bold', bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.1'))


    plt.title(f'Microwatersheds - Minimum Total Pond Area {min_total_pond_area} Acres. Max Number of Ponds {max_num_ponds}')
    plt.show()

    # Rename columns 
    microwatersheds_filter_gdf.rename(columns={'Avg_SUM_Annu_5': 'Total_Nitrogen_(Lb/Yr)'}, inplace=True)
    microwatersheds_filter_gdf.rename(columns={'Avg_SUM_Annu_8': 'Total_Phosphorous_(Lb/Yr)'}, inplace=True)
    microwatersheds_filter_gdf.rename(columns={'Pond_Area_Percentage': 'Pond Area /_MWS Area_Percentage'}, inplace=True)
    microwatersheds_filter_gdf.rename(columns={'Pondshed_to_MWS_Percentage': 'Pondshed Area /_MWS Area_Percentage'}, inplace=True)
    microwatersheds_filter_gdf.rename(columns={'Microwatershed_ID': 'Microwshed_ID'}, inplace=True)

    # Select only the specified columns and order by Total_Pond_Area_Acres
    columns_to_display = ['Microwshed_ID', 
                        'Pond_Count', 
                        'Area_Acres', 
                        #   'Average_Pond_Area_Acres', 
                        'Total_Pond_Area_Acres', 
                        'Total_Pondshed_Area_Acres',
                        'Pondshed_to_Pond_Ratio',
                        'Pond Area /_MWS Area_Percentage',
                        'Pondshed Area /_MWS Area_Percentage',
                        'Pond_Controllable_Volume_Ac-Ft', 
                        'Total_Nitrogen_(Lb/Yr)', 
                        'Total_Phosphorous_(Lb/Yr)', 
                        'Percent_Impervious', 
                        'Percent_Urban']
    filter_df = microwatersheds_filter_gdf[columns_to_display].sort_values(by='Total_Pond_Area_Acres', ascending=False)
    # filter_df = filter_df[filter_df['Microwatershed_ID'] == 121]

    format_columns = {
        'Microwshed_ID': '{:.0f}',
        'Pond_Count': '{:.0f}',
        'Area_Acres': '{:.2f}',
        # 'Average_Pond_Area_Acres': '{:.2f}',
        'Total_Pond_Area_Acres': '{:.2f}',
        'Total_Pondshed_Area_Acres': '{:.2f}',
        'Pondshed_to_Pond_Ratio': '{:.2f}',
        'Pond Area /_MWS Area_Percentage': '{:.2f}',
        'Pondshed Area /_MWS Area_Percentage': '{:.2f}',
        'Pond_Controllable_Volume_Ac-Ft': '{:.2f}',
        'Total_Nitrogen_(Lb/Yr)': '{:.2f}',
        'Total_Phosphorous_(Lb/Yr)': '{:.2f}',
        'Percent_Impervious': '{:.2f}',
        'Percent_Urban': '{:.2f}'
    }

    for col, fmt in format_columns.items():
        filter_df[col] = filter_df[col].map(fmt.format)

    # Print the DataFrame
    print(filter_df.head(20))

    # Save plot to pdf
    pdf_pages.savefig(fig)


    # Add DataFrame to the PDF
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('tight')
    ax.axis('off')
    table_data = filter_df.head(20).values  # Assuming filter_df is your DataFrame
    columns = [col.replace('_', '\n') for col in filter_df.columns.tolist()]
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center', loc='center')

    # Function for coloring columns
    def create_color_map(df, column, cmap, alpha=0.5):
        norm = plt.Normalize(df[column].astype(float).min(), df[column].astype(float).max())
        col_colors = cmap(norm(df[column].astype(float)))
        col_colors[:, -1] = alpha  # Set alpha transparency
        return col_colors

    columns_to_color = {
        'Pond_Count': plt.cm.plasma,
        'Area_Acres': plt.cm.plasma,
        'Total_Pond_Area_Acres': plt.cm.plasma,
        'Total_Pondshed_Area_Acres': plt.cm.plasma,
        'Pondshed_to_Pond_Ratio': plt.cm.plasma,
        # 'Average_Pond_Area_Acres': plt.cm.plasma,
        'Pond Area /_MWS Area_Percentage': plt.cm.plasma,
        'Pondshed Area /_MWS Area_Percentage': plt.cm.plasma,
        'Pond_Controllable_Volume_Ac-Ft': plt.cm.plasma,
        'Total_Nitrogen_(Lb/Yr)': plt.cm.plasma,
        'Total_Phosphorous_(Lb/Yr)': plt.cm.plasma,
        'Percent_Impervious': plt.cm.plasma,
        'Percent_Urban': plt.cm.plasma,
    }

    col_colors = {col: create_color_map(filter_df, col, cmap) for col, cmap in columns_to_color.items()}


    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)  # Adjust the size of the table

    # Apply col_colors to the columns
    for col, color_map in col_colors.items():
        col_index = columns.index(col.replace('_', '\n'))
        for i in range(len(table_data)):
            cell = table[(i+1, col_index)]  # (row, column)
            cell.set_facecolor(color_map[i])
            cell.set_text_props(color='black')

    # Adjust header cell height if needed
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_height(0.12)  # Adjust height for better visibility

    # Also export to a csv
    # Get the current datetime and format it
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H%M")  # Format: 2024-11-22_0230
    filter_df.to_csv(rf'outputs\\tables\\SummaryTable_{file}_{datetime_str}.csv', index=False)

    pdf_pages.savefig(fig)  # Save the DataFrame table to the PDF

    return pdf_pages, filter_df, microwatersheds_filter_gdf

def export_microwatersheds(polygon):
    # Export Microwatershed output to a shapefile

    # Get the current datetime and format it
    import os
    from datetime import datetime
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H%M")  # Format: 2024-11-22_0230

    # Define the output file path
    os.makedirs(rf'outputs\\shp\\{datetime_str}', exist_ok=True)
    output_file_path = f"outputs\shp\{datetime_str}\Microwatersheds_{file}_{datetime_str}.shp"

    # Export the GeoDataFrame to a shapefile
    polygon.to_file(output_file_path, driver='ESRI Shapefile')

def close_pdf(pdf_pages):
    # Close the PDF file
    pdf_pages.close()

def interactive_map(ponds_intersect, microwatersheds_all_gdf, branches_):
    # Folium plotting
    import folium
    from streamlit_folium import st_folium

    # Select only the specified columns and order by Pond_Count
    columns_to_display = ['Pond_ID', 'Area_Acres_right', 'geometry']
    ponds_simple = ponds_intersect[columns_to_display].sort_values(by='Area_Acres_right', ascending=False)
    
    # separate by stream order
    ones = microwatersheds_all_gdf[microwatersheds_all_gdf['Order']==1]
    twos = microwatersheds_all_gdf[microwatersheds_all_gdf['Order']==2]
    threes = microwatersheds_all_gdf[microwatersheds_all_gdf['Order']==3] # skip fours

    # Map 'em!
    cols = ['Microwatershed_ID', 'Area_Acres', 'Order', 'Pond_Count', 'Total_Pond_Area_Acres', 'Average_Pond_Area_Acres', 'Pond_Area_Percentage', 'BasinGeo']
    cols_branches = ['Index','Length','Relief','Order','Slope','geometry','LocalPP_X','LocalPP_Y','Final_Chain_Val']
    # pond_cols = ['Pond_ID', 'area']

    m = folium.Map(location=[28.205, -80.70], zoom_start=10)  # Set your initial location
    branches_[cols_branches].explore(m=m, color='black')
    ones[cols].explore(m=m,color='green',tiles='Stamen Terrain')
    twos[cols].explore(m=m,color='purple',tiles='Stamen Terrain')
    threes[cols].explore(m=m,color='yellow',tiles='Stamen Terrain')
    ponds_simple.explore(m=m, color='blue', tiles='Stamen Terrain')
    folium.LayerControl().add_to(m) 
    # m

    return m

# Function to create the interactive map and table
def dash_map(filter_df, microwatersheds_filter_gdf):
    import folium
    
    # Create a base map
    m = folium.Map(location=[28.2, -80.7], zoom_start=4)
    
    # Add polygons to the map
    for _, row in microwatersheds_filter_gdf.iterrows():
        folium.GeoJson(row['BasinGeo']).add_to(m)
    
    # Display the map in Streamlit
    st.components.v1.html(m._repr_html_(), height=600)

    # Display the table in Streamlit
    selected_row = st.selectbox("Select a Microwatershed ID", filter_df['Microwatershed_ID'])
    
    # Highlight selected polygon
    if selected_row:
        selected_id = selected_row
        selected_polygon = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Microwatershed_ID'] == selected_id]
        folium.GeoJson(
            selected_polygon['BasinGeo'].values[0],
            style_function=lambda x: {'fillColor': 'yellow'}
        ).add_to(m)
        
        # Update the map with the highlighted polygon
        st.components.v1.html(m._repr_html_(), height=600)



def main(file, epsg, units, aggregation, flow_file_path, burn_width, burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds):

    # If testing locally, use path
    c_path = 'C:\\Users\\alden.summerville\\Documents\\dev-local\\IRL-MIA-development\\'
    # If deployed, use relative paths
    # c_path = ''

    ## Run all the consequetive functions
    dem_path, grid, dem = read_reproject_dem(c_path, file)

    pdf_path, pdf_pages = initialize_pdf(c_path, file, epsg, units, aggregation, flow_file_path, burn_width, burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds)
    
    dem_agg_path, grid, dem = aggregate_dem(c_path, dem_path, aggregation)

    crs_dem = confirm_crs(dem_agg_path)

    grid, dem, dem_agg_burn_path = burn_flowlines(c_path, crs_dem, dem_agg_path, burn_width, burn_value, dem, file, aggregation)

    pdf_pages = plot_burned_dem(dem, grid, pdf_pages, crs_dem, aggregation, burn_value, burn_width, units)

    branches_, microwatersheds_gdf = delineate_microwatersheds(dem_agg_burn_path, river_network_min_flow_acc)

    pdf_pages = plot_microwatersheds(dem, grid, microwatersheds_gdf, branches_, pdf_pages)

    microwatersheds_gdf = calc_mws_areas(microwatersheds_gdf)

    summary_df, ponds_intersect, microwatersheds_all_gdf = overlay_ponds(c_path, microwatersheds_gdf)

    pdf_pages = plot_pond_overlay(grid, dem, microwatersheds_gdf, ponds_intersect, pdf_pages)

    microwatersheds_all_gdf, pondsheds = pondshed_buffer(ponds_intersect, microwatersheds_all_gdf, tolerance=1e-3)

    # Load nutrients layer
    nutrients = gpd.read_file(r'data-inputs\\Nutrients\\Nutrients_Brevard_4326.shp')
    microwatersheds_all_gdf = summarize_nutrients(nutrients, microwatersheds_all_gdf, 'SUM_Annu_5')

    impervious_raster = r'data-inputs\\ImperviousArea\\FlaImperviousArea_4326.tif'
    microwatersheds_all_gdf = calculate_impervious_percentage(impervious_raster, microwatersheds_all_gdf)

    land_cover = gpd.read_file(r'data-inputs\\LandCover\\Land_Cover_FLA_4326.shp')
    microwatersheds_all_gdf = urban_area(land_cover, microwatersheds_all_gdf)

    export_microwatersheds(microwatersheds_all_gdf)

    pdf_pages, filter_df, microwatersheds_filter_gdf = filter_mws_characteristics(microwatersheds_all_gdf, grid, dem, ponds_intersect, pdf_pages, min_total_pond_area, max_num_ponds, pondsheds)

    close_pdf(pdf_pages)

    # dash_map(filter_df, microwatersheds_filter_gdf)

    # m = interactive_map(ponds_intersect, microwatersheds_nutrients_gdf, branches_)

    return pdf_path


# Streamlit app layout
st.title("Microwatershed Impact Assessment - Python Tool")

# DEM file inputs
dems = ["Terrain_Grant_Valkaria_ClipNoData_NAD83",
        "UpperCanalMosa_1m_NAD83"]

# User inputs
file = st.selectbox("Enter the DEM file name (without extension):", ["Pineda_Scalgo_8m_NAD83", "Terrain_Grant_Valkaria_ClipNoData_NAD83", "Terrain_Grant_Valkaria_ClipNoData_FLA", "PinedaScalgo_1m_NAD83", "Pineda_Scalgo_AggMedian16_NAD83", "Melbourne_NAD83", "Tomako_FLA", "UpperCanalMosa_1m_NAD83"])
epsg = st.selectbox("Enter the EPSG code (2881 for StatePlane Florida East. 26917 for NAD83 Zone 17N):", ["26917", "2881"])
units = st.selectbox("Enter the units of the DEM", ["Meters", "US Foot"])
aggregation = st.number_input("DEM Aggregation Factor:", min_value=0, value=1)
flow_file_path = st.selectbox("Enter the clipped flowlines path:", ["IRL-Flowlines-Export_NAD83.shp", "IRL-Pineda-Flowlines-Export_NAD83.shp", "IRL-UpperCanalFlowlines-Export_NAD83.shp"])
burn_width = st.number_input("Burn Width:", min_value=1, value=1)
burn_value = st.number_input("Burn Value:", value=0)
river_network_min_flow_acc = st.number_input("Minimum Flow Accumulation - Channels:", min_value=0, value=3000)
min_total_pond_area = st.number_input("Minimum Total Pond Area per Microwatershed:", min_value=0, value=15)
max_num_ponds = st.number_input("Max Number of Ponds per Microwatershed:", min_value=0, value=50)


# Button to run the main function
if st.button("Run"):
    with st.spinner("Running..."):
        pdf_path = main(file, epsg, units, aggregation, flow_file_path, burn_width, burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds)
    st.success("Complete. A report with the key figures is saved to the outputs folder.")
    
    # Open the PDF file
    if os.path.exists(pdf_path):
        if os.name == 'nt':  # For Windows
            os.startfile(pdf_path)
        else:  # For macOS and Linux
            subprocess.call(['open', pdf_path])  # macOS
            # subprocess.call(['xdg-open', pdf_path])  # Linux
    else:
        st.error("The PDF file was not found.")

    # Store the map in session state
    # st.session_state.map = m


# # Render the map if it exists in session state
# if 'map' in st.session_state:
#     st_folium(st.session_state.map, width=700, height=500)