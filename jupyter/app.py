
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
import pandas as pd
from shapely import geometry
from rasterstats import zonal_stats
# from streamlit_folium import st_folium

# sample inputs
# file = 'Terrain_Grant_Valkaria_ClipNoData_NAD83'
# aggregation = 16
# flow_file_path = 'IRL-Flowlines-Export_NAD83.shp'
# burn_width=4
# flowlines_burn_value = -3  # Adjust this value as needed
# river_network_min_flow_acc = 1000
# min_total_pond_area = 20

## Define Key Functions

def read_reproject_dem(c_path, file):
    # Import initial 1m resolution DEM (to be downsampled)
    print("Reading in raster...")
    file = file
    dem_path = fr'{c_path}jupyter\\data-inputs\\DEM\\{file}.tif'

    grid = Grid.from_raster(dem_path)
    grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
    dem = grid.read_raster(dem_path)

    print("Converting to mercator...")
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
                "data-inputs\\temp\\temp_reprojected_raster.tif",
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


    dem_path = 'data-inputs\\temp\\temp_reprojected_raster.tif'
    grid = Grid.from_raster(dem_path)
    grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
    dem = grid.read_raster(dem_path)

    return dem_path, grid, dem

def initialize_pdf(c_path, file, epsg, units, aggregation, flow_file_path, burn_width, flowlines_burn_value, pond_burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds):
    # Create a PDF file to save the plots
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    # Get the current datetime and format it
    now = datetime.now()
    datetime_str = now.strftime("%Y-%m-%d_%H%M")  # Format: 2024-11-22_0230

    pdf_path = f'{c_path}jupyter\\outputs\\Output_{file}_{flowlines_burn_value}Burn_{min_total_pond_area}MinTotPondArea_{max_num_ponds}MaxNumPonds_{datetime_str}.pdf'
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
    Burn Value: {flowlines_burn_value}
    Pond Burn Value: {pond_burn_value}
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

    print("Downsampling initialized...")

    import rasterio
    from rasterio.enums import Resampling

    # PARAMETER: set the upscale factor for downsampling
    aggregation = aggregation
    upscale_factor = 1/aggregation

    with rasterio.open(dem_path) as dataset:
        # Print resolution
        original_resolution = dataset.res
        print(f"Original resolution: {original_resolution}")

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

        # Print the new resolution
        new_resolution = dataset.res
        print(f"New resolution: {new_resolution}")

        # Write the downsampled data to a new file
        with rasterio.open(
            fr'{c_path}jupyter\\data-inputs\\temp\\{file}_{aggregation}_Agg.tif',
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

    dem_path = fr'{c_path}jupyter\\data-inputs\\temp\\{file}_{aggregation}_Agg.tif'
    grid = Grid.from_raster(dem_path)
    grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
    dem = grid.read_raster(dem_path)

    print("Downsampling complete")

    return dem_path, grid, dem

def confirm_crs(dem_path):
    # LOG: Confirm CRS
    import rasterio
    # Open the raster file
    with rasterio.open(dem_path) as src:
        crs_dem = src.crs
        print(crs_dem)

    # set crs variable to whatever the crs is
    # for now, manually, becuase the rasterio lib isn't returning the CRS correctly for the FLA projection
    crs_dem = epsg
    # crs_dem = str(crs_dem).replace("EPSG:", "")

    return crs_dem

def condition_dem(grid, dem, c_path):
    from rasterio.transform import from_origin

    print("Conditioning DEM pre-burn...")
    pit_filled_dem = grid.fill_pits(dem)
    
    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)

    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)


    # Extract necessary information
    crs = grid.crs  # CRS from the pysheds grid object
    res = grid.affine[0]  # Resolution (assuming square cells)
    x_min, y_max = grid.bbox[0], grid.bbox[3]  # Bounding box (xmin, ymax)
    transform = from_origin(x_min, y_max, res, res)  # Create affine transform
    output_tif = fr'{c_path}jupyter\\data-inputs\\temp\\TempPostCondition_DEM.tif'

    # Save the array as a GeoTIFF
    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=inflated_dem.shape[0],
        width=inflated_dem.shape[1],
        count=1,
        dtype=inflated_dem.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(inflated_dem, 1)

    dem_path_cd = output_tif
    grid_cd = Grid.from_raster(dem_path_cd)
    dem_cd = grid.read_raster(dem_path_cd)

    return grid_cd, dem_cd, dem_path_cd

def burn_features(c_path, crs_dem, dem_path, burn_width, flowlines_burn_value, osm_burn_value, pond_burn_value, dem, file, aggregation):
    # Burn the NHD Flowlines into the DEM

    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    import numpy as np
    from shapely.geometry import box

    if flowlines_burn_value != 0:
        # Load the flowlines dataset
        # flow_file_path = 'IRL-Flowlines-Export_NAD83.shp'
        flowlines = gpd.read_file(fr'{c_path}jupyter\\data-inputs\\NHD-Flowlines\\{flow_file_path}')
        flowlines = flowlines.to_crs(epsg=4269)

        with rasterio.open(dem_path) as src:
            bounds = src.bounds
            # Create a bounding box geometry
            bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
            bbox_gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src.crs)

        print("clipping flowlines...")
        # Clip flowlines to DEM extent
        flowlines_clip = gpd.clip(flowlines, bbox_gdf)

        # Convert back to CRS with meters to do the buffer
        flowlines_clip = flowlines_clip.to_crs(epsg=26917)
        # PARAMETER: burn width
        flowlines_clip['geometry'] = flowlines_clip.geometry.buffer(burn_width)  # Buffer by half the desired width
        # Convert back to CRS with lat lon
        flowlines_clip = flowlines_clip.to_crs(epsg=4269)

        # Load the shorelines dataset
        shoreline = gpd.read_file(r'data-inputs\\Shoreline\\FLA-Shoreline_4269.shp')

        print("clipping shorelines...")
        shoreline_clip = gpd.clip(shoreline, bbox_gdf)

        shoreline_clip = shoreline_clip[shoreline_clip['ATTRIBUTE'] != 'Land']
        shoreline_clip = shoreline_clip[shoreline_clip['Shape_Area'] > 0.000002]

        print("performing shoreline clip...")
        # Clip the channels polyline by the shoreline dataset
        flowlines_clip = gpd.clip(flowlines_clip, shoreline_clip)

        # Ensure the flowlines are in the same CRS as the DEM
        # file = 'Terrain_Grant_Valkaria_ClipNoData_AggMedian16_NAD83'
        # NOTE: replace dem_path with dem_path if the dem isn't being aggregated
        with rasterio.open(dem_path) as src:
            # flowlines = flowlines.to_crs(src.crs)
            transform = src.transform
            out_shape = src.shape

        print("Length of flowlines: ", len(flowlines_clip))

        # Rasterize the flowlines
        flowline_raster = rasterize(
            [(geom, 1) for geom in flowlines_clip.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=True
        )

        # Read the DEM
        with rasterio.open(dem_path) as src:
            dem = src.read(1)  # Read the first band

        # PARAMETER: Burn the flowlines into the DEM
        flowlines_burn_value = flowlines_burn_value  # Adjust this value as needed
        dem_burned = np.where(flowline_raster == 1, dem + flowlines_burn_value, dem)

        # Save the modified DEM
        with rasterio.open(
            fr'{c_path}jupyter\\data-inputs\\temp\\{file}_{aggregation}_Agg_Burned.tif', 
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

        dem_path = fr'{c_path}jupyter\\data-inputs\\temp\\{file}_{aggregation}_Agg_Burned.tif'
        grid = Grid.from_raster(dem_path)
        grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
        dem = grid.read_raster(dem_path)

        print("Flowlines have been burned into the DEM and saved as a new file.")
    else:
        dem_path = dem_path
        # grid = Grid.from_raster(dem_path)
        # grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
        # dem = grid.read_raster(dem_path)
        print("No flowlines were burned into the DEM.")

    if osm_burn_value != 0:
        # load OSM network
        with rasterio.open(dem_path) as src:
            bounds = src.bounds
            crs = src.crs
        # Create a bounding box geometry
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        bbox_gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src.crs)
        # Convert the bounding box to a format suitable for OSMnx
        minx, miny, maxx, maxy = bbox.bounds

        # # Download the street network within the bounding box
        # G = ox.graph.graph_from_bbox(bbox.bounds, network_type='all')
        # # Convert the graph to a GeoDataFrame
        # nodes, edges = ox.graph_to_gdfs(G)

        # TEMP read in edges
        edges = gpd.read_file(r'data-inputs\\StreetNetwork\\PinedaOsmExport.shp')
        # Ensure the CRS matches the source CRS
        edges = edges.to_crs(crs)

        # Convert back to CRS with meters to do the buffer
        edges = edges.to_crs(epsg=26917)
        # PARAMETER: burn width
        edges['geometry'] = edges.geometry.buffer(2)  # Buffer by half the desired width
        # Convert back to CRS with lat lon
        edges = edges.to_crs(epsg=4269)

        # Ensure the flowlines are in the same CRS as the DEM
        # file = 'Terrain_Grant_Valkaria_ClipNoData_AggMedian16_NAD83'
        with rasterio.open(dem_path) as src:
            # flowlines = flowlines.to_crs(src.crs)
            transform = src.transform
            out_shape = src.shape


        # Rasterize the flowlines
        edges_raster = rasterize(
            [(geom, 1) for geom in edges.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=True
        )

        # Read the DEM
        with rasterio.open(dem_path) as src:
            dem = src.read(1)  # Read the first band

        # Ensure edges array has the same shape as dem
        # edges_resized = np.resize(edges, dem.shape)

        dem_burned = np.where(edges_raster == 1, dem - osm_burn_value, dem)

        # Save the modified DEM
        with rasterio.open(
            fr'data-inputs\\temp\\{file}_{aggregation}_Agg_Burned_OSM.tif', 
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

        dem_path = fr'data-inputs\\temp\\{file}_{aggregation}_Agg_Burned_OSM.tif'
        grid = Grid.from_raster(dem_path)
        grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
        dem = grid.read_raster(dem_path)

        print("OSM Street network has been burned into the DEM and saved as a new file.")
    else:
        dem_path = dem_path
        # grid = Grid.from_raster(dem_path)
        # grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
        # dem = grid.read_raster(dem_path)
        print("No OSM burn value provided. Skipping the burn process.")

    if pond_burn_value !=0:
        # Load the pond_burn dataset
        pond_burn = gpd.read_file(r'data-inputs\\IRL-Ponds-Export\\IRL-Ponds-Export_4269.shp')
        pond_burn = pond_burn.to_crs(epsg=4269)

        with rasterio.open(dem_path) as src:
            bounds = src.bounds
        # Create a bounding box geometry
        bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        bbox_gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src.crs)

        # Clip flowlines to DEM extent
        pond_burn_clip = gpd.clip(pond_burn, bbox_gdf)

        with rasterio.open(dem_path) as src:
            transform = src.transform
            out_shape = src.shape


        # Rasterize the flowlines
        pond_burn_raster = rasterize(
            [(geom, 1) for geom in pond_burn_clip.geometry],
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype='uint8',
            all_touched=True
        )

        # Read the DEM
        with rasterio.open(dem_path) as src:
            dem = src.read(1)  # Read the first band


        dem_burned = np.where(pond_burn_raster == 1, dem - flowlines_burn_value, dem)

        # Save the modified DEM
        with rasterio.open(
            fr'data-inputs\\temp\\{file}_{aggregation}_Agg_Pond_Burned.tif', 
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

        dem_path = fr'data-inputs\\temp\\{file}_{aggregation}_Agg_Pond_Burned.tif'
        grid = Grid.from_raster(dem_path)
        grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
        dem = grid.read_raster(dem_path)

        print("Ponds have been burned into the DEM and saved as a new file.")
    else:
        dem_path = dem_path
        # grid = Grid.from_raster(dem_path)
        # grid_clip = Grid.from_raster(dem_path) # to be clipped by the delineation extent to preserve the original grid
        # dem = grid.read_raster(dem_path)
        print("No burn value provided. Skipping the burn process.")

    return grid, dem, dem_path

def plot_burned_dem(dem, grid, pdf_pages, crs_dem, aggregation, flowlines_burn_value, burn_width, units):
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
    plt.title(f'Digital Elevation Map - {aggregation}x{aggregation} - {flowlines_burn_value} depth, {burn_width*2} wide Burn', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    # Save plot to pdf
    pdf_pages.savefig(fig)

    return pdf_pages

def delineate_microwatersheds(grid, dem, dem_cd, river_network_min_flow_acc, condition_burn):
    # NEW METHOD to delineate the catchments and stream orders 
    import make_catchments

    basins_, branches_ = make_catchments.generate_catchments(grid, dem, condition_burn, dem_cd, acc_thresh=river_network_min_flow_acc,so_filter_max=4, shoreline_clip=True)
    print("Pysheds complete")

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
    print("Calculate MWS areas")
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

    # print(microwatersheds_gdf)

    return microwatersheds_gdf

def overlay_ponds(c_path, microwatersheds_gdf):
    # Pull in ponds dataset and intersect
    print("Overlay ponds")
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
    # print(summary_df)

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
    print("Pondshed buffer")

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

    # Step 3: Initialize an empty GeoDataFrame to store the clipped results
    clipped_buffers_list = []

    for idx, pond in buff_dissolved.iterrows():
        mws_id = pond['Microwatershed_ID']
        mws_geom = mws_all_gdf[mws_all_gdf['Microwatershed_ID'] == mws_id].geometry.iloc[0]
        clipped_buff = gpd.clip(gpd.GeoDataFrame([pond], columns=buff_dissolved.columns), mws_geom)
        clipped_buffers_list.append(clipped_buff)

    clipped_buffers = pd.concat(clipped_buffers_list, ignore_index=True)
    clipped_buffers = gpd.GeoDataFrame(clipped_buffers, geometry='geometry', crs=buff_dissolved.crs)

    buff_clipped = clipped_buffers

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
    print("Summarize nutrients")
    
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

    # (Nutrient Load) * (Pondshed Area Percentage) to estimate how much of the nutrient load might be reduced
    microwatersheds_gdf[f'Controllable_Nitrogen_(Lb/Yr)'] = microwatersheds_gdf['Avg_SUM_Annu_5'] * microwatersheds_gdf[f'Pondshed_to_MWS_Percentage'] / 100

    return microwatersheds_gdf

def calculate_impervious_percentage(raster_path, microwatersheds_gdf):
    import geopandas as gpd
    from rasterstats import zonal_stats

    # Compute zonal statistics (sum of impervious pixels and total pixel count)
    stats = zonal_stats(
        microwatersheds_gdf,
        raster_path,
        stats=["sum", "count"],  # Sum of impervious pixels, and total pixel count
        all_touched=True  # Ensures all intersecting pixels are included
    )

    # Extract values
    impervious_pixel_sums = [stat["sum"] for stat in stats]  # Sum of pixels classified as 1
    total_pixels = [stat["count"] for stat in stats]  # Total pixels in each polygon

    # Calculate percentage of impervious area
    microwatersheds_gdf["Percent_Impervious"] = [
        (impervious / total * 100) if total > 0 else 0
        for impervious, total in zip(impervious_pixel_sums, total_pixels)
    ]

    # Calculate impervious area in acres
    microwatersheds_gdf["ImperviousAreaAcres"] = (
        microwatersheds_gdf["Area_Acres"] * microwatersheds_gdf["Percent_Impervious"] / 100
    )

    return microwatersheds_gdf

def calculate_pondshed_impervious(raster_path, microwatersheds_gdf, pondsheds):
    # Compute zonal statistics (sum of impervious pixels and total pixel count)
    stats = zonal_stats(
        pondsheds,
        raster_path,
        stats=["sum", "count"],  # Sum of impervious pixels, and total pixel count
        all_touched=True  # Ensures all intersecting pixels are included
    )

    # Extract values
    impervious_pixel_sums = [stat["sum"] for stat in stats]  # Sum of pixels classified as 1
    total_pixels = [stat["count"] for stat in stats]  # Total pixels in each polygon

    # Calculate percentage of impervious area
    pondsheds["Percent_Impervious"] = [
        (impervious / total * 100) if total > 0 else 0
        for impervious, total in zip(impervious_pixel_sums, total_pixels)
    ]

    # Calculate impervious area in acres
    pondsheds["PondshedImperviousAreaAcres"] = (
        pondsheds["Pondshed_Area_Ac"] * pondsheds["Percent_Impervious"] / 100
    )

    # Sum runoff, nitrogen, and phosphorus for each microwatershed
    pondshed_grouped = pondsheds.groupby('Microwatershed_ID', as_index=False).agg({
        'PondshedImperviousAreaAcres': 'sum'
    })

    microwatersheds_gdf = microwatersheds_gdf.merge(pondshed_grouped, on='Microwatershed_ID', how='left')

    return microwatersheds_gdf

def get_annual_rainfall(gdf, raster_path):
    # Compute zonal statistics (mean value of raster within each polygon)
    stats = zonal_stats(
        gdf,
        raster_path,
        stats="mean",  # You can use other statistics like "min", "max", etc.
        all_touched=True # to ensure all pixels intersecting the polygon are included
    )

    # Extract the mean rainfall values and add them as a new column
    gdf['AnnualRainfallMm'] = [stat['mean'] for stat in stats] # dataset has units of mm
    gdf['AnnualRainfallInches'] = gdf['AnnualRainfallMm'] / 25.4

    return gdf

def annual_control_volume(gdf):
    """
    Processes a GeoDataFrame to calculate and add the following columns:
    - 'AnnualRunoffMGYr': Annual runoff in MG/yr.
    - 'Annual_Volume_Treated_MG/Yr': Incremental annual wet weather capture in MG/yr.
    - 'Annual_Volume_Treated_MG/Yr_PerPond': Volume treated per pond.
    
    Parameters:
        gdf (GeoDataFrame): Input GeoDataFrame with necessary columns.
    
    Returns:
        GeoDataFrame: Updated GeoDataFrame with new calculated columns.
    """
    
    def calculate_annual_runoff(impervious_area_ac, annual_rainfall_in, runoff_coefficient=1.0):
        """Calculate Annual Runoff (MG/yr)."""
        return impervious_area_ac * annual_rainfall_in * runoff_coefficient * 0.027154
    
    def calculate_incremental_wet_weather_capture(impervious_area_ac, annual_rainfall_in, annual_runoff_mgyr, passive_volume_acft, cmac_volume_acft):
        import math

        # Calculate Incremental Annual Wet Weather Capture (MG/yr).
        if impervious_area_ac == 0:
            # If area is zero, set it to a negligible number to ensure the division is possible
            impervious_area_ac = 0.001
            passive_volume_inIA = (passive_volume_acft / impervious_area_ac) * 12
            cmac_volume_inIA = (cmac_volume_acft / impervious_area_ac) * 12
        else:
            passive_volume_inIA = (passive_volume_acft / impervious_area_ac) * 12
            cmac_volume_inIA = (cmac_volume_acft / impervious_area_ac) * 12

        
        A, B = 25.07, 20.864
        intercept, precip_coef, ln_precip, ln_Vol = 124.24, 0.3415, -22.161, 27.417
        
        passive_efficiency = max(min((A + B * np.log(passive_volume_inIA)) / 100, 0.99), 0.05)
        cmac_efficiency = max(min((intercept + precip_coef * annual_rainfall_in + ln_precip * math.log(annual_rainfall_in) + ln_Vol * math.log(cmac_volume_inIA)) / 100, 0.99), 0.05)
        
        return annual_runoff_mgyr * (cmac_efficiency - passive_efficiency)
    
    # Calculate and add columns
    gdf['AnnualRunoffMGYr'] = gdf.apply(lambda row: calculate_annual_runoff(row['ImperviousAreaAcres'], row['AnnualRainfallInches']), axis=1)
    gdf['Annual_Volume_Treated_MG/Yr'] = gdf.apply(
        lambda row: calculate_incremental_wet_weather_capture(
            row['ImperviousAreaAcres'],
            row['AnnualRainfallInches'],
            row['AnnualRunoffMGYr'],
            row['Pond_Controllable_Volume_Ac-Ft'],
            row['Pond_Controllable_Volume_Ac-Ft']
        ),
        axis=1
    )
    gdf['Annual_Volume_Treated_MG/Yr_PerPond'] = np.where(
        gdf['Pond_Count'] == 0,
        0,
        gdf['Annual_Volume_Treated_MG/Yr'] / gdf['Pond_Count']
    )
    
    return gdf

def urban_area(overlay_gdf, microwatersheds_gdf):
    # Ensure both GeoDataFrames use the same CRS
    print("Calculate urban area")
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

def calculate_nutrients(overlay_gdf, microwatersheds_gdf):

    # Create areas of intersection
    land_use_intersect = gpd.overlay(microwatersheds_gdf, overlay_gdf, how='intersection')
 
    # Convert back to CRS with meters to calculate areas
    land_use_intersect = land_use_intersect.to_crs(epsg=26917)
    # PARAMETER: burn width
    land_use_intersect['LULC_Area_Acres'] = land_use_intersect.area / 4046.85642
    # Convert back to CRS with lat lon
    land_use_intersect = land_use_intersect.to_crs(epsg=4269)
    
    # Read in lookup table
    csv_path = r'data-inputs\LandCover\FlLandUseForNutrients.csv'
    # Notable columns in lookup table: 'FlLandUse' (contains the classifications, but some instances have multiple classes separated by commas), 'RunnoffCoefficient', 'TotalNitrogenEMC_(mg/L)', 'TotalPhosEMC_(mg/L)'
    lookup_df = pd.read_csv(csv_path)
    # Explode lookup table so each row corresponds to a single land-use class
    lookup_df['FlLandUse'] = lookup_df['FlLandUse'].str.split(',')  # Convert to list
    lookup_df = lookup_df.explode('FlLandUse')  # Create separate rows for each class
    lookup_df['FlLandUse'] = lookup_df['FlLandUse'].str.strip()  # Remove whitespace

    # Merge with land_use_intersect
    land_use_merge = land_use_intersect.merge(
        lookup_df, left_on='LEVEL2_L_1', right_on='FlLandUse', how='left'
    )


    # Calculate Runoff for each classification
    # Calculate Runoff_m3 = LULC_Area_Acres * RunoffCoefficient (from lookup table column RunoffCoefficient, for a given class) * 0.186 (1 in 10 rainfall volume for 24 hr event)
    land_use_merge['Runoff_m3'] = land_use_merge['LULC_Area_Acres'] * land_use_merge['RunoffCoefficient'] * 0.186

    # Calculate N and P. 0.001 is a conversion factor from mg to kg and 2.2 is the factor for kg to lb
    land_use_merge['Nitrogen_lb'] = land_use_merge['Runoff_m3'] * land_use_merge['TotalNitrogenEMC_(mg/L)'] * 0.001 * 2.20462
    land_use_merge['Phosphorous_lb'] = land_use_merge['Runoff_m3'] * land_use_merge['TotalPhosEMC_(mg/L)'] * 0.001 * 2.20462

    # Sum runoff, nitrogen, and phosphorus for each microwatershed
    nutrient_summary = land_use_merge.groupby('Microwatershed_ID', as_index=False).agg({
        'LULC_Area_Acres': 'sum',
        'Runoff_m3': 'sum',
        'Nitrogen_lb': 'sum',
        'Phosphorous_lb': 'sum'
    })

    # Merge the Runoff_m3 column back into the microwatersheds_gdf
    microwatersheds_gdf = microwatersheds_gdf.merge(nutrient_summary, on='Microwatershed_ID', how='left')

    return microwatersheds_gdf

def filter_mws_characteristics(microwatersheds_all_gdf, grid, dem, ponds_intersect, pdf_pages, min_total_pond_area, max_num_ponds, pondsheds):
    # Filter MWS characteristics
    print("Filter MWS attributes")

    # Total Pond Area - likely the most important
    min_total_pond_area = min_total_pond_area
    microwatersheds_filter_gdf = microwatersheds_all_gdf[microwatersheds_all_gdf['Total_Pond_Area_Acres'] >= min_total_pond_area]
    # Pond Count - second of importance (likely won't implement the tech on a high number of ponds)
    max_pond_count = max_num_ponds
    microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Pond_Count'] <= max_pond_count]
    # MWS Area - less important (?) becuase a large area could still have favorable above characteristics
    max_mws_area = 500
    # microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Area_Acres'] <= max_mws_area]

    # Filter out catchments with an order above 2
    microwatersheds_filter_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Order'] < 3]

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
    microwatersheds_filter_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='black', alpha=0.5)
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
    microwatersheds_filter_gdf.rename(columns={'Pond_Area_Percentage': 'Pond /_MWS Area_Percentage'}, inplace=True)
    microwatersheds_filter_gdf.rename(columns={'Pondshed_to_MWS_Percentage': 'Pondshed /_MWS Area_Percentage'}, inplace=True)
    microwatersheds_filter_gdf.rename(columns={'Microwatershed_ID': 'Microwshed_ID'}, inplace=True)
    microwatersheds_filter_gdf.rename(columns={'Nitrogen_lb': 'Nitrogen_Lb_ByLandUse'}, inplace=True)

    # Select only the specified columns and order by Total_Pond_Area_Acres
    columns_to_display = ['Microwshed_ID', 
                        'Pond_Count', 
                        'Area_Acres', 
                        #   'Average_Pond_Area_Acres', 
                        'Total_Pond_Area_Acres', 
                        'Total_Pondshed_Area_Acres',
                        # 'Pondshed_to_Pond_Ratio',
                        # 'Pond /_MWS Area_Percentage',
                        'Pondshed /_MWS Area_Percentage',
                        'Pond_Controllable_Volume_Ac-Ft', 
                        'Annual_Volume_Treated_MG/Yr', 
                        'Nitrogen_Lb_ByLandUse', 
                        'Total_Nitrogen_(Lb/Yr)', 
                        'Controllable_Nitrogen_(Lb/Yr)',
                        # 'Total_Phosphorous_(Lb/Yr)', 
                        'Percent_Impervious', 
                        'Percent_Urban']
    filter_df = microwatersheds_filter_gdf[columns_to_display].sort_values(by='Total_Pondshed_Area_Acres', ascending=False)
    # filter_df = filter_df[filter_df['Microwatershed_ID'] == 121]

    format_columns = {
        'Microwshed_ID': '{:.0f}',
        'Pond_Count': '{:.0f}',
        'Area_Acres': '{:.2f}',
        # 'Average_Pond_Area_Acres': '{:.2f}',
        'Total_Pond_Area_Acres': '{:.2f}',
        'Total_Pondshed_Area_Acres': '{:.2f}',
        # 'Pondshed_to_Pond_Ratio': '{:.2f}',
        # 'Pond /_MWS Area_Percentage': '{:.2f}',
        'Pondshed /_MWS Area_Percentage': '{:.2f}',
        'Pond_Controllable_Volume_Ac-Ft': '{:.2f}',
        'Annual_Volume_Treated_MG/Yr': '{:.2f}',
        'Nitrogen_Lb_ByLandUse': '{:.4f}',
        'Total_Nitrogen_(Lb/Yr)': '{:.2f}',
        'Controllable_Nitrogen_(Lb/Yr)': '{:.2f}',
        # 'Total_Phosphorous_(Lb/Yr)': '{:.2f}',
        'Percent_Impervious': '{:.2f}',
        'Percent_Urban': '{:.2f}'
    }

    for col, fmt in format_columns.items():
        filter_df[col] = filter_df[col].map(fmt.format)

    # Print the DataFrame
    # print(filter_df.head(20))

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
        # 'Pondshed_to_Pond_Ratio': plt.cm.plasma,
        # 'Average_Pond_Area_Acres': plt.cm.plasma,
        # 'Pond /_MWS Area_Percentage': plt.cm.plasma,
        'Pondshed /_MWS Area_Percentage': plt.cm.plasma,
        'Pond_Controllable_Volume_Ac-Ft': plt.cm.plasma,
        'Annual_Volume_Treated_MG/Yr': plt.cm.plasma,
        'Nitrogen_Lb_ByLandUse': plt.cm.plasma,
        'Total_Nitrogen_(Lb/Yr)': plt.cm.plasma,
        'Controllable_Nitrogen_(Lb/Yr)': plt.cm.plasma,
        # 'Total_Phosphorous_(Lb/Yr)': plt.cm.plasma,
        'Percent_Impervious': plt.cm.plasma,
        'Percent_Urban': plt.cm.plasma,
    }

    col_colors = {col: create_color_map(filter_df, col, cmap) for col, cmap in columns_to_color.items()}


    table.auto_set_font_size(False)
    table.set_fontsize(7)
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
    output_file_path = f"outputs\shp\{datetime_str}\Microwatersheds_{file}_{datetime_str}.gpkg"

    # Export the GeoDataFrame to a shapefile
    polygon.to_file(output_file_path, driver='GPKG')

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



def main(file, epsg, units, aggregation, flow_file_path, condition_burn, burn_width, flowlines_burn_value, osm_burn_value, pond_burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds=50):

    # If testing locally, use path
    c_path = 'C:\\Users\\alden.summerville\\Documents\\dev-local\\IRL-MIA-development\\'
    # If deployed, use relative paths
    # c_path = ''

    ## Run all the consequetive functions
    dem_path, grid, dem = read_reproject_dem(c_path, file)

    pdf_path, pdf_pages = initialize_pdf(c_path, file, epsg, units, aggregation, flow_file_path, burn_width, flowlines_burn_value, pond_burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds)
    
    if aggregation != 1:
        dem_path, grid, dem = aggregate_dem(c_path, dem_path, aggregation)
    else:
        print("Skipped aggregation")
        dem_path = dem_path

    crs_dem = confirm_crs(dem_path)

    grid_cd, dem_cd, dem_path_cd = condition_dem(grid, dem, c_path)

    if flowlines_burn_value or pond_burn_value or osm_burn_value != 0:
        if condition_burn == 1:
            grid_cd, dem_cd, dem_path_cd = burn_features(c_path, crs_dem, dem_path_cd, burn_width, flowlines_burn_value, osm_burn_value, pond_burn_value, dem_cd, file, aggregation)
        grid, dem, dem_path = burn_features(c_path, crs_dem, dem_path, burn_width, flowlines_burn_value, osm_burn_value, pond_burn_value, dem, file, aggregation)
    else:
        print("Skipped burning")
        dem_path = dem_path

    pdf_pages = plot_burned_dem(dem, grid, pdf_pages, crs_dem, aggregation, flowlines_burn_value, burn_width, units)

    branches_, microwatersheds_gdf = delineate_microwatersheds(grid, dem, dem_cd, river_network_min_flow_acc, condition_burn)

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
    microwatersheds_all_gdf = calculate_pondshed_impervious(impervious_raster, microwatersheds_all_gdf, pondsheds)

    rainfall_data = 'data-inputs\\temp\\temp_reprojected_raster_rainfall.tif'
    microwatersheds_all_gdf = get_annual_rainfall(microwatersheds_all_gdf, rainfall_data)

    microwatersheds_all_gdf = annual_control_volume(microwatersheds_all_gdf)

    land_cover = gpd.read_file(r'data-inputs\\LandCover\\Land_Cover_IRL_4326.shp')
    microwatersheds_all_gdf = urban_area(land_cover, microwatersheds_all_gdf)
    microwatersheds_all_gdf = calculate_nutrients(land_cover, microwatersheds_all_gdf)

    pdf_pages, filter_df, microwatersheds_filter_gdf = filter_mws_characteristics(microwatersheds_all_gdf, grid, dem, ponds_intersect, pdf_pages, min_total_pond_area, max_num_ponds, pondsheds)

    export_microwatersheds(microwatersheds_all_gdf)

    close_pdf(pdf_pages)

    # dash_map(filter_df, microwatersheds_filter_gdf)

    # m = interactive_map(ponds_intersect, microwatersheds_nutrients_gdf, branches_)

    return pdf_path


# Streamlit app layout
st.title("Microwatershed Impact Assessment - Python Tool")

# DEM file inputs
dems = ["Pineda_Scalgo_8m_NAD83", 
        "Terrain_Grant_Valkaria_ClipNoData_NAD83",
        "Melbourne_NAD83",
        "Tomako_FLA",
        "MerritIsland_1m_26917",
        "PalmBay_1m_26917",
        "GrandHarbor_1m_26917",
        "SouthPatrickTIFF",
        "UpperCanalMosaic_1_10",
        "UpperCanalMosa_1m_NAD83",
        "IRL-full-region9",
        "IRL-full-region6",
        "IRL-full-region3",
        "IRL-full-region2",
        "IRL-full-region12",
        "IRL-full-region13",
        "IRL-full-region"]

flowline_datasets = ["FLA-NHD-Flowlines_NAD83.shp"]

# User inputs
file = st.selectbox("Enter the DEM file name (without extension):", dems)
epsg = st.selectbox("Enter the EPSG code (2881 for StatePlane Florida East. 26917 for NAD83 Zone 17N):", ["26917", "2881"])
units = st.selectbox("Enter the units of the DEM", ["Meters", "US Foot"])
river_network_min_flow_acc = st.number_input("Minimum Flow Accumulation - Channels:", min_value=0, value=2500)

aggregation = st.number_input("DEM Aggregation Factor:", min_value=0, value=1)
pre_condition_smooth = st.number_input("Pre-condition smooth factor:", min_value=0, value=0)

flow_file_path = st.selectbox("Enter the clipped flowlines path:", flowline_datasets)

condition_burn = st.selectbox("Burn after condition?", [0,1])
burn_width = st.number_input("Burn Width:", min_value=1, value=1)
flowlines_burn_value = st.number_input("Burn Value:", value=0.0)
osm_burn_value = st.number_input("OSM Burn Value:", value=0.0)
pond_burn_value = st.number_input("Pond Burn Value:", value=0.0)

min_total_pond_area = st.number_input("Minimum Total Pond Area per Microwatershed:", min_value=0, value=15)
max_num_ponds = st.number_input("Max Number of Ponds per Microwatershed:", min_value=0, value=25)


# Button to run the main function
if st.button("Run"):
    with st.spinner("Running..."):
        pdf_path = main(file, epsg, units, aggregation, flow_file_path, condition_burn, burn_width, flowlines_burn_value, osm_burn_value, pond_burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds)
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