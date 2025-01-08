
import streamlit as st
import os
import subprocess

from pysheds.grid import Grid
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
import rasterio
from collections import defaultdict
import geopandas as gpd
from shapely import geometry
from streamlit_folium import st_folium

# key inputs
# file = 'Terrain_Grant_Valkaria_ClipNoData_NAD83'
# aggregation = 16
# flow_file_path = 'IRL-Flowlines-Export_NAD83.shp'
# burn_width=4
# burn_value = -3  # Adjust this value as needed
# river_network_min_flow_acc = 1000
# min_total_pond_area = 20

def read_reproject_dem():
    pass




def main(file, epsg, units, aggregation, flow_file_path, burn_width, burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds):

    # PARAMETER - input DEM
    # Note, whatever CRS the tif is in will influence the units
    # Typically either NAD 1983 UTM Zone 17N - EPSG:26917
    # Or NAD 1983 HARN Florida GDL Albers (Meters) - EPSG:3087

    # If testing locally, use path
    c_path = 'C:\\Users\\alden.summerville\\Documents\\dev-local\\IRL-MIA-development\\'
    # If deployed, use relative paths
    # c_path = ''


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
    
    # Create a PDF file to save the plots
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt
    pdf_path = f'{c_path}jupyter\\outputs\\Output_{file}_{burn_value}Burn_{min_total_pond_area}MinTotPondArea_{max_num_ponds}MaxNumPonds.pdf'
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

    
    # Confirm area units check out (square meters)
    import rasterio
    import numpy as np

    # Open the raster file
    with rasterio.open(dem_agg_path) as src:
        # Read the data
        data = src.read(1)
        
        # Get the pixel size (resolution)
        pixel_width, pixel_height = src.res
        pixel_size = pixel_width * pixel_height
        
        # Calculate the number of valid (non-NaN) pixels
        valid_pixels = np.count_nonzero(~np.isnan(data))
        
        # Calculate the total area
        total_area = valid_pixels * pixel_size
        
        if units == "Meters":
            print(f'Total area: {total_area} square meters')
            print(f'Total area: {total_area/4046.85642} acres')
        elif units == "US Foot":
            print(f'Total area: {total_area} square feet')
            print(f'Total area: {total_area/43560} acres')


    
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
    plt.title('Digital elevation map', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    
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
    plt.title(f'Digital Elevation Map - {aggregation}x{aggregation} - {burn_value} depth, {burn_width*2} wide Burn - EPSG:{crs_dem}, {units}', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    # Save plot to pdf
    pdf_pages.savefig(fig)


    # DEM Conditioning steps
    # NOTE: don't need to run if the above steps were run

    # condition the DEM
    pit_filled_dem = grid.fill_pits(dem)

    flooded_dem = grid.fill_depressions(pit_filled_dem)
        

    inflated_dem = grid.resolve_flats(flooded_dem)

    print(inflated_dem.dtype)


    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    # if 'nodata' error (numpy issue)  
    # fdir = grid.flowdir(inflated_dem, dirmap=dirmap, nodata_out=np.int64(0))

    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)

    
    # calculate flow direction
    fig = plt.figure(figsize=(8,6))
    fig.patch.set_alpha(0)

    plt.imshow(fdir, extent=grid.extent, cmap='viridis', zorder=2)
    boundaries = ([0] + sorted(list(dirmap)))
    plt.colorbar(boundaries= boundaries,
                values=sorted(dirmap))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Flow direction grid', size=14)
    plt.grid(zorder=-1)
    plt.tight_layout()

    
    # Calculate and plot flow accumulation

    #unproj dem
    acc = grid.accumulation(fdir, dirmap=dirmap)
    fig, ax = plt.subplots(figsize=(8,6))
    fig.patch.set_alpha(0)
    plt.grid('on', zorder=0)
    im = ax.imshow(acc, extent=grid.extent, zorder=2,
                cmap='cubehelix',
                norm=colors.LogNorm(1, acc.max()),
                interpolation='bilinear')
    # ponds.plot()
    plt.colorbar(im, ax=ax, label='Upstream Cells')
    plt.title('Flow Accumulation', size=14)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()

    # Save plot to pdf
    pdf_pages.savefig(fig)

    
    # Plot the distribution of flow accumulation

    import matplotlib.pyplot as plt

    # Assuming 'acc' is the flow accumulation raster array
    # Flatten the array to get all values
    acc_values = acc.flatten()

    # Plot the distribution of the flow accumulation values
    plt.figure(figsize=(10, 6))
    plt.hist(acc_values, bins=50, color='blue', edgecolor='black')
    plt.title('Distribution of Flow Accumulation Values')
    plt.xlabel('Flow Accumulation Value')
    plt.ylabel('Frequency')
    plt.yscale('log')  # Use logarithmic scale for better visualization
    plt.grid(True)
    plt.tight_layout()
    pdf_pages.savefig(plt.gcf())  # Save the histogram plot to the PDF

    plt.show()


    
    # Find river network branches within wider area
    # NOTE: if you want to delineate catchments based on these channels, this 'brances' dataset will be the input

    # PARAMETER - river_network_min_flow_acc
    river_network_min_flow_acc = river_network_min_flow_acc
    # unprojected DEM
    branches = grid.extract_river_network(fdir, acc > river_network_min_flow_acc, dirmap=dirmap,  nodata_out=np.int64(0))

    sns.set_palette('husl')
    fig, ax = plt.subplots(figsize=(8,6))

    plt.xlim(grid.bbox[0], grid.bbox[2])
    plt.ylim(grid.bbox[1], grid.bbox[3])
    ax.set_aspect('equal')

    for branch in branches['features']:
        line = np.asarray(branch['geometry']['coordinates'])
        plt.plot(line[:, 0], line[:, 1])

    plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.4)
        
    plt.title(f'D8 Channels - Min Flow Acc of {river_network_min_flow_acc}', size=14)
    print("Saving figure...")
    # Save plot to pdf
    pdf_pages.savefig(fig)
    # plt.close(fig)  # Close the figure to free up memory

    
    
    from collections import defaultdict
    import numpy as np

    node_counts = defaultdict(int)
    precision = 10

    for branch in branches['features']:
        line = np.asarray(branch['geometry']['coordinates'])
        
        # # Add the starting point to the node_counts
        # starting_point = tuple(np.round(line[0], precision))
        # node_counts[starting_point] += 1

        # # Add the ending point to the node_counts
        # ending_point = tuple(np.round(line[-1], precision))
        # node_counts[ending_point] += 1
        
        for point in line:
            rounded_point = tuple(np.round(point, precision))
            node_counts[rounded_point] += 1

    # Identify junctions (points that appear more than once)
    junctions = [node for node, count in node_counts.items() if count > 1]

    print(f"Junction points: {junctions}")
    print(len(junctions))


    
    # # Delineate MICROWATERSHEDS

    # #unproj DEM
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import matplotlib.colors as mcolors
    # import geopandas as gpd
    # from shapely.geometry import shape
    # from rasterio.features import shapes

    # # Create a figure for plotting all catchments
    # fig, ax = plt.subplots(figsize=(8,6))

    # # Set the plot boundaries and aspect ratio
    # plt.xlim(grid.bbox[0], grid.bbox[2])
    # plt.ylim(grid.bbox[1], grid.bbox[3])
    # plt.gca().set_aspect('equal')

    # # Define colormap for unique catchment colors
    # colors = plt.cm.get_cmap('tab20b', len(junctions))

    # # INPUT PARAMETER: Define the minimum and maximum accumulation thresholds
    # mws_min_flow_acc = 1
    # mws_max_flow_acc = 500000

    # # Initialize an empty list to store microwatersheds vectors
    # geometries = []
    # values = []

    # # Loop over each junction and plot its catchment
    # for i, junction in enumerate(junctions):
    #     x, y = junction
        
    #     # Snap the junction to the nearest cell where accumulation is between min_acc and max_acc
    #     mask = (acc > mws_min_flow_acc) & (acc < mws_max_flow_acc)
    #     x_snap, y_snap = grid.snap_to_mask(mask, (x, y))
        
    #     # Delineate the catchment for the snapped point
    #     catch = grid.catchment(x=x_snap, y=y_snap, fdir=fdir, dirmap=dirmap, xytype='coordinate')
    #     # print(type(catch))
        
    #     # Convert the catchment to vector and add to list (to be transformed into a gdf)
    #     data = catch.astype(np.float32)
    #     transform = catch.affine
    #     mask = data != catch.nodata
    #     # Extract shapes (vector geometries) from the raster
    #     shapes_generator = shapes(data, mask=mask, transform=transform)
    #     # print(shapes_generator)
    #     # Convert shapes to Shapely geometries
    #     for geom, value in shapes_generator:
    #         geometries.append(shape(geom))
    #         values.append(value)
        
    #     # Plot the catchment with a unique color
    #     plt.imshow(np.where(catch, catch, np.nan), extent=grid.extent, cmap=mcolors.ListedColormap([colors(i)]), alpha=0.5)

    # # Plot the river network on top of the catchments
    # for branch in branches['features']:
    #     line = np.asarray(branch['geometry']['coordinates'])
    #     plt.plot(line[:, 0], line[:, 1], color='black')

    # # Highlight all junction points
    # x, y = zip(*junctions)
    # plt.scatter(x, y, color='red', s=5, zorder=5)

    # # im3 = ax.imshow(flats, extent=grid.extent, zorder=3, cmap='Greys_r', alpha=0.6)

    # # Create geodataframe of microwatersheds
    # microwatersheds_gdf = gpd.GeoDataFrame({'geometry': geometries, 'value': values})
    # microwatersheds_gdf.set_crs(epsg=crs_dem, inplace=True)
    # # print(microwatersheds_gdf)

    # # Add title and show the combined plot
    # plt.title('Microwatersheds - Delineated from Channel Junction Points')
    # plt.show()

    # # Save plot to pdf
    # pdf_pages.savefig(fig)

    # confirm crs
    # crs = microwatersheds_gdf.crs
    # print(crs)

    # NEW METHOD to delineate the catchments and stream orders 
    import folium
    import make_catchments

    basins_, branches_ = make_catchments.generate_catchments(dem_path,acc_thresh=river_network_min_flow_acc,so_filter=4)

    # Visualize output
    mws_with_stream_order = basins_.copy()
    # mws_with_stream_order

    # To use the basins wrapper output (switch the use the gdf)

    microwatersheds_gdf = mws_with_stream_order

    # Create a figure for plotting all catchments
    fig, ax = plt.subplots(figsize=(8,6))
    plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
    # Set the plot boundaries and aspect ratio
    plt.xlim(grid.bbox[0], grid.bbox[2])
    plt.ylim(grid.bbox[1], grid.bbox[3])
    plt.gca().set_aspect('equal')
    mws_with_stream_order.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
    branches_.plot(ax=ax, aspect=1, color='black')
    # Plot ponds
    # Add title and show the combined plot
    plt.title('Microwatersheds - Delineated from Channel Junction Points')
    plt.show()
    # Save plot to pdf
    pdf_pages.savefig(fig)


    # print(microwatersheds_gdf)
    fig, ax = plt.subplots()
    microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='black', alpha=0.7)
    # im3 = ax.imshow(flats, extent=grid.extent, zorder=3, cmap='Greys_r', alpha=0.5)
    plt.show()

    
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


    # Create the plot
    fig, ax = plt.subplots()
    microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='black', alpha=0.5)

    # Add labels
    # for idx, row in microwatersheds_gdf.iterrows():
    #     plt.annotate(text=row['Microwatershed_ID'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
    #                 xytext=(3, 3), textcoords='offset points', fontsize=8, color='black')

    plt.show()


    
    # Pull in ponds dataset and intersect

    import pandas as pd
    import geopandas as gpd
    from shapely import geometry, ops

    # Load ponds data
    # NOTE make sure to read in the shapefile with a CRS aligned with the DEM
    ponds = gpd.read_file(fr'{c_path}jupyter\\data-inputs\\IRL-Ponds-Export\\IRL-Ponds-Export_4269.shp')
    # ponds.to_crs(epsg=crs_dem, inplace=True)

    # Reproject to a suitable projected CRS (e.g., UTM Zone 17N)
    ponds_projected = ponds.to_crs(epsg=26917)

    # Calculate area in square meters
    ponds_projected['Area_SqMeters'] = ponds_projected.geometry.area

    # Convert to acres
    ponds_projected['area'] = ponds_projected['Area_SqMeters'] / 4046.85642

    # Create a unique Pond_ID for each pond
    ponds_projected['Pond_ID'] = range(1, len(ponds_projected) + 1)

    # Create a new GeoDataFrame with Pond_ID and Area_Acres
    area_acres_gdf = ponds_projected[['Pond_ID', 'area']]

    # Merge the area data back to the original GeoDataFrame using Pond_ID
    ponds = ponds.merge(area_acres_gdf, on='Pond_ID', how='left')
    
    # # OLD Calculate the area of each pond
    # if units == "Meters":
    #     ponds['area'] = ponds.geometry.area/4046.85642
    # elif units == "US Foot":
    #     ponds['area'] = ponds.geometry.area/43560

    # NOTE Filter out ponds with an area less than 1 acre
    ponds = ponds[ponds['area'] >= 0.25]

    # Find intersecting ponds
    ponds_intersect = gpd.sjoin(ponds, microwatersheds_gdf, how='inner', predicate='intersects')

    # Count the number of intersecting ponds for each microwatershed
    pond_counts = ponds_intersect.groupby('index_right').size().reset_index(name='Pond_Count')

    # Sum the area of intersecting ponds for each microwatershed
    pond_area_sum = ponds_intersect.groupby('index_right')['area'].sum().reset_index(name='Total_Pond_Area_Acres')

    # Calculate the average pond area within each microwatershed
    pond_area_avg = ponds_intersect.groupby('index_right')['area'].mean().reset_index(name='Average_Pond_Area_Acres')

    # Combine pond_counts, pond_area_sum, and pond_area_avg into a single DataFrame
    pond_summary = pond_counts.merge(pond_area_sum, on='index_right').merge(pond_area_avg, on='index_right')

    # Merge the combined summary DataFrame back into the microwatersheds_gdf
    microwatersheds_all_gdf = microwatersheds_gdf.merge(pond_summary, left_index=True, right_on='index_right', how='left')

    # Fill NaN values with 0 (if there are microwatersheds with no intersecting ponds)
    microwatersheds_all_gdf['Pond_Count'] = microwatersheds_all_gdf['Pond_Count'].fillna(0)
    microwatersheds_all_gdf['Total_Pond_Area_Acres'] = microwatersheds_all_gdf['Total_Pond_Area_Acres'].fillna(0)
    microwatersheds_all_gdf['Average_Pond_Area_Acres'] = microwatersheds_all_gdf['Average_Pond_Area_Acres'].fillna(0)

    # Calculate the ratio of total pond area to the area of the microwatershed
    microwatersheds_all_gdf['Pond_Area_Ratio'] = microwatersheds_all_gdf['Total_Pond_Area_Acres'] / microwatersheds_all_gdf['Area_Acres'] *100

    # Select only the specified columns and order by Pond_Count
    columns_to_display = ['Microwatershed_ID', 'Area_Acres', 'Order', 'Pond_Count', 'Total_Pond_Area_Acres', 'Average_Pond_Area_Acres', 'Pond_Area_Ratio']
    summary_df = microwatersheds_all_gdf[columns_to_display].sort_values(by='Pond_Count', ascending=False)

    # Print the DataFrame
    print(summary_df)



    
    # Print MWS and intersecting ponds

    # Create a figure for plotting all catchments
    fig, ax = plt.subplots(figsize=(8,6))

    # Set the plot boundaries and aspect ratio
    plt.xlim(grid.bbox[0], grid.bbox[2])
    plt.ylim(grid.bbox[1], grid.bbox[3])
    plt.gca().set_aspect('equal')
    #plot DEM with high transparency
    plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
    microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
    # Plot ponds
    ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='blue')

    # Add title and show the combined plot
    plt.title('Microwatersheds - Pond Overlay')
    plt.show()

    # Save plot to pdf
    pdf_pages.savefig(fig)

    
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

    # Print MWS and intersecting ponds

    # print(microwatersheds_gdf)
    fig, ax = plt.subplots(figsize=(8,6))
    # Set the plot boundaries and aspect ratio
    plt.xlim(grid.bbox[0], grid.bbox[2])
    plt.ylim(grid.bbox[1], grid.bbox[3])
    plt.gca().set_aspect('equal')
    cmap = plt.get_cmap('tab20')
    #plot DEM with high transparency
    plt.imshow(dem, extent=grid.extent, cmap='terrain', norm=norm, zorder=1, alpha=0.25)
    microwatersheds_filter_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='white', alpha=0.5)
    ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='blue')

    # Add labels
    # for idx, row in microwatersheds_filter_gdf.iterrows():
    #     plt.annotate(text=row['Microwatershed_ID'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
    #                 xytext=(3, 3), textcoords='offset points', fontsize=8, color='black')

    plt.title(f'Microwatersheds - Minimum Total Pond Area {min_total_pond_area} Acres. Max Number of Ponds {max_num_ponds}')
    plt.show()

    # Select only the specified columns and order by Total_Pond_Area_Acres
    columns_to_display = ['Microwatershed_ID', 'Area_Acres', 'Pond_Count', 'Total_Pond_Area_Acres', 'Average_Pond_Area_Acres', 'Pond_Area_Ratio']
    filter_df = microwatersheds_filter_gdf[columns_to_display].sort_values(by='Total_Pond_Area_Acres', ascending=False)

    # Format the DataFrame columns
    filter_df['Microwatershed_ID'] = filter_df['Microwatershed_ID'].astype(int)  # No decimal places
    filter_df['Pond_Count'] = filter_df['Pond_Count'].astype(int)  # No decimal places
    filter_df['Area_Acres'] = filter_df['Area_Acres'].map('{:.2f}'.format)  # Two decimal places
    filter_df['Total_Pond_Area_Acres'] = filter_df['Total_Pond_Area_Acres'].map('{:.2f}'.format)  # Two decimal places
    filter_df['Average_Pond_Area_Acres'] = filter_df['Average_Pond_Area_Acres'].map('{:.2f}'.format)  # Two decimal places
    filter_df['Pond_Area_Ratio'] = filter_df['Pond_Area_Ratio'].map('{:.4f}'.format)  # Four decimal places

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
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)  # Adjust the size of the table

    # Adjust header cell height if needed
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header row
            cell.set_height(0.12)  # Adjust height for better visibility

    pdf_pages.savefig(fig)  # Save the DataFrame table to the PDF
    
    # Filter for a single MWS

    MWS_ID = 22

    # Total Pond Area - likely the most important
    microwatersheds_inspect_gdf = microwatersheds_filter_gdf[microwatersheds_filter_gdf['Microwatershed_ID'] == MWS_ID]

    # Print MWS and intersecting ponds

    # print(microwatersheds_gdf)
    fig, ax = plt.subplots(figsize=(8,6))
    microwatersheds_inspect_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='black', alpha=0.5)
    ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='black')

    # Add labels
    # for idx, row in microwatersheds_inspect_gdf.iterrows():
    #     plt.annotate(text=row['Microwatershed_ID'], xy=(row.geometry.centroid.x, row.geometry.centroid.y),
    #                 xytext=(3, 3), textcoords='offset points', fontsize=8, color='black')

    plt.show()

    # Select only the specified columns and order by Pond_Count
    columns_to_display = ['Microwatershed_ID', 'Area_Acres', 'Pond_Count', 'Total_Pond_Area_Acres']
    inspect_df = microwatersheds_inspect_gdf[columns_to_display].sort_values(by='Total_Pond_Area_Acres', ascending=False)

    # Print the DataFrame
    print(inspect_df)

    
    # To develop:

    # If happy with the output, save the MWS polygons and Ponds to shapefiles

    
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming catchment_polygon, ponds_intersect, branches, and junctions are already defined
    # Create a GeoDataFrame from the catchment_polygon
    # catchment_gdf = gpd.GeoDataFrame(geometry=[catchment_polygon])

    # Plot the GeoDataFrame with a fixed aspect ratio
    fig, ax = plt.subplots(figsize=(8,6))
    microwatersheds_gdf.plot(ax=ax, aspect=1, cmap='tab20', edgecolor='black', alpha=0.5)

    # Plot the ponds_intersect on the same plot
    ponds_intersect.plot(ax=ax, aspect=1, color='blue', edgecolor='black')

    # Plot the river network on top of the catchments
    for branch in branches['features']:
        line = np.asarray(branch['geometry']['coordinates'])
        plt.plot(line[:, 0], line[:, 1], color='black')

    # Highlight all junction points
    x, y = zip(*junctions)
    plt.scatter(x, y, color='red', s=5, zorder=5)

    plt.show()

    
    # Close the PDF file
    pdf_pages.close()


    # Folium plotting
    from streamlit_folium import st_folium

    # Select only the specified columns and order by Pond_Count
    columns_to_display = ['Pond_ID', 'area', 'geometry']
    ponds_simple = ponds_intersect[columns_to_display].sort_values(by='area', ascending=False)
    
    # separate by stream order
    ones = microwatersheds_all_gdf[microwatersheds_all_gdf['Order']==1]
    twos = microwatersheds_all_gdf[microwatersheds_all_gdf['Order']==2]
    threes = microwatersheds_all_gdf[microwatersheds_all_gdf['Order']==3] # skip fours

    # Map 'em!
    cols = ['Microwatershed_ID', 'Area_Acres', 'Order', 'Pond_Count', 'Total_Pond_Area_Acres', 'Average_Pond_Area_Acres', 'Pond_Area_Ratio', 'BasinGeo']
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

    return pdf_path, m


# Streamlit app layout
st.title("Microwatershed Impact Assessment - Python Tool")

# DEM file inputs
dems = ["Terrain_Grant_Valkaria_ClipNoData_NAD83",
        "UpperCanalMosa_1m_NAD83"]

# User inputs
file = st.selectbox("Enter the DEM file name (without extension):", ["Terrain_Grant_Valkaria_ClipNoData_NAD83", "Terrain_Grant_Valkaria_ClipNoData_FLA", "PinedaScalgo_1m_NAD83", "Pineda_Scalgo_AggMedian16_NAD83", "Pineda_Scalgo_8m_NAD83", "Melbourne_NAD83", "Tomako_FLA", "UpperCanalMosa_1m_NAD83"])
epsg = st.selectbox("Enter the EPSG code (2881 for StatePlane Florida East. 26917 for NAD83 Zone 17N):", ["26917", "2881"])
units = st.selectbox("Enter the units of the DEM", ["Meters", "US Foot"])
aggregation = st.number_input("DEM Aggregation Factor:", min_value=0, value=16)
flow_file_path = st.selectbox("Enter the clipped flowlines path:", ["IRL-Flowlines-Export_NAD83.shp", "IRL-Pineda-Flowlines-Export_NAD83.shp", "IRL-UpperCanalFlowlines-Export_NAD83.shp"])
burn_width = st.number_input("Burn Width:", min_value=1, value=3)
burn_value = st.number_input("Burn Value:", value=-2)
river_network_min_flow_acc = st.number_input("Minimum Flow Accumulation - Channels:", min_value=0, value=1000)
min_total_pond_area = st.number_input("Minimum Total Pond Area per Microwatershed:", min_value=0, value=20)
max_num_ponds = st.number_input("Max Number of Ponds per Microwatershed:", min_value=0, value=50)


# Button to run the main function
if st.button("Run"):
    with st.spinner("Running..."):
        pdf_path, m = main(file, epsg, units, aggregation, flow_file_path, burn_width, burn_value, river_network_min_flow_acc, min_total_pond_area, max_num_ponds)
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