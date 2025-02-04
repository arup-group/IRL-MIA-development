from pysheds.grid import Grid
import pandas as pd
import random
import geopandas as gpd
from shapely import geometry, ops
import os
import numpy as np
import folium
from pyproj import Geod
import glob
import time
from shapely.geometry import Point
from shapely.errors import ShapelyDeprecationWarning
import warnings
import argparse
from tqdm import tqdm

# filter warnings for now - code will need updating for Shapely 2.0+
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning) 

# global variables
geod = Geod(ellps="WGS84")

######################################################################
#                      Helper Functions
######################################################################

def calc_attributes(row,dem,profiles,so):
    '''Takes a stream segment as input and assigns various 
       elements related to its geometry and stream order. 
       Topographic metrics such as relief and slope are 
       approximations only.'''
    
    index = row.name 
    row['Length'] = geod.geometry_length(row['geometry'])
    row['Relief'] = int(np.ptp(dem.flat[profiles[index]]))
    
    row['Order'] = int(np.min(so.flat[profiles[index]]))
    # note: 'so.flat[profiles[index]]' returns a 1d array of length n
    # where all values are equal to the stream order for that profile.
    # so if it is a 1st order channel, the array has all 1's. 
    
    row['Slope'] = np.rad2deg(np.arctan2(row['Relief'],row['Length']))
    row['Index'] = row.name
    return row 

def make_connections(profiles, connections):
    '''creates full connections map for stream network.'''
    
    profile_list = []
    connection_list = []
    
    for i in range(0,len(profiles)): # loop through each segment profile
        
        index = i
        chain = []
        connection_map = []
        
        # create full profile and connection map
        while True:
            chain.extend(profiles[index]) # add segment
            connection_map.extend([index]) # add that segment id
            next_index = connections[index] # get idx of the connecting segment
            
            # if next idx is the same as before, then the full profile is done
            if index == next_index:  
                break
            index = next_index # otherwise go to the next connecting segment
        
        profile_list.append(chain) # add full profile
        connection_list.append(connection_map) # add full connection mapping
    
    return profile_list, connection_list

def make_profile(row,so,coords,profile_list,connection_list):
    index = row.name # stream profile index 
    row_order = row['Order']
    profile = profile_list[index] # grab full profile for segment
        
    so_list = so.flat[profile].tolist() # get stream orders along full profile
    so_list.reverse() # reversing to perform the next steps easier...
    
    final_idx = len(so_list) - so_list.index(row_order) - 1
    # ^^^ get last instance of this stream order in the full profile. 
    
    row['LocalPP'] = np.flip(coords[profile[final_idx]]) 
    # ^^^ use idx to get the appropriate coords for the pour point.
    # np.flip() puts it in the right order
    row['LocalPP_X'] = row['LocalPP'][0]
    row['LocalPP_Y'] = row['LocalPP'][1]
    
    
    row['Profile'] = profile_list[index] # add full profile
    row['Chain'] = connection_list[index] # add full chain mapping    
    
    row['Final_SO'] = so.flat[profile][-1]
    # idx of final segment along profile
    row['Final_Chain_Val'] = connection_list[index][-1] 
    return row

def make_basin(row,grid,dem,fdir,
               routing,algorithm):
    '''Calculate basin geometry via pour point 
       derived from above functions.'''
    
    # index = row.name
    # row_order = row['Order']    
    x, y = row['LocalPP_X'], row['LocalPP_Y']
    c = grid.catchment(x=x,y=y,fdir=fdir,routing=routing,algorithm=algorithm)
    grid.clip_to(c)
    # clipped_catch = grid.view(c)
    catchment_polygon = ops.unary_union([geometry.shape(shape) 
                                     for shape, value in grid.polygonize()])
    grid.viewfinder = dem.viewfinder
    
    row['BasinGeo']  = catchment_polygon
    
    return row

######################################################################
#                      Main Function
######################################################################

def generate_catchments(grid, dem,acc_thresh=3000,so_filter=3,
                        routing='d8',algorithm='iterative', shoreline_clip=False):
    '''Full workflow integrating the above functions.
       Process is as follows: 
       
       1. Read and process DEM
       2. Create stream network and connection map
       3. Determine relevent pour point locations
       4. Generate all catchments.
       5. Return gdf of all catchment data and stream network.
       
       '''
    print('Starting pysheds...')
    # grid = Grid.from_raster(path)
    # dem = grid.read_raster(path)
    
    local_start = time.time()
    
    pit_filled_dem = grid.fill_pits(dem)
    
    # Fill depressions in DEM
    flooded_dem = grid.fill_depressions(pit_filled_dem)
    
    # Resolve flats in DEM
    inflated_dem = grid.resolve_flats(flooded_dem)
    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    print("Pysheds - started post conditioning burn")
    # Try burning here?
    import geopandas as gpd
    import rasterio
    from rasterio.features import rasterize
    import numpy as np
    from shapely.geometry import box
    # Save the reprojected raster
    crs_path = 'data-inputs\\temp\\temp_reprojected_raster.tif'
    with rasterio.open(crs_path) as src:
        temp = src.read(1)  # Read the first band

    with rasterio.open(
        "data-inputs\\temp\\TempPostCondition_DEM.tif",
        "w",
        driver="GTiff",
        width=inflated_dem.shape[1],
        height=inflated_dem.shape[0],
        count=1,
        dtype=inflated_dem.dtype,
        crs=src.crs,
        transform=src.transform
    ) as dst:
        dst.write(inflated_dem, 1)

    dem_path = "data-inputs\\temp\\TempPostCondition_DEM.tif"
    grid_cd = Grid.from_raster(dem_path)
    dem_cd = grid_cd.read_raster(dem_path)
    print("Pysheds - wrote raster")

    # PARAMETER: Burn the flowlines into the DEM
    burn_value = 0  # Adjust this value as needed


    if burn_value != 0:
        # Load the flowlines dataset
        print('Pysheds - in burn value code')
        flowlines = gpd.read_file(r'data-inputs\\NHD-Flowlines\\FLA-NHD-Flowlines_NAD83.shp')
        flowlines = flowlines.to_crs(epsg=4269)

        # with rasterio.open(dem_path) as src:
        #     bounds = src.bounds
        # # Create a bounding box geometry
        # bbox = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
        # bbox_gdf = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src.crs)

        # # Clip flowlines to DEM extent
        # flowlines_clip = gpd.clip(flowlines, bbox_gdf)

        # # Convert back to CRS with meters to do the buffer
        # flowlines_clip = flowlines_clip.to_crs(epsg=26917)
        # # PARAMETER: burn width
        # flowlines_clip['geometry'] = flowlines_clip.geometry.buffer(1)  # Buffer by half the desired width
        # # Convert back to CRS with lat lon
        # flowlines_clip = flowlines_clip.to_crs(epsg=4269)

        # # Load the shorelines dataset
        # shoreline = gpd.read_file(r'data-inputs\\Shoreline\\FLA-Shoreline_4269.shp')

        # from shapely.geometry import Polygon
        # # Clip the shoreline dataset to the AOI
        # area_of_interest = Polygon([(grid.bbox[0], grid.bbox[1]), (grid.bbox[0], grid.bbox[3]), (grid.bbox[2], grid.bbox[3]), (grid.bbox[2], grid.bbox[1]), (grid.bbox[0], grid.bbox[1])])
        # area_of_interest = gpd.GeoDataFrame(index=[0], crs="EPSG:4269", geometry=[area_of_interest])

        # shoreline_clip_temp = gpd.clip(shoreline, area_of_interest)

        # shoreline_clip_temp = shoreline_clip_temp[shoreline_clip_temp['ATTRIBUTE'] != 'Land']
        # shoreline_clip_temp = shoreline_clip_temp[shoreline_clip_temp['Shape_Area'] > 0.000002]

        # print("performing shoreline clip...")
        # # Clip the channels polyline by the shoreline dataset
        # flowlines_clip = gpd.clip(flowlines_clip, shoreline_clip_temp)

        # Ensure the flowlines are in the same CRS as the DEM
        # file = 'Terrain_Grant_Valkaria_ClipNoData_AggMedian16_NAD83'
        with rasterio.open(dem_path) as src:
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
        with rasterio.open(dem_path) as src:
            dem = src.read(1)  # Read the first band


        dem_burned = np.where(flowline_raster == 1, dem - burn_value, dem)

        # Save the modified DEM
        with rasterio.open(
            fr'data-inputs\\temp\\Agg_Burned.tif', 
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

        dem_agg_burn_path = fr'data-inputs\\temp\\Agg_Burned.tif'
        grid = Grid.from_raster(dem_agg_burn_path)
        grid_clip = Grid.from_raster(dem_agg_burn_path) # to be clipped by the delineation extent to preserve the original grid
        dem = grid.read_raster(dem_agg_burn_path)

        print("Flowlines have been burned into the DEM and saved as a new file.")
    else:
        print("No burn value provided. Skipping the burn process.")

    print("Pysheds flow direction...")
    # Compute flow directions
    # -------------------------------------
    # fdir = grid.flowdir(dem, dirmap=dirmap)
    fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    
    # make flow accumulation raster
    print('Pysheds flow accumulaton map...')
    acc = grid.accumulation(fdir, dirmap=dirmap)
    # set mask
    acc_mask = (acc_thresh < acc )
    # calculate stream order
    so = grid.stream_order(fdir=fdir,mask=acc_mask)
    # update mask
    mask = acc_mask
    
    # make river network
    print('Pysheds stream network...')
    branches = grid.extract_river_network(fdir=fdir,mask=mask) # returns geojson
    print(len(branches['features']), "branches generated")

    # Clip branches to IRL region****
    print("Clipping branches to IRL region")
    irl_region = gpd.read_file(r'data-inputs\\IRL-boundary\\IRL-AOI_4326.shp')
    branches = gpd.GeoDataFrame.from_features(branches,crs='epsg:4326')
    # print(type(branches), branches.crs)
    branches = gpd.clip(branches, irl_region)
    print("Post IRL clip")

    if shoreline_clip == True:
        print("Pysheds branches - reading in shorelines")
        # Load the shorelines dataset
        shoreline = gpd.read_file(r'data-inputs\\Shoreline\\FLA-Shoreline_4269.shp')
        print("shorelines read in successfully")
        # make main GeoDataFrame -- the indices here will match those of all profile
        # and connection lists that follow
        branch_gdf_i = branches.to_crs('epsg:4269')

        from shapely.geometry import Polygon
        # Clip the shoreline dataset to the AOI
        area_of_interest = Polygon([(grid.bbox[0], grid.bbox[1]), (grid.bbox[0], grid.bbox[3]), (grid.bbox[2], grid.bbox[3]), (grid.bbox[2], grid.bbox[1]), (grid.bbox[0], grid.bbox[1])])
        area_of_interest = gpd.GeoDataFrame(index=[0], crs="EPSG:4269", geometry=[area_of_interest])

        shoreline_clip = gpd.clip(shoreline, area_of_interest)
        # Filter shoreline dataset for only shorelines and meaningful area sizes
        shoreline_clip = shoreline_clip[shoreline_clip['ATTRIBUTE'] != 'Land']
        shoreline_clip = shoreline_clip[shoreline_clip['Shape_Area'] > 0.000002] # Equivalent to approx > 5 acres
        print("shorelines filtered by attribute column")
        

        branch_gdf = gpd.clip(branch_gdf_i, shoreline_clip)
        print("Post shoreline clip")
        print(len(branch_gdf), "branches remain")
    else:
        branch_gdf = gpd.GeoDataFrame.from_features(branches,crs='epsg:4326')
    
    

    # generate profiles for each individual segment
    profiles, connections = grid.extract_profiles(fdir=fdir,mask=mask,include_endpoint=False)
     
    branch_gdf = branch_gdf.apply(lambda x: calc_attributes(x,dem,profiles,so),axis=1)
    coords = dem.coords
    
    profile_list, connection_list = make_connections(profiles, connections)
    
    # create all profile and connection lists w/ pour points
    print('Calculating pour points...')
    branch_gdf = branch_gdf.apply(lambda x: make_profile(x,so,coords,profile_list,connection_list),axis=1)
    
    branch_gdf_copy = branch_gdf.copy() # unfiltered copy to return at end
    
    # some of these sections will have duplicate orders and pour points - drop them
    unique = branch_gdf[['Order','LocalPP_X','LocalPP_Y','Final_Chain_Val']].drop_duplicates()
    
    branch_gdf = branch_gdf.loc[unique.index]
    
    # filter by stream order
    if so_filter:
        branch_gdf = branch_gdf[branch_gdf['Order']<=so_filter]
    
    print('Pysheds generating',len(branch_gdf),'catchments...')
    branch_gdf = branch_gdf.apply(lambda x: make_basin(x,grid,dem,fdir,routing,algorithm),axis=1)

    b_copy = branch_gdf.copy()
    b_copy.set_geometry('BasinGeo',inplace=True)
    b_copy.set_crs('epsg:4326',inplace=True)
    
    copy_for_area = b_copy.copy()
    copy_for_area.to_crs('epsg:6933',inplace=True)
    copy_for_area.geometry.area
    copy_for_area['AreaSqKm'] = copy_for_area.geometry.area  / 1000000

    b_copy['AreaSqKm'] = copy_for_area['AreaSqKm']
    
    basin_drop_cols = ['geometry','LocalPP'] #,'Profile','Chain']
    branch_drop_cols = ['LocalPP'] #,'Profile','Chain']
    
    b_copy.drop(basin_drop_cols,axis=1,inplace=True)
    branch_gdf_copy.drop(branch_drop_cols,axis=1,inplace=True)
    
    b_copy['Profile'] = b_copy['Profile'].astype(str)
    b_copy['Chain'] = b_copy['Chain'].astype(str)

    branch_gdf_copy['Profile'] = branch_gdf_copy['Profile'].astype(str)
    branch_gdf_copy['Chain'] = branch_gdf_copy['Chain'].astype(str)
    
    local_total = (time.time() - local_start)
    
    print('Total runtime is',str(local_total/60),'minutes')
    
    return b_copy, branch_gdf_copy



    
    