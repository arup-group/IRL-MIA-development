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

import osmnx as ox
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np
from shapely.geometry import box

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
    
    # row['Slope'] = np.rad2deg(np.arctan2(row['Relief'],row['Length']))
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
        
    so_list = np.array(so.flat[profile]) # get stream orders along full profile
    # so_list.reverse() # reversing to perform the next steps easier...
    
    final_idx = len(so_list) - np.where(so_list[::-1] == row_order)[0][-1] - 1
    # final_idx = len(so_list) - so_list.index(row_order) - 1
    # ^^^ get last instance of this stream order in the full profile. 
    
    local_pp = np.flip(coords[profile[final_idx]]) 
    # ^^^ use idx to get the appropriate coords for the pour point.
    # np.flip() puts it in the right order
    # row['LocalPP_X'] = row['LocalPP'][0]
    # row['LocalPP_Y'] = row['LocalPP'][1]
    pour = pd.Series({'LocalPP_X': local_pp[0], 'LocalPP_Y': local_pp[1]})
    
    # Commented out because we don't use the Profile or Chain or Final_SO attributes
    # row['Profile'] = profile_list[index] # add full profile
    # row['Chain'] = connection_list[index] # add full chain mapping    
    
    # row['Final_SO'] = so.flat[profile][-1]
    # # idx of final segment along profile
    # row['Final_Chain_Val'] = connection_list[index][-1] 
    return pour

def make_basin(row,grid,dem,fdir,
               routing,algorithm):
    '''Calculate basin geometry via pour point 
       derived from above functions.'''
    
    # index = row.name
    # row_order = row['Order']    
    x, y = row['LocalPP_X'], row['LocalPP_Y']
    c = grid.catchment(x=x,y=y,fdir=fdir,routing=routing,algorithm=algorithm)
    # NOTE test if can remove this clipping
    grid.clip_to(c)
    # clipped_catch = grid.view(c)
    # NOTE test with the below
    # catchment_polygon = ops.unary_union(shape(shape) for shape, _ in grid.polygonize())
    catchment_polygon = ops.unary_union([geometry.shape(shape) 
                                     for shape, value in grid.polygonize()])
    # NOTE test if this can be commented out if this clip isn't needed
    grid.viewfinder = dem.viewfinder
    
    row['BasinGeo']  = catchment_polygon
    
    return row

######################################################################
#                      Main Function
######################################################################

def generate_catchments(grid, dem, dem_path, condition, acc_thresh=3000,so_filter_min=1, so_filter_max=4,
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
    step_start = time.time()
    
    if condition == 1:
        print("Pysheds - conditioning DEM")
        pit_filled_dem = grid.fill_pits(dem)
    
        # Fill depressions in DEM
        flooded_dem = grid.fill_depressions(pit_filled_dem)
    
        # Resolve flats in DEM
        inflated_dem = grid.resolve_flats(flooded_dem)

        local_total = (time.time() - step_start)
        print('Runtime ',str(local_total/60),'minutes')
        step_start = time.time()

    # Specify directional mapping
    dirmap = (64, 128, 1, 2, 4, 8, 16, 32)

    print("Pysheds flow direction...")
    # Compute flow directions
    # -------------------------------------
    # fdir = grid.flowdir(dem, dirmap=dirmap)
    if condition == 0:
        fdir = grid.flowdir(dem, dirmap=dirmap)
    if condition == 1:
        fdir = grid.flowdir(inflated_dem, dirmap=dirmap)
    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()

    # make flow accumulation raster
    print('Pysheds flow accumulaton map...')
    acc = grid.accumulation(fdir, dirmap=dirmap)
    # set mask
    acc_mask = (acc_thresh < acc )
    # calculate stream order
    so = grid.stream_order(fdir=fdir,mask=acc_mask)
    # update mask
    mask = acc_mask
    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()
    
    # make river network
    print('Pysheds stream network...')
    branches = grid.extract_river_network(fdir=fdir,mask=mask) # returns geojson
    print(len(branches['features']), "branches generated")

    # Filter stream order before clipping?

    branch_gdf = gpd.GeoDataFrame.from_features(branches,crs='epsg:4326')
    print(branch_gdf.columns)
    branches_all = branch_gdf

    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()

    # generate profiles for each individual segment
    print('Pysheds generating profiles...')
    profiles, connections = grid.extract_profiles(fdir=fdir,mask=mask,include_endpoint=False)
    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()
    
    print('Pysheds calculating attributes...')
    branch_gdf = branch_gdf.apply(lambda x: calc_attributes(x,dem,profiles,so),axis=1)
    print(branch_gdf.columns)
    coords = dem.coords

    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()

    # Filter by stream order
    
    print('Pysheds making connections...')
    profile_list, connection_list = make_connections(profiles, connections)
    
    # filter by stream order
    if so_filter_max:
        branch_gdf = branch_gdf[(branch_gdf['Order']>=so_filter_min) & (branch_gdf['Order']<=so_filter_max)]

    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()

    # create all profile and connection lists w/ pour points
    print('Calculating pour points...')
    # branch_gdf = branch_gdf.apply(lambda x: make_profile(x,so,coords,profile_list,connection_list),axis=1)
    branch_gdf[['LocalPP_X', 'LocalPP_Y']] = branch_gdf.apply(lambda x: make_profile(x, so, coords, profile_list, connection_list), axis=1)

    # branch_drop_cols = ['Profile','Chain'] #,'Profile','Chain']
    # branch_gdf.drop(branch_drop_cols,axis=1,inplace=True)
    
    # branch_gdf_copy = branch_gdf.copy() # unfiltered copy to return at end
    
    # some of these sections will have duplicate orders and pour points - drop them
    unique = branch_gdf[['Order','LocalPP_X','LocalPP_Y']].drop_duplicates()
    
    branch_gdf = branch_gdf.loc[unique.index]

    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()
    

    # Clip branches to IRL region****
    print("Clipping branches to IRL region")
    irl_region = gpd.read_file(r'data-inputs\\IRL-boundary\\IRL-AOI_4326.shp')
    # branches = gpd.GeoDataFrame.from_features(branches,crs='epsg:4326')
    branches = branch_gdf
    # print(type(branches), branches.crs)
    branches = gpd.clip(branches, irl_region)
    print("Post IRL clip")
    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()

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
        
        local_total = (time.time() - step_start)
        print('Runtime ',str(local_total/60),'minutes')
        step_start = time.time()

        branch_gdf = gpd.clip(branch_gdf_i, shoreline_clip)
        print("Post shoreline clip")
        print(len(branch_gdf), "branches remain")

        local_total = (time.time() - step_start)
        print('Runtime ',str(local_total/60),'minutes')
        step_start = time.time()
    else:
        branch_gdf = gpd.GeoDataFrame.from_features(branches,crs='epsg:4326')
    
    print('Pysheds generating',len(branch_gdf),'catchments...')
    basins_gdf = branch_gdf.apply(lambda x: make_basin(x,grid,dem,fdir,routing,algorithm),axis=1)
    print("Pysheds - make basins finished")
    
    local_total = (time.time() - step_start)
    print('Runtime ',str(local_total/60),'minutes')
    step_start = time.time()

    # Set geometry and CRS directly on basins_gdf
    basins_gdf.set_geometry('BasinGeo', inplace=True)
    basins_gdf.set_crs('epsg:4326', inplace=True)
    
    copy_for_area = basins_gdf.copy()
    copy_for_area.to_crs('epsg:6933',inplace=True)
    copy_for_area['AreaSqKm'] = copy_for_area.geometry.area  / 1000000

    basins_gdf['AreaSqKm'] = copy_for_area['AreaSqKm']
    
    basin_drop_cols = ['geometry'] #'LocalPP','Profile','Chain']
    # branch_drop_cols = ['LocalPP','Profile','Chain'] #,'Profile','Chain']
    
    basins_gdf.drop(basin_drop_cols,axis=1,inplace=True)
    # basins_gdf_copy.drop(branch_drop_cols,axis=1,inplace=True)
    
    local_total = (time.time() - local_start)
    
    print('Total runtime is',str(local_total/60),'minutes')
    
    return basins_gdf, branches_all



    
    