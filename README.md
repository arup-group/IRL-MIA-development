# Indian River Lagoon Smart Watershed Project

## Purpose
The "Microwatershed Impact Assessment" open-source tool, at a high-level, is a geospatial workflow which breaks down larger watersheds into smaller, more manageable ‘microwatersheds’ and allows users to determine locations of hydraulically connected sytems which include ponds that are optimal for smart control.

The key inputs into this process (either read in via API in the scripts, or included in the 'data-inputs' folder)  are a Digital Elevation Model (DEM, up to 10m resolution) that represents the terrain of an area, and a GIS dataset containing the ponds in that area. That area of interest can be up to 2,500 square miles, with a computation time of less than an hour. 

With the inputs, the MIA utilizes the “Pysheds” open-source Python library to pre-process the DEM, delineate flow channels based on a flow accumulation output, and then delineate granular "microwatersheds" based on those flow channels. Additional information and statistics are then automatically calculated attributed to each microwatershed, including the area of the microwatershed, the number of ponds (>1 acre surface area), the total controllable volume of those ponds (linearly interpolated from pond surface area), the annual volume treated by ponds (calculation method provided by Opti), the nitrogen load for a common storm event, and total pondshed area (total area of a microwatershed that flows into ponds). The nitrogen load is based on calculated runoff for each FL DEP Land Use class within a microwatershed, and an Event Mean Concentration value derived from the 2021 FL DEP SWIL model.

## Repo Contents
The primary contents of the repo are jupyter notebooks that run the analysis, and a folder for the data-inputs (shapefile, tif, csv)

The 'jupyter' folder notably contains the 2 key notebooks used to conduct the analysis:
- 1_ConditionDEM_Whitebox.ipynb
- 2_DelineateMicrowatersheds.ipynb

In addition, the 'make_catchments' python script is derived from a Pysheds wrapper (https://github.com/eorland/nested_watersheds) and is utilized in the second notebook.

The relevant subfolders are the 'data-inputs' and 'outputs' folders:
- data-inputs: this folder contains a series of subfolders containing the necessary dataset inputs into the delineation process. 

## Setup

### In your terminal (this assumes a bash terminal):

1. **Create the virtual environments**
    Due to package dependencies, we need to create 2 separate environments, one for the DEM processing, and one for the microwatershed delineation.

    We used Python version 3.9.13, so install that specific version of python, and install it in your venv.
    Let's start with the DEM processing env:
    ```bash
    virtualenv --python="<path to your version...>/Python39/python.exe" venv39_dem
    ```

2. **Activate the environment**
    ```bash
    source venv39_dem/Scripts/activate
    ```

3. **Install the requirements**
    Note: this may take a couple minutes to install all of the packages.

    ```bash
    pip install -r venv39_dem.txt
    ```

4. **Create second venv**
    Now let's create the microwatershed environment:
    ```bash
    deactivate
    virtualenv --python="<path to your version...>/Python39/python.exe" venv39_mws
    ```

5. **Activate the environment**
    ```bash
    source venv39_mws/Scripts/activate
    ```

6. **Install the requirements**
    Note: this may take a couple minutes to install all of the packages.

    ```bash
    pip install -r venv39_mws.txt
    ```

7. **Proceed to the '1_ConditionDEM_Whitebox.ipynb' to begin the analysis!**
    Remember to select the given virtual environment as the notebook kernel to ensure the packages are all installed.

    [DEM Conditioning Notebook](jupyter\1_ConditionDEM_Whitebox.ipynb)