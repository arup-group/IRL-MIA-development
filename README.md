# Indian River Lagoon Smart Watershed Project - The Nature Conservancy

## Purpose
Python scripts to delineate "microwatersheds", assess and prioritize them, and return summary information.

## Contents
The primary contents of the repo are jupyter notebooks that run the analysis, and a folder for the data-inputs (DEM, shapefiles)

We've created separate notebooks of similar contents to test different DEMs/areas of interest.

## Setup

### In your terminal (this assumes a bash terminal):

1. **Navigate to the 'jupyter' directory**
    ```bash
    cd jupyter
    ```

2. **Create a virtual environment**
    ```bash
    python -m venv venv
    ```

    We've mainly used Python version 3.9.13, so it might be worth installing and using that specific version in your venv:
    ```bash
    virtualenv --python="<path to your version...>/Python39/python.exe" venv
    ```

3. **Install the requirements**
    ```bash
    pip install -r requirements.txt
    ```

4. **Activate the environment**
    ```bash
    source venv/Scripts/activate
    ```
