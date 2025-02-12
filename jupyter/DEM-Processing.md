# Digital Elevation Model (DEM) Pre-Processing for Flow Direction Computation

## Overview
This document outlines the pre-processing operations for preparing a Digital Elevation Model (DEM) before computing flow direction and flow accumulation. The focus is on processing an urban DEM in coastal Florida, where flat terrain poses unique challenges. The flow direction function applies the conditioned DEM (`dem_cd`) to the input grid and computes directions as follows:

```python
fdir = grid.flowdir(dem_cd, dirmap=dirmap)
```

where:
- `grid` represents the input DEM
- `dem_cd` is the output of the DEM conditioning steps

The following pre-processing steps can be applied to both `grid` and `dem_cd`, and their sequence affects the final results. The typical workflow is outlined below.

---

## 1. Aggregating the DEM to a Lower Resolution

### **Overview**
Reducing the resolution of the DEM (e.g., from 1m to 8m) to decrease noise and enhance flow path smoothness.

### **Purpose/Intention**
- Reduces granularity and noise in high-resolution DEMs
- Improves catchment delineations by preventing unrealistic micro-depressions
- Enhances computational efficiency

### **Key Parameters**
- **Target resolution:** Typically 8m
- **Aggregation method:** Mean or median

### **Considerations**
- Reducing resolution too much can remove small-scale features essential for hydrological modeling.
- Aggregation may generalize or remove key hydrological features.

### **Risks**
- Loss of fine-scale hydrological details
- Over-smoothing can lead to inaccuracies in flow direction

---

## 2. Smoothing the Input DEM

### **Overview**
Applying a median filter to reduce noise before further processing.

### **Purpose/Intention**
- Reduces small elevation variations that might cause erratic flow paths
- Helps smooth transitions between different terrain features

### **Key Parameters**
- **Filter type:** Median filter
- **Kernel size:** Depends on the desired smoothing effect

### **Considerations**
- Should be applied cautiously to avoid over-smoothing
- May remove important fine-scale features

### **Risks**
- Over-smoothing can obscure critical hydrological features
- Smoothing before burning features may reduce their impact

---

## 3. Burning Flowlines or OSM Street Network into the DEM

### **Overview**
Embedding hydrologically relevant features (e.g., flowlines, street networks) into the DEM to guide flow paths.

### **Purpose/Intention**
- Ensures water follows known pathways (e.g., rivers, streets)
- Improves representation of human-modified landscapes

### **Key Parameters**
- **Burn width:** At least the resolution of the DEM (e.g., 8m for an 8m DEM)
- **Burn depth:** Determines the impact of burned features on flow paths

### **Considerations**
- Whether to burn features **before or after aggregation** (burning before may preserve details, but burning after ensures proper alignment with coarser grid resolution)
- Feature width must be large enough to be meaningful at the chosen DEM resolution

### **Risks**
- Burning too aggressively may overly influence flow patterns
- Insufficient burning can lead to features being ignored in hydrological analysis

---

## 4. Conditioning the DEM (Filling Pits, Depressions, Resolving Flats)

### **Overview**
Correcting surface inconsistencies that may disrupt hydrological flow modeling.

### **Purpose/Intention**
- Removes artificial depressions and flat areas
- Ensures continuous flow paths

### **Key Parameters**
- **Pit-filling algorithm:** Commonly used to fill depressions
- **Flat resolution method:** Used to resolve ambiguous flow directions

### **Considerations**
- Should be applied **after** aggregation and burning to ensure pre-processed features are conditioned properly
- Over-conditioning may create unrealistic flow paths

### **Risks**
- Excessive pit-filling can alter natural depressions
- May create unnatural flow paths in areas that naturally hold water

---

## 5. Smoothing the Conditioned DEM

### **Overview**
Applying a median filter after DEM conditioning to further refine terrain features.

### **Purpose/Intention**
- Further reduces noise after conditioning
- Creates smoother transitions for flow modeling

### **Key Parameters**
- **Filter type:** Median filter
- **Kernel size:** Adjusted to balance smoothing and detail preservation

### **Considerations**
- Should be applied cautiously to avoid diminishing the effects of conditioning
- Helps to remove residual artifacts from conditioning

### **Risks**
- Over-smoothing may counteract previous conditioning efforts
- May unintentionally alter flow paths

---

## 6. Burning Flowlines or OSM into the Conditioned DEM

### **Overview**
Embedding hydrologically relevant features **after** conditioning to enforce desired flow paths.

### **Purpose/Intention**
- Reinforces flow paths after DEM smoothing and conditioning
- Ensures hydrologically significant features are incorporated into the final DEM used for flow direction

### **Key Parameters**
- **Burn width:** At least the resolution of the DEM
- **Burn depth:** Adjusted based on feature prominence

### **Considerations**
- Burning at this stage ensures features are not lost due to prior smoothing or conditioning
- The output is typically very smooth after conditioning, so burning features at this point may introduce critical flow path adjustments

### **Risks**
- Burning after conditioning may disrupt the uniformity of smoothed areas
- Requires careful selection of burn depth to avoid excessive influence

---

## Conclusion
This document outlines the key steps involved in preparing a DEM for flow direction and accumulation computations in an urban coastal environment. The order of operations affects the final results, and careful consideration should be given to:
1. **Resolution aggregation** to balance detail and computational efficiency.
2. **Smoothing** to reduce noise without obscuring important features.
3. **Feature burning** to enhance hydrological realism.
4. **DEM conditioning** to remove artificial depressions and ensure continuous flow paths.
5. **Final smoothing and feature burning** to refine the conditioned DEM before applying the flow direction function.

Further research or expert consultation is recommended to optimize the workflow based on the specific study area and objectives.

