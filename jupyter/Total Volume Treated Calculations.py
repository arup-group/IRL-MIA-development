# Standalone Python Script to Calculate Incremental Annual Wet Weather Capture

import math

def calculate_annual_runoff(impervious_area_ac, annual_rainfall_in, runoff_coefficient=1.0):
    """
    Calculate Annual Runoff (MG/yr).

    Parameters:
        impervious_area_ac (float): Impervious Drainage Area in acres.
        annual_rainfall_in (float): Annual Rainfall in inches.
        runoff_coefficient (float): Runoff coefficient (default is 1.0).

    Returns:
        float: Annual runoff in MG/yr.
    """
    # Conversion factor: 1 acre-inch = 0.027154 MG
    return (impervious_area_ac * annual_rainfall_in * runoff_coefficient * 0.027154)

def calculate_incremental_wet_weather_capture(
    impervious_area_ac, annual_rainfall_in, annual_runoff_mgyr, passive_volume_acft, cmac_volume_acft):
    """
    Calculate Incremental Annual Wet Weather Capture (MG/yr).

    Parameters:
        impervious_area_ac (float): Impervious Drainage Area in acres.
        annual_rainfall_in (float): Annual Rainfall in inches.
        annual_runoff_mgyr (float): Annual runoff in MG/yr.
        passive_volume_acft (float): Passive detention volume in acre-feet.
        cmac_volume_acft (float): CMAC control volume in acre-feet.

    Returns:
        float: Incremental annual wet weather capture in MG/yr.
    """
    # Convert storage volume in ac-ft to storage volume in inches per impervious acre
    passive_volume_inIA = (passive_volume_acft / impervious_area_ac) * 12
    cmac_volume_inIA = (cmac_volume_acft / impervious_area_ac) * 12

    # Specify passive detention efficiency constants from Opti model
    A = 25.07
    B = 20.864

    # Specify CMAC detention efficiency constants from Opti regression model
    intercept = 124.24
    precip_coef = 0.3415
    ln_precip = -22.161
    ln_Vol = 27.417

    # Passive stormwater capture efficiency (capped between 5% and 99% efficient)
    passive_efficiency = max(min((A + B * math.log(passive_volume_inIA))/100 , 0.99), 0.05)

    # CMAC stormwater capture efficiency (capped between 5% and 99%)
    cmac_efficiency = max(min((intercept + precip_coef * annual_rainfall_in + ln_precip * math.log(annual_rainfall_in) + ln_Vol * math.log(cmac_volume_inIA))/100,0.99), 0.05)

    # Incremental capture based on CMAC efficiency
    incremental_capture_mgyr = annual_runoff_mgyr * (cmac_efficiency - passive_efficiency)

    return incremental_capture_mgyr


# Inputs
impervious_area_ac = 72.52  # Example: acres
annual_rainfall_in = 44.0  # Example: inches
passive_volume_acft = .6  # Example: acre-feet
cmac_volume_acft = .6  # Example: acre-feet

# Calculate annual runoff (MG/yr)
annual_runoff_mgyr = calculate_annual_runoff(impervious_area_ac, annual_rainfall_in)

# Calculate incremental wet weather capture
incremental_capture = calculate_incremental_wet_weather_capture(
    impervious_area_ac, annual_rainfall_in, annual_runoff_mgyr, passive_volume_acft, cmac_volume_acft
)

# Output results
print(f"Annual Runoff (MG/yr): {annual_runoff_mgyr:.2f}")
print(f"Incremental Wet Weather Capture (MG/yr): {incremental_capture:.2f}")
