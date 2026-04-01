# import packages
import os
import time


# Define spatial limits of the research corridor
# The coordinates must end with 000 in order to match the coordinates of the input raster data
e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000

##################################################################################
# Define variables for monetisation

# Construction costs
c_openhighway = 15200 # CHF/m
c_tunnel = 416000 # CHF/m
c_bridge = 63900 # CHF/m
ramp = 102000000 # CHF

# Maintenance costs
c_structural_maint = 1.2 / 100 # % of cosntruction costs
c_om_openhighway = 89.7 # CHF/m/a
c_om_tunnel = 89.7 # CHF/m/a
c_om_bridge = 368.8 # CHF/m/a
maintenance_duration = 50 # years

# Value of travel time savings (VTTS)
VTTS = 32.2 # CHF/h
travel_time_duration = 50 # years

# Noise costs
noise_distance = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
noise_values = [7254, 5536, 4055, 2812, 1799, 1019, 467, 130, 20]
noise_duration = 50 # years

# Climate effects
co2_highway = 2780 # CHF/m/50a
co2_tunnel = 3750 # CHF/m/50a

# Nature and Landscape
fragmentation = 165.6 # CHF/m2/a
fragmentation_duration = 50 # years
habitat_loss = 33.6 # CHF/m2/a
habitat_loss_duration = 30 # years

# Land reallocation
forest_reallocation = 0.889 # CHF/m2/a
meadow_reallocation = 0.1014 # CHF/m2/a
reallocation_duration = 50  # years


