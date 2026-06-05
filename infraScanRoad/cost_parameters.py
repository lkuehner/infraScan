
##################################################################################
# Define variables for monetisation
duration = 50  # years

# Construction costs -> adjusted for 2023 by Kuehner
c_openhighway = 15300 # CHF/m #old: 15200
c_tunnel = 416200 # CHF/m # old: 416000
c_bridge = 63900 # CHF/m
ramp = 102100000 # CHF #old: 102000000

# Maintenance costs -> adjusted for 2023
c_structural_maint = 1.2 / 100 # % of cosntruction costs
c_om_openhighway = 89.7 # CHF/m/a
c_om_tunnel = 89.7 # CHF/m/a
c_om_bridge = 368.8 # CHF/m/a


# Lifetime of Instrastructure (SN 641 820)
openhighway_lifetime = 60 # years
ramp_lifetime = 70 # years
tunnel_lifetime = 50 # years
bridge_lifetime = 50 # years

# Value of travel time savings (VTTS)
VTTS = 31.4 # CHF/h -> adjusted for 2023 (Marggi: 32.2)


# Noise costs -> adjusted for 2023
noise_distance = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
noise_values = [7302, 5573, 4082, 2831, 1811, 1025, 470, 131, 33]
# old: noise_values = [7254, 5536, 4055, 2812, 1799, 1019, 467, 130, 20]

# Climate effects -> adjusted for 2023
co2_highway = 2780 # CHF/m/50a
co2_tunnel = 3750 # CHF/m/50a

# Nature and Landscape -> adjusted for 2023
fragmentation = 163.6 # CHF/m2/a # old: 165..6
habitat_loss = 33.2 # CHF/m2/a # old: 33.6

# Land reallocation -> adjusted for 2023
forest_reallocation = 0.889 # CHF/m2/a
meadow_reallocation = 0.109 # CHF/m2/a # old: 0.1014

