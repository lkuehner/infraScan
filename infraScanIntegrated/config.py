# import packages


from infraScan.infraScanRail import settings as rail_settings



def configure_rail():
    rail_settings.PIPELINE_CONFIG.visualization_mode = "none"
    rail_settings.PIPELINE_CONFIG.grouping_strategy = "baseline"
    rail_settings.capacity_threshold = 2.0
    rail_settings.max_enhancement_iterations = 10



# Shared integrated settings
# Keep the literal assumptions here so road/rail-specific modules can consume
# them from a single source of truth.

# Spatial limits of the research corridor.
# Coordinates must end with 000 to match the input raster grid.
e_min, e_max = 2687000, 2708000
n_min, n_max = 1237000, 1254000

# Scenario control --> integrated only for generated scenarios possible
scenario_type = "GENERATED"
amount_of_scenarios = 100
representative_scenarios_count = 10
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050

# Monetisation / cross-model values?
road_development_sample_size = 1000