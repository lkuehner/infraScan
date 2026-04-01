
# Define spatial limits of the research corridor
# The coordinates must end with 000 in order to match the coordinates of the input raster data
e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000



# only cross-model settings


scenario_type = "GENERATED"
amount_of_scenarios = 2
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050

discount_rate = 0.03

rail_value_of_time = 14.43
road_value_of_time = 23.29

rail_annualization_factor = 0.1 * 1 * 250
road_annualization_factor = 0.1 * 1 * 250

road_development_sample_size = 1000
