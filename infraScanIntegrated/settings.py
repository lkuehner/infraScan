
# Shared integrated settings
# Keep the literal assumptions here so road/rail-specific modules can consume
# them from a single source of truth.

# Spatial limits of the research corridor.
# Coordinates must end with 000 to match the input raster grid.
e_min, e_max = 2687000, 2708000
n_min, n_max = 1237000, 1254000

# Scenario control
scenario_type = "GENERATED"
amount_of_scenarios = 100
representative_scenarios_count = 3
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050

# Monetisation / cross-model values
discount_rate = 0.03
rail_VTT = 14.43
road_VTT = 23.29
rail_annualization_factor = 0.1 * 1 * 250
road_annualization_factor = 0.1 * 1 * 250
road_development_sample_size = 1000

# Shared literature assumptions for modal behaviour
# Verkehrsperspektiven 2050 start shares
rail_modal_split_start = 0.209
road_modal_split_start = 0.731
other_modal_split_start = 0.06

# Verkehrsperspektiven 2050 anchor values used to extend all three modal split paths
# up to 2100 with a constant annual growth rate.
rail_modal_split_target = 0.243
other_modal_split_target = 0.081
road_modal_split_target = 0.676

# Growth assumptions used for sampling the modal behaviour scenarios.
# Rail follows the original infraScanRail assumption directly. Road and other use
# the same setup, calibrated from the corresponding Verkehrsperspektiven anchor
# values over the 2018-2100 horizon.

def _annualized_growth_rate(start_value: float, target_value: float, start_year: int, target_year: int) -> float:
	horizon = max(1, int(target_year) - int(start_year))
	if start_value <= 0:
		return 0.0
	return (target_value / start_value) ** (1.0 / horizon) - 1.0


rail_modal_split_avg_growth_rate = 0.0045
road_modal_split_avg_growth_rate = _annualized_growth_rate(
	road_modal_split_start,
	road_modal_split_target,
	start_year_scenario,
	end_year_scenario,
)
other_modal_split_avg_growth_rate = _annualized_growth_rate(
	other_modal_split_start,
	other_modal_split_target,
	start_year_scenario,
	end_year_scenario,
)

# These per-mode volatility inputs are still used, but no longer as direct share
# noise like in the rail-only generator. In the current logistic-normal setup
# they are combined into the latent joint-process volatility.
rail_modal_split_start_std_dev = 0.015
rail_modal_split_end_std_dev = 0.045
rail_modal_split_std_dev_shocks = 0.02

road_modal_split_start_std_dev = 0.015
road_modal_split_end_std_dev = 0.015
road_modal_split_std_dev_shocks = 0.01

other_modal_split_start_std_dev = 0.005
other_modal_split_end_std_dev = 0.015
other_modal_split_std_dev_shocks = 0.01

# Parameters used only by the current joint logistic-normal modal split model.
modal_split_latent_correlation = 0.15
modal_split_latent_reversion = 0.04
modal_split_latent_std_scale = 2.5
modal_split_min_share = 0.03
modal_split_warmup_years = 12

# Distance per person assumptions
distance_per_person_start = 39.79
distance_per_person_avg_growth_rate = -0.0027
distance_per_person_start_std_dev = 0.005
distance_per_person_end_std_dev = 0.015
distance_per_person_std_dev_shocks = 0.015