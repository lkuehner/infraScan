
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

# Verkehrsperspektiven 2050 anchor values.
# In the integrated joint model, all three modal drifts are calibrated from the
# 2018 start shares to these 2050 anchors and then extended with the same yearly
# rate up to 2100.
rail_modal_split_target = 0.243
other_modal_split_target = 0.081
road_modal_split_target = 0.676

# Growth assumptions used for sampling the modal behaviour scenarios.
# The symmetric integrated setup derives the mean drift for rail, road and other
# directly from the 2018 start values and the Verkehrsperspektiven-2050 anchors.
# This keeps the three-mode implementation internally consistent.

def _annualized_growth_rate(start_value: float, target_value: float, start_year: int, target_year: int) -> float:
	horizon = max(1, int(target_year) - int(start_year))
	if start_value <= 0:
		return 0.0
	return (target_value / start_value) ** (1.0 / horizon) - 1.0


rail_modal_split_avg_growth_rate = _annualized_growth_rate(
	rail_modal_split_start,
	rail_modal_split_target,
	start_year_scenario,
	start_valuation_year,
)
road_modal_split_avg_growth_rate = _annualized_growth_rate(
	road_modal_split_start,
	road_modal_split_target,
	start_year_scenario,
	start_valuation_year,
)
other_modal_split_avg_growth_rate = _annualized_growth_rate(
	other_modal_split_start,
	other_modal_split_target,
	start_year_scenario,
	start_valuation_year,
)

# These per-mode volatility inputs are still used, but no longer as direct share
# noise like in the rail-only generator. In the current logistic-normal setup
# they are combined into the latent joint-process volatility. The values mirror
# the original rail calibration idea: sigma grows from 0.015 in 2018 to 0.045
# in 2100 and tau is fixed at 0.02, so the Verkehrsperspektiven anchors should
# remain inside the central 90% of the generated sample.
rail_modal_split_start_std_dev = 0.015
rail_modal_split_end_std_dev = 0.045
rail_modal_split_std_dev_shocks = 0.02

road_modal_split_start_std_dev = 0.015
road_modal_split_end_std_dev = 0.045
road_modal_split_std_dev_shocks = 0.02

other_modal_split_start_std_dev = 0.015
other_modal_split_end_std_dev = 0.045
other_modal_split_std_dev_shocks = 0.02

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
