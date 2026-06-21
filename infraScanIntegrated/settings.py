"""
infraScanIntegrated settings

Defaults for the integrated orchestrator. These values may be overridden
interactively in main_integrated.py for the current run only.
"""

# --------------------------------------------------------------
# INTEGRATED RUN CONTROL
# --------------------------------------------------------------

RUN_MODE = "integrated"  # "legacy_rail", "legacy_road", "integrated"
INCLUDE_STANDALONE = True

RUN_RAIL = True
RUN_ROAD = True
PLOT_LEGACY_RAIL = True
PLOT_LEGACY_ROAD = True
PLOT_INTEGRATED = True

# Control the use of cached intermediate outputs from the rail and road models
# in infraScanRail.settings the detailed cache settings can be defined
# in infraScanRoad the cache is controlled by checkpoints files under data/infraScanRoad/cache
use_cache_rail = True # False: full rerun of the rail model
use_cache_road_checkpoints = False # False: full rerun of the road model

# Rail standalone prompts are not usable in sbatch (e.g. EULER) integrated runs. These values
# are applied only by infraScanIntegrated/pipeline_integrated.py.
rail_visualization_mode = "all"  # "manual", "none", "all"
rail_grouping_strategy = "baseline"  # "manual", "conservative", "baseline", "optimal"
rail_capacity_threshold = 2.0
rail_max_enhancement_iterations = 3
rail_use_existing_capacity_prep = True
rail_intervention_costs_reviewed = True


# --------------------------------------------------------------
# PHASE 0: INTEGRATED APPRAISALCONFIGURATION 
# --------------------------------------------------------------	

# Shared valuation defaults for integrated scoring and comparable standalone
# outputs produced within the integrated run.
appraisal_years = 40
rail_VTTS = 25.24 # CHF/h NIBA adjusted 2023
road_VTTS = 26.85 # CHF/h EBeN adjusted 2023

# SN-641821
discount_rate = 0.02

#SN 641 822a (NISTRA)
real_wage_growth = 0.0069 

# --------------------------------------------------------------
# PHASE IV: Settings for Scenario Generation
# --------------------------------------------------------------	

# Shared scenario and valuation defaults for integrated runs.
scenario_type = "GENERATED"
amount_of_scenarios = 100
representative_scenarios_count = 10
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050



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

def annualized_growth_rate(start_value: float, target_value: float, start_year: int, target_year: int) -> float:
	horizon = max(1, int(target_year) - int(start_year))
	if start_value <= 0:
		return 0.0
	return (target_value / start_value) ** (1.0 / horizon) - 1.0


rail_modal_split_avg_growth_rate = annualized_growth_rate(
	rail_modal_split_start,
	rail_modal_split_target,
	start_year_scenario,
	start_valuation_year,
)
road_modal_split_avg_growth_rate = annualized_growth_rate(
	road_modal_split_start,
	road_modal_split_target,
	start_year_scenario,
	start_valuation_year,
)
other_modal_split_avg_growth_rate = annualized_growth_rate(
	other_modal_split_start,
	other_modal_split_target,
	start_year_scenario,
	start_valuation_year,
)

# In the logistic-normal setup these per-mode volatility inputs are combined into 
# the latent joint-process volatility. The values mirror
# the calibration idea: sigma grows from 0.015 in 2018 to 0.045
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
