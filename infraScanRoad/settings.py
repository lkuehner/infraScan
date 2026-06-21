# import packages
import os
import re
import time

import pandas as pd

# Set the active project root manually for Euler or local execution.
MAIN = "/cluster/home/lkuehner/MSc_Thesis/"
#MAIN = "/Volumes/WD_Windows/MSc_Thesis"

# True downloads fresh OSM data; False only reads the raw OSM cache.
online_access = False
OSM_CACHE_DIR = os.path.join(MAIN, "infraScan", "infraScanRoad", "cache")
CHECKPOINT_DIR = os.path.join(MAIN, "data", "infraScanRoad", "checkpoints")
PLOT_DIRECTORY = os.path.join(MAIN, "plots", "road_standalone")

##################################################################################
# Define settings 
TRAVEL_TIME_METHODS = {"aggregate": "   Aggregate travel time savings on network level"
"                                       including local_accessibility effects"
"                                   !!! tts use non-equivalent network extents"
"                                       local accessibility uses a fixed trip-generation factor (1.14) and is not OD-demand consistent.",
                        "od": "OD demand-weighted travel timae savings"}

travel_time_savings_method = "od" # TODO: od or aggregate (only with STATIC scenarios)


##################################################################################
# Define spatial limits of the research corridor
# The coordinates must end with 000 in order to match the coordinates of the input raster data
e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000


##################################################################################
# Define Scenario Generation

#
SCENARIO_TYPE = {"GENERATED": "Generated scenarios based on random sampling of the input parameters",
                 "STATIC": "Existing scenarios based on the Swiss population scenario of the BFS"
                            "TRAVEL_TIME_METHODS = aggregate only for STATIC scenarios"}
scenario_type = "GENERATED"  # TODO: GENERATED or STATIC

if travel_time_savings_method == "aggregate":
    scenario_type = "STATIC"  # aggregate travel time savings only for STATIC scenarios

# Optional local cap for GENERATED runs (set to None to disable cap)
amount_of_scenarios = 100
generated_representative_scenarios_count = 3
generated_select_representative_scenarios = True
generated_selected_scenarios = None # choose representative scenarios manually (e.g. ["scenario_1", "scenario_50")
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050

# Spatial allocation of generated commune OD demand to road Voronoi catchments.
# Population is scenario-specific; employment uses the current empl23 raster
# because no generated employment scenarios are available.
road_od_blend_pop_rate = 1.0
road_od_blend_empl_rate = 1.0

# Phase 6 terminal output. Enable the first two options when detailed solver
# diagnostics for every status-quo/development scenario are needed.
travel_time_show_solver_output = False
travel_time_show_task_starts = False
travel_time_suppress_known_warnings = True

# Optional debug throttles for Phase 6 travel-time computation
# When enabled, these limits are applied in both aggregate and OD modes.
travel_time_debug_enabled = True  # True or False
aggregate_debug_max_developments = None # e.g. 1
aggregate_debug_developments_ids = None # [2, 103, 469, 895, 249, 662, 201, 689, 775, 28, 750, 789, 27, 25, 334]  # Explicit ID_new list for aggregate debug runs; overrides aggregate_debug_max_developments when set
od_max_developments = None  # e.g. 1
od_debug_development_ids = None #[334, 0, 669, 57,423,673,15,938,767,761] #[254, 109, 28, 267] #[2, 103, 469, 895, 249, 662, 201, 689, 775, 750, 789, 334]  # Explicit ID_new list for OD debug runs; overrides od_max_developments when set

def get_travel_time_debug_scenarios():
    if not travel_time_debug_enabled:
        return None

    if generated_selected_scenarios is not None:
        if isinstance(generated_selected_scenarios, str):
            return [generated_selected_scenarios]
        return list(generated_selected_scenarios)

    if scenario_type == "STATIC":
        return ["low", "medium", "high"]


def get_aggregate_debug_development_ids():
    if aggregate_debug_developments_ids is None:
        return None

    if isinstance(aggregate_debug_developments_ids, (int, str)):
        return [int(aggregate_debug_developments_ids)]

    return [int(x) for x in aggregate_debug_developments_ids]


def get_od_debug_development_ids():
    if od_debug_development_ids is None:
        return None

    if isinstance(od_debug_development_ids, (int, str)):
        return [int(od_debug_development_ids)]

    return [int(x) for x in od_debug_development_ids]


def get_representative_generated_scenarios(
    n_scenarios=None,
    n_representatives=None,
):
    """
    Select representative generated scenarios from the full generated set.
    The helper picks low-, mid-, and high-demand cases (or more evenly spread
    demand-ranked cases) using the exported OD matrices at the valuation year.

    Falls back to evenly spread scenario ids when no OD matrices are available.
    """
    if n_scenarios is None:
        n_scenarios = amount_of_scenarios
    if n_representatives is None:
        n_representatives = generated_representative_scenarios_count

    n_scenarios = max(0, int(n_scenarios))
    n_representatives = max(0, int(n_representatives))
    if n_scenarios == 0 or n_representatives == 0:
        return []
    if n_representatives >= n_scenarios:
        return [f"scenario_{idx}" for idx in range(1, n_scenarios + 1)]

    od_dir = os.path.join("data", "infraScanRoad", "traffic_flow", "od")
    demand_by_scenario = []

    if os.path.isdir(od_dir):
        pattern = re.compile(r"od_matrix_(scenario_\d+)\.csv$")
        for filename in os.listdir(od_dir):
            match = pattern.match(filename)
            if not match:
                continue

            scenario_name = match.group(1)
            path = os.path.join(od_dir, filename)
            try:
                od_df = pd.read_csv(path, index_col=0)
                total_demand = pd.to_numeric(od_df.to_numpy().ravel(), errors="coerce").sum()
                demand_by_scenario.append((scenario_name, float(total_demand)))
            except Exception:
                continue

    if demand_by_scenario:
        demand_by_scenario = sorted(demand_by_scenario, key=lambda item: item[1])
        if len(demand_by_scenario) <= n_representatives:
            return [scenario_name for scenario_name, _ in demand_by_scenario]

        positions = [
            round(idx * (len(demand_by_scenario) - 1) / (n_representatives - 1))
            for idx in range(n_representatives)
        ] if n_representatives > 1 else [len(demand_by_scenario) // 2]
        selected = []
        seen = set()
        for pos in positions:
            scenario_name = demand_by_scenario[pos][0]
            if scenario_name not in seen:
                selected.append(scenario_name)
                seen.add(scenario_name)
        return selected

    if n_representatives == 1:
        positions = [(n_scenarios + 1) // 2]
    else:
        positions = [
            round(1 + idx * (n_scenarios - 1) / (n_representatives - 1))
            for idx in range(n_representatives)
        ]
    positions = sorted(set(int(pos) for pos in positions))
    return [f"scenario_{idx}" for idx in positions]
