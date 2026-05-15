# import packages
import os
import re
import time

import pandas as pd

MAIN = "/Volumes/WD_Windows/MSc_Thesis/"


##################################################################################
# Define settings 
TRAVEL_TIME_METHODS = {"aggregate": "Aggregate travel time savings on network level",
                        "od": "OD demand-weighted travel timae savings"}
travel_time_savings_method = "od" # TODO: od or aggregate


##################################################################################
# Define spatial limits of the research corridor
# The coordinates must end with 000 in order to match the coordinates of the input raster data
e_min, e_max = 2687000, 2708000     # 2688000, 2704000 - 2688000, 2705000
n_min, n_max = 1237000, 1254000     # 1238000, 1252000 - 1237000, 1252000


##################################################################################
# Define Scenario Generation

#
SCENARIO_TYPE = {"GENERATED": "Generated scenarios based on random sampling of the input parameters",
                 "STATIC": "Existing scenarios based on the Swiss population scenario of the BFS"}
scenario_type = "GENERATED"  # TODO: GENERATED or STATIC


# Optional local cap for GENERATED runs (set to None to disable cap)
amount_of_scenarios = 100
generated_representative_scenarios_count = 3
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050


# Optional debug throttles for Phase 6 travel-time computation
# When enabled, these limits are applied in both aggregate and OD modes.
travel_time_debug_enabled = True  # True or False
travel_time_debug_scenarios = ("scenario_76", "scenario_45", "scenario_67")  # None -> auto by scenario_type (STATIC: low/medium/high, GENERATED: scenario_1..N)
aggregate_debug_max_developments = 10  # e.g. 1
aggregate_debug_developments_ids = [2, 103, 469, 895, 249, 662, 201, 689, 775, 28, 750, 789, 27, 25, 334]  # Explicit ID_new list for aggregate debug runs; overrides aggregate_debug_max_developments when set
od_max_developments = 10  # e.g. 1
od_debug_development_ids = [2, 103, 469, 895, 249, 662, 201, 689, 775, 28, 750, 789, 27, 25, 334]  # Explicit ID_new list for OD debug runs; overrides od_max_developments when set

def get_travel_time_debug_scenarios():
    if not travel_time_debug_enabled:
        return None

    if travel_time_debug_scenarios is not None:
        if isinstance(travel_time_debug_scenarios, str):
            return [travel_time_debug_scenarios]
        return list(travel_time_debug_scenarios)

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


##################################################################################
# Define variables for monetisation

discount_rate = 0.02 #SN-641821

# Construction costs -> adjusted for 2023
c_openhighway = 15200 # CHF/m
c_tunnel = 416000 # CHF/m
c_bridge = 63900 # CHF/m
ramp = 102000000 # CHF

# Maintenance costs -> adjusted for 2023
c_structural_maint = 1.2 / 100 # % of cosntruction costs
c_om_openhighway = 89.7 # CHF/m/a
c_om_tunnel = 89.7 # CHF/m/a
c_om_bridge = 368.8 # CHF/m/a
maintenance_duration = 50 # years

# Value of travel time savings (VTTS)
VTTS = 32.2 # CHF/h -> adjusted for 2023 

travel_time_duration = 50 # years

# Noise costs -> adjusted for 2023
noise_distance = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]
noise_values = [7254, 5536, 4055, 2812, 1799, 1019, 467, 130, 20]
noise_duration = 50 # years

# Climate effects -> adjusted for 2023
co2_highway = 2780 # CHF/m/50a
co2_tunnel = 3750 # CHF/m/50a

# Nature and Landscape -> adjusted for 2023
fragmentation = 165.6 # CHF/m2/a
fragmentation_duration = 50 # years
habitat_loss = 33.6 # CHF/m2/a
habitat_loss_duration = 30 # years

# Land reallocation -> adjusted for 2023
forest_reallocation = 0.889 # CHF/m2/a
meadow_reallocation = 0.1014 # CHF/m2/a
reallocation_duration = 50  # years
