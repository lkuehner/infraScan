from typing import Iterable, Optional

import pandas as pd

from . import common_cost_parameters
from ..infraScanRoad import cost_parameters as road_cost_parameters
from ..infraScanRail import cost_parameters as rail_cost_parameters


RESULT_COLUMNS = [
    "development",
    "scenario",
    "score_id",
    "standalone_value",
    "integrated_value",
]

def make_result_row(
    development,
    scenario,
    score_id: str,
    standalone_value,
    integrated_value,
) -> dict:
    return {
        "development": development,
        "scenario": scenario,
        "score_id": score_id,
        "standalone_value": standalone_value,
        "integrated_value": integrated_value,
    }


def filter_score_results(
    result_df: pd.DataFrame,
    score_id: Optional[str] = None,
    scenario: Optional[str] = None,
) -> pd.DataFrame:
    filtered = result_df.copy()
    if score_id is not None:
        filtered = filtered[filtered["score_id"] == score_id]
    if scenario is not None:
        filtered = filtered[filtered["scenario"] == scenario]
    return filtered.reset_index(drop=True)

# --------------------------------------------------------------------
# Helper functions for calculating integrated values
# --------------------------------------------------------------------

def capital_recovery_factor(rate: float, lifetime_years: int) -> float:
    growth = (1.0 + rate) ** lifetime_years
    return rate * growth / (growth - 1.0)


def dynamization_factor(growth_rate: float, appraisal_years: int, discount_rate: float) -> float:
    # Calculating for the average year of the appraisal period, i.e. half of the appraisal years (as in NISTRA)
    midpoint_years = appraisal_years / 2.0
    value_increase = (1.0 + growth_rate) ** midpoint_years - 1.0
    net_value_factor = ((1 + growth_rate) ** midpoint_years) / ((1+discount_rate) **  midpoint_years)
    return (value_increase * net_value_factor) + 1.0




# --------------------------------------------------------------------
# ROAD: Calculate integrated values 
# --------------------------------------------------------------------

def build_road_construction_result_df(
    construction_df: pd.DataFrame,
    scenarios: Iterable[str],
    score_id: str = "road_construction_cost",
) -> pd.DataFrame:
    """
    Road construction costs
    - standalone values [CHF] are taken from the 'cost_' column of infraScanRoad/costs/construction.gpkg
    - integrated values [CHF/y] are calculated per construction component using element-specific lifetimes
    """
    rows = []

    crf_open_highway = capital_recovery_factor(
        rate=common_cost_parameters.discount_rate,
        lifetime_years=road_cost_parameters.openhighway_lifetime,
    )
    crf_bridge = capital_recovery_factor(
        rate=common_cost_parameters.discount_rate,
        lifetime_years=road_cost_parameters.bridge_lifetime,
    )
    crf_tunnel = capital_recovery_factor(
        rate=common_cost_parameters.discount_rate,
        lifetime_years=road_cost_parameters.tunnel_lifetime,
    )
    crf_ramp = capital_recovery_factor(
        rate=common_cost_parameters.discount_rate,
        lifetime_years=road_cost_parameters.ramp_lifetime,
    )

    # Construction costs are development-specific, not scenario-specific.
    # Can be repeated for each scenario
    for row in construction_df.itertuples(index=False):
        cost_open_highway = getattr(row, "cost_open_highway", 0.0)
        cost_bridge = getattr(row, "cost_bridge", 0.0)
        cost_tunnel = getattr(row, "cost_tunnel", 0.0)
        cost_ramp = getattr(row, "cost_ramp", 0.0)

        result_row = {
            "development": row.ID_new,
            "score_id": score_id,
            "standalone_value": row.building_costs,
            "integrated_value": (
                cost_open_highway * crf_open_highway
                + cost_bridge * crf_bridge
                + cost_tunnel * crf_tunnel
                + cost_ramp * crf_ramp
            ),
        }

        for scenario in scenarios:
            rows.append({
                **result_row,
                "scenario": scenario,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def build_road_maint_result_df(
    maintenance_df: pd.DataFrame,
    scenarios: Iterable[str],
    score_id: str = "road_maint_cost",
) -> pd.DataFrame:
    """
    Road maintenance costs 
    - standalone values [CHF/y] are taken from the 'maintenance_annual' column of infraScanRoad/costs/maintenance.gpkg
    - integrated values [CHF/y] are calculated with as maintenance * dynamization_factor
    """
    rows = []


    # Calculate dynamization factor for converting maintenance costs to annualized costs
    # Using 1% growth rate for maintenance costs (NISTRA) and 2% discount rate (SN-641821) 
    dyn_factor = dynamization_factor(
        growth_rate=common_cost_parameters.road_maintenance_operating_cost_growth,
        appraisal_years=common_cost_parameters.appraisal_years,
        discount_rate=common_cost_parameters.discount_rate,
    )


    # Maintenance costs are development-specific, not scenario-specific.
    # Can be repeated for each scenario
    for row in maintenance_df.itertuples(index=False):
        result_row = {
            "development": row.ID_new,
            "score_id": score_id,
            "standalone_value": row.maintenance_annual,
            "integrated_value": row.maintenance_annual * dyn_factor,
        }

        for scenario in scenarios:
            rows.append({
                **result_row,
                "scenario": scenario,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)



def build_road_tts_result_df(
    tts_df: pd.DataFrame,
    scenarios: Iterable[str],
    score_id: str = "road_tts_cost",
) -> pd.DataFrame:
    """
    Road TTS costs
    - standalone values [CHF/y] are taken from the 'tts' column of infraScanRoad/costs/traveltime_savings_od.csv
    - integrated values [CHF/y] are calculated with as tts * dynamization_factor
    """
    rows = []


    # Calculate dynamization factor for converting tts costs to annualized costs
    # Using real wage growth rate of 0.69% and 2% discount rate
    dyn_factor = dynamization_factor(
        growth_rate=common_cost_parameters.real_wage_growth,
        appraisal_years=common_cost_parameters.appraisal_years,
        discount_rate=common_cost_parameters.discount_rate,
    )


    # TTS costs are development-specific, but also scenario-specific 
    # (depending on the traffic volumes in each scenario).
    for row in tts_df.itertuples(index=False):
        development = row.development

        for col in tts_df.columns:
            if not col.startswith("tt_"):
                continue

            scenario = col.removeprefix("tt_")
            standalone_value = getattr(row, col)
            integrated_value = standalone_value * dyn_factor

            rows.append({
                "development": development,
                "scenario": scenario,
                "score_id": score_id,
                "standalone_value": standalone_value,
                "integrated_value": integrated_value,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)



# --------------------------------------------------------------------
# RAIL: Calculate integrated values 
# --------------------------------------------------------------------


def build_rail_construction_result_df(
    construction_df: pd.DataFrame,
    scenarios: Iterable[str],
    score_id: str = "rail_construction_cost",
) -> pd.DataFrame:
    rows = []
    """
    Rail construction costs 
    - standalone values [CHF] are taken from the 'TotalConstructionCost' column of infraScanRail/costs/construction_cost.csv
    - integrated values [CHF/y] are calculated by applying the capital recovery factor to the standalone values
    """
    
    # Calculate capital recovery factor for converting construction costs to annualized costs 
    crf_track = capital_recovery_factor(
        rate=common_cost_parameters.discount_rate,
        lifetime_years=rail_cost_parameters.track_lifetime,
    )
    crf_bridge = capital_recovery_factor(
        rate=common_cost_parameters.discount_rate,
        lifetime_years=rail_cost_parameters.bridge_lifetime,
    )
    crf_tunnel = capital_recovery_factor(
        rate=common_cost_parameters.discount_rate,
        lifetime_years=rail_cost_parameters.tunnel_lifetime,
    )

    # Construction costs are development-specific, not scenario-specific.
    # Can be repeated for each scenario
    for row in construction_df.itertuples(index=False):
        cost_track = getattr(row, "TrackConstructionCost", 0.0)
        cost_bridge = getattr(row, "BridgeConstructionCost", 0.0)
        cost_tunnel = getattr(row, "TunnelConstructionCost", 0.0)

        result_row = {
            "development": row.ID_new,
            "score_id": score_id,
            "standalone_value": row.building_costs,
            "integrated_value": (
                cost_track * crf_track
                + cost_bridge * crf_bridge
                + cost_tunnel * crf_tunnel
            ),
        }

        for scenario in scenarios:
            rows.append({
                **result_row,
                "scenario": scenario,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def build_rail_maint_result_df(
    maintenance_df: pd.DataFrame,
    scenarios: Iterable[str],
    score_id: str = "rail_maint_cost",
) -> pd.DataFrame:
    """
    Rail maintenance costs 
    - standalone values [CHF/y] are taken from the 'YearlyMaintenanceCost' column of infraScanRail/costs/construction_cost.csv
    - integrated values [CHF/y] are calculated with as yearly maintenance * dynamization_factor
    """
    rows = []

    dyn_factor = common_cost_parameters.rail_maintenance_cost_growth


    # Maintenance costs are development-specific, not scenario-specific.
    # Can be repeated for each scenario
    for row in maintenance_df.itertuples(index=False):
        result_row = {
            "development": row.ID_new,
            "score_id": score_id,
            "standalone_value": row.YearlyMaintenanceCost,
            "integrated_value": row.YearlyMaintenanceCost * dyn_factor,
        }

        for scenario in scenarios:
            rows.append({
                **result_row,
                "scenario": scenario,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)

def build_rail_operation_result_df(
    maintenance_df: pd.DataFrame,
    scenarios: Iterable[str],
    score_id: str = "rail_operation_cost",
) -> pd.DataFrame:
    """
    Rail operation costs 
    - standalone values [CHF/y] are taken from the 'uncoveredOperatingCost' column of infraScanRail/costs/construction_cost.csv
    - integrated values [CHF/y] are calculated with as yearly maintenance * dynamization_factor
    """
    rows = []

    dyn_factor = common_cost_parameters.rail_operation_cost_growth


    # Operation costs are development-specific, not scenario-specific.
    # Can be repeated for each scenario
    for row in maintenance_df.itertuples(index=False):
        result_row = {
            "development": row.ID_new,
            "score_id": score_id,
            "standalone_value": row.uncoveredOperatingCost,
            "integrated_value": row.uncoveredOperatingCost * dyn_factor,
        }

        for scenario in scenarios:
            rows.append({
                **result_row,
                "scenario": scenario,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def build_rail_tts_result_df(
    tts_df: pd.DataFrame,
    score_id: str = "rail_tts_cost",
) -> pd.DataFrame:
    """
    Rail TTS costs
    - standalone values [CHF/y] are taken from the 'monetized_savings_yearly' column of infraScanRail/costs/traveltime_savings.cs
    - integrated values [CHF/y] are calculated with as tts * dynamization_factor
    """
    rows = []


    # Calculate dynamization factor for converting tts costs to annualized costs
    # Using real wage growth rate of 0.69% and 2% discount rate
    dyn_factor = dynamization_factor(
        growth_rate=common_cost_parameters.real_wage_growth,
        appraisal_years=common_cost_parameters.appraisal_years,
        discount_rate=common_cost_parameters.discount_rate,
    )
    tts_df["year"] = tts_df["year"].astype(int)
    tts_df = tts_df[tts_df["year"] == common_cost_parameters.prognosis_year].copy()

    for row in tts_df.itertuples(index=False):
            rows.append({
                "development": row.development,
                "scenario": row.scenario,
                "score_id": score_id,
                "standalone_value": row.monetized_savings_yearly,
                "integrated_value": row.monetized_savings_yearly * dyn_factor,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)

def build_rail_externalities_result_df(
    train_km_df: pd.DataFrame,
    scenarios: Iterable[str],
) -> pd.DataFrame:
    """
    Rail externalities costs.
    - standalone_value has not been determined 
    - integrated_value is the simple monetization of the change in train-km (or btkm) 
      with the per-unit externality costs literature (NIBA)
    """
    rows = []

    for row in train_km_df.itertuples(index=False):
        development = row.Development
        delta_btkm = row.DeltaBTKM
        delta_train_km = row.DeltaTrainKM

        score_values = {
            "rail_noise_cost": delta_btkm * common_cost_parameters.rail_noise_costs,
            "rail_airpollution_cost": delta_btkm * common_cost_parameters.rail_airpollution_costs,
            "rail_co2_cost": delta_btkm * common_cost_parameters.rail_co2_costs,
            "rail_accident_cost": delta_train_km * common_cost_parameters.rail_accident_costs,
        }

        for scenario in scenarios:
            for score_id, integrated_value in score_values.items():
                rows.append({
                    "development": development,
                    "scenario": scenario,
                    "score_id": score_id,
                    "standalone_value": None,
                    "integrated_value": integrated_value,
                })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)

# --------------------------------------------------------------------
# Quantitative scoring functions (accessibility)
# --------------------------------------------------------------------

# TODO