"""Build harmonized integrated score tables from road and rail model outputs.

The module reads the exported standalone result files, derives the comparable
integrated score components, and writes consolidated road-rail score tables for
the integrated plotting workflow.

Exports are written to:
  - data/infraScanIntegrated/costs/score_results/
"""


import sqlite3
import warnings
from typing import Iterable, Optional
from pathlib import Path
try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None
import numpy as np
import pandas as pd
from shapely.ops import unary_union
from . import common_cost_parameters

from . import paths as integrated_paths
from ..infraScanRoad import cost_parameters as road_cost_parameters
from ..infraScanRail import cost_parameters as rail_cost_parameters
from ..infraScanRail import paths as rail_paths


RESULT_COLUMNS = [
    "development",
    "scenario",
    "score_id",
    "standalone_value",
    "integrated_value",
]

STANDALONE_ANNUAL_YEARS = common_cost_parameters.appraisal_years
SETTLEMENT_LANDCOVER_SHP = ( Path(rail_paths.MAIN)/ "data"/ "landuse_landcover"/ "landcover"/ "Landcover"/ "swissTLMRegio_LandCover.shp")
NOISE_BUFFER_METERS = 50.0 # SN VSS 41 828, assuming that noise impacts are relevant within a 50m buffer in residential areas


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


def load_settlement_footprint():
    if not SETTLEMENT_LANDCOVER_SHP.exists():
        return None, gpd.GeoDataFrame(columns=["OBJVAL", "geometry"])

    landcover = gpd.read_file(SETTLEMENT_LANDCOVER_SHP)
    objval = landcover["OBJVAL"].fillna("").astype(str).str.lower()
    settlement = landcover[
        objval.str.contains("siedl", regex=False)
        | objval.str.contains("stadtzentr", regex=False)
    ].copy()

    if settlement.empty:
        return None, gpd.GeoDataFrame(columns=["OBJVAL", "geometry"])

    settlement = settlement[~settlement.geometry.is_empty & settlement.geometry.notna()]
    if settlement.empty:
        return None, settlement

    settlement_valid = settlement[settlement.geometry.is_valid].copy()
    if settlement_valid.empty:
        return None, settlement

    return unary_union(settlement_valid.geometry), settlement


def safe_buffer(geometry, buffer_m: float = NOISE_BUFFER_METERS):
    if geometry is None or geometry.is_empty or not geometry.is_valid:
        return None

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            buffered = geometry.buffer(buffer_m)
    except (RuntimeWarning, ValueError):
        return None

    if buffered.is_empty or not buffered.is_valid or buffered.area == 0:
        return None
    return buffered


def compute_settlement_buffer_share(
    geometry,
    settlement_footprint,
    buffer_m: float = NOISE_BUFFER_METERS,
) -> float:
    buffered = safe_buffer(geometry, buffer_m=buffer_m)
    if buffered is None or settlement_footprint is None:
        return 0.0

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            intersection_area = buffered.intersection(settlement_footprint).area
    except (RuntimeWarning, ValueError):
        return 0.0

    return max(0.0, min(1.0, intersection_area / buffered.area))


def load_rail_noise_relevant_share_by_development() -> dict:
    development_dir = Path(rail_paths.MAIN) / rail_paths.DEVELOPMENT_DIRECTORY
    settlement_footprint, _ = load_settlement_footprint()
    if settlement_footprint is None or not development_dir.exists():
        return {}

    noise_share_by_development = {}
    for file_path in sorted(
        path for path in development_dir.glob("*.gpkg")
        if not path.name.startswith("._")
    ):
        dev_gdf = gpd.read_file(file_path)
        new_segments = dev_gdf[dev_gdf.get("new_dev") == "Yes"].copy()
        if not new_segments.empty and "dev_id" in new_segments.columns:
            development = new_segments["dev_id"].iloc[0]
        else:
            development = file_path.stem

        if new_segments.empty:
            noise_share_by_development[development] = 0.0
            continue

        new_segments["segment_length_m"] = new_segments.geometry.length.astype(float)
        new_segments["tunnel_length_m"] = pd.to_numeric(
            new_segments.get("Tunnel m", 0.0),
            errors="coerce",
        ).fillna(0.0)
        new_segments["surface_length_m"] = (
            new_segments["segment_length_m"] - new_segments["tunnel_length_m"]
        ).clip(lower=0.0)
        new_segments["surface_share"] = np.where(
            new_segments["segment_length_m"] > 0,
            new_segments["surface_length_m"] / new_segments["segment_length_m"],
            0.0,
        )
        new_segments["settlement_buffer_share"] = new_segments.geometry.apply(
            lambda geom: compute_settlement_buffer_share(
                geometry=geom,
                settlement_footprint=settlement_footprint,
            )
        )
        new_segments["noise_relevant_share"] = (
            new_segments["surface_share"] * new_segments["settlement_buffer_share"]
        )
        new_segments["Frequency"] = pd.to_numeric(
            new_segments.get("Frequency", 0.0),
            errors="coerce",
        ).fillna(0.0)
        new_segments["segment_weight"] = (
            new_segments["segment_length_m"] * new_segments["Frequency"]
        )

        total_weight = float(new_segments["segment_weight"].sum())
        if total_weight <= 0:
            noise_share_by_development[development] = 0.0
            continue

        noise_share_by_development[development] = float(
            (
                new_segments["segment_weight"]
                * new_segments["noise_relevant_share"]
            ).sum()
            / total_weight
        )

    return noise_share_by_development




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
    - standalone values [CHF/y] are simple annual proxies based on construction cost / appraisal years
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
            "standalone_value": -(row.building_costs / STANDALONE_ANNUAL_YEARS),
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
            "standalone_value": -row.maintenance_annual,
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


def monetize_road_externalities_detail(
    detail_df: pd.DataFrame,
    annualization_factor: float = 2.5 * 250,
) -> pd.DataFrame:
    """
    Monetize road externalities from link-level delta-vkm inputs.
    """
    detail = detail_df.copy()

    # Convert peak-hour delta-vkm to yearly delta-vkm.
    detail["delta_vkm_annualized"] = detail["delta_vkm"] * annualization_factor

    # Noise is only monetized on the residential exposed, non-tunnel share of the link
    detail["delta_vkm_annualized_noise_relevant"] = (
        detail["delta_vkm_annualized"] * detail["noise_relevant_share"]
    )

    # Accidents, air pollution and CO2 are monetized on the full traffic delta.
    detail["road_accident_cost_annual"] = (
        detail["delta_vkm_annualized"] * common_cost_parameters.road_accident_costs
    )
    detail["road_airpollution_cost_annual"] = (
        detail["delta_vkm_annualized"] * common_cost_parameters.road_airpollution_costs
    )
    detail["road_co2_cost_annual"] = (
        detail["delta_vkm_annualized"] * common_cost_parameters.road_co2_costs
    )
    detail["road_noise_cost_annual"] = (
        detail["delta_vkm_annualized_noise_relevant"] * common_cost_parameters.road_noise_costs
    )

    # Land consumption is only assigned to the new above-ground link of each development.
    detail["road_land_consumption_cost"] = (
        (detail["link_role"] == "new_link").astype(float)
        * detail["surface_length_m"]
        * 20.0 # Assumption of 20m width of land consumption for new highway links
        * 5 # VSS 41 828
        / 10_000.0
        * common_cost_parameters.land_consumption_costs
    )

    return detail


def build_road_externalities_result_df(
    detail_df: pd.DataFrame,
    total_costs_df: Optional[pd.DataFrame] = None,
    score_ids: Optional[Iterable[str]] = None,
    annualization_factor: float = 2.5 * 250,
) -> pd.DataFrame:
    """
    Road externalities costs.
    - standalone_value uses simple annual proxies from the spatially explicit road outputs in total_costs_od
    - integrated_value is the simple monetization of the change in veh-km
      with the per-unit externality costs literature (NIBA)
    """
    rows = []
    detail = monetize_road_externalities_detail(
        detail_df=detail_df,
        annualization_factor=annualization_factor,
    )
    grouped = detail.groupby(["development", "scenario"], as_index=False).agg(
        road_accident_cost_annual=("road_accident_cost_annual", "sum"),
        road_airpollution_cost_annual=("road_airpollution_cost_annual", "sum"),
        road_co2_cost_annual=("road_co2_cost_annual", "sum"),
        road_noise_cost_annual=("road_noise_cost_annual", "sum"),
        road_land_consumption_cost=("road_land_consumption_cost", "sum"),
    )

    standalone_columns = {
        "road_climate_cost": "climate_cost",
        "road_land_consumption_cost": "land_realloc",
        "road_ecological_disruption_cost": "nature",
        "road_noise_cost": "noise_s1",
    }
    score_column_map = {
        "road_accident_cost": "road_accident_cost_annual",
        "road_airpollution_cost": "road_airpollution_cost_annual",
        "road_co2_cost": "road_co2_cost_annual",
        "road_noise_cost": "road_noise_cost_annual",
        "road_land_consumption_cost": "road_land_consumption_cost",
        "road_climate_cost": None,
        "road_ecological_disruption_cost": None,
    }
    selected_score_ids = tuple(score_ids) if score_ids is not None else tuple(score_column_map)
    air_pollution_dyn_factor = dynamization_factor(
        growth_rate=common_cost_parameters.real_wage_growth,
        appraisal_years=common_cost_parameters.appraisal_years,
        discount_rate=common_cost_parameters.discount_rate,
    )
    externality_dyn_factors = {
        "road_accident_cost": common_cost_parameters.dyn_accident_costs,
        "road_airpollution_cost": air_pollution_dyn_factor,
        "road_co2_cost": common_cost_parameters.dyn_climate_effects,
        "road_noise_cost": common_cost_parameters.dyn_noise_pollution,
        "road_land_consumption_cost": 1.0,
    }

    standalone_lookup = {}
    if total_costs_df is not None:
        standalone = total_costs_df.copy()
        development_col = "ID_new" if "ID_new" in standalone.columns else "development"
        keep_cols = [development_col] + [col for col in standalone_columns.values() if col in standalone.columns]
        standalone = standalone[keep_cols].drop_duplicates(subset=[development_col])
        standalone_lookup = standalone.set_index(development_col).to_dict("index")

    for row in grouped.itertuples(index=False):
        standalone_values = standalone_lookup.get(row.development, {})
        for score_id in selected_score_ids:
            standalone_value = standalone_values.get(standalone_columns.get(score_id))
            if standalone_value is not None and pd.notna(standalone_value):
                standalone_value = -(abs(float(standalone_value)) / STANDALONE_ANNUAL_YEARS)
            integrated_column = score_column_map[score_id]
            integrated_value = None if integrated_column is None else getattr(row, integrated_column)
            if integrated_value is not None and pd.notna(integrated_value):
                if score_id == "road_land_consumption_cost":
                    integrated_value = abs(float(integrated_value))
                else:
                    integrated_value = float(integrated_value) * externality_dyn_factors.get(score_id, 1.0)
            rows.append({
                "development": row.development,
                "scenario": row.scenario,
                "score_id": score_id,
                "standalone_value": standalone_value,
                "integrated_value": integrated_value,
            })

    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def prepare_road_construction(construction_gpkg: Path) -> pd.DataFrame:
    """Prepare the road construction input table for registry scoring."""
    construction = read_gpkg_table(
        construction_gpkg,
        columns=["ID_new", "cost_path", "cost_bridge", "cost_tunnel", "cost_ramp", "building_costs"],
    )

    for col in ["cost_path", "cost_bridge", "cost_tunnel", "cost_ramp", "building_costs"]:
        if col in construction.columns:
            construction[col] = pd.to_numeric(construction[col], errors="coerce").abs()

    construction["cost_open_highway"] = construction["cost_path"]
    if "cost_ramp" not in construction.columns:
        construction["cost_ramp"] = (
            construction["building_costs"]
            - construction["cost_path"]
            - construction["cost_bridge"]
            - construction["cost_tunnel"]
        ).clip(lower=0.0)
    return construction


def build_road_results(
    road_costs_dir: Path,
    road_externalities_detail_path: Path | None,
) -> pd.DataFrame:
    """Build the long-format road score table from standalone road outputs."""
    construction_gpkg = road_costs_dir / "total_costs_od.gpkg"
    total_costs_csv = road_costs_dir / "total_costs_od.csv"
    tts_csv = road_costs_dir / "traveltime_savings_od_yearly.csv"
    if not tts_csv.exists():
        tts_csv = road_costs_dir / "traveltime_savings_od.csv"
    maintenance_gpkg = road_costs_dir / "maintenance.gpkg"

    if not construction_gpkg.exists():
        raise FileNotFoundError(f"Missing road construction gpkg: {construction_gpkg}")
    if not total_costs_csv.exists():
        raise FileNotFoundError(f"Missing road total costs csv: {total_costs_csv}")
    if not tts_csv.exists():
        raise FileNotFoundError(
            "Missing road travel time savings csv. Expected one of:\n"
            f"- {road_costs_dir / 'traveltime_savings_od_yearly.csv'}\n"
            f"- {road_costs_dir / 'traveltime_savings_od.csv'}"
        )

    construction = prepare_road_construction(construction_gpkg)
    road_tt_wide = pd.read_csv(tts_csv)
    total_costs_df = pd.read_csv(total_costs_csv)
    standalone_externality_cols = ["climate_cost", "land_realloc", "nature", "noise_s1"]
    missing_standalone_cols = [
        col for col in standalone_externality_cols if col not in total_costs_df.columns
    ]
    if missing_standalone_cols:
        gpkg_cols = ["ID_new"] + missing_standalone_cols
        total_costs_gpkg = read_gpkg_table(construction_gpkg, columns=gpkg_cols)
        total_costs_df = total_costs_df.merge(total_costs_gpkg, on="ID_new", how="left")

    if "ID_new" in road_tt_wide.columns and "development" not in road_tt_wide.columns:
        road_tt_wide = road_tt_wide.rename(columns={"ID_new": "development"})

    road_tt_scenarios = {
        col.removeprefix("tt_") for col in road_tt_wide.columns if col.startswith("tt_")
    }
    road_total_scenarios = {
        col.removeprefix("total_") for col in total_costs_df.columns if col.startswith("total_scenario_")
    }
    scenarios = sorted(
        list(road_tt_scenarios & road_total_scenarios),
        key=lambda scenario: int(str(scenario).split("_")[-1]) if str(scenario).split("_")[-1].isdigit() else -1,
    )

    result_frames = [
        build_road_construction_result_df(construction, scenarios),
        build_road_tts_result_df(road_tt_wide, scenarios),
    ]

    if maintenance_gpkg.exists():
        try:
            maintenance_df = read_gpkg_table(
                maintenance_gpkg,
                columns=["ID_new", "maintenance_annual"],
            )
            if "maintenance_annual" in maintenance_df.columns:
                result_frames.append(build_road_maint_result_df(maintenance_df, scenarios))
            else:
                maintenance_df = read_gpkg_table(maintenance_gpkg, columns=["ID_new", "maintenance"])
                if "maintenance" in maintenance_df.columns:
                    maintenance_df["maintenance_annual"] = pd.to_numeric(
                        maintenance_df["maintenance"],
                        errors="coerce",
                    )
                    result_frames.append(
                        build_road_maint_result_df(
                            maintenance_df[["ID_new", "maintenance_annual"]],
                            scenarios,
                        )
                    )
                else:
                    warnings.warn("Road maintenance.gpkg has neither 'maintenance_annual' nor 'maintenance'. Skipping road maintenance.")
        except Exception as exc:
            warnings.warn(f"Road maintenance.gpkg is not readable. Skipping road maintenance. ({exc})")
    else:
        warnings.warn("Road maintenance.gpkg not found. Skipping road maintenance.")

    if road_externalities_detail_path is not None and road_externalities_detail_path.exists():
        detail_df = pd.read_csv(road_externalities_detail_path)
        result_frames.append(
            build_road_externalities_result_df(
                detail_df=detail_df,
                total_costs_df=total_costs_df,
            )
        )
    else:
        warnings.warn("Road externalities detail file not found. Skipping integrated road externalities.")

    road_result = pd.concat(result_frames, ignore_index=True)
    road_result["mode"] = "Road"
    road_result["development"] = (
        road_result["development"]
        .astype(str)
        .str.replace("Development_", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )
    road_result["scenario"] = road_result["scenario"].astype(str)
    return road_result[["mode"] + RESULT_COLUMNS]




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
    - standalone values [CHF/y] are simple annual proxies based on TotalConstructionCost / appraisal years
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
            "standalone_value": row.building_costs / STANDALONE_ANNUAL_YEARS,
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
    old_discounted_tts_df: Optional[pd.DataFrame] = None,
    score_id: str = "rail_tts_cost",
) -> pd.DataFrame:
    """
    Rail TTS costs
    - standalone values [CHF/y] use the old discounted benefit path divided by appraisal years, if available
    - integrated values [CHF/y] are calculated as yearly tts * dynamization_factor
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

    standalone_lookup = {}
    if old_discounted_tts_df is not None and not old_discounted_tts_df.empty:
        old_discounted = old_discounted_tts_df.copy()
        old_discounted["development"] = old_discounted["development"].astype(str)
        old_discounted["scenario"] = old_discounted["scenario"].astype(str)
        standalone_lookup = {
            (str(row.development), str(row.scenario)): float(row.standalone_value)
            for row in old_discounted.itertuples(index=False)
        }

    for row in tts_df.itertuples(index=False):
            standalone_value = standalone_lookup.get(
                (str(row.development), str(row.scenario)),
            )
            rows.append({
                "development": row.development,
                "scenario": row.scenario,
                "score_id": score_id,
                "standalone_value": standalone_value,
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
    land_consumption_by_development = {}
    noise_relevant_share_by_development = load_rail_noise_relevant_share_by_development()

    construction_cost_path = pd.io.common.stringify_path(
        Path(rail_paths.MAIN) / rail_paths.CONSTRUCTION_COSTS
    )
    try:
        construction_df = pd.read_csv(construction_cost_path)
        for cost_row in construction_df.itertuples(index=False):
            land_consumption_cost = 0.0

            # Passing siding / Doppelspurausbau:
            # Prefer exported length. If unavailable, back-calculate from construction costs.
            passing_siding_length_m = getattr(cost_row, "CapInt_PassingSidingLength_m", None)
            if passing_siding_length_m is None:
                passing_siding_length_m = (
                    getattr(cost_row, "CapInt_ConstructionCost", 0.0)
                    / rail_cost_parameters.segment_siding_costs
                    * 1000.0
                    if getattr(cost_row, "CapInt_ConstructionCost", 0.0) > 0
                    else 0.0
                )
            if passing_siding_length_m > 0:
                land_consumption_cost += (
                    passing_siding_length_m
                    * 15.0
                    / 10_000.0
                    * common_cost_parameters.land_consumption_costs
                )

            # New track:
            # Prefer exported length. If unavailable, back-calculate from track construction costs.
            free_track_length_m = getattr(cost_row, "Dev_FreeTrackLength_m", None)
            if free_track_length_m is None:
                free_track_length_m = (
                    getattr(cost_row, "Dev_TrackConstructionCost", 0.0)
                    / rail_cost_parameters.track_cost_per_meter
                    if getattr(cost_row, "Dev_TrackConstructionCost", 0.0) > 0
                    else 0.0
                )
            if free_track_length_m > 0:
                land_consumption_cost += (
                    free_track_length_m
                    * 25.0
                    / 10_000.0
                    * common_cost_parameters.land_consumption_costs
                )

            land_consumption_by_development[getattr(cost_row, "Development")] = land_consumption_cost
    except FileNotFoundError:
        pass

    air_pollution_dyn_factor = dynamization_factor(
        growth_rate=common_cost_parameters.real_wage_growth,
        appraisal_years=common_cost_parameters.appraisal_years,
        discount_rate=common_cost_parameters.discount_rate,
    )

    for row in train_km_df.itertuples(index=False):
        development = row.Development
        delta_btkm = row.DeltaBTKM
        delta_train_km = row.DeltaTrainKM
        noise_relevant_share = noise_relevant_share_by_development.get(development, 1.0)

        score_values = {
            "rail_noise_cost": delta_btkm * noise_relevant_share * common_cost_parameters.rail_noise_costs * common_cost_parameters.dyn_noise_pollution,
            "rail_airpollution_cost": delta_btkm * common_cost_parameters.rail_airpollution_costs * air_pollution_dyn_factor,
            "rail_co2_cost": delta_btkm * common_cost_parameters.rail_co2_costs * common_cost_parameters.dyn_climate_effects,
            "rail_accident_cost": delta_train_km * common_cost_parameters.rail_accident_costs * common_cost_parameters.dyn_accident_costs,
            "rail_land_consumption_cost": land_consumption_by_development.get(development, 0.0),
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


def build_rail_results(
    rail_costs_dir: Path,
    rail_train_km_path: Path | None,
) -> pd.DataFrame:
    """Build the long-format rail score table from standalone rail outputs."""
    construction_csv = rail_costs_dir / "construction_cost.csv"
    tts_csv = rail_costs_dir / "traveltime_savings.csv"
    discounted_tts_path = integrated_paths.RAIL_DISCOUNTED_TTS_CSV

    if not construction_csv.exists():
        raise FileNotFoundError(f"Missing rail construction csv: {construction_csv}")
    if not tts_csv.exists():
        raise FileNotFoundError(f"Missing rail travel time savings csv: {tts_csv}")

    construction_raw = pd.read_csv(construction_csv)
    tts_df = pd.read_csv(tts_csv)
    discounted_tts_df = None
    if discounted_tts_path is not None and discounted_tts_path.exists():
        discounted_raw = pd.read_csv(discounted_tts_path, usecols=["development", "scenario", "benefit"])
        discounted_tts_df = (
            discounted_raw.groupby(["development", "scenario"], as_index=False)
            .agg(standalone_value=("benefit", "sum"))
        )
        discounted_tts_df["standalone_value"] = (
            discounted_tts_df["standalone_value"] / STANDALONE_ANNUAL_YEARS
        )

    construction_registry_input = construction_raw.rename(
        columns={
            "Development": "ID_new",
            "TotalConstructionCost": "building_costs",
        }
    ).copy()
    static_registry_input = construction_raw.rename(columns={"Development": "ID_new"}).copy()

    scenarios = sorted(
        tts_df["scenario"].astype(str).unique().tolist(),
        key=lambda scenario: int(str(scenario).split("_")[-1]) if str(scenario).split("_")[-1].isdigit() else -1,
    )

    result_frames = [
        build_rail_construction_result_df(construction_registry_input, scenarios),
        build_rail_maint_result_df(static_registry_input, scenarios),
        build_rail_operation_result_df(static_registry_input, scenarios),
        build_rail_tts_result_df(
            tts_df.copy(),
            old_discounted_tts_df=discounted_tts_df,
        ),
    ]

    if rail_train_km_path is not None and rail_train_km_path.exists():
        train_km_df = pd.read_csv(rail_train_km_path)
        result_frames.append(
            build_rail_externalities_result_df(
                train_km_df=train_km_df,
                scenarios=scenarios,
            )
        )
    else:
        warnings.warn("Rail train_km.csv not found. Skipping rail externalities.")

    rail_result = pd.concat(result_frames, ignore_index=True)
    rail_result["mode"] = "Rail"
    rail_result["development"] = (
        rail_result["development"]
        .astype(str)
        .str.replace("Development_", "", regex=False)
        .str.replace(r"\.0$", "", regex=True)
    )
    rail_result["scenario"] = rail_result["scenario"].astype(str)
    return rail_result[["mode"] + RESULT_COLUMNS]



# --------------------------------------------------------------------
# Combine and export results
# --------------------------------------------------------------------

def assemble_score_results_long(
    road_result: pd.DataFrame,
    rail_result: pd.DataFrame,
) -> pd.DataFrame:
    """Combine road and rail score rows into one consistently ordered table.

    Args:
        road_result: Road score rows with columns ["mode"] + RESULT_COLUMNS.
        rail_result: Rail score rows with columns ["mode"] + RESULT_COLUMNS.

    Returns:
        One long-format score table sorted by mode, development, scenario and score id.
    """
    score_long = pd.concat([road_result, rail_result], ignore_index=True)

    shared_selection_path = Path(integrated_paths.SHARED_SELECTION_PATH)
    if shared_selection_path.exists():
        selected_scenarios = (
            pd.read_csv(shared_selection_path, usecols=["scenario"])["scenario"]
            .dropna()
            .astype(str)
            .tolist()
        )
        score_long["scenario"] = score_long["scenario"].astype(str)
        score_long = score_long[score_long["scenario"].isin(selected_scenarios)].copy()

    road_externalities_detail = integrated_paths.ROAD_EXTERNALITIES_DETAIL_CSV
    if road_externalities_detail.exists():
        link_flows = pd.read_csv(
            road_externalities_detail,
            usecols=["development", "link_role", "vkm_development"],
        )
        link_flows["development"] = link_flows["development"].astype(str).str.replace(r"\.0$", "", regex=True)
        link_flows["vkm_development"] = pd.to_numeric(link_flows["vkm_development"], errors="coerce").fillna(0.0)
        valid_road_developments = set(
            link_flows.loc[
                (link_flows["link_role"] == "new_link")
                & (link_flows["vkm_development"] > 0),
                "development",
            ]
        )
        normalized_development = score_long["development"].astype(str).str.replace(r"\.0$", "", regex=True)
        score_long = score_long[
            score_long["mode"].ne("Road") | normalized_development.isin(valid_road_developments)
        ].copy()

    score_long["scenario_sort"] = score_long["scenario"].astype(str).str.split("_").str[-1]
    score_long["scenario_sort"] = pd.to_numeric(score_long["scenario_sort"], errors="coerce").fillna(-1)
    score_long = score_long.sort_values(
        ["mode", "development", "scenario_sort", "score_id"]
    ).drop(columns=["scenario_sort"]).reset_index(drop=True)
    return score_long


def build_tidy_score_values(score_long: pd.DataFrame) -> pd.DataFrame:
    """Reshape the long score table into a tidy value table.

    Args:
        score_long: Long-format score table with standalone and integrated values.

    Returns:
        Tidy table with one value per row and a value_mode column.
    """
    tidy = score_long.melt(
        id_vars=["mode", "development", "scenario", "score_id"],
        value_vars=["standalone_value", "integrated_value"],
        var_name="value_mode",
        value_name="value",
    )
    tidy["value_mode"] = tidy["value_mode"].replace(
        {
            "standalone_value": "standalone",
            "integrated_value": "integrated",
        }
    )
    scenario_sort = pd.to_numeric(
        tidy["scenario"].astype(str).str.split("_").str[-1],
        errors="coerce",
    ).fillna(-1)
    tidy = tidy.assign(scenario_sort=scenario_sort).sort_values(
        ["mode", "development", "scenario_sort", "score_id", "value_mode"]
    ).drop(columns=["scenario_sort"]).reset_index(drop=True)
    return tidy


def read_gpkg_table(path: Path, table: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    """Read one geopackage table and optionally keep only selected columns."""
    if gpd is not None:
        gdf = gpd.read_file(path)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        if columns is None:
            return df
        keep_cols = [col for col in columns if col in df.columns]
        return df[keep_cols].copy()

    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        if table is None:
            contents = pd.read_sql("SELECT table_name, data_type FROM gpkg_contents", con)
            table = contents.loc[contents["data_type"].isin(["features", "attributes"]), "table_name"].iloc[0]
        query = f"SELECT * FROM '{table}'" if columns is None else f"SELECT {', '.join(columns)} FROM '{table}'"
        return pd.read_sql(query, con)
    finally:
        con.close()

def export_score_results(output_dir: Path | None = None) -> dict:
    """Build and export the integrated road-rail score tables.

    Args:
        output_dir: Optional custom export directory.
            If None, uses the integrated default score-results directory.

    Returns:
        Dict with the exported DataFrames and their output paths.
    """
    if output_dir is None:
        output_dir = integrated_paths.SCORE_RESULTS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)

    road_result = build_road_results(
        road_costs_dir=integrated_paths.ROAD_COSTS_DIR,
        road_externalities_detail_path=(
            integrated_paths.ROAD_EXTERNALITIES_DETAIL_CSV
            if integrated_paths.ROAD_EXTERNALITIES_DETAIL_CSV.exists()
            else None
        ),
    )
    rail_result = build_rail_results(
        rail_costs_dir=integrated_paths.RAIL_COSTS_DIR,
        rail_train_km_path=(
            integrated_paths.RAIL_TRAIN_KM_CSV
            if integrated_paths.RAIL_TRAIN_KM_CSV.exists()
            else None
        ),
    )

    score_long = assemble_score_results_long(road_result, rail_result)
    score_tidy = build_tidy_score_values(score_long)

    score_long_path = (
        integrated_paths.SCORE_RESULTS_LONG_PATH
        if output_dir == integrated_paths.SCORE_RESULTS_DIR
        else output_dir / "score_results_long.csv"
    )
    score_tidy_path = (
        integrated_paths.SCORE_RESULTS_TIDY_PATH
        if output_dir == integrated_paths.SCORE_RESULTS_DIR
        else output_dir / "score_results_tidy.csv"
    )

    score_long.to_csv(score_long_path, index=False)
    score_tidy.to_csv(score_tidy_path, index=False)

    print(f"Wrote {score_long_path}")
    print(f"Wrote {score_tidy_path}")

    return {
        "score_long": score_long,
        "score_tidy": score_tidy,
        "score_long_path": score_long_path,
        "score_tidy_path": score_tidy_path,
    }
