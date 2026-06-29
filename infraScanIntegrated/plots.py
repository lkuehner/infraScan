from pathlib import Path
from typing import Iterable
import os
import pickle
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import geopandas as gpd
import rasterio
from PIL import Image, ImageChops, ImageDraw, ImageFont
from shapely.geometry import LineString, MultiLineString, box

from infraScan.infraScanIntegrated import paths as integrated_paths
from infraScan.infraScanRail import paths as rail_paths
from infraScan.infraScanRail import plot_parameter as rail_plot_parameter
from infraScan.infraScanRail.network_plot import _offset_polyline_uniform
from infraScan.infraScanRoad.externalities_comp import compute_settlement_buffer_share
from infraScan.infraScanIntegrated.scoring_registry import load_settlement_footprint


#TO DO: make automatic based on generated_selected_scenarios
SCENARIOS = ('scenario_10', 'scenario_100', 'scenario_13', 'scenario_16', 'scenario_19', 'scenario_23', 'scenario_24', 'scenario_26', 'scenario_28', 'scenario_30', 'scenario_35', 'scenario_36', 'scenario_40', 'scenario_47', 'scenario_49', 'scenario_7', 'scenario_70', 'scenario_71', 'scenario_79', 'scenario_97')
#("scenario_76", "scenario_45", "scenario_67")

RAIL_COMPARISON_YEAR = 2050

COST_COLORS = {
    "construction": "#263852",
    "maintenance": "#547285",
    "operating": "#B2C2D3",
    "accident": "#532929",
    "air": "#E2D99E",
    "co2": "#BC6348",
    "noise": "#8B3A3A",
    "land": "#9E9E9E",
    "externalities": "#D5A834",
    "tts": "#91B58D",
}

# Centralized paths (adjust these in paths.py)
RAIL_TOTAL_COSTS_CSV = integrated_paths.RAIL_COSTS_DIR / "total_costs.csv"
RAIL_TT_SAVINGS_CSV = integrated_paths.RAIL_COSTS_DIR / "traveltime_savings.csv"
ROAD_TOTAL_COSTS_CSV = integrated_paths.ROAD_COSTS_DIR / "total_costs_od.csv"
ROAD_TT_OD_CSV = integrated_paths.ROAD_COSTS_DIR / "traveltime_savings_od.csv"
ROAD_TT_DETAILED_CSV = integrated_paths.ROAD_DATA_ROOT / "traffic_flow" / "od" / "od_tt_savings_detailed.csv"
ROAD_OD_TT_UNWEIGHTED_CSV = integrated_paths.ROAD_DATA_ROOT / "traffic_flow" / "od" / "developments_od_tt.csv"
ROAD_OD_TT_RAW_BEFORE_WEIGHTING_CSV = integrated_paths.ROAD_DATA_ROOT / "traffic_flow" / "od" / "od_tt_raw_before_weighting.csv"
ROAD_OD_TT_STATUS_QUO_CSV = integrated_paths.ROAD_DATA_ROOT / "traffic_flow" / "od" / "status_quo_od_tt.csv"

RAIL_TRAVELTIME_CACHE = integrated_paths.RAIL_NETWORK_PATH / "travel_time" / "cache" / "od_times.pkl"
RAIL_TRAVELTIME_SAVINGS_DIR = integrated_paths.RAIL_NETWORK_PATH / "travel_time" / "TravelTime_Savings"
ROAD_TRAVELTIME_RASTER = integrated_paths.ROAD_NETWORK_PATH / "travel_time" / "travel_time_raster.tif"
DEV_DIR = integrated_paths.DATA_ROOT / rail_paths.DEVELOPMENT_DIRECTORY
ANALYSIS_DIR = integrated_paths.INTEGRATED_COSTS_DIR / "score_analysis"
SCORE_RESULTS_DIR = integrated_paths.SCORE_RESULTS_DIR
GENERATED_PLOTS_DIR = integrated_paths.GENERATED_PLOTS_DIR
ROAD_EXTERNALITY_DETAIL_CSV = integrated_paths.ROAD_EXTERNALITIES_DETAIL_CSV
RAIL_TOTAL_COSTS_WITH_GEOMETRY_GPKG = integrated_paths.RAIL_COSTS_DIR / "total_costs_with_geometry.gpkg"
ROAD_NEW_LINKS_REALISTIC_GPKG = integrated_paths.ROAD_NETWORK_PROCESSED_DIR / "new_links_realistic.gpkg"
ROAD_CONSTRUCTION_GPKG = integrated_paths.ROAD_COSTS_DIR / "construction.gpkg"
ROAD_GENERATED_NODES_GPKG = integrated_paths.ROAD_NETWORK_PROCESSED_DIR / "generated_nodes.gpkg"
ROAD_CORRIDOR_POINTS_GPKG = integrated_paths.ROAD_NETWORK_PROCESSED_DIR / "points_corridor_attribute.gpkg"
RAIL_SPLIT_S_BAHN_LINES_GPKG = integrated_paths.RAIL_NETWORK_PATH / "processed" / "split_s_bahn_lines.gpkg"
RAIL_UPDATED_NEW_LINKS_GPKG = integrated_paths.RAIL_NETWORK_PATH / "processed" / "updated_new_links.gpkg"
RAIL_NEW_RAILWAY_LINES_GPKG = integrated_paths.RAIL_NETWORK_PATH / "processed" / "new_railway_lines.gpkg"
OUTERBOUNDARY_SHP = integrated_paths.DATA_ROOT / "data" / "_basic_data" / "outerboundary.shp"
LAKE_DATA_ZH_GPKG = integrated_paths.DATA_ROOT / "data" / "landuse_landcover" / "processed" / "lake_data_zh.gpkg"
CITIES_SHP = integrated_paths.DATA_ROOT / "data" / "manually_gathered_data" / "cities.shp"
RAIL_OVERVIEW_LINE_OFFSET_M = 85.0

MODE_CONFIG = {
    "Rail": {"base_vtt": 25.24},
    "Road": {"base_vtt": 26.85},
}

VTT_SOURCE_PAIRS = pd.DataFrame(
    [
        {"source": "Schmid et al. (2021)", "rail_vtt":
         15.20, "road_vtt": 31.40},
        {"source": "VSS-Norm by guidelines", 
         "rail_vtt": 25.24, "road_vtt": 26.85},
        {"source": "VSS-Norm by distance", 
         "rail_vtt": 24.25, "road_vtt": 38.68},
        {"source": "Average VSS-Norm", 
         "rail_vtt": 16.63, "road_vtt": 26.85},
    ]
)
VTT_SOURCE_PAIRS["label"] = VTT_SOURCE_PAIRS.apply(
    lambda row: f"{row['source']}: Rail {row['rail_vtt']:.2f} / Road {row['road_vtt']:.2f}",
    axis=1,
)


def _offset_line_geometry(geometry, offset_m: float):
    if geometry is None or geometry.is_empty or np.isclose(offset_m, 0.0):
        return geometry
    if isinstance(geometry, LineString):
        coords = _offset_polyline_uniform(list(geometry.coords), offset_m)
        return LineString(coords) if len(coords) >= 2 else geometry
    if isinstance(geometry, MultiLineString):
        shifted_parts = []
        for part in geometry.geoms:
            coords = _offset_polyline_uniform(list(part.coords), offset_m)
            shifted_parts.append(LineString(coords) if len(coords) >= 2 else part)
        return MultiLineString(shifted_parts)
    return geometry


def _shade_mode_groups(ax, modes: list[str] | pd.Series) -> None:
    mode_list = [str(mode) for mode in list(modes)]
    if not mode_list:
        return

    start = 0
    groups = []
    for idx in range(1, len(mode_list) + 1):
        if idx == len(mode_list) or mode_list[idx] != mode_list[start]:
            groups.append((mode_list[start], start, idx - 1))
            start = idx

    background = {"Rail": "#F0F0F0", "Road": "#FFFFFF"}
    for mode, first, last in groups:
        ax.axvspan(first - 0.5, last + 0.5, color=background.get(mode, "#F7F7F7"), zorder=0)
        ax.text(
            (first + last) / 2.0,
            1.02,
            mode,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            color="0.35",
            zorder=1,
            clip_on=False,
        )





# ------------------------------------------
# Data loading and transformation functions
# ------------------------------------------

def load_rail_final_costs_from_sources(scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    df = pd.read_csv(RAIL_TOTAL_COSTS_CSV)

    sline = df["Sline"].astype(str)
    dev_id = df["development"].astype(str).str.removeprefix("Development_").astype(int)
    line_name = np.where(dev_id < 101000, (dev_id - 100000).astype(str) + "_" + sline, "X" + (dev_id - 101000).astype(str))

    rows = []
    for scenario in scenarios:
        suffix = scenario.split("_")[-1]
        rows.append(
            pd.DataFrame(
                {
                    "mode": "Rail",
                    "development": dev_id.astype(str),
                    "line_name": line_name,
                    "scenario": scenario,
                    "net_benefit_mio_chf": df[f"Net Benefit Scenario {suffix} [in Mio. CHF]"],
                    "monetized_savings_mio_chf": df[f"Monetized Savings Scenario {suffix} [in Mio. CHF]"],
                    "construction_cost_mio_chf": df["Construction Cost [in Mio. CHF]"],
                    "maintenance_cost_mio_chf": df["Maintenance Costs [in Mio. CHF]"],
                    "uncovered_operating_cost_mio_chf": df["Uncovered Operating Costs [in Mio. CHF]"],
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def load_road_final_costs_from_sources(scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    totals_path = pd.read_csv(ROAD_TOTAL_COSTS_CSV)
    tt_path = pd.read_csv(ROAD_TT_OD_CSV)
    construction = gpd.read_file(integrated_paths.ROAD_COSTS_DIR / "construction.gpkg")[["ID_new", "building_costs"]]
    maintenance = gpd.read_file(integrated_paths.ROAD_COSTS_DIR / "maintenance.gpkg")[["ID_new", "maintenance"]]
    externalities = gpd.read_file(integrated_paths.ROAD_COSTS_DIR / "externalities.gpkg")[["ID_new", "climate_cost", "land_realloc", "nature"]]
    noise = gpd.read_file(integrated_paths.ROAD_COSTS_DIR / "noise.gpkg")[["ID_new", "noise_s1"]]

    totals_df = (
        construction.merge(maintenance, on="ID_new", how="left")
        .merge(externalities, on="ID_new", how="left").merge(noise, on="ID_new", how="left")
    )

    totals_df["externalities_chf"] = totals_df["climate_cost"] + totals_df["land_realloc"] + totals_df["nature"] + totals_df["noise_s1"]

    rows = []
    for scenario in SCENARIOS:
        scenario_df = (
            totals_path[["ID_new", f"total_{scenario}"]]
            .merge(tt_path[["development", f"tt_{scenario}"]], left_on="ID_new", right_on="development", how="left")
            .merge(totals_df[["ID_new", "building_costs", "maintenance", "externalities_chf"]], on="ID_new", how="left")
        )
        scenario_df["mode"] = "Road"
        # Normalize numeric IDs (remove trailing .0 if present) and convert to string
        scenario_df["development"] = scenario_df["ID_new"].astype(str).str.replace(r"\.0$", "", regex=True)
        scenario_df["line_name"] = scenario_df["development"]
        scenario_df["scenario"] = scenario
        scenario_df["net_benefit_mio_chf"] = scenario_df[f"total_{scenario}"] / 1_000_000
        scenario_df["monetized_savings_mio_chf"] = scenario_df[f"tt_{scenario}"] / 1_000_000
        scenario_df["construction_cost_mio_chf"] = scenario_df["building_costs"] / 1_000_000
        scenario_df["maintenance_cost_mio_chf"] = scenario_df["maintenance"] / 1_000_000
        scenario_df["other_cost_mio_chf"] = scenario_df["externalities_chf"] / 1_000_000
        rows.append(
            scenario_df[
                [
                    "mode",
                    "development",
                    "line_name",
                    "scenario",
                    "net_benefit_mio_chf",
                    "monetized_savings_mio_chf",
                    "construction_cost_mio_chf",
                    "maintenance_cost_mio_chf",
                    "other_cost_mio_chf",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)

def create_combined_cost_csv() -> Path:
    combined_cost_csv = GENERATED_PLOTS_DIR / "rail_road_final_costs_total.csv"
    if combined_cost_csv.exists():
        return combined_cost_csv
    combined = pd.concat([load_rail_final_costs_from_sources(), load_road_final_costs_from_sources()], ignore_index=True, sort=False)
    combined.to_csv(combined_cost_csv, index=False)
    return combined_cost_csv


def create_rail_dev_id_lookup_table() -> pd.DataFrame:
    dev_dir = DEV_DIR
    dev_ids = sorted(
        str(int(float(os.path.splitext(path.name)[0])))
        for path in dev_dir.iterdir()
        if path.is_file() and not path.name.startswith("._")
    )
    return pd.DataFrame({"dev_id": dev_ids}, index=range(1, len(dev_ids) + 1))


def load_rail_weighted_tts(rail_top: list[str] | None = None, annualize: bool = True) -> pd.DataFrame:
    """Load rail demand-weighted travel-time savings before monetization."""
    if not RAIL_TT_SAVINGS_CSV.exists():
        return pd.DataFrame(columns=["mode", "development", "scenario", "weighted_tts_person_h_per_year"])

    selected_scenarios = _selected_scenarios("Rail")
    df = pd.read_csv(RAIL_TT_SAVINGS_CSV)
    df["development"] = df["development"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["scenario"] = df["scenario"].astype(str)
    df = df[
        df["scenario"].isin(selected_scenarios)
        & (pd.to_numeric(df["year"], errors="coerce") == RAIL_COMPARISON_YEAR)
    ].copy()
    if rail_top is not None:
        df = df[df["development"].isin([str(dev) for dev in rail_top])].copy()
    values = pd.to_numeric(df["tt_savings_daily"], errors="coerce")
    df["weighted_tts_person_h_per_year"] = values * 365.0 if annualize else values
    df = df[np.isfinite(df["weighted_tts_person_h_per_year"])].copy()
    df["mode"] = "Rail"
    return df[["mode", "development", "scenario", "weighted_tts_person_h_per_year"]]


def load_road_weighted_tts(road_top: list[str] | None = None, annualize: bool = True) -> pd.DataFrame:
    """Load road demand-weighted travel-time savings before monetization."""
    df = pd.read_csv(ROAD_TT_DETAILED_CSV)
    selected_scenarios = _selected_scenarios("Road")

    df["development"] = df["development"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["scenario"] = df["scenario"].astype(str)
    df = df[df["scenario"].isin(selected_scenarios)].copy()
    if road_top is not None:
        df = df[df["development"].isin([str(dev) for dev in road_top])].copy()
    values = pd.to_numeric(df["tt_savings_peak"], errors="coerce") / 60.0 * 2.5
    df["weighted_tts_person_h_per_year"] = values * 250.0 if annualize else values
    df = df[np.isfinite(df["weighted_tts_person_h_per_year"])].copy()
    df["mode"] = "Road"
    return df[["mode", "development", "scenario", "weighted_tts_person_h_per_year"]]


def build_weighted_tts_by_scenario(
    rail_top: list[str] | None = None,
    road_top: list[str] | None = None,
    annualize: bool = True,
) -> pd.DataFrame:
    """Build the common demand-weighted TTS table used by TTS plots."""
    rail = load_rail_weighted_tts(rail_top, annualize=annualize)
    road = load_road_weighted_tts(road_top, annualize=annualize)
    frames = [frame for frame in [rail, road] if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["mode", "development", "scenario", "weighted_tts_person_h_per_year"])
    tts = pd.concat(frames, ignore_index=True, sort=False)
    if ROAD_EXTERNALITY_DETAIL_CSV.exists():
        link_flows = pd.read_csv(
            ROAD_EXTERNALITY_DETAIL_CSV,
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
        tts = tts[
            tts["mode"].ne("Road")
            | tts["development"].astype(str).isin(valid_road_developments)
        ].copy()
    return tts



def combined_order(data: pd.DataFrame) -> list[str]:
    """Order developments by mean TTS, rail first then road."""
    rail_order = (
        data[data["mode"] == "Rail"]
        .groupby("development")["weighted_tts_person_h_per_year"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    road_order = (
        data[data["mode"] == "Road"]
        .groupby("development")["weighted_tts_person_h_per_year"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return [f"Rail {dev}" for dev in rail_order] + [f"Road {dev}" for dev in road_order]


def _style_axes(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)


def _load_component_overview(mode: str) -> pd.DataFrame:
    score_df = pd.read_csv(SCORE_RESULTS_DIR / "score_results_long.csv")
    score_df = score_df[score_df["mode"] == mode].copy()
    score_df["development"] = score_df["development"].astype(str)

    integrated = score_df[["development", "score_id", "integrated_value"]].rename(
        columns={"integrated_value": "value_chf"}
    )
    integrated["value_mode"] = "integrated"

    standalone = score_df[["development", "score_id", "standalone_value"]].rename(
        columns={"standalone_value": "value_chf"}
    )
    standalone["value_mode"] = "standalone_annual_proxy"

    df = pd.concat([integrated, standalone], ignore_index=True)
    df["value_chf"] = pd.to_numeric(df["value_chf"], errors="coerce")
    df = df.dropna(subset=["value_chf"])
    df["value_mio_chf"] = df["value_chf"] / 1_000_000.0
    df = (
        df.groupby(["development", "score_id", "value_mode"], as_index=False)
        .agg(value_mio_chf=("value_mio_chf", "mean"))
    )

    if mode == "Rail":
        rail_labels = _rail_label_lookup()
        df["development_label"] = df["development"].map(rail_labels).fillna(df["development"])
    else:
        df["development_label"] = df["development"]

    tts_score = "rail_tts_cost" if mode == "Rail" else "road_tts_cost"
    order = (
        df[(df["value_mode"] == "integrated") & (df["score_id"] == tts_score)]
        .sort_values("value_mio_chf", ascending=False)["development"]
        .drop_duplicates()
        .tolist()
    )
    df["plot_order"] = df["development"].map({dev: idx for idx, dev in enumerate(order)})
    return df


def _selected_scenarios(mode: str | None = None) -> list[str]:
    score_df = pd.read_csv(SCORE_RESULTS_DIR / "score_results_long.csv", usecols=["mode", "scenario"])
    if mode is not None:
        score_df = score_df[score_df["mode"] == mode].copy()
    scenarios = score_df["scenario"].astype(str).drop_duplicates().tolist()
    return sorted(scenarios, key=lambda value: int(value.split("_")[-1]))


def _build_integrated_bcr_top10_by_mode_plot_data(top_n_per_mode: int = 10) -> pd.DataFrame:
    rows = []
    for mode in ["Rail", "Road"]:
        component_df = _load_component_overview(mode)
        integrated = component_df[component_df["value_mode"] == "integrated"].copy()
        tts_score = "rail_tts_cost" if mode == "Rail" else "road_tts_cost"
        integrated["is_tts"] = integrated["score_id"].eq(tts_score)
        bcr = (
            integrated.assign(
                cost_mio=np.where(integrated["is_tts"], 0.0, integrated["value_mio_chf"].abs()),
                tts_mio=np.where(integrated["is_tts"], integrated["value_mio_chf"], 0.0),
            )
            .groupby(["development", "development_label"], as_index=False)
            .agg(cost_mio=("cost_mio", "sum"), tts_mio=("tts_mio", "sum"))
        )
        bcr = bcr[bcr["cost_mio"] > 0].copy()
        bcr["bcr_mean"] = bcr["tts_mio"] / bcr["cost_mio"]
        bcr["net_benefit_mean"] = bcr["tts_mio"] - bcr["cost_mio"]
        selected = bcr.sort_values("net_benefit_mean", ascending=False).head(top_n_per_mode).copy()
        selected["mode"] = mode
        selected["ranking_label_short"] = selected["development_label"]
        selected["plot_order"] = np.arange(len(selected))

        plot_df = integrated.drop(columns=["plot_order"], errors="ignore").merge(
            selected[["mode", "development", "ranking_label_short", "bcr_mean", "net_benefit_mean", "plot_order"]],
            on="development",
            how="inner",
        )
        plot_df["mode"] = mode
        plot_df["value_mio_chf"] = np.where(
            plot_df["is_tts"],
            plot_df["value_mio_chf"],
            -plot_df["value_mio_chf"].abs(),
        )
        rows.append(
            plot_df[["mode", "development", "ranking_label_short", "bcr_mean", "net_benefit_mean", "plot_order", "score_id", "value_mio_chf"]]
        )

    plot_df = pd.concat(rows, ignore_index=True)
    mode_offset = plot_df["mode"].map({"Rail": 0, "Road": top_n_per_mode}).fillna(0)
    plot_df["plot_order"] = plot_df["plot_order"] + mode_offset
    return plot_df


def _build_tts_summary_by_development() -> pd.DataFrame:
    tts = build_weighted_tts_by_scenario()
    return (
        tts.groupby(["mode", "development"], as_index=False)
        .agg(
            mean_tts_person_h_per_year=("weighted_tts_person_h_per_year", "mean"),
            std_tts_person_h_per_year=("weighted_tts_person_h_per_year", "std"),
        )
    )


def _rail_label_lookup() -> dict[str, str]:
    df = pd.read_csv(
        integrated_paths.DATA_ROOT / "data" / "infraScanRail" / "costs" / "total_costs.csv",
        usecols=["development", "Sline"],
    ).drop_duplicates()
    dev = df["development"].astype(str).str.removeprefix("Development_").str.replace(r"\.0$", "", regex=True)
    sline = df["Sline"].astype(str)
    dev_num = dev.astype(int)
    label = np.where(
        sline.isin(["G", "P"]),
        (dev_num - 100000).astype(str) + "_" + sline,
        sline,
    )
    return dict(zip(dev.astype(str), label.astype(str)))


def _component_color(score_id: str) -> str:
    score_id = str(score_id)
    if "construction" in score_id:
        return COST_COLORS["construction"]
    if "maint" in score_id:
        return COST_COLORS["maintenance"]
    if "operation" in score_id:
        return COST_COLORS["operating"]
    if "accident" in score_id:
        return COST_COLORS["accident"]
    if "airpollution" in score_id:
        return COST_COLORS["air"]
    if score_id.endswith("co2_cost"):
        return COST_COLORS["co2"]
    if "noise" in score_id:
        return COST_COLORS["noise"]
    if "land_consumption" in score_id or "ecological" in score_id:
        return COST_COLORS["land"]
    if "climate" in score_id:
        return COST_COLORS["co2"]
    if score_id.endswith("tts_cost"):
        return COST_COLORS["tts"]
    return "#777777"


def _component_label(score_id: str) -> str:
    label_map = {
        "construction_cost": "Construction costs",
        "maint_cost": "Maintenance costs",
        "operation_cost": "Operating costs",
        "accident_cost": "Accident costs",
        "airpollution_cost": "Air pollution costs",
        "co2_cost": "CO2 costs",
        "noise_cost": "Noise costs",
        "land_consumption_cost": "Land consumption costs",
        "climate_cost": "CO2 costs",
        "ecological_disruption_cost": "Land consumption costs",
        "tts_cost": "Travel time savings",
    }
    for suffix, label in label_map.items():
        if score_id.endswith(suffix):
            return label
    return score_id


def _top_ranked_developments(mode: str, top_n: int = 5) -> pd.DataFrame:
    plot_df = _build_integrated_bcr_top10_by_mode_plot_data(top_n)
    top_mode = (
        plot_df[plot_df["mode"] == mode][
            ["development", "ranking_label_short", "bcr_mean", "net_benefit_mean", "plot_order"]
        ]
        .drop_duplicates()
        .sort_values(["plot_order", "net_benefit_mean"], ascending=[True, False])
        .head(top_n)
        .reset_index(drop=True)
    )
    top_mode["development"] = top_mode["development"].astype(str)
    top_mode["rank"] = np.arange(1, len(top_mode) + 1)
    return top_mode


def _normalize_development_id(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.removeprefix("Development_")
        .str.replace(r"\.0$", "", regex=True)
    )


def _development_color_palette() -> list[str]:
    return [
        color for color in rail_plot_parameter.zvv_colors
        if color.lower() not in {"#e94b4b", "#555555"}
    ]


def _development_color_lookup(top_rail: pd.DataFrame, top_road: pd.DataFrame) -> dict[tuple[str, str], str]:
    palette = _development_color_palette()
    rail_colors = palette[:len(top_rail)]
    road_colors = palette[len(top_rail):len(top_rail) + len(top_road)]
    lookup = {}
    for idx, row in top_rail.reset_index(drop=True).iterrows():
        lookup[("Rail", str(row["development"]))] = rail_colors[idx % len(rail_colors)]
    for idx, row in top_road.reset_index(drop=True).iterrows():
        lookup[("Road", str(row["development"]))] = road_colors[idx % len(road_colors)]
    return lookup


def _scenario_bcr_net_benefit(mode_filter: str | None = None) -> pd.DataFrame:
    score_df = pd.read_csv(SCORE_RESULTS_DIR / "score_results_long.csv")
    score_df["development"] = score_df["development"].astype(str).str.replace(r"\.0$", "", regex=True)
    if mode_filter is not None:
        score_df = score_df[score_df["mode"] == mode_filter].copy()
    score_df["value_mio_chf"] = pd.to_numeric(score_df["integrated_value"], errors="coerce") / 1_000_000.0
    score_df = score_df.dropna(subset=["value_mio_chf"]).copy()
    tts_scores = {"Rail": "rail_tts_cost", "Road": "road_tts_cost"}
    score_df["is_tts"] = score_df.apply(lambda row: row["score_id"] == tts_scores[row["mode"]], axis=1)
    out = (
        score_df.assign(
            cost_mio=np.where(score_df["is_tts"], 0.0, score_df["value_mio_chf"].abs()),
            tts_mio=np.where(score_df["is_tts"], score_df["value_mio_chf"], 0.0),
        )
        .groupby(["mode", "development", "scenario"], as_index=False)
        .agg(cost_mio=("cost_mio", "sum"), tts_mio=("tts_mio", "sum"))
    )
    out = out[out["cost_mio"] > 0].copy()
    out["bcr"] = out["tts_mio"] / out["cost_mio"]
    out["net_benefit_mio_chf"] = out["tts_mio"] - out["cost_mio"]
    return out


def _add_north_arrow(ax, x: float, y: float, size: float) -> None:
    arrow = mpatches.Polygon(
        [
            (x, y + size * 0.50),
            (x - size * 0.20, y - size * 0.20),
            (x, y - size * 0.05),
            (x + size * 0.20, y - size * 0.20),
        ],
        closed=True,
        facecolor="black",
        edgecolor="black",
        linewidth=0.8,
        zorder=20,
    )
    ax.add_patch(arrow)
    ax.text(x, y - size * 0.42, "N", ha="center", va="center", fontsize=12, zorder=20)


def _add_scale_bar(ax, x: float, y: float, length_m: float, label: str) -> None:
    tick_height = length_m * 0.06
    ax.plot([x, x + length_m], [y, y], color="black", linewidth=2.2, zorder=20)
    for tick_x in [x, x + length_m / 2, x + length_m]:
        ax.plot(
            [tick_x, tick_x],
            [y - tick_height / 2, y + tick_height / 2],
            color="black",
            linewidth=1.8,
            zorder=20,
        )
    ax.text(x + length_m + tick_height * 0.7, y, label, ha="left", va="center", fontsize=10, zorder=20)


def plot_top5_rail_highway_overview(output_path: Path, top_n: int = 5) -> None:
    top_rail = _top_ranked_developments("Rail", top_n)
    top_road = _top_ranked_developments("Road", top_n)

    rail_developments = gpd.read_file(RAIL_TOTAL_COSTS_WITH_GEOMETRY_GPKG).copy()
    rail_developments["development"] = _normalize_development_id(rail_developments["development"])
    rail_developments = rail_developments.merge(
        top_rail[["development", "ranking_label_short", "rank", "bcr_mean"]],
        on="development",
        how="inner",
    )
    rail_developments = rail_developments.to_crs("EPSG:2056")
    rail_path_lines = gpd.GeoDataFrame(columns=["name", "path", "geometry"], crs=rail_developments.crs)
    if RAIL_NEW_RAILWAY_LINES_GPKG.exists():
        rail_path_lines = gpd.read_file(RAIL_NEW_RAILWAY_LINES_GPKG).to_crs(rail_developments.crs)
        if {"name", "geometry"}.issubset(rail_path_lines.columns):
            generated_line_geometries = rail_path_lines.set_index(rail_path_lines["name"].astype(str))["geometry"].to_dict()
            rail_developments["geometry"] = rail_developments.apply(
                lambda row: generated_line_geometries.get(str(row["ranking_label_short"]), row.geometry),
                axis=1,
            )
            rail_developments = gpd.GeoDataFrame(rail_developments, geometry="geometry", crs=rail_developments.crs)

    road_developments = gpd.read_file(ROAD_CONSTRUCTION_GPKG).copy()
    road_developments["development"] = road_developments["ID_new"].astype(str)
    road_developments = road_developments.merge(
        top_road[["development", "ranking_label_short", "rank", "bcr_mean"]],
        on="development",
        how="inner",
    )
    road_developments = road_developments.to_crs("EPSG:2056")

    road_development_points = gpd.read_file(ROAD_GENERATED_NODES_GPKG).copy()
    road_development_points["development"] = road_development_points["ID_new"].astype(str)
    road_development_points = road_development_points.merge(
        top_road[["development", "ranking_label_short", "rank", "bcr_mean"]],
        on="development",
        how="inner",
    )
    road_development_points = road_development_points.to_crs("EPSG:2056")

    rail_network = gpd.read_file(RAIL_SPLIT_S_BAHN_LINES_GPKG).to_crs("EPSG:2056")
    highway_network = gpd.read_file(integrated_paths.ROAD_HIGHWAY_NETWORK_GPKG).to_crs("EPSG:2056")
    rail_stations = gpd.read_file(integrated_paths.RAIL_NETWORK_PATH / "processed" / "points.gpkg").to_crs("EPSG:2056")

    rail_station_ids = set()
    top_rail_ids = pd.to_numeric(top_rail["development"], errors="coerce").dropna().astype(int).tolist()
    if RAIL_UPDATED_NEW_LINKS_GPKG.exists() and top_rail_ids:
        updated_new_links = gpd.read_file(RAIL_UPDATED_NEW_LINKS_GPKG)
        updated_new_links = updated_new_links[
            pd.to_numeric(updated_new_links["dev_id"], errors="coerce").isin(top_rail_ids)
        ].copy()
        for id_col in ["from_ID_new", "to_ID"]:
            if id_col in updated_new_links.columns:
                rail_station_ids.update(
                    pd.to_numeric(updated_new_links[id_col], errors="coerce")
                    .dropna()
                    .astype(int)
                    .tolist()
                )

    top_rail_line_names = set(top_rail["ranking_label_short"].astype(str))
    if not rail_path_lines.empty and top_rail_line_names:
        if {"name", "path"}.issubset(rail_path_lines.columns):
            selected_rail_path_lines = rail_path_lines[
                rail_path_lines["name"].astype(str).isin(top_rail_line_names)
            ].copy()
            for path_value in selected_rail_path_lines["path"].dropna().astype(str):
                rail_station_ids.update(
                    int(node_id)
                    for node_id in path_value.split(",")
                    if node_id.strip().isdigit()
                )

    station_names = [
        "Aathal", "Wetzikon", "Uster", "Schwerzenbach", "Rüti ZH", "Pfäffikon ZH",
        "Nänikon-Greifensee", "Kempten", "Illnau", "Hinwil", "Fehraltorf", "Effretikon",
        "Dübendorf", "Dietlikon", "Bubikon", "Saland", "Bauma", "Esslingen", "Forch",
        "Männedorf", "Küsnacht ZH", "Glattbrugg", "Kloten", "Kemptthal", "Zürich Rehalp",
        "Herrliberg-Feldmeilen", "Horgen", "Thalwil", "Wila", "Schwerzenbach",
    ]
    station_mask = rail_stations["NAME"].isin(station_names)
    if rail_station_ids and "ID_point" in rail_stations.columns:
        station_mask = station_mask | pd.to_numeric(rail_stations["ID_point"], errors="coerce").isin(rail_station_ids)
    rail_stations = rail_stations[station_mask].copy()
    highway_access = gpd.read_file(ROAD_CORRIDOR_POINTS_GPKG).to_crs("EPSG:2056")
    if "intersection" in highway_access.columns:
        highway_access = highway_access[pd.to_numeric(highway_access["intersection"], errors="coerce") == 0].copy()

    boundary = gpd.read_file(OUTERBOUNDARY_SHP).to_crs("EPSG:2056")
    focus_layers = [rail_developments, road_developments, road_development_points]
    bounds = np.array([layer.total_bounds for layer in focus_layers if not layer.empty])
    xmin, ymin = bounds[:, 0].min(), bounds[:, 1].min()
    xmax, ymax = bounds[:, 2].max(), bounds[:, 3].max()
    x_pad_west = max((xmax - xmin) * 0.05, 2200)
    y_pad_south = max((ymax - ymin) * 0.05, 2200)
    x_pad_east = max((xmax - xmin) * 0.14, 5500)
    y_pad_north = max((ymax - ymin) * 0.14, 5500)
    clip_bounds = (xmin - x_pad_west, ymin - y_pad_south, xmax + x_pad_east, ymax + y_pad_north)
    x_shift = max((clip_bounds[2] - clip_bounds[0]) * 0.035, 1500)
    clip_bounds = (
        clip_bounds[0] + x_shift,
        clip_bounds[1],
        clip_bounds[2] + x_shift,
        clip_bounds[3],
    )
    clip_geom = box(*clip_bounds)

    lakes = None
    if LAKE_DATA_ZH_GPKG.exists():
        lakes = gpd.read_file(LAKE_DATA_ZH_GPKG).to_crs("EPSG:2056")
        if "GEWAESSERN" in lakes.columns:
            lakes = lakes[lakes["GEWAESSERN"].isin(["Zürichsee", "Greifensee", "Pfäffikersee"])].copy()
    city_path = CITIES_SHP
    if not city_path.exists():
        city_path = CITIES_SHP.with_name("Cities.shp")
    cities = None
    if city_path.exists():
        cities = gpd.read_file(city_path).to_crs("EPSG:2056")

    highway_network = gpd.clip(highway_network, clip_geom)
    rail_network = gpd.clip(rail_network, clip_geom)
    road_developments = gpd.clip(road_developments, clip_geom)
    road_development_points = gpd.clip(road_development_points, clip_geom)
    rail_developments = gpd.clip(rail_developments, clip_geom)
    rail_stations = gpd.clip(rail_stations, clip_geom)
    highway_access = gpd.clip(highway_access, clip_geom)
    if lakes is not None and not lakes.empty:
        lakes = gpd.clip(lakes, clip_geom)
    if cities is not None and not cities.empty:
        cities = gpd.clip(cities, clip_geom)

    fig, ax = plt.subplots(figsize=(13, 11), dpi=250)

    if lakes is not None and not lakes.empty:
        lakes.plot(
            ax=ax,
            color="#A9D3E9",
            edgecolor="white",
            linewidth=0.7,
            zorder=1,
        )

    if not rail_network.empty:
        rail_network.plot(ax=ax, color="#ABABAB", linewidth=1.8, alpha=1, zorder=2)

    if not highway_network.empty:
        highway_network.plot(ax=ax, color="#000000", linewidth=1.6, alpha=1.0, zorder=3)

    if not rail_stations.empty:
        rail_stations.plot(
            ax=ax,
            color="#ABABAB",
            markersize=15,
            zorder=4,
        )

    if not highway_access.empty:
        highway_access.plot(
            ax=ax,
            color="#000000",
            markersize=15,
            zorder=5,
            alpha=0.8
        )

    development_color_lookup = _development_color_lookup(top_rail, top_road)
    development_legend_handles = []

    rail_developments_ranked = rail_developments.sort_values("rank").copy()
    rail_count = len(rail_developments_ranked)
    for rail_idx, (_, row) in enumerate(rail_developments_ranked.iterrows()):
        color = development_color_lookup[("Rail", str(row["development"]))]
        offset_m = (rail_idx - (rail_count - 1) / 2.0) * RAIL_OVERVIEW_LINE_OFFSET_M
        rail_geometry = _offset_line_geometry(row.geometry, offset_m)
        gpd.GeoSeries([rail_geometry], crs=rail_developments.crs).plot(
            ax=ax,
            color=color,
            linewidth=3.2,
            zorder=7,
        )
        development_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=3.6,
                label=f"Rail ID: {row['ranking_label_short']}",
            )
        )

    for idx, row in road_developments.sort_values("rank").iterrows():
        color = development_color_lookup[("Road", str(row["development"]))]
        gpd.GeoSeries([row.geometry], crs=road_developments.crs).plot(
            ax=ax,
            color=color,
            linewidth=2.3,
            linestyle="-",
            zorder=6,
        )
        development_legend_handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                lw=2.8,
                linestyle="-",
                label=f"Road ID: {row['ranking_label_short']}",
            )
        )

    for idx, row in road_development_points.sort_values("rank").iterrows():
        color = development_color_lookup[("Road", str(row["development"]))]
        gpd.GeoSeries([row.geometry], crs=road_development_points.crs).plot(
            ax=ax,
            color=color,
            linewidth=1.5,
            markersize=15,
            zorder=8,
        )

    if cities is not None and not cities.empty:
        if "location" in cities.columns:
            for idx, row in cities.iterrows():
                label = str(row["location"])
                xytext = (0, -8)
                ha = "center"
                va = "top"
                if label == "Hinwil":
                    xytext = (7, 0)
                    ha = "left"
                    va = "center"
                elif label in {"Dübendorf", "Uster"}:
                    xytext = (0, -13)
                ax.annotate(
                    label,
                    xy=row.geometry.coords[0],
                    ha=ha,
                    va=va,
                    xytext=xytext,
                    textcoords="offset points",
                    fontsize=12,
                    zorder=10,
                )

    context_legend_handles = [
        Line2D([0], [0], color="#000000", lw=2.6, label="Highway network"),
        Line2D([0], [0], color="#ABABAB", lw=2.6, label="Rail network"),
        Line2D([0], [0], marker="o", color="#ABABAB", markerfacecolor="#ABABAB", markersize=4, linewidth=0, label="Rail stations"),
        Line2D([0], [0], marker="o", color="#000000", markerfacecolor="#000000", markersize=4, linewidth=0, label="Highway access points"),
    ]
    legend_handles = development_legend_handles + context_legend_handles

    ax.set_xlim(clip_bounds[0], clip_bounds[2])
    ax.set_ylim(clip_bounds[1], clip_bounds[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("0.7")
        spine.set_linewidth(0.8)
        spine.set_linestyle("--")
    scale_length_m = 5000
    _add_scale_bar(
        ax,
        x=clip_bounds[0] + scale_length_m * 0.55,
        y=clip_bounds[1] + scale_length_m * 0.22,
        length_m=scale_length_m,
        label="5 km",
    )
    _add_north_arrow(
        ax,
        x=clip_bounds[0] + scale_length_m * 0.32,
        y=clip_bounds[1] + scale_length_m * 0.32,
        size=scale_length_m * 0.40,
    )

    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0, 0.83, 1])
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_mode_standalone_vs_integrated(mode: str, output_path: Path, max_developments: int | None = None) -> None:
    component_df = _load_component_overview(mode)
    rail_labels = _rail_label_lookup() if mode == "Rail" else {}
    tts_score = "rail_tts_cost" if mode == "Rail" else "road_tts_cost"
    if "plot_order" in component_df.columns and component_df["plot_order"].notna().any():
        order = (
            component_df[
                (component_df["value_mode"] == "integrated")
                & (component_df["score_id"] == tts_score)
            ][["development", "plot_order"]]
            .drop_duplicates()
            .sort_values("plot_order")["development"]
            .astype(str)
            .tolist()
        )
    else:
        order = (
            component_df[
                (component_df["value_mode"] == "integrated")
                & (component_df["score_id"] == tts_score)
            ]
            .sort_values("value_mio_chf", ascending=False)["development"]
            .astype(str)
            .tolist()
        )
    if max_developments is not None:
        order = order[:max_developments]

    pivot = component_df.pivot_table(
        index=["development", "development_label", "value_mode"],
        columns="score_id",
        values="value_mio_chf",
        aggfunc="mean",
    ).reset_index()
    pivot["development"] = pd.Categorical(pivot["development"], categories=order, ordered=True)
    pivot = pivot.sort_values(["development", "value_mode"])

    x = np.arange(len(order))
    width = 0.36
    offsets = {"standalone_annual_proxy": -width / 2, "integrated": width / 2}

    if mode == "Rail":
        component_order = [
            "rail_construction_cost",
            "rail_maint_cost",
            "rail_operation_cost",
            "rail_accident_cost",
            "rail_airpollution_cost",
            "rail_co2_cost",
            "rail_noise_cost",
            "rail_land_consumption_cost",
            "rail_tts_cost",
        ]
    else:
        component_order = [
            "road_construction_cost",
            "road_maint_cost",
            "road_accident_cost",
            "road_airpollution_cost",
            "road_co2_cost",
            "road_noise_cost",
            "road_land_consumption_cost",
            "road_climate_cost",
            "road_ecological_disruption_cost",
            "road_tts_cost",
        ]

    fig, ax = plt.subplots(figsize=(max(16, len(order) * 0.28), 8))
    for value_mode, offset in offsets.items():
        subset = pivot[pivot["value_mode"] == value_mode].copy()
        subset = subset.groupby("development", as_index=False).mean(numeric_only=True)
        subset = subset.set_index("development").reindex(order).reset_index()
        alpha = 0.45 if value_mode == "standalone_annual_proxy" else 0.90
        negative_bottom = np.zeros(len(order))
        positive_bottom = np.zeros(len(order))
        for score_id in component_order:
            if score_id not in subset.columns:
                continue
            values = subset[score_id].fillna(0.0).to_numpy()
            if score_id == tts_score:
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    bottom=positive_bottom,
                    color=_component_color(score_id),
                    edgecolor="white",
                    linewidth=0.2,
                    alpha=alpha,
                    label=_component_label(score_id) if value_mode == "integrated" else None,
                )
                positive_bottom += values
            else:
                plot_values = -np.abs(values)
                hatch = None
                if (
                    mode == "Road"
                    and value_mode == "standalone_annual_proxy"
                    and score_id == "road_noise_cost"
                ):
                    hatch = None #"//////"
                ax.bar(
                    x + offset,
                    plot_values,
                    width=width,
                    bottom=negative_bottom,
                    color=_component_color(score_id),
                    edgecolor="white",
                    linewidth=0.2,
                    hatch=hatch,
                    alpha=alpha,
                    label=_component_label(score_id) if value_mode == "integrated" else None,
                )
                negative_bottom += plot_values

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    labels = (
        pivot[["development", "development_label"]]
        .drop_duplicates()
        .groupby("development", as_index=False)
        .first()
        .set_index("development")
        .reindex(order)["development_label"]
        .fillna(pd.Series(order, index=order))
        .tolist()
    )
    if mode == "Rail":
        labels = [rail_labels.get(dev, lbl) for dev, lbl in zip(order, labels)]
    ax.set_xticklabels(labels, rotation=90, fontsize=8 if mode == "Rail" else 7)
    ax.set_ylabel("Annual value [Mio. CHF/year]")
    ax.set_xlabel("Development")
    ax.set_title(f"{mode}: integrated vs standalone annualized stacked costs and TTS")
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend(
        [h for h, _ in filtered],
        [l for _, l in filtered],
        frameon=False,
        bbox_to_anchor=(1.01, 1),
        loc="upper left",
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_integrated_bcr_top10_by_mode(
    output_path: Path,
    top_n_per_mode: int = 5,
    font_size: int = 9,
) -> None:
    plot_df = _build_integrated_bcr_top10_by_mode_plot_data(top_n_per_mode)
    plot_df["development"] = plot_df["development"].astype(str)
    rail_labels = _rail_label_lookup()
    plot_df["ranking_label_short"] = np.where(
        plot_df["mode"].eq("Rail"),
        plot_df["development"].map(rail_labels).fillna(plot_df["ranking_label_short"]),
        plot_df["ranking_label_short"],
    )
    development_color_lookup = _development_color_lookup(
        _top_ranked_developments("Rail", top_n_per_mode),
        _top_ranked_developments("Road", top_n_per_mode),
    )
    ordered_selection = (
        plot_df[["mode", "development", "plot_order", "bcr_mean", "net_benefit_mean"]]
        .drop_duplicates()
        .sort_values(["mode", "net_benefit_mean", "plot_order"], ascending=[True, False, True])
        .groupby("mode", group_keys=False)
        .head(top_n_per_mode)
        .copy()
    )
    ordered_selection = ordered_selection.sort_values(["net_benefit_mean", "plot_order"], ascending=[False, True])
    ordered_selection["plot_order"] = np.arange(len(ordered_selection))
    plot_df = (
        plot_df.merge(
            ordered_selection[["mode", "development", "plot_order"]],
            on=["mode", "development"],
            how="inner",
            suffixes=("", "_new"),
        )
        .drop(columns="plot_order")
        .rename(columns={"plot_order_new": "plot_order"})
        .sort_values("plot_order")
    )

    component_order = [
        "rail_construction_cost", "rail_maint_cost", "rail_operation_cost",
        "rail_accident_cost", "rail_airpollution_cost", "rail_co2_cost",
        "rail_noise_cost", "rail_land_consumption_cost", "rail_tts_cost",
        "road_construction_cost", "road_maint_cost",
        "road_accident_cost", "road_airpollution_cost", "road_co2_cost",
        "road_climate_cost", "road_noise_cost", "road_land_consumption_cost",
        "road_ecological_disruption_cost", "road_tts_cost",
    ]
    pivot = plot_df.pivot_table(
        index=["plot_order", "mode", "development", "ranking_label_short", "bcr_mean"],
        columns="score_id",
        values="value_mio_chf",
        aggfunc="mean",
    ).reset_index().sort_values("plot_order").reset_index(drop=True)

    x = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    _shade_mode_groups(ax, pivot["mode"])
    neg = np.zeros(len(pivot))
    pos = np.zeros(len(pivot))
    for score_id in component_order:
        if score_id not in pivot.columns:
            continue
        values = pivot[score_id].fillna(0.0).to_numpy()
        if score_id.endswith("tts_cost"):
            tts_colors = [
                development_color_lookup.get((str(row["mode"]), str(row["development"])), COST_COLORS["tts"])
                for _, row in pivot.iterrows()
            ]
            ax.bar(
                x,
                values,
                width=0.65,
                bottom=pos,
                color=tts_colors,
                edgecolor="none",
                linewidth=0,
                label=None,
            )
            ax.bar(
                x,
                values,
                width=0.65,
                bottom=pos,
                color="none",
                edgecolor="white",
                linewidth=0.4,
                hatch="///",
                label=None,
            )
            pos += values
        else:
            plot_values = -np.abs(values)
            ax.bar(
                x,
                plot_values,
                width=0.65,
                bottom=neg,
                color=_component_color(score_id),
                edgecolor="white",
                linewidth=0.2,
                label=_component_label(score_id) if score_id.startswith("rail_") else None,
            )
            neg += plot_values

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["ranking_label_short"], rotation=90, fontsize=font_size + 5)
    ax.tick_params(axis="y", labelsize=font_size + 5)
    ax.set_xlabel("Development ID", fontsize=font_size + 7)
    ax.set_ylabel("Annual value [Mio. CHF/year]", fontsize=font_size + 7)

    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    filtered.append(
        (
            mpatches.Patch(facecolor="#777777", edgecolor="white", hatch="///", label="Travel time savings"),
            "Travel time savings",
        )
    )
    ax.legend(
        [h for h, _ in filtered],
        [l for _, l in filtered],
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=4,
        fontsize=font_size + 2,
    )
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.92)
    fig.savefig(output_path, dpi=200, transparent=True)
    plt.close(fig)


def build_vtt_ratio_plot_df() -> pd.DataFrame:
    component_frames = []
    for mode in ["Rail", "Road"]:
        component_df = _load_component_overview(mode)
        integrated = component_df[component_df["value_mode"] == "integrated"].copy()
        pivot = (
            integrated.pivot_table(
                index="development",
                columns="score_id",
                values="value_mio_chf",
                aggfunc="mean",
            )
            .reset_index()
        )
        tts_score = "rail_tts_cost" if mode == "Rail" else "road_tts_cost"
        cost_cols = [col for col in pivot.columns if col.endswith("_cost") and col != tts_score]
        pivot["tts_integrated"] = pivot[tts_score].fillna(0.0)
        pivot["cost_base_integrated"] = pivot[cost_cols].fillna(0.0).abs().sum(axis=1)
        pivot["mode"] = mode
        component_frames.append(
            pivot[["mode", "development", "tts_integrated", "cost_base_integrated"]]
        )

    analysis_df = pd.concat(component_frames, ignore_index=True)

    ratio_rows = []
    for row in VTT_SOURCE_PAIRS.itertuples(index=False):
        rail_base = analysis_df[analysis_df["mode"] == "Rail"].copy()
        road_base = analysis_df[analysis_df["mode"] == "Road"].copy()

        rail_base["adjusted_tts"] = rail_base["tts_integrated"] * (row.rail_vtt / MODE_CONFIG["Rail"]["base_vtt"])
        rail_base["adjusted_ratio"] = rail_base["adjusted_tts"] / rail_base["cost_base_integrated"]
        rail_base["source"] = row.source
        rail_base["label"] = row.label

        road_base["adjusted_tts"] = road_base["tts_integrated"] * (row.road_vtt / MODE_CONFIG["Road"]["base_vtt"])
        road_base["adjusted_ratio"] = road_base["adjusted_tts"] / road_base["cost_base_integrated"]
        road_base["source"] = row.source
        road_base["label"] = row.label

        ratio_rows.extend([rail_base, road_base])

    ratio_plot_df = pd.concat(ratio_rows, ignore_index=True)
    ratio_plot_df["label"] = pd.Categorical(
        ratio_plot_df["label"],
        categories=VTT_SOURCE_PAIRS["label"].tolist(),
        ordered=True,
    )
    return ratio_plot_df


def build_vtt_ratio_mean_scenario_ratio_plot_df() -> pd.DataFrame:
    annual = pd.read_csv(ANALYSIS_DIR / "annual_overview_by_development_scenario.csv")
    integrated = annual[annual["value_mode"] == "integrated"].copy()
    integrated["cost_annual"] = pd.to_numeric(integrated["cost_annual"], errors="coerce")
    integrated["tts_annual"] = pd.to_numeric(integrated["tts_annual"], errors="coerce")
    integrated = integrated[integrated["cost_annual"] > 0].copy()

    ratio_rows = []
    for row in VTT_SOURCE_PAIRS.itertuples(index=False):
        pair_df = integrated.copy()
        pair_df["adjusted_tts"] = np.where(
            pair_df["mode"].eq("Rail"),
            pair_df["tts_annual"] * (row.rail_vtt / MODE_CONFIG["Rail"]["base_vtt"]),
            pair_df["tts_annual"] * (row.road_vtt / MODE_CONFIG["Road"]["base_vtt"]),
        )
        pair_df["adjusted_ratio"] = pair_df["adjusted_tts"] / pair_df["cost_annual"]

        development_mean = (
            pair_df.groupby(["mode", "development"], as_index=False)
            .agg(adjusted_ratio=("adjusted_ratio", "mean"))
            .assign(source=row.source, label=row.label)
        )
        ratio_rows.append(development_mean)

    ratio_plot_df = pd.concat(ratio_rows, ignore_index=True)
    ratio_plot_df["label"] = pd.Categorical(
        ratio_plot_df["label"],
        categories=VTT_SOURCE_PAIRS["label"].tolist(),
        ordered=True,
    )
    return ratio_plot_df


def build_externality_normalized_df() -> pd.DataFrame:
    score_df = pd.read_csv(SCORE_RESULTS_DIR / "score_results_long.csv")
    score_df["development"] = score_df["development"].astype(str)
    score_df["integrated_value"] = pd.to_numeric(score_df["integrated_value"], errors="coerce")

    road_ext_scores = [
        "road_accident_cost",
        "road_airpollution_cost",
        "road_co2_cost",
        "road_noise_cost",
        "road_land_consumption_cost",
    ]
    rail_ext_scores = [
        "rail_accident_cost",
        "rail_airpollution_cost",
        "rail_co2_cost",
        "rail_noise_cost",
        "rail_land_consumption_cost",
    ]

    ext_df = score_df[
        (
            (score_df["mode"] == "Road") & score_df["score_id"].isin(road_ext_scores)
        )
        | (
            (score_df["mode"] == "Rail") & score_df["score_id"].isin(rail_ext_scores)
        )
    ].copy()

    summary = (
        ext_df.groupby(["mode", "development", "score_id"], as_index=False)
        .agg(mean_value_chf=("integrated_value", "mean"))
    )
    pivot = summary.pivot_table(
        index=["mode", "development"],
        columns="score_id",
        values="mean_value_chf",
        aggfunc="mean",
    ).reset_index()

    pivot["total_externalities_chf"] = 0.0
    pivot["noise_externality_chf"] = np.nan

    for col in road_ext_scores + rail_ext_scores:
        if col not in pivot.columns:
            pivot[col] = np.nan

    road_mask = pivot["mode"] == "Road"
    rail_mask = pivot["mode"] == "Rail"

    pivot.loc[road_mask, "total_externalities_chf"] = pivot.loc[road_mask, road_ext_scores].fillna(0.0).sum(axis=1)
    pivot.loc[rail_mask, "total_externalities_chf"] = pivot.loc[rail_mask, rail_ext_scores].fillna(0.0).sum(axis=1)
    pivot.loc[road_mask, "noise_externality_chf"] = pivot.loc[road_mask, "road_noise_cost"]
    pivot.loc[rail_mask, "noise_externality_chf"] = pivot.loc[rail_mask, "rail_noise_cost"]

    road_lengths = pd.read_csv(ROAD_EXTERNALITY_DETAIL_CSV)
    road_lengths["development"] = road_lengths["development"].astype(str)
    road_new = road_lengths[road_lengths["link_role"] == "new_link"].copy()
    road_geom = (
        road_new.groupby("development", as_index=False)
        .agg(
            new_link_km=("link_length_m_geometry", lambda s: float(pd.to_numeric(s, errors="coerce").mean()) / 1000.0),
            surface_route_km=("surface_length_m", lambda s: float(pd.to_numeric(s, errors="coerce").mean()) / 1000.0),
            settlement_exposed_km=(
                "noise_relevant_share",
                lambda s: np.nan,
            ),
            road_noise_relevant_share=("noise_relevant_share", lambda s: float(pd.to_numeric(s, errors="coerce").mean())),
        )
    )
    road_geom["settlement_exposed_km"] = road_geom["surface_route_km"] * road_geom["road_noise_relevant_share"]
    road_geom["development_label"] = road_geom["development"]

    settlement_footprint, _ = load_settlement_footprint()
    rail_labels = _rail_label_lookup()
    rail_rows = []
    for gpkg_path in sorted(DEV_DIR.glob("*.gpkg")):
        if gpkg_path.name.startswith("._"):
            continue
        development = gpkg_path.stem.replace(".0", "")
        try:
            dev_gdf = gpd.read_file(gpkg_path)
        except Exception:
            continue
        if "new_dev" not in dev_gdf.columns:
            continue
        new_segments = dev_gdf[dev_gdf["new_dev"].astype(str).str.lower() == "yes"].copy()
        if new_segments.empty:
            continue
        segment_length_m = new_segments.geometry.length.astype(float)
        tunnel_length_m = pd.to_numeric(new_segments.get("Tunnel m", 0.0), errors="coerce").fillna(0.0)
        surface_length_m = np.maximum(segment_length_m - tunnel_length_m, 0.0)
        settlement_share = new_segments.geometry.apply(
            lambda geom: compute_settlement_buffer_share(geom, settlement_footprint)
        ).astype(float)
        rail_rows.append(
            {
                "development": development,
                "new_link_km": float(segment_length_m.sum()) / 1000.0,
                "surface_route_km": float(surface_length_m.sum()) / 1000.0,
                "settlement_exposed_km": float((surface_length_m * settlement_share).sum()) / 1000.0,
                "development_label": rail_labels.get(development, development),
            }
        )

    rail_geom = pd.DataFrame(rail_rows)

    geom = pd.concat(
        [
            road_geom.assign(mode="Road"),
            rail_geom.assign(mode="Rail"),
        ],
        ignore_index=True,
        sort=False,
    )

    comparison = pivot.merge(geom, on=["mode", "development"], how="left")
    comparison["development_label"] = comparison["development_label"].fillna(comparison["development"])
    comparison["total_externalities_mio_chf"] = comparison["total_externalities_chf"] / 1_000_000.0
    comparison["noise_externality_mio_chf"] = comparison["noise_externality_chf"] / 1_000_000.0
    comparison["externality_per_new_link_km"] = comparison["total_externalities_mio_chf"] / comparison["new_link_km"]
    comparison["externality_per_surface_route_km"] = comparison["total_externalities_mio_chf"] / comparison["surface_route_km"]
    comparison["noise_per_settlement_exposed_km"] = comparison["noise_externality_mio_chf"] / comparison["settlement_exposed_km"]
    return comparison


def plot_externality_per_km_violin(comparison_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = comparison_df.melt(
        id_vars=["mode", "development", "development_label"],
        value_vars=["externality_per_new_link_km", "externality_per_surface_route_km"],
        var_name="metric",
        value_name="value_mio_chf_per_km",
    ).dropna()
    metric_labels = {
        "externality_per_new_link_km": "Externalities / new link-km",
        "externality_per_surface_route_km": "Externalities / surface route-km",
    }
    plot_df["metric"] = plot_df["metric"].map(metric_labels)
    palette = {"Rail": "#D2B8E3", "Road": "#608BA7"}

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.violinplot(
        data=plot_df,
        x="metric",
        y="value_mio_chf_per_km",
        hue="mode",
        palette=palette,
        cut=0,
        inner="quartile",
        linewidth=0.9,
        dodge=True,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(0.80)
        collection.set_edgecolor("none")
    sns.stripplot(
        data=plot_df,
        x="metric",
        y="value_mio_chf_per_km",
        hue="mode",
        dodge=True,
        jitter=0.10,
        color="black",
        alpha=0.22,
        size=2.2,
        zorder=5,
        ax=ax,
    )
    ax.set_title("Integrated externalities normalized by route length")
    ax.set_xlabel("")
    ax.set_ylabel("Mean annual externalities [Mio. CHF / km]")
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    ax.legend([h for h, _ in filtered[:2]], [l for _, l in filtered[:2]], title="Mode", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_externality_total_vs_new_link_km(comparison_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {"Rail": "#8B6FB2", "Road": "#3D79A1"}
    for mode in ["Rail", "Road"]:
        sub = comparison_df[comparison_df["mode"] == mode].copy()
        ax.scatter(
            sub["new_link_km"],
            sub["total_externalities_mio_chf"],
            s=36,
            alpha=0.75,
            color=palette[mode],
            label=mode,
        )
    ax.set_title("Integrated externalities versus new link length")
    ax.set_xlabel("New link length [km]")
    ax.set_ylabel("Mean annual externalities [Mio. CHF/year]")
    _style_axes(ax)
    ax.legend(frameon=False, title="Mode")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_noise_vs_settlement_exposed_km(comparison_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))
    palette = {"Rail": "#8B6FB2", "Road": "#3D79A1"}
    for mode in ["Rail", "Road"]:
        sub = comparison_df[comparison_df["mode"] == mode].copy()
        ax.scatter(
            sub["settlement_exposed_km"],
            sub["noise_externality_mio_chf"],
            s=36,
            alpha=0.75,
            color=palette[mode],
            label=mode,
        )
    ax.set_title("Integrated noise costs versus settlement-exposed route length")
    ax.set_xlabel("Settlement-exposed route length [km]")
    ax.set_ylabel("Mean annual noise costs [Mio. CHF/year]")
    _style_axes(ax)
    ax.legend(frameon=False, title="Mode")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_vtt_ratio_violin(output_path: Path) -> None:
    ratio_plot_df = build_vtt_ratio_plot_df()
    palette = {"Rail": "#B070AF", "Road": "#0E4F84"}

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.violinplot(
        data=ratio_plot_df,
        x="label",
        y="adjusted_ratio",
        hue="mode",
        palette=palette,
        cut=0,
        inner="quartile",
        linewidth=0.9,
        dodge=True,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(0.80)
        collection.set_edgecolor("none")

    sns.stripplot(
        data=ratio_plot_df,
        x="label",
        y="adjusted_ratio",
        hue="mode",
        dodge=True,
        jitter=0.10,
        color = "black",
        alpha=0.5,
        size=1.9,
        zorder=5,
        ax=ax,
    )

    ax.axhline(1, color="black", linewidth=1.0)
    #ax.set_title("CBA sensitivity to value of travel time assumptions", fontsize=13, pad=12)
    ax.set_xlabel("VTTS (CHF/hour)", fontsize=10)
    ax.set_ylabel("CBA ratio", fontsize=10)
    ax.set_xticklabels([label.get_text().replace(": ", ":\n") for label in ax.get_xticklabels()])
    ax.tick_params(axis="both", labelsize=10)
    plt.xticks(ha="center")
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            filtered_handles.append(handle)
            filtered_labels.append(label)
            seen.add(label)
    ax.legend(filtered_handles[:2], filtered_labels[:2], title="Mode", frameon=False, fontsize=10, title_fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_vtt_ratio_violin_mean_scenario_ratios(output_path: Path) -> None:
    ratio_plot_df = build_vtt_ratio_mean_scenario_ratio_plot_df()
    palette = {"Rail": "#8A5A9E", "Road": "#4C92C3"}

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.violinplot(
        data=ratio_plot_df,
        x="label",
        y="adjusted_ratio",
        hue="mode",
        palette=palette,
        cut=0,
        inner="quartile",
        linewidth=0.9,
        dodge=True,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(0.80)
        collection.set_edgecolor("none")

    sns.stripplot(
        data=ratio_plot_df,
        x="label",
        y="adjusted_ratio",
        hue="mode",
        dodge=True,
        jitter=0.10,
        color="black",
        alpha=0.5,
        size=2.4,
        zorder=5,
        ax=ax,
    )

    ax.axhline(1, color="black", linewidth=1.0)
    ax.set_title("Mean of scenario-specific development CBA ratios by cross-modal VTT sources")
    ax.set_xlabel("VTTS (CHF/hour)")
    ax.set_ylabel("Mean scenario CBA ratio")
    plt.xticks(ha="right")
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            filtered_handles.append(handle)
            filtered_labels.append(label)
            seen.add(label)
    ax.legend(filtered_handles[:2], filtered_labels[:2], title="Mode", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_vtt_ratio_violin_by_scenario(output_path: Path) -> None:
    annual = pd.read_csv(ANALYSIS_DIR / "annual_overview_by_development_scenario.csv")
    integrated = annual[annual["value_mode"] == "integrated"].copy()
    integrated["cost_base_integrated"] = integrated["cost_annual"].abs()

    ratio_rows = []
    for row in VTT_SOURCE_PAIRS.itertuples(index=False):
        rail_base = integrated[integrated["mode"] == "Rail"].copy()
        road_base = integrated[integrated["mode"] == "Road"].copy()

        rail_base["adjusted_tts"] = rail_base["tts_annual"] * (row.rail_vtt / MODE_CONFIG["Rail"]["base_vtt"])
        rail_base["adjusted_ratio"] = rail_base["adjusted_tts"] / rail_base["cost_base_integrated"]
        rail_base["source"] = row.source
        rail_base["label"] = row.label

        road_base["adjusted_tts"] = road_base["tts_annual"] * (row.road_vtt / MODE_CONFIG["Road"]["base_vtt"])
        road_base["adjusted_ratio"] = road_base["adjusted_tts"] / road_base["cost_base_integrated"]
        road_base["source"] = row.source
        road_base["label"] = row.label
        ratio_rows.extend([rail_base, road_base])

    ratio_plot_df = pd.concat(ratio_rows, ignore_index=True)
    ratio_plot_df["label"] = pd.Categorical(
        ratio_plot_df["label"],
        categories=VTT_SOURCE_PAIRS["label"].tolist(),
        ordered=True,
    )
    palette = {"Rail": "#6788B6", "Road": "#2E5C88"}

    fig, ax = plt.subplots(figsize=(18, 7))
    sns.violinplot(
        data=ratio_plot_df,
        x="label",
        y="adjusted_ratio",
        hue="mode",
        palette=palette,
        cut=0,
        inner="quartile",
        linewidth=0.9,
        dodge=True,
        ax=ax,
    )
    for collection in ax.collections:
        collection.set_alpha(0.80)
        collection.set_edgecolor("none")

    sns.stripplot(
        data=ratio_plot_df,
        x="label",
        y="adjusted_ratio",
        hue="mode",
        dodge=True,
        jitter=0.12,
        alpha=0.18,
        size=1.8,
        color="black",
        zorder=5,
        ax=ax,
    )

    ax.axhline(1, color="black", linewidth=1.0)
    ax.set_title("Scenario-level CBA ratio by manually paired VTT sources")
    ax.set_xlabel("VTT source pair")
    ax.set_ylabel("CBA ratio")
    plt.xticks(rotation=25, ha="right")
    _style_axes(ax)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels):
        if label not in seen:
            filtered_handles.append(handle)
            filtered_labels.append(label)
            seen.add(label)
    ax.legend(filtered_handles[:2], filtered_labels[:2], title="Mode", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    

 


# ------------------------------------------
# Plotting functions
# ------------------------------------------

def plot_combined_top5_final_cost_savings(out_dir: Path) -> None:
    """Stacked bar chart: top 5 rail + top 5 road developments with cost breakdown."""
    rail_data = load_rail_final_costs_from_sources()
    road_data = load_road_final_costs_from_sources()
    
    rail_top = (
        rail_data.groupby(["development", "line_name"], as_index=False)
        .agg({
            "construction_cost_mio_chf": "mean",
            "maintenance_cost_mio_chf": "mean",
            "uncovered_operating_cost_mio_chf": "mean",
            "monetized_savings_mio_chf": "mean",
            "net_benefit_mio_chf": "mean",
        })
        .sort_values("net_benefit_mio_chf", ascending=False)
        .head(5)
        .assign(mode="Rail", label=lambda d: d["line_name"])
    )

    road_top = (
        road_data.groupby(["development", "line_name"], as_index=False)
        .agg({
            "construction_cost_mio_chf": "mean",
            "maintenance_cost_mio_chf": "mean",
            "other_cost_mio_chf": "mean",
            "monetized_savings_mio_chf": "mean",
            "net_benefit_mio_chf": "mean",
        })
        .sort_values("net_benefit_mio_chf", ascending=False)
        .head(5)
        .assign(mode="Road", label=lambda d: d["line_name"])
        .rename(columns={"other_cost_mio_chf": "externalities_cost_mio_chf"})
    )
    combined = pd.concat([rail_top, road_top], ignore_index=True)
    
    # Ensure all cost columns exist (rail doesn't have externalities, road doesn't have uncovered_operating)
    combined["uncovered_operating_cost_mio_chf"] = combined["uncovered_operating_cost_mio_chf"].fillna(0)
    combined["externalities_cost_mio_chf"] = combined.get("externalities_cost_mio_chf", 0)

    x_pos = np.arange(len(combined))
    bar_width = 0.6

    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    _shade_mode_groups(ax, combined["mode"])

    ax.bar(
        x_pos,
        -combined["construction_cost_mio_chf"],
        width=bar_width,
        color=COST_COLORS["construction"],
        label="Construction costs",
    )
    ax.bar(
        x_pos,
        -combined["maintenance_cost_mio_chf"],
        width=bar_width,
        bottom=-combined["construction_cost_mio_chf"],
        color=COST_COLORS["maintenance"],
        label="Maintenance costs",
    )
    ax.bar(
        x_pos,
        -combined["uncovered_operating_cost_mio_chf"],
        width=bar_width,
        bottom=-(combined["construction_cost_mio_chf"] + combined["maintenance_cost_mio_chf"]),
        color=COST_COLORS["operating"],
        label="Uncovered operating costs",
        )
    ax.bar(
        x_pos,
        -combined["externalities_cost_mio_chf"],
        width=bar_width,
        bottom=-(
            combined["construction_cost_mio_chf"]
            + combined["maintenance_cost_mio_chf"]
            + combined["uncovered_operating_cost_mio_chf"]
        ),
        color=COST_COLORS["externalities"],
        label="Externalities",
    )

    ax.bar(
        x_pos,
        combined["monetized_savings_mio_chf"],
        width=bar_width,
        color=COST_COLORS["tts"],
        label="Travel time savings",
    )

    ax.axhline(y=0, color="black", linestyle="-")
    ax.axhline(y=ax.get_ylim()[0],  color="0.7", linestyle="--", alpha=0.7)
    ax.axvline(x=4.5, color="black", linestyle="-", alpha=0.7, linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(combined["label"], rotation=90, fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title("Costs and benefits of development alternatives over all scenarios", fontsize=14, pad=20)
    ax.set_xlabel("Development ID", fontsize=12, labelpad=20)
    ax.set_ylabel("Average total value over all scenarios [Mio. CHF]", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ax.text(2, ax.get_ylim()[1] * 0.95, "Rail top 5", ha="center", fontsize=11)
    ax.text(7, ax.get_ylim()[1] * 0.95, "Road top 5", ha="center", fontsize=11)

    handles = [
        mpatches.Patch(color=COST_COLORS["construction"], label="Construction costs"),
        mpatches.Patch(color=COST_COLORS["maintenance"], label="Maintenance costs"),
        mpatches.Patch(color=COST_COLORS["operating"], label="Uncovered operating costs"),
        mpatches.Patch(color=COST_COLORS["externalities"], label="Externalities"),
        mpatches.Patch(color=COST_COLORS["tts"], label="Travel time savings"),
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(out_dir / "combined_top5_final_cost_savings.png", dpi=600)
    plt.close()

def plot_all_rail_final_cost_savings(output_dir: Path) -> None:
    rail_data = load_rail_final_costs_from_sources()
    # Aggregate by development (preserve summary order by net benefit)
    summary = (
        rail_data.groupby(["development", "line_name"], as_index=False)
        .agg(
            {
                "construction_cost_mio_chf": "mean",
                "maintenance_cost_mio_chf": "mean",
                "uncovered_operating_cost_mio_chf": "mean",
                "monetized_savings_mio_chf": "mean",
                "net_benefit_mio_chf": "mean",
            }
        )
        .sort_values("net_benefit_mio_chf", ascending=False)
        .reset_index(drop=True)
    )

    x_pos = np.arange(len(summary))
    bar_width = 0.6

    plt.figure(figsize=(max(7, len(summary) * 0.62), 5), dpi=300)
    plt.bar(x_pos, -summary["construction_cost_mio_chf"], width=bar_width, color=COST_COLORS["construction"])
    plt.bar(
        x_pos,
        -summary["maintenance_cost_mio_chf"],
        width=bar_width,
        bottom=-summary["construction_cost_mio_chf"],
        color=COST_COLORS["maintenance"],
    )
    plt.bar(
        x_pos,
        -summary["uncovered_operating_cost_mio_chf"],
        width=bar_width,
        bottom=-(summary["construction_cost_mio_chf"] + summary["maintenance_cost_mio_chf"]),
        color=COST_COLORS["operating"],
    )


    plt.bar(x_pos, summary["monetized_savings_mio_chf"], width=bar_width, color=COST_COLORS["tts"])
    plt.axhline(y=0, color="black", linestyle="-")
    plt.xticks(x_pos, summary["line_name"], rotation=90)
    plt.title("Costs and benefits of rail development alternatives over all scenarios", fontsize=14, pad=20)
    plt.xlabel("Development ID", fontsize=10, labelpad=20)
    plt.ylabel("Average total value over all scenarios [Mio. CHF]", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    handles = [
        mpatches.Patch(color=COST_COLORS["construction"], label="Construction costs"),
        mpatches.Patch(color=COST_COLORS["maintenance"], label="Maintenance costs"),
        mpatches.Patch(color=COST_COLORS["operating"], label="Uncovered operating costs"),
        mpatches.Patch(color=COST_COLORS["tts"], label="Travel time savings"),
    ]
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.legend(handles=handles, loc = "upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / "rail_all_final_cost_savings.png", dpi=600)
    plt.close()


def plot_all_road_final_cost_savings(output_dir: Path) -> None:
    road_data = load_road_final_costs_from_sources()
    # Aggregate and sort by net benefit
    summary = (
        road_data.groupby(["development", "line_name"], as_index=False)
        .agg(
            {
                "construction_cost_mio_chf": "mean",
                "maintenance_cost_mio_chf": "mean",
                "other_cost_mio_chf": "mean",
                "monetized_savings_mio_chf": "mean",
                "net_benefit_mio_chf": "mean",
            }
        )
        .sort_values("net_benefit_mio_chf", ascending=False)
        .reset_index(drop=True)
    )

    x_pos = np.arange(len(summary))
    bar_width = 0.6

    plt.figure(figsize=(max(7, len(summary) * 0.62), 5), dpi=300)
    plt.bar(x_pos, -summary["construction_cost_mio_chf"], width=bar_width, color=COST_COLORS["construction"])
    plt.bar(
        x_pos,
        -summary["maintenance_cost_mio_chf"],
        width=bar_width,
        bottom=-summary["construction_cost_mio_chf"],
        color=COST_COLORS["maintenance"],
    )
    plt.bar(
        x_pos,
        -summary["other_cost_mio_chf"],
        width=bar_width,
        bottom=-(summary["construction_cost_mio_chf"] + summary["maintenance_cost_mio_chf"]),
        color=COST_COLORS["externalities"],
    )

    plt.bar(x_pos, summary["monetized_savings_mio_chf"], width=bar_width, color=COST_COLORS["tts"])

    plt.axhline(y=0, color="black", linestyle="-")
    plt.xticks(x_pos, summary["line_name"], rotation=90)
    plt.xlabel("Development ID", fontsize=10, labelpad=20)
    plt.ylabel("Average total value over all scenarios [Mio. CHF]", fontsize=10)
    plt.title("Costs and benefits of road development alternatives over all scenarios", fontsize=14, pad=20)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    handles = [
        mpatches.Patch(color=COST_COLORS["construction"], label="Construction costs"),
        mpatches.Patch(color=COST_COLORS["maintenance"], label="Maintenance costs"),
        mpatches.Patch(color=COST_COLORS["externalities"], label="Externalities"),
        mpatches.Patch(color=COST_COLORS["tts"], label="Travel time savings"),
    ]
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / "road_all_final_cost_savings.png", dpi=600)
    plt.close()
   

def plot_final_tts_boxplot_top5(output_dir: Path | None = None) -> Path:
    """
    Final cross-mode TTS boxplot:
    - Rail: demand-weighted daily TTS before monetization
    - Road: demand-weighted daily TTS before monetization
    """
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "combined_top5_tts_boxplot.png"
    
    selection_df = (
        _build_integrated_bcr_top10_by_mode_plot_data(5)[["mode", "development", "plot_order", "ranking_label_short", "bcr_mean", "net_benefit_mean"]]
        .drop_duplicates()
        .sort_values("net_benefit_mean", ascending=False)
    )
    selection_df["development"] = selection_df["development"].astype(str)
    rail_top = selection_df.loc[selection_df["mode"] == "Rail", "development"].tolist()
    road_top = selection_df.loc[selection_df["mode"] == "Road", "development"].tolist()
    display_labels = {
        f"{row.mode} {row.development}": str(row.ranking_label_short)
        for row in selection_df.itertuples(index=False)
    }
    
    
    data = build_weighted_tts_by_scenario(rail_top, road_top, annualize=False).copy()
    if data.empty:
        return output_path
    data["label"] = data["mode"] + " " + data["development"].astype(str)
    order = [f"{row.mode} {row.development}" for row in selection_df.itertuples(index=False)]
    order = [label for label in order if label in set(data["label"])]
    development_color_lookup = _development_color_lookup(
        _top_ranked_developments("Rail", 5),
        _top_ranked_developments("Road", 5),
    )

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    positions = np.arange(len(order))
    _shade_mode_groups(ax, [label.split(" ", 1)[0] for label in order])
    for pos, label in enumerate(order):
        mode, development = label.split(" ", 1)
        values = data[data["label"] == label]["weighted_tts_person_h_per_year"].dropna().to_numpy()
        color = development_color_lookup.get((mode, development), "#777777")
        ax.boxplot(
            [values],
            positions=[pos],
            widths=0.58,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none", "markeredgecolor": "0.4"},
            boxprops={"facecolor": color, "alpha": 0.85, "edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([display_labels.get(label, label.split(" ", 1)[1]) for label in order], rotation=90)
    ax.set_xlabel("Development ID", fontsize=15)
    ax.set_ylabel("Demand-weighted TTS [person-h/day]", fontsize=15)
    ax.tick_params(axis="y", labelsize=13)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    _style_axes(ax)

    handles = [Line2D([0], [0], marker="o", color="black", label="Mean", markersize=5, linestyle="None")]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=13)

    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.92)
    fig.savefig(output_path, dpi=200, transparent=True)
    plt.close(fig)
    return output_path


def export_unweighted_od_tt_savings_top5(output_dir: Path | None = None) -> Path:
    """Export unweighted OD-pair travel-time savings for the integrated top 5 per mode."""
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "combined_top5_unweighted_od_tt_savings.csv"

    selection_df = (
        _build_integrated_bcr_top10_by_mode_plot_data(5)[["mode", "development", "plot_order", "ranking_label_short", "bcr_mean", "net_benefit_mean"]]
        .drop_duplicates()
        .sort_values("net_benefit_mean", ascending=False)
    )
    selection_df["development"] = selection_df["development"].astype(str)

    rows = []
    for row in selection_df.itertuples(index=False):
        if row.mode == "Rail":
            rail_path = RAIL_TRAVELTIME_SAVINGS_DIR / f"TravelTime_Savings_Dev_{row.development}.csv"
            if not rail_path.exists():
                continue
            tt = pd.read_csv(rail_path)
            tt["unweighted_tt_savings_min"] = -pd.to_numeric(tt["delta_time"], errors="coerce")
            source = "rail_od_station_pairs"
        else:
            if not ROAD_OD_TT_RAW_BEFORE_WEIGHTING_CSV.exists():
                continue
            tt = pd.read_csv(ROAD_OD_TT_RAW_BEFORE_WEIGHTING_CSV)
            tt["development"] = tt["development"].astype(str).str.replace(r"\.0$", "", regex=True)
            tt = tt[tt["development"] == str(row.development)].copy()
            tt["unweighted_tt_savings_min"] = pd.to_numeric(tt["raw_tt_savings_min"], errors="coerce")
            source = "road_raw_commune_od_pairs_by_scenario"

        savings = tt["unweighted_tt_savings_min"].dropna()
        if savings.empty:
            continue
        positive_savings = savings[savings > 0]
        top20_savings = savings.sort_values(ascending=False).head(20)
        rows.append(
            {
                "mode": row.mode,
                "development": row.development,
                "development_label": row.ranking_label_short,
                "source": source,
                "n_unweighted_od_observations": int(savings.shape[0]),
                "mean_unweighted_tt_savings_min": float(savings.mean()),
                "median_unweighted_tt_savings_min": float(savings.median()),
                "std_unweighted_tt_savings_min": float(savings.std()),
                "min_unweighted_tt_savings_min": float(savings.min()),
                "max_unweighted_tt_savings_min": float(savings.max()),
                "share_positive_tt_savings": float((savings > 0).mean()),
                "n_positive_od_observations": int(positive_savings.shape[0]),
                "mean_positive_unweighted_tt_savings_min": float(positive_savings.mean()) if not positive_savings.empty else 0.0,
                "median_positive_unweighted_tt_savings_min": float(positive_savings.median()) if not positive_savings.empty else 0.0,
                "mean_top20_unweighted_tt_savings_min": float(top20_savings.mean()),
                "min_top20_unweighted_tt_savings_min": float(top20_savings.min()),
                "max_top20_unweighted_tt_savings_min": float(top20_savings.max()),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_path, index=False)
    return output_path


def plot_integrated_bcr_boxplot_top5(output_dir: Path | None = None) -> Path:
    """Scenario-level integrated BCR distribution for the top 5 rail and road developments."""
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "integrated_bcr_top5_boxplot.png"

    selection_df = (
        _build_integrated_bcr_top10_by_mode_plot_data(5)[["mode", "development", "plot_order", "ranking_label_short", "bcr_mean", "net_benefit_mean"]]
        .drop_duplicates()
        .sort_values(["mode", "net_benefit_mean", "plot_order"], ascending=[True, False, True])
        .groupby("mode", group_keys=False)
        .head(5)
        .sort_values("net_benefit_mean", ascending=False)
        .reset_index(drop=True)
    )
    selection_df["development"] = selection_df["development"].astype(str)

    bcr_df = _scenario_bcr_net_benefit().merge(
        selection_df[["mode", "development"]],
        on=["mode", "development"],
        how="inner",
    )
    bcr_df = bcr_df.merge(selection_df, on=["mode", "development"], how="left")

    order_rows = selection_df.sort_values("net_benefit_mean", ascending=False).reset_index(drop=True)
    development_color_lookup = _development_color_lookup(
        _top_ranked_developments("Rail", 5),
        _top_ranked_developments("Road", 5),
    )

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=200)
    positions = np.arange(len(order_rows))
    _shade_mode_groups(ax, order_rows["mode"])
    for pos, row in enumerate(order_rows.itertuples(index=False)):
        values = bcr_df[
            (bcr_df["mode"] == row.mode)
            & (bcr_df["development"] == row.development)
        ]["bcr"].dropna().to_numpy()
        color = development_color_lookup.get((str(row.mode), str(row.development)), "#777777")
        ax.boxplot(
            [values],
            positions=[pos],
            widths=0.58,
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4},
            flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "none", "markeredgecolor": "0.4"},
            boxprops={"facecolor": color, "alpha": 0.85, "edgecolor": "black", "linewidth": 0.8},
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
        )

    ax.axhline(y=1, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(order_rows["ranking_label_short"], rotation=90, fontsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel("Development ID", fontsize=16)
    ax.set_ylabel("Benefit-cost ratio [-]", fontsize=16)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    _style_axes(ax)
    handles = [
        Line2D([0], [0], marker="o", color="black", label="Mean", markersize=5, linestyle="None"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="BCR = 1"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=13)
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.30, top=0.92)
    fig.savefig(output_path, dpi=200, transparent=True)
    plt.close(fig)
    return output_path


def plot_mean_net_benefit_ecdf_by_mode(output_dir: Path | None = None) -> Path:
    """ECDF of mean integrated net benefits by development, split by mode."""
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "mean_net_benefit_ecdf_by_mode.png"

    summary = (
        _scenario_bcr_net_benefit()[["mode", "development", "net_benefit_mio_chf"]]
        .groupby(["mode", "development"], as_index=False)
        .agg(mean_net_benefit_mio_chf=("net_benefit_mio_chf", "mean"))
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    mode_colors = {"Rail": "#B070AF", "Road": "#0E4F84"}

    for mode in ["Rail", "Road"]:
        mode_df = summary[summary["mode"] == mode].sort_values("mean_net_benefit_mio_chf").reset_index(drop=True)
        if mode_df.empty:
            continue
        x = mode_df["mean_net_benefit_mio_chf"].to_numpy()
        y = np.arange(1, len(mode_df) + 1) / len(mode_df)
        ax.step(x, y, where="post", linewidth=2.2, color=mode_colors[mode], label=f"{mode} (n={len(mode_df)})")
        ax.plot(
            x,
            np.full_like(x, 0.015 if mode == "Rail" else 0.035, dtype=float),
            linestyle="None",
            marker="|",
            markersize=14,
            markeredgewidth=1.4,
            color=mode_colors[mode],
            alpha=0.95,
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Mean net benefit per development [Mio. CHF/year]", fontsize=14)
    ax.set_ylabel("Cumulative probability", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, linestyle="--", alpha=0.35)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=11, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    return output_path


def plot_net_benefit_ecdf_all_developments_by_mode(output_dir: Path | None = None) -> Path:
    """Scenario-level ECDF of integrated net benefits for every development."""
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "net_benefit_ecdf_all_developments_by_mode.png"

    data = _scenario_bcr_net_benefit()[["mode", "development", "net_benefit_mio_chf"]].copy()
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    mode_colors = {"Rail": "#B070AF", "Road": "#0E4F84"}
    mode_counts = {}

    for mode in ["Rail", "Road"]:
        mode_df = data[data["mode"] == mode].copy()
        mode_counts[mode] = mode_df["development"].nunique()
        for _, dev_df in mode_df.groupby("development"):
            values = dev_df["net_benefit_mio_chf"].dropna().sort_values().to_numpy()
            if values.size == 0:
                continue
            y = np.arange(1, values.size + 1) / values.size
            ax.step(
                values,
                y,
                where="post",
                color=mode_colors[mode],
                linewidth=1.05,
                alpha=0.18 if mode == "Road" else 0.28,
            )

    for y, label in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%")]:
        ax.axhline(y, color="0.65", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(1.002, y, label, transform=ax.get_yaxis_transform(), va="center", ha="left", color="0.5", fontsize=9)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Scenario-level distribution of integrated net benefits", fontsize=14, pad=12)
    ax.set_xlabel("Integrated net benefit [Mio. CHF/year]", fontsize=12)
    ax.set_ylabel("Cumulative probability", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.grid(True, linestyle="--", alpha=0.32)
    _style_axes(ax)
    handles = [
        Line2D([0], [0], color=mode_colors["Rail"], linewidth=2.2, label=f"Rail (n={mode_counts.get('Rail', 0)})"),
        Line2D([0], [0], color=mode_colors["Road"], linewidth=2.2, label=f"Road (n={mode_counts.get('Road', 0)})"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    return output_path


def plot_bcr_ecdf_all_developments_by_mode(output_dir: Path | None = None) -> Path:
    """Scenario-level ECDF of integrated BCR for every development."""
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "bcr_ecdf_all_developments_by_mode.png"

    data = _scenario_bcr_net_benefit()[["mode", "development", "bcr"]].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["bcr"])

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    mode_colors = {"Rail": "#B070AF", "Road": "#0E4F84"}
    mode_counts = {}

    for mode in ["Rail", "Road"]:
        mode_df = data[data["mode"] == mode].copy()
        mode_counts[mode] = mode_df["development"].nunique()
        for _, dev_df in mode_df.groupby("development"):
            values = dev_df["bcr"].dropna().sort_values().to_numpy()
            if values.size == 0:
                continue
            y = np.arange(1, values.size + 1) / values.size
            ax.step(
                values,
                y,
                where="post",
                color=mode_colors[mode],
                linewidth=1.35,
                alpha=0.26 if mode == "Road" else 0.36,
            )

    for y, label in [(0.25, "25%"), (0.50, "50%"), (0.75, "75%")]:
        ax.axhline(y, color="0.65", linestyle=":", linewidth=1.0, alpha=0.7)
        ax.text(1.002, y, label, transform=ax.get_yaxis_transform(), va="center", ha="left", color="0.5", fontsize=12)

    ax.axvline(1, color="black", linestyle="-", linewidth=1.1, alpha=0.9)
    ax.set_title("Scenario-level distribution of integrated benefit-cost ratios", fontsize=18, pad=14)
    ax.set_xlabel("Integrated benefit-cost ratio [-]", fontsize=16)
    ax.set_ylabel("Cumulative probability", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, linestyle="--", alpha=0.32)
    _style_axes(ax)
    handles = [
        Line2D([0], [0], color=mode_colors["Rail"], linewidth=2.2, label=f"Rail (n={mode_counts.get('Rail', 0)})"),
        Line2D([0], [0], color=mode_colors["Road"], linewidth=2.2, label=f"Road (n={mode_counts.get('Road', 0)})"),
        Line2D([0], [0], color="black", linewidth=1.1, label="BCR = 1"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=14, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    return output_path


def plot_mean_bcr_lollipop_all_developments(output_dir: Path | None = None) -> Path:
    """Ranked point plot of mean integrated BCR for all developments."""
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "mean_bcr_lollipop_all_developments.png"

    summary = (
        _scenario_bcr_net_benefit()[["mode", "development", "bcr"]]
        .groupby(["mode", "development"], as_index=False)
        .agg(
            mean_bcr=("bcr", "mean"),
            bcr_p25=("bcr", lambda values: values.quantile(0.25)),
            bcr_p75=("bcr", lambda values: values.quantile(0.75)),
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["mean_bcr"])
        .sort_values("mean_bcr", ascending=True)
        .reset_index(drop=True)
    )
    rail_labels = _rail_label_lookup()
    summary["display_label"] = np.where(
        summary["mode"].eq("Rail"),
        summary["development"].map(rail_labels).fillna(summary["development"]),
        summary["development"],
    )
    summary["rank"] = np.arange(len(summary))

    fig_height = max(8, len(summary) * 0.16)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)
    mode_colors = {"Rail": "#B070AF", "Road": "#0E4F84"}

    for mode in ["Rail", "Road"]:
        sub = summary[summary["mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(
            sub["mean_bcr"],
            sub["rank"],
            s=28,
            color=mode_colors[mode],
            label=mode,
            zorder=3,
        )
        lower = (sub["mean_bcr"] - sub["bcr_p25"]).clip(lower=0)
        upper = (sub["bcr_p75"] - sub["mean_bcr"]).clip(lower=0)
        ax.errorbar(
            sub["mean_bcr"],
            sub["rank"],
            xerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=mode_colors[mode],
            elinewidth=1.2,
            capsize=2.2,
            alpha=0.7,
            zorder=2,
        )

    ax.set_yticks(summary["rank"])
    ax.set_yticklabels(summary["display_label"], fontsize=7)
    ax.set_xlabel("Mean benefit-cost ratio with interquartile range", fontsize=14)
    ax.set_ylabel("All developments by mean BCR", fontsize=14)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    _style_axes(ax)
    ax.axvline(1, color="#222222", linestyle="-", linewidth=1.8, alpha=0.9, zorder=2.5)
    ax.legend(frameon=False, fontsize=15, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    return output_path


def plot_mean_bcr_lollipop_top25(output_dir: Path | None = None) -> Path:
    """Ranked lollipop plot of the top 25 mean integrated BCRs."""
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "mean_bcr_lollipop_top25.png"

    summary = (
        _scenario_bcr_net_benefit()[["mode", "development", "bcr"]]
        .groupby(["mode", "development"], as_index=False)
        .agg(
            mean_bcr=("bcr", "mean"),
            bcr_p25=("bcr", lambda values: values.quantile(0.25)),
            bcr_p75=("bcr", lambda values: values.quantile(0.75)),
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["mean_bcr"])
        .sort_values("mean_bcr", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )
    rail_labels = _rail_label_lookup()
    summary["display_label"] = np.where(
        summary["mode"].eq("Rail"),
        summary["development"].map(rail_labels).fillna(summary["development"]),
        summary["development"],
    )
    summary["rank"] = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    mode_colors = {"Rail": "#B070AF", "Road": "#0E4F84"}

    for mode in ["Rail", "Road"]:
        sub = summary[summary["mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(
            sub["rank"],
            sub["mean_bcr"],
            s=34,
            color=mode_colors[mode],
            label=mode,
            zorder=3,
        )
        lower = (sub["mean_bcr"] - sub["bcr_p25"]).clip(lower=0)
        upper = (sub["bcr_p75"] - sub["mean_bcr"]).clip(lower=0)
        ax.errorbar(
            sub["rank"],
            sub["mean_bcr"],
            yerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor=mode_colors[mode],
            elinewidth=1.8,
            capsize=3.0,
            alpha=0.75,
            zorder=2,
        )

    ax.set_xticks(summary["rank"])
    ax.set_xticklabels(summary["display_label"], rotation=90, fontsize=10)
    ax.set_xlabel("Top 25 developments by mean BCR", fontsize=14)
    ax.set_ylabel("Mean benefit-cost ratio with interquartile range", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    _style_axes(ax)
    ax.axhline(1, color="#222222", linestyle="-", linewidth=1.8, alpha=0.9, zorder=2.5)
    ax.legend(frameon=False, fontsize=15, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, transparent=True)
    plt.close(fig)
    return output_path


def build_rail_integrated_appraisal_plot_df() -> pd.DataFrame:
    """Map integrated rail appraisal scores to the rail Benefits_Combined plot schema."""
    score_df = pd.read_csv(SCORE_RESULTS_DIR / "score_results_long.csv")
    rail = score_df[score_df["mode"].eq("Rail")].copy()
    pivot = (
        rail.pivot_table(
            index=["development", "scenario"],
            columns="score_id",
            values="integrated_value",
            aggfunc="sum",
        )
        .reset_index()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )
    externality_scores = [
        "rail_accident_cost",
        "rail_airpollution_cost",
        "rail_co2_cost",
        "rail_noise_cost",
        "rail_land_consumption_cost",
    ]
    for column in [
        "rail_construction_cost",
        "rail_maint_cost",
        "rail_operation_cost",
        "rail_tts_cost",
        *externality_scores,
    ]:
        if column not in pivot.columns:
            pivot[column] = 0.0

    return pd.DataFrame(
        {
            "development": pivot["development"].astype(float),
            "scenario": pivot["scenario"].astype(str),
            "year": RAIL_COMPARISON_YEAR,
            "TotalConstructionCost": pivot["rail_construction_cost"],
            "TotalMaintenanceCost": pivot["rail_maint_cost"],
            "TotalUncoveredOperatingCost": pivot["rail_operation_cost"],
            "TotalExternalityCost": pivot[externality_scores].sum(axis=1),
            "monetized_savings_total": pivot["rail_tts_cost"],
            "__annualized": True,
        }
    )


def plot_rail_integrated_appraisal_benefits_combined(output_dir: Path | None = None) -> Path:
    """Create Rail Benefits_Combined-style plots from the integrated rail appraisal."""
    if output_dir is None:
        output_dir = integrated_paths.INTEGRATED_PLOTS_DIR / "rail_integrated"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_df = build_rail_integrated_appraisal_plot_df()
    plot_df.to_csv(output_dir / "rail_integrated_appraisal_plot_input.csv", index=False)

    from infraScan.infraScanRail import plots as rail_result_plots

    railway_lines = gpd.read_file(RAIL_NEW_RAILWAY_LINES_GPKG)
    rail_result_plots.create_and_save_plots(
        df=plot_df,
        railway_lines=railway_lines,
        plot_directory=str(output_dir),
        plot_preferences={
            "small_developments": False,
            "grouped_by_connection": True,
            "ranked_groups": False,
            "combined_with_maps": True,
        },
    )
    for path in [output_dir / "Benefits", output_dir / "Benefits_Ranked"]:
        if path.exists():
            shutil.rmtree(path)
    for path in output_dir.glob("railway_lines_*.png"):
        path.unlink()
    return output_dir


def plot_integrated_overview_shared_layout(output_dir: Path | None = None) -> Path:
    """Combine the map, TTS, BCR, and stacked-value panels into one overview PNG."""
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "integrated_overview_shared_layout.png"

    def crop_white(path: Path, pad: int = 10) -> Image.Image:
        img = Image.open(path).convert("RGB")
        diff = ImageChops.difference(img, Image.new("RGB", img.size, "white"))
        bbox = ImageChops.multiply(diff, diff).getbbox()
        return img.crop((max(bbox[0] - pad, 0), max(bbox[1] - pad, 0), min(bbox[2] + pad, img.width), min(bbox[3] + pad, img.height))) if bbox else img

    def fit(img: Image.Image, width: int, height: int) -> Image.Image:
        scale = min(width / img.width, height / img.height)
        return img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.LANCZOS)

    chart_w, chart_h, map_w, map_h, label_h, margin, gap_x, gap_y = 1500, 812, 1500, 930, 48, 50, 70, 55
    canvas = Image.new("RGB", (margin * 2 + chart_w * 2 + gap_x, margin * 2 + label_h + map_h + gap_y + label_h + chart_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_label, font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 32), ImageFont.truetype("DejaVuSans.ttf", 28)
    except OSError:
        font_label = font_title = ImageFont.load_default()

    panels = [
        ("A", "Development locations", "integrated_top5_rail_highway_overview.png", margin, margin, map_w, map_h),
        ("B", "Benefit-cost ratio", "integrated_bcr_top5_boxplot.png", margin + chart_w + gap_x, margin + (map_h - chart_h) // 2, chart_w, chart_h),
        ("C", "Demand-weighted TTS", "combined_top5_tts_boxplot.png", margin, margin + label_h + map_h + gap_y, chart_w, chart_h),
        ("D", "Stacked annual values", "integrated_bcr_top5_by_mode_stacked.png", margin + chart_w + gap_x, margin + label_h + map_h + gap_y, chart_w, chart_h),
    ]
    for letter, title, filename, x, y, box_w, box_h in panels:
        img = fit(crop_white(Path(output_dir) / filename, 14 if letter == "A" else 10), box_w, box_h)
        draw.text((x, y), letter, fill="black", font=font_label)
        draw.text((x + 54, y + 1), title, fill="black", font=font_title)
        canvas.paste(img, (x + (box_w - img.width) // 2, y + label_h + (box_h - img.height) // 2))

    canvas.save(output_path, quality=95)
    return output_path


# ------------------------------------------
# Main execution
# ------------------------------------------

def main() -> None:
    GENERATED_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    externality_comparison = build_externality_normalized_df()
    externality_comparison.to_csv(
        GENERATED_PLOTS_DIR / "externality_normalized_comparison.csv",
        index=False,
    )

    # Optional standalone-vs-integrated comparison plots.
    # plot_mode_standalone_vs_integrated(
    #     mode="Road",
    #     output_path=GENERATED_PLOTS_DIR / "road_stacked_integrated_vs_standalone_annual.png",
    # )
    # plot_mode_standalone_vs_integrated(
    #     mode="Road",
    #     output_path=GENERATED_PLOTS_DIR / "road_stacked_integrated_vs_standalone_annual_top40.png",
    #     max_developments=40,
    # )
    # plot_mode_standalone_vs_integrated(
    #     mode="Rail",
    #     output_path=GENERATED_PLOTS_DIR / "rail_stacked_integrated_vs_standalone_annual.png",
    # )
    plot_integrated_bcr_top10_by_mode(
        output_path=GENERATED_PLOTS_DIR / "integrated_bcr_top5_by_mode_stacked.png",
        top_n_per_mode=5,
    )
    plot_top5_rail_highway_overview(
        output_path=GENERATED_PLOTS_DIR / "integrated_top5_rail_highway_overview.png",
        top_n=5,
    )
    plot_final_tts_boxplot_top5(GENERATED_PLOTS_DIR)
    export_unweighted_od_tt_savings_top5(GENERATED_PLOTS_DIR)
    plot_integrated_bcr_boxplot_top5(GENERATED_PLOTS_DIR)
    plot_integrated_overview_shared_layout(GENERATED_PLOTS_DIR)
    plot_rail_integrated_appraisal_benefits_combined()
    plot_net_benefit_ecdf_all_developments_by_mode(GENERATED_PLOTS_DIR)
    plot_bcr_ecdf_all_developments_by_mode(GENERATED_PLOTS_DIR)
    # plot_mean_bcr_lollipop_all_developments(GENERATED_PLOTS_DIR)
    plot_mean_bcr_lollipop_top25(GENERATED_PLOTS_DIR)
    plot_vtt_ratio_violin(
        output_path=GENERATED_PLOTS_DIR / "vtt_ratio_violin_by_mode.png",
    )
    # plot_externality_total_vs_new_link_km(
    #     externality_comparison,
    #     output_path=GENERATED_PLOTS_DIR / "externality_total_vs_new_link_km.png",
    # )

    print("Saved plots to:", GENERATED_PLOTS_DIR)


if __name__ == "__main__":
    main()
