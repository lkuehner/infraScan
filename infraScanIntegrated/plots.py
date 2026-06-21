from pathlib import Path
from typing import Iterable
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
import geopandas as gpd
import rasterio

from infraScan.infraScanIntegrated import paths as integrated_paths
from infraScan.infraScanRail import paths as rail_paths
from infraScan.infraScanRoad.externalities_comp import compute_settlement_buffer_share
from infraScan.infraScanIntegrated.scoring_registry import load_settlement_footprint


#TO DO: make automatic based on generated_selected_scenarios
SCENARIOS = ('scenario_26', 'scenario_70', 'scenario_89', 'scenario_100', 'scenario_75', 'scenario_96', 'scenario_44', 'scenario_19', 'scenario_64', 'scenario_78')
#("scenario_76", "scenario_45", "scenario_67")

RAIL_COMPARISON_YEAR = 2050

COST_COLORS = {
    "construction": "#263852",
    "maintenance": "#547285",
    "operating": "#B2C2D3",
    "accident": "#A53D3D",
    "air": "#BF5A5A",
    "co2": "#D07A7A",
    "noise": "#8F2F2F",
    "land": "#838383",
    "externalities": "#B64B4B",
    "tts": "#91B58D",
}

# Centralized paths (adjust these in paths.py)
RAIL_TOTAL_COSTS_CSV = integrated_paths.RAIL_COSTS_DIR / "total_costs.csv"
RAIL_TT_SAVINGS_CSV = integrated_paths.RAIL_COSTS_DIR / "traveltime_savings.csv"
ROAD_TOTAL_COSTS_CSV = integrated_paths.ROAD_COSTS_DIR / "total_costs_od.csv"
ROAD_TT_OD_CSV = integrated_paths.ROAD_COSTS_DIR / "traveltime_savings_od.csv"
ROAD_TT_DETAILED_CSV = integrated_paths.ROAD_DATA_ROOT / "traffic_flow" / "od" / "od_tt_savings_detailed.csv"
ROAD_OD_TT_UNWEIGHTED_CSV = integrated_paths.ROAD_DATA_ROOT / "traffic_flow" / "od" / "developments_od_tt.csv"
ROAD_OD_TT_STATUS_QUO_CSV = integrated_paths.ROAD_DATA_ROOT / "traffic_flow" / "od" / "status_quo_od_tt.csv"

RAIL_TRAVELTIME_CACHE = integrated_paths.RAIL_NETWORK_PATH / "travel_time" / "cache" / "od_times.pkl"
RAIL_TRAVELTIME_SAVINGS_DIR = integrated_paths.RAIL_NETWORK_PATH / "travel_time" / "TravelTime_Savings"
ROAD_TRAVELTIME_RASTER = integrated_paths.ROAD_NETWORK_PATH / "travel_time" / "travel_time_raster.tif"
DEV_DIR = integrated_paths.DATA_ROOT / rail_paths.DEVELOPMENT_DIRECTORY
ANALYSIS_DIR = integrated_paths.INTEGRATED_COSTS_DIR / "score_analysis"
SCORE_RESULTS_DIR = integrated_paths.SCORE_RESULTS_DIR
GENERATED_PLOTS_DIR = integrated_paths.GENERATED_PLOTS_DIR
ROAD_EXTERNALITY_DETAIL_CSV = integrated_paths.ROAD_EXTERNALITIES_DETAIL_CSV

MODE_CONFIG = {
    "Rail": {"base_vtt": 25.24},
    "Road": {"base_vtt": 26.85},
}

VTT_SOURCE_PAIRS = pd.DataFrame(
    [
        {"source": "Literature", "rail_vtt":
         15.20, "road_vtt": 31.40},
        {"source": "VSS-Norm by guidelines", 
         "rail_vtt": 25.24, "road_vtt": 26.85},
        {"source": "VSS-Norm by distance", 
         "rail_vtt": 24.25, "road_vtt": 38.68},
        {"source": "VSS-Norm by purpose", 
         "rail_vtt": 16.63, "road_vtt": 26.85},
    ]
)
VTT_SOURCE_PAIRS["label"] = VTT_SOURCE_PAIRS.apply(
    lambda row: f"{row['source']}: Rail {row['rail_vtt']:.2f} / Road {row['road_vtt']:.2f}",
    axis=1,
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


def load_rail_weighted_tts(rail_top: list[str] | None = None) -> pd.DataFrame:
    """Load rail demand-weighted annual travel-time savings before monetization."""
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
    df["weighted_tts_person_h_per_year"] = pd.to_numeric(df["tt_savings_daily"], errors="coerce") * 365.0
    df = df[np.isfinite(df["weighted_tts_person_h_per_year"])].copy()
    df["mode"] = "Rail"
    return df[["mode", "development", "scenario", "weighted_tts_person_h_per_year"]]


def load_road_weighted_tts(road_top: list[str] | None = None) -> pd.DataFrame:
    """Load road demand-weighted annual travel-time savings before monetization."""
    df = pd.read_csv(ROAD_TT_DETAILED_CSV)
    selected_scenarios = _selected_scenarios("Road")

    df["development"] = df["development"].astype(str).str.replace(r"\.0$", "", regex=True)
    df["scenario"] = df["scenario"].astype(str)
    df = df[df["scenario"].isin(selected_scenarios)].copy()
    if road_top is not None:
        df = df[df["development"].isin([str(dev) for dev in road_top])].copy()
    df["weighted_tts_person_h_per_year"] = pd.to_numeric(df["tt_savings_peak"], errors="coerce") / 60.0 * 2.5 * 250.0
    df = df[np.isfinite(df["weighted_tts_person_h_per_year"])].copy()
    df["mode"] = "Road"
    return df[["mode", "development", "scenario", "weighted_tts_person_h_per_year"]]


def build_weighted_tts_by_scenario(
    rail_top: list[str] | None = None,
    road_top: list[str] | None = None,
) -> pd.DataFrame:
    """Build the common demand-weighted TTS table used by TTS plots."""
    rail = load_rail_weighted_tts(rail_top)
    road = load_road_weighted_tts(road_top)
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
        selected = bcr.sort_values("bcr_mean", ascending=False).head(top_n_per_mode).copy()
        selected["mode"] = mode
        selected["ranking_label_short"] = selected["development_label"]
        selected["plot_order"] = np.arange(len(selected))

        plot_df = integrated.drop(columns=["plot_order"], errors="ignore").merge(
            selected[["mode", "development", "ranking_label_short", "bcr_mean", "plot_order"]],
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
            plot_df[["mode", "development", "ranking_label_short", "bcr_mean", "plot_order", "score_id", "value_mio_chf"]]
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
        (dev_num - 99999).astype(str) + "_" + sline,
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
    if "land_consumption" in score_id:
        return COST_COLORS["land"]
    if "climate" in score_id or "ecological" in score_id:
        return COST_COLORS["externalities"]
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
        "climate_cost": "Climate costs",
        "ecological_disruption_cost": "Ecological disruption costs",
        "tts_cost": "Travel time savings",
    }
    for suffix, label in label_map.items():
        if score_id.endswith(suffix):
            return label
    return score_id


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
    ordered_selection = (
        plot_df[["mode", "development", "plot_order", "bcr_mean"]]
        .drop_duplicates()
        .sort_values(["mode", "bcr_mean", "plot_order"], ascending=[True, False, True])
        .groupby("mode", group_keys=False)
        .head(top_n_per_mode)
        .copy()
    )
    ordered_selection = ordered_selection.sort_values(["bcr_mean", "plot_order"], ascending=[False, True])
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
        "road_noise_cost", "road_land_consumption_cost", "road_tts_cost",
    ]
    pivot = plot_df.pivot_table(
        index=["plot_order", "ranking_label_short", "bcr_mean"],
        columns="score_id",
        values="value_mio_chf",
        aggfunc="mean",
    ).reset_index().sort_values("plot_order").reset_index(drop=True)

    y = np.arange(len(pivot))
    fig, ax = plt.subplots(figsize=(14, max(8, len(pivot) * 0.95)))
    neg = np.zeros(len(pivot))
    pos = np.zeros(len(pivot))
    for score_id in component_order:
        if score_id not in pivot.columns:
            continue
        values = pivot[score_id].fillna(0.0).to_numpy()
        if score_id.endswith("tts_cost"):
            ax.barh(y, values, height=0.55, left=pos, color=_component_color(score_id),
                    edgecolor="white", linewidth=0.2, label=_component_label(score_id) if score_id.startswith("rail_") else None)
            pos += values
        else:
            ax.barh(y, values, height=0.55, left=neg, color=_component_color(score_id),
                    edgecolor="white", linewidth=0.2,
                    label=_component_label(score_id) if score_id.startswith("rail_") else None)
            neg += values

    ax.set_yticks(y)
    ax.set_yticklabels(pivot["ranking_label_short"], fontsize=font_size + 6)
    ax.tick_params(axis="x", labelsize=font_size + 4)
    ax.set_xlabel("Annual value [Mio. CHF/year]", fontsize=font_size + 6)
    ax.set_ylabel("Development ID", fontsize=font_size + 6)
    ax.set_title(
        f"Top {top_n_per_mode} integrated benefit-cost ratios by mode",
        fontsize=font_size + 4,
    )
    ax.invert_yaxis()
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

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
        fontsize=font_size + 1,
    )
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(output_path, dpi=200, transparent=True)
    plt.close(fig)


def plot_weighted_tts_mean_std(output_path: Path) -> None:
    df = _build_tts_summary_by_development()
    rail_labels = _rail_label_lookup()
    fig, axes = plt.subplots(1, 2, figsize=(20, 18), sharex=False)

    road_all = df[df["mode"] == "Road"].copy()
    rail_all = df[df["mode"] == "Rail"].copy()
    shared_positive_xmax = max(
        float((road_all["mean_tts_person_h_per_year"] + road_all["std_tts_person_h_per_year"].fillna(0.0)).max()) if not road_all.empty else 0.0,
        float((rail_all["mean_tts_person_h_per_year"] + rail_all["std_tts_person_h_per_year"].fillna(0.0)).max()) if not rail_all.empty else 0.0,
    )
    shared_positive_xmax = max(shared_positive_xmax * 1.05, 1.0)

    for ax, mode in zip(axes, ["Road", "Rail"]):
        sub = df[df["mode"] == mode].copy()
        sub = sub.sort_values("mean_tts_person_h_per_year", ascending=False).reset_index(drop=True)
        y = np.arange(len(sub))
        lower = sub["mean_tts_person_h_per_year"] - sub["std_tts_person_h_per_year"].fillna(0.0)
        upper = sub["mean_tts_person_h_per_year"] + sub["std_tts_person_h_per_year"].fillna(0.0)
        ax.hlines(y, lower, upper,
                  color="#8EC5E8", linewidth=2)
        ax.scatter(sub["mean_tts_person_h_per_year"], y, color="#0E5A9C", s=24, zorder=3)
        ax.axvline(0, color="black", linestyle="--", linewidth=0.9)
        xmin = float(lower.min()) if not sub.empty else 0.0
        pad = max(shared_positive_xmax * 0.05, 1.0)
        ax.set_xlim(xmin - pad, shared_positive_xmax)
        ax.set_yticks(y)
        if mode == "Rail":
            ylabels = [rail_labels.get(dev, dev) for dev in sub["development"].astype(str)]
        else:
            ylabels = sub["development"].astype(str).tolist()
        ax.set_yticklabels(ylabels, fontsize=7 if mode == "Road" else 8)
        ax.invert_yaxis()
        ax.set_xlabel("Demand-weighted travel time savings [person-h/year]")
        ax.set_ylabel("Development ID")
        ax.set_title(f"{mode}: mean TTS with standard deviation across scenarios")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
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
    palette = {"Rail": "#B070AF", "Road": "#6FB2DE"}

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
        color = "black",
        alpha=0.5,
        size=2.4,
        zorder=5,
        ax=ax,
    )

    ax.axhline(1, color="black", linewidth=1.0)
    ax.set_title("Development-level CBA ratio by cross-modal VTT sources")
    ax.set_xlabel("VTTS (CHF/hour)")
    ax.set_ylabel("CBA ratio")
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

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

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
    ax.set_xticklabels(combined["label"], rotation=90)
    ax.set_title("Costs and benefits of development alternatives over all scenarios", fontsize=14, pad=20)
    ax.set_xlabel("Development ID", fontsize=10, labelpad=20)
    ax.set_ylabel("Average total value over all scenarios [Mio. CHF]", fontsize=10)
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
    - Rail: demand-weighted annual TTS before monetization
    - Road: demand-weighted annual TTS before monetization
    """
    if output_dir is None:
        output_dir = GENERATED_PLOTS_DIR
    output_path = Path(output_dir) / "combined_top5_tts_boxplot.png"
    
    selection_df = (
        _build_integrated_bcr_top10_by_mode_plot_data(5)[["mode", "development", "plot_order", "ranking_label_short", "bcr_mean"]]
        .drop_duplicates()
        .sort_values("bcr_mean", ascending=False)
    )
    selection_df["development"] = selection_df["development"].astype(str)
    rail_top = selection_df.loc[selection_df["mode"] == "Rail", "development"].tolist()
    road_top = selection_df.loc[selection_df["mode"] == "Road", "development"].tolist()
    display_labels = {
        f"{row.mode} {row.development}": str(row.ranking_label_short)
        for row in selection_df.itertuples(index=False)
    }
    
    
    data = build_weighted_tts_by_scenario(rail_top, road_top).copy()
    if data.empty:
        return output_path
    data["label"] = data["mode"] + " " + data["development"].astype(str)
    order = [f"{row.mode} {row.development}" for row in selection_df.itertuples(index=False)]
    order = [label for label in order if label in set(data["label"])]
    palette = {label: "#fff3b0" if label.startswith("Rail") else "#e09f3e" for label in order}

    fig, ax = plt.subplots(figsize=(18, 6), dpi=300)

    sns.boxplot(
        data=data,
        x="label",
        y="weighted_tts_person_h_per_year",
        order=order,
        palette=palette,
        linewidth=0.9,
        showfliers=False,
        width=0.58,
        medianprops={"color": "black", "linewidth": 2.0},
        ax=ax,
    )

    ax.axhline(y=ax.get_ylim()[0], color="0.7", linestyle="--", alpha=0.7)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([display_labels.get(label, label.split(" ", 1)[1]) for label in order], rotation=90)
    ax.set_title("Distribution of demand-weighted travel time savings for developments selected in the integrated BCR ranking", fontsize=14, pad=20)
    ax.set_xlabel("Development ID", fontsize=10, labelpad=20)
    ax.set_ylabel("Demand-weighted travel time savings [person-h/year]", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    handles = [
        mpatches.Patch(color="#fff3b0", label="Rail selected scenarios"),
        mpatches.Patch(color="#e09f3e", label="Road selected scenarios"),
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_path, dpi=600)
    plt.close(fig)
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

    plot_mode_standalone_vs_integrated(
        mode="Road",
        output_path=GENERATED_PLOTS_DIR / "road_stacked_integrated_vs_standalone_annual.png",
        max_developments=30,
    )
    plot_mode_standalone_vs_integrated(
        mode="Rail",
        output_path=GENERATED_PLOTS_DIR / "rail_stacked_integrated_vs_standalone_annual.png",
    )
    plot_integrated_bcr_top10_by_mode(
        output_path=GENERATED_PLOTS_DIR / "integrated_bcr_top5_by_mode_stacked.png",
        top_n_per_mode=5,
    )
    plot_weighted_tts_mean_std(
        output_path=GENERATED_PLOTS_DIR / "weighted_tts_mean_std_by_mode.png",
    )
    # plot_combined_top5_final_cost_savings(GENERATED_PLOTS_DIR)
    # plot_all_rail_final_cost_savings(GENERATED_PLOTS_DIR)
    # plot_all_road_final_cost_savings(GENERATED_PLOTS_DIR)
    plot_final_tts_boxplot_top5(GENERATED_PLOTS_DIR)
    plot_vtt_ratio_violin(
        output_path=GENERATED_PLOTS_DIR / "vtt_ratio_violin_by_mode.png",
    )
    # plot_vtt_ratio_violin_mean_scenario_ratios(
    #     output_path=GENERATED_PLOTS_DIR / "vtt_ratio_violin_by_mode_mean_scenario_ratios.png",
    # )
    # plot_vtt_ratio_violin_by_scenario(
    #     output_path=GENERATED_PLOTS_DIR / "vtt_ratio_violin_by_mode_scenarios.png",
    # )
    # plot_externality_per_km_violin(
    #     externality_comparison,
    #     output_path=GENERATED_PLOTS_DIR / "externality_per_km_violin.png",
    # )
    plot_externality_total_vs_new_link_km(
        externality_comparison,
        output_path=GENERATED_PLOTS_DIR / "externality_total_vs_new_link_km.png",
    )
    # plot_noise_vs_settlement_exposed_km(
    #     externality_comparison,
    #     output_path=GENERATED_PLOTS_DIR / "noise_vs_settlement_exposed_km.png",
    # )

    print("Saved plots to:", GENERATED_PLOTS_DIR)


if __name__ == "__main__":
    main()
