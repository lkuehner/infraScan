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

from infraScan.infraScanRail import paths as rail_paths
from infraScan.infraScanRail.TT_Delay import analyze_travel_times

SCENARIOS = ("scenario_76", "scenario_45", "scenario_67")
RAIL_COMPARISON_YEAR = 2050

COST_COLORS = {
    "construction": "#a6bddb",
    "maintenance": "#3690c0",
    "operating": "#1f5a89",
    "externalities": "#092245",
    "tts": "#efdd74",
}

# Centralized paths (adjust these at top to configure where data and outputs live)
DATA_ROOT = Path(rail_paths.MAIN)
COST_OUTPUT_DIR = DATA_ROOT / "plots" / "Integrated" / "CBA_Comparison"
TTS_OUTPUT_DIR = DATA_ROOT / "plots" / "Integrated" / "TTS_Comparison"

RAIL_COSTS_PATH = DATA_ROOT / "data" / "infraScanRail" / "costs"
ROAD_COSTS_PATH = DATA_ROOT / "data" / "infraScanRoad" / "costs"
RAIL_NETWORK_PATH = DATA_ROOT / "data" / "infraScanRail" / "Network"
ROAD_NETWORK_PATH = DATA_ROOT / "data" / "infraScanRoad" / "Network"

RAIL_TOTAL_COSTS_CSV = RAIL_COSTS_PATH / "total_costs.csv"
RAIL_TT_SAVINGS_CSV = RAIL_COSTS_PATH / "traveltime_savings.csv"
ROAD_TOTAL_COSTS_CSV = ROAD_COSTS_PATH / "total_costs_od.csv"
ROAD_TT_OD_CSV = ROAD_COSTS_PATH / "traveltime_savings_od.csv"
ROAD_TT_DETAILED_CSV = DATA_ROOT / "data" / "infraScanRoad" / "traffic_flow" / "od" / "od_tt_savings_detailed.csv"

RAIL_TRAVELTIME_CACHE = RAIL_NETWORK_PATH / "travel_time" / "cache" / "od_times.pkl"
RAIL_TRAVELTIME_SAVINGS_DIR = RAIL_NETWORK_PATH / "travel_time" / "TravelTime_Savings"
ROAD_TRAVELTIME_RASTER = ROAD_NETWORK_PATH / "travel_time" / "travel_time_raster.tif"
DEV_DIR = DATA_ROOT / rail_paths.DEVELOPMENT_DIRECTORY





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
    construction = gpd.read_file(ROAD_COSTS_PATH / "construction.gpkg")[["ID_new", "building_costs"]]
    maintenance = gpd.read_file(ROAD_COSTS_PATH / "maintenance.gpkg")[["ID_new", "maintenance"]]
    externalities = gpd.read_file(ROAD_COSTS_PATH / "externalities.gpkg")[["ID_new", "climate_cost", "land_realloc", "nature"]]
    noise = gpd.read_file(ROAD_COSTS_PATH / "noise.gpkg")[["ID_new", "noise_s1"]]

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
    combined_cost_csv = COST_OUTPUT_DIR / "rail_road_final_costs_total.csv"
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


def load_rail_od_savings_top(rail_top: list[str]) -> pd.DataFrame:
    """Load rail OD-level travel time savings using analyze_travel_times."""
    workspace_root = Path(__file__).resolve().parents[2]
    lookup = create_rail_dev_id_lookup_table()

    cache_path = RAIL_TRAVELTIME_CACHE
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)

    dev_id_to_position = {
        str(dev_id): idx
        for idx, dev_id in enumerate(lookup["dev_id"].astype(str).tolist())
    }
    selected_positions = [dev_id_to_position[dev_id] for dev_id in rail_top]
    selected_od_times = [cache["od_times_dev"][idx] for idx in selected_positions]
    selected_lookup = pd.DataFrame({"dev_id": rail_top}, index=range(1, len(rail_top) + 1))

    original_cwd = Path.cwd()
    try:
        os.chdir(workspace_root)
        analyze_travel_times(
            od_times_status_quo=cache["od_times_status_quo"],
            od_times_dev=selected_od_times,
            od_nodes=list(cache["od_times_status_quo"][0]["from_station"].unique()),
            dev_id_lookup_table=selected_lookup,
        )
    finally:
        os.chdir(original_cwd)

    savings_dir = RAIL_TRAVELTIME_SAVINGS_DIR
    frames: list[pd.DataFrame] = []
    for dev_id in rail_top:
        csv_path = savings_dir / f"TravelTime_Savings_Dev_{dev_id}.csv"
        df = pd.read_csv(csv_path)
        df["development"] = str(dev_id)
        df["mode"] = "Rail"
        df["scenario"] = "all_selected_scenarios"
        # In analyze_travel_times, savings are status quo minus development time.
        df["tts_minutes"] = pd.to_numeric(df["status_quo_time"], errors="coerce") - pd.to_numeric(
            df["new_time"], errors="coerce"
        )
        # For the final comparison plot, keep only affected OD relations.
        df = df[np.isfinite(df["tts_minutes"]) & (np.abs(df["tts_minutes"]) > 1e-9)].copy()
        frames.append(df[["mode", "development", "scenario", "origin", "destination", "tts_minutes"]])

    return pd.concat(frames, ignore_index=True)


def load_road_affected_cell_savings_top(road_top: list[str]) -> pd.DataFrame:
    """Load road raster-based affected-cell travel time savings."""
    with rasterio.open(ROAD_TRAVELTIME_RASTER) as src:
        sq_tt = src.read(1).astype(float)

    frames: list[pd.DataFrame] = []
    for dev_id in road_top:
        with rasterio.open(ROAD_NETWORK_PATH / "travel_time" / "developments" / f"dev{dev_id}_travel_time_raster.tif") as src:
            dev_tt = src.read(1).astype(float)
        with rasterio.open(ROAD_NETWORK_PATH / "travel_time" / "developments" / f"dev{dev_id}_source_id_raster.tif") as src:
            dev_source_id = src.read(1)

        delta_min = (sq_tt - dev_tt) / 60.0
        affected_mask = dev_source_id == 9999
        affected_values = delta_min[affected_mask]
        affected_values = affected_values[np.isfinite(affected_values)]

        frames.append(
            pd.DataFrame(
                {
                    "mode": "Road",
                    "development": str(dev_id),
                    "scenario": "affected_cells",
                    "tts_minutes": affected_values,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def load_od_level_tts_top(rail_top: list[str], road_top: list[str]) -> pd.DataFrame:
    """Combine rail OD savings and road affected-cell savings for final boxplot comparison."""
    rail = load_rail_od_savings_top(rail_top)
    road = load_road_affected_cell_savings_top(road_top)
    return pd.concat([rail, road], ignore_index=True, sort=False)



def combined_order(data: pd.DataFrame) -> list[str]:
    """Order developments by mean TTS, rail first then road."""
    rail_order = (
        data[data["mode"] == "Rail"]
        .groupby("development")["tts_minutes"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    road_order = (
        data[data["mode"] == "Road"]
        .groupby("development")["tts_minutes"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return [f"Rail {dev}" for dev in rail_order] + [f"Road {dev}" for dev in road_order]

    

 


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
    plt.savefig(COST_OUTPUT_DIR / "combined_top5_final_cost_savings.png", dpi=600)
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
    - Rail: affected OD savings from analyze_travel_times
    - Road: raster-based affected-cell savings
    """
    if output_dir is None:
        output_dir = TTS_OUTPUT_DIR
    output_path = Path(output_dir) / "combined_top5_tts_boxplot.png"
    
    # Get top 5 developments from costs
    rail_data = load_rail_final_costs_from_sources()
    road_data = load_road_final_costs_from_sources()
    
    rail_top = (
        rail_data.groupby(["development", "line_name"], as_index=False)
        .agg({"net_benefit_mio_chf": "mean"})
        .sort_values("net_benefit_mio_chf", ascending=False)
        # Get the top 5 
        .head(5)["development"].astype(str).tolist()
    )
    road_top = (
        road_data.groupby(["development", "line_name"], as_index=False)
        .agg({"net_benefit_mio_chf": "mean"})
        .sort_values("net_benefit_mio_chf", ascending=False)
        # Get the top 5 
        .head(5)["development"].astype(str).tolist()
    )
    
    # Get display labels for top 5 developments
    display_labels: dict[str, str] = {}
    rail_top_df = (
        rail_data[rail_data["development"].astype(str).isin(rail_top)]
        .drop_duplicates(["development", "line_name"]) 
    )
    for _, row in rail_top_df.iterrows():
        display_labels[f"Rail {str(row['development'])}"] = str(row["line_name"])
    road_top_df = (
        road_data[road_data["development"].astype(str).isin(road_top)]
        .drop_duplicates(["development", "line_name"]) 
    )
    for _, row in road_top_df.iterrows():
        display_labels[f"Road {str(row['development'])}"] = str(row["line_name"])
    
    
    # Load OD-level TTS data (rail from analyze_travel_times, road from rasters)
    data = load_od_level_tts_top(rail_top, road_top).copy() 
    data["label"] = data["mode"] + " " + data["development"].astype(str)
    order = combined_order(data)
    palette = {label: "#fff3b0" if label.startswith("Rail") else "#e09f3e" for label in order}

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

    sns.boxplot(
        data=data,
        x="label",
        y="tts_minutes",
        order=order,
        palette=palette,
        linewidth=0.9,
        showfliers=False,
        width=0.58,
        medianprops={"color": "black", "linewidth": 2.0},
        ax=ax,
    )

    ax.axvline(x=4.5, color="black", linestyle="-", alpha=0.7, linewidth=0.5)
    ax.axhline(y=ax.get_ylim()[0], color="0.7", linestyle="--", alpha=0.7)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([display_labels.get(label, label.split(" ", 1)[1]) for label in order], rotation=90)
    ax.set_title("Distribution of travel time savings for top development alternatives", fontsize=14, pad=20)
    ax.set_xlabel("Development ID", fontsize=10, labelpad=20)
    ax.set_ylabel("Travel time savings [minutes]", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ax.text(2, ax.get_ylim()[1] + 0.1, "Rail top 5", ha="center", va="top", fontsize=11)
    ax.text(7, ax.get_ylim()[1] + 0.1, "Road top 5", ha="center", va="top", fontsize=11)

    handles = [
        mpatches.Patch(color="#fff3b0", label="Rail affected OD relations"),
        mpatches.Patch(color="#e09f3e", label="Road raster-cell savings"),
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
    COST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    # create_combined_cost_csv() is optional — plots load sources directly
    plot_combined_top5_final_cost_savings(COST_OUTPUT_DIR)
    plot_final_tts_boxplot_top5(TTS_OUTPUT_DIR)

    # rail and road separately 
    plot_all_rail_final_cost_savings(COST_OUTPUT_DIR)
    plot_all_road_final_cost_savings(COST_OUTPUT_DIR)

    print("Saved plots to:", COST_OUTPUT_DIR)


if __name__ == "__main__":
    main()
