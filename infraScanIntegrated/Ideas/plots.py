from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.patches as mpatches
try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

from infraScan.infraScanRail import paths as rail_paths


SCENARIOS = ("scenario_76", "scenario_45", "scenario_67")
RAIL_COMPARISON_YEAR = 2050
DATA_ROOT = Path(rail_paths.MAIN)
COST_OUTPUT_DIR = DATA_ROOT / "plots" / "Integrated" / "CBA_Comparison"
TTS_OUTPUT_DIR = DATA_ROOT / "plots" / "Integrated" / "TTS_Comparison"

"""
RAIL_COST_COLUMNS = [ "development", "Sline", "Net Benefit Scenario 76 [in Mio. CHF]", "Net Benefit Scenario 45 [in Mio. CHF]",
                      "Net Benefit Scenario 67 [in Mio. CHF]", "Monetized Savings Scenario 76 [in Mio. CHF]", "Monetized Savings Scenario 45 [in Mio. CHF]",
                      "Monetized Savings Scenario 67 [in Mio. CHF]", "Construction Cost [in Mio. CHF]", "Maintenance Costs [in Mio. CHF]", "Uncovered Operating Costs [in Mio. CHF]",
                 ]

ROAD_COST_TOTAL_COLUMNS = ["ID_new", "total_scenario_76", "total_scenario_45", "total_scenario_67"]
ROAD_TTS_DETAIL_COLUMNS = ["origin", "destination", "demand", "travel_time", "scenario", "development"]


"""
RAIL_TTS_COLUMNS = ["development", "scenario", "year", "tt_savings_daily"]
ROAD_TTS_COLUMNS = ["development", "scenario", "tt_savings_peak"]
# ------------------------------------------
# Helper Functions
# ------------------------------------------


def _load_csv(path: Path, columns: list[str], source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")
    df = pd.read_csv(path)
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")
    return df

ZVV_COLORS = [
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#EECA3B",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
]

def _line_colors(order: list[str]) -> dict[str, str]:
    return {line_name: ZVV_COLORS[i % len(ZVV_COLORS)] for i, line_name in enumerate(order)}



def _line_name_from_rail_row(row: pd.Series) -> str:
    development = int(str(row["development"]).removeprefix("Development_"))
    sline = str(row["Sline"])
    if development < 101000:
        return f"{development - 100000}_{sline}"
    return f"X{development - 101000}"


def _normalize_development_id(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.strip()
    return normalized.str.replace(r"\.0$", "", regex=True)

def _ordered_line_names(data: pd.DataFrame) -> list[str]:
    return (
        data.groupby("line_name")["net_benefit_mio_chf"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )



# ------------------------------------------
# Data loading and transformation functions
# ------------------------------------------

def load_rail_final_costs_from_sources(scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    path = DATA_ROOT / "data" / "infraScanRail" / "costs" / "total_costs.csv"
    df = pd.read_csv(path)

    df["development_id"] = df["development"].astype(str).str.removeprefix("Development_").astype(int)
    df["line_name"] = df.apply(_line_name_from_rail_row, axis=1)

    rows = []
    for scenario in scenarios:
        suffix = scenario.split("_")[-1]
        rows.append(
            pd.DataFrame(
                {
                    "mode": "Rail",
                    "development": _normalize_development_id(df["development_id"]),
                    "line_name": df["line_name"],
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
    if gpd is None:
        raise ImportError("geopandas is required to load road final-cost sources.")

    totals_path = pd.read_csv(DATA_ROOT / "data" / "infraScanRoad" / "costs" / "total_costs_od.csv")
    tt_path = pd.read_csv(DATA_ROOT / "data" / "infraScanRoad" / "costs" / "traveltime_savings_od.csv")
    construction = gpd.read_file(DATA_ROOT / "data" / "infraScanRoad" / "costs" / "construction.gpkg")[["ID_new", "building_costs"]]
    maintenance = gpd.read_file(DATA_ROOT / "data" / "infraScanRoad" / "costs" / "maintenance.gpkg")[["ID_new", "maintenance"]]
    externalities = gpd.read_file(DATA_ROOT / "data" / "infraScanRoad" / "costs" / "externalities.gpkg")[["ID_new", "climate_cost", "land_realloc", "nature"]]
    noise = gpd.read_file(DATA_ROOT / "data" / "infraScanRoad" / "costs" / "noise.gpkg")[["ID_new", "noise_s1"]]

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
        scenario_df["development"] = _normalize_development_id(scenario_df["ID_new"])
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

def create_combined_cost_csvs() -> Path:
    long_csv = COST_OUTPUT_DIR / "rail_road_final_costs_long.csv"
    if long_csv.exists():
        return long_csv
    combined = pd.concat([load_rail_final_costs_from_sources(), load_road_final_costs_from_sources()], ignore_index=True, sort=False)
    combined.to_csv(long_csv, index=False)
    return long_csv


def load_rail_tts_from_sources(year: int = RAIL_COMPARISON_YEAR, scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    path = DATA_ROOT / "data" / "infraScanRail" / "costs" / "traveltime_savings.csv"
    df = pd.read_csv(path)
    df = df[df["scenario"].isin(list(scenarios)) & (df["year"] == year)].copy()
    df["mode"] = "Rail"
    df["development"] = _normalize_development_id(df["development"])
    df["tt_savings_daily_minutes"] = df["tt_savings_daily"].astype(float)
    return df[["mode", "development", "scenario", "tt_savings_daily_minutes"]]



def load_road_tts_from_sources(scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    path = DATA_ROOT / "data" / "infraScanRoad" / "traffic_flow" / "od" / "od_tt_savings_detailed.csv"
    df = pd.read_csv(path)
    df = df[df["scenario"].isin(list(scenarios))].copy()
    df["mode"] = "Road"
    df["development"] = _normalize_development_id(df["development"])
    df["tt_savings_daily_minutes"] = df["tt_savings_peak"].astype(float)
    return df[["mode", "development", "scenario", "tt_savings_daily_minutes"]]

def create_combined_tts_csv() -> Path:
    long_csv = TTS_OUTPUT_DIR / "rail_road_tts_minutes_long.csv"
    if long_csv.exists():
        return long_csv
    combined = pd.concat([load_rail_tts_from_sources(), load_road_tts_from_sources()], ignore_index=True)
    combined.to_csv(long_csv, index=False)
    return long_csv

# ------------------------------------------
# Plotting functions
# ------------------------------------------

def plot_combined_top5_final_cost_savings(cost_csv: Path, out_dir: Path) -> None:
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
    cost_colors = {
        "construction": "#a6bddb",
        "maintenance": "#3690c0",
        "operating": "#1f5a89",
        "externalities": "#092245",
        "tts": "#f8eeb3",
    }

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

    ax.bar(
        x_pos,
        -combined["construction_cost_mio_chf"],
        width=bar_width,
        color=cost_colors["construction"],
        label="Construction costs",
    )
    ax.bar(
        x_pos,
        -combined["maintenance_cost_mio_chf"],
        width=bar_width,
        bottom=-combined["construction_cost_mio_chf"],
        color=cost_colors["maintenance"],
        label="Maintenance costs",
    )
    ax.bar(
        x_pos,
        -combined["uncovered_operating_cost_mio_chf"],
        width=bar_width,
        bottom=-(combined["construction_cost_mio_chf"] + combined["maintenance_cost_mio_chf"]),
        color=cost_colors["operating"],
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
        color=cost_colors["externalities"],
        label="Externalities",
    )

    ax.bar(
        x_pos,
        combined["monetized_savings_mio_chf"],
        width=bar_width,
        color=cost_colors["tts"],
        label="Travel time savings",
    )

    ax.axhline(y=0, color="black", linestyle="-")
    ax.axhline(y=ax.get_ylim()[0],  color="0.7", linestyle="--", alpha=0.7)
    ax.axvline(x=4.5, color="black", linestyle="-", alpha=0.7, linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(combined["label"], rotation=90)
    ax.set_title("Costs and benefits of development alternatives over all scenarios", fontsize=14, pad=20)
    ax.set_xlabel("Development ID", fontsize=10, labelpad=20)
    ax.set_ylabel("Average total costs over all scenarios [Mio. CHF]", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ax.text(2, ax.get_ylim()[1] * 0.95, "Rail top 5", ha="center", fontsize=11)
    ax.text(7, ax.get_ylim()[1] * 0.95, "Road top 5", ha="center", fontsize=11)

    handles = [
        mpatches.Patch(color=cost_colors["construction"], label="Construction costs"),
        mpatches.Patch(color=cost_colors["maintenance"], label="Maintenance costs"),
        mpatches.Patch(color=cost_colors["operating"], label="Uncovered operating costs"),
        mpatches.Patch(color=cost_colors["externalities"], label="Externalities"),
        mpatches.Patch(color=cost_colors["tts"], label="Travel time savings"),
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(COST_OUTPUT_DIR / "combined_top5_final_cost_savings.png", dpi=600)
    plt.close()



def plot_combined_top5_final_tts_boxplot(tts_csv: Path, cost_csv: Path, out_dir: Path) -> None:
    tt = pd.read_csv(tts_csv)
    tt["development"] = _normalize_development_id(tt["development"])

    cost = pd.read_csv(cost_csv)
    cost["development"] = _normalize_development_id(cost["development"])
    top5 = (
        cost.groupby("development")["net_benefit_mio_chf"].mean().abs().sort_values(ascending=False).head(5).index.tolist()
    )
    sub = tt[tt["development"].isin(top5)]
    if sub.empty:
        raise ValueError(
            "No TTS rows matched the selected top-5 developments. "
            "Check whether development IDs are stored consistently across the cached cost and TTS CSVs."
        )
    order = sub.groupby("development")["tt_savings_daily_minutes"].median().abs().sort_values(ascending=False).index.tolist()

    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=sub, x="development", y="tt_savings_daily_minutes", hue="mode", order=order)
    ax.set_title("Top 5 developments — TTS minutes distribution")
    ax.set_xlabel("Development")
    ax.set_ylabel("TTS change (minutes)")
    plt.tight_layout()
    plt.savefig(out_dir / "combined_top5_final_tts_boxplot.png", dpi=300)
    plt.close()


def plot_road_final_cost_savings(output_dir: Path) -> None:
    """Road top 5 with stacked costs and colored TTS bars per development."""
    road_data = load_road_final_costs_from_sources()
    
    order = _ordered_line_names(road_data)
    line_colors = _line_colors(order)
    summary = (
        road_data.groupby(["development", "line_name"], as_index=False)
        .agg({
            "construction_cost_mio_chf": "mean",
            "maintenance_cost_mio_chf": "mean",
            "other_cost_mio_chf": "mean",
            "monetized_savings_mio_chf": "mean",
            "net_benefit_mio_chf": "mean",
        })
        .set_index("line_name")
        .loc[order]
        .reset_index()
    )

    x_pos = np.arange(len(order))
    bar_width = 0.6
    cost_colors = {
        "construction": "#a6bddb",
        "maintenance": "#3690c0",
        "other": "#034e7b",
    }

    plt.figure(figsize=(max(7, len(order) * 0.62), 5), dpi=300)
    plt.bar(x_pos, -summary["construction_cost_mio_chf"], width=bar_width, color=cost_colors["construction"], label="Construction costs")
    plt.bar(x_pos, -summary["maintenance_cost_mio_chf"], width=bar_width, bottom=-summary["construction_cost_mio_chf"],
            color=cost_colors["maintenance"], label="Maintenance costs")
    plt.bar(x_pos, -summary["other_cost_mio_chf"], width=bar_width,
            bottom=-(summary["construction_cost_mio_chf"] + summary["maintenance_cost_mio_chf"]),
            color=cost_colors["other"], label="Externalities")

    for i, line_name in enumerate(order):
        plt.bar(x_pos[i], summary.loc[summary["line_name"] == line_name, "monetized_savings_mio_chf"].iloc[0],
                width=bar_width, color=line_colors[line_name], hatch="////", edgecolor="black")

    plt.axhline(y=0, color="black", linestyle="-")
    plt.xticks(x_pos, summary["line_name"], rotation=90)
    plt.xlabel("Development ID", fontsize=12)
    plt.ylabel("Value [Mio. CHF]", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    handles = [
        mpatches.Patch(color=cost_colors["construction"], label="Construction costs"),
        mpatches.Patch(color=cost_colors["maintenance"], label="Maintenance costs"),
        mpatches.Patch(color=cost_colors["other"], label="Externalities"),
        mpatches.Patch(facecolor="none", hatch="////", edgecolor="black", label="Travel time savings"),
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / "road_final_cost_savings.png", dpi=600)
    plt.close()

def plot_rail_final_cost_savings(output_dir: Path) -> None:
    """Rail top 5 with stacked costs and colored TTS bars per line."""
    rail_data = load_rail_final_costs_from_sources()
    
    order = _ordered_line_names(rail_data)
    line_colors = _line_colors(order)
    summary = (
        rail_data.groupby(["development", "line_name"], as_index=False)
        .agg({
            "construction_cost_mio_chf": "mean",
            "maintenance_cost_mio_chf": "mean",
            "uncovered_operating_cost_mio_chf": "mean",
            "monetized_savings_mio_chf": "mean",
            "net_benefit_mio_chf": "mean",
        })
        .set_index("line_name")
        .loc[order]
        .reset_index()
    )

    x_pos = np.arange(len(order))
    bar_width = 0.6
    cost_colors = {
        "construction": "#a6bddb",
        "maintenance": "#3690c0",
        "operating": "#034e7b",
    }

    plt.figure(figsize=(max(7, len(order) * 0.62), 5), dpi=300)
    plt.bar(x_pos, -summary["construction_cost_mio_chf"], width=bar_width, color=cost_colors["construction"], label="Construction costs")
    plt.bar(x_pos, -summary["maintenance_cost_mio_chf"], width=bar_width, bottom=-summary["construction_cost_mio_chf"],
            color=cost_colors["maintenance"], label="Maintenance costs")
    plt.bar(x_pos, -summary["uncovered_operating_cost_mio_chf"], width=bar_width,
            bottom=-(summary["construction_cost_mio_chf"] + summary["maintenance_cost_mio_chf"]),
            color=cost_colors["operating"], label="Uncovered operating costs")

    for i, line_name in enumerate(order):
        plt.bar(x_pos[i], summary.loc[summary["line_name"] == line_name, "monetized_savings_mio_chf"].iloc[0],
                width=bar_width, color=line_colors[line_name], hatch="////", edgecolor="black")

    plt.axhline(y=0, color="black", linestyle="-")
    plt.xticks(x_pos, summary["line_name"], rotation=90)
    plt.xlabel("Line", fontsize=12)
    plt.ylabel("Value [Mio. CHF]", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    handles = [
        mpatches.Patch(color=cost_colors["construction"], label="Construction costs"),
        mpatches.Patch(color=cost_colors["maintenance"], label="Maintenance costs"),
        mpatches.Patch(color=cost_colors["operating"], label="Uncovered operating costs"),
        mpatches.Patch(facecolor="none", hatch="////", edgecolor="black", label="Travel time savings"),
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / "rail_final_cost_savings.png", dpi=600)
    plt.close()



# ------------------------------------------
# Main execution
# ------------------------------------------
 


def main() -> None:
    COST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cost_csv = create_combined_cost_csvs()
    tts_csv = create_combined_tts_csv()

    plot_combined_top5_final_cost_savings(cost_csv, COST_OUTPUT_DIR)
    plot_combined_top5_final_tts_boxplot(tts_csv, cost_csv, TTS_OUTPUT_DIR)

    # rail and road separately 
    plot_rail_final_cost_savings(COST_OUTPUT_DIR)
    plot_road_final_cost_savings(COST_OUTPUT_DIR)

    print("Saved plots to:", COST_OUTPUT_DIR, TTS_OUTPUT_DIR)


if __name__ == "__main__":
    main()
