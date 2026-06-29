"""
Validate selected road link flows against ASTRA traffic counts.

This script follows the same matching logic as `link_traffic_to_map`
in `Scouring.py`, but applies the station matching automatically to
a larger set of validation links across multiple active scenarios.

link_traffic_to_map:    to assign a share of OD-based model flow
                        to corridor links as a proxy for highway traffic
flow_validation:        to validate modelled link flows against ASTRA counts
                        in the assessment area                     
"""


from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd

import settings


INPUT_DIR_DEFAULT = Path("/Volumes/WD_Windows/draft/data/infraScanRoad/traffic_flow/link_flows_2025")
ASTRA_GPKG_DEFAULT = Path("/Volumes/WD_Windows/MSc_Thesis/data/infraScanRoad/traffic_flow/ASTRA_traffic_count2025.gpkg")
EDGES_GPKG_DEFAULT = Path("/Volumes/WD_Windows/MSc_Thesis/data/infraScanRoad/Network/processed/edges_with_attribute.gpkg")
OUTPUT_DIR_PLOT = Path("/Volumes/WD_Windows/draft/plots/road_standalone")

LINK_IDS = [64, 90, 96, 83, 77, 129, 175, 102, 132]
MANUAL_MLOCNR_BY_EDGE = {178: 761}  # match ID_edge 178 manual to mlocnr 761 (Uster-Hinwil)
MANUAL_NT_BY_EDGE = {90: 1087} # add the flow manually to edge 90 


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate selected road link flows against ASTRA traffic counts.")
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR_DEFAULT)
    parser.add_argument("--astra-path", type=Path, default=ASTRA_GPKG_DEFAULT)
    parser.add_argument("--edges-path", type=Path, default=EDGES_GPKG_DEFAULT)
    parser.add_argument("--output-dir-plot", type=Path, default=OUTPUT_DIR_PLOT)
    parser.add_argument("--scenario", type=str, default=None, help="Scenario number or scenario_<n>.")
    return parser.parse_args()


def normalize_scenario_name(value: str) -> str:
    match = re.search(r"(\d+)$", str(value))
    if not match:
        raise ValueError(f"Could not parse scenario from '{value}'.")
    return f"scenario_{int(match.group(1))}"


def scenario_number(value: str) -> int:
    return int(normalize_scenario_name(value).split("_")[1])


def get_active_scenarios() -> list[str]:
    return [normalize_scenario_name(scen) for scen in settings.get_travel_time_debug_scenarios()]


def discover_available_scenarios(input_dir: Path) -> list[str]:
    scenarios = set()
    for path in input_dir.glob("dev0_scenario_*.csv"):
        match = re.search(r"dev0_(scenario_\d+)\.csv$", path.name)
        if match:
            scenarios.add(match.group(1))
    return sorted(scenarios, key=scenario_number)


def resolve_scenarios(input_dir: Path, scenario_arg: str | None) -> list[str]:
    active_scenarios = set(get_active_scenarios())
    available_scenarios = discover_available_scenarios(input_dir)
    selected = [scen for scen in available_scenarios if scen in active_scenarios]
    if scenario_arg is not None:
        scenario_name = normalize_scenario_name(scenario_arg)
        if scenario_name not in active_scenarios:
            raise ValueError(f"{scenario_name} is not in the active scenario list.")
        if scenario_name not in available_scenarios:
            raise FileNotFoundError(f"No dev0 CSV found for {scenario_name} in {input_dir}.")
        return [scenario_name]
    return selected


def load_edges(edges_path: Path) -> gpd.GeoDataFrame:
    edges = gpd.read_file(edges_path)
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges[edges["ID_edge"].isin(LINK_IDS)].copy()
    # Capacity is used as an additional reference marker in the plot.
    edges["capacity"] = pd.to_numeric(edges["capacity"], errors="coerce")
    edges = edges.sort_values("ID_edge").reset_index(drop=True)
    return edges


def load_astra(astra_path: Path, crs) -> gpd.GeoDataFrame:
    astra = gpd.read_file(astra_path).to_crs(crs)
    for col in ["aindicator_direction", "aindicator_nt", "aindicator_mspw", "aindicator_aspw", "mlocnr"]:
        astra[col] = pd.to_numeric(astra[col], errors="coerce")
    return astra


def select_match_for_edge(edge_row: pd.Series, astra: gpd.GeoDataFrame) -> dict[str, object]:
    edge_id = int(edge_row["ID_edge"])
    edge_geom = edge_row.geometry

    candidates = astra.copy()
    if edge_id in MANUAL_MLOCNR_BY_EDGE:
        candidates = candidates[candidates["mlocnr"] == MANUAL_MLOCNR_BY_EDGE[edge_id]].copy()
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in buffer", category=RuntimeWarning)
            # Flat / mitre keeps the buffer parallel to the link and avoids round ends.
            search_area = edge_geom.buffer(1000, cap_style=2, join_style=2)
        candidates = candidates[candidates.geometry.within(search_area)].copy()

    # Keep only ASTRA points with at least one useful traffic-count field.
    candidates["has_count_value"] = candidates[["aindicator_nt", "aindicator_mspw", "aindicator_aspw"]].notna().any(axis=1)
    candidates = candidates[candidates["has_count_value"]].copy()
    if candidates.empty:
        row = {
            "ID_edge": edge_id,
            "mlocnr": pd.NA,
            "mlocname": pd.NA,
            "streetdesignation": pd.NA,
            "aindicator_direction": pd.NA,
            "distance_to_link_m": pd.NA,
            "aindicator_nt": pd.NA,
            "aindicator_mspw": pd.NA,
            "aindicator_aspw": pd.NA,
            "astra_peak_hour_avg": pd.NA,
        }
        if edge_id in MANUAL_NT_BY_EDGE:
            row["aindicator_nt"] = MANUAL_NT_BY_EDGE[edge_id]
        return row

    # Pick the nearest ASTRA point when several stations fall inside the buffer.
    candidates["distance_to_link_m"] = candidates.geometry.distance(edge_geom)
    candidates = candidates.sort_values(["distance_to_link_m", "mlocnr"], kind="mergesort")
    selected = candidates.iloc[0]
    return {
        "ID_edge": edge_id,
        "mlocnr": int(selected["mlocnr"]) if pd.notna(selected["mlocnr"]) else pd.NA,
        "mlocname": selected.get("mlocname"),
        "streetdesignation": selected.get("streetdesignation"),
        "aindicator_direction": selected.get("aindicator_direction"),
        "distance_to_link_m": float(selected["distance_to_link_m"]),
        "aindicator_nt": selected.get("aindicator_nt"),
        "aindicator_mspw": selected.get("aindicator_mspw"),
        "aindicator_aspw": selected.get("aindicator_aspw"),
        "astra_peak_hour_avg": (
            (selected.get("aindicator_mspw") + selected.get("aindicator_aspw")) / 2
            if pd.notna(selected.get("aindicator_mspw")) and pd.notna(selected.get("aindicator_aspw"))
            else pd.NA
        ),
    }


def build_match_table(edges: gpd.GeoDataFrame, astra: gpd.GeoDataFrame) -> pd.DataFrame:
    rows = [select_match_for_edge(edge_row, astra) for _, edge_row in edges.iterrows()]
    return pd.DataFrame(rows).sort_values("ID_edge").reset_index(drop=True)


def load_flow_table(input_dir: Path, scenario: str) -> pd.DataFrame:
    flow_csv = input_dir / f"dev0_{scenario}.csv"
    if not flow_csv.exists():
        raise FileNotFoundError(f"Missing flow CSV: {flow_csv}")
    flows = pd.read_csv(flow_csv)
    flows["ID_edge"] = flows["ID_edge"].astype(int)
    flows = flows[flows["ID_edge"].isin(LINK_IDS)].copy()
    flows = flows[["ID_edge", "flow"]].drop_duplicates(subset=["ID_edge"]).sort_values("ID_edge")
    return flows.reset_index(drop=True)


def write_scenario_output(output_dir: Path, scenario: str, scenario_table: pd.DataFrame) -> Path:
    output_path = output_dir / f"{scenario}_links_traffic_count.csv"
    output_cols = [
        "ID_edge",
        "flow",
        "mlocnr",
        "mlocname",
        "streetdesignation",
        "aindicator_direction",
        "distance_to_link_m",
        "aindicator_nt",
        "aindicator_mspw",
        "aindicator_aspw",
        "astra_peak_hour_avg",
    ]
    scenario_table[output_cols].to_csv(output_path, index=False)
    return output_path




# ---------------------------------------------------------
# Plotting functions
# --------------------------------------------------------

def build_plot_table(scenario_tables: list[pd.DataFrame]) -> pd.DataFrame:
    plot_df = pd.concat(scenario_tables, ignore_index=True)
    plot_df = plot_df[["scenario", "ID_edge", "flow", "capacity", "aindicator_nt", "astra_peak_hour_avg"]].copy()
    plot_df["_scenario_order"] = plot_df["scenario"].map(scenario_number)
    plot_df = plot_df.sort_values(["ID_edge", "_scenario_order"]).reset_index(drop=True)
    return plot_df.drop(columns="_scenario_order")


def plot_flow_ranges(plot_df: pd.DataFrame, output_path: Path) -> None:
    summary = (
        plot_df.groupby("ID_edge", as_index=False)
        .agg(
            flow_min=("flow", "min"),
            flow_max=("flow", "max"),
            capacity=("capacity", "first"),
            aindicator_nt=("aindicator_nt", "first"),
            astra_peak_hour_avg=("astra_peak_hour_avg", "first"),
        )
        .sort_values("ID_edge")
    )

    x_positions = list(range(len(summary)))
    fig, ax = plt.subplots(figsize=(18, 8))

    range_color = "#7A7A7A"
    nt_color = "#F4A261"
    peak_color = "#D62828"
    capacity_color = "#000000"
    cap_half_width = 0.28

    for xpos, row in zip(x_positions, summary.itertuples(index=False)):
        ax.vlines(xpos, row.flow_min, row.flow_max, color=range_color, linewidth=2.0, zorder=1)
        ax.hlines(row.flow_min, xpos - cap_half_width, xpos + cap_half_width, color=range_color, linewidth=2.0, zorder=1)
        ax.hlines(row.flow_max, xpos - cap_half_width, xpos + cap_half_width, color=range_color, linewidth=2.0, zorder=1)

        if pd.notna(row.aindicator_nt):
            ax.scatter(xpos - 0.06, row.aindicator_nt, color=nt_color, s=90, zorder=3)
        if pd.notna(row.astra_peak_hour_avg):
            ax.scatter(xpos + 0.06, row.astra_peak_hour_avg, color=peak_color, s=90, zorder=3)
        if pd.notna(row.capacity):
            ax.scatter(xpos, row.capacity, color=capacity_color, s=70, marker="D", zorder=3)

    ax.set_title("Comparison Szenario-Range with Traffic Counts, 2025", fontsize=20)
    ax.set_ylabel("Vehicles per Hour", fontsize=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(edge_id) for edge_id in summary["ID_edge"]], rotation=45, ha="right", fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.grid(True, axis="y", alpha=0.2)
    ax.grid(True, axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    legend_handles = [
        mlines.Line2D([], [], color=range_color, linewidth=3.0, label="Szenario-Range Modell"),
        mlines.Line2D([], [], color=nt_color, marker="o", linestyle="None", markersize=12, label="Traffic Count Hourly"),
        mlines.Line2D([], [], color=peak_color, marker="o", linestyle="None", markersize=12, label="Traffic Count Peak Hour"),
        mlines.Line2D([], [], color=capacity_color, marker="D", linestyle="None", markersize=10, label="Capacity Limit"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=16)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir_csv = args.input_dir / "processed"
    output_dir_plot = args.output_dir_plot
    output_dir_csv.mkdir(parents=True, exist_ok=True)
    output_dir_plot.mkdir(parents=True, exist_ok=True)


    scenarios = resolve_scenarios(args.input_dir, args.scenario)
    if not scenarios:
        raise ValueError(f"No active scenarios found in {args.input_dir}.")

    edges = load_edges(args.edges_path)
    astra = load_astra(args.astra_path, edges.crs)
    match_table = build_match_table(edges, astra)

    scenario_tables = []
    for scenario in scenarios:
        flows = load_flow_table(args.input_dir, scenario)
        edge_meta = edges[["ID_edge", "capacity"]].copy()
        scenario_table = flows.merge(edge_meta, on="ID_edge", how="left")
        scenario_table = scenario_table.merge(match_table, on="ID_edge", how="left")
        scenario_table["scenario"] = scenario
        scenario_tables.append(scenario_table)
        output_path = write_scenario_output(output_dir_csv, scenario, scenario_table)
        matched_count = int(scenario_table["mlocnr"].notna().sum())
        unmatched_count = int(scenario_table["mlocnr"].isna().sum())
        overridden_edges = scenario_table.loc[scenario_table["ID_edge"].isin(MANUAL_MLOCNR_BY_EDGE), "ID_edge"].tolist()
        print(
            f"{scenario}: wrote {output_path} | matched={matched_count} "
            f"unmatched={unmatched_count} overrides={overridden_edges}"
        )

    plot_df = build_plot_table(scenario_tables)
    plot_png = output_dir_plot / "links_traffic_count_plot.png"
    plot_flow_ranges(plot_df, plot_png)
    print(f"Wrote: {plot_png}")


if __name__ == "__main__":
    main()
