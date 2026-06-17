"""
VTT sensitivity analysis for integrated rail and road results.

This script reuses the integrated comparison output that already exists in
`notebook_outputs/mode_comparison_by_development_scenario.csv`. It varies the
value of travel time (VTT) for rail and road separately and rescales the
monetized travel-time component linearly:

    TTS_new = TTS_base * (VTT_new / VTT_base)

Because only the monetized TTS term changes, the adjusted net benefit is:

    NB_new = NB_base - TTS_base + TTS_new

Outputs:
- `vtt_sensitivity_summary.csv`
- `vtt_sensitivity_surface.csv`
- `vtt_sensitivity_development_summary.csv`
- `vtt_sensitivity_side_by_side.png`
- `vtt_sensitivity_3d_surface.png`
- `vtt_sensitivity_development_heatmaps.png`
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

#matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


blue_yellow = LinearSegmentedColormap.from_list(
    "blue_yellow_zero",
    [
        (0.00, "#1870C9"),  # negative: blue
        (0.49, "#2166AC"),
        (0.50, "#FFFEFE"),  # zero: clear break
        (0.51, "#FFD92F"),
        (1.00, "#F6C900"),  # positive: yellow
    ],
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infraScan.infraScanRail import cost_parameters as rail_cost_parameters
from infraScan.infraScanRoad import cost_parameters as road_cost_parameters
from infraScan.infraScanIntegrated import common_cost_parameters


INPUT_CSV = Path(__file__).resolve().parent / "notebook_outputs" / "mode_comparison_by_development_scenario.csv"
OUTPUT_DIR = Path("/Volumes/WD_Windows/MSc_Thesis/plots/Integrated/sensitivity_analysis")

MODE_CONFIG = {
    "Rail": {
        "base_vtt": float(rail_cost_parameters.VTTS),
        "range": (15.19, 25.24),
        "color": "#8D88B2",
    },
    "Road": {
        "base_vtt": float(road_cost_parameters.VTTS),
        "range": (26.85, 31.40),
        "color": "#3D79A1",
    },
}

REQUIRED_COLUMNS = {
    "development",
    "scenario",
    "mode",
    "tts_integrated",
    "net_benefit_integrated",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VTT sensitivity analysis for integrated road and rail outputs.")
    parser.add_argument("--input", type=Path, default=INPUT_CSV, help="Path to mode comparison CSV.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR, help="Directory for plots and CSV outputs.")
    parser.add_argument("--grid-size", type=int, default=41, help="Number of VTT points per mode.")
    parser.add_argument(
        "--road-costs-dir",
        type=Path,
        default=None,
        help="Optional external infraScanRoad costs directory used to refresh the road rows before plotting.",
    )
    parser.add_argument(
        "--write-refreshed-input",
        action="store_true",
        help="Write the merged comparison CSV with refreshed road rows into the output directory.",
    )
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Generate only the selected filtered outputs for the current road/rail review.",
    )
    parser.add_argument(
        "--exclude-road-developments",
        type=int,
        nargs="*",
        default=[],
        help="Road development IDs to exclude from the selected analysis.",
    )
    parser.add_argument(
        "--top-road-developments",
        type=int,
        default=40,
        help="Number of top-ranked road developments to keep in the selected road heatmap.",
    )
    return parser.parse_args()


def load_comparison_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["tts_integrated"] = pd.to_numeric(df["tts_integrated"], errors="coerce")
    df["net_benefit_integrated"] = pd.to_numeric(df["net_benefit_integrated"], errors="coerce")
    df = df.dropna(subset=["mode", "development", "scenario", "tts_integrated", "net_benefit_integrated"])
    return df


def _read_sqlite_table_with_fallback(path: Path, table: str) -> pd.DataFrame:
    try_paths = [path]
    temp_dir = None

    # Some mounted Euler files are visible but not directly openable through sqlite.
    # Copying to a local temp file makes the read robust.
    if str(path).startswith("/Volumes/"):
        temp_dir = tempfile.TemporaryDirectory(prefix="infrascan_road_costs_")
        copied_path = Path(temp_dir.name) / path.name
        shutil.copy2(path, copied_path)
        try_paths.insert(0, copied_path)

    last_error = None
    try:
        for candidate in try_paths:
            try:
                with sqlite3.connect(candidate) as connection:
                    return pd.read_sql_query(f"SELECT * FROM {table}", connection)
            except Exception as exc:  # pragma: no cover - fallback handling
                last_error = exc
        raise RuntimeError(f"Failed to read table '{table}' from {path}: {last_error}")
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def refresh_road_rows_from_costs(df: pd.DataFrame, road_costs_dir: Path) -> pd.DataFrame:
    road_costs_dir = road_costs_dir.resolve()
    tt_path = road_costs_dir / "traveltime_savings_od.csv"
    total_path = road_costs_dir / "total_costs_od.csv"
    construction_path = road_costs_dir / "construction.gpkg"
    maintenance_path = road_costs_dir / "maintenance.gpkg"
    externalities_path = road_costs_dir / "externalities.gpkg"
    noise_path = road_costs_dir / "noise.gpkg"

    required_paths = [tt_path, total_path, construction_path, maintenance_path, externalities_path, noise_path]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required road cost input(s): " + ", ".join(missing))

    tt_df = pd.read_csv(tt_path)
    total_df = pd.read_csv(total_path)
    construction_df = _read_sqlite_table_with_fallback(construction_path, "construction")
    maintenance_df = _read_sqlite_table_with_fallback(maintenance_path, "maintenance")
    externalities_df = _read_sqlite_table_with_fallback(externalities_path, "externalities")
    noise_df = _read_sqlite_table_with_fallback(noise_path, "noise")

    construction_df = construction_df.rename(columns={"ID_new": "development"})
    maintenance_df = maintenance_df.rename(columns={"ID_new": "development"})
    externalities_df = externalities_df.rename(columns={"ID_new": "development"})
    noise_df = noise_df.rename(columns={"ID_new": "development"})
    total_df = total_df.rename(columns={"ID_new": "development"})

    base_cols = [
        "development",
        "cost_path",
        "cost_bridge",
        "cost_tunnel",
        "building_costs",
    ]
    construction_df = construction_df[base_cols].copy()
    construction_df["cost_ramp"] = (
        construction_df["building_costs"]
        - construction_df["cost_path"]
        - construction_df["cost_bridge"]
        - construction_df["cost_tunnel"]
    )

    maintenance_df = maintenance_df[["development", "maintenance"]].copy()
    externalities_df = externalities_df[["development", "climate_cost", "land_realloc", "nature"]].copy()
    noise_df = noise_df[["development", "noise_s1"]].copy()

    from infraScan.infraScanIntegrated.scoring_registry import capital_recovery_factor, dynamization_factor

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
    dyn_maint = dynamization_factor(
        growth_rate=common_cost_parameters.road_maintenance_operating_cost_growth,
        appraisal_years=common_cost_parameters.appraisal_years,
        discount_rate=common_cost_parameters.discount_rate,
    )
    dyn_tts = dynamization_factor(
        growth_rate=common_cost_parameters.real_wage_growth,
        appraisal_years=common_cost_parameters.appraisal_years,
        discount_rate=common_cost_parameters.discount_rate,
    )

    road_static = (
        construction_df
        .merge(maintenance_df, on="development", how="inner")
        .merge(externalities_df, on="development", how="inner")
        .merge(noise_df, on="development", how="inner")
    )
    road_static["externalities_raw"] = (
        road_static["climate_cost"]
        + road_static["land_realloc"]
        + road_static["nature"]
        + road_static["noise_s1"]
    )
    road_static["construction_integrated"] = (
        road_static["cost_path"] * crf_open_highway
        + road_static["cost_bridge"] * crf_bridge
        + road_static["cost_tunnel"] * crf_tunnel
        + road_static["cost_ramp"] * crf_ramp
    )
    road_static["maintenance_integrated"] = road_static["maintenance"] * dyn_maint
    road_static["cost_base_raw"] = road_static["building_costs"] + road_static["maintenance"]
    road_static["cost_base_integrated"] = (
        road_static["construction_integrated"] + road_static["maintenance_integrated"]
    )

    tt_cols = sorted(col for col in tt_df.columns if col.startswith("tt_"))
    scenario_rows = []
    for tt_col in tt_cols:
        scenario = tt_col.removeprefix("tt_")
        total_col = f"total_{scenario}"
        if total_col not in total_df.columns:
            continue

        scenario_df = (
            road_static
            .merge(tt_df[["development", tt_col]], on="development", how="inner")
            .merge(total_df[["development", total_col]], on="development", how="inner")
            .copy()
        )
        scenario_df["scenario"] = scenario
        scenario_df["mode"] = "Road"
        scenario_df["tts_raw"] = pd.to_numeric(scenario_df[tt_col], errors="coerce")
        scenario_df["tts_integrated"] = scenario_df["tts_raw"] * dyn_tts
        scenario_df["net_benefit_raw"] = pd.to_numeric(scenario_df[total_col], errors="coerce")
        scenario_df["net_benefit_integrated"] = (
            scenario_df["tts_integrated"]
            - scenario_df["cost_base_integrated"]
            - scenario_df["externalities_raw"]
        )
        scenario_df["maintenance_derived"] = scenario_df["maintenance"]
        scenario_df["construction_raw"] = scenario_df["building_costs"]
        scenario_df["maintenance_raw"] = scenario_df["maintenance"]
        scenario_df["operation_raw"] = 0.0
        scenario_df["operation_integrated"] = 0.0

        scenario_rows.append(
            scenario_df[
                [
                    "development",
                    "scenario",
                    "tts_raw",
                    "net_benefit_raw",
                    "building_costs",
                    "externalities_raw",
                    "maintenance",
                    "maintenance_derived",
                    "construction_raw",
                    "construction_integrated",
                    "maintenance_raw",
                    "maintenance_integrated",
                    "tts_integrated",
                    "operation_raw",
                    "operation_integrated",
                    "cost_base_raw",
                    "cost_base_integrated",
                    "net_benefit_integrated",
                    "mode",
                ]
            ]
        )

    if not scenario_rows:
        raise ValueError(f"No usable road TT scenario columns found in {tt_path}")

    refreshed_road = pd.concat(scenario_rows, ignore_index=True)
    refreshed_road["development"] = pd.to_numeric(refreshed_road["development"], errors="coerce").astype("Int64")

    merged = df[df["mode"] != "Road"].copy()
    for col in df.columns:
        if col not in refreshed_road.columns:
            refreshed_road[col] = np.nan
    refreshed_road = refreshed_road[df.columns]
    refreshed_road["mode"] = "Road"

    merged = pd.concat([merged, refreshed_road], ignore_index=True, sort=False)
    merged["development"] = pd.to_numeric(merged["development"], errors="coerce").astype("Int64")
    merged["scenario"] = merged["scenario"].astype(str)
    return merged


def make_vtt_grid(mode: str, grid_size: int) -> np.ndarray:
    start, end = MODE_CONFIG[mode]["range"]
    return np.linspace(start, end, grid_size)


def adjusted_net_benefit(base_nb: pd.Series, base_tts: pd.Series, base_vtt: float, new_vtt: float) -> pd.Series:
    scale = new_vtt / base_vtt
    return base_nb - base_tts + (base_tts * scale)


def adjusted_ratio(base_tts: pd.Series, base_cost: pd.Series, base_vtt: float, new_vtt: float) -> float:
    """Calculate CBA as mean TTS / mean Cost (not mean of individual ratios)."""
    scaled_tts = base_tts * (new_vtt / base_vtt)
    cost = pd.to_numeric(base_cost, errors="coerce")
    mean_tts = np.nanmean(scaled_tts)
    mean_cost = np.nanmean(cost)
    if mean_cost != 0 and np.isfinite(mean_cost):
        return mean_tts / mean_cost
    else:
        return np.nan


def summarize_mode(df: pd.DataFrame, mode: str, grid_size: int) -> pd.DataFrame:
    mode_df = df[df["mode"] == mode].copy()
    base_vtt = MODE_CONFIG[mode]["base_vtt"]
    rows = []

    for vtt in make_vtt_grid(mode, grid_size):
        nb = adjusted_net_benefit(
            base_nb=mode_df["net_benefit_integrated"],
            base_tts=mode_df["tts_integrated"],
            base_vtt=base_vtt,
            new_vtt=vtt,
        )
        per_dev_mean = mode_df.assign(adjusted_net_benefit=nb).groupby("development", as_index=False)["adjusted_net_benefit"].mean()

        rows.append(
            {
                "mode": mode,
                "vtt_chf_per_hour": vtt,
                "base_vtt_chf_per_hour": base_vtt,
                "mean_net_benefit_rows": float(nb.mean()),
                "median_net_benefit_rows": float(nb.median()),
                "share_positive_rows": float((nb > 0).mean()),
                "mean_net_benefit_developments": float(per_dev_mean["adjusted_net_benefit"].mean()),
                "median_net_benefit_developments": float(per_dev_mean["adjusted_net_benefit"].median()),
                "share_positive_developments": float((per_dev_mean["adjusted_net_benefit"] > 0).mean()),
                "positive_developments": int((per_dev_mean["adjusted_net_benefit"] > 0).sum()),
                "total_developments": int(per_dev_mean["development"].nunique()),
                "total_rows": int(len(mode_df)),
            }
        )

    return pd.DataFrame(rows)


def summarize_mode_by_development(df: pd.DataFrame, mode: str, grid_size: int) -> pd.DataFrame:
    mode_df = df[df["mode"] == mode].copy()
    base_vtt = MODE_CONFIG[mode]["base_vtt"]
    rows = []

    for vtt in make_vtt_grid(mode, grid_size):
        nb = adjusted_net_benefit(
            base_nb=mode_df["net_benefit_integrated"],
            base_tts=mode_df["tts_integrated"],
            base_vtt=base_vtt,
            new_vtt=vtt,
        )
        work = mode_df.assign(adjusted_net_benefit=nb)
        per_dev = work.groupby("development", as_index=False).agg(
            mean_net_benefit=("adjusted_net_benefit", "mean"),
            median_net_benefit=("adjusted_net_benefit", "median"),
            share_positive_scenarios=("adjusted_net_benefit", lambda s: float((s > 0).mean())),
            positive_scenarios=("adjusted_net_benefit", lambda s: int((s > 0).sum())),
            total_scenarios=("adjusted_net_benefit", "size"),
        )
        per_dev["mode"] = mode
        per_dev["vtt_chf_per_hour"] = vtt
        per_dev["base_vtt_chf_per_hour"] = base_vtt
        rows.append(per_dev)

    return pd.concat(rows, ignore_index=True)


def build_surface(summary_df: pd.DataFrame) -> pd.DataFrame:
    rail = summary_df[summary_df["mode"] == "Rail"].copy()
    road = summary_df[summary_df["mode"] == "Road"].copy()

    surface = rail.merge(road, how="cross", suffixes=("_rail", "_road"))
    surface["combined_mean_net_benefit_developments"] = (
        surface["mean_net_benefit_developments_rail"] + surface["mean_net_benefit_developments_road"]
    )
    surface["delta_mean_net_benefit_developments"] = (
        surface["mean_net_benefit_developments_rail"] - surface["mean_net_benefit_developments_road"]
    )
    surface["delta_share_positive_developments"] = (
        surface["share_positive_developments_rail"] - surface["share_positive_developments_road"]
    )
    return surface


def build_correlated_surface(summary_df: pd.DataFrame) -> pd.DataFrame:
    rail = summary_df[summary_df["mode"] == "Rail"].sort_values("vtt_chf_per_hour").reset_index(drop=True)
    road = summary_df[summary_df["mode"] == "Road"].sort_values("vtt_chf_per_hour").reset_index(drop=True)
    n = min(len(rail), len(road))
    if n == 0:
        return pd.DataFrame()

    correlated = pd.DataFrame(
        {
            "vtt_chf_per_hour_rail": rail.loc[: n - 1, "vtt_chf_per_hour"].to_numpy(),
            "vtt_chf_per_hour_road": road.loc[: n - 1, "vtt_chf_per_hour"].to_numpy(),
            "mean_net_benefit_developments_rail": rail.loc[: n - 1, "mean_net_benefit_developments"].to_numpy(),
            "mean_net_benefit_developments_road": road.loc[: n - 1, "mean_net_benefit_developments"].to_numpy(),
            "share_positive_developments_rail": rail.loc[: n - 1, "share_positive_developments"].to_numpy(),
            "share_positive_developments_road": road.loc[: n - 1, "share_positive_developments"].to_numpy(),
        }
    )
    correlated["combined_mean_net_benefit_developments"] = (
        correlated["mean_net_benefit_developments_rail"]
        + correlated["mean_net_benefit_developments_road"]
    )
    correlated["combined_share_positive_developments"] = (
        0.5
        * (
            correlated["share_positive_developments_rail"]
            + correlated["share_positive_developments_road"]
        )
    )
    return correlated


def filter_selected_analysis_input(df: pd.DataFrame, excluded_road_ids: list[int]) -> pd.DataFrame:
    filtered = df.copy()
    filtered["development"] = pd.to_numeric(filtered["development"], errors="coerce").astype("Int64")
    if excluded_road_ids:
        excluded_set = {int(dev) for dev in excluded_road_ids}
        filtered = filtered[
            ~(
                (filtered["mode"] == "Road")
                & (filtered["development"].astype("Int64").isin(excluded_set))
            )
        ].copy()
    return filtered


def select_top_road_developments_by_ratio(df: pd.DataFrame, top_n: int) -> list[int]:
    road_df = df[df["mode"] == "Road"].copy()
    road_df["base_ratio"] = adjusted_ratio(
        base_tts=road_df["tts_integrated"],
        base_cost=road_df["cost_base_integrated"],
        base_vtt=MODE_CONFIG["Road"]["base_vtt"],
        new_vtt=MODE_CONFIG["Road"]["base_vtt"],
    )
    ranked = (
        road_df.groupby("development", as_index=False)["base_ratio"]
        .mean()
        .sort_values("base_ratio", ascending=False)
    )
    return ranked.head(top_n)["development"].astype(int).tolist()


def summarize_ratio_line_correlated(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    road_df = df[df["mode"] == "Road"].copy()
    rail_df = df[df["mode"] == "Rail"].copy()

    # Restrict road to developments with positive travel-time savings so the
    # correlated ratio path reflects beneficial road projects only.
    road_df = road_df[road_df["tts_integrated"] > 0].copy()

    common_scenarios = sorted(set(road_df["scenario"].astype(str)).intersection(set(rail_df["scenario"].astype(str))))
    if common_scenarios:
        road_df = road_df[road_df["scenario"].astype(str).isin(common_scenarios)].copy()
        rail_df = rail_df[rail_df["scenario"].astype(str).isin(common_scenarios)].copy()

    rail_vtts = make_vtt_grid("Rail", grid_size)
    road_vtts = make_vtt_grid("Road", grid_size)
    n = min(len(rail_vtts), len(road_vtts))
    rows = []
    for idx in range(n):
        rail_vtt = float(rail_vtts[idx])
        road_vtt = float(road_vtts[idx])
        rail_ratio = adjusted_ratio(
            base_tts=rail_df["tts_integrated"],
            base_cost=rail_df["cost_base_integrated"],
            base_vtt=MODE_CONFIG["Rail"]["base_vtt"],
            new_vtt=rail_vtt,
        )
        road_ratio = adjusted_ratio(
            base_tts=road_df["tts_integrated"],
            base_cost=road_df["cost_base_integrated"],
            base_vtt=MODE_CONFIG["Road"]["base_vtt"],
            new_vtt=road_vtt,
        )
        rows.append(
            {
                "step": idx,
                "vtt_chf_per_hour_rail": rail_vtt,
                "vtt_chf_per_hour_road": road_vtt,
                "mean_ratio_rail": rail_ratio,
                "mean_ratio_road": road_ratio,
            }
        )
    return pd.DataFrame(rows)


def summarize_ratio_grid(df: pd.DataFrame, grid_size: int) -> pd.DataFrame:
    """Create full grid of benefit/cost ratios for all VTT combinations."""
    road_df = df[df["mode"] == "Road"].copy()
    rail_df = df[df["mode"] == "Rail"].copy()
    road_df = road_df[road_df["tts_integrated"] > 0].copy()

    common_scenarios = sorted(set(road_df["scenario"].astype(str)).intersection(set(rail_df["scenario"].astype(str))))
    if common_scenarios:
        road_df = road_df[road_df["scenario"].astype(str).isin(common_scenarios)].copy()
        rail_df = rail_df[rail_df["scenario"].astype(str).isin(common_scenarios)].copy()

    rail_vtts = make_vtt_grid("Rail", grid_size)
    road_vtts = make_vtt_grid("Road", grid_size)
    rows = []
    for rail_vtt in rail_vtts:
        for road_vtt in road_vtts:
            rail_ratio = adjusted_ratio(
                base_tts=rail_df["tts_integrated"],
                base_cost=rail_df["cost_base_integrated"],
                base_vtt=MODE_CONFIG["Rail"]["base_vtt"],
                new_vtt=float(rail_vtt),
            )
            road_ratio = adjusted_ratio(
                base_tts=road_df["tts_integrated"],
                base_cost=road_df["cost_base_integrated"],
                base_vtt=MODE_CONFIG["Road"]["base_vtt"],
                new_vtt=float(road_vtt),
            )
            rows.append(
                {
                    "vtt_chf_per_hour_rail": float(rail_vtt),
                    "vtt_chf_per_hour_road": float(road_vtt),
                    "mean_ratio_rail": rail_ratio,
                    "mean_ratio_road": road_ratio,
                }
            )
    return pd.DataFrame(rows)

def plot_side_by_side(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), dpi=300, sharex="col")
    mode_order = ["Rail", "Road"]

    for col_idx, mode in enumerate(mode_order):
        mode_df = summary_df[summary_df["mode"] == mode].sort_values("vtt_chf_per_hour")
        color = MODE_CONFIG[mode]["color"]
        base_vtt = MODE_CONFIG[mode]["base_vtt"]

        ax_top = axes[0, col_idx]
        ax_bottom = axes[1, col_idx]

        ax_top.plot(
            mode_df["vtt_chf_per_hour"],
            mode_df["mean_net_benefit_developments"] / 1_000_000.0,
            color=color,
            linewidth=2.2,
        )
        ax_top.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        ax_top.axvline(base_vtt, color="#7f7f7f", linewidth=1.0, linestyle=":")
        ax_top.set_title(f"{mode}: mean net benefit per development", fontsize=12)
        ax_top.set_ylabel("Mean net benefit [Mio. CHF/y]", fontsize=10)
        ax_top.grid(alpha=0.25, linewidth=0.6)

        ax_bottom.plot(
            mode_df["vtt_chf_per_hour"],
            100.0 * mode_df["share_positive_developments"],
            color=color,
            linewidth=2.2,
        )
        ax_bottom.axvline(base_vtt, color="#7f7f7f", linewidth=1.0, linestyle=":")
        ax_bottom.set_xlabel("VTT [CHF/h]", fontsize=10)
        ax_bottom.set_ylabel("Positive developments [%]", fontsize=10)
        ax_bottom.grid(alpha=0.25, linewidth=0.6)

        base_row = mode_df.iloc[(mode_df["vtt_chf_per_hour"] - base_vtt).abs().argmin()]
        label = (
            f"base {base_vtt:.2f}\n"
            f"{base_row['positive_developments']}/{base_row['total_developments']} positive"
        )
        ax_bottom.annotate(
            label,
            xy=(base_row["vtt_chf_per_hour"], 100.0 * base_row["share_positive_developments"]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=8,
            color="#333333",
        )

    fig.suptitle("VTT sensitivity: rail and road side by side", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_3d_surface(surface_df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(15, 6), dpi=300)
    ax_combined = fig.add_subplot(1, 2, 1, projection="3d")
    ax_delta = fig.add_subplot(1, 2, 2, projection="3d")

    rail_values = np.sort(surface_df["vtt_chf_per_hour_rail"].unique())
    road_values = np.sort(surface_df["vtt_chf_per_hour_road"].unique())
    rail_grid, road_grid = np.meshgrid(rail_values, road_values, indexing="ij")

    combined_grid = (
        surface_df.pivot(
            index="vtt_chf_per_hour_rail",
            columns="vtt_chf_per_hour_road",
            values="combined_mean_net_benefit_developments",
        ).sort_index().sort_index(axis=1).to_numpy() / 1_000_000.0
    )
    delta_grid = (
        surface_df.pivot(
            index="vtt_chf_per_hour_rail",
            columns="vtt_chf_per_hour_road",
            values="delta_mean_net_benefit_developments",
        ).sort_index().sort_index(axis=1).to_numpy() / 1_000_000.0
    )

    combined_surface = ax_combined.plot_surface(
        rail_grid,
        road_grid,
        combined_grid,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    delta_surface = ax_delta.plot_surface(
        rail_grid,
        road_grid,
        delta_grid,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )

    for ax, title, zlabel in [
        (ax_combined, "Combined mean net benefit", "Net benefit [Mio. CHF/y]"),
        (ax_delta, "Rail minus road mean net benefit", "Delta net benefit [Mio. CHF/y]"),
    ]:
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xlabel("Rail VTT [CHF/h]", fontsize=10, labelpad=8)
        ax.set_ylabel("Road VTT [CHF/h]", fontsize=10, labelpad=8)
        ax.set_zlabel(zlabel, fontsize=10, labelpad=8)
        ax.view_init(elev=28, azim=-132)

    fig.colorbar(combined_surface, ax=ax_combined, shrink=0.7, pad=0.08, label="Combined [Mio. CHF/y]")
    fig.colorbar(delta_surface, ax=ax_delta, shrink=0.7, pad=0.08, label="Rail - Road [Mio. CHF/y]")

    fig.suptitle("VTT sensitivity surfaces", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_3d_correlated(correlated_df: pd.DataFrame, output_path: Path) -> None:
    if correlated_df.empty:
        return

    fig = plt.figure(figsize=(15, 6), dpi=300)
    ax_nb = fig.add_subplot(1, 2, 1, projection="3d")
    ax_share = fig.add_subplot(1, 2, 2, projection="3d")

    x = correlated_df["vtt_chf_per_hour_rail"].to_numpy()
    y = correlated_df["vtt_chf_per_hour_road"].to_numpy()
    z_nb = correlated_df["combined_mean_net_benefit_developments"].to_numpy() / 1_000_000.0
    z_share = 100.0 * correlated_df["combined_share_positive_developments"].to_numpy()

    for ax, z, title, zlabel, cmap in [
        (ax_nb, z_nb, "Correlated rail-road VTT path: combined mean net benefit", "Net benefit [Mio. CHF/y]", "viridis"),
        (ax_share, z_share, "Correlated rail-road VTT path: combined positive share", "Positive share [%]", "plasma"),
    ]:
        line = ax.plot(x, y, z, color="#3b3b3b", linewidth=1.4, alpha=0.9)
        scatter = ax.scatter(x, y, z, c=z, cmap=cmap, s=34, depthshade=True)
        del line
        ax.set_xlabel("Rail VTT [CHF/h]", fontsize=10, labelpad=8)
        ax.set_ylabel("Road VTT [CHF/h]", fontsize=10, labelpad=8)
        ax.set_zlabel(zlabel, fontsize=10, labelpad=8)
        ax.set_title(title, fontsize=12, pad=10)
        ax.view_init(elev=26, azim=-128)
        fig.colorbar(scatter, ax=ax, shrink=0.72, pad=0.08)

    fig.suptitle("Correlated VTT sensitivity", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_development_heatmaps(development_df: pd.DataFrame, output_path: Path) -> None:
    mode_order = ["Rail", "Road"]
    fig_height = 0.18 * max(
        development_df[development_df["mode"] == "Rail"]["development"].nunique(),
        development_df[development_df["mode"] == "Road"]["development"].nunique(),
    )
    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, fig_height)), dpi=220)
    shared_norm = TwoSlopeNorm(vmin=-80.0, vcenter=0.0, vmax=20.0)
    shared_cmap = "BrBG_r"
    image = None

    for ax, mode in zip(axes, mode_order):
        mode_df = development_df[development_df["mode"] == mode].copy()
        base_vtt = MODE_CONFIG[mode]["base_vtt"]
        available_vtt = np.sort(mode_df["vtt_chf_per_hour"].unique())
        nearest_base_vtt = float(available_vtt[np.argmin(np.abs(available_vtt - base_vtt))])

        base_order = (
            mode_df.loc[np.isclose(mode_df["vtt_chf_per_hour"], nearest_base_vtt)]
            .sort_values("mean_net_benefit", ascending=False)["development"]
            .tolist()
        )
        pivot = mode_df.pivot(
            index="development",
            columns="vtt_chf_per_hour",
            values="mean_net_benefit",
        ).reindex(base_order)
        pivot_mio = pivot / 1_000_000.0

        image = ax.imshow(
            pivot_mio.to_numpy(),
            aspect="auto",
            cmap=shared_cmap,
            norm=shared_norm,
            origin="upper",
        )
        ax.grid(False)

        tick_positions = np.arange(len(pivot.columns))
        tick_labels = [f"{value:.2f}" for value in pivot.columns]
        step = max(1, len(tick_positions) // 8)
        ax.set_xticks(tick_positions[::step])
        ax.set_xticklabels(tick_labels[::step], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels([str(int(dev)) for dev in pivot.index], fontsize=5 if mode == "Road" else 7)
        ax.set_xlabel("VTT [CHF/h]", fontsize=10)
        ax.set_ylabel("Development", fontsize=10)
        ax.set_title(f"{mode}: mean net benefit by development", fontsize=12)

        base_idx = int(np.argmin(np.abs(pivot.columns.to_numpy(dtype=float) - nearest_base_vtt)))
        ax.axvline(base_idx, color="black", linewidth=1.0, linestyle=":")

        positive_line = (
            mode_df.groupby("vtt_chf_per_hour", as_index=False)
            .agg(share_positive_developments=("mean_net_benefit", lambda s: 100.0 * float((s > 0).mean())))
            .sort_values("vtt_chf_per_hour")
        )
        positive_line = positive_line.set_index("vtt_chf_per_hour").reindex(pivot.columns).reset_index()
        ax_share = ax.twinx()
        share_values = positive_line["share_positive_developments"].to_numpy(dtype=float)
        ax_share.plot(
            np.arange(len(pivot.columns)),
            share_values,
            color="#2b2b2b",
            linewidth=1.8,
            alpha=0.9,
            zorder=4,
        )
        ax_share.set_ylim(0, 100)
        ax_share.set_yticks(np.arange(0, 101, 20))
        ax_share.set_ylabel("Positive developments [%]", fontsize=9, color="#2b2b2b")
        ax_share.tick_params(axis="y", labelsize=8, colors="#2b2b2b")
        ax_share.grid(False)

    cax = fig.add_axes([0.935, 0.16, 0.018, 0.68])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Mean net benefit [Mio. CHF/y]", fontsize=9)

    fig.suptitle(
        "Development-level VTT sensitivity\nRows are developments, columns are VTT values, colors show mean net benefit, right axis shows the share of positive developments",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 0.91, 0.97))
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_selected_road_ratio_heatmap(df: pd.DataFrame, top_developments: list[int], grid_size: int, output_path: Path) -> None:
    road_df = df[(df["mode"] == "Road") & (df["development"].astype("Int64").isin(top_developments))].copy()
    rows = []
    for vtt in make_vtt_grid("Road", grid_size):
        ratio = adjusted_ratio(
            base_tts=road_df["tts_integrated"],
            base_cost=road_df["cost_base_integrated"],
            base_vtt=MODE_CONFIG["Road"]["base_vtt"],
            new_vtt=vtt,
        )
        per_dev = road_df.assign(adjusted_ratio=ratio).groupby("development", as_index=False)["adjusted_ratio"].mean()
        per_dev["vtt_chf_per_hour"] = vtt
        rows.append(per_dev)

    heatmap_df = pd.concat(rows, ignore_index=True)
    dev_order = (
        heatmap_df.loc[np.isclose(heatmap_df["vtt_chf_per_hour"], MODE_CONFIG["Road"]["base_vtt"], atol=0.1)]
        .sort_values("adjusted_ratio", ascending=False)["development"]
        .astype(int)
        .tolist()
    )
    if not dev_order:
        dev_order = [int(dev) for dev in top_developments]

    
    pivot = heatmap_df.pivot(index="development", columns="vtt_chf_per_hour", values="adjusted_ratio")
    pivot = pivot.reindex(dev_order)
    vmax = float(np.nanmax(np.abs(pivot_mio.to_numpy())))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    image = ax.imshow(
        pivot_mio.to_numpy(),
        aspect="auto",
        cmap=blue_yellow,
        norm=norm,
        origin="upper",
    )



    fig, ax = plt.subplots(figsize=(12, max(8, 0.22 * len(dev_order))), dpi=240)
    image = ax.imshow(
        pivot.to_numpy(),
        aspect="auto",
        cmap="RdBu_r",
        norm=norm,
        origin="upper",
    )
    tick_positions = np.arange(len(pivot.columns))
    tick_labels = [f"{value:.2f}" for value in pivot.columns]
    step = max(1, len(tick_positions) // 8)
    ax.set_xticks(tick_positions[::step])
    ax.set_xticklabels(tick_labels[::step], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(int(dev)) for dev in pivot.index], fontsize=6)
    ax.set_xlabel("Road VTT [CHF/h]", fontsize=10)
    ax.set_ylabel("Road development (top 40 by mean ratio)", fontsize=10)
    ax.set_title("Road top-40 benefit/cost ratio sensitivity", fontsize=13)

    base_idx = int(np.argmin(np.abs(pivot.columns.to_numpy(dtype=float) - MODE_CONFIG["Road"]["base_vtt"])))
    ax.axvline(base_idx, color="black", linewidth=1.0, linestyle=":")
    cbar = plt.colorbar(image, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("Mean benefit/cost ratio", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig




def plot_selected_correlated_ratio_lines(correlated_ratio_df, output_path):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=240)

    ax.plot(
        correlated_ratio_df["vtt_chf_per_hour_rail"],
        correlated_ratio_df["mean_ratio_road"],
        color=MODE_CONFIG["Road"]["color"],
        linewidth=2.3,
        label="Road ratio",
    )

    ax.plot(
        correlated_ratio_df["vtt_chf_per_hour_rail"],
        correlated_ratio_df["mean_ratio_rail"],
        color=MODE_CONFIG["Rail"]["color"],
        linewidth=2.3,
        label="Rail ratio",
    )

    # Horizontal reference line at y = 1
    ax.axhline(
        1,
        color="black",
        linewidth=1.2,
        linestyle="-",
        zorder=0,
    )

    # Vertical reference lines
    ax.axvline(
        15.2,
        color="#666666",
        linewidth=1.0,
        linestyle="--",
    )

    ax.axvline(
        24.2,
        color="#666666",
        linewidth=1.0,
        linestyle="--",
    )

    ax.set_xlabel("Rail VTT [CHF/h]", fontsize=10)
    ax.set_ylabel("Mean benefit/cost ratio", fontsize=10)

    ax.set_title(
        "Correlated rail-road VTT path: mean benefit/cost ratio",
        fontsize=13,
    )

    ax.grid(alpha=0.25, linewidth=0.6)

    # Remove frame/spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional:
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    ax.legend(frameon=False)

    road_start, road_end = MODE_CONFIG["Road"]["range"]

    secax = ax.secondary_xaxis("top")
    secax.set_xlabel("Road VTT [CHF/h] (correlated path)", fontsize=10)

    # FEWER TICKS
    tick_idx = np.linspace(
        0,
        len(correlated_ratio_df) - 1,
        6,   # number of ticks
        dtype=int,
    )

    tick_positions = correlated_ratio_df["vtt_chf_per_hour_rail"].iloc[tick_idx]

    secax.set_xticks(tick_positions)

    road_labels = np.linspace(
        road_end,
        road_start,
        len(tick_positions),
    )

    secax.set_xticklabels(
        [f"{x:.0f}" for x in road_labels]
    )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig
def build_normalized_correlated_ratio_df(correlated_ratio_df: pd.DataFrame) -> pd.DataFrame:
    normalized = correlated_ratio_df.copy()
    if normalized.empty:
        return normalized

    rail_base_vtt = MODE_CONFIG["Rail"]["base_vtt"]
    road_base_vtt = MODE_CONFIG["Road"]["base_vtt"]

    rail_base_idx = (normalized["vtt_chf_per_hour_rail"] - rail_base_vtt).abs().idxmin()
    road_base_idx = (normalized["vtt_chf_per_hour_road"] - road_base_vtt).abs().idxmin()

    rail_base_ratio = float(normalized.loc[rail_base_idx, "mean_ratio_rail"])
    road_base_ratio = float(normalized.loc[road_base_idx, "mean_ratio_road"])

    normalized["rail_vtt_pct_change"] = 100.0 * (
        normalized["vtt_chf_per_hour_rail"] / rail_base_vtt - 1.0
    )
    normalized["road_vtt_pct_change"] = 100.0 * (
        normalized["vtt_chf_per_hour_road"] / road_base_vtt - 1.0
    )
    normalized["rail_ratio_pct_change"] = 100.0 * (
        normalized["mean_ratio_rail"] / rail_base_ratio - 1.0
    )
    normalized["road_ratio_pct_change"] = 100.0 * (
        normalized["mean_ratio_road"] / road_base_ratio - 1.0
    )
    return normalized


def plot_normalized_correlated_ratio_lines(correlated_ratio_df: pd.DataFrame, output_path: Path):
    normalized = build_normalized_correlated_ratio_df(correlated_ratio_df)
    if normalized.empty:
        return None

    fig, ax = plt.subplots(figsize=(12, 6), dpi=240)

    ax.plot(
        normalized["rail_vtt_pct_change"],
        normalized["rail_ratio_pct_change"],
        color=MODE_CONFIG["Rail"]["color"],
        linewidth=2.3,
        label="Rail ratio",
    )
    ax.plot(
        normalized["road_vtt_pct_change"],
        normalized["road_ratio_pct_change"],
        color=MODE_CONFIG["Road"]["color"],
        linewidth=2.3,
        label="Road ratio",
    )

    ax.axhline(0, color="black", linewidth=1.2, linestyle="-", zorder=0)
    ax.axvline(0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Change from base VTT [%]", fontsize=10)
    ax.set_ylabel("Change from base mean benefit/cost ratio [%]", fontsize=10)
    ax.set_title("Correlated rail-road VTT path: normalized mean benefit/cost ratio", fontsize=13)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def plot_correlated_ratio_contour(df: pd.DataFrame, output_path: Path) -> None:
    """Contour plot of benefit/cost ratio for both modes across VTT ranges."""
    fig, (ax_rail, ax_road) = plt.subplots(1, 2, figsize=(15, 6), dpi=240)
    
    rail_vtts = np.sort(df["vtt_chf_per_hour_rail"].unique())
    road_vtts = np.sort(df["vtt_chf_per_hour_road"].unique())
    rail_grid, road_grid = np.meshgrid(rail_vtts, road_vtts, indexing="ij")
    
    rail_ratio_grid = df.pivot(
        index="vtt_chf_per_hour_rail",
        columns="vtt_chf_per_hour_road",
        values="mean_ratio_rail",
    ).sort_index().sort_index(axis=1).to_numpy()
    
    road_ratio_grid = df.pivot(
        index="vtt_chf_per_hour_rail",
        columns="vtt_chf_per_hour_road",
        values="mean_ratio_road",
    ).sort_index().sort_index(axis=1).to_numpy()
    
    contour_rail = ax_rail.contourf(rail_grid, road_grid, rail_ratio_grid, levels=20, cmap="RdYlGn")
    contour_road = ax_road.contourf(rail_grid, road_grid, road_ratio_grid, levels=20, cmap="RdYlGn")
    
    ax_rail.set_title("Rail: Mean benefit/cost ratio", fontsize=12)
    ax_road.set_title("Road: Mean benefit/cost ratio", fontsize=12)
    
    for ax in [ax_rail, ax_road]:
        ax.set_xlabel("Rail VTT [CHF/h]", fontsize=10)
        ax.set_ylabel("Road VTT [CHF/h]", fontsize=10)
        ax.grid(alpha=0.3, linewidth=0.5)
    
    fig.colorbar(contour_rail, ax=ax_rail, label="Rail B/C ratio")
    fig.colorbar(contour_road, ax=ax_road, label="Road B/C ratio")
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig
    return fig


def plot_correlated_ratio_3d(df: pd.DataFrame, output_path: Path) -> None:
    """3D surface plot of benefit/cost ratio for both modes."""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 6), dpi=240)
    ax_rail = fig.add_subplot(121, projection="3d")
    ax_road = fig.add_subplot(122, projection="3d")
    
    rail_vtts = np.sort(df["vtt_chf_per_hour_rail"].unique())
    road_vtts = np.sort(df["vtt_chf_per_hour_road"].unique())
    rail_grid, road_grid = np.meshgrid(rail_vtts, road_vtts, indexing="ij")
    
    rail_ratio_grid = df.pivot(
        index="vtt_chf_per_hour_rail",
        columns="vtt_chf_per_hour_road",
        values="mean_ratio_rail",
    ).sort_index().sort_index(axis=1).to_numpy()
    
    road_ratio_grid = df.pivot(
        index="vtt_chf_per_hour_rail",
        columns="vtt_chf_per_hour_road",
        values="mean_ratio_road",
    ).sort_index().sort_index(axis=1).to_numpy()
    
    surf_rail = ax_rail.plot_surface(
        rail_grid, road_grid, rail_ratio_grid,
        cmap="RdYlGn", linewidth=0, antialiased=True, alpha=0.9
    )
    surf_road = ax_road.plot_surface(
        rail_grid, road_grid, road_ratio_grid,
        cmap="RdYlGn", linewidth=0, antialiased=True, alpha=0.9
    )
    
    ax_rail.set_title("Rail: Mean benefit/cost ratio", fontsize=12)
    ax_road.set_title("Road: Mean benefit/cost ratio", fontsize=12)
    
    for ax in [ax_rail, ax_road]:
        ax.set_xlabel("Rail VTT [CHF/h]", fontsize=10)
        ax.set_ylabel("Road VTT [CHF/h]", fontsize=10)
        ax.set_zlabel("B/C ratio", fontsize=10)
        ax.view_init(elev=28, azim=-132)
    
    fig.colorbar(surf_rail, ax=ax_rail, shrink=0.7, label="Rail B/C")
    fig.colorbar(surf_road, ax=ax_road, shrink=0.7, label="Road B/C")
    
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    return fig


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_comparison_data(args.input)
    if args.road_costs_dir is not None:
        df = refresh_road_rows_from_costs(df, args.road_costs_dir)
        if args.write_refreshed_input:
            refreshed_input_path = args.output_dir / "mode_comparison_by_development_scenario_refreshed.csv"
            df.to_csv(refreshed_input_path, index=False)

    if args.selected_only:
        df = filter_selected_analysis_input(df, args.exclude_road_developments)
        top_road_developments = select_top_road_developments_by_ratio(df, args.top_road_developments)
        correlated_ratio_df = summarize_ratio_line_correlated(df, args.grid_size)
        ratio_grid_df = summarize_ratio_grid(df, args.grid_size)

        road_heatmap_path = args.output_dir / "vtt_sensitivity_road_top40_ratio_heatmap.png"
        ratio_lines_path = args.output_dir / "vtt_sensitivity_correlated_ratio_lines.png"
        normalized_ratio_lines_path = args.output_dir / "vtt_sensitivity_correlated_ratio_lines_normalized.png"
        ratio_contour_path = args.output_dir / "vtt_sensitivity_ratio_contour.png"
        ratio_3d_path = args.output_dir / "vtt_sensitivity_ratio_3d.png"
        top_road_csv_path = args.output_dir / "vtt_sensitivity_road_top40_selected.csv"
        correlated_ratio_csv_path = args.output_dir / "vtt_sensitivity_correlated_ratio_lines.csv"
        normalized_correlated_ratio_csv_path = args.output_dir / "vtt_sensitivity_correlated_ratio_lines_normalized.csv"

        plot_selected_road_ratio_heatmap(df, top_road_developments, args.grid_size, road_heatmap_path)
        plot_selected_correlated_ratio_lines(correlated_ratio_df, ratio_lines_path)
        plot_normalized_correlated_ratio_lines(correlated_ratio_df, normalized_ratio_lines_path)
        plot_correlated_ratio_contour(ratio_grid_df, ratio_contour_path)
        plot_correlated_ratio_3d(ratio_grid_df, ratio_3d_path)

        pd.DataFrame({"development": top_road_developments}).to_csv(top_road_csv_path, index=False)
        correlated_ratio_df.to_csv(correlated_ratio_csv_path, index=False)
        build_normalized_correlated_ratio_df(correlated_ratio_df).to_csv(normalized_correlated_ratio_csv_path, index=False)

        print(f"Selected road heatmap: {road_heatmap_path}")
        print(f"Selected correlated ratio plot: {ratio_lines_path}")
        print(f"Selected ratio contour plot: {ratio_contour_path}")
        print(f"Selected ratio 3D plot: {ratio_3d_path}")
        print(f"Selected road IDs: {top_road_csv_path}")
        print(f"Correlated ratio CSV: {correlated_ratio_csv_path}")
        return

    summary_parts = [summarize_mode(df, mode, args.grid_size) for mode in ("Rail", "Road")]
    summary_df = pd.concat(summary_parts, ignore_index=True)
    development_parts = [summarize_mode_by_development(df, mode, args.grid_size) for mode in ("Rail", "Road")]
    development_df = pd.concat(development_parts, ignore_index=True)
    surface_df = build_surface(summary_df)
    correlated_df = build_correlated_surface(summary_df)

    summary_path = args.output_dir / "vtt_sensitivity_summary.csv"
    surface_path = args.output_dir / "vtt_sensitivity_surface.csv"
    correlated_path = args.output_dir / "vtt_sensitivity_correlated.csv"
    development_path = args.output_dir / "vtt_sensitivity_development_summary.csv"
    plot_side_by_side(summary_df, args.output_dir / "vtt_sensitivity_side_by_side.png")
    plot_3d_surface(surface_df, args.output_dir / "vtt_sensitivity_3d_surface.png")
    plot_3d_correlated(correlated_df, args.output_dir / "vtt_sensitivity_3d_correlated.png")
    plot_development_heatmaps(development_df, args.output_dir / "vtt_sensitivity_development_heatmaps.png")

    summary_df.to_csv(summary_path, index=False)
    surface_df.to_csv(surface_path, index=False)
    correlated_df.to_csv(correlated_path, index=False)
    development_df.to_csv(development_path, index=False)

    print(f"Input: {args.input}")
    print(f"Summary CSV: {summary_path}")
    print(f"Surface CSV: {surface_path}")
    print(f"Correlated CSV: {correlated_path}")
    print(f"Development CSV: {development_path}")
    print(f"Plots: {args.output_dir}")


if __name__ == "__main__":
    main()
