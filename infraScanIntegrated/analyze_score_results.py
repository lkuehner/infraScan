from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import common_cost_parameters
from ..infraScanRail import paths as rail_paths


DATA_ROOT = Path(rail_paths.MAIN)
SCORE_RESULTS_LONG = Path("infraScan/infraScanIntegrated/outputs/score_results/score_results_long.csv")
OUTPUT_DIR = Path("infraScan/infraScanIntegrated/outputs/score_analysis")

ROAD_TTS_DETAIL = (
    DATA_ROOT
    / "euler"
    / "infraScanRoad_trust_2iter_alldev_10sce"
    / "traffic_flow"
    / "od"
    / "od_tt_savings_detailed.csv"
)
ROAD_VKM_MONETIZATION = (
    DATA_ROOT
    / "euler"
    / "infraScanRoad_trust_2iter_alldev_10sce"
    / "traffic_flow"
    / "road_externalities_inputs"
    / "road_externalities_monetization.csv"
)
RAIL_TTS = DATA_ROOT / "data" / "infraScanRail" / "costs" / "traveltime_savings.csv"
RAIL_TOTAL_COSTS = DATA_ROOT / "data" / "infraScanRail" / "costs" / "total_costs.csv"
CONSTRUCTION_PROXY_YEARS = 40.0

COST_COLORS = {
    "construction": "#a6bddb",
    "maintenance": "#3690c0",
    "operating": "#1f5a89",
    "accident": "#d95f0e",
    "air": "#756bb1",
    "co2": "#31a354",
    "noise": "#dd1c77",
    "land": "#636363",
    "tts": "#fedf2f",
}


ROAD_OVERVIEW_COST_SCORES = [
    "road_construction_cost",
    "road_maint_cost",
    "road_accident_cost",
    "road_airpollution_cost",
    "road_co2_cost",
    "road_noise_cost",
    "road_land_consumption_cost",
]
RAIL_OVERVIEW_COST_SCORES = [
    "rail_construction_cost",
    "rail_maint_cost",
    "rail_operation_cost",
    "rail_accident_cost",
    "rail_airpollution_cost",
    "rail_co2_cost",
    "rail_noise_cost",
    "rail_land_consumption_cost",
]
ROAD_TTS_SCORE = "road_tts_cost"
RAIL_TTS_SCORE = "rail_tts_cost"


def scenario_number(scenario: str) -> int:
    try:
        return int(str(scenario).split("_")[-1])
    except ValueError:
        return -1


def normalize_development(values: pd.Series) -> pd.Series:
    normalized = values.astype(str)
    normalized = normalized.str.replace("Development_", "", regex=False)
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized


def save_analysis_notes(path: Path) -> None:
    notes = [
        "Rail externalities are included using the corrected train_km.csv with development-specific IDs.",
        "Road overview excludes developments without any flow on the new link across all scenarios.",
        "Annual overview excludes road_climate_cost and road_ecological_disruption_cost because they are not directly comparable to the integrated annual values.",
        "Road TTS minutes are currently only available as demand-weighted aggregate tt_savings_peak from od_tt_savings_detailed.csv.",
        "Rail TTS minutes are derived from tt_savings_daily * 60 at prognosis_year.",
        "Road scenario line plots show total TTS, network TTS and access-only TTS (origin + destination).",
    ]
    path.write_text("\n".join(notes))


def load_score_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(SCORE_RESULTS_LONG)
    df["development"] = normalize_development(df["development"])
    df["scenario"] = df["scenario"].astype(str)

    duplicate_counts = (
        df.groupby(["mode", "development", "scenario", "score_id"], as_index=False)
        .size()
        .rename(columns={"size": "row_count"})
    )

    collapsed = (
        df.groupby(["mode", "development", "scenario", "score_id"], as_index=False)
        .agg(
            standalone_value=("standalone_value", "mean"),
            integrated_value=("integrated_value", "mean"),
        )
    )
    return collapsed, duplicate_counts


def build_coverage_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    return (
        score_df.groupby(["mode", "score_id"], as_index=False)
        .agg(
            n_developments=("development", "nunique"),
            n_scenarios=("scenario", "nunique"),
            n_rows=("score_id", "size"),
            standalone_missing=("standalone_value", lambda s: int(s.isna().sum())),
            integrated_missing=("integrated_value", lambda s: int(s.isna().sum())),
        )
        .sort_values(["mode", "score_id"])
    )


def annual_proxy_value(row: pd.Series) -> float:
    score_id = row["score_id"]
    standalone_value = row["standalone_value"]
    integrated_value = row["integrated_value"]

    road_standalone_is_annual = {
        "road_construction_cost",
        "road_maint_cost",
        "road_tts_cost",
        "road_climate_cost",
        "road_land_consumption_cost",
        "road_ecological_disruption_cost",
        "road_noise_cost",
    }
    construction_scores = {
        "rail_construction_cost",
    }
    standalone_is_annual = {
        "road_tts_cost",
        "road_maint_cost",
        "rail_tts_cost",
        "rail_maint_cost",
        "rail_operation_cost",
    }
    if score_id in road_standalone_is_annual and pd.notna(standalone_value):
        return float(standalone_value)
    if score_id in construction_scores and pd.notna(standalone_value):
        return float(standalone_value) / CONSTRUCTION_PROXY_YEARS
    if score_id in standalone_is_annual and pd.notna(standalone_value):
        return float(standalone_value)
    return float(integrated_value) if pd.notna(integrated_value) else np.nan


def get_road_developments_without_new_link_flow() -> set[str]:
    vkm = pd.read_csv(ROAD_VKM_MONETIZATION)
    vkm["development"] = normalize_development(vkm["development"])
    summary = (
        vkm.groupby("development", as_index=False)
        .agg(
            max_abs_new_link=("delta_vkm_peak_hour_new_link", lambda s: float(s.abs().max())),
            scenario_count=("scenario", "nunique"),
        )
    )
    return set(summary.loc[summary["max_abs_new_link"] <= 1e-9, "development"].astype(str))


def build_annual_overview(score_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    road_excluded_devs = get_road_developments_without_new_link_flow()
    road_valid_devs = (
        set(
        score_df.loc[score_df["score_id"] == ROAD_TTS_SCORE, "development"]
        ) & set(
        score_df.loc[score_df["score_id"] == "road_construction_cost", "development"]
        )
    ) - road_excluded_devs

    overview = score_df[
        (
            (score_df["mode"] == "Road")
            & (
                score_df["score_id"].isin([ROAD_TTS_SCORE] + ROAD_OVERVIEW_COST_SCORES)
            )
            & score_df["development"].isin(road_valid_devs)
        )
        | (
            (score_df["mode"] == "Rail")
            & (
                score_df["score_id"].isin([RAIL_TTS_SCORE] + RAIL_OVERVIEW_COST_SCORES)
            )
        )
    ].copy()

    overview["standalone_annual_proxy"] = overview.apply(annual_proxy_value, axis=1)

    annual_long = overview.melt(
        id_vars=["mode", "development", "scenario", "score_id"],
        value_vars=["standalone_annual_proxy", "integrated_value"],
        var_name="value_mode",
        value_name="annual_value",
    )
    annual_long["value_mode"] = annual_long["value_mode"].replace(
        {
            "standalone_annual_proxy": "standalone_annual_proxy",
            "integrated_value": "integrated",
        }
    )
    tts_summary = (
        annual_long[annual_long["score_id"].str.endswith("_tts_cost")]
        .groupby(["mode", "development", "scenario", "value_mode"], as_index=False)
        .agg(tts_annual=("annual_value", "sum"))
    )
    cost_summary = (
        annual_long[~annual_long["score_id"].str.endswith("_tts_cost")]
        .groupby(["mode", "development", "scenario", "value_mode"], as_index=False)
        .agg(cost_annual=("annual_value", "sum"))
    )
    annual_summary = tts_summary.merge(
        cost_summary,
        on=["mode", "development", "scenario", "value_mode"],
        how="outer",
    )
    annual_summary["cost_annual_negative"] = -annual_summary["cost_annual"]
    annual_summary["net_annual"] = annual_summary["tts_annual"] - annual_summary["cost_annual"]
    return annual_long, annual_summary


def plot_overview_bars(annual_summary: pd.DataFrame, output_dir: Path) -> None:
    for mode in ["Road", "Rail"]:
        mode_df = annual_summary[annual_summary["mode"] == mode].copy()
        if mode_df.empty:
            continue

        order = (
            mode_df[mode_df["value_mode"] == "integrated"]
            .groupby("development", as_index=False)["net_annual"]
            .median()
            .sort_values("net_annual", ascending=False)["development"]
            .astype(str)
            .tolist()
        )

        plot_df = (
            mode_df.groupby(["development", "value_mode"], as_index=False)
            .agg(
                tts_annual=("tts_annual", "median"),
                cost_annual_negative=("cost_annual_negative", "median"),
                net_annual=("net_annual", "median"),
            )
        )
        plot_df["development"] = pd.Categorical(
            plot_df["development"].astype(str),
            categories=order,
            ordered=True,
        )
        plot_df = plot_df.sort_values(["development", "value_mode"])

        y_base = np.arange(len(order))
        fig_height = max(8, len(order) * 0.18)
        fig, ax = plt.subplots(figsize=(16, fig_height))

        offsets = {
            "standalone_annual_proxy": -0.18,
            "integrated": 0.18,
        }
        colors = {
            "tts": "#2b8a3e",
            "cost": "#c92a2a",
        }

        for value_mode, offset in offsets.items():
            subset = plot_df[plot_df["value_mode"] == value_mode].copy()
            subset = subset.set_index(subset["development"].astype(str)).reindex(order).reset_index(drop=True)
            y = y_base + offset
            ax.barh(
                y,
                subset["tts_annual"],
                height=0.32,
                color=colors["tts"],
                alpha=0.85 if value_mode == "integrated" else 0.45,
                label=f"TTS {value_mode}" if mode == "Road" else None,
            )
            ax.barh(
                y,
                subset["cost_annual_negative"],
                height=0.32,
                color=colors["cost"],
                alpha=0.85 if value_mode == "integrated" else 0.45,
                label=f"Costs {value_mode}" if mode == "Road" else None,
            )

        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_yticks(y_base)
        ax.set_yticklabels(order, fontsize=7 if mode == "Road" else 9)
        ax.invert_yaxis()
        ax.set_xlabel("Annual value [CHF/year]")
        ax.set_ylabel("Development")
        ax.set_title(f"{mode}: annual TTS (positive) and annual costs (negative)")
        ax.grid(True, axis="x", alpha=0.25)

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=colors["tts"], alpha=0.45),
            plt.Rectangle((0, 0), 1, 1, color=colors["tts"], alpha=0.85),
            plt.Rectangle((0, 0), 1, 1, color=colors["cost"], alpha=0.45),
            plt.Rectangle((0, 0), 1, 1, color=colors["cost"], alpha=0.85),
        ]
        labels = [
            "TTS standalone annual proxy",
            "TTS integrated",
            "Costs standalone annual proxy",
            "Costs integrated",
        ]
        ax.legend(handles, labels, loc="best", frameon=False)
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode.lower()}_annual_overview_bars.png", dpi=200)
        plt.close(fig)


def build_road_access_only_annual_summary(score_df: pd.DataFrame) -> pd.DataFrame:
    road_excluded_devs = get_road_developments_without_new_link_flow()
    road_valid_devs = (
        set(score_df.loc[score_df["score_id"] == ROAD_TTS_SCORE, "development"])
        & set(score_df.loc[score_df["score_id"] == "road_construction_cost", "development"])
    ) - road_excluded_devs

    road_tts = pd.read_csv(ROAD_TTS_DETAIL)
    road_tts["development"] = normalize_development(road_tts["development"])
    road_tts["scenario"] = road_tts["scenario"].astype(str)
    road_tts = road_tts[road_tts["development"].isin(road_valid_devs)].copy()

    road_tts["access_only_minutes"] = (
        pd.to_numeric(road_tts["origin_access_savings"], errors="coerce").fillna(0.0)
        + pd.to_numeric(road_tts["destination_access_savings"], errors="coerce").fillna(0.0)
    )
    total_minutes = pd.to_numeric(road_tts["tt_savings_peak"], errors="coerce")
    total_chf_yearly = pd.to_numeric(road_tts["monetized_savings_yearly"], errors="coerce")
    road_tts["access_only_chf_yearly"] = np.where(
        total_minutes.abs() > 1e-9,
        total_chf_yearly * road_tts["access_only_minutes"] / total_minutes,
        np.nan,
    )

    cost_df = score_df[
        (score_df["mode"] == "Road")
        & score_df["score_id"].isin(ROAD_OVERVIEW_COST_SCORES)
        & score_df["development"].isin(road_valid_devs)
    ].copy()
    cost_df["cost_annual"] = cost_df.apply(annual_proxy_value, axis=1)
    cost_summary = (
        cost_df.groupby(["development", "scenario"], as_index=False)
        .agg(cost_annual=("cost_annual", "sum"))
    )

    access_summary = (
        road_tts.groupby(["development", "scenario"], as_index=False)
        .agg(
            tts_annual=("access_only_chf_yearly", "sum"),
            access_only_minutes=("access_only_minutes", "sum"),
        )
    )

    annual_summary = access_summary.merge(
        cost_summary,
        on=["development", "scenario"],
        how="left",
    )
    annual_summary["mode"] = "Road"
    annual_summary["value_mode"] = "access_only_integrated_proxy"
    annual_summary["cost_annual_negative"] = -annual_summary["cost_annual"]
    annual_summary["net_annual"] = annual_summary["tts_annual"] - annual_summary["cost_annual"]
    return annual_summary


def plot_road_access_only_overview_bars(annual_summary: pd.DataFrame, output_dir: Path) -> None:
    mode_df = annual_summary.copy()
    if mode_df.empty:
        return

    order = (
        mode_df.groupby("development", as_index=False)["net_annual"]
        .median()
        .sort_values("net_annual", ascending=False)["development"]
        .astype(str)
        .tolist()
    )
    plot_df = (
        mode_df.groupby("development", as_index=False)
        .agg(
            tts_annual=("tts_annual", "median"),
            cost_annual_negative=("cost_annual_negative", "median"),
            net_annual=("net_annual", "median"),
            access_only_minutes=("access_only_minutes", "median"),
        )
    )
    plot_df["development"] = pd.Categorical(
        plot_df["development"].astype(str),
        categories=order,
        ordered=True,
    )
    plot_df = plot_df.sort_values("development")

    y_base = np.arange(len(order))
    fig_height = max(8, len(order) * 0.18)
    fig, ax = plt.subplots(figsize=(16, fig_height))

    ax.barh(
        y_base,
        plot_df["tts_annual"],
        height=0.34,
        color="#2b8a3e",
        alpha=0.85,
        label="Access + egress TTS",
    )
    ax.barh(
        y_base,
        plot_df["cost_annual_negative"],
        height=0.34,
        color="#c92a2a",
        alpha=0.85,
        label="Costs",
    )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels(order, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Annual value [CHF/year]")
    ax.set_ylabel("Development")
    ax.set_title("Road: annual access + egress TTS (positive) and annual costs (negative)")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "road_access_only_annual_overview_bars.png", dpi=200)
    plt.close(fig)


def build_rail_component_overview(score_df: pd.DataFrame) -> pd.DataFrame:
    externality_scores = {
        "rail_accident_cost",
        "rail_airpollution_cost",
        "rail_co2_cost",
        "rail_noise_cost",
        "rail_land_consumption_cost",
    }
    rail_scores = [
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
    rail_df = score_df[
        (score_df["mode"] == "Rail") & (score_df["score_id"].isin(rail_scores))
    ].copy()
    rail_df["annual_value"] = rail_df.apply(annual_proxy_value, axis=1)

    standalone = (
        rail_df.groupby(["development", "score_id"], as_index=False)
        .agg(value=("annual_value", "median"))
        .assign(value_mode="standalone_annual_proxy")
    )
    standalone.loc[standalone["score_id"].isin(externality_scores), "value"] = np.nan
    integrated = (
        rail_df.groupby(["development", "score_id"], as_index=False)
        .agg(value=("integrated_value", "median"))
        .assign(value_mode="integrated")
    )
    rail_labels = pd.read_csv(RAIL_TOTAL_COSTS, usecols=["development", "Sline"]).copy()
    rail_labels["development"] = normalize_development(rail_labels["development"])
    rail_labels["development_label"] = (
        rail_labels["Sline"].astype(str)
        + " ("
        + rail_labels["development"].astype(str)
        + ")"
    )

    combined = pd.concat([standalone, integrated], ignore_index=True)
    combined["development"] = normalize_development(combined["development"])
    combined = combined.merge(
        rail_labels[["development", "development_label"]],
        on="development",
        how="left",
    )
    combined["development_label"] = combined["development_label"].fillna(combined["development"].astype(str))
    combined["value_mio_chf"] = combined["value"] / 1_000_000.0
    rail_order = (
        combined[
            (combined["value_mode"] == "integrated")
            & (combined["score_id"] == "rail_tts_cost")
        ][["development", "value_mio_chf"]]
        .drop_duplicates()
        .sort_values("value_mio_chf", ascending=False)
        .reset_index(drop=True)
    )
    rail_order["plot_order"] = np.arange(len(rail_order))
    combined = combined.merge(
        rail_order[["development", "plot_order"]],
        on="development",
        how="left",
    )
    return combined


def build_road_component_overview(score_df: pd.DataFrame) -> pd.DataFrame:
    road_excluded_devs = get_road_developments_without_new_link_flow()
    road_scores = [
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
    road_df = score_df[
        (score_df["mode"] == "Road")
        & (score_df["score_id"].isin(road_scores))
        & (~score_df["development"].isin(road_excluded_devs))
    ].copy()

    def road_standalone_component_value(row: pd.Series) -> float:
        score_id = row["score_id"]
        standalone_value = row["standalone_value"]

        if pd.notna(standalone_value):
            return float(standalone_value)
        return np.nan

    road_df["annual_value"] = road_df.apply(road_standalone_component_value, axis=1)

    standalone = (
        road_df.groupby(["development", "score_id"], as_index=False)
        .agg(value=("annual_value", "median"))
        .assign(value_mode="standalone_annual_proxy")
    )
    integrated = (
        road_df.groupby(["development", "score_id"], as_index=False)
        .agg(value=("integrated_value", "median"))
        .assign(value_mode="integrated")
    )

    combined = pd.concat([standalone, integrated], ignore_index=True)
    combined["development"] = normalize_development(combined["development"])
    combined["development_label"] = combined["development"].astype(str)
    combined["value_mio_chf"] = combined["value"] / 1_000_000.0
    road_order = (
        combined[
            (combined["value_mode"] == "integrated")
            & (combined["score_id"] == "road_tts_cost")
        ][["development", "value_mio_chf"]]
        .drop_duplicates()
        .sort_values("value_mio_chf", ascending=False)
        .reset_index(drop=True)
    )
    road_order["plot_order"] = np.arange(len(road_order))
    combined = combined.merge(
        road_order[["development", "plot_order"]],
        on="development",
        how="left",
    )
    return combined


def build_road_standalone_annual_cost_table(component_df: pd.DataFrame) -> pd.DataFrame:
    standalone = component_df[
        component_df["value_mode"] == "standalone_annual_proxy"
    ].copy()
    pivot = (
        standalone.pivot_table(
            index=["development", "development_label"],
            columns="score_id",
            values="value",
            aggfunc="median",
        )
        .reset_index()
    )
    for col in [
        "road_construction_cost",
        "road_maint_cost",
        "road_tts_cost",
        "road_climate_cost",
        "road_land_consumption_cost",
        "road_ecological_disruption_cost",
        "road_noise_cost",
    ]:
        if col not in pivot.columns:
            pivot[col] = np.nan

    pivot["road_externalities_total_cost"] = (
        pivot["road_climate_cost"].fillna(0.0)
        + pivot["road_land_consumption_cost"].fillna(0.0)
        + pivot["road_ecological_disruption_cost"].fillna(0.0)
        + pivot["road_noise_cost"].fillna(0.0)
    )
    pivot["road_total_cost_without_tts"] = (
        pivot["road_construction_cost"].fillna(0.0)
        + pivot["road_maint_cost"].fillna(0.0)
        + pivot["road_externalities_total_cost"].fillna(0.0)
    )
    mio_cols = [
        "road_construction_cost",
        "road_maint_cost",
        "road_tts_cost",
        "road_climate_cost",
        "road_land_consumption_cost",
        "road_ecological_disruption_cost",
        "road_noise_cost",
        "road_externalities_total_cost",
        "road_total_cost_without_tts",
    ]
    for col in mio_cols:
        pivot[f"{col}_mio_chf"] = pivot[col] / 1_000_000.0
    return pivot


def build_integrated_bcr_outputs(
    score_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    road_excluded_devs = get_road_developments_without_new_link_flow()
    road_valid_devs = (
        set(score_df.loc[score_df["score_id"] == ROAD_TTS_SCORE, "development"])
        & set(score_df.loc[score_df["score_id"] == "road_construction_cost", "development"])
    ) - road_excluded_devs

    rail_label_lookup = (
        pd.read_csv(RAIL_TOTAL_COSTS, usecols=["development", "Sline"])
        .assign(development=lambda df: normalize_development(df["development"]))
        .assign(
            development_label=lambda df: (
                df["Sline"].astype(str) + " (" + df["development"].astype(str) + ")"
            )
        )[["development", "development_label"]]
        .drop_duplicates()
    )

    road_scores = [ROAD_TTS_SCORE] + ROAD_OVERVIEW_COST_SCORES
    rail_scores = [RAIL_TTS_SCORE] + RAIL_OVERVIEW_COST_SCORES

    integrated = score_df[
        (
            (score_df["mode"] == "Road")
            & score_df["development"].isin(road_valid_devs)
            & score_df["score_id"].isin(road_scores)
        )
        | (
            (score_df["mode"] == "Rail")
            & score_df["score_id"].isin(rail_scores)
        )
    ].copy()
    integrated = integrated[pd.notna(integrated["integrated_value"])].copy()

    integrated["cost_signed"] = np.where(
        integrated["score_id"].str.endswith("_tts_cost"),
        np.nan,
        np.where(
            integrated["mode"] == "Rail",
            -pd.to_numeric(integrated["integrated_value"], errors="coerce"),
            pd.to_numeric(integrated["integrated_value"], errors="coerce"),
        ),
    )
    integrated["tts_value"] = np.where(
        integrated["score_id"].str.endswith("_tts_cost"),
        pd.to_numeric(integrated["integrated_value"], errors="coerce"),
        np.nan,
    )

    tts_by_scenario = (
        integrated[pd.notna(integrated["tts_value"])]
        .groupby(["mode", "development", "scenario"], as_index=False)
        .agg(tts_annual_chf=("tts_value", "sum"))
    )
    costs_by_scenario = (
        integrated[pd.notna(integrated["cost_signed"])]
        .groupby(["mode", "development", "scenario"], as_index=False)
        .agg(cost_annual_chf_negative=("cost_signed", "sum"))
    )

    scenario_df = tts_by_scenario.merge(
        costs_by_scenario,
        on=["mode", "development", "scenario"],
        how="outer",
    )
    scenario_df["cost_annual_chf_magnitude"] = -scenario_df["cost_annual_chf_negative"]
    scenario_df["bcr"] = np.where(
        scenario_df["cost_annual_chf_magnitude"] > 0,
        scenario_df["tts_annual_chf"] / scenario_df["cost_annual_chf_magnitude"],
        np.nan,
    )

    summary_df = (
        scenario_df.groupby(["mode", "development"], as_index=False)
        .agg(
            scenario_count=("scenario", "nunique"),
            tts_median_chf=("tts_annual_chf", "median"),
            tts_mean_chf=("tts_annual_chf", "mean"),
            cost_median_chf_negative=("cost_annual_chf_negative", "median"),
            cost_mean_chf_negative=("cost_annual_chf_negative", "mean"),
            bcr_median=("bcr", "median"),
            bcr_mean=("bcr", "mean"),
            bcr_std=("bcr", "std"),
        )
    )
    summary_df["cost_median_chf_magnitude"] = -summary_df["cost_median_chf_negative"]
    summary_df["cost_mean_chf_magnitude"] = -summary_df["cost_mean_chf_negative"]
    summary_df["tts_median_mio_chf"] = summary_df["tts_median_chf"] / 1_000_000.0
    summary_df["cost_median_mio_chf_negative"] = summary_df["cost_median_chf_negative"] / 1_000_000.0
    summary_df["cost_median_mio_chf_magnitude"] = summary_df["cost_median_chf_magnitude"] / 1_000_000.0
    summary_df["net_median_mio_chf"] = (
        summary_df["tts_median_mio_chf"] + summary_df["cost_median_mio_chf_negative"]
    )

    summary_df = summary_df.merge(rail_label_lookup, on="development", how="left")
    summary_df["development_label"] = np.where(
        summary_df["mode"] == "Rail",
        summary_df["development_label"].fillna(summary_df["development"].astype(str)),
        summary_df["development"].astype(str),
    )
    summary_df["ranking_label"] = np.where(
        summary_df["mode"] == "Rail",
        "Rail " + summary_df["development_label"].astype(str),
        "Road " + summary_df["development"].astype(str),
    )

    top10_df = (
        summary_df[summary_df["bcr_median"] > 0]
        .sort_values(["bcr_median", "tts_median_chf"], ascending=[False, False])
        .head(10)
        .copy()
    )
    top10_df["rank"] = np.arange(1, len(top10_df) + 1)
    return scenario_df, summary_df, top10_df


def plot_road_standalone_annual_bars(cost_table: pd.DataFrame, output_dir: Path) -> None:
    plot_df = cost_table.copy()
    plot_df["development"] = plot_df["development"].astype(str)
    plot_df = plot_df.sort_values("road_tts_cost_mio_chf", ascending=False)

    x = np.arange(len(plot_df))
    width = 0.62

    fig_width = max(18, len(plot_df) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    construction = plot_df["road_construction_cost_mio_chf"].fillna(0.0).to_numpy()
    maintenance = plot_df["road_maint_cost_mio_chf"].fillna(0.0).to_numpy()
    climate = plot_df["road_climate_cost_mio_chf"].fillna(0.0).to_numpy()
    land = plot_df["road_land_consumption_cost_mio_chf"].fillna(0.0).to_numpy()
    ecology = plot_df["road_ecological_disruption_cost_mio_chf"].fillna(0.0).to_numpy()
    noise = plot_df["road_noise_cost_mio_chf"].fillna(0.0).to_numpy()
    tts = plot_df["road_tts_cost_mio_chf"].fillna(0.0).to_numpy()

    ax.bar(x, construction, width=width, color=COST_COLORS["construction"], label="Construction costs")
    ax.bar(x, maintenance, width=width, bottom=construction, color=COST_COLORS["maintenance"], label="Maintenance costs")
    ax.bar(x, climate, width=width, bottom=construction + maintenance, color=COST_COLORS["co2"], label="Climate costs")
    ax.bar(x, land, width=width, bottom=construction + maintenance + climate, color=COST_COLORS["land"], label="Land consumption costs")
    ax.bar(x, ecology, width=width, bottom=construction + maintenance + climate + land, color=COST_COLORS["operating"], label="Ecological disruption costs")
    ax.bar(x, noise, width=width, bottom=construction + maintenance + climate + land + ecology, color=COST_COLORS["noise"], label="Noise costs")
    ax.bar(x, tts, width=width, color=COST_COLORS["tts"], label="Travel time savings")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["development"], rotation=90, fontsize=7)
    ax.set_ylabel("Annual value [Mio. CHF/year]")
    ax.set_xlabel("Development")
    ax.set_title("Road standalone annual costs and TTS")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "road_standalone_annual_bars.png", dpi=200)
    plt.close(fig)


def plot_integrated_bcr_top10(top10_df: pd.DataFrame, output_dir: Path) -> None:
    if top10_df.empty:
        return

    plot_df = top10_df.sort_values("bcr_median", ascending=False).copy()
    x = np.arange(len(plot_df))

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(
        x,
        plot_df["cost_median_mio_chf_negative"],
        color="#295a8a",
        alpha=0.85,
        label="Integrated annual costs",
    )
    ax.bar(
        x,
        plot_df["tts_median_mio_chf"],
        color=COST_COLORS["tts"],
        alpha=0.9,
        label="Integrated annual TTS",
    )

    for xpos, (_, row) in zip(x, plot_df.iterrows()):
        upper_anchor = max(float(row["tts_median_mio_chf"]), 0.0)
        ax.text(
            xpos,
            upper_anchor + 0.6,
            f"B/C={row['bcr_median']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["ranking_label"], rotation=90, fontsize=8)
    ax.set_ylabel("Median annual value over scenarios [Mio. CHF/year]")
    ax.set_xlabel("Development")
    ax.set_title("Top 10 integrated benefit-cost ratios across Rail and Road")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "integrated_bcr_top10_ranking.png", dpi=200)
    plt.close(fig)


def build_integrated_bcr_top10_by_mode_plot_data(
    score_df: pd.DataFrame,
    bcr_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    road_excluded_devs = get_road_developments_without_new_link_flow()
    road_valid_devs = (
        set(score_df.loc[score_df["score_id"] == ROAD_TTS_SCORE, "development"])
        & set(score_df.loc[score_df["score_id"] == "road_construction_cost", "development"])
    ) - road_excluded_devs

    rail_top = (
        bcr_summary_df[(bcr_summary_df["mode"] == "Rail") & (bcr_summary_df["bcr_median"] > 0)]
        .sort_values(["bcr_median", "tts_median_chf"], ascending=[False, False])
        .head(10)
        .copy()
    )
    rail_top["plot_group"] = "Rail top 10"

    road_top = (
        bcr_summary_df[
            (bcr_summary_df["mode"] == "Road")
            & (bcr_summary_df["bcr_median"] > 0)
            & (bcr_summary_df["development"].isin(road_valid_devs))
        ]
        .sort_values(["bcr_median", "tts_median_chf"], ascending=[False, False])
        .head(10)
        .copy()
    )
    road_top["plot_group"] = "Road top 10"

    selection = pd.concat([rail_top, road_top], ignore_index=True)
    if selection.empty:
        return selection

    selection["plot_order"] = np.arange(len(selection))
    selection["ranking_label_short"] = selection["development_label"].astype(str)

    rail_components = [
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
    road_components = [
        "road_construction_cost",
        "road_maint_cost",
        "road_accident_cost",
        "road_airpollution_cost",
        "road_co2_cost",
        "road_noise_cost",
        "road_land_consumption_cost",
        "road_tts_cost",
    ]

    integrated = score_df[pd.notna(score_df["integrated_value"])].copy()
    integrated = integrated[
        (
            (integrated["mode"] == "Rail")
            & integrated["score_id"].isin(rail_components)
            & integrated["development"].isin(rail_top["development"])
        )
        | (
            (integrated["mode"] == "Road")
            & integrated["score_id"].isin(road_components)
            & integrated["development"].isin(road_top["development"])
        )
    ].copy()

    integrated["plot_value_mio_chf"] = np.where(
        integrated["score_id"].str.endswith("_tts_cost"),
        pd.to_numeric(integrated["integrated_value"], errors="coerce") / 1_000_000.0,
        np.where(
            integrated["mode"] == "Rail",
            -pd.to_numeric(integrated["integrated_value"], errors="coerce") / 1_000_000.0,
            pd.to_numeric(integrated["integrated_value"], errors="coerce") / 1_000_000.0,
        ),
    )

    plot_components = (
        integrated.groupby(["mode", "development", "score_id"], as_index=False)
        .agg(value_mio_chf=("plot_value_mio_chf", "median"))
        .merge(
            selection[
                [
                    "mode",
                    "development",
                    "development_label",
                    "ranking_label",
                    "ranking_label_short",
                    "bcr_median",
                    "plot_group",
                    "plot_order",
                ]
            ],
            on=["mode", "development"],
            how="left",
        )
    )
    return plot_components


def plot_integrated_bcr_top10_by_mode_stacked(plot_df: pd.DataFrame, output_dir: Path) -> None:
    if plot_df.empty:
        return

    component_map = {
        "rail_construction_cost": ("Construction costs", "construction"),
        "rail_maint_cost": ("Maintenance costs", "maintenance"),
        "rail_operation_cost": ("Operating costs", "operating"),
        "rail_accident_cost": ("Accident costs", "accident"),
        "rail_airpollution_cost": ("Air pollution costs", "air"),
        "rail_co2_cost": ("CO2 costs", "co2"),
        "rail_noise_cost": ("Noise costs", "noise"),
        "rail_land_consumption_cost": ("Land consumption costs", "land"),
        "rail_tts_cost": ("TTS", "tts"),
        "road_construction_cost": ("Construction costs", "construction"),
        "road_maint_cost": ("Maintenance costs", "maintenance"),
        "road_accident_cost": ("Accident costs", "accident"),
        "road_airpollution_cost": ("Air pollution costs", "air"),
        "road_co2_cost": ("CO2 costs", "co2"),
        "road_noise_cost": ("Noise costs", "noise"),
        "road_land_consumption_cost": ("Land consumption costs", "land"),
        "road_tts_cost": ("TTS", "tts"),
    }
    plot_sequence = [
        "rail_construction_cost",
        "rail_maint_cost",
        "rail_operation_cost",
        "rail_accident_cost",
        "rail_airpollution_cost",
        "rail_co2_cost",
        "rail_noise_cost",
        "rail_land_consumption_cost",
        "rail_tts_cost",
        "road_construction_cost",
        "road_maint_cost",
        "road_accident_cost",
        "road_airpollution_cost",
        "road_co2_cost",
        "road_noise_cost",
        "road_land_consumption_cost",
        "road_tts_cost",
    ]

    ordered_devs = (
        plot_df[["mode", "development", "ranking_label_short", "bcr_median", "plot_group", "plot_order"]]
        .drop_duplicates()
        .sort_values("plot_order")
    )
    x = np.arange(len(ordered_devs))
    fig_width = max(18, len(ordered_devs) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    pivot = plot_df.pivot_table(
        index=["mode", "development", "ranking_label_short", "bcr_median", "plot_group", "plot_order"],
        columns="score_id",
        values="value_mio_chf",
        aggfunc="median",
    ).reset_index()
    pivot = pivot.sort_values("plot_order")

    negative_bottom = np.zeros(len(pivot))
    positive_bottom = np.zeros(len(pivot))

    for score_id in plot_sequence:
        if score_id not in pivot.columns:
            continue
        values = pivot[score_id].fillna(0.0).to_numpy()
        _, color_key = component_map[score_id]
        if score_id.endswith("_tts_cost"):
            ax.bar(
                x,
                values,
                width=0.7,
                bottom=positive_bottom,
                color=COST_COLORS[color_key],
                edgecolor="white",
                linewidth=0.2,
                label="Travel time savings" if score_id == "rail_tts_cost" else None,
            )
            positive_bottom += values
        else:
            ax.bar(
                x,
                values,
                width=0.7,
                bottom=negative_bottom,
                color=COST_COLORS[color_key],
                edgecolor="white",
                linewidth=0.2,
                label=component_map[score_id][0] if score_id in {
                    "rail_construction_cost",
                    "rail_maint_cost",
                    "rail_operation_cost",
                    "rail_accident_cost",
                    "rail_airpollution_cost",
                    "rail_co2_cost",
                    "rail_noise_cost",
                    "rail_land_consumption_cost",
                } else None,
            )
            negative_bottom += values

    for xpos, (_, row) in zip(x, pivot.iterrows()):
        ax.text(
            xpos,
            max(positive_bottom[xpos], 0.0) + 0.5,
            f"{row['bcr_median']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(9.5, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot["ranking_label_short"], rotation=90, fontsize=8)
    ax.set_ylabel("Median annual value over scenarios [Mio. CHF/year]")
    ax.set_xlabel("Development")
    ax.set_title("Top 10 integrated benefit-cost ratios by mode with stacked cost components")
    ax.grid(True, axis="y", alpha=0.25)

    ymax = ax.get_ylim()[1]
    ax.text(4.5, ymax * 0.95, "Rail top 10", ha="center", fontsize=11)
    ax.text(14.5, ymax * 0.95, "Road top 10", ha="center", fontsize=11)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["construction"]),
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["maintenance"]),
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["operating"]),
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["accident"]),
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["air"]),
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["co2"]),
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["noise"]),
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["land"]),
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS["tts"]),
    ]
    labels = [
        "Construction costs",
        "Maintenance costs",
        "Operating costs",
        "Accident costs",
        "Air pollution costs",
        "CO2 costs",
        "Noise costs",
        "Land consumption costs",
        "Travel time savings",
    ]
    ax.legend(handles, labels, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")

    fig.tight_layout(rect=[0, 0, 0.88, 1])
    fig.savefig(output_dir / "integrated_bcr_top10_by_mode_stacked.png", dpi=200)
    plt.close(fig)


def plot_rail_stacked_integrated_vs_standalone(component_df: pd.DataFrame, output_dir: Path) -> None:
    component_map = {
        "rail_construction_cost": ("Construction costs", "construction", True),
        "rail_maint_cost": ("Maintenance costs", "maintenance", True),
        "rail_operation_cost": ("Operating costs", "operating", True),
        "rail_accident_cost": ("Accident costs", "accident", True),
        "rail_airpollution_cost": ("Air pollution costs", "air", True),
        "rail_co2_cost": ("CO2 costs", "co2", True),
        "rail_noise_cost": ("Noise costs", "noise", True),
        "rail_land_consumption_cost": ("Land consumption costs", "land", True),
        "rail_tts_cost": ("TTS", "tts", False),
    }

    order = (
        component_df[
            (component_df["value_mode"] == "integrated")
            & (component_df["score_id"] == "rail_tts_cost")
        ]
        .sort_values("value_mio_chf", ascending=False)["development"]
        .astype(str)
        .tolist()
    )
    label_lookup = (
        component_df[["development", "development_label"]]
        .drop_duplicates()
        .assign(development=lambda df: df["development"].astype(str))
        .set_index("development")["development_label"]
        .to_dict()
    )
    pivot = component_df.pivot_table(
        index=["development", "value_mode"],
        columns="score_id",
        values="value_mio_chf",
        aggfunc="median",
    ).reset_index()
    pivot["development"] = pd.Categorical(
        pivot["development"].astype(str),
        categories=order,
        ordered=True,
    )
    pivot = pivot.sort_values(["development", "value_mode"])

    x = np.arange(len(order))
    width = 0.36
    offsets = {
        "standalone_annual_proxy": -width / 2,
        "integrated": width / 2,
    }

    fig_height = 8
    fig_width = max(16, len(order) * 0.28)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for value_mode, offset in offsets.items():
        subset = pivot[pivot["value_mode"] == value_mode].copy()
        subset = subset.set_index(subset["development"].astype(str)).reindex(order).reset_index(drop=True)
        for score_id in component_map:
            if score_id in subset.columns:
                subset[score_id] = subset[score_id].fillna(0.0)

        negative_bottom = np.zeros(len(order))
        positive_bottom = np.zeros(len(order))

        for score_id, (label, color_key, is_cost) in component_map.items():
            values = subset[score_id].to_numpy() if score_id in subset.columns else np.zeros(len(order))
            if is_cost:
                ax.bar(
                    x + offset,
                    -values,
                    width=width,
                    bottom=negative_bottom,
                    color=COST_COLORS[color_key],
                    edgecolor="white",
                    linewidth=0.2,
                    label=label if value_mode == "integrated" else None,
                )
                negative_bottom -= values
            else:
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    bottom=positive_bottom,
                    color=COST_COLORS[color_key],
                    edgecolor="white",
                    linewidth=0.2,
                    label=label if value_mode == "integrated" else None,
                )
                positive_bottom += values

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_lookup.get(dev, dev) for dev in order], rotation=90, fontsize=8)
    ax.set_ylabel("Annual value [Mio. CHF/year]")
    ax.set_xlabel("Development")
    ax.set_title("Rail: integrated vs standalone annualized stacked costs and TTS")
    ax.grid(True, axis="y", alpha=0.25)

    mode_handles = [
        plt.Rectangle((0, 0), 1, 1, color="white", ec="black", linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, color="lightgray", ec="black", linewidth=0.8),
    ]
    mode_labels = ["Standalone annualized (left bar)", "Integrated (right bar)"]
    component_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS[color_key])
        for _, color_key, _ in component_map.values()
    ]
    component_labels = [label for label, _, _ in component_map.values()]
    legend1 = ax.legend(component_handles, component_labels, frameon=False, loc="upper right")
    ax.add_artist(legend1)
    ax.legend(mode_handles, mode_labels, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "rail_stacked_integrated_vs_standalone_annual.png", dpi=200)
    plt.close(fig)


def plot_road_stacked_integrated_vs_standalone(
    component_df: pd.DataFrame,
    standalone_cost_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    component_map = {
        "road_construction_cost": ("Construction costs", "construction", True),
        "road_maint_cost": ("Maintenance costs", "maintenance", True),
        "road_accident_cost": ("Accident costs", "accident", True),
        "road_airpollution_cost": ("Air pollution costs", "air", True),
        "road_co2_cost": ("CO2 costs", "co2", True),
        "road_noise_cost": ("Noise costs", "noise", True),
        "road_land_consumption_cost": ("Land consumption costs", "land", True),
        "road_climate_cost": ("Climate costs", "co2", True),
        "road_ecological_disruption_cost": ("Ecological disruption costs", "operating", True),
        "road_tts_cost": ("TTS", "tts", False),
    }

    order = (
        component_df[
            (component_df["value_mode"] == "integrated")
            & (component_df["score_id"] == "road_tts_cost")
        ]
        .sort_values("value_mio_chf", ascending=False)["development"]
        .astype(str)
        .tolist()
    )
    label_lookup = (
        component_df[["development", "development_label"]]
        .drop_duplicates()
        .assign(development=lambda df: df["development"].astype(str))
        .set_index("development")["development_label"]
        .to_dict()
    )
    integrated_pivot = component_df[
        component_df["value_mode"] == "integrated"
    ].pivot_table(
        index=["development"],
        columns="score_id",
        values="value_mio_chf",
        aggfunc="median",
    ).reset_index()
    integrated_pivot["value_mode"] = "integrated"

    standalone_pivot = standalone_cost_table[
        [
            "development",
            "road_construction_cost_mio_chf",
            "road_maint_cost_mio_chf",
            "road_climate_cost_mio_chf",
            "road_land_consumption_cost_mio_chf",
            "road_ecological_disruption_cost_mio_chf",
            "road_noise_cost_mio_chf",
            "road_tts_cost_mio_chf",
        ]
    ].rename(
        columns={
            "road_construction_cost_mio_chf": "road_construction_cost",
            "road_maint_cost_mio_chf": "road_maint_cost",
            "road_climate_cost_mio_chf": "road_climate_cost",
            "road_land_consumption_cost_mio_chf": "road_land_consumption_cost",
            "road_ecological_disruption_cost_mio_chf": "road_ecological_disruption_cost",
            "road_noise_cost_mio_chf": "road_noise_cost",
            "road_tts_cost_mio_chf": "road_tts_cost",
        }
    )
    standalone_pivot["development"] = standalone_pivot["development"].astype(str)
    standalone_pivot["value_mode"] = "standalone_annual_proxy"
    standalone_pivot["road_accident_cost"] = np.nan
    standalone_pivot["road_airpollution_cost"] = np.nan
    standalone_pivot["road_co2_cost"] = np.nan

    pivot = pd.concat([standalone_pivot, integrated_pivot], ignore_index=True, sort=False)
    pivot["development"] = pd.Categorical(
        pivot["development"].astype(str),
        categories=order,
        ordered=True,
    )
    pivot = pivot.sort_values(["development", "value_mode"])

    x = np.arange(len(order))
    width = 0.36
    offsets = {
        "standalone_annual_proxy": -width / 2,
        "integrated": width / 2,
    }

    fig_height = 8
    fig_width = max(18, len(order) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for value_mode, offset in offsets.items():
        subset = pivot[pivot["value_mode"] == value_mode].copy()
        subset = subset.set_index(subset["development"].astype(str)).reindex(order).reset_index(drop=True)
        for score_id in component_map:
            if score_id in subset.columns:
                subset[score_id] = subset[score_id].fillna(0.0)

        negative_bottom = np.zeros(len(order))
        positive_bottom = np.zeros(len(order))

        for score_id, (label, color_key, is_cost) in component_map.items():
            values = subset[score_id].to_numpy() if score_id in subset.columns else np.zeros(len(order))
            if is_cost:
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    bottom=negative_bottom,
                    color=COST_COLORS[color_key],
                    edgecolor="white",
                    linewidth=0.2,
                    label=label if value_mode == "integrated" else None,
                )
                negative_bottom += values
            else:
                ax.bar(
                    x + offset,
                    values,
                    width=width,
                    bottom=positive_bottom,
                    color=COST_COLORS[color_key],
                    edgecolor="white",
                    linewidth=0.2,
                    label=label if value_mode == "integrated" else None,
                )
                positive_bottom += values

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([label_lookup.get(dev, dev) for dev in order], rotation=90, fontsize=7)
    ax.set_ylabel("Annual value [Mio. CHF/year]")
    ax.set_xlabel("Development")
    ax.set_title("Road: integrated vs standalone annualized stacked costs and TTS")
    ax.grid(True, axis="y", alpha=0.25)

    mode_handles = [
        plt.Rectangle((0, 0), 1, 1, color="white", ec="black", linewidth=0.8),
        plt.Rectangle((0, 0), 1, 1, color="lightgray", ec="black", linewidth=0.8),
    ]
    mode_labels = ["Standalone annualized (left bar)", "Integrated (right bar)"]
    component_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COST_COLORS[color_key])
        for _, color_key, _ in component_map.values()
    ]
    component_labels = [label for label, _, _ in component_map.values()]
    legend1 = ax.legend(component_handles, component_labels, frameon=False, loc="upper right")
    ax.add_artist(legend1)
    ax.legend(mode_handles, mode_labels, frameon=False, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_dir / "road_stacked_integrated_vs_standalone_annual.png", dpi=200)
    plt.close(fig)


def plot_road_cost_bars_integrated_vs_standalone(
    cost_table: pd.DataFrame,
    component_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    Grouped vertical bar chart: standalone annual costs vs integrated annual costs per development.

    Standalone totals come from road_standalone_annual_cost_table (costs already forced negative,
    TTS keeps its sign). Integrated totals are derived from road_component_overview by negating
    cost component values (matching the sign convention used in the stacked plot) and taking TTS
    directly. Bars are sorted by integrated net annual value (TTS + costs, descending).
    """
    ROAD_COST_SCORES = [
        "road_construction_cost",
        "road_maint_cost",
        "road_accident_cost",
        "road_airpollution_cost",
        "road_co2_cost",
        "road_noise_cost",
        "road_land_consumption_cost",
        "road_climate_cost",
        "road_ecological_disruption_cost",
    ]

    integrated_comp = component_df[component_df["value_mode"] == "integrated"].copy()
    integrated_comp["development"] = integrated_comp["development"].astype(str)

    integrated_cost = (
        integrated_comp[integrated_comp["score_id"].isin(ROAD_COST_SCORES)]
        .groupby("development", as_index=False)
        .agg(total_cost_mio_chf=("value_mio_chf", lambda s: float(-s.sum())))
    )
    integrated_tts = (
        integrated_comp[integrated_comp["score_id"] == "road_tts_cost"][["development", "value_mio_chf"]]
        .rename(columns={"value_mio_chf": "tts_mio_chf"})
    )
    integrated_df = integrated_cost.merge(integrated_tts, on="development", how="left")
    integrated_df["net_mio_chf"] = (
        integrated_df["tts_mio_chf"].fillna(0.0) + integrated_df["total_cost_mio_chf"].fillna(0.0)
    )

    standalone_df = cost_table[
        ["development", "road_total_cost_without_tts_mio_chf", "road_tts_cost_mio_chf"]
    ].copy()
    standalone_df["development"] = standalone_df["development"].astype(str)

    order = (
        integrated_df.sort_values("net_mio_chf", ascending=False)["development"]
        .tolist()
    )
    order = [d for d in order if d in standalone_df["development"].values]
    if not order:
        return

    x = np.arange(len(order))
    width = 0.36
    fig_width = max(18, len(order) * 0.16)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    sa = standalone_df.set_index("development").reindex(order)
    ig = integrated_df.set_index("development").reindex(order)

    sa_cost = sa["road_total_cost_without_tts_mio_chf"].fillna(0.0).to_numpy()
    sa_tts = sa["road_tts_cost_mio_chf"].fillna(0.0).to_numpy()
    ig_cost = ig["total_cost_mio_chf"].fillna(0.0).to_numpy()
    ig_tts = ig["tts_mio_chf"].fillna(0.0).to_numpy()

    ax.bar(x - width / 2, sa_tts, width=width, color=COST_COLORS["tts"], alpha=0.5, label="TTS standalone")
    ax.bar(x - width / 2, sa_cost, width=width, color="#c92a2a", alpha=0.5, label="Total cost standalone")
    ax.bar(x + width / 2, ig_tts, width=width, color=COST_COLORS["tts"], alpha=0.9, label="TTS integrated")
    ax.bar(x + width / 2, ig_cost, width=width, color="#c92a2a", alpha=0.9, label="Total cost integrated")

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel("Annual value [Mio. CHF/year]")
    ax.set_xlabel("Development")
    ax.set_title("Road: total annual TTS and costs — standalone vs integrated (sorted by integrated net value)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_dir / "road_total_cost_integrated_vs_standalone.png", dpi=200)
    plt.close(fig)


def build_tts_frame(score_df: pd.DataFrame) -> pd.DataFrame:
    road_excluded_devs = get_road_developments_without_new_link_flow()
    road_scenarios = sorted(
        score_df.loc[score_df["mode"] == "Road", "scenario"].dropna().unique().tolist(),
        key=scenario_number,
    )
    road_valid_devs = sorted((
        set(score_df.loc[score_df["score_id"] == ROAD_TTS_SCORE, "development"])
        & set(score_df.loc[score_df["score_id"] == "road_construction_cost", "development"])
    ) - road_excluded_devs)

    road_tts = pd.read_csv(ROAD_TTS_DETAIL)
    road_tts["development"] = normalize_development(road_tts["development"])
    road_tts["scenario"] = road_tts["scenario"].astype(str)
    road_tts = road_tts[
        road_tts["scenario"].isin(road_scenarios)
        & road_tts["development"].isin(road_valid_devs)
    ].copy()
    road_tts = road_tts.rename(
        columns={
            "tt_savings_peak": "tts_minutes",
            "monetized_savings_yearly": "tts_chf_yearly",
        }
    )
    road_tts["mode"] = "Road"
    road_tts["tts_hours"] = road_tts["tts_minutes"] / 60.0
    road_tts["tts_minutes_metric"] = "weighted_peak_minutes"

    rail_tts = pd.read_csv(RAIL_TTS)
    rail_tts["development"] = normalize_development(rail_tts["development"])
    rail_tts["scenario"] = rail_tts["scenario"].astype(str)
    rail_tts["year"] = pd.to_numeric(rail_tts["year"], errors="coerce")
    rail_tts = rail_tts[rail_tts["year"] == common_cost_parameters.prognosis_year].copy()
    rail_tts = rail_tts[rail_tts["scenario"].isin(score_df.loc[score_df["mode"] == "Rail", "scenario"].unique())].copy()
    rail_tts["mode"] = "Rail"
    rail_tts["tts_hours"] = pd.to_numeric(rail_tts["tt_savings_daily"], errors="coerce")
    rail_tts["tts_minutes"] = rail_tts["tts_hours"] * 60.0
    rail_tts["tts_chf_yearly"] = pd.to_numeric(rail_tts["monetized_savings_yearly"], errors="coerce")
    rail_tts["tts_minutes_metric"] = "daily_minutes"

    keep_cols = ["mode", "development", "scenario", "tts_hours", "tts_minutes", "tts_chf_yearly", "tts_minutes_metric"]
    return pd.concat(
        [
            road_tts[keep_cols],
            rail_tts[keep_cols],
        ],
        ignore_index=True,
    )


def summarize_tts_by_development(tts_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        tts_df.groupby(["mode", "development"], as_index=False)
        .agg(
            scenario_count=("scenario", "nunique"),
            mean_tts_minutes=("tts_minutes", "mean"),
            median_tts_minutes=("tts_minutes", "median"),
            std_tts_minutes=("tts_minutes", "std"),
            min_tts_minutes=("tts_minutes", "min"),
            max_tts_minutes=("tts_minutes", "max"),
            q1_tts_minutes=("tts_minutes", lambda s: s.quantile(0.25)),
            q3_tts_minutes=("tts_minutes", lambda s: s.quantile(0.75)),
            mean_tts_chf_yearly=("tts_chf_yearly", "mean"),
            median_tts_chf_yearly=("tts_chf_yearly", "median"),
            std_tts_chf_yearly=("tts_chf_yearly", "std"),
            n_positive=("tts_minutes", lambda s: int((s > 0).sum())),
            n_negative=("tts_minutes", lambda s: int((s < 0).sum())),
        )
    )
    summary["iqr_tts_minutes"] = summary["q3_tts_minutes"] - summary["q1_tts_minutes"]
    return summary.sort_values(["mode", "median_tts_minutes"], ascending=[True, False])


def plot_tts_boxplots(tts_df: pd.DataFrame, output_dir: Path) -> None:
    for mode in ["Road", "Rail"]:
        subset = tts_df[tts_df["mode"] == mode].copy()
        if subset.empty:
            continue
        order = (
            subset.groupby("development", as_index=False)["tts_minutes"]
            .median()
            .sort_values("tts_minutes", ascending=False)["development"]
            .astype(str)
            .tolist()
        )
        groups = [subset.loc[subset["development"].astype(str) == dev, "tts_minutes"].dropna().values for dev in order]
        fig_width = max(12, len(order) * (0.13 if mode == "Road" else 0.22))
        fig, ax = plt.subplots(figsize=(fig_width, 7))
        ax.boxplot(groups, showfliers=False, patch_artist=True, widths=0.65)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(np.arange(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=90, fontsize=6 if mode == "Road" else 8)
        ax.set_ylabel("TTS [minutes]")
        ax.set_xlabel("Development")
        ax.set_title(f"{mode}: TTS distribution across scenarios")
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / f"{mode.lower()}_tts_boxplot_all_developments.png", dpi=200)
        plt.close(fig)


def plot_tts_mean_std(tts_summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    for ax, mode in zip(axes, ["Road", "Rail"]):
        subset = tts_summary[tts_summary["mode"] == mode].copy()
        ax.scatter(
            subset["mean_tts_minutes"],
            subset["std_tts_minutes"],
            s=20,
            alpha=0.75,
            color="#1c7ed6",
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(f"{mode}: mean vs std of TTS across scenarios")
        ax.set_xlabel("Mean TTS [minutes]")
        ax.set_ylabel("Std TTS [minutes]")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "tts_mean_vs_std.png", dpi=200)
    plt.close(fig)


def analyze_vkm(road_valid_devs: set[str]) -> pd.DataFrame:
    vkm = pd.read_csv(ROAD_VKM_MONETIZATION)
    vkm["development"] = normalize_development(vkm["development"])
    vkm["scenario"] = vkm["scenario"].astype(str)
    vkm = vkm[vkm["development"].isin(road_valid_devs)].copy()
    summary = (
        vkm.groupby("development", as_index=False)
        .agg(
            scenario_count=("scenario", "nunique"),
            mean_delta_vkm_annualized=("delta_vkm_annualized", "mean"),
            median_delta_vkm_annualized=("delta_vkm_annualized", "median"),
            std_delta_vkm_annualized=("delta_vkm_annualized", "std"),
            mean_delta_vkm_peak_hour=("delta_vkm_peak_hour", "mean"),
            median_delta_vkm_peak_hour=("delta_vkm_peak_hour", "median"),
            std_delta_vkm_peak_hour=("delta_vkm_peak_hour", "std"),
        )
    )
    return vkm, summary


def build_road_tts_components(road_valid_devs: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    road_tts = pd.read_csv(ROAD_TTS_DETAIL)
    road_tts["development"] = normalize_development(road_tts["development"])
    road_tts["scenario"] = road_tts["scenario"].astype(str)
    road_tts = road_tts[road_tts["development"].isin(road_valid_devs)].copy()
    road_tts["access_only_savings"] = (
        pd.to_numeric(road_tts["origin_access_savings"], errors="coerce").fillna(0.0)
        + pd.to_numeric(road_tts["destination_access_savings"], errors="coerce").fillna(0.0)
    )

    component_long = pd.concat(
        [
            road_tts[["development", "scenario", "tt_savings_peak"]].rename(
                columns={"tt_savings_peak": "tts_minutes"}
            ).assign(component="total"),
            road_tts[["development", "scenario", "network_savings"]].rename(
                columns={"network_savings": "tts_minutes"}
            ).assign(component="network"),
            road_tts[["development", "scenario", "access_only_savings"]].rename(
                columns={"access_only_savings": "tts_minutes"}
            ).assign(component="access_only"),
            road_tts[["development", "scenario", "origin_access_savings"]].rename(
                columns={"origin_access_savings": "tts_minutes"}
            ).assign(component="origin_access"),
            road_tts[["development", "scenario", "destination_access_savings"]].rename(
                columns={"destination_access_savings": "tts_minutes"}
            ).assign(component="destination_access"),
        ],
        ignore_index=True,
    )

    summary = (
        component_long.groupby(["development", "component"], as_index=False)
        .agg(
            scenario_count=("scenario", "nunique"),
            mean_tts_minutes=("tts_minutes", "mean"),
            median_tts_minutes=("tts_minutes", "median"),
            std_tts_minutes=("tts_minutes", "std"),
            min_tts_minutes=("tts_minutes", "min"),
            max_tts_minutes=("tts_minutes", "max"),
            n_positive=("tts_minutes", lambda s: int((s > 0).sum())),
            n_negative=("tts_minutes", lambda s: int((s < 0).sum())),
        )
    )
    return component_long, summary


def plot_road_tts_component_lines(component_df: pd.DataFrame, output_dir: Path) -> None:
    total_summary = component_df[component_df["component"] == "total"].groupby("development", as_index=False).agg(
        median_tts_minutes=("tts_minutes", "median"),
        n_positive=("tts_minutes", lambda s: int((s > 0).sum())),
        n_negative=("tts_minutes", lambda s: int((s < 0).sum())),
    )

    top_positive = (
        total_summary.sort_values("median_tts_minutes", ascending=False)
        .head(8)["development"]
        .astype(str)
        .tolist()
    )
    top_negative = (
        total_summary.sort_values("median_tts_minutes", ascending=True)
        .head(8)["development"]
        .astype(str)
        .tolist()
    )
    sign_switching = (
        total_summary[(total_summary["n_positive"] > 0) & (total_summary["n_negative"] > 0)]
        .sort_values("median_tts_minutes", ascending=False)["development"]
        .astype(str)
        .tolist()
    )
    selected_devs = []
    for dev in top_positive + top_negative + sign_switching[:8]:
        if dev not in selected_devs:
            selected_devs.append(dev)

    selected_df = component_df[component_df["development"].astype(str).isin(selected_devs)].copy()
    selected_df["scenario_num"] = selected_df["scenario"].map(scenario_number)

    for component in ["total", "network", "access_only"]:
        subset = selected_df[selected_df["component"] == component].copy()
        if subset.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 8))
        for dev, dev_df in subset.groupby("development"):
            dev_df = dev_df.sort_values("scenario_num")
            ax.plot(
                dev_df["scenario_num"],
                dev_df["tts_minutes"],
                linewidth=1.6,
                marker="o",
                markersize=3.5,
                alpha=0.9,
                label=str(dev),
            )

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("TTS [minutes]")
        ax.set_title(f"Road {component} TTS across scenarios for selected developments")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"road_tts_{component}_scenario_lines.png", dpi=200)
        plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    score_df, duplicate_counts = load_score_results()
    coverage = build_coverage_summary(score_df)

    annual_long, annual_summary = build_annual_overview(score_df)
    plot_overview_bars(annual_summary, OUTPUT_DIR)
    integrated_bcr_scenario, integrated_bcr_summary, integrated_bcr_top10 = build_integrated_bcr_outputs(score_df)
    plot_integrated_bcr_top10(integrated_bcr_top10, OUTPUT_DIR)
    integrated_bcr_top10_by_mode_plot = build_integrated_bcr_top10_by_mode_plot_data(
        score_df,
        integrated_bcr_summary,
    )
    plot_integrated_bcr_top10_by_mode_stacked(integrated_bcr_top10_by_mode_plot, OUTPUT_DIR)
    road_access_only_annual_summary = build_road_access_only_annual_summary(score_df)
    plot_road_access_only_overview_bars(road_access_only_annual_summary, OUTPUT_DIR)
    road_component_overview = build_road_component_overview(score_df)
    road_standalone_annual_costs = build_road_standalone_annual_cost_table(road_component_overview)
    plot_road_stacked_integrated_vs_standalone(
        road_component_overview,
        road_standalone_annual_costs,
        OUTPUT_DIR,
    )
    plot_road_standalone_annual_bars(road_standalone_annual_costs, OUTPUT_DIR)
    plot_road_cost_bars_integrated_vs_standalone(road_standalone_annual_costs, road_component_overview, OUTPUT_DIR)
    rail_component_overview = build_rail_component_overview(score_df)
    plot_rail_stacked_integrated_vs_standalone(rail_component_overview, OUTPUT_DIR)

    tts_df = build_tts_frame(score_df)
    tts_summary = summarize_tts_by_development(tts_df)
    plot_tts_boxplots(tts_df, OUTPUT_DIR)
    plot_tts_mean_std(tts_summary, OUTPUT_DIR)

    road_excluded_devs = get_road_developments_without_new_link_flow()
    road_valid_devs = (
        set(
        score_df.loc[score_df["score_id"] == ROAD_TTS_SCORE, "development"]
        ) & set(
        score_df.loc[score_df["score_id"] == "road_construction_cost", "development"]
        )
    ) - road_excluded_devs
    road_vkm_df, road_vkm_summary = analyze_vkm(road_valid_devs)
    road_tts_components_df, road_tts_components_summary = build_road_tts_components(road_valid_devs)
    plot_road_tts_component_lines(road_tts_components_df, OUTPUT_DIR)

    duplicate_counts.to_csv(OUTPUT_DIR / "score_result_duplicate_counts.csv", index=False)
    coverage.to_csv(OUTPUT_DIR / "score_result_coverage_summary.csv", index=False)
    annual_long.to_csv(OUTPUT_DIR / "annual_overview_components_long.csv", index=False)
    annual_summary.to_csv(OUTPUT_DIR / "annual_overview_by_development_scenario.csv", index=False)
    integrated_bcr_scenario.to_csv(OUTPUT_DIR / "integrated_bcr_by_development_scenario.csv", index=False)
    integrated_bcr_summary.to_csv(OUTPUT_DIR / "integrated_bcr_summary_by_development.csv", index=False)
    integrated_bcr_top10.to_csv(OUTPUT_DIR / "integrated_bcr_top10.csv", index=False)
    integrated_bcr_top10_by_mode_plot.to_csv(
        OUTPUT_DIR / "integrated_bcr_top10_by_mode_plot_data.csv",
        index=False,
    )
    road_access_only_annual_summary.to_csv(
        OUTPUT_DIR / "road_access_only_annual_overview_by_development_scenario.csv",
        index=False,
    )
    road_component_overview.to_csv(
        OUTPUT_DIR / "road_component_overview_integrated_vs_standalone.csv",
        index=False,
    )
    road_standalone_annual_costs.to_csv(
        OUTPUT_DIR / "road_standalone_annual_cost_table.csv",
        index=False,
    )
    rail_component_overview.to_csv(
        OUTPUT_DIR / "rail_component_overview_integrated_vs_standalone.csv",
        index=False,
    )
    tts_df.to_csv(OUTPUT_DIR / "tts_scenario_values.csv", index=False)
    tts_summary.to_csv(OUTPUT_DIR / "tts_summary_by_development.csv", index=False)
    road_vkm_df.to_csv(OUTPUT_DIR / "road_vkm_scenario_values.csv", index=False)
    road_vkm_summary.to_csv(OUTPUT_DIR / "road_vkm_summary_by_development.csv", index=False)
    pd.DataFrame({"development": sorted(road_excluded_devs)}).to_csv(
        OUTPUT_DIR / "road_developments_without_new_link_flow.csv",
        index=False,
    )
    road_tts_components_df.to_csv(OUTPUT_DIR / "road_tts_components_scenario_values.csv", index=False)
    road_tts_components_summary.to_csv(OUTPUT_DIR / "road_tts_components_summary_by_development.csv", index=False)
    save_analysis_notes(OUTPUT_DIR / "analysis_notes.txt")

    print(f"Wrote analysis outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
