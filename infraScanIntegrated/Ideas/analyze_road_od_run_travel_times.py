from __future__ import annotations

from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


BASE = Path("/cluster/home/lkuehner/MSc_Thesis")
RUNS_DIR = BASE / "data/infraScanRoad/traffic_flow/od/runs"
OUTPUT_DIR = BASE / "infraScan/infraScanIntegrated/outputs/road_od_run_travel_time_checks"


def scenario_number(value: str) -> int:
    match = re.search(r"(\d+)$", str(value))
    return int(match.group(1)) if match else -1


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce")
    weights = pd.to_numeric(weights, errors="coerce").clip(lower=0)
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def summarize_od_tt(df: pd.DataFrame, run: str, case_type: str) -> pd.DataFrame:
    work = df.copy()
    work["travel_time_min"] = pd.to_numeric(work["travel_time"], errors="coerce") * 60.0
    work["demand"] = pd.to_numeric(work["demand"], errors="coerce")
    work["development"] = work["development"].astype(str).str.replace(r"\.0$", "", regex=True)
    work["case_type"] = case_type
    work["run"] = run

    rows = []
    for keys, group in work.groupby(["run", "case_type", "development", "scenario"], dropna=False):
        rows.append(
            {
                "run": keys[0],
                "case_type": keys[1],
                "development": keys[2],
                "scenario": keys[3],
                "od_pairs": len(group),
                "total_demand": group["demand"].sum(),
                "mean_tt_min_unweighted": group["travel_time_min"].mean(),
                "median_tt_min_unweighted": group["travel_time_min"].median(),
                "p95_tt_min_unweighted": group["travel_time_min"].quantile(0.95),
                "max_tt_min_unweighted": group["travel_time_min"].max(),
                "mean_tt_min_weighted": weighted_mean(group["travel_time_min"], group["demand"]),
                "std_tt_min_unweighted": group["travel_time_min"].std(),
            }
        )
    return pd.DataFrame(rows)


def load_all_runs() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    savings = []

    for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
        run = run_dir.name
        sq_path = run_dir / "status_quo_od_tt.csv"
        dev_path = run_dir / "developments_od_tt.csv"
        savings_path = run_dir / "od_tt_savings_detailed.csv"

        if sq_path.exists():
            summaries.append(summarize_od_tt(pd.read_csv(sq_path), run, "status_quo"))
        if dev_path.exists():
            summaries.append(summarize_od_tt(pd.read_csv(dev_path), run, "development"))
        if savings_path.exists():
            current = pd.read_csv(savings_path)
            current["run"] = run
            current["development"] = current["development"].astype(str).str.replace(r"\.0$", "", regex=True)
            savings.append(current)

    return pd.concat(summaries, ignore_index=True), pd.concat(savings, ignore_index=True)


def make_plots(summary: pd.DataFrame, savings: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    common_scenarios = sorted(
        set.intersection(*(set(summary.loc[summary["run"] == run, "scenario"]) for run in summary["run"].unique())),
        key=scenario_number,
    )
    common_devs = sorted(
        set.intersection(
            *(set(summary.loc[(summary["run"] == run) & (summary["case_type"] == "development"), "development"]) for run in summary["run"].unique())
        ),
        key=lambda value: int(value) if str(value).isdigit() else 999999,
    )

    common_summary = summary[
        summary["scenario"].isin(common_scenarios)
        & (
            (summary["case_type"] == "status_quo")
            | (summary["development"].isin(common_devs))
        )
    ].copy()
    common_savings = savings[
        savings["scenario"].isin(common_scenarios)
        & savings["development"].isin(common_devs)
    ].copy()

    summary.to_csv(OUTPUT_DIR / "od_travel_time_summary_all_runs.csv", index=False)
    common_summary.to_csv(OUTPUT_DIR / "od_travel_time_summary_common_cases.csv", index=False)
    savings.to_csv(OUTPUT_DIR / "tts_savings_all_runs.csv", index=False)
    common_savings.to_csv(OUTPUT_DIR / "tts_savings_common_cases.csv", index=False)

    run_order = sorted(summary["run"].unique())
    scenario_order = common_scenarios

    sq = common_summary[common_summary["case_type"] == "status_quo"].copy()
    sq["scenario"] = pd.Categorical(sq["scenario"], categories=scenario_order, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    sns.lineplot(data=sq, x="scenario", y="mean_tt_min_weighted", hue="run", marker="o", hue_order=run_order, ax=axes[0])
    axes[0].set_title("Status quo: demand-weighted OD travel time")
    axes[0].set_ylabel("Travel time [min]")
    axes[0].tick_params(axis="x", rotation=45)
    sns.lineplot(data=sq, x="scenario", y="mean_tt_min_unweighted", hue="run", marker="o", hue_order=run_order, ax=axes[1])
    axes[1].set_title("Status quo: unweighted OD travel time")
    axes[1].set_ylabel("Travel time [min]")
    axes[1].tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "status_quo_weighted_vs_unweighted_tt_by_run.png", dpi=220, bbox_inches="tight")
    plt.close()

    dev = common_summary[common_summary["case_type"] == "development"].copy()
    dev["scenario"] = pd.Categorical(dev["scenario"], categories=scenario_order, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    sns.boxplot(data=dev, x="run", y="mean_tt_min_weighted", ax=axes[0], color="#9ecae1")
    axes[0].set_title("Development cases: demand-weighted mean OD TT")
    axes[0].set_ylabel("Travel time [min]")
    axes[0].tick_params(axis="x", rotation=30)
    sns.boxplot(data=dev, x="run", y="mean_tt_min_unweighted", ax=axes[1], color="#fdae6b")
    axes[1].set_title("Development cases: unweighted mean OD TT")
    axes[1].set_ylabel("Travel time [min]")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "development_weighted_vs_unweighted_tt_boxplot_by_run.png", dpi=220, bbox_inches="tight")
    plt.close()

    common_savings["scenario"] = pd.Categorical(common_savings["scenario"], categories=scenario_order, ordered=True)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=False)
    sns.lineplot(
        data=common_savings,
        x="scenario",
        y="tt_savings_peak",
        hue="development",
        style="run",
        marker="o",
        ax=axes[0],
    )
    axes[0].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_title("TTS by development/scenario/run")
    axes[0].set_ylabel("TTS [person-min]")
    axes[0].tick_params(axis="x", rotation=45)
    sns.boxplot(data=common_savings, x="run", y="tt_savings_peak", ax=axes[1], color="#a1d99b")
    axes[1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1].set_title("TTS distribution across common dev-scenario cases")
    axes[1].set_ylabel("TTS [person-min]")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tts_common_cases_by_run.png", dpi=220, bbox_inches="tight")
    plt.close()

    comparison = (
        common_summary.pivot_table(
            index=["case_type", "development", "scenario"],
            columns="run",
            values=["mean_tt_min_weighted", "mean_tt_min_unweighted"],
            aggfunc="first",
        )
        .reset_index()
    )
    comparison.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in comparison.columns]
    comparison.to_csv(OUTPUT_DIR / "od_tt_common_case_wide_comparison.csv", index=False)

    print(f"Saved outputs to: {OUTPUT_DIR}")
    print("Runs:", ", ".join(run_order))
    print("Common scenarios:", ", ".join(common_scenarios))
    print("Common developments:", ", ".join(common_devs))
    print()
    print("Status quo summary, common scenarios:")
    print(
        sq.groupby("run")[["mean_tt_min_weighted", "mean_tt_min_unweighted", "p95_tt_min_unweighted"]]
        .mean()
        .round(3)
        .to_string()
    )
    print()
    print("Development summary, common cases:")
    print(
        dev.groupby("run")[["mean_tt_min_weighted", "mean_tt_min_unweighted", "p95_tt_min_unweighted"]]
        .mean()
        .round(3)
        .to_string()
    )
    print()
    print("TTS summary, common cases:")
    print(
        common_savings.groupby("run")["tt_savings_peak"]
        .agg(["mean", "median", "std", "min", "max"])
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    summary_df, savings_df = load_all_runs()
    make_plots(summary_df, savings_df)
