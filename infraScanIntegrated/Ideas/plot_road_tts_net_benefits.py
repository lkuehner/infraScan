from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot road travel-time-savings distributions by development and "
            "net-benefit distributions by scenario from existing outputs."
        )
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help="MSc_Thesis root containing data/ and infraScan/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "road_tts_net_benefits",
        help="Directory for PNG and CSV outputs.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional scenario filter, e.g. scenario_19 scenario_29 scenario_30.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Number of developments shown in the per-development TTS plots.",
    )
    parser.add_argument(
        "--show-lowest",
        action="store_true",
        help="Show the lowest mean-TTS developments instead of the highest.",
    )
    parser.add_argument(
        "--all-developments",
        action="store_true",
        help="Show all developments in the per-development TTS plots.",
    )
    return parser.parse_args()


def _format_axis(ax) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_tts(data_root: Path, scenarios: list[str] | None) -> pd.DataFrame:
    path = data_root / "data" / "infraScanRoad" / "traffic_flow" / "od" / "od_tt_savings_detailed.csv"
    df = pd.read_csv(path)
    required = {
        "development",
        "scenario",
        "origin_access_savings",
        "network_savings",
        "destination_access_savings",
        "tt_savings_peak",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["development"] = df["development"].astype(int)
    df["scenario"] = df["scenario"].astype(str)
    if scenarios:
        df = df[df["scenario"].isin(scenarios)].copy()
    df["access_savings"] = df["origin_access_savings"] + df["destination_access_savings"]
    df["total_tts"] = df["tt_savings_peak"]

    demand_path = data_root / "data" / "infraScanRoad" / "traffic_flow" / "od" / "status_quo_od_tt.csv"
    demand = pd.read_csv(demand_path)
    demand["scenario"] = demand["scenario"].astype(str)
    demand = demand.groupby("scenario", as_index=False)["demand"].sum().rename(
        columns={"demand": "status_quo_total_demand"}
    )
    df = df.merge(demand, on="scenario", how="left")
    for col in [
        "origin_access_savings",
        "network_savings",
        "destination_access_savings",
        "access_savings",
        "total_tts",
    ]:
        df[f"{col}_per_demand"] = df[col] / df["status_quo_total_demand"]
    return df


def load_net_benefits(data_root: Path, scenarios: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    path = data_root / "data" / "infraScanRoad" / "costs" / "total_costs_od.csv"
    df = pd.read_csv(path)
    if "ID_new" not in df.columns:
        raise ValueError(f"{path} is missing ID_new.")

    available = [c for c in df.columns if c.startswith("total_scenario_")]
    if scenarios:
        scenario_cols = [f"total_{s}" for s in scenarios if f"total_{s}" in df.columns]
    else:
        scenario_cols = available

    if not scenario_cols:
        raise ValueError(
            f"No scenario total columns found in {path}. Available scenario columns: {available}"
        )

    return df, scenario_cols


def select_developments(summary: pd.DataFrame, top_n: int, all_developments: bool, show_lowest: bool) -> list[int]:
    ordered = summary.sort_values("mean_total_tts", ascending=show_lowest)
    if not all_developments:
        ordered = ordered.head(top_n)
    return ordered["development"].astype(int).tolist()


def plot_tts_boxplot_by_development(
    tts: pd.DataFrame,
    selected_developments: list[int],
    output_dir: Path,
) -> None:
    plot_df = tts[tts["development"].isin(selected_developments)].copy()
    order = (
        plot_df.groupby("development")["total_tts"]
        .mean()
        .sort_values(ascending=False)
        .index.astype(int)
        .tolist()
    )
    series = [plot_df.loc[plot_df["development"] == dev, "total_tts"].to_numpy() for dev in order]
    means = [np.nanmean(values) for values in series]

    width = max(12, 0.28 * len(order))
    fig, ax = plt.subplots(figsize=(width, 6), dpi=240)
    bp = ax.boxplot(
        series,
        patch_artist=True,
        showmeans=False,
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.7},
        medianprops={"color": "black", "linewidth": 1.2},
        whiskerprops={"color": "black", "linewidth": 0.7},
        capprops={"color": "black", "linewidth": 0.7},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_edgecolor("black")
        patch.set_linewidth(0.7)

    ax.scatter(np.arange(1, len(order) + 1), means, color="#b2182b", s=14, zorder=4, label="mean")
    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xticks(np.arange(1, len(order) + 1))
    ax.set_xticklabels([str(dev) for dev in order], rotation=90)
    ax.set_xlabel("Development ID")
    ax.set_ylabel("Peak generalized travel-time savings")
    ax.set_title("Road TTS by development across scenarios")
    ax.legend(loc="upper right")
    _format_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "tts_boxplot_by_development.png", bbox_inches="tight")
    plt.close(fig)


def _stack_bars(ax, x: np.ndarray, values: pd.DataFrame, colors: dict[str, str]) -> None:
    positive_bottom = np.zeros(len(values))
    negative_bottom = np.zeros(len(values))
    for col in ["origin_access_savings", "network_savings", "destination_access_savings"]:
        y = values[col].to_numpy(dtype=float)
        bottoms = np.where(y >= 0, positive_bottom, negative_bottom)
        ax.bar(x, y, bottom=bottoms, color=colors[col], label=col.replace("_savings", "").replace("_", " "))
        positive_bottom += np.where(y >= 0, y, 0)
        negative_bottom += np.where(y < 0, y, 0)


def plot_mean_tts_components(
    tts_summary: pd.DataFrame,
    selected_developments: list[int],
    output_dir: Path,
) -> None:
    comp = tts_summary[tts_summary["development"].isin(selected_developments)].copy()
    comp = comp.sort_values("mean_total_tts", ascending=False)
    x = np.arange(len(comp))

    width = max(12, 0.28 * len(comp))
    fig, ax = plt.subplots(figsize=(width, 6), dpi=240)
    colors = {
        "origin_access_savings": "#7fc97f",
        "network_savings": "#386cb0",
        "destination_access_savings": "#beaed4",
    }
    _stack_bars(ax, x, comp, colors)
    ax.plot(x, comp["mean_total_tts"], color="#b2182b", marker="o", linewidth=1.0, markersize=3, label="total mean")
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["development"].astype(str), rotation=90)
    ax.set_xlabel("Development ID")
    ax.set_ylabel("Mean peak generalized travel-time savings")
    ax.set_title("Mean TTS components across scenarios")
    ax.legend(loc="upper right", ncols=2)
    _format_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_tts_components_by_development.png", bbox_inches="tight")
    plt.close(fig)


def plot_mean_tts_components_per_demand(
    tts_summary: pd.DataFrame,
    selected_developments: list[int],
    output_dir: Path,
) -> None:
    comp = tts_summary[tts_summary["development"].isin(selected_developments)].copy()
    comp = comp.sort_values("mean_total_tts_per_demand", ascending=False)
    x = np.arange(len(comp))

    width = max(12, 0.28 * len(comp))
    fig, ax = plt.subplots(figsize=(width, 6), dpi=240)
    colors = {
        "origin_access_savings_per_demand": "#7fc97f",
        "network_savings_per_demand": "#386cb0",
        "destination_access_savings_per_demand": "#beaed4",
    }
    positive_bottom = np.zeros(len(comp))
    negative_bottom = np.zeros(len(comp))
    labels = {
        "origin_access_savings_per_demand": "origin access",
        "network_savings_per_demand": "network",
        "destination_access_savings_per_demand": "destination access",
    }
    for col in labels:
        y = comp[col].to_numpy(dtype=float)
        bottoms = np.where(y >= 0, positive_bottom, negative_bottom)
        ax.bar(x, y, bottom=bottoms, color=colors[col], label=labels[col])
        positive_bottom += np.where(y >= 0, y, 0)
        negative_bottom += np.where(y < 0, y, 0)

    ax.plot(
        x,
        comp["mean_total_tts_per_demand"],
        color="#b2182b",
        marker="o",
        linewidth=1.0,
        markersize=3,
        label="total mean",
    )
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(comp["development"].astype(str), rotation=90)
    ax.set_xlabel("Development ID")
    ax.set_ylabel("Mean peak savings per status-quo demand unit")
    ax.set_title("Mean TTS components per demand across scenarios")
    ax.legend(loc="upper right", ncols=2)
    _format_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "mean_tts_components_per_demand_by_development.png", bbox_inches="tight")
    plt.close(fig)


def plot_net_benefit_distribution(
    costs: pd.DataFrame,
    scenario_cols: list[str],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=240)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#666666", "#e7298a"]

    long_rows = []
    for i, col in enumerate(scenario_cols):
        values = pd.to_numeric(costs[col], errors="coerce").dropna() / 1_000_000.0
        values = values.sort_values().reset_index(drop=True)
        rank = np.arange(1, len(values) + 1)
        scenario = col.replace("total_", "")
        ax.plot(rank, values, color=colors[i % len(colors)], linewidth=1.8, label=scenario)
        long_rows.append(pd.DataFrame({"scenario": scenario, "rank": rank, "net_benefit_mio_chf": values}))

    ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
    ax.set_xlabel("Developments sorted by net benefit")
    ax.set_ylabel("Net benefit [Mio. CHF]")
    ax.set_title("Distribution of net benefits for all developments")
    ax.legend(title="Scenario")
    _format_axis(ax)
    fig.tight_layout()
    fig.savefig(output_dir / "net_benefit_distribution_lines.png", bbox_inches="tight")
    plt.close(fig)

    pd.concat(long_rows, ignore_index=True).to_csv(
        output_dir / "net_benefit_distribution_long.csv",
        index=False,
    )


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = args.scenarios if args.scenarios else None
    tts = load_tts(data_root, scenarios)
    tts_summary = (
        tts.groupby("development", as_index=False)
        .agg(
            mean_total_tts=("total_tts", "mean"),
            median_total_tts=("total_tts", "median"),
            std_total_tts=("total_tts", "std"),
            min_total_tts=("total_tts", "min"),
            max_total_tts=("total_tts", "max"),
            origin_access_savings=("origin_access_savings", "mean"),
            network_savings=("network_savings", "mean"),
            destination_access_savings=("destination_access_savings", "mean"),
            access_savings=("access_savings", "mean"),
            origin_access_savings_per_demand=("origin_access_savings_per_demand", "mean"),
            network_savings_per_demand=("network_savings_per_demand", "mean"),
            destination_access_savings_per_demand=("destination_access_savings_per_demand", "mean"),
            access_savings_per_demand=("access_savings_per_demand", "mean"),
            mean_total_tts_per_demand=("total_tts_per_demand", "mean"),
            n_scenarios=("scenario", "nunique"),
        )
        .sort_values("mean_total_tts", ascending=False)
    )
    tts_summary.to_csv(output_dir / "tts_mean_by_development.csv", index=False)

    selected_developments = select_developments(
        summary=tts_summary,
        top_n=args.top_n,
        all_developments=args.all_developments,
        show_lowest=args.show_lowest,
    )

    plot_tts_boxplot_by_development(tts, selected_developments, output_dir)
    plot_mean_tts_components(tts_summary, selected_developments, output_dir)
    plot_mean_tts_components_per_demand(tts_summary, selected_developments, output_dir)

    costs, scenario_cols = load_net_benefits(data_root, scenarios)
    plot_net_benefit_distribution(costs, scenario_cols, output_dir)

    print(f"Output directory: {output_dir}")
    print("Created:")
    print(f"  {output_dir / 'tts_boxplot_by_development.png'}")
    print(f"  {output_dir / 'mean_tts_components_by_development.png'}")
    print(f"  {output_dir / 'mean_tts_components_per_demand_by_development.png'}")
    print(f"  {output_dir / 'net_benefit_distribution_lines.png'}")
    print(f"  {output_dir / 'tts_mean_by_development.csv'}")
    print(f"  {output_dir / 'net_benefit_distribution_long.csv'}")


if __name__ == "__main__":
    main()
