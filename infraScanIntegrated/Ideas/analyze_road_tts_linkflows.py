from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALPHA = 0.25
GAMMA = 2.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Road OD travel-time savings and diagnose worst link "
            "flows/delays from existing infraScanRoad outputs."
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
        default=Path(__file__).resolve().parents[1] / "outputs" / "road_tts_linkflow_analysis",
        help="Directory for analysis CSV outputs.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional scenario filter, e.g. scenario_19 scenario_29.",
    )
    parser.add_argument(
        "--developments",
        nargs="*",
        type=int,
        default=None,
        help="Optional development filter, e.g. 0 2 334.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of worst links to keep per development-scenario case.",
    )
    parser.add_argument(
        "--include-status-quo",
        action="store_true",
        help="Also summarize worst status-quo links per scenario.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write CSV outputs, skip PNG plots.",
    )
    return parser.parse_args()


def read_tts_components(data_root: Path) -> pd.DataFrame:
    path = data_root / "data" / "infraScanRoad" / "traffic_flow" / "od" / "od_tt_savings_detailed.csv"
    df = pd.read_csv(path)
    required = {
        "development",
        "scenario",
        "origin_access_savings",
        "network_savings",
        "destination_access_savings",
        "tt_savings_peak",
        "monetized_savings_yearly",
        "monetized_savings",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df["development"] = df["development"].astype(int)
    df["scenario"] = df["scenario"].astype(str)
    df["access_savings"] = df["origin_access_savings"] + df["destination_access_savings"]
    df["total_tts"] = df["tt_savings_peak"]
    df["positive_tts"] = df["total_tts"] > 0
    return df


def load_edge_attributes(data_root: Path) -> pd.DataFrame:
    path = data_root / "data" / "infraScanRoad" / "Network" / "processed" / "edges_with_attribute.gpkg"
    edges = gpd.read_file(path)
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges["capacity"] = pd.to_numeric(edges["capacity"], errors="coerce")
    edges["ffs"] = pd.to_numeric(edges["ffs"], errors="coerce")
    edges["edge_length_m"] = edges.geometry.length
    return pd.DataFrame(edges.drop(columns="geometry"))


def enrich_link_flows(flow_path: Path, edges: pd.DataFrame) -> pd.DataFrame:
    flows = pd.read_csv(flow_path)
    flows["ID_edge"] = flows["ID_edge"].astype(int)
    flows["flow"] = pd.to_numeric(flows["flow"], errors="coerce")

    df = flows.merge(
        edges[["ID_edge", "capacity", "ffs", "edge_length_m"]],
        on="ID_edge",
        how="left",
    )
    length_m = df["length_m"].fillna(df["edge_length_m"])
    capacity = df["capacity"].replace(0, np.nan)
    ffs = df["ffs"].replace(0, np.nan)

    df["flow_capacity_ratio"] = df["flow"] / capacity
    df["fftt_min"] = (length_m / 1000.0) / ffs * 60.0
    df["tt_min"] = df["fftt_min"] * (
        1.0 + ALPHA * np.power(df["flow_capacity_ratio"], GAMMA)
    )
    df["delay_min"] = df["tt_min"] - df["fftt_min"]
    return df


def worst_links_for_case(
    data_root: Path,
    edges: pd.DataFrame,
    development: int,
    scenario: str,
    top_n: int,
) -> pd.DataFrame:
    path = (
        data_root
        / "data"
        / "infraScanRoad"
        / "traffic_flow"
        / "od"
        / "link_flows"
        / f"dev{development}_{scenario}.csv"
    )
    if not path.exists():
        return pd.DataFrame(
            [
                {
                    "development": development,
                    "scenario": scenario,
                    "missing_link_flow_file": str(path),
                }
            ]
        )

    links = enrich_link_flows(path, edges)
    worst_vcr = links.nlargest(top_n, "flow_capacity_ratio", keep="all").copy()
    worst_vcr["rank_type"] = "flow_capacity_ratio"
    worst_vcr["rank_value"] = worst_vcr["flow_capacity_ratio"]

    worst_delay = links.nlargest(top_n, "delay_min", keep="all").copy()
    worst_delay["rank_type"] = "delay_min"
    worst_delay["rank_value"] = worst_delay["delay_min"]

    cols = [
        "development",
        "scenario",
        "rank_type",
        "rank_value",
        "ID_edge",
        "length_m",
        "flow",
        "capacity",
        "flow_capacity_ratio",
        "fftt_min",
        "tt_min",
        "delay_min",
    ]
    out = pd.concat([worst_vcr, worst_delay], ignore_index=True)
    out["development"] = development
    out["scenario"] = scenario
    return out[cols]


def worst_status_quo_links(
    data_root: Path,
    edges: pd.DataFrame,
    scenario: str,
    top_n: int,
) -> pd.DataFrame:
    path = (
        data_root
        / "data"
        / "infraScanRoad"
        / "traffic_flow"
        / "od"
        / "link_flows"
        / f"status_quo_{scenario}.csv"
    )
    if not path.exists():
        return pd.DataFrame(
            [{"development": "status_quo", "scenario": scenario, "missing_link_flow_file": str(path)}]
        )

    links = enrich_link_flows(path, edges)
    worst_vcr = links.nlargest(top_n, "flow_capacity_ratio", keep="all").copy()
    worst_vcr["rank_type"] = "flow_capacity_ratio"
    worst_vcr["rank_value"] = worst_vcr["flow_capacity_ratio"]

    worst_delay = links.nlargest(top_n, "delay_min", keep="all").copy()
    worst_delay["rank_type"] = "delay_min"
    worst_delay["rank_value"] = worst_delay["delay_min"]

    out = pd.concat([worst_vcr, worst_delay], ignore_index=True)
    out["development"] = "status_quo"
    out["scenario"] = scenario
    return out[
        [
            "development",
            "scenario",
            "rank_type",
            "rank_value",
            "ID_edge",
            "length_m",
            "flow",
            "capacity",
            "flow_capacity_ratio",
            "fftt_min",
            "tt_min",
            "delay_min",
        ]
    ]


def _format_axis(ax) -> None:
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_tts_distribution(component_summary: pd.DataFrame, plot_dir: Path) -> None:
    scenarios = list(component_summary["scenario"].drop_duplicates())
    fig, axes = plt.subplots(
        1,
        len(scenarios),
        figsize=(5.2 * len(scenarios), 4.2),
        dpi=220,
        sharey=True,
    )
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        subset = component_summary[component_summary["scenario"] == scenario]
        ax.hist(subset["total_tts"], bins=24, color="#5b8bb2", edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="#8c2d2d", linewidth=1.1, linestyle="--")
        ax.set_title(scenario)
        ax.set_xlabel("Peak travel-time savings")
        _format_axis(ax)

    axes[0].set_ylabel("Developments")
    fig.suptitle("Distribution of road travel-time savings", y=1.02)
    fig.tight_layout()
    fig.savefig(plot_dir / "tts_distribution_by_scenario.png", bbox_inches="tight")
    plt.close(fig)


def plot_positive_tts_top(positive: pd.DataFrame, plot_dir: Path, top_n: int = 20) -> None:
    if positive.empty:
        return

    scenarios = list(positive["scenario"].drop_duplicates())
    fig, axes = plt.subplots(
        len(scenarios),
        1,
        figsize=(12, max(3.8, 0.35 * top_n * len(scenarios))),
        dpi=220,
        squeeze=False,
    )

    for ax, scenario in zip(axes[:, 0], scenarios):
        subset = (
            positive[positive["scenario"] == scenario]
            .nlargest(top_n, "total_tts")
            .sort_values("total_tts")
        )
        y = np.arange(len(subset))
        ax.barh(y, subset["network_savings"], color="#386cb0", label="network")
        ax.barh(
            y,
            subset["access_savings"],
            left=subset["network_savings"],
            color="#7fc97f",
            label="access",
        )
        ax.set_yticks(y)
        ax.set_yticklabels(subset["development"].astype(str))
        ax.set_title(f"Top positive TTS developments - {scenario}")
        ax.set_xlabel("Peak travel-time savings")
        ax.set_ylabel("Development")
        ax.legend(loc="lower right")
        _format_axis(ax)

    fig.tight_layout()
    fig.savefig(plot_dir / "positive_tts_top_developments.png", bbox_inches="tight")
    plt.close(fig)


def plot_scenario_summary(by_scenario: pd.DataFrame, plot_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=220)

    x = np.arange(len(by_scenario))
    labels = by_scenario["scenario"].astype(str)
    axes[0].bar(x, by_scenario["developments"], color="#bdbdbd", label="all")
    axes[0].bar(x, by_scenario["positive_developments"], color="#4daf4a", label="positive")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=30, ha="right")
    axes[0].set_ylabel("Developments")
    axes[0].set_title("Positive developments by scenario")
    axes[0].legend()
    _format_axis(axes[0])

    axes[1].bar(x - 0.18, by_scenario["mean_network_savings"], width=0.36, color="#386cb0", label="network")
    axes[1].bar(x + 0.18, by_scenario["mean_access_savings"], width=0.36, color="#7fc97f", label="access")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=30, ha="right")
    axes[1].set_ylabel("Mean peak savings")
    axes[1].set_title("Mean components by scenario")
    axes[1].legend()
    _format_axis(axes[1])

    fig.tight_layout()
    fig.savefig(plot_dir / "tts_summary_by_scenario.png", bbox_inches="tight")
    plt.close(fig)


def plot_worst_link_rankings(worst: pd.DataFrame, plot_dir: Path, prefix: str, top_n: int = 20) -> None:
    if worst.empty or "missing_link_flow_file" in worst.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=220)
    configs = [
        ("flow_capacity_ratio", "Max flow/capacity ratio", "max_flow_capacity_ratio"),
        ("delay_min", "Max delay [min]", "max_delay_min"),
    ]

    for ax, (rank_type, title, col_name) in zip(axes, configs):
        subset = worst[worst["rank_type"] == rank_type].copy()
        if subset.empty:
            ax.set_axis_off()
            continue
        grouped = (
            subset.groupby("ID_edge")
            .agg(
                rank_value=("rank_value", "max"),
                appearances=("ID_edge", "size"),
                mean_flow=("flow", "mean"),
                mean_capacity=("capacity", "mean"),
            )
            .rename(columns={"rank_value": col_name})
            .reset_index()
            .nlargest(top_n, col_name)
            .sort_values(col_name)
        )
        y = np.arange(len(grouped))
        ax.barh(y, grouped[col_name], color="#f07f3c", edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(grouped["ID_edge"].astype(str))
        ax.set_xlabel(title)
        ax.set_ylabel("ID_edge")
        ax.set_title(f"{title} ({prefix})")
        _format_axis(ax)

    fig.tight_layout()
    fig.savefig(plot_dir / f"{prefix}_worst_link_rankings.png", bbox_inches="tight")
    plt.close(fig)


def plot_worst_links_for_best_cases(
    worst_positive: pd.DataFrame,
    positive: pd.DataFrame,
    plot_dir: Path,
    max_cases: int = 8,
) -> None:
    if worst_positive.empty or positive.empty:
        return

    best_cases = positive.nlargest(max_cases, "total_tts")[["development", "scenario", "total_tts"]]
    subset = worst_positive.merge(best_cases[["development", "scenario"]], on=["development", "scenario"], how="inner")
    subset = subset[subset["rank_type"] == "delay_min"].copy()
    if subset.empty:
        return

    subset["case"] = subset["development"].astype(str) + " | " + subset["scenario"].astype(str)
    idx = subset.groupby("case")["delay_min"].idxmax()
    plot_df = subset.loc[idx].merge(best_cases, on=["development", "scenario"], how="left")
    plot_df = plot_df.sort_values("total_tts")

    fig, ax = plt.subplots(figsize=(11, 5), dpi=220)
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["delay_min"], color="#984ea3", edgecolor="black", linewidth=0.4)
    labels = (
        "dev "
        + plot_df["development"].astype(str)
        + " / "
        + plot_df["scenario"].astype(str)
        + " / edge "
        + plot_df["ID_edge"].astype(str)
    )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Worst-link delay [min]")
    ax.set_title("Worst delayed link in the highest positive TTS cases")
    _format_axis(ax)
    fig.tight_layout()
    fig.savefig(plot_dir / "best_positive_cases_worst_delay_link.png", bbox_inches="tight")
    plt.close(fig)


def make_plots(
    component_summary: pd.DataFrame,
    positive: pd.DataFrame,
    by_scenario: pd.DataFrame,
    worst: pd.DataFrame,
    output_dir: Path,
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_tts_distribution(component_summary, plot_dir)
    plot_positive_tts_top(positive, plot_dir)
    plot_scenario_summary(by_scenario, plot_dir)
    plot_worst_link_rankings(worst, plot_dir, prefix="all_cases")

    if not positive.empty and not worst.empty:
        positive_keys = positive[["development", "scenario"]].drop_duplicates()
        worst_positive = worst.merge(positive_keys, on=["development", "scenario"], how="inner")
        plot_worst_link_rankings(worst_positive, plot_dir, prefix="positive_tts_cases")
        plot_worst_links_for_best_cases(worst_positive, positive, plot_dir)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tts = read_tts_components(data_root)
    if args.scenarios:
        tts = tts[tts["scenario"].isin(args.scenarios)].copy()
    if args.developments:
        tts = tts[tts["development"].isin(args.developments)].copy()

    component_cols = [
        "development",
        "scenario",
        "origin_access_savings",
        "destination_access_savings",
        "access_savings",
        "network_savings",
        "total_tts",
        "positive_tts",
        "monetized_savings_yearly",
        "monetized_savings",
    ]
    component_summary = tts[component_cols].sort_values(
        ["positive_tts", "total_tts"], ascending=[False, False]
    )
    component_summary.to_csv(output_dir / "tts_component_summary.csv", index=False)

    positive = component_summary[component_summary["positive_tts"]].copy()
    positive.to_csv(output_dir / "positive_tts_developments.csv", index=False)

    edges = load_edge_attributes(data_root)
    worst_rows = []
    for row in tts[["development", "scenario"]].drop_duplicates().itertuples(index=False):
        worst_rows.append(
            worst_links_for_case(
                data_root=data_root,
                edges=edges,
                development=int(row.development),
                scenario=str(row.scenario),
                top_n=args.top_n,
            )
        )

    worst = pd.concat(worst_rows, ignore_index=True) if worst_rows else pd.DataFrame()
    worst.to_csv(output_dir / "worst_links_by_case.csv", index=False)

    if not positive.empty and not worst.empty:
        positive_keys = positive[["development", "scenario"]].drop_duplicates()
        worst_positive = worst.merge(positive_keys, on=["development", "scenario"], how="inner")
        worst_positive.to_csv(output_dir / "worst_links_positive_tts.csv", index=False)

    if args.include_status_quo:
        status_rows = [
            worst_status_quo_links(data_root, edges, scenario, args.top_n)
            for scenario in sorted(tts["scenario"].unique())
        ]
        pd.concat(status_rows, ignore_index=True).to_csv(
            output_dir / "worst_status_quo_links_by_scenario.csv",
            index=False,
        )

    by_scenario = (
        component_summary.groupby("scenario")
        .agg(
            developments=("development", "nunique"),
            positive_developments=("positive_tts", "sum"),
            mean_total_tts=("total_tts", "mean"),
            median_total_tts=("total_tts", "median"),
            min_total_tts=("total_tts", "min"),
            max_total_tts=("total_tts", "max"),
            mean_network_savings=("network_savings", "mean"),
            mean_access_savings=("access_savings", "mean"),
        )
        .reset_index()
    )
    by_scenario.to_csv(output_dir / "tts_summary_by_scenario.csv", index=False)

    if not args.no_plots:
        make_plots(
            component_summary=component_summary,
            positive=positive,
            by_scenario=by_scenario,
            worst=worst,
            output_dir=output_dir,
        )

    print(f"Output directory: {output_dir}")
    print("\nPositive TTS developments:")
    if positive.empty:
        print("  none")
    else:
        print(
            positive[
                ["development", "scenario", "total_tts", "network_savings", "access_savings"]
            ].to_string(index=False)
        )

    print("\nScenario summary:")
    print(by_scenario.to_string(index=False))


if __name__ == "__main__":
    main()
