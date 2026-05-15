from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from infraScan.infraScanRail import paths as rail_paths


SCENARIOS = ("scenario_76", "scenario_45", "scenario_67")
RAIL_COMPARISON_YEAR = 2050


def _data_root() -> Path:
    return Path(rail_paths.MAIN)


def _output_dir() -> Path:
    out = _data_root() / "plots" / "Integrated_TTS_Comparison"
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return out
    except PermissionError:
        fallback = Path(__file__).resolve().parents[2] / "plots" / "Integrated_TTS_Comparison"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _cached_combined_candidates(output_dir: Path) -> list[Path]:
    return [
        output_dir / "rail_road_tts_minutes_long.csv",
        Path(__file__).resolve().parents[2] / "plots" / "Integrated_TTS_Comparison" / "rail_road_tts_minutes_long.csv",
    ]


def load_cached_combined(
    output_dir: Path,
    scenarios: Iterable[str] = SCENARIOS,
) -> pd.DataFrame | None:
    expected_scenarios = set(map(str, scenarios))
    for path in _cached_combined_candidates(output_dir):
        if path.exists():
            df = pd.read_csv(path)
            if "scenario" not in df.columns:
                continue
            cached_scenarios = set(df["scenario"].dropna().astype(str).unique())
            if expected_scenarios.issubset(cached_scenarios):
                return df[df["scenario"].astype(str).isin(expected_scenarios)].copy()
    return None


def load_rail_tts(year: int = RAIL_COMPARISON_YEAR, scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    path = _data_root() / "data" / "infraScanRail" / "costs" / "traveltime_savings.csv"
    df = pd.read_csv(path)
    df = df[df["scenario"].isin(list(scenarios)) & (df["year"] == year)].copy()
    df["mode"] = "Rail"
    df["development"] = df["development"].astype(str)
    df["tt_savings_daily_minutes"] = df["tt_savings_daily"].astype(float)
    return df[["mode", "development", "scenario", "tt_savings_daily_minutes"]]


def load_road_tts(scenarios: Iterable[str] = SCENARIOS) -> pd.DataFrame:
    path = _data_root() / "data" / "infraScanRoad" / "traffic_flow" / "od" / "od_tt_savings_detailed.csv"
    df = pd.read_csv(path)
    df = df[df["scenario"].isin(list(scenarios))].copy()
    if "tt_savings_peak" not in df.columns:
        available = ", ".join(df.columns)
        raise ValueError(
            "Road TTS source is missing 'tt_savings_peak'. "
            f"Available columns: {available}. "
            "Regenerate the road peak-savings output first."
        )
    df["mode"] = "Road"
    df["development"] = df["development"].astype(str)
    df["tt_savings_daily_minutes"] = df["tt_savings_peak"].astype(float)
    return df[["mode", "development", "scenario", "tt_savings_daily_minutes"]]


def build_road_same_basis_direct_check(
    scenarios: Iterable[str] = SCENARIOS,
) -> pd.DataFrame:
    """
    Simple apples-to-apples road check:
    keep the status-quo OD demand basis fixed and only swap in the development
    travel times for the OD pairs that exist in both tables.
    """
    sq_path = _data_root() / "data" / "infraScanRoad" / "traffic_flow" / "od" / "status_quo_od_tt.csv"
    dev_path = _data_root() / "data" / "infraScanRoad" / "traffic_flow" / "od" / "developments_od_tt.csv"

    sq = pd.read_csv(sq_path)
    dev = pd.read_csv(dev_path)

    sq = sq[sq["scenario"].isin(list(scenarios))].copy()
    dev = dev[dev["scenario"].isin(list(scenarios))].copy()

    rows = []
    for scenario in scenarios:
        sq_scen = sq[sq["scenario"] == scenario][["origin", "destination", "demand", "travel_time"]].copy()
        sq_scen = sq_scen.rename(columns={"travel_time": "tt_sq", "demand": "demand_sq"})
        if sq_scen.empty:
            continue

        for development, dev_scen in dev[dev["scenario"] == scenario].groupby("development"):
            dev_scen = dev_scen[["origin", "destination", "travel_time"]].copy()
            dev_scen = dev_scen.rename(columns={"travel_time": "tt_dev"})

            merged = sq_scen.merge(dev_scen, on=["origin", "destination"], how="inner")
            if merged.empty:
                continue

            sq_total_tt = float((merged["demand_sq"] * merged["tt_sq"]).sum())
            dev_total_tt_on_sq_basis = float((merged["demand_sq"] * merged["tt_dev"]).sum())
            tt_savings_daily_same_basis = sq_total_tt - dev_total_tt_on_sq_basis

            rows.append(
                {
                    "development": str(development),
                    "scenario": str(scenario),
                    "sq_total_tt_on_sq_basis": sq_total_tt,
                    "dev_total_tt_on_sq_basis": dev_total_tt_on_sq_basis,
                    "tt_savings_daily_same_basis": tt_savings_daily_same_basis,
                    "matched_sq_pairs": int(len(merged)),
                }
            )

    return pd.DataFrame(rows)


def summarize_road_same_basis(direct_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        direct_df.groupby("scenario")["tt_savings_daily_same_basis"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    summary["positive_developments"] = (
        direct_df.assign(is_positive=lambda d: d["tt_savings_daily_same_basis"] > 0)
        .groupby("scenario")["is_positive"]
        .sum()
        .values
    )
    return summary


def plot_road_same_basis_focus(direct_df: pd.DataFrame, output_dir: Path) -> None:
    focus_scenarios = ("scenario_28", "scenario_45")
    fig, axes = plt.subplots(1, len(focus_scenarios), figsize=(12, 5), dpi=300, sharey=True)

    for ax, scenario in zip(axes, focus_scenarios):
        subset = direct_df[direct_df["scenario"] == scenario].copy()
        subset = subset.sort_values("tt_savings_daily_same_basis", ascending=False)
        ax.bar(
            subset["development"].astype(str),
            subset["tt_savings_daily_same_basis"],
            color="#c75b39",
            edgecolor="black",
            linewidth=0.6,
        )
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_title(f"Road direct check | {scenario}", fontsize=11)
        ax.set_xlabel("Development", fontsize=10)
        ax.tick_params(axis="x", rotation=90)

    axes[0].set_ylabel("Daily TT savings on SQ basis [minutes]", fontsize=11)
    fig.suptitle("Road direct same-basis check using status-quo OD demand", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "road_same_basis_direct_check.png", bbox_inches="tight")
    plt.close(fig)


def build_summary(combined: pd.DataFrame) -> pd.DataFrame:
    summary = (
        combined.groupby(["mode", "scenario"])["tt_savings_daily_minutes"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )
    positive = (
        combined.assign(is_positive=lambda d: d["tt_savings_daily_minutes"] > 0)
        .groupby(["mode", "scenario"])["is_positive"]
        .sum()
        .reset_index(name="positive_developments")
    )
    return summary.merge(positive, on=["mode", "scenario"], how="left")


def plot_boxplots(combined: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 5), dpi=300, sharey=True)
    colors = {"Rail": "#1f4e79", "Road": "#c75b39"}

    for ax, scenario in zip(axes, SCENARIOS):
        subset = combined[combined["scenario"] == scenario]
        series = [
            subset[subset["mode"] == "Rail"]["tt_savings_daily_minutes"].values,
            subset[subset["mode"] == "Road"]["tt_savings_daily_minutes"].values,
        ]
        boxplot_kwargs = dict(
            patch_artist=True,
            showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
            flierprops={"markersize": 3},
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"color": "black", "linewidth": 0.8},
            capprops={"color": "black", "linewidth": 0.8},
        )
        try:
            bp = ax.boxplot(
                series,
                tick_labels=["Rail", "Road"],
                **boxplot_kwargs,
            )
        except TypeError:
            bp = ax.boxplot(
                series,
                labels=["Rail", "Road"],
                **boxplot_kwargs,
            )
        for patch, mode in zip(bp["boxes"], ["Rail", "Road"]):
            patch.set_facecolor(colors[mode])
            patch.set_alpha(0.9)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_title(scenario.replace("_", " ").title(), fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel("Travel-time savings [minutes]", fontsize=11)

    fig.suptitle(
        f"Rail vs Road travel-time savings by scenario\nRail daily savings vs Road peak savings | Rail fixed at year {RAIL_COMPARISON_YEAR}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "rail_road_tts_boxplots_minutes.png", bbox_inches="tight")
    plt.close(fig)


def plot_violin_overview(combined: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 5), dpi=300, sharey=True)
    colors = {"Rail": "#1f4e79", "Road": "#c75b39"}
    modes = ["Rail", "Road"]

    for ax, scenario in zip(axes, SCENARIOS):
        subset = combined[combined["scenario"] == scenario].copy()
        series = [
            subset[subset["mode"] == mode]["tt_savings_daily_minutes"].values
            for mode in modes
        ]

        violin = ax.violinplot(
            series,
            positions=[1, 2],
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )

        for body, mode in zip(violin["bodies"], modes):
            body.set_facecolor(colors[mode])
            body.set_edgecolor("black")
            body.set_alpha(0.75)
            body.set_linewidth(0.8)

        violin["cmedians"].set_color("black")
        violin["cmedians"].set_linewidth(1.4)

        for pos, mode in enumerate(modes, start=1):
            mode_values = subset[subset["mode"] == mode]["tt_savings_daily_minutes"].values
            x_values = [pos] * len(mode_values)
            ax.scatter(
                x_values,
                mode_values,
                s=12,
                alpha=0.45,
                color=colors[mode],
                edgecolors="black",
                linewidths=0.25,
                zorder=3,
            )
            ax.text(
                pos,
                0.98,
                f"n={len(mode_values)}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=9,
                color="#333333",
            )

        ax.set_xticks([1, 2])
        ax.set_xticklabels(modes)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_title(scenario.replace("_", " ").title(), fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel("Travel-time savings [minutes]", fontsize=11)

    fig.suptitle(
        f"Rail vs Road travel-time savings by scenario\nViolin overview | Rail daily savings vs Road peak savings | Rail fixed at year {RAIL_COMPARISON_YEAR}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "rail_road_tts_violin_minutes.png", bbox_inches="tight")
    plt.close(fig)


def plot_top_developments(combined: pd.DataFrame, output_dir: Path) -> None:
    focus_scenarios = ("scenario_28", "scenario_45")
    colors = {"Rail": "#1f4e79", "Road": "#c75b39"}

    fig, axes = plt.subplots(len(focus_scenarios), 2, figsize=(14, 10), dpi=300)

    for row_idx, scenario in enumerate(focus_scenarios):
        for col_idx, mode in enumerate(("Rail", "Road")):
            ax = axes[row_idx, col_idx]
            subset = combined[(combined["scenario"] == scenario) & (combined["mode"] == mode)].copy()
            subset = subset.sort_values("tt_savings_daily_minutes", ascending=False).head(10)
            ax.barh(
                subset["development"].astype(str),
                subset["tt_savings_daily_minutes"],
                color=colors[mode],
                edgecolor="black",
                linewidth=0.6,
            )
            ax.invert_yaxis()
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            ax.set_title(f"{mode} | {scenario}", fontsize=11)
            if row_idx == len(focus_scenarios) - 1:
                ax.set_xlabel("Travel-time savings [minutes]", fontsize=10)

    fig.suptitle(
        f"Top 10 developments by travel-time savings\nRail daily savings vs Road peak savings | Rail year {RAIL_COMPARISON_YEAR}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "rail_road_top10_tts_minutes.png", bbox_inches="tight")
    plt.close(fig)


def plot_development_boxplots(combined: pd.DataFrame, output_dir: Path, mode: str) -> None:
    subset = combined[combined["mode"] == mode].copy()
    if subset.empty:
        return

    stats = (
        subset.groupby("development")["tt_savings_daily_minutes"]
        .mean()
        .sort_values(ascending=False)
    )
    dev_order = stats.index.tolist()
    series = [
        subset[subset["development"] == dev]["tt_savings_daily_minutes"].values
        for dev in dev_order
    ]

    fig_width = max(12, len(dev_order) * 0.33)
    fig, ax = plt.subplots(figsize=(fig_width, 6), dpi=300)
    color = "#1f4e79" if mode == "Rail" else "#c75b39"

    boxplot_kwargs = dict(
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4},
        flierprops={"markersize": 3},
        medianprops={"color": "black", "linewidth": 1.3},
        whiskerprops={"color": "black", "linewidth": 0.8},
        capprops={"color": "black", "linewidth": 0.8},
    )
    try:
        bp = ax.boxplot(series, tick_labels=dev_order, **boxplot_kwargs)
    except TypeError:
        bp = ax.boxplot(series, labels=dev_order, **boxplot_kwargs)

    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.set_xlabel("Development", fontsize=11)
    ax.set_ylabel("Travel-time savings [minutes]", fontsize=11)
    ax.set_title(f"{mode} travel-time savings by development across scenarios", fontsize=13)
    ax.tick_params(axis="x", rotation=90)

    fig.tight_layout()
    fig.savefig(output_dir / f"{mode.lower()}_development_boxplots_tts_minutes.png", bbox_inches="tight")
    plt.close(fig)


def plot_ranked_development_bars(combined: pd.DataFrame, output_dir: Path, mode: str) -> None:
    subset = combined[combined["mode"] == mode].copy()
    if subset.empty:
        return

    summary = (
        subset.groupby("development")["tt_savings_daily_minutes"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    summary["color"] = summary["tt_savings_daily_minutes"].apply(
        lambda value: "#1f4e79" if value >= 0 else "#c75b39"
    )

    fig_height = max(6, len(summary) * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_height), dpi=300)

    ax.barh(
        summary["development"].astype(str),
        summary["tt_savings_daily_minutes"],
        color=summary["color"],
        edgecolor="black",
        linewidth=0.6,
    )
    ax.axvline(0, color="gray", linewidth=0.9, linestyle="--")
    ax.grid(axis="x", linestyle="--", alpha=0.6)
    ax.set_xlabel("Mean travel-time savings across scenarios [minutes]", fontsize=11)
    ax.set_ylabel("Development", fontsize=11)
    ax.set_title(f"{mode} developments ranked by mean travel-time savings", fontsize=13)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(output_dir / f"{mode.lower()}_ranked_tts_bars_minutes.png", bbox_inches="tight")
    plt.close(fig)


def plot_scenario_means(summary: pd.DataFrame, output_dir: Path) -> None:
    plot_df = summary.copy()
    plot_df["scenario_label"] = plot_df["scenario"].str.replace("_", " ").str.title()

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = range(len(plot_df["scenario"].unique()))
    width = 0.35
    scenarios = list(plot_df["scenario"].unique())

    rail_vals = [plot_df[(plot_df["scenario"] == s) & (plot_df["mode"] == "Rail")]["mean"].iloc[0] for s in scenarios]
    road_vals = [plot_df[(plot_df["scenario"] == s) & (plot_df["mode"] == "Road")]["mean"].iloc[0] for s in scenarios]

    ax.bar([i - width / 2 for i in x], rail_vals, width=width, color="#1f4e79", edgecolor="black", label="Rail")
    ax.bar([i + width / 2 for i in x], road_vals, width=width, color="#c75b39", edgecolor="black", label="Road")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios])
    ax.set_ylabel("Mean travel-time savings [minutes]")
    ax.set_title(f"Mean travel-time savings by scenario | Rail daily vs Road peak | Rail year {RAIL_COMPARISON_YEAR}")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "rail_road_mean_tts_minutes.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = _output_dir()

    combined = load_cached_combined(output_dir)
    if combined is None:
        rail = load_rail_tts()
        road = load_road_tts()
        combined = pd.concat([rail, road], ignore_index=True)

    summary = build_summary(combined)
    summary.to_csv(output_dir / "rail_road_tts_minutes_summary.csv", index=False)
    combined.to_csv(output_dir / "rail_road_tts_minutes_long.csv", index=False)

    plot_boxplots(combined, output_dir)
    plot_violin_overview(combined, output_dir)
    plot_top_developments(combined, output_dir)
    plot_development_boxplots(combined, output_dir, mode="Rail")
    plot_development_boxplots(combined, output_dir, mode="Road")
    plot_ranked_development_bars(combined, output_dir, mode="Rail")
    plot_ranked_development_bars(combined, output_dir, mode="Road")
    plot_scenario_means(summary, output_dir)

    road_direct = build_road_same_basis_direct_check()
    road_direct.to_csv(output_dir / "road_same_basis_direct_check.csv", index=False)
    summarize_road_same_basis(road_direct).to_csv(
        output_dir / "road_same_basis_direct_check_summary.csv",
        index=False,
    )
    plot_road_same_basis_focus(road_direct, output_dir)

    print(f"Saved comparison outputs to: {output_dir}")
    print("Files created:")
    for name in [
        "rail_road_tts_minutes_summary.csv",
        "rail_road_tts_minutes_long.csv",
        "rail_road_tts_boxplots_minutes.png",
        "rail_road_tts_violin_minutes.png",
        "rail_road_top10_tts_minutes.png",
        "rail_development_boxplots_tts_minutes.png",
        "road_development_boxplots_tts_minutes.png",
        "rail_ranked_tts_bars_minutes.png",
        "road_ranked_tts_bars_minutes.png",
        "rail_road_mean_tts_minutes.png",
        "road_same_basis_direct_check.csv",
        "road_same_basis_direct_check_summary.csv",
        "road_same_basis_direct_check.png",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
