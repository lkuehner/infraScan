from __future__ import annotations

from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

from infraScan.infraScanRail import paths as rail_paths


SCENARIOS = ("scenario_76", "scenario_45", "scenario_67")
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


def _data_root() -> Path:
    return Path(rail_paths.MAIN)


def _output_dir() -> Path:
    out = _data_root() / "plots" / "Integrated_Final_Costs"
    try:
        out.mkdir(parents=True, exist_ok=True)
        probe = out / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return out
    except PermissionError:
        fallback = Path(__file__).resolve().parents[2] / "plots" / "Integrated_Final_Costs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _line_name_from_rail_row(row: pd.Series) -> str:
    development = int(str(row["development"]).removeprefix("Development_"))
    sline = str(row["Sline"])
    if development < 101000:
        return f"{development - 100000}_{sline}"
    return f"X{development - 101000}"


def load_rail_final_costs() -> pd.DataFrame:
    path = _data_root() / "data" / "infraScanRail" / "costs" / "total_costs.csv"
    df = pd.read_csv(path)

    df["development_id"] = df["development"].astype(str).str.removeprefix("Development_").astype(int)
    df["line_name"] = df.apply(_line_name_from_rail_row, axis=1)

    rows: list[dict] = []
    for scenario in SCENARIOS:
        suffix = scenario.split("_")[-1]
        rows.append(
            pd.DataFrame(
                {
                    "mode": "Rail",
                    "development": df["development_id"].astype(str),
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


def load_road_final_costs() -> pd.DataFrame:
    totals_path = _data_root() / "data" / "infraScanRoad" / "costs" / "total_costs_od.csv"
    tt_path = _data_root() / "data" / "infraScanRoad" / "costs" / "traveltime_savings_od.csv"
    local_path = _data_root() / "data" / "infraScanRoad" / "costs" / "local_accessibility.csv"

    totals = pd.read_csv(totals_path)
    tt = pd.read_csv(tt_path)
    local = pd.read_csv(local_path).rename(columns={"ID_develop": "ID_new"})

    rows: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        total_col = f"total_{scenario}"
        tt_col = f"tt_{scenario}"
        local_col = f"local_{scenario}"
        if local_col not in local.columns:
            local_col = scenario

        merged = (
            totals[["ID_new", total_col]]
            .merge(tt[["development", tt_col]], left_on="ID_new", right_on="development", how="left")
            .merge(local[["ID_new", local_col]], on="ID_new", how="left")
        )
        merged = merged.rename(
            columns={
                total_col: "net_benefit_chf",
                tt_col: "monetized_savings_chf",
                local_col: "local_accessibility_chf",
            }
        )
        merged["mode"] = "Road"
        merged["development"] = merged["ID_new"].astype(str)
        merged["line_name"] = merged["development"]
        merged["scenario"] = scenario
        merged["net_benefit_mio_chf"] = merged["net_benefit_chf"] / 1_000_000
        merged["monetized_savings_mio_chf"] = merged["monetized_savings_chf"] / 1_000_000
        merged["local_accessibility_mio_chf"] = merged["local_accessibility_chf"] / 1_000_000
        rows.append(merged)

    combined = pd.concat(rows, ignore_index=True)
    return combined[
        [
            "mode",
            "development",
            "line_name",
            "scenario",
            "net_benefit_mio_chf",
            "monetized_savings_mio_chf",
            "local_accessibility_mio_chf",
        ]
    ]


def load_road_final_cost_components() -> pd.DataFrame:
    if gpd is None:
        raise ImportError("geopandas is required to build the Road cost-savings stacked plot.")

    base = _data_root() / "data" / "infraScanRoad" / "costs"
    totals = pd.read_csv(base / "total_costs_od.csv")
    tt = pd.read_csv(base / "traveltime_savings_od.csv")
    construction = gpd.read_file(base / "construction.gpkg")[["ID_new", "building_costs"]]
    maintenance = gpd.read_file(base / "maintenance.gpkg")[["ID_new", "maintenance"]]
    externalities = gpd.read_file(base / "externalities.gpkg")[["ID_new", "climate_cost", "land_realloc", "nature"]]
    noise = gpd.read_file(base / "noise.gpkg")[["ID_new", "noise_s1"]]

    base_df = (
        construction.merge(maintenance, on="ID_new", how="inner")
        .merge(externalities, on="ID_new", how="inner")
        .merge(noise, on="ID_new", how="inner")
    )
    base_df["externalities_chf"] = (
        base_df["climate_cost"] + base_df["land_realloc"] + base_df["nature"] + base_df["noise_s1"]
    )

    rows: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        scenario_df = (
            totals[["ID_new", f"total_{scenario}"]]
            .merge(tt[["development", f"tt_{scenario}"]], left_on="ID_new", right_on="development", how="left")
            .merge(base_df[["ID_new", "building_costs", "maintenance", "externalities_chf"]], on="ID_new", how="left")
        )
        scenario_df["mode"] = "Road"
        scenario_df["development"] = scenario_df["ID_new"].astype(str)
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


def _ordered_line_names(data: pd.DataFrame) -> list[str]:
    return (
        data.groupby("line_name")["net_benefit_mio_chf"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )


def _line_colors(order: list[str]) -> dict[str, str]:
    return {line_name: ZVV_COLORS[i % len(ZVV_COLORS)] for i, line_name in enumerate(order)}


def plot_final_net_benefit_boxplot(data: pd.DataFrame, output_dir: Path, mode: str) -> None:
    subset = data[data["mode"] == mode].copy()
    if subset.empty:
        return

    order = _ordered_line_names(subset)
    colors = [_line_colors(order)[line] for line in order]
    fig_width = max(7, len(order) * 0.58)

    plt.figure(figsize=(fig_width, 5), dpi=300)
    ax = sns.boxplot(
        data=subset,
        x="line_name",
        y="net_benefit_mio_chf",
        hue="line_name",
        order=order,
        palette=colors,
        dodge=False,
        width=0.4,
        linewidth=0.8,
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 5},
        fliersize=3,
        showfliers=True,
    )
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_xlim(-0.5, len(order) - 0.5)
    plt.axhline(y=0, color="red", linestyle="-", alpha=0.5)
    plt.xlabel("Line" if mode == "Rail" else "Development ID", fontsize=12)
    plt.ylabel("Net benefits over all scenarios\n[Mio. CHF]", fontsize=12)
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    legend_handles = [
        mlines.Line2D([0], [0], marker="o", color="black", label="Mean", markersize=5),
        mpatches.Patch(color=colors[0], label="Line Colour"),
    ]
    plt.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / f"{mode.lower()}_final_net_benefit_boxplot.png", dpi=600)
    plt.close()


def plot_mode_comparison_boxplots(data: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 5), dpi=300, sharey=True)
    colors = {"Rail": "#4C78A8", "Road": "#54A24B"}

    for ax, scenario in zip(axes, SCENARIOS):
        subset = data[data["scenario"] == scenario]
        series = [
            subset[subset["mode"] == "Rail"]["net_benefit_mio_chf"].values,
            subset[subset["mode"] == "Road"]["net_benefit_mio_chf"].values,
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
            bp = ax.boxplot(series, tick_labels=["Rail", "Road"], **boxplot_kwargs)
        except TypeError:
            bp = ax.boxplot(series, labels=["Rail", "Road"], **boxplot_kwargs)

        for patch, mode in zip(bp["boxes"], ["Rail", "Road"]):
            patch.set_facecolor(colors[mode])
            patch.set_alpha(0.85)
            patch.set_edgecolor("black")
            patch.set_linewidth(0.8)

        ax.axhline(0, color="red", linewidth=0.8, linestyle="-", alpha=0.5)
        ax.grid(axis="y", linestyle="--", alpha=0.7)
        ax.set_title(scenario.replace("_", " ").title(), fontsize=12)
        if ax is axes[0]:
            ax.set_ylabel("Net benefits [Mio. CHF]", fontsize=11)

    fig.suptitle("Rail vs Road final net benefits by scenario", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_dir / "rail_road_final_net_benefit_boxplots.png", bbox_inches="tight")
    plt.close(fig)


def plot_rail_cost_savings_bars(rail_data: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_line_names(rail_data)
    line_colors = _line_colors(order)
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
    plt.bar(
        x_pos,
        -summary["construction_cost_mio_chf"],
        width=bar_width,
        color=cost_colors["construction"],
        label="Construction costs",
    )
    plt.bar(
        x_pos,
        -summary["maintenance_cost_mio_chf"],
        width=bar_width,
        bottom=-summary["construction_cost_mio_chf"],
        color=cost_colors["maintenance"],
        label="Uncovered maintenance costs",
    )
    plt.bar(
        x_pos,
        -summary["uncovered_operating_cost_mio_chf"],
        width=bar_width,
        bottom=-(summary["construction_cost_mio_chf"] + summary["maintenance_cost_mio_chf"]),
        color=cost_colors["operating"],
        label="Uncovered operating costs",
    )

    for i, line_name in enumerate(order):
        plt.bar(
            x_pos[i],
            summary.loc[summary["line_name"] == line_name, "monetized_savings_mio_chf"].iloc[0],
            width=bar_width,
            color=line_colors[line_name],
            hatch="////",
            edgecolor="black",
        )

    plt.axhline(y=0, color="black", linestyle="-")
    plt.xticks(x_pos, summary["line_name"], rotation=90)
    plt.xlabel("Linie", fontsize=12)
    plt.ylabel("Wert in Mio. CHF", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    handles = [
        mpatches.Patch(color=cost_colors["construction"], label="Baukosten"),
        mpatches.Patch(color=cost_colors["maintenance"], label="ungedeckte Unterhaltskosten"),
        mpatches.Patch(color=cost_colors["operating"], label="ungedeckte Betriebskosten"),
        mpatches.Patch(facecolor="none", hatch="////", edgecolor="black", label="Reisezeiteinsparnisse"),
        mpatches.Patch(facecolor="#31a354", edgecolor="#31a354", label="Linienfarbe"),
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / "rail_final_cost_savings.png", dpi=600)
    plt.close()


def plot_road_cost_savings_bars(road_data: pd.DataFrame, output_dir: Path) -> None:
    order = _ordered_line_names(road_data)
    line_colors = _line_colors(order)
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
    plt.bar(
        x_pos,
        -summary["maintenance_cost_mio_chf"],
        width=bar_width,
        bottom=-summary["construction_cost_mio_chf"],
        color=cost_colors["maintenance"],
        label="Maintenance costs",
    )
    plt.bar(
        x_pos,
        -summary["other_cost_mio_chf"],
        width=bar_width,
        bottom=-(summary["construction_cost_mio_chf"] + summary["maintenance_cost_mio_chf"]),
        color=cost_colors["other"],
        label="Externalities",
    )

    for i, line_name in enumerate(order):
        plt.bar(
            x_pos[i],
            summary.loc[summary["line_name"] == line_name, "monetized_savings_mio_chf"].iloc[0],
            width=bar_width,
            color=line_colors[line_name],
            hatch="////",
            edgecolor="black",
        )

    plt.axhline(y=0, color="black", linestyle="-")
    plt.xticks(x_pos, summary["line_name"], rotation=90)
    plt.xlabel("Development ID", fontsize=12)
    plt.ylabel("Wert in Mio. CHF", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    handles = [
        mpatches.Patch(color=cost_colors["construction"], label="Baukosten"),
        mpatches.Patch(color=cost_colors["maintenance"], label="Unterhaltskosten"),
        mpatches.Patch(color=cost_colors["other"], label="Externalities"),
        mpatches.Patch(facecolor="none", hatch="////", edgecolor="black", label="Reisezeiteinsparnisse"),
        mpatches.Patch(facecolor="#31a354", edgecolor="#31a354", label="Linienfarbe"),
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / "road_final_cost_savings.png", dpi=600)
    plt.close()


def plot_combined_top5_cost_savings(rail_data: pd.DataFrame, road_data: pd.DataFrame, output_dir: Path) -> None:
    rail_top = (
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
        .head(5)
        .assign(mode="Rail", label=lambda d: d["line_name"])
        .rename(columns={"uncovered_operating_cost_mio_chf": "uncovered_operating_cost_mio_chf"})
    )
    road_top = (
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
        .head(5)
        .assign(mode="Road", label=lambda d: "R" + d["line_name"])
        .rename(columns={"other_cost_mio_chf": "externalities_cost_mio_chf"})
    )
    combined = pd.concat([rail_top, road_top], ignore_index=True)

    x_pos = np.arange(len(combined))
    bar_width = 0.6
    cost_colors = {
        "construction": "#a6bddb",
        "maintenance": "#3690c0",
        "operating": "#1f5a89",
        "externalities": "#4d4d4d",
        "tts": "#f2c94c",
    }

    combined["uncovered_operating_cost_mio_chf"] = combined.get("uncovered_operating_cost_mio_chf", 0).fillna(0)
    combined["externalities_cost_mio_chf"] = combined.get("externalities_cost_mio_chf", 0).fillna(0)

    plt.figure(figsize=(11, 5), dpi=300)
    plt.bar(x_pos, -combined["construction_cost_mio_chf"], width=bar_width, color=cost_colors["construction"], label="Construction costs")
    plt.bar(
        x_pos,
        -combined["maintenance_cost_mio_chf"],
        width=bar_width,
        bottom=-combined["construction_cost_mio_chf"],
        color=cost_colors["maintenance"],
        label="Maintenance costs",
    )
    plt.bar(
        x_pos,
        -combined["uncovered_operating_cost_mio_chf"],
        width=bar_width,
        bottom=-(combined["construction_cost_mio_chf"] + combined["maintenance_cost_mio_chf"]),
        color=cost_colors["operating"],
        label="Uncovered operating costs",
    )
    plt.bar(
        x_pos,
        -combined["externalities_cost_mio_chf"],
        width=bar_width,
        bottom=-(combined["construction_cost_mio_chf"] + combined["maintenance_cost_mio_chf"] + combined["uncovered_operating_cost_mio_chf"]),
        color=cost_colors["externalities"],
        label="Externalities",
    )

    plt.bar(
        x_pos,
        combined["monetized_savings_mio_chf"],
        width=bar_width,
        color=cost_colors["tts"],
        edgecolor="black",
        label="Travel time savings",
    )

    plt.axhline(y=0, color="black", linestyle="-")
    plt.axvline(x=4.5, color="gray", linestyle="--", alpha=0.7)
    plt.xticks(x_pos, combined["label"], rotation=90)
    plt.xlabel("Development", fontsize=12)
    plt.ylabel("Value in CHF million", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.text(2, plt.ylim()[1] * 0.95, "Rail top 5", ha="center", va="top", fontsize=11)
    plt.text(7, plt.ylim()[1] * 0.95, "Road top 5", ha="center", va="top", fontsize=11)

    handles = [
        mpatches.Patch(color=cost_colors["construction"], label="Construction costs"),
        mpatches.Patch(color=cost_colors["maintenance"], label="Maintenance costs"),
        mpatches.Patch(color=cost_colors["operating"], label="Uncovered operating costs"),
        mpatches.Patch(color=cost_colors["externalities"], label="Externalities"),
        mpatches.Patch(color=cost_colors["tts"], label="Travel time savings"),
    ]
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1))

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_dir / "combined_top5_final_cost_savings.png", dpi=600)
    plt.close()


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.groupby(["mode", "scenario"])["net_benefit_mio_chf"]
        .agg(["count", "mean", "median", "min", "max", "std"])
        .reset_index()
    )


def main() -> None:
    output_dir = _output_dir()
    rail = load_rail_final_costs()
    road = load_road_final_costs()
    road_components = load_road_final_cost_components()
    combined = pd.concat([rail, road], ignore_index=True, sort=False)

    combined.to_csv(output_dir / "rail_road_final_costs_long.csv", index=False)
    build_summary(combined).to_csv(output_dir / "rail_road_final_costs_summary.csv", index=False)

    plot_mode_comparison_boxplots(combined, output_dir)
    plot_final_net_benefit_boxplot(combined, output_dir, mode="Rail")
    plot_final_net_benefit_boxplot(combined, output_dir, mode="Road")
    plot_rail_cost_savings_bars(rail, output_dir)
    plot_road_cost_savings_bars(road_components, output_dir)
    plot_combined_top5_cost_savings(rail, road_components, output_dir)

    print(f"Saved final-cost comparison outputs to: {output_dir}")
    print("Files created:")
    for name in [
        "rail_road_final_costs_long.csv",
        "rail_road_final_costs_summary.csv",
        "rail_road_final_net_benefit_boxplots.png",
        "rail_final_net_benefit_boxplot.png",
        "road_final_net_benefit_boxplot.png",
        "rail_final_cost_savings.png",
        "road_final_cost_savings.png",
        "combined_top5_final_cost_savings.png",
    ]:
        print(f"  - {name}")


if __name__ == "__main__":
    main()
