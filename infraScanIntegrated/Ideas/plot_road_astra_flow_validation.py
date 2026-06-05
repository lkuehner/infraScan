from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# Edit these settings for quick plot adjustments.
BASE = Path("/cluster/home/lkuehner/MSc_Thesis")
INPUT_CSV = (
    BASE
    / "infraScan/infraScanIntegrated/outputs/road_od_run_travel_time_checks/scenario_44_links/"
    / "scenario_44_15iter_links_flow_with_astra_counts_expanded_no_missing.csv"
)
OUTPUT_PNG = INPUT_CSV.with_name("scenario_44_15iter_model_flow_vs_astra_counts_and_capacity_no_missing_custom.png")

TITLE = "1 Scenario, status quo: model flow vs traffic counts"
X_LABEL = "Link ID"
Y_LABEL = "Flow [veh / hour]"
FIGSIZE = (13, 6)
DPI = 220

# Keep only links with actual ASTRA count values.
DROP_MISSING_COUNTS = True

# Keep only rows where a model flow exists. Set False if you want to show ASTRA-only rows.
DROP_MISSING_MODEL_FLOW = True

# Label format on x-axis. Available columns include ID_edge and link_group.
X_LABEL_TEMPLATE = "{ID_edge}"

# Column names in the CSV.
MODEL_FLOW_COL = "flow_status_quo_15iter"
ASTRA_PEAK_COL = "traffic_count_peak_hour_avg"
ASTRA_DWV24_COL = "traffic_count_dwv_per_hour"
CAPACITY_COL = "capacity"

# Select which series to draw and how they should appear.
SERIES = [
    (CAPACITY_COL, "Model capacity", "#B0B0B0"),
    (MODEL_FLOW_COL, "Model flow", "#0E4F84"),
    (ASTRA_PEAK_COL, "Traffic count (peak hour)", "#FF8C00"),
    (ASTRA_DWV24_COL, "Traffic count (AADT)", "#FFBF00"),
]


def main() -> None:
    df = pd.read_csv(INPUT_CSV)

    if DROP_MISSING_COUNTS:
        df = df[df[ASTRA_PEAK_COL].notna() | df[ASTRA_DWV24_COL].notna()].copy()

    if DROP_MISSING_MODEL_FLOW:
        df = df[df[MODEL_FLOW_COL].notna()].copy()

    df = df.sort_values(["link_group", "ID_edge"]).reset_index(drop=True)
    df["plot_label"] = df.apply(lambda row: X_LABEL_TEMPLATE.format(**row.to_dict()), axis=1)

    long = df.melt(
        id_vars=["ID_edge", "link_group", "plot_label"],
        value_vars=[column for column, _, _ in SERIES],
        var_name="source",
        value_name="value",
    )
    long = long.dropna(subset=["value"])
    source_labels = {column: label for column, label, _ in SERIES}
    source_colors = {label: color for column, label, color in SERIES}
    long["source"] = long["source"].map(source_labels)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
        }
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(
        data=long,
        x="plot_label",
        y="value",
        hue="source",
        hue_order=list(source_colors),
        palette=source_colors,
        ax=ax,
    )

    ax.set_title(TITLE)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.tick_params(axis="x", rotation=55)
    ax.grid(True, axis="y", alpha=0.35)
    ax.grid(False, axis="x")
    ax.legend(title="Source", loc="upper left", frameon=True)

    plt.tight_layout()
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Read:  {INPUT_CSV}")
    print(f"Wrote: {OUTPUT_PNG}")
    print(f"Rows plotted: {len(df)}")


if __name__ == "__main__":
    main()
