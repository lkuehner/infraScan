import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from infraScan.infraScanIntegrated import settings as integrated_settings
from infraScan.infraScanIntegrated.random_scenarios_multimode import (
    generate_joint_modal_split_scenarios_simple,
)


OUTPUT_DIR = WORKSPACE_ROOT / "plots" / "Integrated" / "modal_split_simple"


def plot_mode_comparison(
    mode_name: str,
    simple_df: pd.DataFrame,
    output_path: Path,
    federal_2050_range: tuple[float, float] | None = None,
) -> None:
    year_stats = (
        simple_df.groupby("year")["modal_split"]
        .agg(min="min", max="max", mean="mean", std="std")
        .reset_index()
    )
    year_stats["mean_plus_1_65std"] = year_stats["mean"] + 1.65 * year_stats["std"]
    year_stats["mean_minus_1_65std"] = year_stats["mean"] - 1.65 * year_stats["std"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.fill_between(
        year_stats["year"],
        year_stats["min"],
        year_stats["max"],
        color="grey",
        alpha=0.3,
        label="Gesamter Bereich",
    )
    ax.plot(
        year_stats["year"],
        year_stats["mean_plus_1_65std"],
        color="red",
        linestyle="-",
        alpha=0.7,
        label="+1,65σ (95%)",
    )
    ax.plot(
        year_stats["year"],
        year_stats["mean_minus_1_65std"],
        color="red",
        linestyle="-",
        alpha=0.7,
        label="-1,65σ (5%)",
    )
    ax.plot(
        year_stats["year"],
        year_stats["mean"],
        color="grey",
        linestyle="--",
        alpha=0.8,
        label="Mittelwert",
    )

    sample_id = simple_df["scenario"].drop_duplicates().sample(n=1, random_state=42).iloc[0]
    sample_df = simple_df[simple_df["scenario"] == sample_id].sort_values("year")
    ax.plot(
        sample_df["year"],
        sample_df["modal_split"],
        color="blue",
        linewidth=2,
        label=f"Beispielszenario {sample_id}",
    )

    if 2050 in year_stats["year"].values and federal_2050_range is not None:
        lower_bound, upper_bound = federal_2050_range
        marker_color = "#E08D3C"
        if abs(upper_bound - lower_bound) < 1e-12:
            marker_description = f"Verkehrsperspektive 2050 ({lower_bound*100:.1f}%)"
            ax.plot([2050], [lower_bound], marker="_", markersize=14, color=marker_color, label=marker_description)
        else:
            marker_description = f"Verkehrsperspektive 2050 ({lower_bound*100:.1f}-{upper_bound*100:.1f}%)"
            ax.vlines(
                x=2050,
                ymin=lower_bound,
                ymax=upper_bound,
                colors=marker_color,
                linestyles="solid",
                linewidth=2,
                label=marker_description,
            )
            ax.plot([2050], [lower_bound], marker="_", markersize=10, color=marker_color)
            ax.plot([2050], [upper_bound], marker="_", markersize=10, color=marker_color)

    ax.set_title(f"{mode_name} Modal-Split-Szenarien: Bereich, Mittelwert und 90% Konfidenzintervall")
    ax.set_xlabel("Jahr")
    ax.set_ylabel("Modal-Split (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.set_ylim(0.0, 1.0)
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    start_year = integrated_settings.start_year_scenario
    end_year = integrated_settings.end_year_scenario
    n_scenarios = integrated_settings.amount_of_scenarios

    rail_simple, road_simple, other_simple = generate_joint_modal_split_scenarios_simple(
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
    )

    rail_simple.to_csv(OUTPUT_DIR / "rail_modal_split_simple.csv", index=False)
    road_simple.to_csv(OUTPUT_DIR / "road_modal_split_simple.csv", index=False)
    other_simple.to_csv(OUTPUT_DIR / "other_modal_split_simple.csv", index=False)

    plot_mode_comparison(
        mode_name="Rail",
        simple_df=rail_simple,
        output_path=OUTPUT_DIR / "rail_modal_split_simple.png",
        federal_2050_range=(0.187, integrated_settings.rail_modal_split_target),
    )
    plot_mode_comparison(
        mode_name="Road",
        simple_df=road_simple,
        output_path=OUTPUT_DIR / "road_modal_split_simple.png",
        federal_2050_range=(
            integrated_settings.road_modal_split_target,
            integrated_settings.road_modal_split_target,
        ),
    )
    plot_mode_comparison(
        mode_name="Other",
        simple_df=other_simple,
        output_path=OUTPUT_DIR / "other_modal_split_simple.png",
        federal_2050_range=(
            integrated_settings.other_modal_split_target,
            integrated_settings.other_modal_split_target,
        ),
    )

    print(f"Saved CSVs and plots to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
