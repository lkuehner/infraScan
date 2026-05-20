from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EDGES_PATH = Path("/Volumes/WD_Windows/MSc_Thesis/data/infraScanRoad/Network/processed/edges_with_attribute.gpkg")
FLOW_PATH = Path("/Volumes/WD_Windows/MSc_Thesis/data/infraScanRoad/traffic_flow/developments/D_i/Xi_sum_status_quo_scenario_45.csv")
OUT_PNG = Path("/Users/laura/Desktop/infraScan_lkuehner/plots/road_link_tt_statusquo_scenario45.png")
OUT_CSV = Path("/Users/laura/Desktop/infraScan_lkuehner/plots/road_link_tt_statusquo_scenario45_summary.csv")


def main() -> None:
    edges = gpd.read_file(EDGES_PATH).sort_values("ID_edge").reset_index(drop=True)
    flow = pd.read_csv(FLOW_PATH, header=None, names=["flow"])

    if len(edges) != len(flow):
        raise ValueError(f"Edge/flow length mismatch: {len(edges)} vs {len(flow)}")

    edges["flow"] = flow["flow"].astype(float)

    # Same units and parameters as in convert_data_to_input() and CostFun().
    edges["length_km"] = edges.geometry.length / 1000.0
    edges["fftt_h"] = edges["length_km"] / pd.to_numeric(edges["ffs"])
    alpha = 0.25
    gamma = 2.4
    edges["tt_h"] = edges["fftt_h"] * (1.0 + alpha * np.power(edges["flow"] / edges["capacity"], gamma))
    edges["tt_min"] = edges["tt_h"] * 60.0
    edges["fftt_min"] = edges["fftt_h"] * 60.0
    edges["delay_min"] = edges["tt_min"] - edges["fftt_min"]

    summary = edges[["ID_edge", "flow", "fftt_min", "tt_min", "delay_min"]].sort_values("tt_min", ascending=False)
    summary.to_csv(OUT_CSV, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=250)
    base_kw = dict(color="#d9d9d9", linewidth=0.5)

    edges.plot(ax=axes[0], **base_kw)
    edges.plot(
        ax=axes[0],
        column="tt_min",
        cmap="viridis",
        linewidth=2.2,
        legend=True,
        legend_kwds={"label": "Link travel time [min]", "shrink": 0.75},
    )
    axes[0].set_title("Road status quo link TT (scenario_45)")
    axes[0].set_axis_off()

    edges.plot(ax=axes[1], **base_kw)
    edges.plot(
        ax=axes[1],
        column="delay_min",
        cmap="magma",
        linewidth=2.2,
        legend=True,
        legend_kwds={"label": "Congestion delay vs free flow [min]", "shrink": 0.75},
    )
    axes[1].set_title("Delay component per link")
    axes[1].set_axis_off()

    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches="tight")

    print(summary.head(15).to_string(index=False))
    print("\nDescribe tt_min")
    print(edges["tt_min"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_string())
    print(f"\nSaved {OUT_PNG}")
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
