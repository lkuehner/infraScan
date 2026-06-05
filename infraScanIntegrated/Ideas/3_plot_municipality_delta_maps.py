"""
Create chloropleth maps for municipality accessibility deltas.
Colors represent the improvement in accessibility from each development scenario.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/Volumes/WD_Windows/MSc_Thesis/data")
RAIL = DATA / "infraScanRail"
IN = ROOT / "infraScan" / "plots" / "rail_accessibility_maps" / "municipality_lineplots"
OUT = ROOT / "infraScan" / "plots" / "rail_accessibility_maps" / "municipality_delta_maps"

COMMUNE_SHP = DATA / "_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp"
CSV_PATH = IN / "rail_accessibility_municipalities.csv"

TOP_N = 6


def create_green_gray_red_colormap():
    """Create custom colormap: red (worse) → light gray (0) → green (better)"""
    colors = [
        "#8B0000",  # dark red (negative = slower/worse)
        "#C54B4B",  # red
        "#F5F5F5",  # light gray (0 = no change)
        "#9CF49C",  # light green
        "#014B01",  # forest green (positive = faster/better)
    ]
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list("green_gray_red", colors, N=n_bins)
    return cmap


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading municipality data...")
    communes = gpd.read_file(COMMUNE_SHP)
    communes["BFS"] = pd.to_numeric(communes["BFS"], errors="coerce")
    communes = communes.dropna(subset=["BFS"]).copy()
    communes["BFS"] = communes["BFS"].astype(int)
    communes = communes[communes["BFS"] > 0].copy()

    df = pd.read_csv(CSV_PATH)
    df["BFS"] = pd.to_numeric(df["BFS"], errors="coerce")
    df = df.dropna(subset=["BFS"]).copy()
    df["BFS"] = df["BFS"].astype(int)

    gdf = communes.merge(df, on="BFS", how="left")

    # Find all dev columns (top N)
    dev_cols = [col for col in gdf.columns if col.startswith("dev_")][:TOP_N]

    print(f"Creating delta maps for {len(dev_cols)} developments...")
    
    # Calculate global min/max for consistent scale across all maps
    print("Computing global scale across all developments...")
    all_deltas_pct = []
    all_deltas_abs = []
    
    for dev_col in dev_cols:
        gdf_temp = gdf.copy()
        gdf_temp["dev_value"] = gdf_temp[dev_col].fillna(gdf_temp["base_accessibility"])
        gdf_temp["delta"] = gdf_temp["base_accessibility"] - gdf_temp["dev_value"]
        gdf_temp["delta_pct"] = (gdf_temp["delta"] / gdf_temp["base_accessibility"] * 100).fillna(0)
        
        valid_pct = gdf_temp["delta_pct"].dropna()
        valid_abs = gdf_temp["delta"].dropna()
        
        if len(valid_pct) > 0:
            all_deltas_pct.extend(valid_pct.values)
        if len(valid_abs) > 0:
            all_deltas_abs.extend(valid_abs.values)
    
    # Compute global quantiles for symmetric scale
    global_pct_min = np.percentile(all_deltas_pct, 5) if all_deltas_pct else -1
    global_pct_max = np.percentile(all_deltas_pct, 95) if all_deltas_pct else 1
    global_pct_abs = max(abs(global_pct_min), abs(global_pct_max))
    vmin_pct = -global_pct_abs
    vmax_pct = global_pct_abs
    
    global_abs_min = np.percentile(all_deltas_abs, 5) if all_deltas_abs else -1
    global_abs_max = np.percentile(all_deltas_abs, 95) if all_deltas_abs else 1
    global_abs_abs = max(abs(global_abs_min), abs(global_abs_max))
    vmin_abs = -global_abs_abs
    vmax_abs = global_abs_abs
    
    print(f"  Percentage delta range: [{vmin_pct:.2f}%, {vmax_pct:.2f}%]")
    print(f"  Absolute delta range: [{vmin_abs:.2f}, {vmax_abs:.2f}]")
    print("  (negative = faster/better, positive = slower/worse)")

    for dev_col in dev_cols:
        dev_id = dev_col.replace("dev_", "")

        # Fallback: use base_accessibility if dev value is missing
        gdf["dev_value"] = gdf[dev_col].fillna(gdf["base_accessibility"])
        
        # Direction: base - dev (so negative = faster/better, positive = slower/worse)
        gdf["delta"] = gdf["base_accessibility"] - gdf["dev_value"]
        gdf["delta_pct"] = (gdf["delta"] / gdf["base_accessibility"] * 100).fillna(0)

        fig, ax = plt.subplots(figsize=(16, 12), dpi=150)

        # Plot base communes in light gray
        communes.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)

        # Plot colored delta map
        gdf_valid = gdf.dropna(subset=["delta"])
        if len(gdf_valid) > 0:
            cmap = create_green_gray_red_colormap()
            norm = mcolors.TwoSlopeNorm(vmin=vmin_pct, vcenter=0, vmax=vmax_pct)

            gdf_valid.plot(
                column="delta_pct",
                ax=ax,
                legend=True,
                cmap=cmap,
                norm=norm,
                edgecolor="#333333",
                linewidth=0.5,
                legend_kwds={
                    "label": "Accessibility Improvement (%)",
                    "orientation": "vertical",
                    "shrink": 0.6,
                },
            )

        ax.set_title(
            f"Municipality Accessibility Delta: Development {dev_id}\n"
            f"(% faster: green = better, red = worse, gray = no change)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        fig.tight_layout()

        out_file = OUT / f"municipality_delta_dev_{dev_id}.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_file.name}")

        # Also save absolute delta (not percentage)
        fig, ax = plt.subplots(figsize=(16, 12), dpi=150)
        communes.plot(ax=ax, color="#f0f0f0", edgecolor="#cccccc", linewidth=0.5)

        gdf_valid = gdf.dropna(subset=["delta"])
        if len(gdf_valid) > 0:
            cmap = create_green_gray_red_colormap()
            norm = mcolors.TwoSlopeNorm(vmin=vmin_abs, vcenter=0, vmax=vmax_abs)

            gdf_valid.plot(
                column="delta",
                ax=ax,
                legend=True,
                cmap=cmap,
                norm=norm,
                edgecolor="#333333",
                linewidth=0.5,
                legend_kwds={
                    "label": "Accessibility Improvement (absolute)",
                    "orientation": "vertical",
                    "shrink": 0.6,
                },
            )

        ax.set_title(
            f"Municipality Accessibility Delta: Development {dev_id}\n"
            f"(absolute faster: green = better, red = worse, gray = no change)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal")
        fig.tight_layout()

        out_file = OUT / f"municipality_delta_abs_dev_{dev_id}.png"
        fig.savefig(out_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_file.name}")

    print(f"\nAll maps saved to: {OUT}")


if __name__ == "__main__":
    main()
