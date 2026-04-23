from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import rasterio


def interpolate_linestring(linestring, interval: float) -> list:
    length = linestring.length
    num_points = max(2, int(np.ceil(length / interval)))
    return [linestring.interpolate(distance) for distance in np.linspace(0, length, num_points)]


def sample_raster_at_points(points, raster) -> list[float]:
    band = raster.read(1)
    values: list[float] = []
    for point in points:
        row, col = raster.index(point.x, point.y)
        values.append(float(band[row, col]))
    return values


def initial_profile_flags(links: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    links = links.copy()
    links["elevation_difference"] = links["elevation_profile"].apply(lambda values: np.diff(np.array(values)))
    links["elevation_absolute"] = links["elevation_difference"].apply(np.abs)
    links["slope"] = links["elevation_absolute"].apply(lambda values: values / 50 * 100)
    links["slope_mean"] = links["slope"].apply(np.mean)
    # Mirror the currently active code, even though the threshold direction is suspicious.
    links["steep_section"] = links["slope"].apply(lambda values: int((values < 5).sum()))
    links["check_needed"] = (links["slope_mean"] > 5) | (links["steep_section"] > 40)
    return links


def optimize_values_min_changes(values: list[float], max_slope: float) -> list[float] | None:
    max_diff = max_slope * 50
    prob = pulp.LpProblem("SlopeOptimizationMinChanges", pulp.LpMinimize)
    lp_vars = {i: pulp.LpVariable(f"v_{i}") for i in range(len(values))}
    change_vars = {i: pulp.LpVariable(f"c_{i}", 0, 1, cat="Binary") for i in range(len(values))}
    prob += pulp.lpSum(change_vars[i] for i in range(len(values)))

    for i in range(len(values)):
        if i > 0:
            prob += lp_vars[i] - lp_vars[i - 1] <= max_diff
            prob += lp_vars[i - 1] - lp_vars[i] <= max_diff
        prob += lp_vars[i] - values[i] <= 1e9 * change_vars[i]
        prob += values[i] - lp_vars[i] <= 1e9 * change_vars[i]

    prob += lp_vars[0] == values[0]
    prob += change_vars[0] == 0
    prob += lp_vars[len(values) - 1] == values[len(values) - 1]
    prob += change_vars[len(values) - 1] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob.status != pulp.LpStatusOptimal:
        return None
    return [float(pulp.value(lp_vars[i])) for i in range(len(values))]


def check_for_bridge_tunnel(elevation: list[float], new_elevation: list[float], changes: list[bool]) -> list[int]:
    elevation_array = np.array(elevation)
    new_elevation_array = np.array(new_elevation)
    changes_array = np.array(changes)
    flags = np.zeros(len(elevation_array), dtype=int)
    height_diff = new_elevation_array - elevation_array

    for i in range(len(elevation_array) - 1):
        if changes_array[i]:
            if height_diff[i] <= -10 and height_diff[i + 1] <= -10:
                flags[i] = -1
            elif height_diff[i] >= 10 and height_diff[i + 1] >= 10:
                flags[i] = 1
    return flags.tolist()


def plot_profile(row: pd.Series, outdir: Path) -> None:
    x = np.arange(len(row["elevation_profile"])) * 50
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(x, row["elevation_profile"], label="Original", color="gray", linewidth=2)
    ax.plot(x, row["new_elevation"], label="Optimized", color="black", linewidth=1.5)

    for i, flag in enumerate(row["bridge_tunnel_flags"]):
        if flag == -1:
            ax.axvline(x=i * 50, color="lightgray", linewidth=8, alpha=0.7)
        elif flag == 1:
            ax.axvline(x=i * 50, color="lightblue", linewidth=8, alpha=0.7)

    ax.set_title(f"ID_new {row['ID_new']}")
    ax.set_xlabel("Distance (m)")
    ax.set_ylabel("Elevation (m asl)")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(outdir / f"profile_{int(row['ID_new'])}.png", dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export flagged InfraScanRoad links with elevation profiles for manual review.")
    parser.add_argument("--base-path", required=True, help="Path to the data folder or its parent")
    parser.add_argument("--outdir", help="Optional output directory; defaults to infraScanRoad/review_exports under cwd")
    parser.add_argument("--only-flagged", action="store_true", help="Export only links with check_needed=True")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG profile plot export")
    parser.add_argument("--limit", type=int, help="Optionally export only the first N rows after filtering")
    args = parser.parse_args()

    base_path = Path(args.base_path).expanduser()
    if base_path.name != "data":
        base_path = base_path / "data"

    road_base = base_path / "infraScanRoad"
    processed = road_base / "Network" / "processed"
    links_path = processed / "new_links_realistic.gpkg"
    elevation_path = road_base / "elevation_model" / "elevation.tif"
    outdir = Path(args.outdir).expanduser() if args.outdir else Path.cwd() / "infraScan" / "infraScanRoad" / "review_exports"
    outdir.mkdir(parents=True, exist_ok=True)

    links = gpd.read_file(links_path)
    with rasterio.open(elevation_path) as raster:
        links["elevation_profile"] = links["geometry"].apply(
            lambda geom: sample_raster_at_points(interpolate_linestring(geom, 50), raster)
        )

    links = initial_profile_flags(links)
    links["new_elevation"] = links.apply(
        lambda row: optimize_values_min_changes(row["elevation_profile"], 0.07) if row["check_needed"] else row["elevation_profile"],
        axis=1,
    )
    links = links.dropna(subset=["new_elevation"]).copy()
    links["changes"] = links.apply(
        lambda row: (np.array(row["elevation_profile"]) != np.array(row["new_elevation"])).tolist(),
        axis=1,
    )
    links["bridge_tunnel_flags"] = links.apply(
        lambda row: check_for_bridge_tunnel(row["elevation_profile"], row["new_elevation"], row["changes"]),
        axis=1,
    )

    if args.only_flagged:
        links = links[links["check_needed"]].copy()
    if args.limit is not None and args.limit > 0:
        links = links.head(args.limit).copy()

    if not args.no_plots:
        for _, row in links.iterrows():
            plot_profile(row, outdir)

    export = links[
        [
            "ID_new",
            "ID_current",
            "check_needed",
            "slope_mean",
            "steep_section",
            "geometry",
            "elevation_profile",
            "new_elevation",
            "changes",
            "bridge_tunnel_flags",
        ]
    ].copy()

    for col in ["elevation_profile", "new_elevation", "changes", "bridge_tunnel_flags"]:
        export[col] = export[col].apply(json.dumps)

    gpkg_path = outdir / "flagged_links_review.gpkg"
    csv_path = outdir / "flagged_links_review.csv"
    export.to_file(gpkg_path, driver="GPKG")
    export.drop(columns=["geometry"]).to_csv(csv_path, index=False)

    print(f"Wrote review layer: {gpkg_path}")
    print(f"Wrote review table: {csv_path}")
    print(f"Wrote profile plots to: {outdir}")
    print(f"Rows exported: {len(export)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
