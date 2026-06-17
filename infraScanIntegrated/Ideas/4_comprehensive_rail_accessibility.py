"""
Comprehensive rail accessibility workflow for the active canton_ZH setup.

Workflow:
1. Build one consistent municipality accessibility summary for base + all developments.
2. Build a development summary table with total accessibility and delta percentages.
3. Create plots from those summaries:
   - absolute municipality lineplot for all developments
   - total accessibility by development
   - municipality delta-percentage maps for the top 10 developments
"""

from __future__ import annotations

import os
import pickle
import sqlite3
import sys
from pathlib import Path

os.environ["USE_PYGEOS"] = "0"

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from infraScan.infraScanRail import paths as rail_paths
from infraScan.infraScanRail import settings as rail_settings
from shapely.ops import linemerge, unary_union

DATA = Path("/Volumes/WD_Windows/MSc_Thesis/data")
RAIL = DATA / "infraScanRail"
OUT_BASE = ROOT / "infraScan" / "plots" / "rail_accessibility_maps"
OUT_SUMMARY = OUT_BASE / "comprehensive_summary"
OUT_LINEPLOT = OUT_BASE / "lineplot_absolute_values"
OUT_DELTA = OUT_BASE / "municipality_delta_maps_top10"

POINTS_GPKG = RAIL / "Network/processed/points.gpkg"
STATION_OD_PATH = RAIL / "traffic_flow/od/rail/ktzh/od_matrix_stations_ktzh_20.csv"
OD_TIMES_CACHE = RAIL / "Network/travel_time/cache/od_times.pkl"
DEVELOPMENT_DIRECTORY = RAIL / "Network/processed/developments"
COMMUNE_TO_STATION_PATH = RAIL / "Network/processed/Communes_to_railway_stations_ZH.xlsx"
COMMUNE_SHP = DATA / "_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp"
NEW_LINKS_PATH = RAIL / "Network/processed/updated_new_links.gpkg"

TOP_N = 10
NAME_NORMALIZATION = {
    "Niederglatt": "Niederglatt ZH",
    "Oberglatt": "Oberglatt ZH",
}


def load_station_lookup() -> gpd.GeoDataFrame:
    with sqlite3.connect(POINTS_GPKG) as conn:
        lookup = pd.read_sql_query(
            "SELECT ID_point, NAME, XKOORD, YKOORD FROM points",
            conn,
        )

    lookup["ID_point"] = lookup["ID_point"].astype(int)
    lookup["station_name"] = lookup["NAME"].replace(NAME_NORMALIZATION)
    if lookup["XKOORD"].median() < 1_000_000:
        lookup["XKOORD"] = lookup["XKOORD"] + 2_000_000
    if lookup["YKOORD"].median() < 1_000_000:
        lookup["YKOORD"] = lookup["YKOORD"] + 1_000_000

    return gpd.GeoDataFrame(
        lookup[["ID_point", "station_name", "XKOORD", "YKOORD"]],
        geometry=gpd.points_from_xy(lookup["XKOORD"], lookup["YKOORD"]),
        crs="EPSG:2056",
    )[["ID_point", "station_name", "geometry"]].drop_duplicates()


def load_station_od_matrix() -> pd.DataFrame:
    od = pd.read_csv(STATION_OD_PATH)
    od = od.rename(columns={od.columns[0]: "origin"})
    od["origin"] = pd.to_numeric(od["origin"], errors="coerce").astype("Int64")

    rename_map = {
        col: int(float(col))
        for col in od.columns[1:]
        if pd.notna(pd.to_numeric(col, errors="coerce"))
    }
    od = od.rename(columns=rename_map)
    od = od.dropna(subset=["origin"]).copy()
    od["origin"] = od["origin"].astype(int)
    od = od.set_index("origin")
    return od.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def load_commune_station_mapping() -> pd.DataFrame:
    mapping = pd.read_excel(COMMUNE_TO_STATION_PATH)
    mapping = mapping.rename(columns={"Commune_BFS_code": "BFS"})
    mapping = mapping[["BFS", "Commune", "ID_point", "Station"]].copy()
    mapping["BFS"] = pd.to_numeric(mapping["BFS"], errors="coerce")
    mapping["ID_point"] = pd.to_numeric(mapping["ID_point"], errors="coerce")
    mapping = mapping.dropna(subset=["BFS", "ID_point"]).copy()
    mapping["BFS"] = mapping["BFS"].astype(int)
    mapping["ID_point"] = mapping["ID_point"].astype(int)
    return mapping[mapping["BFS"] > 0].drop_duplicates(subset=["BFS"])


def load_communes() -> gpd.GeoDataFrame:
    communes = gpd.read_file(COMMUNE_SHP)
    communes["BFS"] = pd.to_numeric(communes["BFS"], errors="coerce")
    communes = communes.dropna(subset=["BFS"]).copy()
    communes["BFS"] = communes["BFS"].astype(int)
    communes = communes[communes["BFS"] > 0].copy()
    return communes[["BFS", "GEMEINDENA", "geometry"]]


def load_od_times_cache() -> dict:
    with OD_TIMES_CACHE.open("rb") as handle:
        return pickle.load(handle)


def build_dev_position_lookup() -> dict[str, int]:
    dev_ids = sorted(
        str(int(float(path.stem)))
        for path in DEVELOPMENT_DIRECTORY.iterdir()
        if path.is_file() and path.suffix == ".gpkg" and not path.name.startswith("._")
    )
    return {dev_id: idx for idx, dev_id in enumerate(dev_ids)}


def get_active_base_network_path() -> Path:
    rel_path = rail_paths.get_rail_services_path(rail_settings.rail_network)
    return DATA.parent / rel_path


def load_network_overlay(path: Path) -> gpd.GeoDataFrame:
    if not path.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:2056")
    gdf = gpd.read_file(path)
    if gdf.empty or "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:2056")
    return gdf[["geometry"]].copy()


def load_active_development_overlay(dev_id: str) -> gpd.GeoDataFrame:
    if not NEW_LINKS_PATH.exists():
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:2056")

    gdf = gpd.read_file(NEW_LINKS_PATH)
    if gdf.empty or "dev_id" not in gdf.columns or "geometry" not in gdf.columns:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:2056")

    dev_gdf = gdf[gdf["dev_id"].astype(str) == str(dev_id)].copy()
    if dev_gdf.empty:
        return gpd.GeoDataFrame(geometry=[], crs=gdf.crs)

    merged_geom = linemerge(unary_union(dev_gdf.geometry))
    return gpd.GeoDataFrame({"dev_id": [dev_id]}, geometry=[merged_geom], crs=dev_gdf.crs)


def calculate_pj(od_matrix: pd.DataFrame) -> pd.Series:
    pj = od_matrix.sum(axis=0)
    pj.index = pj.index.astype(int)
    return pj


def od_times_to_long_form(
    od_times_df: pd.DataFrame,
    station_lookup: pd.DataFrame,
    allowed_station_ids: set[int],
) -> pd.DataFrame:
    name_to_id = station_lookup.set_index("station_name")["ID_point"].to_dict()

    work = od_times_df.copy()
    work["origin"] = work["from_station"].map(name_to_id)
    work["destination"] = work["to_station"].map(name_to_id)
    work["travel_time_min"] = pd.to_numeric(work["time"], errors="coerce")

    work = work.dropna(subset=["origin", "destination", "travel_time_min"]).copy()
    work["origin"] = work["origin"].astype(int)
    work["destination"] = work["destination"].astype(int)
    work = work[
        work["origin"].isin(allowed_station_ids)
        & work["destination"].isin(allowed_station_ids)
        & (work["origin"] != work["destination"])
    ].copy()

    return work[["origin", "destination", "travel_time_min"]]


def rail_accessibility_decay(t_minutes: float) -> float:
    if t_minutes < 58:
        return np.exp(-0.032 * t_minutes**2)
    return np.exp(-0.023 * t_minutes)


def add_accessibility_metric(tt_df: pd.DataFrame, pj: pd.Series) -> pd.DataFrame:
    work = tt_df.merge(pj.rename("pj"), left_on="destination", right_index=True, how="left")
    work["pj"] = work["pj"].fillna(0.0)
    work["accessibility"] = work["travel_time_min"].apply(rail_accessibility_decay) * work["pj"]
    return (
        work.groupby("origin", as_index=False)["accessibility"]
        .sum()
        .rename(columns={"origin": "ID_point"})
    )


def build_station_accessibility(
    station_lookup: gpd.GeoDataFrame,
    pj: pd.Series,
    base_tt: pd.DataFrame,
    dev_times: dict[str, pd.DataFrame],
    development_ids: list[str],
) -> gpd.GeoDataFrame:
    result = station_lookup.merge(
        add_accessibility_metric(base_tt, pj).rename(
            columns={"accessibility": "base_accessibility"}
        ),
        on="ID_point",
        how="left",
    )

    for dev_id in development_ids:
        dev_tt = dev_times.get(dev_id)
        if dev_tt is None or dev_tt.empty:
            continue
        dev_access = add_accessibility_metric(dev_tt, pj).rename(
            columns={"accessibility": f"dev_{dev_id}"}
        )
        result = result.merge(dev_access, on="ID_point", how="left")

    return result


def build_municipality_accessibility(
    station_accessibility: gpd.GeoDataFrame,
    commune_mapping: pd.DataFrame,
    communes: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    municipality_accessibility = commune_mapping.merge(
        station_accessibility.drop(columns="geometry", errors="ignore"),
        on="ID_point",
        how="left",
    )
    return communes.merge(municipality_accessibility, on="BFS", how="left")


def summarize_developments(
    municipality_df: pd.DataFrame,
    development_ids: list[str],
) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    base_col = municipality_df["base_accessibility"]

    for dev_id in development_ids:
        dev_col = f"dev_{dev_id}"
        if dev_col not in municipality_df.columns:
            continue

        pair_df = municipality_df[["base_accessibility", dev_col]].dropna().copy()
        if pair_df.empty:
            continue

        base_total = pair_df["base_accessibility"].sum()
        dev_total = pair_df[dev_col].sum()
        delta_abs = base_total - dev_total
        delta_pct = np.nan if base_total == 0 else (delta_abs / base_total) * 100.0

        records.append(
            {
                "development": dev_id,
                "n_municipalities": int(len(pair_df)),
                "base_total_accessibility": base_total,
                "dev_total_accessibility": dev_total,
                "delta_total_accessibility": delta_abs,
                "delta_pct_total_accessibility": delta_pct,
                "base_mean_accessibility": pair_df["base_accessibility"].mean(),
                "dev_mean_accessibility": pair_df[dev_col].mean(),
                "delta_mean_accessibility": pair_df["base_accessibility"].mean() - pair_df[dev_col].mean(),
                "n_improved": int((pair_df[dev_col] < pair_df["base_accessibility"]).sum()),
                "n_worsened": int((pair_df[dev_col] > pair_df["base_accessibility"]).sum()),
                "n_unchanged": int((pair_df[dev_col] == pair_df["base_accessibility"]).sum()),
                "base_valid_total": int(base_col.notna().sum()),
            }
        )

    summary = pd.DataFrame(records)
    if not summary.empty:
        summary = summary.sort_values(
            "delta_pct_total_accessibility",
            ascending=False,
        ).reset_index(drop=True)
    return summary


def add_delta_columns(
    municipality_df: pd.DataFrame,
    development_summary: pd.DataFrame,
) -> pd.DataFrame:
    result = municipality_df.copy()
    for dev_id in development_summary["development"]:
        dev_col = f"dev_{dev_id}"
        delta_col = f"delta_{dev_id}"
        delta_pct_col = f"delta_pct_{dev_id}"
        result[delta_col] = result["base_accessibility"] - result[dev_col]
        result[delta_pct_col] = np.where(
            result["base_accessibility"].notna() & (result["base_accessibility"] != 0),
            (result[delta_col] / result["base_accessibility"]) * 100.0,
            np.nan,
        )
    return result


def fill_missing_development_values(
    municipality_df: gpd.GeoDataFrame,
    development_ids: list[str],
) -> gpd.GeoDataFrame:
    result = municipality_df.copy()
    for dev_id in development_ids:
        dev_col = f"dev_{dev_id}"
        if dev_col in result.columns:
            result[dev_col] = result[dev_col].fillna(result["base_accessibility"])
    return result


def create_absolute_lineplot(
    municipality_df: pd.DataFrame,
    development_ids: list[str],
) -> None:
    plot_df = municipality_df.dropna(subset=["base_accessibility"]).copy()
    plot_df = plot_df.sort_values("base_accessibility", ascending=False).reset_index(drop=True)
    dev_cols = [f"dev_{dev_id}" for dev_id in development_ids if f"dev_{dev_id}" in plot_df.columns]

    fig, ax = plt.subplots(figsize=(22, 10), dpi=180)
    x = np.arange(len(plot_df))

    ax.plot(
        x,
        plot_df["base_accessibility"],
        color="black",
        linewidth=2.8,
        label="Status quo",
        zorder=200,
    )

    colors = plt.cm.Greens(np.linspace(0.35, 0.95, max(len(dev_cols), 2)))
    for idx, dev_col in enumerate(dev_cols):
        ax.plot(
            x,
            plot_df[dev_col],
            color=colors[idx],
            linewidth=0.9,
            alpha=0.65,
            label=dev_col.replace("dev_", "Dev "),
        )

    ax.set_xlabel("Municipalities ranked by base accessibility")
    ax.set_ylabel("Accessibility index")
    ax.set_title(f"Rail accessibility by municipality, all {len(dev_cols)} developments")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["GEMEINDENA"], rotation=90, fontsize=6)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, fontsize=7, ncol=6, loc="best")
    fig.tight_layout()
    fig.savefig(OUT_LINEPLOT / "rail_accessibility_all_developments_lines.png", bbox_inches="tight")
    plt.close(fig)


def create_total_accessibility_plot(development_summary: pd.DataFrame) -> None:
    plot_df = development_summary.copy()
    plot_df["label"] = "Dev " + plot_df["development"].astype(str)

    fig, ax = plt.subplots(figsize=(18, 8), dpi=180)
    colors = np.where(
        plot_df["delta_total_accessibility"] >= 0,
        "#0b6e0b",
        "#9b1d20",
    )
    ax.bar(
        plot_df["label"],
        plot_df["delta_total_accessibility"],
        color=colors,
        alpha=0.9,
    )
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Developments ranked by total accessibility change")
    ax.set_ylabel("Delta total accessibility")
    ax.set_title("Total municipality accessibility change by development")
    ax.tick_params(axis="x", rotation=90, labelsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(OUT_LINEPLOT / "rail_accessibility_total_delta_by_development.png", bbox_inches="tight")
    plt.close(fig)


def create_delta_pct_colormap() -> mcolors.LinearSegmentedColormap:
    return mcolors.LinearSegmentedColormap.from_list(
        "rail_delta_signed",
        ["#8b0000", "#d65c5c", "#f2f2f2", "#7fd47f", "#014b01"],
        N=256,
    )


def create_top10_delta_maps(
    municipality_gdf: gpd.GeoDataFrame,
    development_summary: pd.DataFrame,
    station_points: gpd.GeoDataFrame,
) -> None:
    top_devs = development_summary.head(TOP_N)["development"].tolist()
    if not top_devs:
        return

    delta_cols = [f"delta_pct_{dev_id}" for dev_id in top_devs if f"delta_pct_{dev_id}" in municipality_gdf.columns]
    vmax = 0.0
    for col in delta_cols:
        series = municipality_gdf[col].replace([np.inf, -np.inf], np.nan)
        vmax = max(vmax, np.nanmax(np.abs(series.to_numpy(dtype=float))))
    vmax = max(vmax, 0.1)

    base_map = municipality_gdf[["BFS", "geometry"]].drop_duplicates()
    base_network = load_network_overlay(get_active_base_network_path())

    for dev_id in top_devs:
        col = f"delta_pct_{dev_id}"
        if col not in municipality_gdf.columns:
            continue
        dev_network = load_active_development_overlay(dev_id)

        plot_gdf = municipality_gdf.copy()

        fig, ax = plt.subplots(figsize=(15, 12), dpi=180)
        base_map.plot(ax=ax, color="#f2f2f2", edgecolor="#cfcfcf", linewidth=0.5)
        if not base_network.empty:
            base_network.plot(ax=ax, color="#bdbdbd", linewidth=0.7, alpha=0.9, zorder=2)
        if not dev_network.empty:
            dev_network.plot(ax=ax, color="black", linewidth=1.4, alpha=0.95, zorder=3)
        if not station_points.empty:
            station_points.plot(
                ax=ax,
                color="#8f8f8f",
                markersize=7,
                alpha=0.9,
                zorder=4,
            )

        valid = plot_gdf.dropna(subset=[col])
        if not valid.empty:
            valid.plot(
                column=col,
                ax=ax,
                cmap=create_delta_pct_colormap(),
                vmin=-vmax,
                vmax=vmax,
                edgecolor="#8a8a8a",
                linewidth=0.5,
                legend=True,
                norm=mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                zorder=1,
                legend_kwds={
                    "label": "Accessibility change (%)",
                    "orientation": "vertical",
                    "shrink": 0.7,
                },
            )

        title_row = development_summary.loc[
            development_summary["development"] == dev_id,
            "delta_pct_total_accessibility",
        ]
        title_delta = float(title_row.iloc[0]) if not title_row.empty else np.nan
        ax.set_title(
            f"Development {dev_id} - municipality accessibility change\n"
            f"total delta: {title_delta:.2f}%",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_axis_off()
        ax.set_aspect("equal")
        fig.tight_layout()
        fig.savefig(OUT_DELTA / f"municipality_delta_pct_dev_{dev_id}.png", bbox_inches="tight")
        plt.close(fig)


def build_summary_tables() -> tuple[gpd.GeoDataFrame, pd.DataFrame, list[str], gpd.GeoDataFrame]:
    print("Loading active canton_ZH rail inputs...")
    station_lookup = load_station_lookup()
    station_od = load_station_od_matrix()
    commune_mapping = load_commune_station_mapping()
    communes = load_communes()
    cache = load_od_times_cache()
    base_od_times = cache["od_times_status_quo"][0]
    dev_od_times = cache["od_times_dev"]

    station_od_ids = set(station_od.index.astype(int)).intersection(set(station_od.columns.astype(int)))
    base_tt_all = od_times_to_long_form(base_od_times, station_lookup, station_od_ids)
    common_station_ids = set(base_tt_all["origin"]).union(set(base_tt_all["destination"]))
    common_station_ids &= station_od_ids
    print(f"  Common stations across active OD and network times: {len(common_station_ids)}")

    if not common_station_ids:
        raise ValueError("No common station IDs found between active OD and network times.")

    station_lookup = station_lookup[station_lookup["ID_point"].isin(common_station_ids)].copy()
    station_od = station_od.loc[sorted(common_station_ids), sorted(common_station_ids)]
    commune_mapping = commune_mapping[commune_mapping["ID_point"].isin(common_station_ids)].copy()
    pj = calculate_pj(station_od)
    base_tt = base_tt_all[
        base_tt_all["origin"].isin(common_station_ids)
        & base_tt_all["destination"].isin(common_station_ids)
    ].copy()

    dev_position_lookup = build_dev_position_lookup()
    development_ids = [
        dev_id
        for dev_id, pos in sorted(dev_position_lookup.items(), key=lambda item: item[1])
        if pos < len(dev_od_times)
    ]

    dev_times: dict[str, pd.DataFrame] = {}
    for dev_id in development_ids:
        pos = dev_position_lookup[dev_id]
        dev_times[dev_id] = od_times_to_long_form(
            dev_od_times[pos],
            station_lookup,
            common_station_ids,
        )

    print(f"  Computing municipality accessibility for {len(development_ids)} developments...")
    station_result = build_station_accessibility(
        station_lookup=station_lookup,
        pj=pj,
        base_tt=base_tt,
        dev_times=dev_times,
        development_ids=development_ids,
    )
    municipality_result = build_municipality_accessibility(
        station_accessibility=station_result,
        commune_mapping=commune_mapping,
        communes=communes,
    )

    value_columns = ["base_accessibility"] + [
        col for col in municipality_result.columns if col.startswith("dev_")
    ]
    municipality_result[value_columns] = municipality_result[value_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    municipality_result = fill_missing_development_values(
        municipality_result,
        development_ids,
    )

    development_summary = summarize_developments(
        municipality_result.drop(columns="geometry"),
        development_ids,
    )
    municipality_result = add_delta_columns(municipality_result, development_summary)
    return municipality_result, development_summary, development_ids, station_lookup


def save_summaries(
    municipality_result: gpd.GeoDataFrame,
    development_summary: pd.DataFrame,
) -> None:
    gpkg_path = OUT_SUMMARY / "rail_accessibility_municipalities_all_devs.gpkg"
    csv_path = OUT_SUMMARY / "rail_accessibility_municipalities_all_devs.csv"
    dev_csv_path = OUT_SUMMARY / "rail_accessibility_development_summary.csv"

    municipality_result.to_file(gpkg_path, driver="GPKG")
    municipality_result.drop(columns="geometry").to_csv(csv_path, index=False)
    development_summary.to_csv(dev_csv_path, index=False)

    print(f"  Saved municipality summary: {csv_path.name}")
    print(f"  Saved development summary: {dev_csv_path.name}")


def main() -> None:
    OUT_SUMMARY.mkdir(parents=True, exist_ok=True)
    OUT_LINEPLOT.mkdir(parents=True, exist_ok=True)
    OUT_DELTA.mkdir(parents=True, exist_ok=True)

    municipality_result, development_summary, development_ids, station_points = build_summary_tables()
    save_summaries(municipality_result, development_summary)

    print("Creating plots from summary tables...")
    create_absolute_lineplot(
        municipality_result.drop(columns="geometry"),
        development_ids,
    )
    create_total_accessibility_plot(development_summary)
    create_top10_delta_maps(municipality_result, development_summary, station_points)

    print("Done.")
    print(f"  Summary: {OUT_SUMMARY}")
    print(f"  Lineplots: {OUT_LINEPLOT}")
    print(f"  Delta maps: {OUT_DELTA}")


if __name__ == "__main__":
    main()
