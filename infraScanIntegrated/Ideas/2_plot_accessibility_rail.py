"""
Rail accessibility maps for the active infraScanRail canton_ZH setup.

This version follows the active rail workflow:
- station-based OD demand from `od_matrix_stations_ktzh_20.csv`
- station-to-station rail network times from `od_times.pkl`
- no access-time term; only the rail network time is used

The output is a set of municipality lineplots. Accessibility is computed on the
station level and then assigned to municipalities via the active commune-to-
station mapping used by `infraScanRail`.
"""

from __future__ import annotations

import os
import pickle
import sqlite3
from pathlib import Path

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/Volumes/WD_Windows/MSc_Thesis/data")
RAIL = DATA / "infraScanRail"
OUT = ROOT / "infraScan" / "plots" / "rail_accessibility_maps" / "municipality_lineplots"

POINTS_GPKG = RAIL / "Network/processed/points.gpkg"
STATION_OD_PATH = RAIL / "traffic_flow/od/rail/ktzh/od_matrix_stations_ktzh_20.csv"
OD_TIMES_CACHE = RAIL / "Network/travel_time/cache/od_times.pkl"
DEVELOPMENT_DIRECTORY = RAIL / "Network/processed/developments"
COMMUNE_TO_STATION_PATH = RAIL / "Network/processed/Communes_to_railway_stations_ZH.xlsx"
COMMUNE_SHP = DATA / "_basic_data/Gemeindegrenzen/UP_GEMEINDEN_F.shp"

TOP_N = 10
NAME_NORMALIZATION = {
    "Niederglatt": "Niederglatt ZH",
    "Oberglatt": "Oberglatt ZH",
}


def selected_top_developments(n: int = TOP_N) -> list[str]:
    total_costs = pd.read_csv(RAIL / "costs/total_costs.csv")
    total_costs["dev_id"] = (
        total_costs["development"].str.extract(r"Development_(\d+)").astype(str)
    )
    ranked = total_costs.sort_values("total_costs", ascending=False)
    return ranked.head(n)["dev_id"].tolist()


def load_station_lookup() -> pd.DataFrame:
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
    gdf = gpd.GeoDataFrame(
        lookup,
        geometry=gpd.points_from_xy(lookup["XKOORD"], lookup["YKOORD"]),
        crs="EPSG:2056",
    )
    return gdf[["ID_point", "station_name", "geometry"]].drop_duplicates()


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
    od = od.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return od


def load_commune_station_mapping() -> pd.DataFrame:
    mapping = pd.read_excel(COMMUNE_TO_STATION_PATH)
    mapping = mapping.rename(columns={"Commune_BFS_code": "BFS"})
    mapping = mapping[["BFS", "Commune", "ID_point", "Station"]].copy()
    mapping["BFS"] = pd.to_numeric(mapping["BFS"], errors="coerce")
    mapping["ID_point"] = pd.to_numeric(mapping["ID_point"], errors="coerce")
    mapping = mapping.dropna(subset=["BFS", "ID_point"]).copy()
    mapping["BFS"] = mapping["BFS"].astype(int)
    mapping["ID_point"] = mapping["ID_point"].astype(int)
    mapping = mapping[mapping["BFS"] > 0].copy()
    return mapping.drop_duplicates(subset=["BFS"])


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


def add_accessibility_metric(
    tt_df: pd.DataFrame,
    pj: pd.Series,
) -> pd.DataFrame:
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
    developments: list[str],
) -> gpd.GeoDataFrame:
    result = station_lookup.merge(
        add_accessibility_metric(base_tt, pj).rename(
            columns={"accessibility": "base_accessibility"}
        ),
        on="ID_point",
        how="left",
    )

    for dev_id in developments:
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
    municipality_accessibility = communes.merge(
        municipality_accessibility,
        on="BFS",
        how="left",
    )
    return municipality_accessibility


def plot_accessibility_lines(
    df: pd.DataFrame,
    developments: list[str],
    output_prefix: str = "rail_accessibility_municipalities",
) -> None:
    plot_df = df.dropna(subset=["base_accessibility"]).copy()
    plot_df = plot_df.sort_values("base_accessibility", ascending=False).reset_index(drop=True)
    dev_cols = [f"dev_{dev_id}" for dev_id in developments if f"dev_{dev_id}" in plot_df.columns]
    x = np.arange(len(plot_df))
    colors = plt.cm.tab20(np.linspace(0, 1, len(dev_cols)))

    fig, ax = plt.subplots(figsize=(13, 6), dpi=240)
    ax.plot(x, plot_df["base_accessibility"], label="Base", color="black", linewidth=2.2)
    for color, col in zip(colors, dev_cols):
        ax.plot(
            x,
            plot_df[col],
            label=col.replace("dev_", "Dev "),
            linewidth=1.2,
            color=color,
            alpha=0.92,
        )
    ax.set_xlabel("Municipalities ranked by base accessibility")
    ax.set_ylabel("Accessibility index")
    ax.set_title(f"Rail accessibility by municipality, top {len(dev_cols)} developments")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"{output_prefix}_lines.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=240)
    ax.axhline(0, color="black", linewidth=1.0)
    for color, col in zip(colors, dev_cols):
        ax.plot(
            x,
            plot_df[col] - plot_df["base_accessibility"],
            label=col.replace("dev_", "Dev "),
            linewidth=1.3,
            color=color,
            alpha=0.94,
        )
    ax.set_xlabel("Municipalities ranked by base accessibility")
    ax.set_ylabel("Delta accessibility vs base")
    ax.set_title(f"Rail accessibility delta by municipality, top {len(dev_cols)} developments")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(OUT / f"{output_prefix}_delta_lines.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading active canton_ZH station inputs...")
    station_lookup = load_station_lookup()
    station_od = load_station_od_matrix()
    commune_mapping = load_commune_station_mapping()
    communes = load_communes()
    cache = load_od_times_cache()
    base_od_times = cache["od_times_status_quo"][0]
    dev_od_times = cache["od_times_dev"]

    station_od_ids = set(station_od.index.astype(int)).intersection(
        set(station_od.columns.astype(int))
    )
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

    developments = selected_top_developments(TOP_N)
    dev_position_lookup = build_dev_position_lookup()
    dev_times: dict[str, pd.DataFrame] = {}
    for dev_id in developments:
        pos = dev_position_lookup.get(dev_id)
        if pos is None or pos >= len(dev_od_times):
            print(f"Warning: development {dev_id} missing from od_times cache")
            continue
        dev_times[dev_id] = od_times_to_long_form(
            dev_od_times[pos],
            station_lookup,
            common_station_ids,
        )

    print("Computing accessibility per municipality...")
    station_result = build_station_accessibility(
        station_lookup=station_lookup,
        pj=pj,
        base_tt=base_tt,
        dev_times=dev_times,
        developments=developments,
    )
    result = build_municipality_accessibility(
        station_accessibility=station_result,
        commune_mapping=commune_mapping,
        communes=communes,
    )

    value_columns = ["base_accessibility"] + [
        col for col in result.columns if col.startswith("dev_")
    ]
    result[value_columns] = result[value_columns].apply(pd.to_numeric, errors="coerce")

    gpkg_path = OUT / "rail_accessibility_municipalities.gpkg"
    csv_path = OUT / "rail_accessibility_municipalities.csv"
    result.to_file(gpkg_path, driver="GPKG")
    result.drop(columns="geometry").to_csv(csv_path, index=False)

    print("Creating accessibility lineplots...")
    plot_accessibility_lines(result.drop(columns="geometry"), developments)

    print(f"Saved outputs to: {OUT}")


if __name__ == "__main__":
    main()
