import argparse
import os
import re
from typing import List

import geopandas as gpd
import pandas as pd
from tqdm import tqdm

from infraScan.infraScanRoad import scoring as road_scoring
from infraScan.infraScanRoad import settings as road_settings


def _discover_scenarios(base_dir: str) -> List[str]:
    od_dir = os.path.join(base_dir, "data/infraScanRoad/traffic_flow/od")
    files = [f for f in os.listdir(od_dir) if f.startswith("od_matrix_scenario_") and f.endswith(".csv")]
    return sorted([f.replace("od_matrix_", "").replace(".csv", "") for f in files])


def _discover_developments(base_dir: str, scenarios: List[str]) -> List[int]:
    dev_dir = os.path.join(base_dir, "data/infraScanRoad/traffic_flow/od/developments")
    scenario_set = set(scenarios)
    devs = set()
    for filename in os.listdir(dev_dir):
        match = re.match(r"od_matrix_dev(\d+)_(.+)\.csv", filename)
        if match and match.group(2) in scenario_set:
            devs.add(int(match.group(1)))
    return sorted(devs)


def _build_development_network(dev: int):
    links_developments = gpd.read_file("data/infraScanRoad/costs/construction.gpkg")
    if dev not in links_developments["ID_new"].values:
        return None, None

    points_developments = gpd.read_file("data/infraScanRoad/Network/processed/generated_nodes.gpkg")
    points_current = gpd.read_file("data/infraScanRoad/Network/processed/points_with_attribute.gpkg")

    point_temp = points_developments[points_developments["ID_new"] == dev]
    if point_temp.empty:
        return None, None

    points = points_current.copy()
    new_point_row = {
        "intersection": 0,
        "ID_point": 9999,
        "geometry": point_temp.geometry.iloc[0],
        "open_ends": None,
        "within_corridor": True,
        "on_corridor_border": False,
        "generate_traffic": 0,
    }
    points = gpd.GeoDataFrame(pd.concat([points, pd.DataFrame(pd.Series(new_point_row)).T], ignore_index=True))
    points.index = points.index.astype(int)
    points = points.sort_index()
    points["id_dummy"] = points.index.values

    links_current = gpd.read_file("data/infraScanRoad/Network/processed/edges_with_attribute.gpkg")
    edge_temp = links_developments[links_developments["ID_new"] == dev]
    edges = links_current.copy()
    edge_id_max = edges["ID_edge"].astype(int).max()
    index_point_start = points[points["id_dummy"] == edge_temp["ID_current"].values[0]].index[0]

    new_edge_row = {
        "start": index_point_start,
        "end": 9999,
        "geometry": edge_temp["geometry"].iloc[0],
        "ffs": 120,
        "capacity": 2200,
        "start_access": False,
        "end_access": True,
        "polygon_border": False,
        "ID_edge": edge_id_max + 1,
    }
    edges = gpd.GeoDataFrame(pd.concat([edges, pd.DataFrame(pd.Series(new_edge_row)).T], ignore_index=True))
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=["ID_edge"])

    return points, edges


def run_analysis(max_developments: int | None = None):
    os.chdir(road_settings.MAIN)

    scenarios = _discover_scenarios(road_settings.MAIN)
    if not scenarios:
        raise FileNotFoundError("No scenario OD matrices found: data/infraScanRoad/traffic_flow/od/od_matrix_scenario_*.csv")

    developments = _discover_developments(road_settings.MAIN, scenarios)
    if max_developments is not None and max_developments > 0:
        developments = developments[:max_developments]

    if not developments:
        raise FileNotFoundError("No development OD matrices found in data/infraScanRoad/traffic_flow/od/developments")

    points_sq = gpd.read_file("data/infraScanRoad/Network/processed/points_with_attribute.gpkg")
    points_sq.index = points_sq.index.astype(int)
    points_sq = points_sq.sort_index()

    edges_sq = gpd.read_file("data/infraScanRoad/Network/processed/edges_with_attribute.gpkg")
    edges_sq["ID_edge"] = edges_sq["ID_edge"].astype(int)
    edges_sq = edges_sq.sort_values(by=["ID_edge"])

    mon_factor = road_settings.VTTS * 2.5 * 250 * road_settings.travel_time_duration

    rows = []
    for dev in tqdm(developments, desc="Developments"):
        points_dev, edges_dev = _build_development_network(dev)
        if points_dev is None:
            continue

        voronoi_path = f"data/infraScanRoad/Network/travel_time/developments/dev{dev}_Voronoi.gpkg"
        if not os.path.exists(voronoi_path):
            continue
        voronoi_df = gpd.read_file(voronoi_path)

        for scen in scenarios:
            od_path = f"data/infraScanRoad/traffic_flow/od/developments/od_matrix_dev{dev}_{scen}.csv"
            if not os.path.exists(od_path):
                continue

            od_matrix = pd.read_csv(od_path, sep=",", index_col=0)

            tt_dev = road_scoring.travel_flow_optimization(
                OD_matrix=od_matrix,
                points=points_dev.copy(),
                edges=edges_dev.copy(),
                voronoi=voronoi_df,
                dev=dev,
                scen=scen,
            )

            tt_sq = road_scoring.travel_flow_optimization(
                OD_matrix=od_matrix,
                points=points_sq.copy(),
                edges=edges_sq.copy(),
                voronoi=voronoi_df,
                dev="status_quo",
                scen=scen,
            )

            delta_tt = float(tt_sq - tt_dev)
            monetized_signed = delta_tt * mon_factor

            rows.append(
                {
                    "development": int(dev),
                    "scenario": str(scen),
                    "tt_status_quo": float(tt_sq),
                    "tt_development": float(tt_dev),
                    "delta_tt": delta_tt,
                    "tt_signed": monetized_signed,
                    "tt_forced_negative": -abs(monetized_signed),
                }
            )

    if not rows:
        raise RuntimeError("No analysis rows were generated. Check OD/development inputs.")

    out_df = pd.DataFrame(rows)
    costs_dir = "data/infraScanRoad/costs"
    os.makedirs(costs_dir, exist_ok=True)

    detailed_path = os.path.join(costs_dir, "tt_analysis_aggregate_same_basis_detailed.csv")
    out_df.to_csv(detailed_path, index=False)

    wide_signed = out_df.pivot(index="development", columns="scenario", values="tt_signed").reset_index()
    wide_signed.columns.name = None
    wide_signed = wide_signed.rename(columns={c: f"tt_{c}" for c in wide_signed.columns if c != "development"})

    wide_forced_neg = out_df.pivot(index="development", columns="scenario", values="tt_forced_negative").reset_index()
    wide_forced_neg.columns.name = None
    wide_forced_neg = wide_forced_neg.rename(columns={c: f"tt_{c}" for c in wide_forced_neg.columns if c != "development"})

    signed_path = os.path.join(costs_dir, "tt_analysis_aggregate_same_basis_signed.csv")
    forced_path = os.path.join(costs_dir, "tt_analysis_aggregate_same_basis_forced_negative.csv")
    wide_signed.to_csv(signed_path, index=False)
    wide_forced_neg.to_csv(forced_path, index=False)

    print("Saved analysis files:")
    print(f"- {detailed_path}")
    print(f"- {signed_path}")
    print(f"- {forced_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apples-to-apples TT method analysis (aggregate on OD basis)")
    parser.add_argument("--max-developments", type=int, default=None, help="Optional cap for faster analysis runs")
    args = parser.parse_args()

    run_analysis(max_developments=args.max_developments)