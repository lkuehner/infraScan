from __future__ import annotations

import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns

from infraScan.infraScanRail import paths as rail_paths
from infraScan.infraScanRail.TT_Delay import analyze_travel_times
from infraScan.infraScanRoad.scoring import GetHighwayPHDemandPerCommune, GetODMatrix
from infraScan.infraScanIntegrated.Ideas.analysis_final_costs import (
    load_rail_final_costs,
    load_road_final_cost_components,
)


SCENARIOS = ("scenario_76", "scenario_45", "scenario_67")
RAIL_YEAR = 2050


def _data_root() -> Path:
    return Path(rail_paths.MAIN)


def _output_dir() -> Path:
    out = Path(__file__).resolve().parents[2] / "plots" / "Integrated_Final_Costs"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _workspace_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_top5_developments_from_final_costs() -> tuple[list[str], list[str]]:
    rail_data = load_rail_final_costs()
    road_data = load_road_final_cost_components()

    rail_top5 = (
        rail_data.groupby(["development", "line_name"], as_index=False)
        .agg({"net_benefit_mio_chf": "mean"})
        .sort_values("net_benefit_mio_chf", ascending=False)
        .head(5)["development"]
        .astype(str)
        .tolist()
    )
    road_top5 = (
        road_data.groupby(["development", "line_name"], as_index=False)
        .agg({"net_benefit_mio_chf": "mean"})
        .sort_values("net_benefit_mio_chf", ascending=False)
        .head(5)
        ["development"]
        .astype(str)
        .tolist()
    )
    return rail_top5, road_top5


def get_top5_labels_from_final_costs() -> dict[str, str]:
    rail_data = load_rail_final_costs()
    road_data = load_road_final_cost_components()

    rail_top = (
        rail_data.groupby(["development", "line_name"], as_index=False)
        .agg({"net_benefit_mio_chf": "mean"})
        .sort_values("net_benefit_mio_chf", ascending=False)
        .head(5)
    )
    road_top = (
        road_data.groupby(["development", "line_name"], as_index=False)
        .agg({"net_benefit_mio_chf": "mean"})
        .sort_values("net_benefit_mio_chf", ascending=False)
        .head(5)
    )

    labels: dict[str, str] = {}
    for _, row in rail_top.iterrows():
        labels[f"Rail {str(row['development'])}"] = str(row["line_name"])
    for _, row in road_top.iterrows():
        labels[f"Road {str(row['development'])}"] = "R" + str(row["line_name"])
    return labels


def create_rail_dev_id_lookup_table() -> pd.DataFrame:
    dev_dir = _data_root() / rail_paths.DEVELOPMENT_DIRECTORY
    dev_ids = sorted(
        str(int(float(os.path.splitext(path.name)[0])))
        for path in dev_dir.iterdir()
        if path.is_file() and not path.name.startswith("._")
    )
    return pd.DataFrame({"dev_id": dev_ids}, index=range(1, len(dev_ids) + 1))


def run_rail_analyze_travel_times_top5() -> list[Path]:
    root = _data_root()
    workspace_root = _workspace_root()
    rail_top5, _ = get_top5_developments_from_final_costs()
    lookup = create_rail_dev_id_lookup_table()

    cache_path = root / "data" / "infraScanRail" / "Network" / "travel_time" / "cache" / "od_times.pkl"
    with cache_path.open("rb") as handle:
        cache = pickle.load(handle)

    dev_id_to_position = {
        str(dev_id): idx
        for idx, dev_id in enumerate(lookup["dev_id"].astype(str).tolist())
    }
    selected_positions = [dev_id_to_position[dev_id] for dev_id in rail_top5]
    selected_od_times = [cache["od_times_dev"][idx] for idx in selected_positions]
    selected_lookup = pd.DataFrame({"dev_id": rail_top5}, index=range(1, len(rail_top5) + 1))

    original_cwd = Path.cwd()
    try:
        os.chdir(workspace_root)
        analyze_travel_times(
            od_times_status_quo=cache["od_times_status_quo"],
            od_times_dev=selected_od_times,
            od_nodes=list(cache["od_times_status_quo"][0]["from_station"].unique()),
            dev_id_lookup_table=selected_lookup,
        )
    finally:
        os.chdir(original_cwd)

    savings_dir = workspace_root / "data" / "infraScanRail" / "Network" / "travel_time" / "TravelTime_Savings"
    return [savings_dir / f"TravelTime_Savings_Dev_{dev_id}.csv" for dev_id in rail_top5]


def load_rail_od_savings_top5() -> pd.DataFrame:
    rail_top5, _ = get_top5_developments_from_final_costs()
    csv_paths = run_rail_analyze_travel_times_top5()

    frames: list[pd.DataFrame] = []
    for dev_id, csv_path in zip(rail_top5, csv_paths):
        df = pd.read_csv(csv_path)
        df["development"] = str(dev_id)
        df["mode"] = "Rail"
        df["scenario"] = "all_selected_scenarios"
        # In analyze_travel_times, savings are status quo minus development time.
        df["tts_minutes"] = pd.to_numeric(df["status_quo_time"], errors="coerce") - pd.to_numeric(
            df["new_time"], errors="coerce"
        )
        # For the final comparison plot, keep only affected OD relations.
        # This mirrors the road side, where we also keep only affected cells.
        df = df[np.isfinite(df["tts_minutes"]) & (np.abs(df["tts_minutes"]) > 1e-9)].copy()
        frames.append(df[["mode", "development", "scenario", "origin", "destination", "tts_minutes"]])

    return pd.concat(frames, ignore_index=True)


def load_road_affected_cell_savings_top5() -> pd.DataFrame:
    root = _data_root()
    _, road_top5 = get_top5_developments_from_final_costs()
    with rasterio.open(root / "data" / "infraScanRoad" / "Network" / "travel_time" / "travel_time_raster.tif") as src:
        sq_tt = src.read(1).astype(float)

    frames: list[pd.DataFrame] = []
    for dev_id in road_top5:
        with rasterio.open(
            root / "data" / "infraScanRoad" / "Network" / "travel_time" / "developments" / f"dev{dev_id}_travel_time_raster.tif"
        ) as src:
            dev_tt = src.read(1).astype(float)
        with rasterio.open(
            root / "data" / "infraScanRoad" / "Network" / "travel_time" / "developments" / f"dev{dev_id}_source_id_raster.tif"
        ) as src:
            dev_source_id = src.read(1)

        delta_min = (sq_tt - dev_tt) / 60.0
        affected_mask = dev_source_id == 9999
        affected_values = delta_min[affected_mask]
        affected_values = affected_values[np.isfinite(affected_values)]

        frames.append(
            pd.DataFrame(
                {
                    "mode": "Road",
                    "development": str(dev_id),
                    "scenario": "affected_cells",
                    "tts_minutes": affected_values,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def _load_road_population_raster_for_scenario(root: Path, scenario: str) -> np.ndarray:
    candidates = [
        root / "data" / "independent_variable" / "processed" / "scenario" / f"{scenario}_pop.tif",
        root
        / "data"
        / "independent_variable"
        / "processed"
        / "scenario"
        / f"{scenario}_pop_{RAIL_YEAR}.tif",
    ]
    raster_path = next((path for path in candidates if path.exists()), None)
    if raster_path is None:
        raise FileNotFoundError(f"Missing population raster for scenario '{scenario}'. Tried: {candidates}")
    with rasterio.open(raster_path) as src:
        return src.read(1).astype(float)


def _load_road_mass_rasters_for_scenario(root: Path, scenario: str) -> tuple[np.ndarray, np.ndarray]:
    pop = _load_road_population_raster_for_scenario(root, scenario)
    # Current generated-network monetize_tts_network path uses population on both sides.
    return pop, pop.copy()


def _load_road_commune_raster(root: Path) -> np.ndarray:
    raster_path = root / "data" / "_basic_data" / "Gemeindegrenzen" / "gemeinde_zh.tif"
    if not raster_path.exists():
        raise FileNotFoundError(f"Missing commune raster: {raster_path}")
    with rasterio.open(raster_path) as src:
        return src.read(1)


def _road_shares_by_commune_and_catchment(
    commune_raster: np.ndarray, catchment_raster: np.ndarray, pop_raster: np.ndarray
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for commune_id in np.unique(commune_raster):
        if commune_id <= 0:
            continue
        commune_mask = commune_raster == commune_id
        if not np.any(commune_mask):
            continue
        for catchment_id in np.unique(catchment_raster[commune_mask]):
            if catchment_id <= 0:
                continue
            overlap_mask = commune_mask & (catchment_raster == catchment_id)
            pop_mass = float(np.nansum(pop_raster[overlap_mask]))
            if pop_mass <= 0:
                continue
            rows.append(
                {
                    "commune_id": int(commune_id),
                    "catchment_id": int(catchment_id),
                    "pop_mass": pop_mass,
                }
            )

    shares = pd.DataFrame(rows)
    if shares.empty:
        return shares
    shares["commune_total_pop"] = shares.groupby("commune_id")["pop_mass"].transform("sum")
    shares = shares[shares["commune_total_pop"] > 0].copy()
    shares["share"] = shares["pop_mass"] / shares["commune_total_pop"]
    return shares[["commune_id", "catchment_id", "share"]]


def _road_commune_catchment_stats(
    commune_raster: np.ndarray, catchment_raster: np.ndarray, mass_raster: np.ndarray, access_minutes: np.ndarray
) -> pd.DataFrame:
    valid = (
        (catchment_raster > 0)
        & (commune_raster > 0)
        & np.isfinite(mass_raster)
        & np.isfinite(access_minutes)
        & (mass_raster >= 0)
    )
    if not np.any(valid):
        return pd.DataFrame(columns=["commune_id", "catchment_id", "share", "mean_access_min"])

    df = pd.DataFrame(
        {
            "commune_id": commune_raster[valid].astype(int),
            "catchment_id": catchment_raster[valid].astype(int),
            "mass": mass_raster[valid],
            "access_weighted": mass_raster[valid] * access_minutes[valid],
        }
    )
    grouped = df.groupby(["commune_id", "catchment_id"], as_index=False)[["mass", "access_weighted"]].sum()
    grouped = grouped[grouped["mass"] > 0].copy()
    if grouped.empty:
        return pd.DataFrame(columns=["commune_id", "catchment_id", "share", "mean_access_min"])

    grouped["commune_total_mass"] = grouped.groupby("commune_id")["mass"].transform("sum")
    grouped["share"] = grouped["mass"] / grouped["commune_total_mass"]
    grouped["mean_access_min"] = grouped["access_weighted"] / grouped["mass"]
    return grouped[["commune_id", "catchment_id", "share", "mean_access_min"]]


def _restrict_and_renormalize_shares(shares: pd.DataFrame, valid_catchment_ids: set[int]) -> pd.DataFrame:
    if shares.empty:
        return shares
    filtered = shares[shares["catchment_id"].isin(valid_catchment_ids)].copy()
    if filtered.empty:
        return filtered
    filtered["commune_total_share"] = filtered.groupby("commune_id")["share"].transform("sum")
    filtered = filtered[filtered["commune_total_share"] > 0].copy()
    filtered["share"] = filtered["share"] / filtered["commune_total_share"]
    return filtered[["commune_id", "catchment_id", "share"]]


def _restrict_and_renormalize_stats(stats: pd.DataFrame, valid_catchment_ids: set[int]) -> pd.DataFrame:
    if stats.empty:
        return stats
    filtered = stats[stats["catchment_id"].isin(valid_catchment_ids)].copy()
    if filtered.empty:
        return filtered
    filtered["commune_total_share"] = filtered.groupby("commune_id")["share"].transform("sum")
    filtered = filtered[filtered["commune_total_share"] > 0].copy()
    filtered["share"] = filtered["share"] / filtered["commune_total_share"]
    return filtered[["commune_id", "catchment_id", "share", "mean_access_min"]]


def _road_expected_tt_by_commune_od(
    od_long_df: pd.DataFrame, origin_shares: pd.DataFrame, dest_shares: pd.DataFrame, tt_df: pd.DataFrame
) -> pd.DataFrame:
    origin_map = origin_shares.rename(
        columns={"commune_id": "origin_commune", "catchment_id": "origin", "share": "origin_share"}
    )
    dest_map = dest_shares.rename(
        columns={"commune_id": "destination_commune", "catchment_id": "destination", "share": "destination_share"}
    )

    redistributed = od_long_df.merge(origin_map, on="origin_commune", how="inner")
    redistributed = redistributed.merge(dest_map, on="destination_commune", how="inner")
    redistributed = redistributed.merge(tt_df[["origin", "destination", "travel_time"]], on=["origin", "destination"], how="left")
    redistributed = redistributed.dropna(subset=["travel_time"]).copy()

    redistributed["expected_tt_component"] = (
        redistributed["origin_share"] * redistributed["destination_share"] * redistributed["travel_time"]
    )

    return (
        redistributed.groupby(["origin_commune", "destination_commune", "od_demand"], as_index=False)[
            "expected_tt_component"
        ]
        .sum()
        .rename(columns={"expected_tt_component": "expected_tt_hours"})
    )


def _road_expected_generalized_tt_by_commune_od(weighted_df: pd.DataFrame, tt_col: str) -> pd.DataFrame:
    expected = weighted_df.copy()
    expected[tt_col] = pd.to_numeric(expected[tt_col], errors="coerce")
    expected = expected.dropna(subset=[tt_col]).copy()
    expected["expected_tt_component"] = (
        expected["origin_share"] * expected["destination_share"] * expected[tt_col]
    )
    return (
        expected.groupby(["origin_commune", "destination_commune", "od_demand"], as_index=False)[
            "expected_tt_component"
        ]
        .sum()
        .rename(columns={"expected_tt_component": tt_col})
    )


def _road_redistribute_commune_od_with_access(
    od_long_df: pd.DataFrame, origin_stats: pd.DataFrame, dest_stats: pd.DataFrame
) -> pd.DataFrame:
    origin_map = origin_stats.rename(
        columns={
            "commune_id": "origin_commune",
            "catchment_id": "origin",
            "share": "origin_share",
            "mean_access_min": "origin_access_min",
        }
    )
    dest_map = dest_stats.rename(
        columns={
            "commune_id": "destination_commune",
            "catchment_id": "destination",
            "share": "destination_share",
            "mean_access_min": "destination_access_min",
        }
    )
    redistributed = od_long_df.merge(origin_map, on="origin_commune", how="inner")
    redistributed = redistributed.merge(dest_map, on="destination_commune", how="inner")
    redistributed["flow"] = (
        redistributed["od_demand"] * redistributed["origin_share"] * redistributed["destination_share"]
    )
    redistributed = redistributed[redistributed["origin"] != redistributed["destination"]].copy()
    redistributed["origin"] = redistributed["origin"].astype(int)
    redistributed["destination"] = redistributed["destination"].astype(int)
    return redistributed


def load_road_validated_commune_od_savings_top5() -> pd.DataFrame:
    root = _data_root()
    _, road_top5 = get_top5_developments_from_final_costs()

    tt_status_quo = pd.read_csv(root / "data" / "infraScanRoad" / "traffic_flow" / "od" / "status_quo_od_tt.csv")
    tt_developments = pd.read_csv(root / "data" / "infraScanRoad" / "traffic_flow" / "od" / "developments_od_tt.csv")

    tt_status_quo["scenario"] = tt_status_quo["scenario"].astype(str)
    tt_developments["scenario"] = tt_developments["scenario"].astype(str)
    for col in ["origin", "destination", "travel_time"]:
        tt_status_quo[col] = pd.to_numeric(tt_status_quo[col], errors="coerce")
    for col in ["development", "origin", "destination", "travel_time"]:
        tt_developments[col] = pd.to_numeric(tt_developments[col], errors="coerce")

    tt_status_quo = tt_status_quo.dropna(subset=["origin", "destination", "travel_time"]).copy()
    tt_developments = tt_developments.dropna(subset=["development", "origin", "destination", "travel_time"]).copy()
    tt_status_quo["origin"] = tt_status_quo["origin"].astype(int)
    tt_status_quo["destination"] = tt_status_quo["destination"].astype(int)
    tt_developments["development"] = tt_developments["development"].astype(int)
    tt_developments["origin"] = tt_developments["origin"].astype(int)
    tt_developments["destination"] = tt_developments["destination"].astype(int)

    sq_source_path = root / "data" / "infraScanRoad" / "Network" / "travel_time" / "source_id_raster.tif"
    with rasterio.open(sq_source_path) as src:
        sq_source = src.read(1)
    with rasterio.open(root / "data" / "infraScanRoad" / "Network" / "travel_time" / "travel_time_raster.tif") as src:
        sq_access_minutes = src.read(1).astype(float) / 60.0

    original_cwd = Path.cwd()
    try:
        os.chdir(root)
        commune_raster = _load_road_commune_raster(root)
        od = GetHighwayPHDemandPerCommune()
        odmat = GetODMatrix(od).astype(float)
    finally:
        os.chdir(original_cwd)

    od_long = odmat.stack().rename("od_demand").reset_index()
    od_long.columns = ["origin_commune", "destination_commune", "od_demand"]
    od_long["origin_commune"] = pd.to_numeric(od_long["origin_commune"], errors="coerce").astype(int)
    od_long["destination_commune"] = pd.to_numeric(od_long["destination_commune"], errors="coerce").astype(int)
    od_long["od_demand"] = pd.to_numeric(od_long["od_demand"], errors="coerce")
    od_long = od_long[
        (od_long["origin_commune"] != od_long["destination_commune"]) & (od_long["od_demand"] > 0)
    ].copy()

    frames: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        origin_mass_raster, destination_mass_raster = _load_road_mass_rasters_for_scenario(root, scenario)
        sq_tt = tt_status_quo[tt_status_quo["scenario"] == scenario].copy()
        valid_sq_ids = set(sq_tt["origin"]).union(set(sq_tt["destination"]))
        sq_origin_stats = _restrict_and_renormalize_stats(
            _road_commune_catchment_stats(commune_raster, sq_source, origin_mass_raster, sq_access_minutes),
            valid_sq_ids,
        )
        sq_dest_stats = _restrict_and_renormalize_stats(
            _road_commune_catchment_stats(commune_raster, sq_source, destination_mass_raster, sq_access_minutes),
            valid_sq_ids,
        )
        sq_flows = _road_redistribute_commune_od_with_access(od_long, sq_origin_stats, sq_dest_stats)
        sq_weighted = sq_flows.merge(
            sq_tt[["origin", "destination", "travel_time"]], on=["origin", "destination"], how="left"
        )
        sq_weighted["status_quo_generalized_tt"] = (
            sq_weighted["origin_access_min"]
            + sq_weighted["travel_time"]
            + sq_weighted["destination_access_min"]
        )
        sq_expected = _road_expected_generalized_tt_by_commune_od(
            sq_weighted,
            tt_col="status_quo_generalized_tt",
        )

        for dev_id in road_top5:
            dev_tt = tt_developments[
                (tt_developments["scenario"] == scenario) & (tt_developments["development"] == int(dev_id))
            ].copy()
            if dev_tt.empty:
                continue

            with rasterio.open(
                root / "data" / "infraScanRoad" / "Network" / "travel_time" / "developments" / f"dev{dev_id}_source_id_raster.tif"
            ) as src:
                dev_source = src.read(1)
            with rasterio.open(
                root / "data" / "infraScanRoad" / "Network" / "travel_time" / "developments" / f"dev{dev_id}_travel_time_raster.tif"
            ) as src:
                dev_access_minutes = src.read(1).astype(float) / 60.0

            valid_dev_ids = set(dev_tt["origin"]).union(set(dev_tt["destination"]))
            dev_origin_stats = _restrict_and_renormalize_stats(
                _road_commune_catchment_stats(commune_raster, dev_source, origin_mass_raster, dev_access_minutes),
                valid_dev_ids,
            )
            dev_dest_stats = _restrict_and_renormalize_stats(
                _road_commune_catchment_stats(commune_raster, dev_source, destination_mass_raster, dev_access_minutes),
                valid_dev_ids,
            )
            dev_flows = _road_redistribute_commune_od_with_access(od_long, dev_origin_stats, dev_dest_stats)
            dev_weighted = dev_flows.merge(
                dev_tt[["origin", "destination", "travel_time"]], on=["origin", "destination"], how="left"
            )
            dev_weighted["development_generalized_tt"] = (
                dev_weighted["origin_access_min"]
                + dev_weighted["travel_time"]
                + dev_weighted["destination_access_min"]
            )
            dev_expected = _road_expected_generalized_tt_by_commune_od(
                dev_weighted,
                tt_col="development_generalized_tt",
            )

            merged = sq_expected.merge(
                dev_expected,
                on=["origin_commune", "destination_commune", "od_demand"],
                how="inner",
            )
            merged["tts_minutes"] = (
                pd.to_numeric(merged["status_quo_generalized_tt"], errors="coerce")
                - pd.to_numeric(merged["development_generalized_tt"], errors="coerce")
            )
            merged["mode"] = "Road"
            merged["development"] = str(dev_id)
            merged["scenario"] = str(scenario)
            frames.append(
                merged[
                    [
                        "mode",
                        "development",
                        "scenario",
                        "origin_commune",
                        "destination_commune",
                        "od_demand",
                        "tts_minutes",
                    ]
                ].rename(columns={"origin_commune": "origin", "destination_commune": "destination", "od_demand": "demand"})
            )

    return pd.concat(frames, ignore_index=True)


def load_od_level_tts_top5() -> pd.DataFrame:
    rail = load_rail_od_savings_top5()
    # For the road mode, the OD-level comparison plot should show the same
    # underlying object as the raster-based cost comparison: the cell-wise
    # access/egress savings on affected cells. This avoids mixing in a different
    # commune-OD expectation object that can contain positive and negative rows.
    road = load_road_affected_cell_savings_top5()
    return pd.concat([rail, road], ignore_index=True, sort=False)


def load_weighted_tts_top5() -> pd.DataFrame:
    root = _data_root()
    rail_top5, road_top5 = get_top5_developments_from_final_costs()

    rail = pd.read_csv(root / "data" / "infraScanRail" / "costs" / "traveltime_savings.csv")
    rail = rail[
        rail["scenario"].isin(SCENARIOS)
        & (rail["year"] == RAIL_YEAR)
        & (rail["development"].astype(int).astype(str).isin(rail_top5))
    ].copy()
    rail["mode"] = "Rail"
    rail["development"] = rail["development"].astype(int).astype(str)
    # Rail tt_savings_daily is stored in hours; convert to minutes for cross-mode comparison.
    rail["tts_minutes"] = rail["tt_savings_daily"].astype(float) * 60.0

    road = pd.read_csv(root / "data" / "infraScanRoad" / "traffic_flow" / "od" / "od_tt_savings_detailed.csv")
    road = road[
        road["scenario"].isin(SCENARIOS)
        & (road["development"].astype(int).astype(str).isin(road_top5))
    ].copy()
    road["mode"] = "Road"
    road["development"] = road["development"].astype(int).astype(str)
    road["tts_minutes"] = road["tt_savings_peak"].astype(float)

    return pd.concat(
        [
            rail[["mode", "development", "scenario", "tts_minutes"]],
            road[["mode", "development", "scenario", "tts_minutes"]],
        ],
        ignore_index=True,
    )


def load_deweighted_tts_top5_proxy() -> pd.DataFrame:
    """
    Demand-deweighted proxy:
    - Road: divide tt_savings_peak by total scenario OD demand
    - Rail: divide tt_savings_daily by total rail OD demand matrix sum

    This gives an approximate per-trip minutes interpretation.
    """
    root = _data_root()
    weighted = load_weighted_tts_top5()

    road_demand = pd.read_csv(root / "data" / "infraScanRoad" / "traffic_flow" / "od" / "status_quo_od_tt.csv")
    road_demand = (
        road_demand[road_demand["scenario"].isin(SCENARIOS)]
        .groupby("scenario", as_index=False)["demand"]
        .sum()
        .rename(columns={"demand": "total_demand"})
    )

    rail_od = pd.read_csv(
        root / "data" / "infraScanRail" / "traffic_flow" / "od" / "rail" / "ktzh" / "od_matrix_stations_ktzh_20.csv",
        index_col=0,
    )
    rail_total_demand = float(rail_od.to_numpy().sum())

    road = weighted[weighted["mode"] == "Road"].merge(road_demand, on="scenario", how="left")
    road["tts_minutes"] = road["tts_minutes"] / road["total_demand"]

    rail = weighted[weighted["mode"] == "Rail"].copy()
    rail["tts_minutes"] = rail["tts_minutes"] / rail_total_demand

    return pd.concat(
        [
            rail[["mode", "development", "scenario", "tts_minutes"]],
            road[["mode", "development", "scenario", "tts_minutes"]],
        ],
        ignore_index=True,
    )


def _combined_order(data: pd.DataFrame) -> list[str]:
    rail_order = (
        data[data["mode"] == "Rail"]
        .groupby("development")["tts_minutes"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    road_order = (
        data[data["mode"] == "Road"]
        .groupby("development")["tts_minutes"]
        .mean()
        .sort_values(ascending=False)
        .index.tolist()
    )
    return [f"Rail {dev}" for dev in rail_order] + [f"Road {dev}" for dev in road_order]


def _plot_combined_box(ax, data: pd.DataFrame, title: str, ylabel: str) -> None:
    plot_df = data.copy()
    plot_df["label"] = plot_df["mode"] + " " + plot_df["development"].astype(str)
    order = _combined_order(plot_df)
    palette = ["#4C78A8"] * 5 + ["#54A24B"] * 5

    sns.boxplot(
        data=plot_df,
        x="label",
        y="tts_minutes",
        order=order,
        palette=palette,
        linewidth=0.8,
        showfliers=False,
        ax=ax,
    )
    sns.stripplot(
        data=plot_df,
        x="label",
        y="tts_minutes",
        order=order,
        color="black",
        alpha=0.5,
        size=3,
        jitter=True,
        ax=ax,
    )

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Development ID", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis="x", rotation=90, length=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axvline(x=4.5, color="gray", linestyle="--", alpha=0.7)
    upper = ax.get_ylim()[1]
    ax.text(2, upper * 0.97, "Rail top 5", ha="center", va="top", fontsize=11)
    ax.text(7, upper * 0.97, "Road top 5", ha="center", va="top", fontsize=11)


def plot_final_tts_boxplot_top5() -> Path:
    """
    Final cross-mode TTS boxplot:
    - Rail: affected OD savings from analyze_travel_times
    - Road: raster-based affected-cell savings (same object as the access/egress cost comparison)
    """
    data = load_od_level_tts_top5().copy()
    output_path = _output_dir() / "combined_top5_final_tts_boxplot.png"

    data["label"] = data["mode"] + " " + data["development"].astype(str)
    order = _combined_order(data)
    display_labels = get_top5_labels_from_final_costs()
    palette = {label: "#fff3b0" if label.startswith("Rail") else "#e09f3e" for label in order}

    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)

    sns.boxplot(
        data=data,
        x="label",
        y="tts_minutes",
        order=order,
        palette=palette,
        linewidth=0.9,
        showfliers=False,
        width=0.58,
        medianprops={"color": "black", "linewidth": 2.0},
        ax=ax,
    )

    
    ax.axvline(x=4.5, color="black", linestyle="-", alpha=0.7, linewidth=0.5)
    ax.axhline(y=ax.get_ylim()[0],  color="0.7", linestyle="--", alpha=0.7)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([display_labels.get(label, label.split(" ", 1)[1]) for label in order], rotation=90)
    ax.set_title("Distribution of travel time savings for top development alternatives", fontsize=14, pad=20)
    ax.set_xlabel("Development ID", fontsize=10, labelpad=20)
    ax.set_ylabel("Travel time savings [minutes]", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    ax.text(2, ax.get_ylim()[1]+0.1, "Rail top 5", ha="center", va="top", fontsize=11)
    ax.text(7, ax.get_ylim()[1]+0.1, "Road top 5", ha="center", va="top", fontsize=11)

    handles = [
        mpatches.Patch(color="#fff3b0", label="Rail affected OD relations"),
        mpatches.Patch(color="#e09f3e", label="Road raster-cell savings"),
    ]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=False, fontsize=8)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_path, dpi=600)
    plt.close(fig)
    return output_path


def plot_od_level_tts_boxplot_top5() -> Path:
    data = load_od_level_tts_top5()
    output_path = _output_dir() / "top5_tts_od_level_boxplot.png"

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    _plot_combined_box(
        ax,
        data,
        title="Rail and Road top 5 developments",
        ylabel="Travel time savings [minutes]",
    )
    fig.suptitle("Rail OD savings and Road affected-cell savings", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_weighted_tts_violin_top5() -> Path:
    data = load_weighted_tts_top5()
    output_path = _output_dir() / "top5_tts_weighted_violin.png"

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    _plot_combined_box(
        ax,
        data,
        title="Rail and Road top 5 developments",
        ylabel="TTS used for monetization [minutes]",
    )
    fig.suptitle("Travel time savings basis for monetization", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_deweighted_tts_violin_top5() -> Path:
    data = load_deweighted_tts_top5_proxy()
    output_path = _output_dir() / "top5_tts_deweighted_violin.png"

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=300)
    _plot_combined_box(
        ax,
        data,
        title="Rail and Road top 5 developments",
        ylabel="Approx. per-trip TTS [minutes]",
    )
    fig.suptitle("Travel time savings deweighted by total demand (proxy)", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    weighted = plot_weighted_tts_violin_top5()
    deweighted = plot_deweighted_tts_violin_top5()
    od_level = plot_od_level_tts_boxplot_top5()
    final_box = plot_final_tts_boxplot_top5()
    print(f"Saved weighted violin plot to: {weighted}")
    print(f"Saved deweighted violin plot to: {deweighted}")
    print(f"Saved OD-level violin plot to: {od_level}")
    print(f"Saved final TTS boxplot to: {final_box}")


if __name__ == "__main__":
    main()
