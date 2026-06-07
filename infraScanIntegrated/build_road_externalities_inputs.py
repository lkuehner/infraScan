from __future__ import annotations

import csv
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.validation import explain_validity

from .scoring_registry import (
    NOISE_BUFFER_METERS,
    compute_settlement_buffer_share,
    load_settlement_footprint,
    monetize_road_externalities_detail,
)


DATA_ROOT = Path("/Volumes/WD_Windows/MSc_Thesis")
ROAD_RUN_ROOT = DATA_ROOT / "euler" / "infraScanRoad_trust_2iter_alldev_10sce"
# PROCESSED_RASTER_ROOT = DATA_ROOT / "data" / "independent_variable" / "processed"
OUTPUT_DIR = ROAD_RUN_ROOT / "traffic_flow" / "road_externalities_inputs"

LINK_FLOW_DIR = ROAD_RUN_ROOT / "traffic_flow" / "od" / "link_flows"
BASE_EDGE_GPKG = ROAD_RUN_ROOT / "Network" / "processed" / "edges_only_flow.gpkg"
BASE_TUNNEL_GPKG = ROAD_RUN_ROOT / "Network" / "processed" / "edges_tunnels.gpkg"
NEW_LINK_GPKG = ROAD_RUN_ROOT / "Network" / "processed" / "new_links_realistic_tunnel_adjusted.gpkg"
SELECTED_SCENARIOS = (
    "scenario_19",
    "scenario_26",
    "scenario_44",
    "scenario_64",
    "scenario_70",
    "scenario_75",
    "scenario_78",
    "scenario_89",
    "scenario_96",
    "scenario_100",
)

ANNUALIZATION_FACTOR = 2.5 * 250

DEV_RE = re.compile(r"^dev(\d+)_(scenario_\d+)\.csv$")


def scenario_key(scenario_name: str) -> int:
    return int(scenario_name.split("_")[-1])


def read_flow_rows(flow_path: Path) -> list[dict]:
    rows: list[dict] = []
    with flow_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        "ID_edge": int(row["ID_edge"]),
                        "flow": float(row["flow"]),
                        "length_m": float(row["length_m"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def load_base_edges() -> gpd.GeoDataFrame:
    edges = gpd.read_file(BASE_EDGE_GPKG)
    tunnels = gpd.read_file(BASE_TUNNEL_GPKG)
    tunnel_length_map = tunnels.groupby("link_id")["total_tunnel_length"].sum().to_dict()
    edges["tunnel_length_m"] = edges["ID_edge"].map(tunnel_length_map).fillna(0.0)
    return edges[["ID_edge", "flow", "tunnel_length_m", "geometry"]].copy()


def load_new_links() -> gpd.GeoDataFrame:
    links = gpd.read_file(NEW_LINK_GPKG)
    links["total_tunnel_length"] = links["total_tunnel_length"].fillna(0.0)
    aggregated = (
        links.dissolve(
            by="ID_new",
            aggfunc={
                "ID_current": "first",
                "check_needed": "max",
                "total_tunnel_length": "sum",
                "total_bridge_length": "sum",
            },
        )
        .reset_index()
        .rename(columns={"total_tunnel_length": "tunnel_length_m"})
    )
    return aggregated[["ID_new", "ID_current", "tunnel_length_m", "geometry"]].copy()


def geometry_diagnostic_rows(
    gdf: gpd.GeoDataFrame,
    dataset_name: str,
    id_column: str | None,
) -> list[dict]:
    rows: list[dict] = []
    for row in gdf.itertuples(index=False):
        geometry = row.geometry
        identifier = getattr(row, id_column) if id_column is not None and hasattr(row, id_column) else None
        is_missing = geometry is None
        is_empty = False if is_missing else geometry.is_empty
        is_valid = False if is_missing else geometry.is_valid
        rows.append(
            {
                "dataset": dataset_name,
                "feature_id": identifier,
                "geom_type": None if is_missing else geometry.geom_type,
                "is_missing": is_missing,
                "is_empty": is_empty,
                "is_valid": is_valid,
                "validity_reason": "missing geometry"
                if is_missing
                else ("empty geometry" if is_empty else explain_validity(geometry)),
            }
        )
    return rows


def build_exposure_cache(
    base_edges: gpd.GeoDataFrame,
    new_links: gpd.GeoDataFrame,
    settlement_footprint,
) -> dict[tuple[str, int], dict]:
    exposure_cache: dict[tuple[str, int], dict] = {}

    def build_entry(row, key: tuple[str, int]) -> None:
        geometry = row.geometry
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            return

        link_length_m = float(geometry.length)
        tunnel_length_m = float(getattr(row, "tunnel_length_m", 0.0) or 0.0)
        surface_length_m = max(link_length_m - tunnel_length_m, 0.0)
        surface_share = 0.0 if link_length_m == 0 else surface_length_m / link_length_m
        settlement_buffer_share = compute_settlement_buffer_share(
            geometry=geometry,
            settlement_footprint=settlement_footprint,
            buffer_m=NOISE_BUFFER_METERS,
        )
        noise_relevant_share = surface_share * settlement_buffer_share
        # Optional diagnostics if raster-based exposure is needed again:
        # pop_50m = sum_raster_in_buffer(POP_RASTER, geometry)
        # empl_50m = sum_raster_in_buffer(EMPL_RASTER, geometry)

        exposure_cache[key] = {
            "geometry": geometry,
            "link_length_m": link_length_m,
            "tunnel_length_m": tunnel_length_m,
            "surface_length_m": surface_length_m,
            "surface_share": surface_share,
            "settlement_buffer_share": settlement_buffer_share,
            "noise_relevant_share": noise_relevant_share,
            # "pop_50m": pop_50m,
            # "empl_50m": empl_50m,
        }

    for row in base_edges.itertuples(index=False):
        build_entry(row, ("existing", int(row.ID_edge)))

    for row in new_links.itertuples(index=False):
        build_entry(row, ("new", int(row.ID_new)))

    return exposure_cache


def build_link_detail_df(
    scenarios: tuple[str, ...] = SELECTED_SCENARIOS,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    base_edges = load_base_edges()
    new_links = load_new_links()
    settlement_footprint, settlement_gdf = load_settlement_footprint()
    diagnostic_rows = []
    diagnostic_rows.extend(geometry_diagnostic_rows(base_edges, "base_edges", "ID_edge"))
    diagnostic_rows.extend(geometry_diagnostic_rows(new_links, "new_links", "ID_new"))
    diagnostic_rows.extend(geometry_diagnostic_rows(settlement_gdf, "settlement_landcover", None))
    geometry_diagnostics_df = pd.DataFrame(diagnostic_rows)
    exposure_cache = build_exposure_cache(
        base_edges=base_edges,
        new_links=new_links,
        settlement_footprint=settlement_footprint,
    )
    base_geom_map = base_edges.set_index("ID_edge")["geometry"].to_dict()
    new_geom_map = new_links.set_index("ID_new")["geometry"].to_dict()

    rows: list[dict] = []

    for scenario in scenarios:
        status_path = LINK_FLOW_DIR / f"status_quo_{scenario}.csv"
        status_rows = read_flow_rows(status_path)
        status_map = {row["ID_edge"]: row for row in status_rows}
        status_edge_max = max(status_map)

        dev_paths = sorted(LINK_FLOW_DIR.glob(f"dev*_{scenario}.csv"))
        for dev_path in dev_paths:
            dev_match = DEV_RE.match(dev_path.name)
            if dev_match is None:
                continue

            development, _ = dev_match.groups()
            development_id = int(development)
            dev_rows = read_flow_rows(dev_path)
            dev_map = {row["ID_edge"]: row for row in dev_rows}

            for edge_id in sorted(set(status_map).union(dev_map)):
                status_row = status_map.get(edge_id, {})
                dev_row = dev_map.get(edge_id, {})
                flow_status = float(status_row.get("flow", 0.0))
                flow_development = float(dev_row.get("flow", 0.0))
                length_m = float(dev_row.get("length_m", status_row.get("length_m", 0.0)))
                delta_flow = flow_development - flow_status
                delta_vkm = delta_flow * (length_m / 1000.0)

                is_new_link = edge_id > status_edge_max
                if is_new_link:
                    geometry = new_geom_map.get(development_id)
                    exposure_key = ("new", development_id)
                else:
                    geometry = base_geom_map.get(edge_id)
                    exposure_key = ("existing", edge_id)

                if geometry is None:
                    continue

                exposure = exposure_cache[exposure_key]

                rows.append(
                    {
                        "development": development_id,
                        "scenario": scenario,
                        "ID_edge": edge_id,
                        "link_role": "new_link" if is_new_link else "existing_network",
                        "flow_status_quo": flow_status,
                        "flow_development": flow_development,
                        "delta_flow": delta_flow,
                        "length_m_from_flow": length_m,
                        "vkm_status_quo": flow_status * (length_m / 1000.0),
                        "vkm_development": flow_development * (length_m / 1000.0),
                        "delta_vkm": delta_vkm,
                        "link_length_m_geometry": exposure["link_length_m"],
                        "tunnel_length_m": exposure["tunnel_length_m"],
                        "surface_length_m": exposure["surface_length_m"],
                        "surface_share": exposure["surface_share"],
                        "settlement_buffer_share": exposure["settlement_buffer_share"],
                        "noise_relevant_share": exposure["noise_relevant_share"],
                        # "pop_50m": exposure["pop_50m"],
                        # "empl_50m": exposure["empl_50m"],
                        "geometry": geometry,
                    }
                )

    detail_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=base_edges.crs)
    detail_gdf = detail_gdf.sort_values(
        by=["scenario", "development", "ID_edge"],
        key=lambda col: col.map(scenario_key) if col.name == "scenario" else col,
    ).reset_index(drop=True)
    return detail_gdf, geometry_diagnostics_df


def aggregate_road_externalities_detail(
    detail_df,
    annualization_factor: float = 2.5 * 250,
):
    detail = monetize_road_externalities_detail(
        detail_df=detail_df,
        annualization_factor=annualization_factor,
    )
    grouped = (
        detail.groupby(["development", "scenario"], as_index=False)
        .agg(
            delta_vkm_peak_hour=("delta_vkm", "sum"),
            delta_vkm_peak_hour_new_link=(
                "delta_vkm",
                lambda s: s[detail.loc[s.index, "link_role"] == "new_link"].sum(),
            ),
            delta_vkm_peak_hour_existing_network=(
                "delta_vkm",
                lambda s: s[detail.loc[s.index, "link_role"] == "existing_network"].sum(),
            ),
            delta_vkm_peak_hour_noise_relevant=(
                "delta_vkm_annualized_noise_relevant",
                lambda s: s.sum() / annualization_factor,
            ),
            road_accident_cost_annual=("road_accident_cost_annual", "sum"),
            road_airpollution_cost_annual=("road_airpollution_cost_annual", "sum"),
            road_co2_cost_annual=("road_co2_cost_annual", "sum"),
            road_noise_cost_annual=("road_noise_cost_annual", "sum"),
            road_land_consumption_cost=("road_land_consumption_cost", "sum"),
            # pop_50m_sum=("pop_50m", "sum"),
            # empl_50m_sum=("empl_50m", "sum"),
            mean_surface_share=("surface_share", "mean"),
            mean_settlement_buffer_share=("settlement_buffer_share", "mean"),
            mean_noise_relevant_share=("noise_relevant_share", "mean"),
        )
    )
    grouped["delta_vkm_annualized"] = grouped["delta_vkm_peak_hour"] * annualization_factor
    grouped["delta_vkm_annualized_noise_relevant"] = (
        grouped["delta_vkm_peak_hour_noise_relevant"] * annualization_factor
    )
    return grouped


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_gdf_raw, geometry_diagnostics_df = build_link_detail_df()
    detail_gdf = monetize_road_externalities_detail(
        detail_df=detail_gdf_raw,
        annualization_factor=ANNUALIZATION_FACTOR,
    )
    monetization_df = aggregate_road_externalities_detail(
        detail_df=detail_gdf,
        annualization_factor=ANNUALIZATION_FACTOR,
    )

    detail_csv_path = OUTPUT_DIR / "road_externalities_link_detail.csv"
    detail_gpkg_path = OUTPUT_DIR / "road_externalities_link_detail.gpkg"
    monetization_csv_path = OUTPUT_DIR / "road_externalities_monetization.csv"
    geometry_diagnostics_path = OUTPUT_DIR / "road_externalities_geometry_diagnostics.csv"

    detail_gdf.drop(columns="geometry").to_csv(detail_csv_path, index=False)
    detail_gdf.to_file(detail_gpkg_path, driver="GPKG")
    monetization_df.to_csv(monetization_csv_path, index=False)
    geometry_diagnostics_df.to_csv(geometry_diagnostics_path, index=False)

    print(f"Wrote {detail_csv_path}")
    print(f"Wrote {detail_gpkg_path}")
    print(f"Wrote {monetization_csv_path}")
    print(f"Wrote {geometry_diagnostics_path}")


if __name__ == "__main__":
    main()
