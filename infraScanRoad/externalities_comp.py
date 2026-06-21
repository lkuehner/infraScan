from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from shapely.validation import explain_validity

try:
    from . import settings
except ImportError:  # pragma: no cover
    import settings


DATA_ROOT = Path(settings.MAIN)
ROAD_ROOT = DATA_ROOT / "data" / "infraScanRoad"
LINK_FLOW_DIR = ROAD_ROOT / "traffic_flow" / "od" / "link_flows"
RAW_OUTPUT_DIR = ROAD_ROOT / "traffic_flow" / "link_flow_externalities"
BASE_EDGE_GPKG = ROAD_ROOT / "Network" / "processed" / "edges_only_flow.gpkg"
BASE_TUNNEL_GPKG = ROAD_ROOT / "Network" / "processed" / "edges_tunnels.gpkg"
NEW_LINK_GPKG = ROAD_ROOT / "Network" / "processed" / "new_links_realistic_tunnel_adjusted.gpkg"
LANDCOVER_SHP = (
    DATA_ROOT
    / "data"
    / "landuse_landcover"
    / "landcover"
    / "Landcover"
    / "swissTLMRegio_LandCover.shp"
)
NOISE_BUFFER_METERS = 50.0
DEV_RE = re.compile(r"^dev(\d+)_(scenario_\d+)\.csv$")


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


def geometry_diagnostic_rows(gdf: gpd.GeoDataFrame, dataset_name: str, id_column: str | None) -> list[dict]:
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


def compute_settlement_buffer_share(geometry, settlement_footprint, buffer_m: float = NOISE_BUFFER_METERS) -> float:
    if geometry is None or geometry.is_empty or not geometry.is_valid or settlement_footprint is None:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            buffered = geometry.buffer(buffer_m)
    except (RuntimeWarning, ValueError):
        return 0.0
    if buffered.is_empty or not buffered.is_valid or buffered.area == 0:
        return 0.0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            intersection_area = buffered.intersection(settlement_footprint).area
    except (RuntimeWarning, ValueError):
        return 0.0
    return max(0.0, min(1.0, intersection_area / buffered.area))


def build_link_flow_externalities() -> None:
    RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    selected_scenarios = tuple(settings.get_travel_time_debug_scenarios() or [])

    # Load the network geometry used to translate link-flow changes into exposure shares.
    base_edges = gpd.read_file(BASE_EDGE_GPKG)
    tunnels = gpd.read_file(BASE_TUNNEL_GPKG)
    tunnel_length_map = tunnels.groupby("link_id")["total_tunnel_length"].sum().to_dict()
    base_edges["tunnel_length_m"] = base_edges["ID_edge"].map(tunnel_length_map).fillna(0.0)
    base_edges = base_edges[["ID_edge", "tunnel_length_m", "geometry"]].copy()

    new_links = gpd.read_file(NEW_LINK_GPKG)
    new_links["total_tunnel_length"] = new_links["total_tunnel_length"].fillna(0.0)
    new_links = (
        new_links.dissolve(
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
    new_links = new_links[["ID_new", "ID_current", "tunnel_length_m", "geometry"]].copy()

    # Build the settlement footprint once so every link can be scored against the same buffer logic.
    if LANDCOVER_SHP.exists():
        settlement = gpd.read_file(LANDCOVER_SHP)
        objval = settlement["OBJVAL"].fillna("").astype(str).str.lower()
        settlement = settlement[
            objval.str.contains("siedl", regex=False) | objval.str.contains("stadtzentr", regex=False)
        ].copy()
        settlement = settlement[~settlement.geometry.is_empty & settlement.geometry.notna()].copy()
        settlement_valid = settlement[settlement.geometry.is_valid].copy()
        settlement_footprint = unary_union(settlement_valid.geometry) if not settlement_valid.empty else None
    else:
        settlement = gpd.GeoDataFrame(columns=["OBJVAL", "geometry"])
        settlement_footprint = None

    diagnostics = []
    diagnostics.extend(geometry_diagnostic_rows(base_edges, "base_edges", "ID_edge"))
    diagnostics.extend(geometry_diagnostic_rows(new_links, "new_links", "ID_new"))
    diagnostics.extend(geometry_diagnostic_rows(settlement, "settlement_landcover", None))

    exposure_cache: dict[tuple[str, int], dict] = {}
    for row in base_edges.itertuples(index=False):
        geometry = row.geometry
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            continue
        link_length_m = float(geometry.length)
        tunnel_length_m = float(row.tunnel_length_m or 0.0)
        surface_length_m = max(link_length_m - tunnel_length_m, 0.0)
        surface_share = 0.0 if link_length_m == 0 else surface_length_m / link_length_m
        settlement_buffer_share = compute_settlement_buffer_share(geometry, settlement_footprint)
        exposure_cache[("existing", int(row.ID_edge))] = {
            "geometry": geometry,
            "link_length_m": link_length_m,
            "tunnel_length_m": tunnel_length_m,
            "surface_length_m": surface_length_m,
            "surface_share": surface_share,
            "settlement_buffer_share": settlement_buffer_share,
            "noise_relevant_share": surface_share * settlement_buffer_share,
        }

    for row in new_links.itertuples(index=False):
        geometry = row.geometry
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            continue
        link_length_m = float(geometry.length)
        tunnel_length_m = float(row.tunnel_length_m or 0.0)
        surface_length_m = max(link_length_m - tunnel_length_m, 0.0)
        surface_share = 0.0 if link_length_m == 0 else surface_length_m / link_length_m
        settlement_buffer_share = compute_settlement_buffer_share(geometry, settlement_footprint)
        exposure_cache[("new", int(row.ID_new))] = {
            "geometry": geometry,
            "link_length_m": link_length_m,
            "tunnel_length_m": tunnel_length_m,
            "surface_length_m": surface_length_m,
            "surface_share": surface_share,
            "settlement_buffer_share": settlement_buffer_share,
            "noise_relevant_share": surface_share * settlement_buffer_share,
        }

    base_geom_map = base_edges.set_index("ID_edge")["geometry"].to_dict()
    new_geom_map = new_links.set_index("ID_new")["geometry"].to_dict()
    rows: list[dict] = []

    # Compare each development flow file against the same scenario-specific status quo flows.
    for scenario in selected_scenarios:
        status_path = LINK_FLOW_DIR / f"status_quo_{scenario}.csv"
        status_rows = read_flow_rows(status_path)
        if not status_rows:
            continue
        status_map = {row["ID_edge"]: row for row in status_rows}
        status_edge_max = max(status_map)

        for dev_path in sorted(LINK_FLOW_DIR.glob(f"dev*_{scenario}.csv")):
            dev_match = DEV_RE.match(dev_path.name)
            if dev_match is None:
                continue
            development_id = int(dev_match.group(1))
            dev_map = {row["ID_edge"]: row for row in read_flow_rows(dev_path)}

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
                    exposure = exposure_cache.get(("new", development_id))
                else:
                    geometry = base_geom_map.get(edge_id)
                    exposure = exposure_cache.get(("existing", edge_id))

                if geometry is None or exposure is None:
                    continue

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
                        "geometry": geometry,
                    }
                )

    detail_gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=base_edges.crs)
    detail_gdf = detail_gdf.sort_values(
        by=["scenario", "development", "ID_edge"],
        key=lambda col: col.map(lambda value: int(str(value).split("_")[-1])) if col.name == "scenario" else col,
    ).reset_index(drop=True)
    geometry_diagnostics_df = pd.DataFrame(diagnostics)

    # Write the raw link-level table for later monetization inside infraScanIntegrated.
    detail_csv_path = RAW_OUTPUT_DIR / "link_flow_externalities_long.csv"
    detail_gpkg_path = RAW_OUTPUT_DIR / "link_flow_externalities_long.gpkg"
    diagnostics_path = RAW_OUTPUT_DIR / "link_geometry_diagnostics.csv"
    detail_gdf.drop(columns="geometry").to_csv(detail_csv_path, index=False)
    detail_gdf.to_file(detail_gpkg_path, driver="GPKG")
    geometry_diagnostics_df.to_csv(diagnostics_path, index=False)

    print(f"Wrote {detail_csv_path}")
    print(f"Wrote {detail_gpkg_path}")
    print(f"Wrote {diagnostics_path}")


if __name__ == "__main__":
    build_link_flow_externalities()
