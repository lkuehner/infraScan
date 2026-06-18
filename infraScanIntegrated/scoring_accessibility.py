"""
TODO:
EXPLAIN IDEA


"""


from __future__ import annotations

import os
import pickle
import sqlite3
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("USE_PYGEOS", "0")

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from rasterio.plot import plotting_extent
from rasterio.warp import Resampling, reproject

from infraScan.infraScanIntegrated import paths as integrated_paths
from infraScan.infraScanRail import settings as rail_settings
from infraScan.infraScanRoad import settings as road_settings


DATA_ROOT = Path(integrated_paths.MAIN)
INTEGRATED_ANALYSIS_DIR = integrated_paths.INTEGRATED_COSTS_DIR / "score_analysis"
OUTPUT_ROOT = integrated_paths.INTEGRATED_PLOTS_DIR / "accessibility_maps"

ROAD_ROOT = integrated_paths.ROAD_DATA_ROOT
ROAD_SQ_TT = ROAD_ROOT / "traffic_flow" / "od" / "status_quo_od_tt.csv"
ROAD_DEV_TT = ROAD_ROOT / "traffic_flow" / "od" / "developments_od_tt.csv"
ROAD_TRAVELTIME_DIR = ROAD_ROOT / "Network" / "travel_time"
ROAD_DEV_RASTER_DIR = ROAD_TRAVELTIME_DIR / "developments"
ROAD_LINKS_GPKG = integrated_paths.ROAD_NEW_LINKS_GPKG
ROAD_NETWORK_GPKG = integrated_paths.ROAD_HIGHWAY_NETWORK_GPKG

RAIL_ROOT = integrated_paths.RAIL_NETWORK_PATH.parent
RAIL_POINTS_GPKG = RAIL_ROOT / "Network" / "processed" / "points.gpkg"
RAIL_SCENARIO_CACHE_DIR = DATA_ROOT / "data" / "Scenario" / "cache" / "rail"
RAIL_OD_TIMES_CACHE = RAIL_ROOT / "Network" / "travel_time" / "cache" / "od_times.pkl"
RAIL_DEV_DIR = RAIL_ROOT / "Network" / "processed" / "developments"
RAIL_LINKS_GPKG = RAIL_ROOT / "Network" / "processed" / "updated_new_links.gpkg"
RAIL_ACTIVE_SERVICE_NETWORK_GPKG = integrated_paths.RAIL_ACTIVE_SERVICE_NETWORK_GPKG
RAIL_COMMUNE_TO_STATION = RAIL_ROOT / "Network" / "processed" / "Communes_to_railway_stations_ZH.xlsx"
RAIL_STATION_CATCHMENT_SUMMARY = Path(
    "/Volumes/WD_Windows/draft/data/infraScanRail/Catchement_area/Municipal/station_catchment_summary.csv"
)
COMMUNE_SHP = integrated_paths.SWISS_MUNICIPALITY_BOUNDARIES_PATH
LAKES_SHP = integrated_paths.SWISS_LAKES_PATH

POPULATION_RASTER = DATA_ROOT / "data" / "Spatial_Data" / "Land_Use" / "Population" / "population_2023.tif"
EMPLOYMENT_RASTER = DATA_ROOT / "data" / "Spatial_Data" / "Land_Use" / "Employment" / "employment_2023.tif"

ROAD_BETA_PER_HOUR = 3.0
DEVELOPMENT_BLUE = "#0E5FAB"
LAKE_FILL = "#97C6E1"
NETWORK_COLOR = "#303030"
BOUNDARY_COLOR = "#303030"
ZERO_COLOR = "#ECECEC"
ACTIVE_LINE_COLOR = "#5E88B8"
STATION_COLOR = "#2B2B2B"
ABSOLUTE_ACCESS_CMAP = LinearSegmentedColormap.from_list(
    "absolute_accessibility",
    ["#F3F7F2", "#C4D8C0", "#91B58D", "#6E946A", "#4E704D"],
)
DELTA_ACCESS_CMAP = LinearSegmentedColormap.from_list(
    "delta_accessibility",
    [
        (0.00, "#8F2F2F"),
        (0.18, "#A53D3D"),
        (0.34, "#B64B4B"),
        (0.45, "#D07A7A"),
        (0.50, "#FFFFFF"),
        (0.60, "#DCE8D8"),
        (0.76, "#91B58D"),
        (0.88, "#6E946A"),
        (1.00, "#4E704D"),
    ],
)
IMPORTANT_CANTON_NR = 1
ROAD_ASSESSMENT_AREA_CSV = Path("/Volumes/WD_Windows/draft/data/infraScanRoad/Catchement_Area/road_assessment_area.csv")
RAIL_NAME_NORMALIZATION = {
    "Niederglatt": "Niederglatt ZH",
    "Oberglatt": "Oberglatt ZH",
}
RAIL_CATCHMENT_NAME_OVERRIDES = {
    "Glanzenberg": "Glanzenbach",
    "Kilchberg ZH": "Kilchberg",
    "Niederglatt ZH": "Niederglatt",
    "Oberglatt ZH": "Oberglatt",
    "Otelfingen Golfpark": "Otelfingen Rietholz",
    "Pfungen": "Pfungen-Neftenbach",
    "Winterthur Wülflingen": "Einterthur Wülflingen",
    "Zürich Friesenberg": "Friesenberg",
    "Zürich Schweighof": "Schweighof",
    "Zürich Triemli": "Triemli",
}


# Sort scenario labels by their trailing scenario number.
def scenario_sort_key(name: str):
    try:
        return int(str(name).split("_")[-1])
    except Exception:
        return str(name)


# Read the integrated top-10 ranking that decides which absolute maps are exported.
def load_top10_by_mode() -> tuple[list[str], list[str]]:
    df = pd.read_csv(INTEGRATED_ANALYSIS_DIR / "integrated_bcr_top10_by_mode_plot_data.csv")
    df["development"] = df["development"].astype(str)
    df = df[["mode", "development", "plot_order"]].drop_duplicates().sort_values("plot_order")
    rail_top10 = df.loc[df["mode"] == "Rail", "development"].tolist()
    road_top10 = df.loc[df["mode"] == "Road", "development"].tolist()
    return rail_top10, road_top10


def clip_gdf_to_bounds(
    gdf: gpd.GeoDataFrame | None,
    bounds: tuple[float, float, float, float],
    pad: float = 0.0,
) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty:
        return gdf
    xmin, ymin, xmax, ymax = bounds
    return gdf.cx[(xmin - pad):(xmax + pad), (ymin - pad):(ymax + pad)].copy()


def add_reference_layers(
    ax,
    bounds: tuple[float, float, float, float],
    *,
    lakes: gpd.GeoDataFrame | None = None,
    network: gpd.GeoDataFrame | None = None,
    commune_boundaries: gpd.GeoDataFrame | None = None,
    active_line: gpd.GeoDataFrame | None = None,
    stations: gpd.GeoDataFrame | None = None,
) -> None:
    lakes_clip = clip_gdf_to_bounds(lakes, bounds, pad=2_000.0)
    if lakes_clip is not None and not lakes_clip.empty:
        lakes_clip.plot(ax=ax, color=LAKE_FILL, linewidth=0.35, zorder=3)

    network_clip = clip_gdf_to_bounds(network, bounds, pad=2_000.0)
    if network_clip is not None and not network_clip.empty:
        if "Link NR" in network_clip.columns:
            dashed = network_clip[pd.to_numeric(network_clip["Link NR"], errors="coerce") == 178].copy()
            base = network_clip[pd.to_numeric(network_clip["Link NR"], errors="coerce") != 178].copy()
            if not base.empty:
                base.plot(ax=ax, color="black", linewidth=0.9, alpha=0.9, zorder=4)
            if not dashed.empty:
                dashed.plot(ax=ax, color="black", linewidth=1.2, alpha=0.95, linestyle="--", zorder=5)
        else:
            network_clip.plot(ax=ax, color=NETWORK_COLOR, linewidth=0.55, alpha=0.75, zorder=4)

    commune_clip = clip_gdf_to_bounds(commune_boundaries, bounds, pad=2_000.0)
    if commune_clip is not None and not commune_clip.empty:
        commune_clip.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=0.18, alpha=0.55, zorder=5)

    active_line_clip = clip_gdf_to_bounds(active_line, bounds, pad=2_000.0)
    if active_line_clip is not None and not active_line_clip.empty:
        active_line_clip.plot(ax=ax, color=ACTIVE_LINE_COLOR, linewidth=1.5, alpha=0.9, zorder=6)

    stations_clip = clip_gdf_to_bounds(stations, bounds, pad=2_000.0)
    if stations_clip is not None and not stations_clip.empty:
        stations_clip.plot(
            ax=ax,
            color=STATION_COLOR,
            linewidth=0.4,
            markersize=22,
            zorder=7,
        )


# Render a raster-based accessibility map and optionally overlay the development geometry.
def save_raster_plot(
    array: np.ndarray,
    output_path: Path,
    title: str,
    delta: bool,
    transform=None,
    overlay_gdf: gpd.GeoDataFrame | None = None,
    lakes: gpd.GeoDataFrame | None = None,
    network: gpd.GeoDataFrame | None = None,
    commune_boundaries: gpd.GeoDataFrame | None = None,
    active_line: gpd.GeoDataFrame | None = None,
    stations: gpd.GeoDataFrame | None = None,
    bounds_override: tuple[float, float, float, float] | None = None,
    value_range: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), dpi=220)
    masked = np.ma.masked_invalid(array)
    image_kwargs = {}
    extent = None
    if transform is not None:
        extent = plotting_extent(array, transform)
        image_kwargs["extent"] = extent

    if delta:
        if value_range is not None:
            vmin, vmax = value_range
        else:
            vmax = float(np.nanmax(np.abs(masked))) if masked.count() else 1.0
            vmax = max(vmax, 1e-9)
            vmin = -vmax
        image = ax.imshow(
            masked,
            cmap=DELTA_ACCESS_CMAP,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
            **image_kwargs,
        )
    else:
        if value_range is not None:
            vmin, vmax = value_range
        else:
            vmin = float(np.nanmin(masked)) if masked.count() else 0.0
            vmax = float(np.nanmax(masked)) if masked.count() else 1.0
        if not np.isfinite(vmin):
            vmin = 0.0
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-9
        image = ax.imshow(masked, cmap=ABSOLUTE_ACCESS_CMAP, vmin=vmin, vmax=vmax, **image_kwargs)

    reference_bounds = bounds_override
    if reference_bounds is None and extent is not None:
        xmin, xmax, ymin, ymax = extent
        reference_bounds = (xmin, ymin, xmax, ymax)

    if reference_bounds is not None:
        add_reference_layers(
            ax,
            reference_bounds,
            lakes=lakes,
            network=network,
            commune_boundaries=commune_boundaries,
            active_line=active_line,
            stations=stations,
        )

    if overlay_gdf is not None and not overlay_gdf.empty:
        overlay_gdf.plot(ax=ax, color=DEVELOPMENT_BLUE, linewidth=2.2, zorder=8)

    if bounds_override is not None:
        xmin, ymin, xmax, ymax = bounds_override
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

    ax.set_title(title)
    ax.set_axis_off()
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# Reproject the exogenous population or employment raster onto the road analysis grid.
def reproject_raster_to_grid(input_path: Path, target_shape: tuple[int, int], target_transform, target_crs) -> np.ndarray:
    with rasterio.open(input_path) as src:
        source_array = src.read(1).astype(float)
        source_array = np.nan_to_num(source_array, nan=0.0, posinf=0.0, neginf=0.0)
        destination = np.zeros(target_shape, dtype=float)
        reproject(
            source=source_array,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            dst_nodata=0.0,
        )
    return destination


# Build the road-grid exogenous potential rasters used for catchment aggregation.
def load_road_potential_rasters(target_shape: tuple[int, int], target_transform, target_crs) -> tuple[np.ndarray, np.ndarray]:
    pop = reproject_raster_to_grid(POPULATION_RASTER, target_shape, target_transform, target_crs)
    empl = reproject_raster_to_grid(EMPLOYMENT_RASTER, target_shape, target_transform, target_crs)
    return pop, empl


# Map catchment-level values back onto the source-id raster for plotting.
def values_to_source_raster(source_arr: np.ndarray, values: pd.DataFrame, key_col: str, value_col: str) -> np.ndarray:
    out = np.full(source_arr.shape, np.nan, dtype=float)
    if values.empty:
        return out

    mapping = values[[key_col, value_col]].dropna().drop_duplicates(subset=[key_col]).copy()
    mapping[key_col] = pd.to_numeric(mapping[key_col], errors="coerce").astype("Int64")
    value_map = mapping.dropna(subset=[key_col]).set_index(key_col)[value_col].to_dict()

    valid = source_arr > 0
    out[valid] = pd.Series(source_arr[valid].astype(int)).map(value_map).to_numpy()
    return out


# Read and clean the road OD tables used for accessibility averaging.
def load_road_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sq_tt = pd.read_csv(ROAD_SQ_TT)
    dev_tt = pd.read_csv(ROAD_DEV_TT)

    for df in (sq_tt, dev_tt):
        for col in ["origin", "destination", "demand", "travel_time"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["scenario"] = df["scenario"].astype(str)
    dev_tt["development"] = pd.to_numeric(dev_tt["development"], errors="coerce")

    sq_tt = sq_tt.dropna(subset=["origin", "destination", "travel_time"]).copy()
    dev_tt = dev_tt.dropna(subset=["development", "origin", "destination", "travel_time"]).copy()
    sq_tt[["origin", "destination"]] = sq_tt[["origin", "destination"]].astype(int)
    dev_tt[["development", "origin", "destination"]] = dev_tt[["development", "origin", "destination"]].astype(int)

    scenarios = sorted(set(sq_tt["scenario"]).intersection(dev_tt["scenario"]), key=scenario_sort_key)
    if getattr(road_settings, "generated_select_representative_scenarios", False):
        active_scenarios = road_settings.get_representative_generated_scenarios()
        scenarios = [scenario for scenario in active_scenarios if scenario in scenarios]
    return sq_tt, dev_tt, scenarios


# Aggregate road raster cells to catchments and compute exogenous P_j per catchment.
def road_catchment_stats(
    source_arr: np.ndarray,
    access_hr_arr: np.ndarray,
    pop_arr: np.ndarray,
    empl_arr: np.ndarray,
) -> pd.DataFrame:
    valid = (
        (source_arr > 0)
        & np.isfinite(source_arr)
        & np.isfinite(access_hr_arr)
        & np.isfinite(pop_arr)
        & np.isfinite(empl_arr)
    )
    frame = pd.DataFrame(
        {
            "catchment": source_arr[valid].astype(int),
            "access_hr": access_hr_arr[valid],
            "pop": np.clip(pop_arr[valid], a_min=0.0, a_max=None),
            "empl": np.clip(empl_arr[valid], a_min=0.0, a_max=None),
        }
    )
    grouped = frame.groupby("catchment", as_index=False).agg(
        access_hr=("access_hr", "mean"),
        pop=("pop", "sum"),
        empl=("empl", "sum"),
    )
    grouped["Pj"] = grouped["pop"] + 0.5 * grouped["empl"]
    return grouped


# Add access-egress components and exogenous destination potential to road OD rows.
def road_add_generalized_tt(
    tt_df: pd.DataFrame,
    origin_stats: pd.DataFrame,
    dest_access_stats: pd.DataFrame,
    dest_pj_stats: pd.DataFrame,
) -> pd.DataFrame:
    origin_map = origin_stats[["catchment", "access_hr"]].rename(
        columns={"catchment": "origin", "access_hr": "origin_access_hr"}
    )
    dest_access_map = dest_access_stats[["catchment", "access_hr"]].rename(
        columns={"catchment": "destination", "access_hr": "dest_access_hr"}
    )
    dest_pj_map = dest_pj_stats[["catchment", "Pj"]].rename(columns={"catchment": "destination"})
    work = tt_df.merge(origin_map, on="origin", how="left")
    work = work.merge(dest_access_map, on="destination", how="left")
    work = work.merge(dest_pj_map, on="destination", how="left")
    work["origin_access_hr"] = work["origin_access_hr"].fillna(0.0)
    work["dest_access_hr"] = work["dest_access_hr"].fillna(0.0)
    work["Pj"] = work["Pj"].fillna(0.0)
    work["gen_tt_hr"] = work["origin_access_hr"] + work["travel_time"] + work["dest_access_hr"]
    return work


# Compute road accessibility per origin using exogenous destination potential.
def road_accessibility_by_origin(
    tt_df: pd.DataFrame,
    origin_stats: pd.DataFrame,
    dest_access_stats: pd.DataFrame,
    dest_pj_stats: pd.DataFrame,
) -> pd.DataFrame:
    work = road_add_generalized_tt(tt_df, origin_stats, dest_access_stats, dest_pj_stats)
    work["accessibility"] = np.exp(-ROAD_BETA_PER_HOUR * work["gen_tt_hr"]) * work["Pj"]
    return work.groupby("origin", as_index=False)["accessibility"].sum()


# Load the road development geometries that are drawn on the maps.
def load_road_links() -> gpd.GeoDataFrame:
    links = gpd.read_file(ROAD_LINKS_GPKG)
    links["ID_new"] = pd.to_numeric(links["ID_new"], errors="coerce")
    links = links.dropna(subset=["ID_new"]).copy()
    links["ID_new"] = links["ID_new"].astype(int)
    return links[["ID_new", "geometry"]]


def load_road_network() -> gpd.GeoDataFrame:
    network = gpd.read_file(ROAD_NETWORK_GPKG).to_crs("EPSG:2056")
    edge_col = "Link NR" if "Link NR" in network.columns else None
    keep_cols = ["geometry"] + ([edge_col] if edge_col else [])
    return network[keep_cols]


def load_lakes() -> gpd.GeoDataFrame:
    return gpd.read_file(LAKES_SHP).to_crs("EPSG:2056")[["geometry"]]


def load_commune_boundaries() -> gpd.GeoDataFrame:
    communes = gpd.read_file(COMMUNE_SHP).to_crs("EPSG:2056")
    bfs_col = "bfs_nummer" if "bfs_nummer" in communes.columns else "BFS"
    name_col = "name" if "name" in communes.columns else "GEMEINDENA"
    communes["BFS"] = pd.to_numeric(communes[bfs_col], errors="coerce")
    communes = communes.dropna(subset=["BFS"]).copy()
    communes["BFS"] = communes["BFS"].astype(int)
    communes["name"] = communes[name_col].astype(str)
    communes = communes[communes["BFS"] > 0].copy()
    if "kantonsnummer" in communes.columns:
        communes["kantonsnummer"] = pd.to_numeric(communes["kantonsnummer"], errors="coerce")
    return communes[[col for col in ["BFS", "name", "kantonsnummer", "geometry"] if col in communes.columns]]


def load_important_communes() -> gpd.GeoDataFrame:
    communes = load_commune_boundaries()
    if "kantonsnummer" in communes.columns:
        communes = communes[communes["kantonsnummer"] == IMPORTANT_CANTON_NR].copy()
    return communes


def load_canton_boundary() -> gpd.GeoDataFrame:
    communes = load_important_communes()
    dissolved = communes.dissolve()
    return gpd.GeoDataFrame(dissolved[["geometry"]], geometry="geometry", crs=communes.crs).reset_index(drop=True)


def load_road_plot_bounds() -> tuple[float, float, float, float]:
    area = pd.read_csv(ROAD_ASSESSMENT_AREA_CSV, sep=";")
    geometry = gpd.GeoSeries.from_wkt(area["wtk"], crs="EPSG:2056")
    bounds = geometry.total_bounds
    return tuple(float(v) for v in bounds)


def load_rail_network() -> gpd.GeoDataFrame:
    network_path = Path(rail_settings.infra_generation_rail_network)
    if not network_path.is_absolute():
        network_path = DATA_ROOT / network_path
    network = gpd.read_file(network_path).to_crs("EPSG:2056")
    return network[["Service", "geometry"]] if "Service" in network.columns else network[["geometry"]]


def load_rail_active_service_network() -> gpd.GeoDataFrame:
    network = gpd.read_file(RAIL_ACTIVE_SERVICE_NETWORK_GPKG).to_crs("EPSG:2056")
    network["dev_id"] = pd.to_numeric(network["dev_id"], errors="coerce")
    network["Service"] = network["Service"].astype(str)
    network["Sline"] = network["Sline"].astype(str)
    return network[["Service", "Sline", "dev_id", "new_dev", "geometry"]]


def load_rail_stations() -> gpd.GeoDataFrame:
    stations = gpd.read_file(RAIL_POINTS_GPKG).to_crs("EPSG:2056")
    stations["HST"] = pd.to_numeric(stations["HST"], errors="coerce")
    stations = stations[stations["HST"] == 1].copy()
    return stations[["NAME", "geometry"]]


def filter_stations_to_displayed_network(
    stations: gpd.GeoDataFrame,
    network: gpd.GeoDataFrame,
    bounds: tuple[float, float, float, float],
    tolerance_m: float = 250.0,
) -> gpd.GeoDataFrame:
    stations_clip = clip_gdf_to_bounds(stations, bounds, pad=2_000.0)
    network_clip = clip_gdf_to_bounds(network, bounds, pad=2_000.0)
    if stations_clip is None or stations_clip.empty or network_clip is None or network_clip.empty:
        return stations.iloc[0:0].copy()
    buffered_network = network_clip.geometry.buffer(tolerance_m)
    mask = stations_clip.geometry.apply(lambda geom: buffered_network.intersects(geom).any())
    return stations_clip.loc[mask].copy()


def get_rail_active_line(active_network: gpd.GeoDataFrame, development: str) -> gpd.GeoDataFrame:
    dev_num = int(development)
    direct = active_network[active_network["dev_id"] == dev_num].copy()
    if direct.empty:
        return direct
    sline = direct["Sline"].dropna().astype(str).iloc[0]
    service_line = active_network[active_network["Service"].astype(str) == sline].copy()
    return service_line if not service_line.empty else direct


def compute_global_absolute_range(arrays: list[np.ndarray | pd.Series]) -> tuple[float, float]:
    mins = []
    maxs = []
    for arr in arrays:
        values = np.asarray(arr, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        mins.append(float(finite.min()))
        maxs.append(float(finite.max()))
    if not mins:
        return (0.0, 1.0)
    vmin = min(mins)
    vmax = max(maxs)
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return (vmin, vmax)


def compute_global_delta_range(arrays: list[np.ndarray | pd.Series]) -> tuple[float, float]:
    vmax = 0.0
    for arr in arrays:
        values = np.asarray(arr, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            continue
        vmax = max(vmax, float(np.abs(finite).max()))
    vmax = max(vmax, 1e-9)
    return (-vmax, vmax)


# Create all road accessibility maps on the same OD/catchment basis as the TTS comparison.
def create_road_maps() -> None:
    road_output = OUTPUT_ROOT / "road"
    road_output.mkdir(parents=True, exist_ok=True)
    road_plot_bounds = load_road_plot_bounds()

    _, road_top10 = load_top10_by_mode()
    road_top10_set = set(road_top10)
    sq_tt, dev_tt, scenarios = load_road_inputs()
    road_links = load_road_links()
    road_network = load_road_network()
    lakes = load_lakes()

    with rasterio.open(ROAD_TRAVELTIME_DIR / "source_id_raster.tif") as src:
        sq_source = src.read(1)
        road_shape = (src.height, src.width)
        road_transform = src.transform
        road_crs = src.crs
    with rasterio.open(ROAD_TRAVELTIME_DIR / "travel_time_raster.tif") as src:
        sq_access_hr = src.read(1).astype(float) / 3600.0
        sq_transform = src.transform

    pop_arr, empl_arr = load_road_potential_rasters(road_shape, road_transform, road_crs)
    sq_stats = road_catchment_stats(sq_source, sq_access_hr, pop_arr, empl_arr)
    developments = sorted(dev_tt["development"].drop_duplicates().astype(int).tolist())
    sq_sum = np.zeros(sq_source.shape, dtype=float)
    sq_count = 0

    for scenario in scenarios:
        sq_values = road_accessibility_by_origin(
            sq_tt[sq_tt["scenario"] == scenario].copy(),
            sq_stats,
            sq_stats,
            sq_stats,
        )
        sq_sum += np.nan_to_num(
            values_to_source_raster(sq_source, sq_values, "origin", "accessibility"),
            nan=0.0,
        )
        sq_count += 1

    sq_raster_mean = sq_sum / max(sq_count, 1)
    road_dev_results: list[tuple[int, np.ndarray, np.ndarray, gpd.GeoDataFrame]] = []

    for development in developments:
        source_path = ROAD_DEV_RASTER_DIR / f"dev{development}_source_id_raster.tif"
        tt_path = ROAD_DEV_RASTER_DIR / f"dev{development}_travel_time_raster.tif"
        if not source_path.exists() or not tt_path.exists():
            continue

        with rasterio.open(source_path) as src:
            dev_source = src.read(1)
        with rasterio.open(tt_path) as src:
            dev_access_hr = src.read(1).astype(float) / 3600.0
            dev_transform = src.transform

        dev_stats = road_catchment_stats(dev_source, dev_access_hr, pop_arr, empl_arr)
        dev_link = road_links[road_links["ID_new"] == development]
        dev_sum = np.zeros(sq_source.shape, dtype=float)
        sq_common_sum = np.zeros(sq_source.shape, dtype=float)
        dev_count = 0

        for scenario in scenarios:
            sq_scenario_tt = sq_tt[sq_tt["scenario"] == scenario][["origin", "destination", "travel_time"]].copy()
            dev_scenario_tt = dev_tt[
                (dev_tt["scenario"] == scenario)
                & (dev_tt["development"] == development)
            ][["origin", "destination", "travel_time"]].copy()
            if sq_scenario_tt.empty or dev_scenario_tt.empty:
                continue

            common_pairs = sq_scenario_tt[["origin", "destination"]].merge(
                dev_scenario_tt[["origin", "destination"]],
                on=["origin", "destination"],
                how="inner",
            ).drop_duplicates()
            if common_pairs.empty:
                continue

            sq_common_tt = sq_scenario_tt.merge(common_pairs, on=["origin", "destination"], how="inner")
            dev_common_tt = dev_scenario_tt.merge(common_pairs, on=["origin", "destination"], how="inner")

            sq_values = road_accessibility_by_origin(
                sq_common_tt,
                sq_stats,
                sq_stats,
                sq_stats,
            )
            dev_values = road_accessibility_by_origin(
                dev_common_tt,
                dev_stats,
                dev_stats,
                sq_stats,
            )
            if dev_values.empty:
                continue

            sq_common_sum += np.nan_to_num(
                values_to_source_raster(sq_source, sq_values, "origin", "accessibility"),
                nan=0.0,
            )
            dev_sum += np.nan_to_num(
                values_to_source_raster(sq_source, dev_values, "origin", "accessibility"),
                nan=0.0,
            )
            dev_count += 1

        if dev_count == 0:
            continue

        sq_common_mean = sq_common_sum / dev_count
        dev_raster_mean = dev_sum / dev_count
        delta_raster_mean = dev_raster_mean - sq_common_mean
        road_dev_results.append((development, dev_raster_mean, delta_raster_mean, dev_link))

    road_absolute_range = compute_global_absolute_range([sq_raster_mean] + [item[1] for item in road_dev_results])
    road_delta_range = compute_global_delta_range([item[2] for item in road_dev_results])

    save_raster_plot(
        sq_raster_mean,
        road_output / "road_statusquo_mean.png",
        "Road status quo accessibility mean",
        delta=False,
        transform=sq_transform,
        lakes=lakes,
        network=road_network,
        bounds_override=road_plot_bounds,
        value_range=road_absolute_range,
    )

    for development, dev_raster_mean, delta_raster_mean, dev_link in road_dev_results:
        if str(development) in road_top10_set:
            save_raster_plot(
                dev_raster_mean,
                road_output / f"road_development_dev_{development}_mean.png",
                f"Road development accessibility mean, dev {development}",
                delta=False,
                transform=sq_transform,
                overlay_gdf=dev_link,
                lakes=lakes,
                network=road_network,
                bounds_override=road_plot_bounds,
                value_range=road_absolute_range,
            )

        save_raster_plot(
            delta_raster_mean,
            road_output / f"road_delta_dev_{development}_mean.png",
            f"Road accessibility delta mean, dev {development}",
            delta=True,
            transform=sq_transform,
            overlay_gdf=dev_link,
            lakes=lakes,
            network=road_network,
            bounds_override=road_plot_bounds,
            value_range=road_delta_range,
        )


# Load model station IDs, names, and coordinates for the rail network.
def load_rail_station_lookup() -> gpd.GeoDataFrame:
    with sqlite3.connect(RAIL_POINTS_GPKG) as conn:
        lookup = pd.read_sql_query("SELECT ID_point, NAME, XKOORD, YKOORD FROM points", conn)

    lookup["ID_point"] = lookup["ID_point"].astype(int)
    lookup["station_name"] = lookup["NAME"].replace(RAIL_NAME_NORMALIZATION)
    if lookup["XKOORD"].median() < 1_000_000:
        lookup["XKOORD"] = lookup["XKOORD"] + 2_000_000
    if lookup["YKOORD"].median() < 1_000_000:
        lookup["YKOORD"] = lookup["YKOORD"] + 1_000_000
    return gpd.GeoDataFrame(
        lookup[["ID_point", "station_name"]],
        geometry=gpd.points_from_xy(lookup["XKOORD"], lookup["YKOORD"]),
        crs="EPSG:2056",
    )


# Normalize the scenario-specific rail OD matrices to a clean integer station index.
def normalize_rail_od_matrix(od: pd.DataFrame) -> pd.DataFrame:
    work = od.copy()
    if work.index.name == "from_station":
        work = work.reset_index()
    if "from_station" in work.columns:
        work = work.rename(columns={"from_station": "origin"})
    first_col = work.columns[0]
    if first_col != "origin":
        work = work.rename(columns={first_col: "origin"})

    work["origin"] = pd.to_numeric(work["origin"], errors="coerce").round().astype("Int64")
    rename_map = {
        col: int(round(float(col)))
        for col in work.columns[1:]
        if pd.notna(pd.to_numeric(col, errors="coerce"))
    }
    work = work.rename(columns=rename_map)
    work = work.dropna(subset=["origin"]).copy()
    work["origin"] = work["origin"].astype(int)
    work = work.set_index("origin")
    work = work.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return work


# Load the municipality polygons used for the final rail choropleths.
def load_rail_communes() -> gpd.GeoDataFrame:
    return load_important_communes()


# Load the municipality-to-station assignment used for the final rail mapping step.
def load_rail_commune_mapping() -> pd.DataFrame:
    mapping = pd.read_excel(RAIL_COMMUNE_TO_STATION)
    mapping = mapping.rename(columns={"Commune_BFS_code": "BFS"})
    mapping = mapping[["BFS", "Commune", "ID_point", "Station"]].copy()
    mapping["BFS"] = pd.to_numeric(mapping["BFS"], errors="coerce")
    mapping["ID_point"] = pd.to_numeric(mapping["ID_point"], errors="coerce")
    mapping = mapping.dropna(subset=["BFS", "ID_point"]).copy()
    mapping["BFS"] = mapping["BFS"].astype(int)
    mapping["ID_point"] = mapping["ID_point"].astype(int)
    return mapping[mapping["BFS"] > 0].drop_duplicates(subset=["BFS"])


# Load the cached status-quo and development OD travel times for rail.
def load_rail_cache():
    with RAIL_OD_TIMES_CACHE.open("rb") as handle:
        return pickle.load(handle)


# Build the rail development ordering used by the cached OD-time list.
def build_rail_dev_position_lookup() -> dict[str, int]:
    dev_ids = sorted(
        str(int(float(path.stem)))
        for path in RAIL_DEV_DIR.iterdir()
        if path.is_file() and path.suffix == ".gpkg" and not path.name.startswith("._")
    )
    return {dev_id: idx for idx, dev_id in enumerate(dev_ids)}


# Load the rail development geometries that are drawn on the maps.
def load_rail_links() -> gpd.GeoDataFrame:
    links = gpd.read_file(RAIL_LINKS_GPKG)
    links["dev_id"] = pd.to_numeric(links["dev_id"], errors="coerce")
    links = links.dropna(subset=["dev_id"]).copy()
    links["dev_id"] = links["dev_id"].astype(int).astype(str)
    return links[["dev_id", "geometry"]]


# Aggregate the exogenous rail station potential from the station-catchment summary.
def load_rail_station_potential() -> pd.DataFrame:
    station_lookup = load_rail_station_lookup().copy()
    station_lookup["station_name_norm"] = station_lookup["station_name"].astype(str).str.strip()
    summary = pd.read_csv(RAIL_STATION_CATCHMENT_SUMMARY)
    summary["Station_Name"] = summary["Station_Name"].astype(str).str.strip()
    summary["mapped_station_name"] = summary["Station_Name"].replace(RAIL_CATCHMENT_NAME_OVERRIDES)
    summary["Pop"] = pd.to_numeric(summary["Pop"], errors="coerce").fillna(0.0)
    summary["Empl"] = pd.to_numeric(summary["Empl"], errors="coerce").fillna(0.0)
    summary["Pj"] = summary["Pop"] + 0.5 * summary["Empl"]

    merged = summary.merge(
        station_lookup[["ID_point", "station_name_norm"]],
        left_on="mapped_station_name",
        right_on="station_name_norm",
        how="left",
    )
    merged = merged.dropna(subset=["ID_point"]).copy()
    merged["ID_point"] = merged["ID_point"].astype(int)
    grouped = merged.groupby("ID_point", as_index=False)["Pj"].sum()
    return grouped


# Convert cached rail OD travel times to long form on the active model station IDs.
def rail_od_times_to_long(od_times_df: pd.DataFrame, station_lookup: pd.DataFrame, allowed_station_ids: set[int]) -> pd.DataFrame:
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


# Apply the rail accessibility decay function from the Grundlagenbericht.
def rail_accessibility_decay(t_minutes: float) -> float:
    return np.exp(-0.032 * t_minutes)


# Compute rail station accessibility from travel times and exogenous station potential.
def rail_station_accessibility(tt_df: pd.DataFrame, station_pj: pd.DataFrame) -> pd.DataFrame:
    work = tt_df.merge(station_pj.rename(columns={"ID_point": "destination"}), on="destination", how="left")
    work["Pj"] = work["Pj"].fillna(0.0)
    work["accessibility"] = work["travel_time_min"].apply(rail_accessibility_decay) * work["Pj"]
    return work.groupby("origin", as_index=False)["accessibility"].sum().rename(columns={"origin": "ID_point"})


# Push station accessibility values back to municipalities for the final rail map.
def rail_station_to_municipality(station_values: pd.DataFrame, commune_mapping: pd.DataFrame, communes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    municipality_values = commune_mapping.merge(station_values, on="ID_point", how="left")
    return communes.merge(municipality_values, on="BFS", how="left")


# Render a municipality-based rail accessibility map and optionally overlay the development geometry.
def save_gdf_plot(
    gdf: gpd.GeoDataFrame,
    column: str,
    output_path: Path,
    title: str,
    delta: bool,
    overlay_gdf: gpd.GeoDataFrame | None = None,
    lakes: gpd.GeoDataFrame | None = None,
    network: gpd.GeoDataFrame | None = None,
    active_line: gpd.GeoDataFrame | None = None,
    stations: gpd.GeoDataFrame | None = None,
    bounds_override: tuple[float, float, float, float] | None = None,
    value_range: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 8), dpi=220)
    plot_gdf = gdf.copy()
    if delta:
        if value_range is not None:
            vmin, vmax = value_range
        else:
            vmax = float(np.nanmax(np.abs(plot_gdf[column].to_numpy(dtype=float)))) if plot_gdf[column].notna().any() else 1.0
            vmax = max(vmax, 1e-9)
            vmin = -vmax
        plot_gdf.plot(
            column=column,
            cmap=DELTA_ACCESS_CMAP,
            ax=ax,
            linewidth=0.22,
            edgecolor="white",
            legend=True,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
            missing_kwds={"color": "white"},
        )
    else:
        plot_kwargs = {}
        if value_range is not None:
            plot_kwargs["vmin"], plot_kwargs["vmax"] = value_range
        plot_gdf.plot(
            column=column,
            cmap=ABSOLUTE_ACCESS_CMAP,
            ax=ax,
            linewidth=0.22,
            edgecolor="white",
            legend=True,
            missing_kwds={"color": "white"},
            **plot_kwargs,
        )
    add_reference_layers(
        ax,
        bounds_override if bounds_override is not None else tuple(plot_gdf.total_bounds),
        lakes=lakes,
        network=network,
        commune_boundaries=plot_gdf,
        active_line=active_line,
        stations=stations,
    )
    if overlay_gdf is not None and not overlay_gdf.empty:
        overlay_gdf.plot(ax=ax, color=DEVELOPMENT_BLUE, linewidth=2.4, zorder=8)
    if bounds_override is not None:
        xmin, ymin, xmax, ymax = bounds_override
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# Create all rail accessibility maps using exogenous station-catchment potential.
def create_rail_maps() -> None:
    rail_output = OUTPUT_ROOT / "rail"
    rail_output.mkdir(parents=True, exist_ok=True)

    rail_top10, _ = load_top10_by_mode()
    rail_top10_set = set(rail_top10)
    valuation_year = int(rail_settings.start_valuation_year)
    scenario_files = sorted(
        RAIL_SCENARIO_CACHE_DIR.glob("scenario_*.pkl"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not scenario_files:
        raise FileNotFoundError("No rail scenario cache files found for accessibility maps.")

    station_lookup_all = load_rail_station_lookup()
    commune_mapping = load_rail_commune_mapping()
    communes = load_rail_communes()
    rail_links = load_rail_links()
    lakes = load_lakes()
    rail_network = load_rail_network()
    rail_active_network = load_rail_active_service_network()
    canton_boundary = load_canton_boundary()
    rail_bounds = tuple(canton_boundary.total_bounds)
    rail_stations = filter_stations_to_displayed_network(load_rail_stations(), rail_network, rail_bounds)
    station_potential = load_rail_station_potential()
    cache = load_rail_cache()
    base_od_times = cache["od_times_status_quo"][0]
    dev_od_times = cache["od_times_dev"]
    dev_position_lookup = build_rail_dev_position_lookup()
    all_developments = list(dev_position_lookup.keys())
    bfs_index = communes[["BFS", "geometry"]].copy()
    base_sum = pd.Series(0.0, index=bfs_index["BFS"])
    base_count = 0
    scenario_contexts = []

    for scenario_file in scenario_files:
        with scenario_file.open("rb") as handle:
            scenario_od_by_year = pickle.load(handle)
        if valuation_year not in scenario_od_by_year:
            continue

        station_od = normalize_rail_od_matrix(scenario_od_by_year[valuation_year])
        station_ids = set(station_od.index.astype(int)).intersection(set(station_od.columns.astype(int)))
        station_lookup = station_lookup_all[station_lookup_all["ID_point"].isin(station_ids)].copy()
        station_pj = station_potential[station_potential["ID_point"].isin(station_ids)].copy()
        valid_station_ids = set(station_lookup["ID_point"]).intersection(set(station_pj["ID_point"]))
        if not valid_station_ids:
            continue

        station_lookup = station_lookup[station_lookup["ID_point"].isin(valid_station_ids)].copy()
        station_pj = station_pj[station_pj["ID_point"].isin(valid_station_ids)].copy()
        base_tt = rail_od_times_to_long(base_od_times, station_lookup, valid_station_ids)
        if base_tt.empty:
            continue

        scenario_contexts.append((station_lookup, valid_station_ids, station_pj))
        base_station_values = rail_station_accessibility(base_tt, station_pj)
        base_municipality = rail_station_to_municipality(base_station_values, commune_mapping, communes)
        base_municipality["accessibility"] = pd.to_numeric(base_municipality["accessibility"], errors="coerce")
        base_series = base_municipality.set_index("BFS")["accessibility"].reindex(base_sum.index).fillna(0.0)
        base_sum += base_series
        base_count += 1

    if base_count == 0:
        return

    base_mean = (base_sum / base_count).rename("accessibility")
    rail_dev_results: list[tuple[str, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]] = []
    base_municipality_mean = gpd.GeoDataFrame(
        bfs_index.merge(base_mean.reset_index(), on="BFS", how="left"),
        geometry="geometry",
        crs=communes.crs,
    )

    for development in all_developments:
        pos = dev_position_lookup.get(development)
        if pos is None or pos >= len(dev_od_times):
            continue

        dev_sum = pd.Series(0.0, index=bfs_index["BFS"])
        dev_count = 0
        dev_link = rail_links[rail_links["dev_id"] == development]
        active_line = get_rail_active_line(rail_active_network, development)

        for station_lookup, valid_station_ids, station_pj in scenario_contexts:
            dev_tt = rail_od_times_to_long(dev_od_times[pos], station_lookup, valid_station_ids)
            if dev_tt.empty:
                continue

            dev_station_values = rail_station_accessibility(dev_tt, station_pj)
            dev_municipality = rail_station_to_municipality(dev_station_values, commune_mapping, communes)
            dev_municipality["accessibility"] = pd.to_numeric(dev_municipality["accessibility"], errors="coerce")
            dev_series = dev_municipality.set_index("BFS")["accessibility"].reindex(dev_sum.index).fillna(0.0)
            dev_sum += dev_series
            dev_count += 1

        if dev_count == 0:
            continue

        dev_mean = (dev_sum / dev_count).rename("development_accessibility")
        dev_municipality_mean = gpd.GeoDataFrame(
            bfs_index.merge(dev_mean.reset_index(), on="BFS", how="left"),
            geometry="geometry",
            crs=communes.crs,
        )
        delta_mean = dev_municipality_mean[["BFS", "geometry", "development_accessibility"]].merge(
            base_mean.reset_index(),
            on="BFS",
            how="left",
        )
        delta_mean["delta_accessibility"] = delta_mean["development_accessibility"] - delta_mean["accessibility"]
        delta_mean = gpd.GeoDataFrame(delta_mean, geometry="geometry", crs=communes.crs)
        rail_dev_results.append((development, dev_municipality_mean, delta_mean, dev_link, active_line))

    rail_absolute_range = compute_global_absolute_range([base_mean] + [item[1]["development_accessibility"] for item in rail_dev_results])
    rail_delta_range = compute_global_delta_range([item[2]["delta_accessibility"] for item in rail_dev_results])

    save_gdf_plot(
        base_municipality_mean,
        "accessibility",
        rail_output / "rail_statusquo_mean.png",
        "Rail status quo accessibility mean",
        delta=False,
        lakes=lakes,
        network=rail_network,
        stations=rail_stations,
        bounds_override=rail_bounds,
        value_range=rail_absolute_range,
    )

    for development, dev_municipality_mean, delta_mean, dev_link, active_line in rail_dev_results:
        if development in rail_top10_set:
            save_gdf_plot(
                dev_municipality_mean.rename(columns={"development_accessibility": "accessibility"}),
                "accessibility",
                rail_output / f"rail_development_dev_{development}_mean.png",
                f"Rail development accessibility mean, dev {development}",
                delta=False,
                overlay_gdf=dev_link,
                lakes=lakes,
                network=rail_network,
                active_line=active_line,
                stations=rail_stations,
                bounds_override=rail_bounds,
                value_range=rail_absolute_range,
            )

        save_gdf_plot(
            delta_mean,
            "delta_accessibility",
            rail_output / f"rail_delta_dev_{development}_mean.png",
            f"Rail accessibility delta mean, dev {development}",
            delta=True,
            overlay_gdf=dev_link,
            lakes=lakes,
            network=rail_network,
            active_line=active_line,
            stations=rail_stations,
            bounds_override=rail_bounds,
            value_range=rail_delta_range,
        )


# Run the road and rail accessibility map generation workflow.
def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    create_road_maps()
    create_rail_maps()
    print(f"Saved accessibility maps to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
