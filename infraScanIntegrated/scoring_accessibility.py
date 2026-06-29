"""Create accessibility maps for the integrated pipeline.

The module reads the travel-time outputs and spatial inputs produced by the
road and rail pipelines and converts them into map-ready accessibility layers.
It writes three map types for each mode:

- status quo accessibility
- absolute accessibility change
- percentage accessibility change

Road maps are based on road access-point Voronoi catchments. Rail maps are
based on station catchments derived from the municipality-to-station assignment.
The two modes therefore use different spatial units; cross-modal interpretation
should focus on the direction and location of changes within each mode.
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
from matplotlib.lines import Line2D
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject

from infraScan.infraScanIntegrated import paths as integrated_paths
from infraScan.infraScanRail import settings as rail_settings
from infraScan.infraScanRoad import settings as road_settings


DATA_ROOT = integrated_paths.DATA_ROOT
OUTPUT_ROOT = integrated_paths.ACCESSIBILITY_MAPS_DIR

ROAD_SQ_TT = integrated_paths.ROAD_STATUS_QUO_OD_TT_CSV
ROAD_DEV_TT = integrated_paths.ROAD_DEVELOPMENTS_OD_TT_CSV
ROAD_TRAVELTIME_DIR = integrated_paths.ROAD_TRAVELTIME_DIR
ROAD_DEV_RASTER_DIR = integrated_paths.ROAD_DEV_RASTER_DIR
ROAD_VORONOI_GPKG = integrated_paths.ROAD_VORONOI_GPKG
ROAD_LINKS_GPKG = integrated_paths.ROAD_NEW_LINKS_GPKG
ROAD_NETWORK_GPKG = integrated_paths.ROAD_HIGHWAY_NETWORK_GPKG
ROAD_POINTS_GPKG = integrated_paths.ROAD_POINTS_GPKG
ROAD_GENERATED_POINTS_GPKG = integrated_paths.ROAD_GENERATED_POINTS_GPKG

RAIL_POINTS_GPKG = integrated_paths.RAIL_POINTS_GPKG
RAIL_SCENARIO_CACHE_DIR = integrated_paths.RAIL_SCENARIO_CACHE_DIR
RAIL_OD_TIMES_CACHE = integrated_paths.RAIL_OD_TIMES_CACHE
RAIL_DEV_DIR = integrated_paths.RAIL_DEVELOPMENTS_DIR
RAIL_LINKS_GPKG = integrated_paths.RAIL_UPDATED_NEW_LINKS_GPKG
RAIL_OVERVIEW_NETWORK_GPKG = integrated_paths.RAIL_SPLIT_S_BAHN_LINES_GPKG
RAIL_NEW_RAILWAY_LINES_GPKG = integrated_paths.RAIL_NEW_RAILWAY_LINES_GPKG
RAIL_ACTIVE_SERVICE_NETWORK_GPKG = integrated_paths.RAIL_ACTIVE_SERVICE_NETWORK_GPKG
RAIL_COMMUNE_TO_STATION = integrated_paths.RAIL_COMMUNE_TO_STATION_XLSX

COMMUNE_SHP = integrated_paths.SWISS_MUNICIPALITY_BOUNDARIES_PATH
PROCESSED_LAKES_GPKG = integrated_paths.PROCESSED_LAKES_GPKG
POPULATION_RASTER = integrated_paths.POPULATION_RASTER_2023
EMPLOYMENT_RASTER = integrated_paths.EMPLOYMENT_RASTER_2023

ROAD_BETA_PER_HOUR = 3.0
RAIL_BETA_QUADRATIC_MINUTES = 0.032
IMPORTANT_CANTON_NR = 1

DEVELOPMENT_BLUE = "#0E5FAB"
ROAD_DEVELOPMENT_COLOR = "#f4a52e"
RAIL_X10_COLOR = ROAD_DEVELOPMENT_COLOR
LAKE_FILL = "#BDE0FF"
BOUNDARY_COLOR = "#303030"
DEVELOPMENT_COLOR_LOOKUP: dict[tuple[str, str], str] = {}

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
        (0.50, "#F0F0F0"),
        (0.60, "#DCE8D8"),
        (0.76, "#91B58D"),
        (0.88, "#6E946A"),
        (1.00, "#4E704D"),
    ],
)
RAIL_NAME_NORMALIZATION = {
    "Niederglatt": "Niederglatt ZH",
    "Oberglatt": "Oberglatt ZH",
}


def scenario_sort_key(name: str):
    try:
        return int(str(name).split("_")[-1])
    except Exception:
        return str(name)


def compute_relative_accessibility_change(
    development: pd.Series,
    baseline: pd.Series,
    min_baseline: float = 1e-9,
) -> pd.Series:
    """Return percentage accessibility change relative to the status quo."""
    baseline = pd.to_numeric(baseline, errors="coerce")
    development = pd.to_numeric(development, errors="coerce")
    return np.where(
        baseline.abs() > min_baseline,
        (development - baseline) / baseline * 100.0,
        np.nan,
    )


def compute_global_absolute_range(series_list: list[pd.Series]) -> tuple[float, float]:
    values = pd.concat([pd.to_numeric(series, errors="coerce") for series in series_list], ignore_index=True)
    values = values[np.isfinite(values)]
    if values.empty:
        return (0.0, 1.0)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return (vmin, vmax)


def compute_global_centered_range(series_list: list[pd.Series]) -> tuple[float, float]:
    values = pd.concat([pd.to_numeric(series, errors="coerce") for series in series_list], ignore_index=True)
    values = values[np.isfinite(values)]
    if values.empty:
        return (-1.0, 1.0)
    vmax = max(float(values.abs().max()), 1e-9)
    return (-vmax, vmax)


def clip_gdf_to_bounds(
    gdf: gpd.GeoDataFrame | None,
    bounds: tuple[float, float, float, float],
    pad: float = 0.0,
) -> gpd.GeoDataFrame | None:
    if gdf is None or gdf.empty:
        return gdf
    xmin, ymin, xmax, ymax = bounds
    return gdf.cx[(xmin - pad):(xmax + pad), (ymin - pad):(ymax + pad)].copy()


def load_common_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    lakes = gpd.read_file(PROCESSED_LAKES_GPKG).to_crs("EPSG:2056")
    if "GEWAESSERN" in lakes.columns:
        lakes = lakes[lakes["GEWAESSERN"].isin(["Zürichsee", "Greifensee", "Pfäffikersee"])].copy()
    lakes = lakes[["geometry"]]

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
        communes = communes[communes["kantonsnummer"] == IMPORTANT_CANTON_NR].copy()
    communes = communes[[col for col in ["BFS", "name", "kantonsnummer", "geometry"] if col in communes.columns]]

    canton_boundary = gpd.GeoDataFrame(
        communes.dissolve()[["geometry"]],
        geometry="geometry",
        crs=communes.crs,
    ).reset_index(drop=True)
    return lakes, communes, canton_boundary


def add_reference_layers(
    ax,
    bounds: tuple[float, float, float, float],
    *,
    lakes: gpd.GeoDataFrame | None = None,
    network: gpd.GeoDataFrame | None = None,
    boundary: gpd.GeoDataFrame | None = None,
    stations: gpd.GeoDataFrame | None = None,
    access_points: gpd.GeoDataFrame | None = None,
) -> None:
    lakes_clip = clip_gdf_to_bounds(lakes, bounds, pad=2_000.0)
    if lakes_clip is not None and not lakes_clip.empty:
        lakes_clip.plot(ax=ax, color=LAKE_FILL, linewidth=0.35, zorder=4)

    network_clip = clip_gdf_to_bounds(network, bounds, pad=2_000.0)
    if network_clip is not None and not network_clip.empty:
        network_clip.plot(ax=ax, color="black", linewidth=0.55, alpha=0.9, zorder=5)

    boundary_clip = clip_gdf_to_bounds(boundary, bounds, pad=2_000.0)
    if boundary_clip is not None and not boundary_clip.empty:
        boundary_clip.boundary.plot(ax=ax, color=BOUNDARY_COLOR, linewidth=0.18, alpha=0.55, zorder=6)

    station_clip = clip_gdf_to_bounds(stations, bounds, pad=2_000.0)
    if station_clip is not None and not station_clip.empty:
        station_clip.plot(ax=ax, color="black", markersize=4, zorder=7)

    access_clip = clip_gdf_to_bounds(access_points, bounds, pad=2_000.0)
    if access_clip is not None and not access_clip.empty:
        access_clip.plot(ax=ax, color="black", markersize=7, zorder=7)


def save_accessibility_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    output_path: Path,
    title: str,
    *,
    mode: str,
    delta: bool = False,
    value_range: tuple[float, float] | None = None,
    overlay_gdf: gpd.GeoDataFrame | None = None,
    overlay_label: str | None = None,
    overlay_color: str = DEVELOPMENT_BLUE,
    lakes: gpd.GeoDataFrame | None = None,
    network: gpd.GeoDataFrame | None = None,
    boundary: gpd.GeoDataFrame | None = None,
    stations: gpd.GeoDataFrame | None = None,
    access_points: gpd.GeoDataFrame | None = None,
    new_access_point: gpd.GeoDataFrame | None = None,
    bounds_override: tuple[float, float, float, float] | None = None,
    colorbar_label: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_gdf = gdf.copy()
    plot_gdf[column] = pd.to_numeric(plot_gdf[column], errors="coerce")
    plot_gdf = plot_gdf[plot_gdf[column].notna()].copy()
    if plot_gdf.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 8), dpi=220)
    if delta:
        vmin, vmax = value_range or compute_global_centered_range([plot_gdf[column]])
        plot_gdf.plot(
            column=column,
            cmap=DELTA_ACCESS_CMAP,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
            ax=ax,
            linewidth=0.22,
            edgecolor="white",
            legend=True,
            legend_kwds={"shrink": 0.5, "label": colorbar_label or "Accessibility change"},
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
            legend_kwds={"shrink": 0.5, "label": colorbar_label or "Accessibility index"},
            missing_kwds={"color": "white"},
            **plot_kwargs,
        )

    bounds = bounds_override or tuple(plot_gdf.total_bounds)
    add_reference_layers(
        ax,
        bounds,
        lakes=lakes,
        network=network,
        boundary=plot_gdf if boundary is None else boundary,
        stations=stations,
        access_points=access_points,
    )
    if overlay_gdf is not None and not overlay_gdf.empty:
        overlay_gdf.plot(ax=ax, color=overlay_color, linewidth=2.2, zorder=8)
    if new_access_point is not None and not new_access_point.empty:
        new_access_point.plot(ax=ax, color=overlay_color, markersize=18, zorder=9)

    legend_handles = []
    if mode == "road":
        legend_handles.extend(
            [
                Line2D([0], [0], color="black", lw=1.2, label="Current highway network"),
                Line2D([0], [0], color=overlay_color, lw=2.2, label=overlay_label or "Road development"),
            ]
        )
    elif mode == "rail":
        legend_handles.extend(
            [
                Line2D([0], [0], color="black", lw=1.2, label="Current rail network"),
                Line2D([0], [0], color=overlay_color, lw=2.2, label=overlay_label or "Rail development"),
            ]
        )
    if overlay_gdf is not None and not overlay_gdf.empty:
        ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(0.0, -0.03), frameon=False, fontsize=9)

    xmin, ymin, xmax, ymax = bounds
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def reproject_raster_to_grid(input_path: Path, target_shape: tuple[int, int], target_transform, target_crs) -> np.ndarray:
    with rasterio.open(input_path) as src:
        source = src.read(1).astype(float)
        source = np.nan_to_num(source, nan=0.0, posinf=0.0, neginf=0.0)
        destination = np.zeros(target_shape, dtype=float)
        reproject(
            source=source,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_transform,
            dst_crs=target_crs,
            resampling=Resampling.nearest,
            dst_nodata=0.0,
        )
    return destination


def load_road_inputs() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    sq_tt = pd.read_csv(ROAD_SQ_TT)
    dev_tt = pd.read_csv(ROAD_DEV_TT)
    for df in (sq_tt, dev_tt):
        for col in ["origin", "destination", "demand", "travel_time"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["scenario"] = df["scenario"].astype(str)
    dev_tt["development"] = pd.to_numeric(dev_tt["development"], errors="coerce")

    sq_tt = sq_tt.dropna(subset=["origin", "destination", "travel_time"]).copy()
    dev_tt = dev_tt.dropna(subset=["development", "origin", "destination", "travel_time"]).copy()
    sq_tt[["origin", "destination"]] = sq_tt[["origin", "destination"]].astype(int)
    dev_tt[["development", "origin", "destination"]] = dev_tt[["development", "origin", "destination"]].astype(int)

    scenarios = sorted(set(sq_tt["scenario"]).intersection(dev_tt["scenario"]), key=scenario_sort_key)
    if getattr(road_settings, "generated_select_representative_scenarios", False):
        selected = road_settings.get_representative_generated_scenarios()
        scenarios = [scenario for scenario in selected if scenario in scenarios]
    return sq_tt, dev_tt, scenarios


def load_voronoi(path: Path) -> gpd.GeoDataFrame:
    voronoi = gpd.read_file(path).to_crs("EPSG:2056")
    id_col = next((col for col in ("ID_point", "ID", "id") if col in voronoi.columns), None)
    if id_col is None:
        raise KeyError(f"No catchment id column found in {path}")
    voronoi["ID_point"] = pd.to_numeric(voronoi[id_col], errors="coerce").astype("Int64")
    voronoi = voronoi.dropna(subset=["ID_point"]).copy()
    voronoi["ID_point"] = voronoi["ID_point"].astype(int)
    return voronoi[["ID_point", "geometry"]]


def road_catchment_stats(
    source_arr: np.ndarray,
    access_hr_arr: np.ndarray,
    pop_arr: np.ndarray,
    empl_arr: np.ndarray,
) -> pd.DataFrame:
    valid = (
        (source_arr >= 0)
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


def road_accessibility_by_origin(
    tt_df: pd.DataFrame,
    origin_stats: pd.DataFrame,
    dest_access_stats: pd.DataFrame,
    dest_pj_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Road accessibility: A_i = sum_j exp(-beta * c_ij) * P_j."""
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
    work[["origin_access_hr", "dest_access_hr", "Pj"]] = work[["origin_access_hr", "dest_access_hr", "Pj"]].fillna(0.0)
    work["generalized_tt_hr"] = work["origin_access_hr"] + work["travel_time"] + work["dest_access_hr"]
    work["accessibility"] = np.exp(-ROAD_BETA_PER_HOUR * work["generalized_tt_hr"]) * work["Pj"]
    return work.groupby("origin", as_index=False)["accessibility"].sum()


def aggregate_cell_values_to_catchments(
    catchment_arr: np.ndarray,
    value_arr: np.ndarray,
    valid_cells: np.ndarray | None = None,
) -> pd.DataFrame:
    valid = (catchment_arr >= 0) & np.isfinite(catchment_arr) & np.isfinite(value_arr)
    if valid_cells is not None:
        valid &= valid_cells
    if not np.any(valid):
        return pd.DataFrame(columns=["ID_point", "accessibility"])
    frame = pd.DataFrame(
        {
            "ID_point": catchment_arr[valid].astype(int),
            "accessibility": value_arr[valid].astype(float),
        }
    )
    return frame.groupby("ID_point", as_index=False)["accessibility"].mean()


def values_to_source_raster(source_arr: np.ndarray, values: pd.DataFrame, key_col: str, value_col: str) -> np.ndarray:
    out = np.full(source_arr.shape, np.nan, dtype=float)
    mapping = values[[key_col, value_col]].dropna().drop_duplicates(subset=[key_col]).copy()
    mapping[key_col] = pd.to_numeric(mapping[key_col], errors="coerce").astype("Int64")
    value_map = mapping.dropna(subset=[key_col]).set_index(key_col)[value_col].to_dict()
    valid = source_arr >= 0
    out[valid] = pd.Series(source_arr[valid].astype(int)).map(value_map).to_numpy()
    return out


def load_road_map_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    links = gpd.read_file(ROAD_LINKS_GPKG).to_crs("EPSG:2056")
    links["ID_new"] = pd.to_numeric(links["ID_new"], errors="coerce")
    links = links.dropna(subset=["ID_new"]).copy()
    links["ID_new"] = links["ID_new"].astype(int)

    network = gpd.read_file(ROAD_NETWORK_GPKG).to_crs("EPSG:2056")
    network = network[["geometry"]]

    access_points = gpd.read_file(ROAD_POINTS_GPKG).to_crs("EPSG:2056")
    if "intersection" in access_points.columns:
        access_points["intersection"] = pd.to_numeric(access_points["intersection"], errors="coerce")
        access_points = access_points[access_points["intersection"] == 0].copy()
    if "ID_point" in access_points.columns:
        access_points["ID_point"] = pd.to_numeric(access_points["ID_point"], errors="coerce").astype("Int64")
        access_points = access_points.dropna(subset=["ID_point"]).copy()
        access_points["ID_point"] = access_points["ID_point"].astype(int)
    access_points = access_points[[col for col in ["ID_point", "geometry"] if col in access_points.columns]]

    generated_points = gpd.read_file(ROAD_GENERATED_POINTS_GPKG).to_crs("EPSG:2056")
    generated_points["ID_new"] = pd.to_numeric(generated_points["ID_new"], errors="coerce").astype("Int64")
    generated_points = generated_points.dropna(subset=["ID_new"]).copy()
    generated_points["ID_new"] = generated_points["ID_new"].astype(int)
    return links, network, access_points, generated_points[["ID_new", "geometry"]]


def create_road_maps(target_developments: set[int] | None = None) -> None:
    """Create road status quo, absolute change, and percentage change maps."""
    road_output = OUTPUT_ROOT / "road"
    road_output.mkdir(parents=True, exist_ok=True)

    sq_tt, dev_tt, scenarios = load_road_inputs()
    lakes, _, _ = load_common_layers()
    road_links, road_network, road_access_points_all, road_generated_points = load_road_map_layers()

    with rasterio.open(ROAD_TRAVELTIME_DIR / "source_id_raster.tif") as src:
        sq_source = src.read(1)
        road_shape = (src.height, src.width)
        road_transform = src.transform
        road_crs = src.crs
    with rasterio.open(ROAD_TRAVELTIME_DIR / "travel_time_raster.tif") as src:
        sq_access_hr = src.read(1).astype(float) / 3600.0

    valid_sq_cells = (sq_source >= 0) & np.isfinite(sq_source)
    pop_arr = reproject_raster_to_grid(POPULATION_RASTER, road_shape, road_transform, road_crs)
    empl_arr = reproject_raster_to_grid(EMPLOYMENT_RASTER, road_shape, road_transform, road_crs)
    sq_stats = road_catchment_stats(sq_source, sq_access_hr, pop_arr, empl_arr)

    sq_voronoi = load_voronoi(ROAD_VORONOI_GPKG)
    sq_index = pd.Index(sorted(sq_voronoi["ID_point"].unique()), name="ID_point")
    sq_sum = pd.Series(0.0, index=sq_index)
    sq_obs = pd.Series(0, index=sq_index, dtype=int)
    for scenario in scenarios:
        sq_values = road_accessibility_by_origin(sq_tt[sq_tt["scenario"] == scenario].copy(), sq_stats, sq_stats, sq_stats)
        sq_series = sq_values.set_index("origin")["accessibility"].reindex(sq_index)
        sq_sum += sq_series.fillna(0.0)
        sq_obs += sq_series.notna().astype(int)
    sq_mean = (sq_sum / sq_obs.replace(0, np.nan)).rename("accessibility")
    sq_gdf = gpd.GeoDataFrame(sq_voronoi.merge(sq_mean.reset_index(), on="ID_point", how="left"), geometry="geometry", crs=sq_voronoi.crs)

    sq_raster_mean = values_to_source_raster(sq_source, sq_mean.reset_index().rename(columns={"ID_point": "origin"}), "origin", "accessibility")
    sq_raster_mean[~valid_sq_cells] = np.nan

    developments = sorted(dev_tt["development"].drop_duplicates().astype(int).tolist())
    if target_developments is not None:
        developments = [development for development in developments if development in target_developments]

    results: list[tuple[int, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]] = []
    for development in developments:
        source_path = ROAD_DEV_RASTER_DIR / f"dev{development}_source_id_raster.tif"
        tt_path = ROAD_DEV_RASTER_DIR / f"dev{development}_travel_time_raster.tif"
        voronoi_path = ROAD_DEV_RASTER_DIR / f"dev{development}_Voronoi.gpkg"
        if not source_path.exists() or not tt_path.exists() or not voronoi_path.exists():
            continue

        with rasterio.open(source_path) as src:
            dev_source = src.read(1)
        with rasterio.open(tt_path) as src:
            dev_access_hr = src.read(1).astype(float) / 3600.0

        valid_dev_cells = (dev_source >= 0) & np.isfinite(dev_source)
        dev_voronoi = load_voronoi(voronoi_path)
        dev_index = pd.Index(sorted(dev_voronoi["ID_point"].unique()), name="ID_point")
        dev_stats = road_catchment_stats(dev_source, dev_access_hr, pop_arr, empl_arr)

        dev_sum = pd.Series(0.0, index=dev_index)
        dev_obs = pd.Series(0, index=dev_index, dtype=int)
        for scenario in scenarios:
            scenario_tt = dev_tt[(dev_tt["scenario"] == scenario) & (dev_tt["development"] == development)][
                ["origin", "destination", "travel_time"]
            ].copy()
            if scenario_tt.empty:
                continue
            dev_values = road_accessibility_by_origin(scenario_tt, dev_stats, dev_stats, sq_stats)
            dev_series = dev_values.set_index("origin")["accessibility"].reindex(dev_index)
            dev_sum += dev_series.fillna(0.0)
            dev_obs += dev_series.notna().astype(int)

        if dev_obs.sum() == 0:
            continue

        dev_mean = (dev_sum / dev_obs.replace(0, np.nan)).rename("development_accessibility")
        sq_on_dev = aggregate_cell_values_to_catchments(dev_source, sq_raster_mean, valid_cells=valid_dev_cells).rename(
            columns={"accessibility": "status_quo_accessibility"}
        )
        values = dev_voronoi.merge(sq_on_dev, on="ID_point", how="left")
        values = values.merge(dev_mean.reset_index(), on="ID_point", how="left")
        values["delta_accessibility"] = values["development_accessibility"] - values["status_quo_accessibility"]
        values["delta_accessibility_pct"] = compute_relative_accessibility_change(
            values["development_accessibility"],
            values["status_quo_accessibility"],
        )
        values = gpd.GeoDataFrame(values, geometry="geometry", crs=dev_voronoi.crs)
        results.append(
            (
                development,
                values,
                road_links[road_links["ID_new"] == development].copy(),
                road_generated_points[road_generated_points["ID_new"] == development].copy(),
            )
        )

    absolute_range = compute_global_absolute_range([sq_gdf["accessibility"]])
    delta_range = compute_global_centered_range([item[1]["delta_accessibility"] for item in results])
    pct_range = compute_global_centered_range([item[1]["delta_accessibility_pct"] for item in results])
    access_point_ids = set(np.unique(sq_source[valid_sq_cells]).astype(int).tolist())
    road_access_points = road_access_points_all[road_access_points_all["ID_point"].isin(access_point_ids)].copy()

    save_accessibility_map(
        sq_gdf,
        "accessibility",
        road_output / "road_statusquo_mean.png",
        "Road status quo accessibility",
        mode="road",
        value_range=absolute_range,
        lakes=lakes,
        network=road_network,
        boundary=sq_gdf,
        access_points=road_access_points,
    )
    sq_gdf.to_file(road_output / "road_statusquo_accessibility.gpkg", driver="GPKG")

    for development, values, dev_link, dev_point in results:
        values.to_file(road_output / f"road_development_dev_{development}_accessibility_change.gpkg", driver="GPKG")
        values.drop(columns="geometry").to_csv(road_output / f"road_development_dev_{development}_accessibility_change.csv", index=False)
        save_accessibility_map(
            values,
            "delta_accessibility",
            road_output / f"road_delta_dev_{development}_mean.png",
            f"Road {development}, accessibility change",
            mode="road",
            delta=True,
            value_range=delta_range,
            overlay_gdf=dev_link,
            overlay_label=f"Development {development}",
            overlay_color=DEVELOPMENT_COLOR_LOOKUP.get(("Road", str(development)), DEVELOPMENT_BLUE),
            lakes=lakes,
            network=road_network,
            boundary=values,
            access_points=road_access_points,
            new_access_point=dev_point,
            colorbar_label="Accessibility index change",
        )
        save_accessibility_map(
            values,
            "delta_accessibility_pct",
            road_output / f"road_delta_pct_dev_{development}_mean.png",
            f"Road {development}, accessibility change [%]",
            mode="road",
            delta=True,
            value_range=pct_range,
            overlay_gdf=dev_link,
            overlay_label=f"Development {development}",
            overlay_color=DEVELOPMENT_COLOR_LOOKUP.get(("Road", str(development)), DEVELOPMENT_BLUE),
            lakes=lakes,
            network=road_network,
            boundary=values,
            access_points=road_access_points,
            new_access_point=dev_point,
            colorbar_label="Accessibility change [%]",
        )


def load_rail_station_lookup() -> gpd.GeoDataFrame:
    with sqlite3.connect(RAIL_POINTS_GPKG) as conn:
        lookup = pd.read_sql_query("SELECT ID_point, NAME, XKOORD, YKOORD FROM points", conn)
    lookup["ID_point"] = lookup["ID_point"].astype(int)
    lookup["station_name"] = lookup["NAME"].replace(RAIL_NAME_NORMALIZATION)
    if lookup["XKOORD"].median() < 1_000_000:
        lookup["XKOORD"] += 2_000_000
    if lookup["YKOORD"].median() < 1_000_000:
        lookup["YKOORD"] += 1_000_000
    return gpd.GeoDataFrame(
        lookup[["ID_point", "station_name"]],
        geometry=gpd.points_from_xy(lookup["XKOORD"], lookup["YKOORD"]),
        crs="EPSG:2056",
    )


def load_rail_commune_mapping() -> pd.DataFrame:
    mapping = pd.read_excel(RAIL_COMMUNE_TO_STATION).rename(columns={"Commune_BFS_code": "BFS"})
    mapping = mapping[["BFS", "Commune", "ID_point", "Station"]].copy()
    mapping["BFS"] = pd.to_numeric(mapping["BFS"], errors="coerce")
    mapping["ID_point"] = pd.to_numeric(mapping["ID_point"], errors="coerce")
    mapping = mapping.dropna(subset=["BFS", "ID_point"]).copy()
    mapping["BFS"] = mapping["BFS"].astype(int)
    mapping["ID_point"] = mapping["ID_point"].astype(int)
    return mapping[mapping["BFS"] > 0].drop_duplicates(subset=["BFS"])


def summarize_raster_by_commune(raster_path: Path, communes: gpd.GeoDataFrame, value_name: str) -> pd.DataFrame:
    with rasterio.open(raster_path) as src:
        raster = src.read(1).astype(float)
        if src.nodata is not None:
            raster[raster == src.nodata] = np.nan
        raster = np.nan_to_num(raster, nan=0.0, posinf=0.0, neginf=0.0)
        commune_geoms = communes.to_crs(src.crs) if communes.crs != src.crs else communes
        records = []
        for row in commune_geoms[["BFS", "geometry"]].itertuples(index=False):
            mask = geometry_mask([row.geometry], transform=src.transform, invert=True, out_shape=raster.shape)
            records.append({"BFS": int(row.BFS), value_name: max(float(raster[mask].sum()) if np.any(mask) else 0.0, 0.0)})
    return pd.DataFrame(records)


def load_rail_station_potential(commune_mapping: pd.DataFrame, communes: gpd.GeoDataFrame) -> pd.DataFrame:
    communes = communes[communes["BFS"].isin(set(commune_mapping["BFS"].astype(int)))].copy()
    pop = summarize_raster_by_commune(POPULATION_RASTER, communes, "Pop")
    empl = summarize_raster_by_commune(EMPLOYMENT_RASTER, communes, "Empl")
    potential = pop.merge(empl, on="BFS", how="outer").fillna(0.0)
    potential["Pj"] = potential["Pop"] + 0.5 * potential["Empl"]
    station_potential = commune_mapping.merge(potential, on="BFS", how="left")
    station_potential[["Pop", "Empl", "Pj"]] = station_potential[["Pop", "Empl", "Pj"]].fillna(0.0)
    grouped = station_potential.groupby("ID_point", as_index=False)[["Pop", "Empl", "Pj"]].sum()
    grouped["ID_point"] = grouped["ID_point"].astype(int)
    return grouped


def build_rail_catchments(
    commune_mapping: pd.DataFrame,
    communes: gpd.GeoDataFrame,
    station_lookup: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    catchments = communes.merge(commune_mapping[["BFS", "ID_point"]], on="BFS", how="inner")
    catchments = catchments[["ID_point", "geometry"]].dissolve(by="ID_point", as_index=False)
    catchments["ID_point"] = pd.to_numeric(catchments["ID_point"], errors="coerce").astype(int)
    catchments = catchments.merge(station_lookup[["ID_point", "station_name"]].drop_duplicates(), on="ID_point", how="left")
    return gpd.GeoDataFrame(catchments, geometry="geometry", crs=communes.crs)


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
    return work.apply(pd.to_numeric, errors="coerce").fillna(0.0)


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


def rail_station_accessibility(tt_df: pd.DataFrame, station_pj: pd.DataFrame) -> pd.DataFrame:
    """Rail accessibility: A_i = sum_j exp(-0.032 * t_ij^2) * P_j."""
    work = tt_df.merge(station_pj.rename(columns={"ID_point": "destination"}), on="destination", how="left")
    work["Pj"] = work["Pj"].fillna(0.0)
    work["accessibility"] = np.exp(-RAIL_BETA_QUADRATIC_MINUTES * work["travel_time_min"] ** 2) * work["Pj"]
    return work.groupby("origin", as_index=False)["accessibility"].sum().rename(columns={"origin": "ID_point"})


def load_rail_map_layers() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    network = gpd.read_file(RAIL_OVERVIEW_NETWORK_GPKG).to_crs("EPSG:2056")
    network = network[["geometry"]]

    stations = gpd.read_file(RAIL_POINTS_GPKG).to_crs("EPSG:2056")
    stations["HST"] = pd.to_numeric(stations["HST"], errors="coerce")
    stations["ID_point"] = pd.to_numeric(stations["ID_point"], errors="coerce").astype("Int64")
    stations = stations[(stations["HST"] == 1) & stations["ID_point"].notna()].copy()
    stations["ID_point"] = stations["ID_point"].astype(int)
    stations = stations[["ID_point", "NAME", "geometry"]]

    links = gpd.read_file(RAIL_LINKS_GPKG).to_crs("EPSG:2056")
    links["dev_id"] = pd.to_numeric(links["dev_id"], errors="coerce")
    links = links.dropna(subset=["dev_id"]).copy()
    links["dev_id"] = links["dev_id"].astype(int).astype(str)
    return network, stations, links[["dev_id", "geometry"]]


def rail_development_label_lookup() -> dict[str, str]:
    costs_path = DATA_ROOT / "data" / "infraScanRail" / "costs" / "total_costs.csv"
    if not costs_path.exists():
        return {}
    df = pd.read_csv(costs_path, usecols=["development", "Sline"]).drop_duplicates()
    dev = df["development"].astype(str).str.removeprefix("Development_").str.replace(r"\.0$", "", regex=True)
    sline = df["Sline"].astype(str)
    dev_num = pd.to_numeric(dev, errors="coerce")
    labels = np.where(
        sline.isin(["G", "P"]) & dev_num.notna(),
        (dev_num.astype("Int64") - 100000).astype(str) + "_" + sline,
        sline,
    )
    return dict(zip(dev.astype(str), pd.Series(labels).astype(str)))


def rail_development_visual_line(development: str, fallback: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if not RAIL_NEW_RAILWAY_LINES_GPKG.exists():
        return fallback
    labels = rail_development_label_lookup()
    label = labels.get(str(development), str(development))
    visual_lines = gpd.read_file(RAIL_NEW_RAILWAY_LINES_GPKG).to_crs("EPSG:2056")
    if "name" not in visual_lines.columns:
        return fallback
    selected = visual_lines[visual_lines["name"].astype(str) == label].copy()
    return selected[["geometry"]] if not selected.empty else fallback


def build_rail_dev_position_lookup() -> dict[str, int]:
    dev_ids = sorted(
        str(int(float(path.stem)))
        for path in RAIL_DEV_DIR.iterdir()
        if path.is_file() and path.suffix == ".gpkg" and not path.name.startswith("._")
    )
    return {dev_id: idx for idx, dev_id in enumerate(dev_ids)}


def create_rail_maps(target_developments: set[str] | None = None) -> None:
    """Create rail status quo, absolute change, and percentage change maps."""
    rail_output = OUTPUT_ROOT / "rail"
    rail_output.mkdir(parents=True, exist_ok=True)

    valuation_year = int(rail_settings.start_valuation_year)
    lakes, communes, canton_boundary = load_common_layers()
    station_lookup_all = load_rail_station_lookup()
    commune_mapping = load_rail_commune_mapping()
    station_potential = load_rail_station_potential(commune_mapping, communes)
    catchments = build_rail_catchments(commune_mapping, communes, station_lookup_all)
    catchment_index = catchments[["ID_point", "geometry", "station_name"]].copy()
    rail_network, rail_stations, rail_links = load_rail_map_layers()
    rail_stations = rail_stations[rail_stations["ID_point"].isin(catchment_index["ID_point"])].copy()
    rail_bounds = tuple(canton_boundary.total_bounds)

    scenario_files = sorted(RAIL_SCENARIO_CACHE_DIR.glob("scenario_*.pkl"), key=lambda p: int(p.stem.split("_")[-1]))
    if not scenario_files:
        raise FileNotFoundError("No rail scenario cache files found for accessibility maps.")

    with RAIL_OD_TIMES_CACHE.open("rb") as handle:
        cache = pickle.load(handle)
    base_od_times = cache["od_times_status_quo"][0]
    dev_od_times = cache["od_times_dev"]
    dev_position_lookup = build_rail_dev_position_lookup()
    developments = list(dev_position_lookup.keys())
    if target_developments is not None:
        developments = [development for development in developments if development in target_developments]

    base_sum = pd.Series(0.0, index=catchment_index["ID_point"])
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
        base_values = rail_station_accessibility(base_tt, station_pj)
        base_series = base_values.set_index("ID_point")["accessibility"].reindex(base_sum.index).fillna(0.0)
        base_sum += base_series
        base_count += 1
        scenario_contexts.append((station_lookup, valid_station_ids, station_pj))

    if base_count == 0:
        raise RuntimeError("No rail accessibility status quo values could be computed.")

    base_mean = (base_sum / base_count).rename("accessibility")
    base_gdf = gpd.GeoDataFrame(catchment_index.merge(base_mean.reset_index(), on="ID_point", how="left"), geometry="geometry", crs=catchments.crs)

    results: list[tuple[str, str, gpd.GeoDataFrame, gpd.GeoDataFrame]] = []
    labels = rail_development_label_lookup()
    for development in developments:
        pos = dev_position_lookup.get(development)
        if pos is None or pos >= len(dev_od_times):
            continue
        dev_sum = pd.Series(0.0, index=catchment_index["ID_point"])
        dev_count = 0
        for station_lookup, valid_station_ids, station_pj in scenario_contexts:
            dev_tt = rail_od_times_to_long(dev_od_times[pos], station_lookup, valid_station_ids)
            if dev_tt.empty:
                continue
            dev_values = rail_station_accessibility(dev_tt, station_pj)
            dev_series = dev_values.set_index("ID_point")["accessibility"].reindex(dev_sum.index).fillna(0.0)
            dev_sum += dev_series
            dev_count += 1
        if dev_count == 0:
            continue
        dev_mean = (dev_sum / dev_count).rename("development_accessibility")
        values = catchment_index.merge(dev_mean.reset_index(), on="ID_point", how="left")
        values = values.merge(base_mean.reset_index(), on="ID_point", how="left")
        values["delta_accessibility"] = values["development_accessibility"] - values["accessibility"]
        values["delta_accessibility_pct"] = compute_relative_accessibility_change(
            values["development_accessibility"],
            values["accessibility"],
        )
        values = gpd.GeoDataFrame(values, geometry="geometry", crs=catchments.crs)
        dev_line = rail_development_visual_line(development, rail_links[rail_links["dev_id"] == development])
        results.append((development, labels.get(str(development), str(development)), values, dev_line))

    absolute_range = compute_global_absolute_range([base_gdf["accessibility"]])
    delta_range = compute_global_centered_range([item[2]["delta_accessibility"] for item in results])
    pct_range = compute_global_centered_range([item[2]["delta_accessibility_pct"] for item in results])

    save_accessibility_map(
        base_gdf,
        "accessibility",
        rail_output / "rail_statusquo_mean.png",
        "Rail status quo accessibility",
        mode="rail",
        value_range=absolute_range,
        lakes=lakes,
        network=rail_network,
        boundary=communes,
        stations=rail_stations,
        bounds_override=rail_bounds,
    )
    base_gdf.to_file(rail_output / "rail_statusquo_accessibility.gpkg", driver="GPKG")

    for development, label, values, dev_line in results:
        values.to_file(rail_output / f"rail_development_dev_{development}_accessibility_change.gpkg", driver="GPKG")
        values.drop(columns="geometry").to_csv(rail_output / f"rail_development_dev_{development}_accessibility_change.csv", index=False)
        overlay_color = DEVELOPMENT_COLOR_LOOKUP.get(("Rail", str(development)), DEVELOPMENT_BLUE)
        save_accessibility_map(
            values,
            "delta_accessibility",
            rail_output / f"rail_delta_dev_{development}_mean.png",
            f"Rail {label}, accessibility change",
            mode="rail",
            delta=True,
            value_range=delta_range,
            overlay_gdf=dev_line,
            overlay_label=f"Rail {label}",
            overlay_color=overlay_color,
            lakes=lakes,
            network=rail_network,
            boundary=communes,
            stations=rail_stations,
            bounds_override=rail_bounds,
            colorbar_label="Accessibility index change",
        )
        save_accessibility_map(
            values,
            "delta_accessibility_pct",
            rail_output / f"rail_delta_pct_dev_{development}_mean.png",
            f"Rail {label}, accessibility change [%]",
            mode="rail",
            delta=True,
            value_range=pct_range,
            overlay_gdf=dev_line,
            overlay_label=f"Rail {label}",
            overlay_color=overlay_color,
            lakes=lakes,
            network=rail_network,
            boundary=communes,
            stations=rail_stations,
            bounds_override=rail_bounds,
            colorbar_label="Accessibility change [%]",
        )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    create_road_maps()
    create_rail_maps()
    print(f"Saved accessibility maps to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
