import os
from pathlib import Path

# Central project root for integrated runs. Road and rail keep their standalone
# defaults, but the integrated orchestrator pushes this root into both modules.
MAIN = "/cluster/home/lkuehner/MSc_Thesis/"
# MAIN = "/Volumes/WD_Windows/MSc_Thesis"


DATA_ROOT = Path(MAIN)
PACKAGE_ROOT = Path(__file__).resolve().parent
#ROAD_DATA_ROOT = DATA_ROOT / "euler" / "alldev" / "data" / "infraScanRoad"
ROAD_DATA_ROOT = DATA_ROOT / "data" / "infraScanRoad"
INTEGRATED_COSTS_DIR = DATA_ROOT / "data" / "infraScanIntegrated" / "costs"
PLOTS_ROOT = DATA_ROOT / "plots"
INTEGRATED_PLOTS_DIR = PLOTS_ROOT / "integrated"
ROAD_STANDALONE_PLOTS_DIR = INTEGRATED_PLOTS_DIR / "road_standalone"
RAIL_STANDALONE_PLOTS_DIR = INTEGRATED_PLOTS_DIR / "rail_standalone"
SCORE_RESULTS_DIR = INTEGRATED_COSTS_DIR / "score_results"
SCORE_RESULTS_LONG_PATH = SCORE_RESULTS_DIR / "score_results_long.csv"
SCORE_RESULTS_TIDY_PATH = SCORE_RESULTS_DIR / "score_results_tidy.csv"
INTEGRATED_RUN_REPORT_PATH = PACKAGE_ROOT / "integrated_run_report.txt"
RAIL_STANDALONE_RUN_REPORT_PATH = PACKAGE_ROOT / "rail_standalone_run_report.txt"
ROAD_STANDALONE_RUN_REPORT_PATH = PACKAGE_ROOT / "road_standalone_run_report.txt"
GENERATED_PLOTS_DIR = INTEGRATED_PLOTS_DIR / "overview"
ACCESSIBILITY_MAPS_DIR = INTEGRATED_PLOTS_DIR / "accessibility_maps"

ROAD_COSTS_DIR = ROAD_DATA_ROOT / "costs"
ROAD_EXTERNALITIES_DETAIL_CSV = ROAD_DATA_ROOT / "traffic_flow" / "link_flow_externalities" / "link_flow_externalities_long.csv"
RAIL_COSTS_DIR = DATA_ROOT / "data" / "infraScanRail" / "costs"
RAIL_TRAIN_KM_CSV = DATA_ROOT / "data" / "infraScanRail" / "Network" / "processed" / "train_km.csv"
RAIL_DISCOUNTED_TTS_CSV = DATA_ROOT / "data" / "infraScanRail" / "costs" / "costs_and_benefits_discounted.csv"
RAIL_NETWORK_PATH = DATA_ROOT / "data" / "infraScanRail" / "Network"
ROAD_NETWORK_PATH = ROAD_DATA_ROOT / "Network"
ROAD_NETWORK_PROCESSED_DIR = ROAD_NETWORK_PATH / "processed"
ROAD_TRAVELTIME_DIR = ROAD_NETWORK_PATH / "travel_time"
ROAD_DEV_RASTER_DIR = ROAD_TRAVELTIME_DIR / "developments"
ROAD_STATUS_QUO_OD_TT_CSV = ROAD_DATA_ROOT / "traffic_flow" / "od" / "status_quo_od_tt.csv"
ROAD_DEVELOPMENTS_OD_TT_CSV = ROAD_DATA_ROOT / "traffic_flow" / "od" / "developments_od_tt.csv"
ROAD_VORONOI_GPKG = ROAD_TRAVELTIME_DIR / "Voronoi_statusquo.gpkg"
ROAD_EDGES_GPKG = ROAD_NETWORK_PROCESSED_DIR / "edges_with_attribute.gpkg"
ROAD_NEW_LINKS_GPKG = ROAD_NETWORK_PROCESSED_DIR / "new_links.gpkg"
ROAD_POINTS_GPKG = ROAD_NETWORK_PROCESSED_DIR / "points_with_attribute.gpkg"
ROAD_GENERATED_POINTS_GPKG = ROAD_NETWORK_PROCESSED_DIR / "generated_nodes.gpkg"
ROAD_HIGHWAY_NETWORK_GPKG = ROAD_DATA_ROOT / "temp" / "network_highway.gpkg"
SWISS_MUNICIPALITY_BOUNDARIES_PATH = DATA_ROOT / "data" / "Spatial_Data" / "Boundaries" / "SwissBoundaries_Municipalities_2026_CH.gpkg"
SWISS_DISTRICT_BOUNDARIES_PATH = DATA_ROOT / "data" / "Spatial_Data" / "Boundaries" / "SwissBoundaries_Bezirke_2026_CH.gpkg"
SWISS_LAKES_PATH = DATA_ROOT / "data" / "Spatial_Data" / "Land_Use" / "Hydrography" / "swissTLMRegio_Lake.shp"
PROCESSED_LAKES_GPKG = DATA_ROOT / "data" / "landuse_landcover" / "processed" / "lake_data_zh.gpkg"
POPULATION_RASTER_2023 = DATA_ROOT / "data" / "Spatial_Data" / "Land_Use" / "Population" / "population_2023.tif"
EMPLOYMENT_RASTER_2023 = DATA_ROOT / "data" / "Spatial_Data" / "Land_Use" / "Employment" / "employment_2023.tif"

RAIL_NETWORK_PROCESSED_DIR = RAIL_NETWORK_PATH / "processed"
RAIL_TRAVELTIME_CACHE_DIR = RAIL_NETWORK_PATH / "travel_time" / "cache"
RAIL_SCENARIO_CACHE_DIR = DATA_ROOT / "data" / "Scenario" / "cache" / "rail"
RAIL_OD_TIMES_CACHE = RAIL_TRAVELTIME_CACHE_DIR / "od_times.pkl"
RAIL_POINTS_GPKG = RAIL_NETWORK_PROCESSED_DIR / "points.gpkg"
RAIL_DEVELOPMENTS_DIR = RAIL_NETWORK_PROCESSED_DIR / "developments"
RAIL_UPDATED_NEW_LINKS_GPKG = RAIL_NETWORK_PROCESSED_DIR / "updated_new_links.gpkg"
RAIL_SPLIT_S_BAHN_LINES_GPKG = RAIL_NETWORK_PROCESSED_DIR / "split_s_bahn_lines.gpkg"
RAIL_NEW_RAILWAY_LINES_GPKG = RAIL_NETWORK_PROCESSED_DIR / "new_railway_lines.gpkg"
RAIL_COMMUNE_TO_STATION_XLSX = RAIL_NETWORK_PROCESSED_DIR / "Communes_to_railway_stations_ZH.xlsx"
RAIL_ACTIVE_SERVICE_NETWORK_GPKG = DATA_ROOT / "data" / "infraScanRail" / "Network" / "processed" / "combined_network_with_all_modifications.gpkg"

SCENARIO_CACHE_SHARED_DIR = os.path.join(MAIN, "data", "Scenario", "cache", "shared")
SHARED_COMPONENTS_PATH = os.path.join(SCENARIO_CACHE_SHARED_DIR, "shared_scenario_components.pkl")
SHARED_SUMMARY_PATH = os.path.join(SCENARIO_CACHE_SHARED_DIR, "shared_scenario_summary.csv")
SHARED_SELECTION_PATH = os.path.join(SCENARIO_CACHE_SHARED_DIR, "shared_representative_scenarios.csv")
SHARED_POPULATION_RASTER_DIR = os.path.join(SCENARIO_CACHE_SHARED_DIR, "population_rasters")

# Shared input data for integrated scenario orchestration.
# Defaults follow the rail path registry, but can be overridden here centrally.
POPULATION_SCENARIO_CH_BFS_2055 = os.path.join(MAIN, "data", "Scenario", "pop_scenario_switzerland_2055.csv")
POPULATION_SCENARIO_CANTON_ZH_2050 = os.path.join(MAIN, "data", "Scenario", "KTZH_00000705_00001741.csv")
POPULATION_SCENARIO_CH_EUROSTAT_2100 = os.path.join(MAIN, "data", "Scenario", "Eurostat_population_CH_2100.xlsx")
