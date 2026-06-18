import os
from pathlib import Path

from infraScan.infraScanRoad import settings as road_settings


def _resolve_main() -> str:
    candidates = [
        getattr(road_settings, "MAIN", None),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(str(candidate))
        if os.path.isdir(os.path.join(normalized, "data")):
            return normalized

    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "data")):
        return cwd

    return os.path.abspath(cwd)


MAIN = _resolve_main()

DATA_ROOT = Path(MAIN)
ROAD_DATA_ROOT = DATA_ROOT / "euler" / "alldev" / "data" / "infraScanRoad"
INTEGRATED_COSTS_DIR = DATA_ROOT / "data Kopie" / "infraScanIntegrated" / "costs"
INTEGRATED_PLOTS_DIR = DATA_ROOT / "plots" / "Integrated"
SCORE_RESULTS_DIR = INTEGRATED_COSTS_DIR / "score_results"
SCORE_RESULTS_LONG_PATH = SCORE_RESULTS_DIR / "score_results_long.csv"
SCORE_RESULTS_TIDY_PATH = SCORE_RESULTS_DIR / "score_results_tidy.csv"
INTEGRATED_RUN_REPORT_PATH = SCORE_RESULTS_DIR / "integrated_run_report.txt"
GENERATED_PLOTS_DIR = INTEGRATED_PLOTS_DIR / "generated"

ROAD_COSTS_DIR = ROAD_DATA_ROOT / "costs"
ROAD_EXTERNALITIES_DETAIL_CSV = ROAD_DATA_ROOT / "traffic_flow" / "link_flow_externalities" / "link_flow_externalities_long.csv"
RAIL_COSTS_DIR = DATA_ROOT / "data" / "infraScanRail" / "costs"
RAIL_TRAIN_KM_CSV = DATA_ROOT / "data" / "infraScanRail" / "Network" / "processed" / "train_km.csv"
RAIL_DISCOUNTED_TTS_CSV = DATA_ROOT / "data" / "infraScanRail" / "costs" / "costs_and_benefits_discounted.csv"
RAIL_NETWORK_PATH = DATA_ROOT / "data" / "infraScanRail" / "Network"
ROAD_NETWORK_PATH = ROAD_DATA_ROOT / "Network"
ROAD_NETWORK_PROCESSED_DIR = ROAD_NETWORK_PATH / "processed"
ROAD_EDGES_GPKG = ROAD_NETWORK_PROCESSED_DIR / "edges_with_attribute.gpkg"
ROAD_NEW_LINKS_GPKG = ROAD_NETWORK_PROCESSED_DIR / "new_links.gpkg"
ROAD_HIGHWAY_NETWORK_GPKG = ROAD_DATA_ROOT / "temp" / "network_highway.gpkg"
SWISS_MUNICIPALITY_BOUNDARIES_PATH = Path(
    "/Volumes/WD_Windows/MSc_Thesis/data/Spatial_Data/Boundaries/SwissBoundaries_Municipalities_2026_CH.gpkg"
)
SWISS_LAKES_PATH = Path(
    "/Volumes/WD_Windows/MSc_Thesis/data/Spatial_Data/Land_Use/Hydrography/swissTLMRegio_Lake.shp"
)
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
