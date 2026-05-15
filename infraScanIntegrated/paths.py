import os

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
