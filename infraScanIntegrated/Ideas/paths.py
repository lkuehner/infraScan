from pathlib import Path

# ------------------------------------------------------------
# Paths for infraScanIntegrated
# ------------------------------------------------------------

INTEGRATED_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = INTEGRATED_ROOT.parent

ROAD_ROOT = WORKSPACE_ROOT / "infraScanRoad"
RAIL_ROOT = WORKSPACE_ROOT / "infraScanRail"

OUTPUT_ROOT = INTEGRATED_ROOT / "outputs"
REGISTRY_ROOT = OUTPUT_ROOT / "registries"

INTEGRATED_DEVELOPMENT_REGISTRY = REGISTRY_ROOT / "integrated_developments.csv"
INTEGRATED_STATIC_REGISTRY = REGISTRY_ROOT / "integrated_static.csv"
INTEGRATED_SCENARIO_REGISTRY = REGISTRY_ROOT / "integrated_scenarios.csv"
INTEGRATED_SCENARIO_EVAL_REGISTRY = REGISTRY_ROOT / "integrated_scenario_dependent.csv"

PLOT_DIRECTORY = OUTPUT_ROOT / "plots"
PLOT_SCENARIOS = PLOT_DIRECTORY / "scenarios"

def ensure_directories() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    REGISTRY_ROOT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Paths for infraScanRoad
# ------------------------------------------------------------
ROAD_NETWORK_HIGHWAY = ROAD_ROOT / "data/temp/network_highway.gpkg"
ROAD_HIGHWAY_ACCESS_CSV = ROAD_ROOT / "data/manually_gathered_data/highway_access.csv"
ROAD_GENERATED_NODES = ROAD_ROOT / "data/Network/processed/generated_nodes.gpkg"
ROAD_POINTS_CORRIDOR_ATTRIBUTE = ROAD_ROOT / "data/Network/processed/points_corridor_attribute.gpkg"
ROAD_PROTECTED_AREA_CORRIDOR_TIF = ROAD_ROOT / "data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"
ROAD_NEW_LINKS_REALISTIC = ROAD_ROOT / "data/Network/processed/new_links_realistic.gpkg"



# ------------------------------------------------------------
# Paths for infraScanRail
# ------------------------------------------------------------
RAIL_SERVICES_AK2035_PATH= RAIL_ROOT / 'data/temp/railway_services_ak2035.gpkg'
RAIL_SERVICES_AK2035_EXTENDED_PATH = RAIL_ROOT / 'data/temp/railway_services_ak2035_extended.gpkg'
RAIL_SERVICES_2024_PATH= RAIL_ROOT / 'data/temp/network_railway-services.gpkg'
RAIL_SERVICES_AK2024_EXTENDED_PATH = RAIL_ROOT / 'data/temp/network2024_railway_services_extended.gpkg'
NEW_LINKS_UPDATED_PATH = RAIL_ROOT / "data/Network/processed/updated_new_links.gpkg"
NEW_RAILWAY_LINES_PATH = RAIL_ROOT / "data/Network/processed/new_railway_lines.gpkg"
NETWORK_WITH_ALL_MODIFICATIONS = RAIL_ROOT / "data/Network/processed/combined_network_with_all_modifications.gpkg"
DEVELOPMENT_DIRECTORY = RAIL_ROOT / "data/Network/processed/developments"

RAIL_NODES_PATH = RAIL_ROOT / "data/Network/Rail_Node.csv"
RAIL_POINTS_PATH = RAIL_ROOT / "data/Network/processed/points.gpkg"
OD_KT_ZH_PATH = RAIL_ROOT / 'data/_basic_data/KTZH_00001982_00003903.xlsx'
OD_STATIONS_KT_ZH_PATH = RAIL_ROOT / 'data/traffic_flow/od/rail/ktzh/od_matrix_stations_ktzh_20.csv'
COMMUNE_TO_STATION_PATH = RAIL_ROOT / "data/Network/processed/Communes_to_railway_stations_ZH.xlsx"
GRAPH_POS_PATH = RAIL_ROOT / "data/Network/processed/graph_data.pkl"

POPULATION_RASTER = RAIL_ROOT / "data/independent_variable/processed/pop20_corrected.tif"
EMPLOYMENT_RASTER = RAIL_ROOT / "data/independent_variable/processed/empl20_corrected.tif"
POPULATION_SCENARIO_CANTON_ZH_2050 = RAIL_ROOT / "data/Scenario/KTZH_00000705_00001741.csv"
POPULATION_SCENARIO_CH_BFS_2055 = RAIL_ROOT / "data/Scenario/pop_scenario_switzerland_2055.csv"
POPULATION_SCENARIO_CH_EUROSTAT_2100 = RAIL_ROOT / "data/Scenario/Eurostat_population_CH_2100.xlsx"
POPULATION_PER_COMMUNE_ZH_2018 = RAIL_ROOT / "data/Scenario/population_by_gemeinde_2018.csv"
RANDOM_SCENARIO_CACHE_PATH = RAIL_ROOT / "data/Scenario/cache"
DISTRICT_PATH = RAIL_ROOT / "data/_basic_data/Gemeindegrenzen/UP_BEZIRKE_F.shp"

CONSTRUCTION_COSTS =  RAIL_ROOT / "data/costs/construction_cost.csv"
TOTAL_COST_WITH_GEOMETRY = RAIL_ROOT / "data/costs/total_costs_with_geometry.csv"
TOTAL_COST_RAW = RAIL_ROOT / "data/costs/total_costs_raw.csv"
COST_AND_BENEFITS_DISCOUNTED = RAIL_ROOT / "data/costs/costs_and_benefits_dev_discounted.csv"
COSTS_CONNECTION_CURVES = RAIL_ROOT / "data/costs/costs_connection_curves.xlsx"

TTS_CACHE = RAIL_ROOT / "data/Network/travel_time/cache/compute_tts_cache.pkl"



def get_rail_services_path(rail_network_settings):
    """
    Returns the path to the rail services file based on the rail network settings.
    """
    if rail_network_settings == 'AK_2035':
        return RAIL_SERVICES_AK2035_PATH
    elif rail_network_settings == 'AK_2035_extended':
        return RAIL_SERVICES_AK2035_EXTENDED_PATH
    elif rail_network_settings == 'current':
        return RAIL_SERVICES_2024_PATH
    elif rail_network_settings == '2024_extended':
        return RAIL_SERVICES_AK2024_EXTENDED_PATH
    else:
        raise ValueError("Invalid rail network settings provided.")