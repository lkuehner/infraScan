import os
from pathlib import Path

# Use relative path from the script location
# MAIN = str(Path(__file__).parent.resolve()) 
# MAIN = '/Volumes/WD_Windows/MSc_Thesis/infraScanRail'
MAIN = '/Users/laura/Desktop/infraScan_lkuehner/infraScan'

RAIL_SERVICES_AK2035_PATH= 'data/temp/railway_services_ak2035.gpkg'
RAIL_SERVICES_AK2035_EXTENDED_PATH = 'data/temp/railway_services_ak2035_extended.gpkg'
RAIL_SERVICES_2024_PATH= 'data/temp/network_railway-services.gpkg'
RAIL_SERVICES_AK2024_EXTENDED_PATH = 'data/temp/network2024_railway_services_extended.gpkg'
NEW_LINKS_UPDATED_PATH = "data/Network/processed/updated_new_links.gpkg"
NEW_RAILWAY_LINES_PATH = "data/Network/processed/new_railway_lines.gpkg"
NETWORK_WITH_ALL_MODIFICATIONS = "data/Network/processed/combined_network_with_all_modifications.gpkg"
DEVELOPMENT_DIRECTORY = "data/Network/processed/developments"

RAIL_NODES_PATH = "data/Network/Rail_Node.csv"
RAIL_POINTS_PATH = "data/Network/processed/points.gpkg"
OD_KT_ZH_PATH = 'data/_basic_data/KTZH_00001982_00003903.xlsx'
OD_STATIONS_KT_ZH_PATH = 'data/traffic_flow/od/rail/ktzh/od_matrix_stations_ktzh_20.csv'
COMMUNE_TO_STATION_PATH = "data/Network/processed/Communes_to_railway_stations_ZH.xlsx"
GRAPH_POS_PATH = "data/Network/processed/graph_data.pkl"

POPULATION_RASTER = "data/independent_variable/processed/pop20_corrected.tif"
EMPLOYMENT_RASTER = "data/independent_variable/processed/empl20_corrected.tif"
POPULATION_SCENARIO_CANTON_ZH_2050 = "data/Scenario/KTZH_00000705_00001741.csv"
POPULATION_SCENARIO_CH_BFS_2055 = "data/Scenario/pop_scenario_switzerland_2055.csv"
POPULATION_SCENARIO_CH_EUROSTAT_2100 = "data/Scenario/Eurostat_population_CH_2100.xlsx"
POPULATION_PER_COMMUNE_ZH_2018 = "data/Scenario/population_by_gemeinde_2018.csv"
RANDOM_SCENARIO_CACHE_PATH = "data/Scenario/cache"
DISTRICT_PATH = "data/_basic_data/Gemeindegrenzen/UP_BEZIRKE_F.shp"

CONSTRUCTION_COSTS =  "data/costs/construction_cost.csv"
TOTAL_COST_WITH_GEOMETRY = "data/costs/total_costs_with_geometry.csv"
TOTAL_COST_RAW = "data/costs/total_costs_raw.csv"
COST_AND_BENEFITS_DISCOUNTED = "data/costs/costs_and_benefits_dev_discounted.csv"
COSTS_CONNECTION_CURVES = "data/costs/costs_connection_curves.xlsx"

TTS_CACHE = "data/Network/travel_time/cache/compute_tts_cache.pkl"

PLOT_DIRECTORY = "plots"
PLOT_SCENARIOS = "plots/scenarios"

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