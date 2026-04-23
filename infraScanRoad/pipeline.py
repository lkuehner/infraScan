import os
import pickle
import time


import geopandas as gpd
import pandas as pd

from .OSM_network import *
from .data_import import *
from .generate_infrastructure import *
from .plots import *
from .random_scenarios import get_random_scenarios
from .scenarios import *
from .scoring import *
from .traveltime_delay import *
from .voronoi_tiling import *
from . import settings

# ==================================================================================
# PIPELIINE SETTINGS
# ==================================================================================
TRAVEL_TIME_SAVINGS_PIPELINES = {
    "aggregate": {
        "status_quo_fn": tt_optimization_status_quo,
        "developments_fn": tt_optimization_all_developments,
        "monetization_fn": monetize_tts,
        "status_quo_checkpoint": "tt_optimization_status_quo",
        "developments_checkpoint": "tt_optimization_developments",
    },
    "od": {
        "status_quo_fn": tt_optimization_status_quo_by_od,
        "developments_fn": tt_optimization_all_developments_by_od,
        "monetization_fn": monetize_tts_by_od,
        "status_quo_checkpoint": "tt_optimization_status_quo_by_od",
        "developments_checkpoint": "tt_optimization_developments_by_od",
    },
}


# ==================================================================================
# CHECKPOINT UTILITIES
# Checkpoints are saved to the 'checkpoints/' folder as .pkl or sentinel files.
# To re-run a section from scratch, delete its checkpoint file.
# ==================================================================================

CHECKPOINT_DIR = "checkpoints"

def _phase_label_for_checkpoint(name):
    phase_map = {
        "import_raw_data": "PHASE_2",
        "import_raw_data_corridor": "PHASE_2",
        "protected_area_corridor": "PHASE_2",
        "map_access_points": "PHASE_2",
        "generate_infrastructure": "PHASE_3",
        "import_scenario_variables": "PHASE_4",
        "scenario_generation": "PHASE_4",
        "scenario_generation_generated": "PHASE_4",
        "scenario_generation_static": "PHASE_4",
        "import_raw_data_variables": "PHASE_5",
        "protected_area_variables": "PHASE_5",
        "osm_network": "PHASE_5",
        "elevation_tunnel_bridges": "PHASE_5",
        "construction_costs": "PHASE_5",
        "externalities": "PHASE_5",
        "travel_time_voronoi": "PHASE_5",
        "travel_time_developments": "PHASE_5",
        "combine_tt_voronoi": "PHASE_5",
        "accessibility": "PHASE_5",
        "od_matrices": "PHASE_6",
        "tt_optimization_status_quo": "PHASE_6",
        "tt_optimization_developments": "PHASE_6",
        "tt_optimization_status_quo_by_od": "PHASE_6",
        "tt_optimization_developments_by_od": "PHASE_6",
        "aggregate_costs": "PHASE_7",
    }
    return phase_map.get(name, "PHASE_UNKNOWN")

def _phase_token_for_checkpoint(name):
    phase_label = _phase_label_for_checkpoint(name)
    if phase_label.startswith("PHASE_"):
        return phase_label.lower()
    return "phase_unknown"

def _cp_path(name, ext="sentinel"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    phase_token = _phase_token_for_checkpoint(name)
    return os.path.join(CHECKPOINT_DIR, f"{phase_token}_{name}.{ext}")

def _legacy_cp_path(name, ext="sentinel"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{name}.{ext}")

def checkpoint_exists(name):
    return os.path.exists(_cp_path(name)) or os.path.exists(_legacy_cp_path(name))

def save_checkpoint(name):
    """Mark a section as complete (no data to save)."""
    with open(_cp_path(name), "w") as f:
           phase_token = _phase_token_for_checkpoint(name)
           f.write(f"{phase_token}_{name}\n")
    print(f"  [CHECKPOINT] Saved: {name}")

def save_data_checkpoint(name, data):
    """Save a Python object alongside the sentinel."""
    with open(_cp_path(name, "pkl"), "wb") as f:
        pickle.dump(data, f)
    save_checkpoint(name)

def load_data_checkpoint(name):
    pkl = _cp_path(name, "pkl")
    if not os.path.exists(pkl):
        legacy_pkl = _legacy_cp_path(name, "pkl")
        if os.path.exists(legacy_pkl):
            pkl = legacy_pkl
        else:
            raise FileNotFoundError(f"No data checkpoint found for '{name}'")
    with open(pkl, "rb") as f:
        data = pickle.load(f)
    print(f"  [CHECKPOINT] Loaded: {name}")
    return data



# ================================================================================
# PHASE FUNCTIONS - Modular Pipeline Components
# ================================================================================

def phase_1_initialization(runtimes):
    """
    Creates the focus area for the analysis by defining the inner and outer boundaries.
    The boundaries are defined as polygons based on specified coordinates.
    """

    print("\n" + "="*80)
    print("PHASE 1: INITIALIZE VARIABLES")
    print("="*80 + "\n")
    st = time.time()

    
    limits_corridor = [settings.e_min, settings.n_min, 
                       settings.e_max, settings.n_max]

    # Boudary for plot
    boundary_plot = polygon_from_points(e_min=settings.e_min+1000, e_max=settings.e_max-500, 
                                        n_min=settings.n_min+1000, n_max=settings.n_max-2000)

    # Get a polygon as limits for the corridor
    innerboundary = polygon_from_points(e_min=settings.e_min, e_max=settings.e_max, 
                                        n_min=settings.n_min, n_max=settings.n_max)

    # For global operation a margin is added to the boundary
    margin = 3000 # meters
    outerboundary = polygon_from_points(e_min=settings.e_min, e_max=settings.e_max, n_min=settings.n_min, 
                                        n_max=settings.n_max, margin=margin)

    runtimes["Initialize variables"] = time.time() - st

    return limits_corridor,boundary_plot,innerboundary, outerboundary


def phase_2_data_import(limits_corridor,runtimes):
    """
    Import raw geographic data (lakes, cities).
    """
    print("\n" + "="*80)
    print("PHASE 2: DATA IMPORT")
    print("="*80 + "\n")
    st = time.time()

    # Import shapes of lake for plots
    get_lake_data()

    # Import the file containing the locations to be ploted
    import_locations()

    # Define area that is protected for constructing highway links
    if not checkpoint_exists("import_raw_data_corridor"):
        get_protected_area(limits=limits_corridor, suffix="corridor")
        get_unproductive_area(limits=limits_corridor, suffix="corridor")
        landuse(limits=limits_corridor, suffix="corridor")
        save_checkpoint("import_raw_data_corridor")
    else:
        print("  [CHECKPOINT] Skipping: import_raw_data_corridor")

    # Tif file of all unsuitable land cover and protected areas
    # File is stored to 'data\landuse_landcover\processed\zone_no_infra\protected_area_{suffix}.tif'
    if not checkpoint_exists("protected_area_corridor"):
        all_protected_area_to_raster(suffix="corridor")
        save_checkpoint("protected_area_corridor")
    else:
        print("  [CHECKPOINT] Skipping: protected_area_corridor")

    runtimes["Import land use and land cover data"] = time.time() - st


def phase_3_infrastructure_developments(innerboundary, outerboundary, runtimes):
    """
    INFRASTRUCTURE NETWORK
    # 1) Import network
    # 2) Process network
    # 3) Generate developments (new access points) and connection to existing infrastructure
    """

    print("\n" + "="*80)
    print("PHASE 3: INFRASTRUCTURE DEVELOPMENTS")
    print("="*80 + "\n")
    st = time.time()

    # 1) Import network
    # Import the highway network and preprocess it
    # Data are stored as "data/temp/network_highway.gpkg"
    # load_nw()
    
    # Read the network dataset to avoid running the function above
    network = gpd.read_file(r"data/infraScanRoad/temp/network_highway.gpkg")

    # Import manually gathered access points and map them on the highway infrastructure
    # The same point but with adjusted coordinate are saved to "data\access_highway_matched.gpkg"
    df_access = pd.read_csv(r"data/infraScanRoad/manually_gathered_data/highway_access.csv", sep=";")

    if not checkpoint_exists("map_access_points"):
        map_access_points_on_network(current_points=df_access, network=network)
        save_checkpoint("map_access_points")
    else:
        print("  [CHECKPOINT] Skipping: map_access_points")



    """
    # Plot the highway network with the access points (adjusted coordinates)
    current_points = gpd.read_file(r"data\access_highway_matched.shp")
    map_2 = CustomBasemap(boundary=polygon_from_points(current_points.total_bounds), network=network, access_points=current_points, frame=innerboundary)
    map_2.show()
    """

    ##################################################################################
    # 2) Process network

    # Simplify the physical topology of the network
    # One distinct edge between two nodes (currently multiple edges between nodes)
    # Edges are stored in "data\Network\processed\edges.gpkg"
    # Points in simplified network can be intersections ("intersection"==1) or access points ("intersection"==0)
    # Points are stored in "data\Network\processed\points.gpkg"
    reformat_network()


    # Filter the infrastructure elements that lie within a given polygon
    # Points within the corridor are stored in "data\Network\processed\points_corridor.gpkg"
    # Edges within the corridor are stored in "data\Network\processed\edges_corridor.gpkg"
    # Edges crossing the corridor border are stored in "data\Network\processed\edges_on_corridor.gpkg"
    network_in_corridor(polygon=outerboundary)



    # Add attributes to nodes within the corridor (mainly access point T/F)
    # Points with attributes saved as "data\Network\processed\points_attribute.gpkg"
    map_values_to_nodes()

    # Add attributes to the edges
    get_edge_attributes()

    # Add specific elements to the network
    required_manipulations_on_network()

    ##################################################################################
    # 3) Generate developments (new access points) and connection to existing infrastructure

    if not checkpoint_exists("generate_infrastructure"):
        # Make random points within the perimeter (extent) and filter them, so they do not fall within protected or
        # unsuitable area
        # The resulting dataframe of generated nodes is stored in "data\Network\processed\generated_nodes.gpkg"
        num_rand = 1000
        random_gdf = generated_access_points(extent=innerboundary, number=num_rand)
        filter_access_points(random_gdf)
        #filtered_gdf.to_file(r"data/infraScanRoad/Network/processed/generated_nodes.gpkg")

        # Import the generated points as dataframe
        generated_points = gpd.read_file(r"data/infraScanRoad/Network/processed/generated_nodes.gpkg")

        # Import current points as dataframe and filter only access points (no intersection points)
        current_points = gpd.read_file(r"data/infraScanRoad/Network/processed/points_corridor_attribute.gpkg")
        current_access_points = current_points.loc[current_points["intersection"] == 0]

        # Connect the generated points to the existing access points
        # New lines are stored in "data/Network/processed/new_links.gpkg"
        filtered_rand_temp = connect_points_to_network(generated_points, current_access_points)
        nearest_gdf = create_nearest_gdf(filtered_rand_temp)
        create_lines(generated_points, nearest_gdf)

        # Filter the generated links that connect to one of the access point within the considered corridor
        # These access points are defined in the manually defined list of access points
        # The links to corridor are stored in "data/Network/processed/developments_to_corridor_attribute.gpkg"
        # The generated points with link to access point in the corridor are stored in "data/Network/processed/generated_nodes_connecting_corridor.gpkg"
        # The end point [ID_new] of developments_to_corridor_attribute are equivlent to the points in generated_nodes_connecting_corridor
        only_links_to_corridor()

        # Find a routing for the generated links that considers protected areas
        # The new links are stored in "data/Network/processed/new_links_realistic.gpkg"
        # If a point is not accessible due to the banned zoned it is stored in "data/Network/processed/points_inaccessible.csv"
        raster = r'data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif'
        routing_raster(raster_path=raster)

        """
        #plot_corridor(network, limits=limits_corridor, location=location, new_nodes=filtered_rand_gdf, access_link=True)
        map_3 = CustomBasemap(boundary=outerboundary, network=network, frame=innerboundary)
        map_3.new_development(new_nodes=filtered_rand_gdf)
        map_3.show()
        """

        # Compute the voronoi polygons for the status quo and for alle developments based on euclidean distance
        # Dataframe with the voronoi polygons for the status quo is stored in "data/Voronoi/voronoi_status_quo_euclidian.gpkg"
        # Dataframe with the voronoi polygons for the all developments is stored in "data/Voronoi/voronoi_developments_euclidian.gpkg"
        get_voronoi_status_quo()
        limits_variables = get_voronoi_all_developments()
        limits_variables = [2680600, 1227700, 2724300, 1265600]
        print(limits_variables)

        save_data_checkpoint("generate_infrastructure", {"limits_variables": limits_variables})
    else:
        print("  [CHECKPOINT] Skipping: generate_infrastructure")
        limits_variables = load_data_checkpoint("generate_infrastructure")["limits_variables"]

    generated_points = gpd.read_file("data/infraScanRoad/Network/processed/generated_nodes.gpkg")
    current_points = gpd.read_file("data/infraScanRoad/Network/processed/points_corridor_attribute.gpkg")
    current_access_points = current_points.loc[current_points["intersection"] == 0]

    runtimes["Generate infrastructure developments"] = time.time() - st

    return network, limits_variables, generated_points, current_points, current_access_points



def phase_4_scenario_generation(limits_variables, runtimes):
    print("\n" + "="*80)
    print("PHASE 4: SCENARIO GENERATION")
    print("="*80 + "\n")
    st = time.time()


    # Import the raw data, reshape it partially and store it as tif
    # Tif are stored to "data/independent_variable/processed/raw/pop20.tif"
    # File name indicates population (pop) and employment (empl), the year (20), and the extent swisswide (_ch) or only for corridor (no suffix)
    if not checkpoint_exists("import_scenario_variables"):
        import_data(limits_variables)
        save_checkpoint("import_scenario_variables")
    else:
        print("  [CHECKPOINT] Skipping: import_scenario_variables")

    runtimes["Import variable for scenario (population and employment)"] = time.time() - st
    st = time.time()

    # 1) Generate scenarios depending on configured scenario_type
    scenario_generation_checkpoint = f"scenario_generation_{settings.scenario_type.lower()}"

    if not checkpoint_exists(scenario_generation_checkpoint):
        if settings.scenario_type == "STATIC":
            # Import the predicted scenario defined by the canton of Zürich
            scenario_zh = pd.read_csv(r"data/infraScanRoad/Scenario/KTZH_00000705_00001741.csv", sep=";")

            # Define the relative growth per scenario and district
            # The growth rates are stored in "data/temp/data_scenario_n.shp"
            future_scenario_zuerich_2022(scenario_zh)

            # Compute the predicted amount of population and employment in each raster cell (hectar) for each scenario
            # The resulting raster data are stored in "data/independent_variables/scenario/{col}.tif" with col being pop or empl and the scenario
            scenario_to_raster(limits_variables)

            # Aggregate the scenario data over euclidian Voronoi polygons
            polygons_gdf = gpd.read_file(r"data/infraScanRoad/Voronoi/voronoi_developments_euclidian.gpkg")
            scenario_to_voronoi(polygons_gdf, euclidean=True)

            # Convert multiple tif files to one same tif with multiple bands
            stack_tif_files(var="empl")
            stack_tif_files(var="pop")

        elif settings.scenario_type == "GENERATED":
            # Generate stochastic scenarios and export valuation-year OD matrices
            # (and generated population rasters, if configured in random_scenarios).
            get_random_scenarios(
                start_year=settings.start_year_scenario,
                end_year=settings.end_year_scenario,
                num_of_scenarios=settings.amount_of_scenarios,
                use_cache=False,
                do_plot=False,
            )
        else:
            raise ValueError(f"Unsupported scenario_type: {settings.scenario_type}")

        save_checkpoint(scenario_generation_checkpoint)
    else:
        print(f"  [CHECKPOINT] Skipping: {scenario_generation_checkpoint}")

    runtimes["Generate the scenarios"] = time.time() - st
    st = time.time()

    return


def phase_5_costs_and_accesibility(limits_variables, runtimes):
    print("\n" + "="*80)
    print("PHASE 5: COSTS AND ACCESSIBILITY")
    print("="*80 + "\n")
    st = time.time()

    # 1) Redefine protected area for scoring perimeter

    # This operation has already been done above for the corridor limits, here it is applied to the voronoi polygon limits which are bigger than the corridor limits
    if not checkpoint_exists("import_raw_data_variables"):
        get_protected_area(limits=limits_variables, suffix="variables")
        get_unproductive_area(limits=limits_variables, suffix="variables")
        landuse(limits=limits_variables, suffix="variables")
        save_checkpoint("import_raw_data_variables")
    else:
        print("  [CHECKPOINT] Skipping: import_raw_data_variables")

    # Find possible links considering land cover and protected areas
    if not checkpoint_exists("protected_area_variables"):
        all_protected_area_to_raster(suffix="variables")
        save_checkpoint("protected_area_variables")
    else:
        print("  [CHECKPOINT] Skipping: protected_area_variables")


    # 2) Import road network from OSM and rasterize it
    # Import the road network from OSM and rasterize it
    if not checkpoint_exists("osm_network"):
        nw_from_osm(limits_variables) #todo this requires data under data/Network/OSM_road that is not available.
        osm_nw_to_raster(limits_variables)
        save_checkpoint("osm_network")
    else:
        print("  [CHECKPOINT] Skipping: osm_network")
    
    runtimes["Import and rasterize local road network from OSM"] = time.time() - st
    st = time.time()

    # 3) Compute construction costs

    # Compute the elevation profile for each routing to assess the amount
    # First import the elevation model downscale the resolution and store it as raster data to 'data/elevation_model/elevation.tif'
    # resolution = 50 # meter
    #import_elevation_model(new_resolution=resolution)
    #runtimes["Import elevation model in 50 meter resolution"] = time.time() - st
    #st = time.time()

    # Compute the elevation profile for each generated highway routing based on the elevation model
    if not checkpoint_exists("elevation_tunnel_bridges"):
        links_temp = get_road_elevation_profile()
        #links_temp.to_csv(r"data/Network/processed/new_links_realistic_woTunnel.csv")

        # Based on the elevation profile of each links compute the required amount of bridges and tunnels
        # Save the dataset to "data/Network/processed/new_links_realistic_tunnel.gpkg"
        #get_tunnel_candidates(links_temp)
        tunnel_bridges(links_temp)
        save_checkpoint("elevation_tunnel_bridges")
    else:
        print("  [CHECKPOINT] Skipping: elevation_tunnel_bridges")

    runtimes["Optimize eleavtion profile of links to find need for tunnel and bridges"] = time.time() - st
    st = time.time()

    if not checkpoint_exists("construction_costs"):
        construction_costs(highway=settings.c_openhighway, tunnel=settings.c_tunnel, bridge=settings.c_bridge, ramp=settings.ramp)
        maintenance_costs(duration=settings.maintenance_duration, highway=settings.c_om_openhighway, 
                          tunnel=settings.c_om_tunnel, bridge=settings.c_om_bridge, structural=settings.c_structural_maint)
        save_checkpoint("construction_costs")
    else:
        print("  [CHECKPOINT] Skipping: construction_costs")

    runtimes["Compute construction and maintenance costs"] = time.time() - st
    st = time.time() 

    # 4) Compute costs of externalities
    # Compute the costs arrising from externalities for each development (generated points with according link to existing access point)
    # Result stored to "data/Network/processed/new_links_externalities_costs.gpkg"

    if not checkpoint_exists("externalities"):
        print(" -> Externalities")

        externalities_costs(ce_highway=settings.co2_highway, ce_tunnel=settings.co2_tunnel,
                            realloc_forest=settings.forest_reallocation ,realloc_FFF=settings.meadow_reallocation,
                            realloc_dry_meadow=settings.meadow_reallocation, realloc_period=settings.reallocation_duration,
                            nat_fragmentation=settings.fragmentation, fragm_period=settings.fragmentation_duration,
                            nat_loss_habitat=settings.habitat_loss, habitat_period=settings.habitat_loss_duration)
        noise_costs(years=settings.noise_duration, 
                    boundaries=settings.noise_distance, 
                    unit_costs=settings.noise_values)
        save_checkpoint("externalities")
    else:
        print("  [CHECKPOINT] Skipping: externalities")

    # r"data/costs/externalities.gpkg"
    # r"data/costs/noise.gpkg"

    # Add geospatial link to the table with costs
    # Result stored to "data/costs/building_externalities.gpkg"
    #map_coordinates_to_developments()

    # Plot individual cost elements on map
    #gdf_extern_costs = gpd.read_file(r"data/Network/processed/new_links_externalities_costs.gpkg")
    #gdf_constr_costs = gpd.read_file(r"data/Network/processed/new_links_construction_costs.gpkg")
    #gdf_costs = gpd.read_file(r"data/costs/building_externalities.gpkg")
    #tif_path_plot = r"data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"
    #plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, boundary=innerboundary, network=network, access_points=current_access_points)

    runtimes["Compute Externalities"] = time.time() - st
    st = time.time()

    # 5) Get Voronoi tiling based on travel time
    # Based on the rasterised road network from OSM, compute the travel time required to access the closest existing
    # highway access point from each cell in the perimeter. As result, it is also known for each cell which current
    # access points is the closest (its ID)
    # The raster file showing the travel time to the next access point is stored to 'data/Network/travel_time/travel_time_raster.tif'
    # The raster file showing the ID of the closest access point is stored in 'data/Network/travel_time/source_id_raster.tif'
    # Aggregating all cells with same closest access point is equivalent to a travel time based voronoi tiling. This is
    # stored as vector file in "data/Network/travel_time/Voronoi_statusquo.gpkg"
    if not checkpoint_exists("travel_time_voronoi"):
        travel_cost_polygon(limits_variables)
        save_checkpoint("travel_time_voronoi")
    else:
        print("  [CHECKPOINT] Skipping: travel_time_voronoi")

    voronoi_status_quo = gpd.read_file(r"data/infraScanRoad/Voronoi/voronoi_status_quo_euclidian.gpkg")
    voronoi_tt = gpd.read_file(r"data/infraScanRoad/Network/travel_time/Voronoi_statusquo.gpkg")

    # Same operation is made for all developments
    # These are store similarily than above, with id_new beeing the id of the development (ID of generated point)
    # The raster file showing the travel time to the next access point is stored to 'data/Network/travel_time/developments/dev{id_new}_travel_time_raster.tif'
    # The raster file showing the ID of the closest access point is stored in 'data/Network/travel_time/developments/dev{id_new}_source_id_raster.tif'
    # Aggregating all cells with same closest access point is equivalent to a travel time based voronoi tiling. This is
    # stored as vector file in "data/Network/travel_time/developments/dev{id_new}_Voronoi.gpkg"
    if not checkpoint_exists("travel_time_developments"):
        travel_cost_developments(limits_variables)
        save_checkpoint("travel_time_developments")
    else:
        print("  [CHECKPOINT] Skipping: travel_time_developments")

    runtimes["Voronoi tiling: Compute travel time from each raster cell to the closest access point"] = time.time() - st
    st = time.time()

    # Generate one dataframe containing the Voronoi polygons for all developments and all access points within the
    # perimeter. Before the polygons are store in an individual dataset for each development
    # The resulting dataframe is stored to "data/Voronoi/combined_developments.gpkg"
    if not checkpoint_exists("combine_tt_voronoi"):
        folder_path = "data/infraScanRoad/Network/travel_time/developments"
        single_tt_voronoi_ton_one(folder_path)

        # Based on the scenario and the travel time based Voronoi tiling, compute the predicted population and employment
        # in each polygon and for each scenario
        # Resulting dataset is stored to "data/Voronoi/voronoi_developments_tt_values.shp"
        polygon_gdf = gpd.read_file(r"data/infraScanRoad/Voronoi/combined_developments.gpkg")
        scenario_to_voronoi(polygon_gdf, euclidean=False)
        save_checkpoint("combine_tt_voronoi")
    else:
        print("  [CHECKPOINT] Skipping: combine_tt_voronoi")

    runtimes["Aggregate scenarios by Voronoi polygons"] = time.time() - st
    st = time.time()


    #################################################################################
    # 6) Compute access time costs

    # Compute the accessibility for status quo for scenarios
    if not checkpoint_exists("accessibility"):
        accessib_status_quo = accessibility_status_quo(VTT_h=settings.VTTS, 
                                                       duration=settings.travel_time_duration)

        # Compute the benefit in accessibility for each development compared to the status quo
        # The accessibility for each polygon for every development is store in "data/Voronoi/voronoi_developments_local_accessibility.gpkg"
        # The benefit of each development compared to the status quo is stored in 'data/costs/local_accessibility.csv'
        accessibility_developments(accessib_status_quo, VTT_h=settings.VTTS, 
                                   duration=settings.travel_time_duration)  # make this more efficient in terms of for loops and open files
        save_data_checkpoint("accessibility", {"accessib_status_quo": accessib_status_quo})
    else:
        print("  [CHECKPOINT] Skipping: accessibility")
        accessib_status_quo = load_data_checkpoint("accessibility")["accessib_status_quo"]

    runtimes["Compute highway access time benefits"] = time.time() - st
    st = time.time()

    return voronoi_tt


def phase_6_travel_time_savings(runtimes):
    print("\n" + "="*80)
    print("PHASE 6: TRAVEL TIME SAVINGS")
    print("="*80 + "\n")
    st = time.time()

    method = settings.travel_time_savings_method
    pipeline_config = TRAVEL_TIME_SAVINGS_PIPELINES[method] 

    run_status_quo = pipeline_config["status_quo_fn"]
    run_developments = pipeline_config["developments_fn"]
    run_monetization = pipeline_config["monetization_fn"]

    status_quo_checkpoint = pipeline_config["status_quo_checkpoint"]
    developments_checkpoint = pipeline_config["developments_checkpoint"]

    debug_enabled = bool(getattr(settings, "travel_time_debug_enabled", False))
    debug_scenarios = None
    if debug_enabled:
        configured_debug_scenarios = settings.get_travel_time_debug_scenarios()
        if settings.scenario_type == "STATIC":
            debug_scenarios = configured_debug_scenarios
        elif settings.scenario_type == "GENERATED":
            if configured_debug_scenarios:
                debug_scenarios = configured_debug_scenarios
            else:
                debug_scenarios = settings.get_representative_generated_scenarios(
                    n_scenarios=settings.amount_of_scenarios,
                    n_representatives=settings.generated_representative_scenarios_count,
                )

    max_developments = None
    if method == "od":
        max_developments = settings.od_max_developments
    elif method == "aggregate" and debug_enabled:
        max_developments = settings.aggregate_debug_max_developments



    print(f"  -> Using travel time savings method: {method}")

    # Travel time delay on highway
    if not checkpoint_exists("od_matrices"):
        if settings.scenario_type == "STATIC":
            # deterministic/static scenario workflow
            GetVoronoiOD()
            GetVoronoiOD_multi()
        elif settings.scenario_type == "GENERATED":
            # New stochastic scenario workflow:
            GetVoronoiOD_generated_status_quo(
                year=settings.start_valuation_year,
            )
            GetVoronoiOD_multi_generated(
                year=settings.start_valuation_year,
                max_developments=max_developments,
            )
        else:
            raise ValueError(f"Unsupported scenario_type: {settings.scenario_type}")

        save_checkpoint("od_matrices")
    else:
        print("  [CHECKPOINT] Skipping: od_matrices")

    runtimes["Reallocate OD matrices to Voronoi polygons"] = time.time() - st
    st = time.time()

    # To re-run this section only, delete the corresponding phase-prefixed checkpoint,
    # e.g. checkpoints/phase_6_tt_optimization_status_quo.sentinel.
    if not checkpoint_exists(status_quo_checkpoint):
        if method == "od":
            if debug_scenarios is not None:
                run_status_quo(scenarios=debug_scenarios, max_developments=max_developments)
            else:
                run_status_quo(max_developments=max_developments)
        elif debug_scenarios is not None:
            run_status_quo(scenarios=debug_scenarios)
        else:
            run_status_quo()
        save_checkpoint(status_quo_checkpoint)
    else:
        print(f"  [CHECKPOINT] Skipping: {status_quo_checkpoint}")


    if not checkpoint_exists(developments_checkpoint):
        if settings.scenario_type == "STATIC":
            link_traffic_to_map()
            print('Flag: link_traffic_to_map is complete')
        else:
            print('Flag: link_traffic_to_map skipped for GENERATED scenarios')

        if method == "od":
            if debug_scenarios is not None:
                run_developments(scenarios=debug_scenarios, max_developments=max_developments)
            else:
                run_developments(max_developments=max_developments)
        else:
            if debug_scenarios is not None or max_developments is not None:
                run_developments(scenarios=debug_scenarios, max_developments=max_developments)
            else:
                run_developments()

        print('Flag: tt_optimization_all_developments is complete')
        run_monetization(VTTS=settings.VTTS, duration=settings.travel_time_duration)
        save_checkpoint(developments_checkpoint)
    else:
        print(f"  [CHECKPOINT] Skipping: {developments_checkpoint}")


        runtimes["Compute travel time savings"] = time.time() - st
    return 
 

def phase_7_aggregation(runtimes):
    print("\n" + "="*80)
    print("PHASE 7: COST-BENEFIT INTEGRATION")
    print("="*80 + "\n")
    st = time.time()

    # Aggregate the single cost elements to one dataframe
    # Method-specific outputs are stored as total_costs_<method>.gpkg/.csv
    if not checkpoint_exists("aggregate_costs"):
        print(" -> Aggregate costs")
        aggregate_costs()
        save_checkpoint("aggregate_costs")
    else:
        print("  [CHECKPOINT] Skipping: aggregate_costs")

    # Import method-specific overall cost dataframe
    method = settings.travel_time_savings_method
    total_costs_path = fr"data/infraScanRoad/costs/total_costs_{method}.gpkg"
    gdf_costs = gpd.read_file(total_costs_path)
    print(f"  -> Using total costs input: {total_costs_path}")

    # Convert all costs in million CHF
    if settings.scenario_type == "STATIC":
        for col in ["total_low", "total_medium", "total_high"]:
            if col in gdf_costs.columns:
                gdf_costs[col] = (gdf_costs[col] / 1000000).astype(int)
    else:
        for col in ["total_mean", "total_median", "total_std"]:
            if col in gdf_costs.columns:
                gdf_costs[col] = (gdf_costs[col] / 1000000).astype(int)

    runtimes["Aggregate costs"] = time.time() - st

    return gdf_costs

def phase_8_visualization(voronoi_tt, innerboundary, network, 
                          boundary_plot,current_access_points, gdf_costs, 
                          runtimes):
    print("\n" + "="*80)
    print("PHASE 8: VISUALIZATION")
    print("="*80 + "\n")
    st = time.time()

    # Import layers to plot
    tif_path_plot = r"data/landuse_landcover/processed/zone_no_infra/protected_area_corridor.tif"

    links_beeline = gpd.read_file(r"data/infraScanRoad/Network/processed/new_links.gpkg")
    links_realistic = gpd.read_file(r"data/infraScanRoad/Network/processed/new_links_realistic.gpkg")
    print(links_realistic.head(5).to_string())

    # Plot the net benefits for each generated point and interpolate the area in between
    generated_points = gpd.read_file(r"data/infraScanRoad/Network/processed/generated_nodes.gpkg")
    # Get a gpd df with points have an ID_new that is not in links_realistic ID_new
    filtered_rand_gdf = generated_points[~generated_points["ID_new"].isin(links_realistic["ID_new"])]
    #plot_points_gen(points=generated_points, edges=links_beeline, banned_area=tif_path_plot, boundary=boundary_plot, network=network, all_zones=True, plot_name="gen_nodes_beeline")
    #plot_points_gen(points=generated_points, points_2=filtered_rand_gdf, edges=links_realistic, banned_area=tif_path_plot, boundary=boundary_plot, network=network, all_zones=False, plot_name="gen_links_realistic")

    voronoi_dev_2 = gpd.read_file(r"data/infraScanRoad/Network/travel_time/developments/dev779_Voronoi.gpkg")
    plot_voronoi_development(voronoi_tt, voronoi_dev_2, generated_points, boundary=innerboundary, network=network, access_points=current_access_points, plot_name="new_voronoi")

    #plot_voronoi_comp(voronoi_status_quo, voronoi_tt, boundary=boundary_plot, network=network, access_points=current_access_points, plot_name="voronoi")


    # Plot the net benefits for each generated point and interpolate the area in between
    # if plot_name is not False, then the plot is stored in "plot/results/{plot_name}.png"
    if settings.scenario_type == "STATIC":
        plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario low growth", boundary=boundary_plot, network=network,
                         access_points=current_access_points, plot_name="total_costs_low", col="total_low")
        plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario medium growth", boundary=boundary_plot, network=network,
                         access_points=current_access_points, plot_name="total_costs_medium", col="total_medium")
        plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario high growth", boundary=boundary_plot, network=network,
                         access_points=current_access_points, plot_name="total_costs_high", col="total_high")
    else:
        if "total_mean" in gdf_costs.columns:
            plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario mean", boundary=boundary_plot, network=network,
                             access_points=current_access_points, plot_name="total_costs_mean", col="total_mean")
        if "total_median" in gdf_costs.columns:
            plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario median", boundary=boundary_plot, network=network,
                             access_points=current_access_points, plot_name="total_costs_median", col="total_median")
        if "total_std" in gdf_costs.columns:
            plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario std", boundary=boundary_plot, network=network,
                             access_points=current_access_points, plot_name="total_costs_std", col="total_std")

    # Plot single cost element
    local_cols = [c for c in gdf_costs.columns if c.startswith("local_")]
    tt_cols = [c for c in gdf_costs.columns if c.startswith("tt_")]
    externality_cols = [c for c in gdf_costs.columns if c.startswith("externalities_")]

    local_col = "local_s1" if "local_s1" in gdf_costs.columns else (
        "local_s1_pop" if "local_s1_pop" in gdf_costs.columns else (local_cols[0] if local_cols else None)
    )
    tt_medium_col = "tt_medium" if "tt_medium" in gdf_costs.columns else (tt_cols[0] if tt_cols else None)
    tt_low_col = "tt_low" if "tt_low" in gdf_costs.columns else tt_medium_col
    externality_col = "externalities_s1" if "externalities_s1" in gdf_costs.columns else (
        externality_cols[0] if externality_cols else None
    )

    if local_col is None or tt_medium_col is None or externality_col is None:
        missing = []
        if local_col is None:
            missing.append("local_*")
        if tt_medium_col is None:
            missing.append("tt_*")
        if externality_col is None:
            missing.append("externalities_*")
        raise KeyError(f"Missing required Phase 8 columns: {missing}. Available columns: {list(gdf_costs.columns)}")

    print(f"  [PHASE 8] Using columns -> local: {local_col}, tt: {tt_medium_col}, externalities: {externality_col}")

    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="construction",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="construction and maintenance", col="construction_maintenance")
    # Due to errors when plotting convert values to integer
    gdf_costs[local_col] = gdf_costs[local_col].astype(int)
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="access time to highway",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="access_costs",col=local_col)
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="highway travel time",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="tt_costs",col=tt_medium_col)
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="noise emissions",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="externalities_costs", col=externality_col)

    # Plot uncertainty
    if settings.scenario_type == "STATIC":
        uncertainty_cols = ["total_low", "total_medium", "total_high"]
    else:
        uncertainty_cols = [c for c in gdf_costs.columns if c.startswith("total_scenario_")]

    if len(uncertainty_cols) >= 2:
        gdf_costs['mean_costs'] = gdf_costs[uncertainty_cols].mean(axis=1)
        gdf_costs["std"] = gdf_costs[uncertainty_cols].std(axis=1)
    else:
        gdf_costs['mean_costs'] = gdf_costs.get("total_mean", 0)
        gdf_costs["std"] = gdf_costs.get("total_std", 0)

    gdf_costs['cv'] = gdf_costs["std"] / abs(gdf_costs['mean_costs'])
    gdf_costs['cv'] = gdf_costs['cv'] * 10000000

    plot_cost_uncertainty(df_costs=gdf_costs, banned_area=tif_path_plot,
                          boundary=boundary_plot, network=network, col="std",
                          legend_title="Standard deviation\n[Mio. CHF]",
                          access_points=current_access_points, plot_name="uncertainty")

    plot_cost_uncertainty(df_costs=gdf_costs, banned_area=tif_path_plot,
                          boundary=boundary_plot, network=network, col="cv",
                          legend_title="Coefficient of variation/n[0/0'000'000]",
                          access_points=current_access_points, plot_name="cv")

    # Plot the uncertainty of the nbr highest ranked developments as boxplot
    boxplot(gdf_costs, 15)

    if settings.scenario_type == "STATIC":
        overall_bar_col = "total_medium"
        overall_line_cols = ["total_low", "total_medium", "total_high"]
        overall_labels = ["low growth", "medium growth", "high growth"]
    else:
        overall_bar_col = "total_mean" if "total_mean" in gdf_costs.columns else (
            "total_median" if "total_median" in gdf_costs.columns else None
        )
        generated_total_cols = [c for c in gdf_costs.columns if c.startswith("total_scenario_")]
        if generated_total_cols:
            overall_line_cols = generated_total_cols[: min(3, len(generated_total_cols))]
            overall_labels = overall_line_cols
        else:
            overall_line_cols = [c for c in ["total_mean", "total_median", "total_std"] if c in gdf_costs.columns]
            overall_labels = overall_line_cols

    if overall_bar_col is not None:
        plot_benefit_distribution_bar_single(df_costs=gdf_costs, column=overall_bar_col)

    if overall_line_cols:
        plot_benefit_distribution_line_multi(df_costs=gdf_costs, columns=overall_line_cols,
                                             labels=overall_labels, plot_name="overall", legend_title="Tested scenario")

    single_components = ["construction_maintenance", local_col, tt_low_col, externality_col]
    for i in single_components:
        gdf_costs[i] = (gdf_costs[i] / 1000000).astype(int)
    # Plot benefit distribution for all cost elements
    plot_benefit_distribution_line_multi(df_costs=gdf_costs,
                                         columns=["construction_maintenance", local_col, tt_low_col, externality_col],
                                         labels=["construction and maintenance", "access costs", "highway travel time",
                                                 "external costs"], plot_name="single_components",
                                         legend_title="Scoring components")
    #todo plot the uncertainty
    #plot_best_worse(df=gdf_costs)


    # Plot influence of discounting
    """
    map_vor = CustomBasemap(boundary=outerboundary, network=network)
    map_vor.single_development(id=2, new_nodes=filtered_rand_gdf, new_links=new_links)
    map_vor.voronoi(id=2, gdf_voronoi=voronoi_gdf)
    map_vor.show()


    for i in voronoi_gdf["ID"].unique():
        map_vor = CustomBasemap(boundary=outerboundary, network=network)
        map_vor.single_development(id=i, new_nodes=filtered_rand_gdf, new_links=new_links)
        map_vor.voronoi(id=i, gdf_voronoi=voronoi_gdf)
        del map_vor
    

    map_development = CustomBasemap(boundary=outerboundary, network=network, access_points=current_access_points, frame=innerboundary)
    map_development.new_development(new_nodes=filtered_rand_gdf, new_links=lines_gdf)
    map_development.show()
    """


    runtimes["Visualize results"] = time.time() - st
