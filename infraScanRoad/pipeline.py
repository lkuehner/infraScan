import os
import pickle
import time


import geopandas as gpd
import pandas as pd

from .OSM_network import *
from .data_import import *
from .generate_infrastructure import *
from .plots import *
from .scenarios import *
from .scoring import *
from .traveltime_delay import *
from .voronoi_tiling import *
from . import settings


# ==================================================================================
# CHECKPOINT UTILITIES
# Checkpoints are saved to the 'checkpoints/' folder as .pkl or sentinel files.
# To re-run a section from scratch, delete its checkpoint file.
# ==================================================================================

CHECKPOINT_DIR = "checkpoints"

def _cp_path(name, ext="sentinel"):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{name}.{ext}")

def checkpoint_exists(name):
    return os.path.exists(_cp_path(name))

def save_checkpoint(name):
    """Mark a section as complete (no data to save)."""
    with open(_cp_path(name), "w") as f:
        f.write("done")
    print(f"  [CHECKPOINT] Saved: {name}")

def save_data_checkpoint(name, data):
    """Save a Python object alongside the sentinel."""
    with open(_cp_path(name, "pkl"), "wb") as f:
        pickle.dump(data, f)
    save_checkpoint(name)

def load_data_checkpoint(name):
    pkl = _cp_path(name, "pkl")
    if not os.path.exists(pkl):
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
    if not checkpoint_exists("import_raw_data"):
        get_protected_area(limits=limits_corridor)
        get_unproductive_area(limits=limits_corridor)
        landuse(limits=limits_corridor)
        save_checkpoint("import_raw_data")
    else:
        print("  [CHECKPOINT] Skipping: import_raw_data")

    # Tif file of all unsuitable land cover and protected areas
    # File is stored to 'data\landuse_landcover\processed\zone_no_infra\protected_area_{suffix}.tif'

    #all_protected_area_to_raster(suffix="corridor")

    runtimes["Import land use and land cover data"] = time.time() - st


def phase_3_infrastructure_developments(innerboundary, runtimes):
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
    network = gpd.read_file(r"data/temp/network_highway.gpkg")

    # Import manually gathered access points and map them on the highway infrastructure
    # The same point but with adjusted coordinate are saved to "data\access_highway_matched.gpkg"
    df_access = pd.read_csv(r"data/manually_gathered_data/highway_access.csv", sep=";")

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
    #reformat_network()


    # Filter the infrastructure elements that lie within a given polygon
    # Points within the corridor are stored in "data\Network\processed\points_corridor.gpkg"
    # Edges within the corridor are stored in "data\Network\processed\edges_corridor.gpkg"
    # Edges crossing the corridor border are stored in "data\Network\processed\edges_on_corridor.gpkg"
    #network_in_corridor(polygon=outerboundary)



    # Add attributes to nodes within the corridor (mainly access point T/F)
    # Points with attributes saved as "data\Network\processed\points_attribute.gpkg"
    #map_values_to_nodes()

    # Add attributes to the edges
    #get_edge_attributes()

    # Add specific elements to the network
    #required_manipulations_on_network()

    ##################################################################################
    # 3) Generate developments (new access points) and connection to existing infrastructure

    if not checkpoint_exists("generate_infrastructure"):
        # Make random points within the perimeter (extent) and filter them, so they do not fall within protected or
        # unsuitable area
        # The resulting dataframe of generated nodes is stored in "data\Network\processed\generated_nodes.gpkg"
        num_rand = 1000
        random_gdf = generated_access_points(extent=innerboundary, number=num_rand)
        filter_access_points(random_gdf)
        #filtered_gdf.to_file(r"data/Network/processed/generated_nodes.gpkg")

        # Import the generated points as dataframe
        generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")

        # Import current points as dataframe and filter only access points (no intersection points)
        current_points = gpd.read_file(r"data/Network/processed/points_corridor_attribute.gpkg")
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

    generated_points = gpd.read_file("data/Network/processed/generated_nodes.gpkg")
    current_points = gpd.read_file("data/Network/processed/points_corridor_attribute.gpkg")
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

    # 1) Define scenario based on cantonal predictions
    if not checkpoint_exists("scenario_generation"):
        # Import the predicted scenario defined by the canton of Zürich
        scenario_zh = pd.read_csv(r"data/Scenario/KTZH_00000705_00001741.csv", sep=";")

        # Define the relative growth per scenario and district
        # The growth rates are stored in "data/temp/data_scenario_n.shp"
        future_scenario_zuerich_2022(scenario_zh)
        # Plot the growth rates as computed above for population and employment and over three scenarios
        #plot_2x3_subplots(scenario_polygon, outerboundary, network, location)

        # Compute the predicted amount of population and employment in each raster cell (hectar) for each scenario
        # The resulting raster data are stored in "data/independent_variables/scenario/{col}.tif" with col being pop or empl and the scenario
        scenario_to_raster(limits_variables)

        # Aggregate the the scenario data to over the voronoi polygons, here euclidian polygons
        # Store the resulting file to "data/Voronoi/voronoi_developments_euclidian_values.shp"
        polygons_gdf = gpd.read_file(r"data/Voronoi/voronoi_developments_euclidian.gpkg")
        scenario_to_voronoi(polygons_gdf, euclidean=True)

        # Convert multiple tif files to one same tif with multiple bands
        stack_tif_files(var="empl")
        stack_tif_files(var="pop")
        save_checkpoint("scenario_generation")
    else:
        print("  [CHECKPOINT] Skipping: scenario_generation")

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
    #get_protected_area(limits=limits_variables)
    #get_unproductive_area(limits=limits_variables)
    #landuse(limits=limits_variables)

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

    voronoi_status_quo = gpd.read_file(r"data/Voronoi/voronoi_status_quo_euclidian.gpkg")
    voronoi_tt = gpd.read_file(r"data/Network/travel_time/Voronoi_statusquo.gpkg")

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
        folder_path = "data/Network/travel_time/developments"
        single_tt_voronoi_ton_one(folder_path)

        # Based on the scenario and the travel time based Voronoi tiling, compute the predicted population and employment
        # in each polygon and for each scenario
        # Resulting dataset is stored to "data/Voronoi/voronoi_developments_tt_values.shp"
        polygon_gdf = gpd.read_file(r"data/Voronoi/combined_developments.gpkg")
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

    # Travel time delay on highway
    if not checkpoint_exists("od_matrices"):
        # Compute the OD matrix for the current infrastructure under all scenarios
        GetVoronoiOD()
        # od = GetVoronoiOD()

        # Compute the OD matrix for the infrastructure developments under all scenarios
        GetVoronoiOD_multi()
        save_checkpoint("od_matrices")
    else:
        print("  [CHECKPOINT] Skipping: od_matrices")

    runtimes["Reallocate OD matrices to Voronoi polygons"] = time.time() - st
    st = time.time()

    # NOTE: tt_optimization_status_quo() currently crashes with:
    #   NameError: name 'n_demand' is not defined
    #   in scoring.py:1683 inside get_nw_data()
    # Fix scoring.py first, then delete checkpoints/tt_optimization_status_quo.sentinel
    # to re-run only from this point onward.
    if not checkpoint_exists("tt_optimization_status_quo"):
        tt_optimization_status_quo()
        save_checkpoint("tt_optimization_status_quo")
    else:
        print("  [CHECKPOINT] Skipping: tt_optimization_status_quo")

    if not checkpoint_exists("tt_optimization_developments"):
        # check if flow are possible
        link_traffic_to_map()
        print('Flag: link_traffic_to_map is complete')
        # Run travel time optimization for infrastructure developments and all scenarios
        tt_optimization_all_developments()
        print('Flag: tt_optimization_all_developments is complete')
        # Monetize travel time savings
        monetize_tts(VTTS=settings.VTTS, duration=settings.travel_time_duration)
        save_checkpoint("tt_optimization_developments")
    else:
        print("  [CHECKPOINT] Skipping: tt_optimization_developments")

    return 
 

def phase_7_aggregation(runtimes):
    print("\n" + "="*80)
    print("PHASE 7: COST-BENEFIT INTEGRATION")
    print("="*80 + "\n")
    st = time.time()

    # Aggregate the single cost elements to one dataframe
    # New dataframe is stored in "data/costs/total_costs.gpkg"
    # New dataframe also stored in "data/costs/total_costs.csv"
    if not checkpoint_exists("aggregate_costs"):
        print(" -> Aggregate costs")
        aggregate_costs()
        save_checkpoint("aggregate_costs")
    else:
        print("  [CHECKPOINT] Skipping: aggregate_costs")

    # Import to the overall cost dataframe
    gdf_costs = gpd.read_file(r"data/costs/total_costs.gpkg")
    # Convert all costs in million CHF
    gdf_costs["total_low"] = (gdf_costs["total_low"] / 1000000).astype(int)
    gdf_costs["total_medium"] = (gdf_costs["total_medium"] / 1000000).astype(int)
    gdf_costs["total_high"] = (gdf_costs["total_high"] / 1000000).astype(int)

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

    links_beeline = gpd.read_file(r"data/Network/processed/new_links.gpkg")
    links_realistic = gpd.read_file(r"data/Network/processed/new_links_realistic.gpkg")
    print(links_realistic.head(5).to_string())

    # Plot the net benefits for each generated point and interpolate the area in between
    generated_points = gpd.read_file(r"data/Network/processed/generated_nodes.gpkg")
    # Get a gpd df with points have an ID_new that is not in links_realistic ID_new
    filtered_rand_gdf = generated_points[~generated_points["ID_new"].isin(links_realistic["ID_new"])]
    #plot_points_gen(points=generated_points, edges=links_beeline, banned_area=tif_path_plot, boundary=boundary_plot, network=network, all_zones=True, plot_name="gen_nodes_beeline")
    #plot_points_gen(points=generated_points, points_2=filtered_rand_gdf, edges=links_realistic, banned_area=tif_path_plot, boundary=boundary_plot, network=network, all_zones=False, plot_name="gen_links_realistic")

    voronoi_dev_2 = gpd.read_file(r"data/Network/travel_time/developments/dev779_Voronoi.gpkg")
    plot_voronoi_development(voronoi_tt, voronoi_dev_2, generated_points, boundary=innerboundary, network=network, access_points=current_access_points, plot_name="new_voronoi")

    #plot_voronoi_comp(voronoi_status_quo, voronoi_tt, boundary=boundary_plot, network=network, access_points=current_access_points, plot_name="voronoi")


    # Plot the net benefits for each generated point and interpolate the area in between
    # if plot_name is not False, then the plot is stored in "plot/results/{plot_name}.png"
    plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario low growth", boundary=boundary_plot, network=network,
                     access_points=current_access_points, plot_name="total_costs_low",col="total_low")
    plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario medium growth", boundary=boundary_plot, network=network,
                     access_points=current_access_points, plot_name="total_costs_medium",col="total_medium")
    plot_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="scenario high growth", boundary=boundary_plot, network=network,
                     access_points=current_access_points, plot_name="total_costs_high",col="total_high")

    # Plot single cost element

    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="construction",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="construction and maintenance", col="construction_maintenance")
    # Due to erros when plotting convert values to integer
    gdf_costs["local_s1"] = gdf_costs["local_s1"].astype(int)
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="access time to highway",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="access_costs",col="local_s1")
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="highway travel time",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="tt_costs",col="tt_medium")
    plot_single_cost_result(df_costs=gdf_costs, banned_area=tif_path_plot, title_bar="noise emissions",
                            boundary=boundary_plot, network=network, access_points=current_access_points,
                            plot_name="externalities_costs", col="externalities_s1")

    # Plot uncertainty
    gdf_costs['mean_costs'] = gdf_costs[["total_low", "total_medium", "total_high"]].mean(axis=1)
    gdf_costs["std"] = gdf_costs[["total_low", "total_medium", "total_high"]].std(axis=1)
    gdf_costs['cv'] = gdf_costs[["total_low", "total_medium", "total_high"]].std(axis=1) / abs(gdf_costs['mean_costs'])
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

    plot_benefit_distribution_bar_single(df_costs=gdf_costs, column="total_medium")

    plot_benefit_distribution_line_multi(df_costs=gdf_costs, columns=["total_low", "total_medium", "total_high"],
                                         labels=["low growth", "medium growth",
                                                 "high growth"], plot_name="overall", legend_title="Tested scenario")

    single_components = ["construction_maintenance", "local_s1", "tt_low", "externalities_s1"]
    for i in single_components:
        gdf_costs[i] = (gdf_costs[i] / 1000000).astype(int)
    # Plot benefit distribution for all cost elements
    plot_benefit_distribution_line_multi(df_costs=gdf_costs,
                                         columns=["construction_maintenance", "local_s1", "tt_low", "externalities_s1"],
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