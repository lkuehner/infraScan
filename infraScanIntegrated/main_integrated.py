# import packages
import os
import warnings
from pathlib import Path

# Set Matplotlib backend to 'Agg' for non-interactive environments 
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from infraScan.infraScanIntegrated import config as integrated_config
from infraScan.infraScanIntegrated import settings as integrated_settings
from infraScan.infraScanIntegrated import pipeline_integrated as integrated_pipeline
from infraScan.infraScanIntegrated import random_scenarios as integrated_random_scenarios


from infraScan.infraScanRail import pipeline as rail_pipeline
from infraScan.infraScanRail import settings as rail_settings
from infraScan.infraScanRail import paths as rail_paths


from infraScan.infraScanRoad import pipeline as road_pipeline
from infraScan.infraScanRoad import settings as road_settings


def infrascan_integrated():
    # ================================================================================
    # Phase 1: Initialization
    # Shared Initialization (e.g., workspace, settigs, corridor)
    # ================================================================================
    warnings.filterwarnings("ignore")  # TODO: No warnings should be ignored
    runtimes = {}
    

    integrated_config.configure_rail()
    # Global Pipeline?
    # Checkpoints?

    os.chdir(rail_paths.MAIN) # TODO: Adjust for global pipeline

    # Study Area Definition
    limits_corridor,boundary_plot,innerboundary, outerboundary = road_pipeline.phase_1_initialization(runtimes)

    # ================================================================================
    # Phase 2: Data Import
    # ! AGIRIS adjusts this step in the rail module - to be integrated !
    # ================================================================================

    road_pipeline.phase_2_data_import(limits_corridor,runtimes)

    # ================================================================================
    # Phase 3 | RAIL: Infrastructure Developments
    # ================================================================================
   
    # Phase 3.1: Baseline Capacity Analysis
    points, baseline_prep_path, baseline_sections_path, enhanced_network_label = \
        rail_pipeline.phase_3_baseline_capacity_analysis(runtimes)

    # PHASE 3.2: Infrastructure Developments
    dev_id_lookup, capacity_analysis_results = \
        rail_pipeline.phase_4_infrastructure_developments(points, runtimes)

    
    # ================================================================================
    # Phase 4 | RAIL: Scenario independent evaluation
    # ================================================================================
    
    # Phase 4.1: Demand Analysis (OD Matrix)
    rail_pipeline.phase_5_demand_analysis(points, runtimes)

    # Phase 4.2: Travel Time Computation
    od_times_dev, od_times_status_quo, G_status_quo, G_development = \
        rail_pipeline.phase_6_travel_time_computation(dev_id_lookup, runtimes)
    
    # Phase 4.3: Passenger Flow Visualization
    if rail_settings.plot_passenger_flow:
        rail_pipeline.phase_7_passenger_flow_visualization(
            G_development, G_status_quo, dev_id_lookup, runtimes
        )

    # ================================================================================
    # Phase 3 | ROAD: Infrastructure Developments
    # ================================================================================
    network, limits_variables, generated_points, current_points, current_access_points = \
        road_pipeline.phase_3_infrastructure_developments(innerboundary, runtimes)  


    # ================================================================================
    # Phase 4 | ROAD: Scenario independent evaluation
    # ================================================================================
    

    # ================================================================================
    # Phase 5: Scenario Generation
    # ================================================================================
    shared_scenarios_result = None
    if integrated_settings.scenario_type == "GENERATED":
        shared_scenarios_result = integrated_random_scenarios.generate_and_apply_shared_scenarios(
            start_year=integrated_settings.start_year_scenario,
            end_year=integrated_settings.end_year_scenario,
            num_of_scenarios=integrated_settings.amount_of_scenarios,
            representative_scenarios_count=integrated_settings.representative_scenarios_count,
            run_road=True,
            run_rail=True,
            do_plot=False,
        )
    elif rail_settings.OD_type == 'canton_ZH':
        rail_pipeline.phase_8_scenario_generation(runtimes)

    # ================================================================================
    # Phase 6: RAIL | Evaluate Developments under each Scenario
    # ================================================================================

    # Phase 6.1: Travel time Savings
    dev_list, monetized_tt, scenario_list = \
        rail_pipeline.phase_9_travel_time_savings(
            dev_id_lookup, od_times_dev, od_times_status_quo, runtimes
        )

    # Phase 6.2: Construction & Maintenace Cost Estimation
    rail_pipeline.phase_10_new_construction_maintenance_costs(monetized_tt, runtimes)
    
    # Phase 6.3: Cost-Benefit Integration
    rail_pipeline.phase_11_new_cost_benefit_integration(runtimes)

    # Phase 6.4: Cost Aggregation
    rail_pipeline.phase_12_new_cost_aggregation(runtimes)

    # ================================================================================
    # Phase 6: ROAD | Evaluate Developments under each Scenario
    # ================================================================================
    # TODO: Adjust evaluation for the new scenario generation

    voronoi_tt = road_pipeline.phase_5_costs_and_accesibility(limits_variables, runtimes)

    road_pipeline.phase_6_travel_time_savings(runtimes)
    # TODO:OD Travel time to be calculated

    gdf_costs =road_pipeline.phase_7_aggregation(runtimes)

    # ================================================================================
    # Phase 7: Results Visualization
    # ================================================================================

    rail_results = {
    "dev_list": dev_list,
    "monetized_tt": monetized_tt,
    "scenario_list": scenario_list,
    }

    road_results = {
        "gdf_costs": gdf_costs,
    }


    # RAIL
    rail_pipeline.phase_13_results_visualization(runtimes)

    # ROAD
    road_pipeline.phase_8_visualization(voronoi_tt, innerboundary, network, 
                        boundary_plot,current_access_points, gdf_costs, 
                        runtimes)
    
    # TODO: Integrated visualization of road and rail results
    comparison_df = integrated_pipeline.build_integrated_comparison_df(rail_results, road_results)


    # ================================================================================
    # Save Runtimes
    # ================================================================================


    return {
        "runtimes": runtimes,
        "innerboundary": innerboundary,
        "outerboundary": outerboundary,
        "points": points,
        "baseline_prep_path": baseline_prep_path,
        "baseline_sections_path": baseline_sections_path,
        "enhanced_network_label": enhanced_network_label,
        "dev_id_lookup": dev_id_lookup,
        "capacity_analysis_results": capacity_analysis_results,
        "shared_scenarios_result": shared_scenarios_result,
    }


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    infrascan_integrated()
