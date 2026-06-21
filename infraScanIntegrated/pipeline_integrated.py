import os

from infraScan.infraScanIntegrated import common_cost_parameters
from infraScan.infraScanIntegrated import paths as integrated_paths
from infraScan.infraScanIntegrated import plots as integrated_plots
from infraScan.infraScanIntegrated import scoring_registry
from infraScan.infraScanIntegrated import settings as integrated_settings
from infraScan.infraScanIntegrated import random_scenarios_multimode as integrated_random_scenarios

from infraScan.infraScanRail import cost_parameters as rail_cost_parameters
from infraScan.infraScanRail import paths as rail_paths
from infraScan.infraScanRail import pipeline as rail_pipeline
from infraScan.infraScanRail import settings as rail_settings

from infraScan.infraScanRoad import cost_parameters as road_cost_parameters
from infraScan.infraScanRoad import pipeline as road_pipeline
from infraScan.infraScanRoad import settings as road_settings


# ================================================================================
# MODULE 0: 1. INTEGRATED CONFIGURATION
# ================================================================================

def prefix_new_runtime_keys(runtimes, existing_keys, prefix):
    """Rename newly added runtime entries so road and rail keys cannot collide."""
    new_keys = [key for key in list(runtimes.keys()) if key not in existing_keys and not key.startswith(f"{prefix}: ")]
    for key in new_keys:
        runtimes[f"{prefix}: {key}"] = runtimes.pop(key)


def print_integrated_module(module_number, title):
    """Print a compact integrated module header around legacy rail/road phase logs."""
    print("\n" + "#" * 80)
    print(f"INTEGRATED MODULE {module_number}")
    print(title)
    print("#" * 80 + "\n")


def sync_integrated_shared_settings():
    """Push shared integrated scenario settings into the legacy module configs."""
    rail_pipeline.PHASE_LOG_PREFIX = "RAIL"
    rail_paths.MAIN = integrated_paths.MAIN
    rail_paths.PLOT_DIRECTORY = str(integrated_paths.RAIL_STANDALONE_PLOTS_DIR)
    rail_paths.PLOT_SCENARIOS = os.path.join(rail_paths.PLOT_DIRECTORY, "scenarios")

    road_pipeline.PHASE_LOG_PREFIX = "ROAD"
    road_settings.MAIN = integrated_paths.MAIN
    road_settings.PLOT_DIRECTORY = str(integrated_paths.ROAD_STANDALONE_PLOTS_DIR)
    road_settings.OSM_CACHE_DIR = os.path.join(
        road_settings.MAIN, "infraScan","infraScanRoad", "cache",)
    road_settings.CHECKPOINT_DIR = os.path.join(
        road_settings.MAIN, "data", "infraScanRoad", "checkpoints",)


def apply_integrated_overrides_to_rail():
    """Apply integrated run settings to the rail standalone module."""
    rail_settings.scenario_type = integrated_settings.scenario_type
    rail_settings.amount_of_scenarios = integrated_settings.amount_of_scenarios
    rail_settings.start_year_scenario = integrated_settings.start_year_scenario
    rail_settings.end_year_scenario = integrated_settings.end_year_scenario
    rail_settings.start_valuation_year = integrated_settings.start_valuation_year
    rail_settings.plot_passenger_flow = integrated_settings.PLOT_LEGACY_RAIL

    if not integrated_settings.use_cache_rail:
        rail_settings.use_cache_network = False
        rail_settings.use_cache_pt_catchment = False
        rail_settings.use_cache_developments = False
        rail_settings.use_cache_catchmentOD = False
        rail_settings.use_cache_stationsOD = False
        rail_settings.use_cache_traveltime_graph = False
        rail_settings.use_cache_scenarios = False
        rail_settings.use_cache_tts_calc = False

    rail_settings.PIPELINE_CONFIG.visualization_mode = integrated_settings.rail_visualization_mode
    rail_settings.PIPELINE_CONFIG.grouping_strategy = integrated_settings.rail_grouping_strategy
    rail_settings.capacity_threshold = integrated_settings.rail_capacity_threshold
    rail_settings.max_enhancement_iterations = integrated_settings.rail_max_enhancement_iterations
    rail_settings.use_existing_capacity_prep = integrated_settings.rail_use_existing_capacity_prep
    rail_settings.intervention_costs_reviewed = integrated_settings.rail_intervention_costs_reviewed


def apply_integrated_overrides_to_road():
    """Apply integrated run settings to the road standalone module."""
    road_settings.scenario_type = integrated_settings.scenario_type
    road_settings.amount_of_scenarios = integrated_settings.amount_of_scenarios
    road_settings.generated_representative_scenarios_count = integrated_settings.representative_scenarios_count
    road_settings.start_year_scenario = integrated_settings.start_year_scenario
    road_settings.end_year_scenario = integrated_settings.end_year_scenario
    road_settings.start_valuation_year = integrated_settings.start_valuation_year
    road_pipeline.USE_CHECKPOINTS = integrated_settings.use_cache_road_checkpoints


def apply_integrated_cost_overrides():
    """Apply shared valuation assumptions for the integrated approach."""
    common_cost_parameters.prognosis_year = integrated_settings.start_valuation_year
    common_cost_parameters.appraisal_years = integrated_settings.appraisal_years
    common_cost_parameters.discount_rate = integrated_settings.discount_rate
    common_cost_parameters.real_wage_growth = integrated_settings.real_wage_growth

    rail_cost_parameters.VTTS = integrated_settings.rail_VTTS
    rail_cost_parameters.duration = integrated_settings.appraisal_years
    rail_cost_parameters.construction_start_year = integrated_settings.start_valuation_year
    rail_cost_parameters.discount_rate = common_cost_parameters.discount_rate
    rail_cost_parameters.tts_valuation_period = (
        integrated_settings.start_valuation_year,
        integrated_settings.start_valuation_year + integrated_settings.appraisal_years,
    )

    road_cost_parameters.VTTS = integrated_settings.road_VTTS
    road_cost_parameters.duration = integrated_settings.appraisal_years
    road_cost_parameters.maintenance_duration = integrated_settings.appraisal_years
    road_cost_parameters.travel_time_duration = integrated_settings.appraisal_years
    road_cost_parameters.noise_duration = integrated_settings.appraisal_years
    road_cost_parameters.fragmentation_duration = integrated_settings.appraisal_years
    road_cost_parameters.habitat_loss_duration = integrated_settings.appraisal_years
    road_cost_parameters.reallocation_duration = integrated_settings.appraisal_years


# ================================================================================
# MODULE 0: 2. SHARED SETUP, NETWORK PREPARATION 
# ================================================================================
def setup_integrated_inputs(runtimes):
    """Apply integrated overrides and run the shared preprocessing steps."""
    sync_integrated_shared_settings()
    apply_integrated_overrides_to_rail()
    apply_integrated_overrides_to_road()
    apply_integrated_cost_overrides()

    os.chdir(rail_paths.MAIN)

    existing_keys = set(runtimes.keys())
    limits_corridor, boundary_plot, innerboundary, outerboundary = road_pipeline.phase_1_initialization(runtimes)
    road_pipeline.phase_2_data_import(limits_corridor, runtimes)
    prefix_new_runtime_keys(runtimes, existing_keys, "road")

    existing_keys = set(runtimes.keys())
    rail_innerboundary, rail_outerboundary = rail_pipeline.phase_1_initialization(runtimes)
    rail_pipeline.phase_2_data_import(runtimes)
    prefix_new_runtime_keys(runtimes, existing_keys, "rail")

    return {
        "limits_corridor": limits_corridor,
        "boundary_plot": boundary_plot,
        "innerboundary": innerboundary,
        "outerboundary": outerboundary
    }

# ================================================================================
# MODULE 1: INFRASTRUCTURE DEVELOPMENTS
# ================================================================================


def generate_infrastructure_developments(setup_results, runtimes):
    """Generate infrastructure developments."""
    existing_keys = set(runtimes.keys())
    network, limits_variables, generated_points, current_points, current_access_points = (
        road_pipeline.phase_3_infrastructure_developments(
            setup_results["innerboundary"],
            setup_results["outerboundary"],
            runtimes,
        )
    )
    prefix_new_runtime_keys(runtimes, existing_keys, "road")

    existing_keys = set(runtimes.keys())
    points, baseline_prep_path, baseline_sections_path, enhanced_network_label = (
        rail_pipeline.phase_3_baseline_capacity_analysis(runtimes)
    )
    dev_id_lookup, capacity_analysis_results = (
        rail_pipeline.phase_4_infrastructure_developments(points, runtimes)
    )
    prefix_new_runtime_keys(runtimes, existing_keys, "rail")

    return {
        **setup_results,
        "points": points,
        "baseline_prep_path": baseline_prep_path,
        "baseline_sections_path": baseline_sections_path,
        "enhanced_network_label": enhanced_network_label,
        "dev_id_lookup": dev_id_lookup,
        "capacity_analysis_results": capacity_analysis_results,
        "network": network,
        "limits_variables": limits_variables,
        "generated_points": generated_points,
        "current_points": current_points,
        "current_access_points": current_access_points,
    }

# ================================================================================
# MODULE 2: SHARED SCENARIO GENERATION
# ================================================================================

def run_shared_scenario_generation(runtimes, limits_variables):
    """Run the integrated shared-scenario generation when required."""
    if integrated_settings.scenario_type == "GENERATED":
        existing_keys = set(runtimes.keys())
        print(
            "Generating shared scenarios: "
            f"{integrated_settings.amount_of_scenarios} total, "
            f"{integrated_settings.representative_scenarios_count} selected"
        )
        result = integrated_random_scenarios.generate_and_apply_shared_scenarios(
            start_year=integrated_settings.start_year_scenario,
            end_year=integrated_settings.end_year_scenario,
            num_of_scenarios=integrated_settings.amount_of_scenarios,
            representative_scenarios_count=integrated_settings.representative_scenarios_count,
            run_road=True,
            run_rail=True,
            road_limits_variables=limits_variables,
            do_plot=False,
        )
        print("Selected scenarios: " + ", ".join(result["selected_scenarios"]))
        prefix_new_runtime_keys(runtimes, existing_keys, "scenario")
        return result

    if rail_settings.OD_type == "canton_ZH":
        existing_keys = set(runtimes.keys())
        rail_pipeline.phase_8_scenario_generation(runtimes)
        prefix_new_runtime_keys(runtimes, existing_keys, "rail")

    return None


# ================================================================================
# MODULE 3: ASSESSMENT
#        3.1: RAIL SCENARIO EVALUATION
# ================================================================================

def run_rail_evaluation(points, dev_id_lookup, runtimes):
    """Run the rail scenario evaluation and return the integrated rail result bundle."""
    existing_keys = set(runtimes.keys())
    rail_pipeline.phase_5_demand_analysis(points, runtimes)
    od_times_dev, od_times_status_quo, G_status_quo, G_development = rail_pipeline.phase_6_travel_time_computation(
        dev_id_lookup, runtimes
    )
    dev_list, monetized_tt, scenario_list = rail_pipeline.phase_9_travel_time_savings(
        dev_id_lookup, od_times_dev, od_times_status_quo, runtimes
    )
    rail_pipeline.phase_10_new_construction_maintenance_costs(monetized_tt, runtimes)
    rail_pipeline.phase_11_new_cost_benefit_integration(runtimes)
    rail_pipeline.phase_12_new_cost_aggregation(runtimes)
    prefix_new_runtime_keys(runtimes, existing_keys, "rail")

    return {
        "dev_list": dev_list,
        "monetized_tt": monetized_tt,
        "scenario_list": scenario_list,
        "od_times_dev": od_times_dev,
        "od_times_status_quo": od_times_status_quo,
        "G_status_quo": G_status_quo,
        "G_development": G_development,
    }


# ================================================================================
# MODULE 3.2: ROAD SCENARIO EVALUATION
# ================================================================================

def run_road_evaluation(limits_variables, runtimes):
    """Run the road scenario evaluation and return raw outputs for export and plots."""
    existing_keys = set(runtimes.keys())
    voronoi_tt = road_pipeline.phase_5_costs_and_accesibility(limits_variables, runtimes)
    road_pipeline.phase_6_travel_time_savings(runtimes)
    gdf_costs = road_pipeline.phase_7_aggregation(runtimes)
    prefix_new_runtime_keys(runtimes, existing_keys, "road")

    return {
        "voronoi_tt": voronoi_tt,
        "gdf_costs": gdf_costs,
    }


# ================================================================================
# MODULE 3.3: OUTPUTS, EXPORTS, AND PLOTS
# ================================================================================

def build_integrated_comparison_df(rail_results, road_results):
    """Bundle the raw rail and road results for later integrated plotting."""
    return {
        "rail_results": rail_results,
        "road_results": road_results,
    }


def run_integrated_plots():
    """Generate the integrated plot bundle from the exported score tables."""
    integrated_plots.main()
    print("Integrated plots created.")


def write_integrated_run_report(runtimes, shared_scenarios_result):
    """Write the integrated run summary and runtime report to disk."""
    integrated_paths.SCORE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = integrated_paths.INTEGRATED_RUN_REPORT_PATH
    total_time = sum(runtimes.values())
    selected_scenarios = []
    if shared_scenarios_result is not None:
        selected_scenarios = list(shared_scenarios_result.get("selected_scenarios", []))

    with open(report_path, "w") as file:
        file.write("=" * 80 + "\n")
        file.write("INFRASCANINTEGRATED RUN REPORT\n")
        file.write("=" * 80 + "\n\n")

        file.write("SETTINGS\n")
        file.write("-" * 80 + "\n")
        file.write(f"{'Run mode':.<40} {integrated_settings.RUN_MODE}\n")
        file.write(f"{'Valuation year':.<40} {integrated_settings.start_valuation_year}\n")
        file.write(f"{'Appraisal period [years]':.<40} {integrated_settings.appraisal_years}\n")
        file.write(f"{'Discount rate':.<40} {integrated_settings.discount_rate:.2%}\n")
        file.write(f"{'Scenario mode':.<40} {integrated_settings.scenario_type}\n")
        #file.write(f"{'Selected representative scenarios':.<40} {', '.join(selected_scenarios) if selected_scenarios else '-'}\n")
        file.write("\n")

        file.write("RUNTIMES\n")
        file.write("-" * 80 + "\n")
        for part, runtime in runtimes.items():
            mins = int(runtime // 60)
            secs = int(runtime % 60)
            file.write(f"{part:.<50} {mins}m {secs}s ({runtime:.2f}s)\n")
        file.write("\n" + "=" * 80 + "\n")
        total_mins = int(total_time // 60)
        total_secs = int(total_time % 60)
        file.write(f"{'TOTAL TIME':.<50} {total_mins}m {total_secs}s ({total_time:.2f}s)\n")
        file.write("=" * 80 + "\n")

    print(f"Integrated run report saved to: {report_path}")
    return report_path


def finalize_integrated_outputs(
    runtimes,
    rail_results,
    road_results,
    shared_scenarios_result,
    G_development,
    G_status_quo,
    dev_id_lookup,
    voronoi_tt,
    innerboundary,
    network,
    boundary_plot,
    current_access_points,
    gdf_costs,
):
    """Run optional standalone outputs and export the integrated score tables."""
    if integrated_settings.INCLUDE_STANDALONE and integrated_settings.PLOT_LEGACY_RAIL and rail_settings.plot_passenger_flow:
        existing_keys = set(runtimes.keys())
        rail_pipeline.phase_7_passenger_flow_visualization(
            G_development, G_status_quo, dev_id_lookup, runtimes
        )
        prefix_new_runtime_keys(runtimes, existing_keys, "rail")

    if integrated_settings.INCLUDE_STANDALONE and integrated_settings.PLOT_LEGACY_RAIL:
        existing_keys = set(runtimes.keys())
        rail_pipeline.phase_13_results_visualization(runtimes)
        prefix_new_runtime_keys(runtimes, existing_keys, "rail")

    if integrated_settings.INCLUDE_STANDALONE and integrated_settings.PLOT_LEGACY_ROAD:
        existing_keys = set(runtimes.keys())
        road_pipeline.phase_8_visualization(
            voronoi_tt,
            innerboundary,
            network,
            boundary_plot,
            current_access_points,
            gdf_costs,
            runtimes,
        )
        prefix_new_runtime_keys(runtimes, existing_keys, "road")

    score_exports = scoring_registry.export_score_results()
    comparison_df = build_integrated_comparison_df(rail_results, road_results)
    report_path = write_integrated_run_report(runtimes, shared_scenarios_result)

    if integrated_settings.PLOT_INTEGRATED:
        run_integrated_plots()

    return score_exports, comparison_df, report_path


def run_integrated_pipeline():
    """Run the integrated rail-road workflow from setup to final exports."""
    runtimes = {}

    # MODULE 0: shared setup, preprocessing, and network preparation
    print_integrated_module(0, "Configuration, setup, and network preparation")
    setup_results = setup_integrated_inputs(runtimes)

    # MODULE 1: infrastructure developments
    print_integrated_module(1, "Infrastructure developments")
    development_results = generate_infrastructure_developments(
        setup_results,
        runtimes,
    )

    # MODULE 2: shared scenario generation
    print_integrated_module(2, "Shared scenario generation")
    shared_scenarios_result = run_shared_scenario_generation(
        runtimes,
        development_results["limits_variables"],
    )

    # MODULE 3.1: rail scenario evaluation
    print_integrated_module("3.1", "Rail scenario evaluation")
    rail_results = run_rail_evaluation(
        points=development_results["points"],
        dev_id_lookup=development_results["dev_id_lookup"],
        runtimes=runtimes,
    )

    # MODULE 3.2: road scenario evaluation
    print_integrated_module("3.2", "Road scenario evaluation")
    road_outputs = run_road_evaluation(
        limits_variables=development_results["limits_variables"],
        runtimes=runtimes,
    )
    road_results = {
        "gdf_costs": road_outputs["gdf_costs"],
    }

    # MODULE 3.3: optional legacy outputs plus integrated exports and plots
    print_integrated_module("3.3", "Outputs, exports, and plots")
    score_exports, comparison_df, report_path = finalize_integrated_outputs(
        runtimes=runtimes,
        rail_results=rail_results,
        road_results=road_results,
        shared_scenarios_result=shared_scenarios_result,
        G_development=rail_results["G_development"],
        G_status_quo=rail_results["G_status_quo"],
        dev_id_lookup=development_results["dev_id_lookup"],
        voronoi_tt=road_outputs["voronoi_tt"],
        innerboundary=development_results["innerboundary"],
        network=development_results["network"],
        boundary_plot=development_results["boundary_plot"],
        current_access_points=development_results["current_access_points"],
        gdf_costs=road_outputs["gdf_costs"],
    )

    return {
        "runtimes": runtimes,
        "innerboundary": development_results["innerboundary"],
        "outerboundary": development_results["outerboundary"],
        "points": development_results["points"],
        "baseline_prep_path": development_results["baseline_prep_path"],
        "baseline_sections_path": development_results["baseline_sections_path"],
        "enhanced_network_label": development_results["enhanced_network_label"],
        "dev_id_lookup": development_results["dev_id_lookup"],
        "capacity_analysis_results": development_results["capacity_analysis_results"],
        "shared_scenarios_result": shared_scenarios_result,
        "comparison_df": comparison_df,
        "score_long": score_exports["score_long"],
        "score_tidy": score_exports["score_tidy"],
        "score_long_path": score_exports["score_long_path"],
        "score_tidy_path": score_exports["score_tidy_path"],
        "report_path": report_path,
    }
