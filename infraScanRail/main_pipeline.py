# import packages
import os
import warnings

from . import paths
from . import settings
from .pipeline import (
    phase_1_initialization,
    phase_2_data_import,
    phase_3_baseline_capacity_analysis,
    phase_4_infrastructure_developments,
    phase_5_demand_analysis,
    phase_6_travel_time_computation,
    phase_7_passenger_flow_visualization,
    phase_8_scenario_generation,
    phase_9_travel_time_savings,
    phase_10_old_construction_maintenance_costs,
    phase_10_new_construction_maintenance_costs,
    phase_11_old_cost_benefit_integration,
    phase_11_new_cost_benefit_integration,
    phase_12_old_cost_aggregation,
    phase_12_new_cost_aggregation,
    phase_13_results_visualization,
)


# ================================================================================
# MAIN FUNCTION
# ================================================================================

def infrascanrail_cap():
    """
    Enhanced InfraScanRail main pipeline with integrated capacity analysis.

    This orchestrator sequentially calls all 13 phases of the capacity-enhanced pipeline.
    Each phase is now encapsulated as a separate function for easier debugging and testing.
    """
    os.chdir(paths.MAIN)
    warnings.filterwarnings("ignore")  # TODO: No warnings should be ignored
    runtimes = {}

    # ============================================================================
    # CONFIGURABLE AUTO-RESPONSE SETUP
    # ============================================================================
    # Store original input function
    if isinstance(__builtins__, dict):
        _original_input = __builtins__['input']
    else:
        _original_input = __builtins__.input
    
    # Store in config for later use
    settings.PIPELINE_CONFIG._original_input = _original_input

    # ============================================================================
    # GLOBAL CONFIGURATION PROMPTS
    # ============================================================================
    print("PIPELINE CONFIGURATION")
    print("-" * 80)
    
    # A. Visualization Strategy
    print("\nA. VISUALIZATION STRATEGY")
    print("   How should the pipeline handle optional plot generation?")
    print("   1) Manual  - Prompt for each visualization decision (default)")
    print("   2) None    - Skip all optional visualizations")
    print("   3) All     - Generate all optional visualizations")
    
    while True:
        viz_choice = input("\n   Select visualization mode (1-3) [1]: ").strip() or "1"

        if viz_choice in ['1', '2', '3']:
            break
        print("   Invalid selection. Please enter 1, 2, or 3.")
    
    viz_modes = {'1': 'manual', '2': 'none', '3': 'all'}
    settings.PIPELINE_CONFIG.visualization_mode = viz_modes[viz_choice]
    print(f"   → Visualization mode: {settings.PIPELINE_CONFIG.visualization_mode.upper()}")
    
    # B. Grouping Strategy
    print("\nB. CAPACITY GROUPING STRATEGY")
    print("   How should capacity grouping decisions be made?")
    print("   1) Manual       - Prompt for each grouping decision (default)")
    print("   2) Conservative - Always choose lowest capacity option")
    print("   3) Baseline     - Always choose middle option (2)")
    print("   4) Optimal      - Always choose highest capacity option")
    
    while True:
        group_choice = input("\n   Select grouping strategy (1-4) [1]: ").strip() or "1"
        if group_choice in ['1', '2', '3', '4']:
            break
        print("   Invalid selection. Please enter 1, 2, 3, or 4.")
    
    group_modes = {'1': 'manual', '2': 'conservative', '3': 'baseline', '4': 'optimal'}
    settings.PIPELINE_CONFIG.grouping_strategy = group_modes[group_choice]
    print(f"   → Grouping strategy: {settings.PIPELINE_CONFIG.grouping_strategy.upper()}")
    
    # C. Capacity Enhancement Parameters
    print(f"\nC. Configure capacity enhancement parameters:")
    print(f"  Default threshold: {settings.capacity_threshold} tphpd")
    print(f"  Default max iterations: {settings.max_enhancement_iterations}")

    # Get threshold from user
    threshold_input = input(f"\n  Enter capacity threshold (tphpd) or press Enter for default [{settings.capacity_threshold}]: ").strip()
    if threshold_input:
        try:
            settings.capacity_threshold = float(threshold_input)
            print(f"  → Using threshold: {settings.capacity_threshold} tphpd")
        except ValueError:
            print(f"  ⚠ Invalid input. Using default: {settings.capacity_threshold} tphpd")
    else:
        settings.capacity_threshold = settings.capacity_threshold
        print(f"  → Using default threshold: {settings.capacity_threshold} tphpd")

    # Get max iterations from user
    iterations_input = input(f"  Enter max iterations or press Enter for default [{settings.max_enhancement_iterations}]: ").strip()
    if iterations_input:
        try:
            settings.max_enhancement_iterations = int(iterations_input)
            print(f"  → Using max iterations: {settings.max_enhancement_iterations}")
        except ValueError:
            print(f"  ⚠ Invalid input. Using default: {settings.max_enhancement_iterations}")
            settings.max_enhancement_iterations = settings.max_enhancement_iterations
    else:
        settings.max_enhancement_iterations = settings.max_enhancement_iterations
        print(f"  → Using default max iterations: {settings.max_enhancement_iterations}")


    print("-" * 80)
    print("✓ Pipeline configuration complete\n")

    def smart_input(prompt=""):
        """
        Intelligently handle input based on global configuration.
        - Capacity grouping prompts: Use grouping strategy
        - Visualization prompts: Use visualization strategy
        - All other prompts: Use original input
        """
        # Check if this is a capacity grouping prompt
        if ("Enter choice (1-" in prompt or 
            "Select the strategy number" in prompt or
            "Select track allocation strategy" in prompt):
            return settings.PIPELINE_CONFIG.get_grouping_choice(prompt)
        
        # For all other prompts, use original input
        return _original_input(prompt)

    # Replace built-in input with our smart input function
    if isinstance(__builtins__, dict):
        __builtins__['input'] = smart_input
    else:
        __builtins__.input = smart_input

    ##################################################################################
    # PHASE 1: INITIALIZATION & CONFIGURATION
    ##################################################################################
    innerboundary, outerboundary = phase_1_initialization(runtimes)

    ##################################################################################
    # PHASE 2: DATA IMPORT
    ##################################################################################
    phase_2_data_import(runtimes)

    ##################################################################################
    # PHASE 3: BASELINE CAPACITY ANALYSIS
    ##################################################################################
    points, baseline_prep_path, baseline_sections_path, enhanced_network_label = \
        phase_3_baseline_capacity_analysis(runtimes)
 
    ##################################################################################
    # PHASE 4: INFRASTRUCTURE DEVELOPMENTS
    ##################################################################################
    dev_id_lookup, capacity_analysis_results = \
        phase_4_infrastructure_developments(points, runtimes)

    ##################################################################################
    # PHASE 5: DEMAND ANALYSIS (OD MATRIX)
    ##################################################################################
    phase_5_demand_analysis(points, runtimes)

    ##################################################################################
    # PHASE 6: TRAVEL TIME COMPUTATION
    ##################################################################################
    od_times_dev, od_times_status_quo, G_status_quo, G_development = \
        phase_6_travel_time_computation(dev_id_lookup, runtimes)

    ##################################################################################
    # PHASE 7: PASSENGER FLOW VISUALIZATION
    ##################################################################################
    if settings.plot_passenger_flow:
        phase_7_passenger_flow_visualization(
            G_development, G_status_quo, dev_id_lookup, runtimes
        )

    ##################################################################################
    # PHASE 8: SCENARIO GENERATION
    ##################################################################################
    if settings.OD_type == 'canton_ZH':
        phase_8_scenario_generation(runtimes)

    ##################################################################################
    # PHASE 9: TRAVEL TIME SAVINGS
    ##################################################################################
    dev_list, monetized_tt, scenario_list = \
        phase_9_travel_time_savings(
            dev_id_lookup, od_times_dev, od_times_status_quo, runtimes
        )

    ##################################################################################
    # PHASE 10 OLD: CONSTRUCTION & MAINTENANCE COSTS (8 trains/track)
    ##################################################################################
    #phase_10_old_construction_maintenance_costs(monetized_tt, runtimes)

    ##################################################################################
    # PHASE 11 OLD: COST-BENEFIT INTEGRATION (8 trains/track)
    ##################################################################################
    #phase_11_old_cost_benefit_integration(runtimes)

    ##################################################################################
    # PHASE 12 OLD: COST AGGREGATION (8 trains/track)
    ##################################################################################
    #phase_12_old_cost_aggregation(runtimes)

    ##################################################################################
    # PHASE 10 NEW: CONSTRUCTION & MAINTENANCE COSTS (Capacity Interventions)
    ##################################################################################
    phase_10_new_construction_maintenance_costs(monetized_tt, runtimes)

    ##################################################################################
    # PHASE 11 NEW: COST-BENEFIT INTEGRATION (Capacity Interventions)
    ##################################################################################
    phase_11_new_cost_benefit_integration(runtimes)

    ##################################################################################
    # PHASE 12 NEW: COST AGGREGATION (Capacity Interventions)
    ##################################################################################
    phase_12_new_cost_aggregation(runtimes)

    ##################################################################################
    # PHASE 13: RESULTS VISUALIZATION
    ##################################################################################
    phase_13_results_visualization(runtimes)
  
    ##################################################################################
    # SAVE RUNTIMES
    ##################################################################################
    with open('runtimes_cap.txt', 'w') as file:
        file.write("=" * 80 + "\n")
        file.write("INFRASCANRAIL CAPACITY-ENHANCED PIPELINE RUNTIMES\n")
        file.write("=" * 80 + "\n\n")
        total_time = sum(runtimes.values())
        for part, runtime in runtimes.items():
            mins = int(runtime // 60)
            secs = int(runtime % 60)
            file.write(f"{part:.<50} {mins}m {secs}s ({runtime:.2f}s)\n")
        file.write("\n" + "=" * 80 + "\n")
        total_mins = int(total_time // 60)
        total_secs = int(total_time % 60)
        file.write(f"{'TOTAL TIME':.<50} {total_mins}m {total_secs}s ({total_time:.2f}s)\n")
        file.write("=" * 80 + "\n")

    # Restore original input function
    if isinstance(__builtins__, dict):
        __builtins__['input'] = _original_input
    else:
        __builtins__.input = _original_input

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTotal runtime: {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"Runtimes saved to: runtimes_cap.txt\n")

# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == '__main__':
    infrascanrail_cap()