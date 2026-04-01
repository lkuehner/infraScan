# import packages
import os
import warnings


from . import settings
from .pipeline import (
    phase_1_initialization,
    phase_2_data_import,
    phase_3_infrastructure_developments,
    phase_4_scenario_generation,
    phase_5_costs_and_accesibility,
    phase_6_travel_time_savings,
    phase_7_aggregation,
    phase_8_visualization
)


# ================================================================================
# MAIN FUNCTION
# ================================================================================

def infrascanroad():
    """
    Enhanced InfraScanRoad main pipeline.
    
    This orchestrator sequentially calls all 8 phases of the road pipeline.
    Each phase is now encapsulated as a separate function for easier debugging and testing.
    """
    module_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(module_dir)
    os.chdir(project_root)

    warnings.filterwarnings("ignore")  # TODO: No warnings should be ignored
    runtimes = {}


    ##################################################################################
    # PHASE 1: INITIALIZATION & CONFIGURATION
    ##################################################################################
    limits_corridor, boundary_plot, innerboundary, outerboundary = \
        phase_1_initialization(runtimes)

    ##################################################################################
    # PHASE 2: DATA IMPORT
    ##################################################################################
    phase_2_data_import(limits_corridor, runtimes)


    ##################################################################################
    # PHASE 3: INFRASTRUCTURE DEVELOPMENTS
    ##################################################################################
    network, limits_variables, generated_points, current_points, current_access_points = \
        phase_3_infrastructure_developments(innerboundary, runtimes)  


    ##################################################################################
    # PHASE 4: SCENARIO GENERATION
    ##################################################################################
    phase_4_scenario_generation(limits_variables, runtimes)

    ##################################################################################
    # PHASE 5: Costs and Accessibility
    ##################################################################################
    voronoi_tt = phase_5_costs_and_accesibility(limits_variables, runtimes)


    ##################################################################################
    # PHASE 6: Travel Time Savings
    ##################################################################################
    phase_6_travel_time_savings(runtimes)
    
    ##################################################################################
    # PHASE 7: Aggregation
    ##################################################################################
    gdf_costs =phase_7_aggregation(runtimes)
    
    ##################################################################################
    # PHASE 8: Visualization
    ##################################################################################
    phase_8_visualization(voronoi_tt, innerboundary, network, boundary_plot,
                          current_access_points, gdf_costs, 
                          runtimes)
    
    
    ##################################################################################
    # SAVE RUNTIMES
    ##################################################################################
    with open('runtimes.txt', 'w') as file:
        file.write("=" * 80 + "\n")
        file.write("INFRASCANROAD PIPELINE RUNTIMES\n")
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


    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\nTotal runtime: {int(total_time // 60)}m {int(total_time % 60)}s")
    print(f"Runtimes saved to: runtimes.txt\n")

# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == '__main__':
    infrascanroad()