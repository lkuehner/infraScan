# import packages
import os
import time
import pandas as pd

from . import settings as integrated_settings

from infraScan.infraScanRoad import data_import as road_data_import




def phase_1_initialization(runtimes):
    """
    Creates the focus area for the analysis by defining the inner and outer boundaries.
    The boundaries are defined as polygons based on specified coordinates.
    """

    print("\n" + "="*80)
    print("PHASE 1: INITIALIZE VARIABLES")
    print("="*80 + "\n")
    st = time.time()

    
    limits_corridor = [integrated_settings.e_min, integrated_settings.n_min, 
                       integrated_settings.e_max, integrated_settings.n_max]

    # Boudary for plot
    boundary_plot = road_data_import.polygon_from_points(e_min=integrated_settings.e_min+1000, e_max=integrated_settings.e_max-500, 
                                        n_min=integrated_settings.n_min+1000, n_max=integrated_settings.n_max-2000)

    # Get a polygon as limits for the corridor
    innerboundary = road_data_import.polygon_from_points(e_min=integrated_settings.e_min, e_max=integrated_settings.e_max, 
                                        n_min=integrated_settings.n_min, n_max=integrated_settings.n_max)

    # For global operation a margin is added to the boundary
    margin = 3000 # meters
    outerboundary = road_data_import.polygon_from_points(e_min=integrated_settings.e_min, e_max=integrated_settings.e_max, n_min=integrated_settings.n_min, 
                                        n_max=integrated_settings.n_max, margin=margin)

    runtimes["Initialize variables"] = time.time() - st

    return limits_corridor,boundary_plot,innerboundary, outerboundary


# TODO: PHASE 2 DATA IMPORT 

# TODO: PHASE 7 Integrated scenario evaluation
#def collect_rail_results(...): ...
#def collect_road_results(...): ...
#def build_integrated_comparison_df(...): ...


