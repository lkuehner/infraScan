from shapely.geometry import Polygon
from . import paths
import re

#Rail network: Choose either 'AK_2035','AK_2035_extended' or 'current' or '2024_extended'
rail_network = 'AK_2035'

#CACHE
use_cache_network = True #Phase 3.1: Baseline network (current + AK2035)
use_cache_pt_catchment = True #Phase 2.2: Public transport catchment areas (OD matrix)
use_cache_developments = True #Phase 4.1: Infrastructure developments (new networks)
use_cache_catchmentOD = False #Phase 5
use_cache_stationsOD = False #Phase 5
use_cache_traveltime_graph = False #Phase 6: travel time graph with developments
use_cache_scenarios = True #Phase 8: Scenario generation (OD matrices for generated scenarios)
use_cache_tts_calc = True #Phase 9: Travel time savings calculation

# Infrastructure generation modules: Choose either 'EXTEND_LINES', 'NEW_DIRECT_CONNECTIONS' or 'ALL'
infra_generation_modification_type = 'ALL' 
# Rail services: Choose either 'RAIL_SERVICES_AK2035_PATH' or 'RAIL_SERVICES_AK2035_EXTENDED_PATH' or 'RAIL_SERVICES_2024_PATH' or 'RAIL_SERVICES_AK2024_EXTENDED_PATH'
infra_generation_rail_network = paths.RAIL_SERVICES_AK2035_PATH

OD_type = 'canton_ZH' #either 'canton_ZH' or 'pt_catchment_perimeter'
only_demand_from_to_perimeter = True

# Scenario settings: Choose either 'GENERATED' or 'STATIC_9' or 'dummy'
scenario_type = 'GENERATED' 
amount_of_scenarios = 100
start_year_scenario = 2018
end_year_scenario = 2100
start_valuation_year = 2050
#choose which OD

plot_passenger_flow = True
plot_railway_line_load = True




perimeter_infra_generation = Polygon([  #No GeoJSON with this polygon type!
    (2700989.862, 1235663.403),
    (2708491.515, 1239608.529),
    (2694972.602, 1255514.900),
    (2687415.817, 1251056.404)  # closing the polygon
])
perimeter_demand = perimeter_infra_generation



raster_size = (170,210)

pop_scenarios = [
        "pop_urban_", "pop_equal_", "pop_rural_",
        "pop_urba_1", "pop_equa_1", "pop_rura_1",
        "pop_urba_2", "pop_equa_2", "pop_rura_2"]
empl_scenarios = ["empl_urban", "empl_equal", "empl_rural",
                   "empl_urb_1", "empl_equ_1", "empl_rur_1",
                   "empl_urb_2", "empl_equ_2", "empl_rur_2"]

dev_id_start_extended_lines = 100000
dev_id_start_new_direct_connections = 101000

# ═══════════════════════════════════════════════════════════════════════════════
# CAPACITY ANALYSIS SETTINGS (for main_cap.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Phase 3.2: Baseline Capacity
capacity_threshold = 2.0  # Minimum available capacity (trains per hour per direction)
visualize_capacity_analysis = True  # Create plots during capacity analysis

# Phase 3.3: Baseline Enhancement
max_enhancement_iterations = 10  # Max Phase 4 enhancement iterations

# Internal (set dynamically in Phase 3.3)
baseline_network_for_developments = None  # Will be set to enhanced network label (e.g., "2024_extended_enhanced")



# ================================================================================
# GLOBAL PIPELINE CONFIGURATION
# ================================================================================

class PipelineConfig:
    """Global configuration for pipeline execution."""
    
    def __init__(self):
        self.visualization_mode = None  # 'manual', 'none', 'all'
        self.grouping_strategy = None   # 'manual', 'conservative', 'baseline', 'optimal'
        self._original_input = None
    
    def should_generate_plots(self, default_yes: bool = False) -> bool:
        """
        Determine if plots should be generated based on global setting.
        
        Args:
            default_yes: Default answer if mode is manual
        
        Returns:
            bool: True if plots should be generated, None if should prompt
        """
        if self.visualization_mode == 'all':
            return True
        elif self.visualization_mode == 'none':
            return False
        else:  # manual
            return None  # Signal to prompt user
    
    def get_grouping_choice(self, prompt: str) -> str:
        """
        Get grouping strategy choice based on global setting.
        
        Detects available choices from the prompt and selects appropriately.
        
        Args:
            prompt: The original prompt text
        
        Returns:
            str: The choice to make
        """
        if self.grouping_strategy == 'manual':
            # Use original input
            return self._original_input(prompt)
        
        # Detect available choices from prompt
        # Look for patterns like "(1-2)" or "(1-3)" in the prompt      
        choice_match = re.search(r'\(1-(\d+)\)', prompt) 
        if choice_match:
            max_choice = int(choice_match.group(1))
            available_choices = [str(i) for i in range(1, max_choice + 1)]
        else:
            # Fallback: assume 1-2 if we can't detect
            available_choices = ['1', '2']
        
        # Select based on strategy
        if self.grouping_strategy == 'conservative':
            # Always choose lowest option (1)
            choice = '1'
            print(prompt + f"{choice}  [AUTO-SELECTED: Conservative]")
        elif self.grouping_strategy == 'baseline':
            # Always choose option 2 if available, otherwise lowest
            choice = '2' if '2' in available_choices else '1'
            print(prompt + f"{choice}  [AUTO-SELECTED: Baseline]")
        elif self.grouping_strategy == 'optimal':
            # Always choose highest option
            choice = max(available_choices, key=int)
            print(prompt + f"{choice}  [AUTO-SELECTED: Optimal]")
        else:
            # Fallback to manual
            return self._original_input(prompt)
        
        return choice


# Global configuration instance
PIPELINE_CONFIG = PipelineConfig()
