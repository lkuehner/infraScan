# import packages


from infraScan.infraScanRail import settings as rail_settings



def configure_rail():
    rail_settings.PIPELINE_CONFIG.visualization_mode = "none"
    rail_settings.PIPELINE_CONFIG.grouping_strategy = "baseline"
    rail_settings.capacity_threshold = 2.0
    rail_settings.max_enhancement_iterations = 10