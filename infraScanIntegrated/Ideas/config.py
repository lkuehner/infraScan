from __future__ import annotations

from pathlib import Path
from typing import Any


CONFIG: dict[str, Any] = {
    "run": {
        "run_road": True,
        "run_rail": True,
        "execution_mode": "load-existing",  # "load-existing" | "run-subprocess"
        "generate_plots": False,
        "use_cache": True,
        "save_intermediate": True,
        "timeout_seconds": 0,
        "entrypoints": {
            "road": ["python", "main.py"],
            "rail": ["python", "main.py"],
        },
    },
    "scenarios": {
        "scenario_ids": ["low", "medium", "high"],
        "start_year": 2018,
        "end_year": 2100,
        "valuation_year": 2050,
        "corridor_ids": ["corridor_1"],
    },
    "shared_data": {
        "run_shared_preprocessing_once": True,
        "shared_data_root": "infraScanIntegrated/data/shared_inputs",
        "reuse_population_data": True,
        "reuse_land_cover_data": True,
    },
    "integration": {
        "use_integrated_od_for_road": True,
        "od_source": "integrated",
    },
    "economics": {
        "vot_road": 23.29,
        "vot_rail": 14.43,
        "annualization_tau": 0.1,
        "peak_hours_per_day": 1,
        "working_days_per_year": 250,
        "discount_rate": 0.03,
    },
    "scoring": {
        "normalization_method": "minmax",  # "minmax" | "zscore"
        "score_weight_net_benefit": 0.7,
        "score_weight_cba": 0.3,
        "robust_weight_mean": 0.5,
        "robust_weight_q10": 0.3,
        "robust_weight_regret": 0.2,
    },
    "output": {
        "output_subdir": "outputs",
        "export_runtime_report": True,
        "export_pipeline_summary": True,
        "scenarios_subdir": "data/scenarios",
    },
    "inputs": {
        "road_total_costs": "infraScanRoad/data/costs/total_costs.csv",
        "rail_total_costs_raw": "infraScanRail/data/costs/total_costs_raw.csv",
        "rail_total_costs_summary": "infraScanRail/data/costs/total_costs.csv",
    },
}


def get_config() -> dict[str, Any]:
    return CONFIG


def get_output_dir(module_root: Path) -> Path:
    return module_root / str(CONFIG["output"]["output_subdir"])


def get_annualization_factor() -> float:
    economics = CONFIG["economics"]
    return (
        float(economics["annualization_tau"])
        * float(economics["peak_hours_per_day"])
        * float(economics["working_days_per_year"])
    )
