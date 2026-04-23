import os
import pickle
from typing import Any, Dict, Iterable, List

import pandas as pd

from infraScan.infraScanRoad import random_scenarios as road_random
from infraScan.infraScanRoad import settings as road_settings
from infraScan.infraScanRail import random_scenarios as rail_random
from infraScan.infraScanRail import paths as rail_paths
from infraScan.infraScanRail import settings as rail_settings
from . import paths as integrated_paths
from . import settings as integrated_settings

DEFAULT_SHARED_COMPONENTS_PATH = integrated_paths.SHARED_COMPONENTS_PATH
DEFAULT_SHARED_SUMMARY_PATH = integrated_paths.SHARED_SUMMARY_PATH
DEFAULT_SHARED_SELECTION_PATH = integrated_paths.SHARED_SELECTION_PATH


def _resolve_data_root() -> str:
    candidates = [integrated_paths.MAIN]
    for candidate in candidates:
        if not candidate:
            continue
        normalized = os.path.abspath(str(candidate))
        if os.path.isdir(os.path.join(normalized, "data")):
            return normalized

    cwd = os.getcwd()
    if os.path.isdir(os.path.join(cwd, "data")):
        return cwd

    raise FileNotFoundError(
        "Could not resolve infraScan data root. Expected a folder containing 'data/'. "
        "Set settings.MAIN correctly or run from the data root directory."
    )



def build_shared_scenario_components(
    start_year: int,
    end_year: int,
    num_of_scenarios: int,
) -> Dict[str, Any]:
    
    # Ensure rail population scenario paths are aligned with integrated paths
    rail_paths.POPULATION_SCENARIO_CH_BFS_2055 = integrated_paths.POPULATION_SCENARIO_CH_BFS_2055
    rail_paths.POPULATION_SCENARIO_CANTON_ZH_2050 = integrated_paths.POPULATION_SCENARIO_CANTON_ZH_2050
    rail_paths.POPULATION_SCENARIO_CH_EUROSTAT_2100 = integrated_paths.POPULATION_SCENARIO_CH_EUROSTAT_2100

    bezirk_pop_scenarios = rail_random.get_bezirk_population_scenarios()
    population_scenarios = {
        bezirk: rail_random.generate_population_scenarios(
            df,
            start_year,
            end_year,
            num_of_scenarios,
        )
        for bezirk, df in bezirk_pop_scenarios.items()
    }

    modal_split_road = road_random.generate_modal_split_scenarios(
        avg_growth_rate=getattr(road_settings, "avg_growth_rate", 0.0045),
        start_value=getattr(road_settings, "start_value", 0.731),
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
        start_std_dev=getattr(road_settings, "start_std_dev", 0.015),
        end_std_dev=getattr(road_settings, "end_std_dev", 0.045),
        std_dev_shocks=getattr(road_settings, "std_dev_shocks", 0.02),
    )

    modal_split_rail = rail_random.generate_modal_split_scenarios(
        avg_growth_rate=0.0045,
        start_value=0.209,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
        start_std_dev=0.015,
        end_std_dev=0.045,
        std_dev_shocks=0.02,
    )

    distance_per_person = road_random.generate_distance_per_person_scenarios(
        avg_growth_rate=-0.0027,
        start_value=39.79,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
        start_std_dev=0.005,
        end_std_dev=0.015,
        std_dev_shocks=0.015,
    )

    return {
        "meta": {
            "start_year": start_year,
            "end_year": end_year,
            "num_of_scenarios": num_of_scenarios,
        },
        "population_scenarios": population_scenarios,
        "modal_split_road": modal_split_road,
        "modal_split_rail": modal_split_rail,
        "distance_per_person": distance_per_person,
    }


def save_shared_scenario_components(
    components: Dict[str, Any],
    output_path: str = DEFAULT_SHARED_COMPONENTS_PATH,
) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as file:
        pickle.dump(components, file)
    return output_path


def load_shared_scenario_components(
    input_path: str = DEFAULT_SHARED_COMPONENTS_PATH,
) -> Dict[str, Any]:
    with open(input_path, "rb") as file:
        return pickle.load(file)


def _ensure_output_dir(path: str) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def _scenario_name_to_index(scenario_name: str) -> int:
    return int(str(scenario_name).split("_")[-1]) - 1


def _get_value_for_scenario_year(
    df: pd.DataFrame,
    scenario_idx: int,
    year: int,
    value_column: str,
) -> float:
    match = df[(df["scenario"] == scenario_idx) & (df["year"] == year)]
    if match.empty:
        raise KeyError(
            f"Missing value for scenario={scenario_idx}, year={year}, column={value_column}"
        )
    return float(match[value_column].iloc[0])


def build_shared_scenario_summary(
    components: Dict[str, Any],
    valuation_year: int,
) -> pd.DataFrame:
    meta = components["meta"]
    start_year = int(meta["start_year"])
    num_of_scenarios = int(meta["num_of_scenarios"])

    population_scenarios = components["population_scenarios"]
    modal_split_road = components["modal_split_road"]
    modal_split_rail = components["modal_split_rail"]
    distance_per_person = components["distance_per_person"]

    rows: List[Dict[str, float]] = []
    for scenario_idx in range(num_of_scenarios):
        total_population_start = 0.0
        total_population_valuation = 0.0

        for district_df in population_scenarios.values():
            total_population_start += _get_value_for_scenario_year(
                district_df, scenario_idx, start_year, "population"
            )
            total_population_valuation += _get_value_for_scenario_year(
                district_df, scenario_idx, valuation_year, "population"
            )

        road_modal = _get_value_for_scenario_year(
            modal_split_road, scenario_idx, valuation_year, "modal_split"
        )
        rail_modal = _get_value_for_scenario_year(
            modal_split_rail, scenario_idx, valuation_year, "modal_split"
        )
        distance_value = _get_value_for_scenario_year(
            distance_per_person, scenario_idx, valuation_year, "distance_per_person"
        )

        population_growth_factor = (
            total_population_valuation / total_population_start
            if total_population_start > 0
            else 1.0
        )
        road_demand_proxy = total_population_valuation * road_modal * distance_value
        rail_demand_proxy = total_population_valuation * rail_modal * distance_value

        rows.append(
            {
                "scenario": f"scenario_{scenario_idx + 1}",
                "scenario_idx": scenario_idx,
                "valuation_year": valuation_year,
                "total_population_start": total_population_start,
                "total_population_valuation": total_population_valuation,
                "population_growth_factor": population_growth_factor,
                "road_modal_split": road_modal,
                "rail_modal_split": rail_modal,
                "distance_per_person": distance_value,
                "road_demand_proxy": road_demand_proxy,
                "rail_demand_proxy": rail_demand_proxy,
            }
        )

    summary_df = pd.DataFrame(rows)
    score_cols = [
        "population_growth_factor",
        "road_modal_split",
        "rail_modal_split",
        "distance_per_person",
        "road_demand_proxy",
        "rail_demand_proxy",
    ]
    for col in score_cols:
        summary_df[f"{col}_pct_rank"] = summary_df[col].rank(method="average", pct=True)

    summary_df["shared_future_score"] = summary_df[
        [f"{col}_pct_rank" for col in score_cols]
    ].mean(axis=1)
    return summary_df.sort_values("shared_future_score").reset_index(drop=True)


def select_representative_shared_scenarios(
    summary_df: pd.DataFrame,
    n_representatives: int,
) -> pd.DataFrame:
    n_representatives = max(0, int(n_representatives))
    if n_representatives == 0 or summary_df.empty:
        return summary_df.iloc[0:0].copy()
    if n_representatives >= len(summary_df):
        selected = summary_df.copy()
        selected["selection_order"] = range(1, len(selected) + 1)
        return selected

    sorted_df = summary_df.sort_values("shared_future_score").reset_index(drop=True)
    if n_representatives == 1:
        selected_positions = [len(sorted_df) // 2]
    else:
        selected_positions = [
            round(idx * (len(sorted_df) - 1) / (n_representatives - 1))
            for idx in range(n_representatives)
        ]

    selected_rows = []
    used_positions = set()
    for order, target_pos in enumerate(selected_positions, start=1):
        if target_pos not in used_positions:
            used_positions.add(target_pos)
            chosen_pos = target_pos
        else:
            candidate_positions = sorted(
                range(len(sorted_df)),
                key=lambda pos: (abs(pos - target_pos), pos),
            )
            chosen_pos = next(pos for pos in candidate_positions if pos not in used_positions)
            used_positions.add(chosen_pos)

        row = sorted_df.iloc[[chosen_pos]].copy()
        row["selection_order"] = order
        selected_rows.append(row)

    return pd.concat(selected_rows, ignore_index=True)


def save_shared_scenario_summary(
    summary_df: pd.DataFrame,
    output_path: str = DEFAULT_SHARED_SUMMARY_PATH,
) -> str:
    _ensure_output_dir(output_path)
    summary_df.to_csv(output_path, index=False)
    return output_path


def save_representative_scenario_selection(
    selected_df: pd.DataFrame,
    output_path: str = DEFAULT_SHARED_SELECTION_PATH,
) -> str:
    _ensure_output_dir(output_path)
    selected_df.to_csv(output_path, index=False)
    return output_path


def apply_selected_scenarios_to_mode_settings(
    selected_scenarios: Iterable[str],
) -> List[str]:
    selected = list(selected_scenarios)

    road_settings.travel_time_debug_enabled = True
    road_settings.travel_time_debug_scenarios = selected

    return selected


def generate_and_apply_shared_scenarios(
    start_year: int = 2018,
    end_year: int = 2100,
    num_of_scenarios: int = 100,
    representative_scenarios_count: int | None = None,
    components_path: str = DEFAULT_SHARED_COMPONENTS_PATH,
    summary_path: str = DEFAULT_SHARED_SUMMARY_PATH,
    selection_path: str = DEFAULT_SHARED_SELECTION_PATH,
    run_road: bool = True,
    run_rail: bool = True,
    apply_selection_to_modes: bool = True,
    do_plot: bool = False,
) -> Dict[str, Any]:
    data_root = _resolve_data_root()
    previous_cwd = os.getcwd()

    try:
        os.chdir(data_root)

        components = build_shared_scenario_components(
            start_year=start_year,
            end_year=end_year,
            num_of_scenarios=num_of_scenarios,
        )
        saved_path = save_shared_scenario_components(components, output_path=components_path)
        summary_df = build_shared_scenario_summary(
            components,
            valuation_year=integrated_settings.start_valuation_year,
        )
        summary_file = save_shared_scenario_summary(summary_df, output_path=summary_path)

        if representative_scenarios_count is None:
            representative_scenarios_count = integrated_settings.representative_scenarios_count
        selected_df = select_representative_shared_scenarios(
            summary_df,
            n_representatives=representative_scenarios_count,
        )
        selection_file = save_representative_scenario_selection(
            selected_df,
            output_path=selection_path,
        )
        selected_scenarios = selected_df["scenario"].tolist()

        if apply_selection_to_modes:
            apply_selected_scenarios_to_mode_settings(selected_scenarios)

        if run_road:
            road_random.get_random_scenarios(
                start_year=start_year,
                end_year=end_year,
                num_of_scenarios=num_of_scenarios,
                use_cache=False,
                do_plot=do_plot,
                shared_components_path=saved_path,
            )

        if run_rail:
            rail_random.get_random_scenarios(
                start_year=start_year,
                end_year=end_year,
                num_of_scenarios=num_of_scenarios,
                use_cache=False,
                do_plot=do_plot,
                shared_components_path=saved_path,
            )

        return {
            "components_path": saved_path,
            "summary_path": summary_file,
            "selection_path": selection_file,
            "selected_scenarios": selected_scenarios,
            "selected_summary": selected_df,
        }
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    generated = generate_and_apply_shared_scenarios(
        start_year=2018,
        end_year=2100,
        num_of_scenarios=min(int(road_settings.amount_of_scenarios), int(rail_settings.amount_of_scenarios)),
        run_road=True,
        run_rail=True,
        do_plot=False,
    )
    print(f"Shared scenario components generated at: {generated['components_path']}")
    print(f"Shared scenario summary written to: {generated['summary_path']}")
    print(f"Representative scenarios written to: {generated['selection_path']}")
    print(f"Selected scenarios: {generated['selected_scenarios']}")
