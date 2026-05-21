import os
import pickle
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, qmc

from infraScan.infraScanRoad import settings as road_settings
from .. import paths as integrated_paths
from .. import settings as integrated_settings


DEFAULT_SHARED_COMPONENTS_PATH = integrated_paths.SHARED_COMPONENTS_PATH
DEFAULT_SHARED_SUMMARY_PATH = integrated_paths.SHARED_SUMMARY_PATH
DEFAULT_SHARED_SELECTION_PATH = integrated_paths.SHARED_SELECTION_PATH


def _shared_impl():
    from . import random_scenarios as shared_impl
    return shared_impl


def calculate_modal_split_growth_index(
    scenarios_df: pd.DataFrame,
    start_year: int,
) -> pd.DataFrame:
    """
    Finalize a modal split scenario table by calculating growth rates and indices.
    """
    finalized = scenarios_df.copy()
    finalized["modal_split"] = (
        pd.to_numeric(finalized["modal_split"], errors="coerce")
        .clip(lower=0.0, upper=1.0)
    )
    finalized = finalized.sort_values(["scenario", "year"]).reset_index(drop=True)
    finalized["growth_rate"] = (
        finalized.groupby("scenario")["modal_split"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    start_values = (
        finalized[finalized["year"] == start_year][["scenario", "modal_split"]]
        .rename(columns={"modal_split": "start_modal_split"})
    )
    finalized = finalized.merge(start_values, on="scenario", how="left")
    finalized["growth_index"] = np.where(
        finalized["start_modal_split"].fillna(0.0) > 0.0,
        100.0 * finalized["modal_split"] / finalized["start_modal_split"],
        100.0,
    )
    return finalized.drop(columns=["start_modal_split"])


def get_bezirk_population_scenarios() -> Dict[str, pd.DataFrame]:
    """
    Rebuild the district-level population reference tables used by the shared
    scenario generator.
    """
    df_ch = pd.read_csv(integrated_paths.POPULATION_SCENARIO_CH_BFS_2055, sep=",")
    pop_2018 = df_ch.loc[df_ch["Jahr"] == 2018, "Beobachtungen"].values
    pop_2050 = df_ch.loc[df_ch["Jahr"] == 2050, "Referenzszenario A-00-2025"].values
    swiss_growth_factor_18_50 = pop_2050[0] / pop_2018[0]

    df = pd.read_csv(integrated_paths.POPULATION_SCENARIO_CANTON_ZH_2050, sep=";", encoding="utf-8")
    population_summary = (
        df.groupby(["bezirk", "jahr"])["anzahl"]
        .sum()
        .reset_index()
        .rename(columns={"anzahl": "total_population"})
    )

    all_years = pd.Series(range(2011, 2051), name="jahr")
    all_districts = population_summary["bezirk"].unique()
    full_index = pd.MultiIndex.from_product([all_districts, all_years], names=["bezirk", "jahr"])
    population_complete = (
        population_summary.set_index(["bezirk", "jahr"])
        .reindex(full_index)
        .fillna({"total_population": 0})
        .reset_index()
    )
    population_complete["growth_rate"] = (
        population_complete.sort_values(["bezirk", "jahr"]).groupby("bezirk")["total_population"].pct_change()
    )

    district_tables = {}
    for district, group in population_complete.groupby("bezirk"):
        group = group.reset_index(drop=True)
        district_pop_2018 = group.loc[group["jahr"] == 2018, "total_population"].values
        district_pop_2050 = group.loc[group["jahr"] == 2050, "total_population"].values
        growth_factor_18_50 = district_pop_2050[0] / district_pop_2018[0]
        relative_growth = (growth_factor_18_50 - 1) / (swiss_growth_factor_18_50 - 1)
        yearly_growth_factor = relative_growth ** (1 / 32)
        group.attrs["yearly_growth_factor_district_to_CH"] = yearly_growth_factor
        district_tables[district] = group

    eurostat_df = pd.read_excel(integrated_paths.POPULATION_SCENARIO_CH_EUROSTAT_2100)
    eurostat_df.columns = eurostat_df.columns.map(str)
    growth_rate_row = eurostat_df[eurostat_df["unit"] == "GROWTH_RATE"]
    year_columns = [str(year) for year in range(2051, 2101)]
    ch_growth_rates = growth_rate_row[year_columns].iloc[0].astype(float)

    for district, df_district in district_tables.items():
        last_population = df_district.loc[df_district["jahr"] == 2050, "total_population"].values[0]
        scaling_factor = df_district.attrs["yearly_growth_factor_district_to_CH"]
        current_population = last_population
        new_rows = []
        for year in range(2051, 2101):
            adjusted_growth_rate = ch_growth_rates[str(year)] * scaling_factor
            current_population *= (1 + adjusted_growth_rate)
            new_rows.append(
                {
                    "bezirk": district,
                    "jahr": year,
                    "total_population": current_population,
                    "growth_rate": adjusted_growth_rate,
                }
            )
        extension_df = pd.DataFrame(new_rows)
        district_tables[district] = pd.concat([df_district, extension_df], ignore_index=True).reset_index(drop=True)
    return district_tables


def generate_population_scenarios(
    ref_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    n_scenarios: int = 1000,
    start_std_dev: float = 0.01,
    end_std_dev: float = 0.03,
    std_dev_shocks: float = 0.02,
) -> pd.DataFrame:
    """
    Generate one-dimensional scenarios using the same core idea as infraScanRail:
    deterministic growth, LHS perturbations on yearly growth rates, and
    cumulative yearly shocks.
    """
    ref_df = ref_df.sort_values("jahr")
    ref_df = ref_df[(ref_df["jahr"] >= start_year) & (ref_df["jahr"] <= end_year)].reset_index(drop=True)
    years = ref_df["jahr"].values
    ref_growth = ref_df["growth_rate"].values
    n_years = len(years)
    initial_population = ref_df.loc[ref_df["jahr"] == start_year, "total_population"].values[0]

    growth_std_devs = np.linspace(start_std_dev, end_std_dev, n_years)
    sampler = qmc.LatinHypercube(d=n_years, seed=42)
    lhs_samples = sampler.random(n=n_scenarios)
    growth_perturbations = norm.ppf(lhs_samples) * growth_std_devs

    scenario_growth = ref_growth + growth_perturbations
    scenario_growth[:, 0] = 0.0

    shock_sampler = qmc.LatinHypercube(d=n_years, seed=43)
    lhs_shocks = shock_sampler.random(n=n_scenarios)
    et = norm.ppf(lhs_shocks) * std_dev_shocks
    et[:, 0] = 0.0

    cumulative_shocks = np.cumsum(et, axis=1)
    deterministic_growth = np.cumprod(1.0 + scenario_growth, axis=1)
    population_index = deterministic_growth + cumulative_shocks
    pop_scenarios = initial_population * population_index

    scenario_data = []
    for scenario_idx in range(n_scenarios):
        for year_idx, year in enumerate(years):
            if year_idx == 0:
                effective_growth_rate = 0.0
            else:
                effective_growth_rate = (pop_scenarios[scenario_idx, year_idx] / pop_scenarios[scenario_idx, year_idx - 1]) - 1.0
            scenario_data.append(
                {
                    "scenario": scenario_idx,
                    "year": int(year),
                    "population": float(pop_scenarios[scenario_idx, year_idx]),
                    "growth_rate": float(effective_growth_rate),
                    "growth_index": float(100.0 * pop_scenarios[scenario_idx, year_idx] / initial_population),
                }
            )
    return pd.DataFrame(scenario_data)


def generate_distance_per_person_scenarios(
    avg_growth_rate: float,
    start_value: float,
    start_year: int,
    end_year: int,
    n_scenarios: int = 1000,
    start_std_dev: float = 0.01,
    end_std_dev: float = 0.03,
    std_dev_shocks: float = 0.02,
) -> pd.DataFrame:
    """
    Generate distance-per-person scenarios using the same one-dimensional setup
    as the rail random scenario generator.
    """
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)
    growth_factors = np.ones(n_years) * (1.0 + avg_growth_rate)
    growth_factors[0] = 1.0
    cumulative_growth = np.cumprod(growth_factors)
    distance_values = start_value * cumulative_growth
    growth_rates = np.zeros(n_years)
    growth_rates[1:] = avg_growth_rate

    ref_df = pd.DataFrame(
        {
            "jahr": years,
            "total_population": distance_values,
            "growth_rate": growth_rates,
        }
    )
    scenarios_df = generate_population_scenarios(
        ref_df=ref_df,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
        start_std_dev=start_std_dev,
        end_std_dev=end_std_dev,
        std_dev_shocks=std_dev_shocks,
    )
    return scenarios_df.rename(columns={"population": "distance_per_person"})


def _build_reference_modal_split_paths(
    start_year: int,
    end_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    years = np.arange(start_year, end_year + 1)
    year_offset = years - start_year
    raw_reference = np.column_stack(
        [
            integrated_settings.rail_modal_split_start
            * ((1.0 + integrated_settings.rail_modal_split_avg_growth_rate) ** year_offset),
            integrated_settings.road_modal_split_start
            * ((1.0 + integrated_settings.road_modal_split_avg_growth_rate) ** year_offset),
            integrated_settings.other_modal_split_start
            * ((1.0 + integrated_settings.other_modal_split_avg_growth_rate) ** year_offset),
        ]
    )
    reference_shares = raw_reference / raw_reference.sum(axis=1, keepdims=True)
    return years, reference_shares


def _shares_to_log_ratios(shares: np.ndarray) -> np.ndarray:
    safe = np.clip(shares, 1e-12, 1.0)
    return np.column_stack(
        [
            np.log(safe[:, 0] / safe[:, 1]),
            np.log(safe[:, 2] / safe[:, 1]),
        ]
    )


def _log_ratios_to_shares(latent_paths: np.ndarray) -> np.ndarray:
    exp_rail = np.exp(np.clip(latent_paths[:, :, 0], -20.0, 20.0))
    exp_other = np.exp(np.clip(latent_paths[:, :, 1], -20.0, 20.0))
    denominator = 1.0 + exp_rail + exp_other
    rail_share = exp_rail / denominator
    road_share = 1.0 / denominator
    other_share = exp_other / denominator
    return np.stack([rail_share, road_share, other_share], axis=2)


def _matrix_to_dataframe(values: np.ndarray, years: np.ndarray) -> pd.DataFrame:
    rows = []
    n_scenarios, n_years = values.shape
    for scenario_idx in range(n_scenarios):
        for year_idx in range(n_years):
            rows.append(
                {
                    "scenario": scenario_idx,
                    "year": int(years[year_idx]),
                    "modal_split": float(values[scenario_idx, year_idx]),
                }
            )
    return pd.DataFrame(rows)


def generate_joint_modal_split_scenarios_simple(
    start_year: int,
    end_year: int,
    n_scenarios: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate stochastic joint modal split scenarios for rail, road and other
    using the same rail-style ingredients:

    - deterministic reference path
    - LHS perturbations of yearly growth in a two-dimensional state space
    - cumulative yearly shocks

    Because the three shares must sum to 1, only two independent log-ratios are
    simulated: log(rail / road) and log(other / road).
    """
    years, reference_shares = _build_reference_modal_split_paths(start_year, end_year)
    reference_latent = _shares_to_log_ratios(reference_shares)
    n_years = len(years)

    rail_std_path = np.linspace(
        integrated_settings.rail_modal_split_start_std_dev,
        integrated_settings.rail_modal_split_end_std_dev,
        n_years,
    )
    road_std_path = np.linspace(
        integrated_settings.road_modal_split_start_std_dev,
        integrated_settings.road_modal_split_end_std_dev,
        n_years,
    )
    other_std_path = np.linspace(
        integrated_settings.other_modal_split_start_std_dev,
        integrated_settings.other_modal_split_end_std_dev,
        n_years,
    )

    ratio_std_path = np.column_stack(
        [
            np.sqrt(rail_std_path ** 2 + road_std_path ** 2),
            np.sqrt(other_std_path ** 2 + road_std_path ** 2),
        ]
    )
    ratio_shock_std = np.array(
        [
            np.sqrt(
                integrated_settings.rail_modal_split_std_dev_shocks ** 2
                + integrated_settings.road_modal_split_std_dev_shocks ** 2
            ),
            np.sqrt(
                integrated_settings.other_modal_split_std_dev_shocks ** 2
                + integrated_settings.road_modal_split_std_dev_shocks ** 2
            ),
        ],
        dtype=float,
    )

    sampler = qmc.LatinHypercube(d=2 * n_years, seed=42)
    lhs_samples = sampler.random(n=n_scenarios)
    growth_perturbations = norm.ppf(lhs_samples).reshape(n_scenarios, n_years, 2)
    growth_perturbations *= ratio_std_path[None, :, :]
    growth_perturbations[:, 0, :] = 0.0

    shock_sampler = qmc.LatinHypercube(d=2 * n_years, seed=43)
    lhs_shocks = shock_sampler.random(n=n_scenarios)
    et = norm.ppf(lhs_shocks).reshape(n_scenarios, n_years, 2)
    et *= ratio_shock_std[None, None, :]
    et[:, 0, :] = 0.0

    cumulative_shocks = np.cumsum(et, axis=1)
    deterministic_latent = np.broadcast_to(reference_latent[None, :, :], (n_scenarios, n_years, 2))
    latent_paths = deterministic_latent + growth_perturbations + cumulative_shocks

    shares = _log_ratios_to_shares(latent_paths)
    shares[:, 0, 0] = integrated_settings.rail_modal_split_start
    shares[:, 0, 1] = integrated_settings.road_modal_split_start
    shares[:, 0, 2] = integrated_settings.other_modal_split_start

    rail_df = calculate_modal_split_growth_index(_matrix_to_dataframe(shares[:, :, 0], years), start_year=start_year)
    road_df = calculate_modal_split_growth_index(_matrix_to_dataframe(shares[:, :, 1], years), start_year=start_year)
    other_df = calculate_modal_split_growth_index(_matrix_to_dataframe(shares[:, :, 2], years), start_year=start_year)
    return rail_df, road_df, other_df


def build_shared_scenario_components(
    start_year: int,
    end_year: int,
    num_of_scenarios: int,
) -> Dict[str, Any]:
    bezirk_pop_scenarios = get_bezirk_population_scenarios()
    population_scenarios = {
        bezirk: generate_population_scenarios(df, start_year, end_year, num_of_scenarios)
        for bezirk, df in bezirk_pop_scenarios.items()
    }
    modal_split_rail, modal_split_road, modal_split_other = generate_joint_modal_split_scenarios_simple(
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
    )
    distance_per_person = generate_distance_per_person_scenarios(
        avg_growth_rate=integrated_settings.distance_per_person_avg_growth_rate,
        start_value=integrated_settings.distance_per_person_start,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=num_of_scenarios,
        start_std_dev=integrated_settings.distance_per_person_start_std_dev,
        end_std_dev=integrated_settings.distance_per_person_end_std_dev,
        std_dev_shocks=integrated_settings.distance_per_person_std_dev_shocks,
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
        "modal_split_other": modal_split_other,
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


def _get_value_for_scenario_year(
    df: pd.DataFrame,
    scenario_idx: int,
    year: int,
    value_column: str,
) -> float:
    match = df[(df["scenario"] == scenario_idx) & (df["year"] == year)]
    if match.empty:
        raise KeyError(f"Missing value for scenario={scenario_idx}, year={year}, column={value_column}")
    return float(match[value_column].iloc[0])


def build_shared_scenario_summary(
    components: Dict[str, Any],
    valuation_year: int,
) -> pd.DataFrame:
    meta = components["meta"]
    start_year = int(meta["start_year"])
    num_of_scenarios = int(meta["num_of_scenarios"])
    rows: List[Dict[str, float]] = []

    for scenario_idx in range(num_of_scenarios):
        total_population_start = 0.0
        total_population_valuation = 0.0
        for district_df in components["population_scenarios"].values():
            total_population_start += _get_value_for_scenario_year(district_df, scenario_idx, start_year, "population")
            total_population_valuation += _get_value_for_scenario_year(district_df, scenario_idx, valuation_year, "population")

        road_modal = _get_value_for_scenario_year(components["modal_split_road"], scenario_idx, valuation_year, "modal_split")
        rail_modal = _get_value_for_scenario_year(components["modal_split_rail"], scenario_idx, valuation_year, "modal_split")
        other_modal = _get_value_for_scenario_year(components["modal_split_other"], scenario_idx, valuation_year, "modal_split")
        distance_value = _get_value_for_scenario_year(components["distance_per_person"], scenario_idx, valuation_year, "distance_per_person")

        population_growth_factor = total_population_valuation / total_population_start if total_population_start > 0 else 1.0
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
                "other_modal_split": other_modal,
                "distance_per_person": distance_value,
                "road_demand_proxy": road_demand_proxy,
                "rail_demand_proxy": rail_demand_proxy,
                "modal_split_sum": road_modal + rail_modal + other_modal,
                "modal_split_sum_error_abs": abs(1.0 - (road_modal + rail_modal + other_modal)),
            }
        )

    summary_df = pd.DataFrame(rows)
    score_cols = [
        "population_growth_factor",
        "road_modal_split",
        "rail_modal_split",
        "other_modal_split",
        "distance_per_person",
        "road_demand_proxy",
        "rail_demand_proxy",
    ]
    for col in score_cols:
        summary_df[f"{col}_pct_rank"] = summary_df[col].rank(method="average", pct=True)
    summary_df["shared_future_score"] = summary_df[[f"{col}_pct_rank" for col in score_cols]].mean(axis=1)
    return summary_df.sort_values("shared_future_score").reset_index(drop=True)


def select_representative_shared_scenarios(
    summary_df: pd.DataFrame,
    n_representatives: int,
) -> pd.DataFrame:
    return _shared_impl().select_representative_shared_scenarios(summary_df, n_representatives)


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


def _plot_modal_split_band(
    df: pd.DataFrame,
    title: str,
    output_path: str,
    federal_2050_range: tuple[float, float],
) -> None:
    year_stats = (
        df.groupby("year")["modal_split"]
        .agg(min="min", max="max", mean="mean", std="std")
        .reset_index()
    )
    std = year_stats["std"].fillna(0.0)
    year_stats["mean_plus_1_65std"] = year_stats["mean"] + 1.65 * std
    year_stats["mean_minus_1_65std"] = year_stats["mean"] - 1.65 * std

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.fill_between(year_stats["year"], year_stats["min"], year_stats["max"], color="grey", alpha=0.3, label="Gesamter Bereich")
    ax.plot(year_stats["year"], year_stats["mean_plus_1_65std"], color="red", linestyle="-", alpha=0.7, label="+1,65σ (95%)")
    ax.plot(year_stats["year"], year_stats["mean_minus_1_65std"], color="red", linestyle="-", alpha=0.7, label="-1,65σ (5%)")
    ax.plot(year_stats["year"], year_stats["mean"], color="grey", linestyle="--", alpha=0.8, label="Mittelwert")

    sample_id = int(df["scenario"].drop_duplicates().sample(n=1, random_state=42).iloc[0])
    sample_df = df[df["scenario"] == sample_id].sort_values("year")
    ax.plot(sample_df["year"], sample_df["modal_split"], color="blue", linewidth=2, label=f"Beispielszenario {sample_id}")

    lower_bound, upper_bound = federal_2050_range
    marker_color = "#E08D3C"
    marker_description = f"Verkehrsperspektive 2050 ({lower_bound*100:.1f}-{upper_bound*100:.1f}%)"
    ax.vlines(x=2050, ymin=lower_bound, ymax=upper_bound, colors=marker_color, linestyles="solid", linewidth=2, label=marker_description)
    ax.plot([2050], [lower_bound], marker="_", markersize=10, color=marker_color)
    ax.plot([2050], [upper_bound], marker="_", markersize=10, color=marker_color)

    ax.set_xlabel("Jahr")
    ax.set_ylabel("Modal-Split (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%"))
    ax.set_title(title)
    ax.grid(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _plot_value_band(
    df: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: str,
) -> None:
    year_stats = (
        df.groupby("year")[value_column]
        .agg(min="min", max="max", mean="mean", std="std")
        .reset_index()
    )
    std = year_stats["std"].fillna(0.0)
    year_stats["upper_195"] = year_stats["mean"] + 1.95 * std
    year_stats["lower_195"] = year_stats["mean"] - 1.95 * std

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.fill_between(year_stats["year"], year_stats["min"], year_stats["max"], alpha=0.22, color="#7f8c8d", label="Total range")
    ax.plot(year_stats["year"], year_stats["mean"], color="#2c3e50", linestyle="--", linewidth=1.8, label="Mean")
    ax.plot(year_stats["year"], year_stats["upper_195"], color="#8e44ad", linewidth=1.2, linestyle=":", label="Mean ± 1.95σ")
    ax.plot(year_stats["year"], year_stats["lower_195"], color="#8e44ad", linewidth=1.2, linestyle=":")
    sample_id = int(df["scenario"].drop_duplicates().iloc[0])
    sample_df = df[df["scenario"] == sample_id]
    ax.plot(sample_df["year"], sample_df[value_column], color="#2980b9", linewidth=1.6, label=f"Sample scenario {sample_id + 1}")
    ax.set_xlabel("Year")
    ax.set_ylabel(value_column.replace("_", " ").title())
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_shared_scenario_components(
    components: Dict[str, Any],
    summary_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _plot_modal_split_band(
        components["modal_split_rail"],
        "Rail modal split scenarios",
        os.path.join(output_dir, "modal_split_rail.png"),
        federal_2050_range=(0.187, 0.243),
    )
    _plot_modal_split_band(
        components["modal_split_road"],
        "Road modal split scenarios",
        os.path.join(output_dir, "modal_split_road.png"),
        federal_2050_range=(0.670, 0.738),
    )
    _plot_modal_split_band(
        components["modal_split_other"],
        "Other modal split scenarios",
        os.path.join(output_dir, "modal_split_other.png"),
        federal_2050_range=(0.067, 0.089),
    )
    _plot_value_band(
        components["distance_per_person"],
        "distance_per_person",
        "Distance per person scenarios",
        os.path.join(output_dir, "distance_per_person.png"),
    )

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ordered = summary_df.sort_values("shared_future_score").reset_index(drop=True)
    ax.plot(np.arange(len(ordered)), ordered["shared_future_score"], color="#2c3e50", linewidth=1.6)
    if not selected_df.empty:
        selected_lookup = ordered.reset_index().merge(selected_df[["scenario", "selection_order"]], on="scenario", how="inner")
        ax.scatter(selected_lookup["index"], selected_lookup["shared_future_score"], color="#c0392b", s=35, zorder=3)
        for row in selected_lookup.itertuples(index=False):
            ax.text(row.index, row.shared_future_score, f"  {row.selection_order}", va="center", fontsize=8)
    ax.set_xlabel("Scenario rank")
    ax.set_ylabel("Shared future score")
    ax.set_title("Representative scenario selection")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "shared_future_score.png"), bbox_inches="tight")
    plt.close(fig)


def export_generated_population_rasters(*args, **kwargs):
    return _shared_impl().export_generated_population_rasters(*args, **kwargs)


def get_random_scenarios(*args, **kwargs):
    return _shared_impl().get_random_scenarios(*args, **kwargs)


def get_rail_random_scenarios(*args, **kwargs):
    return _shared_impl().get_rail_random_scenarios(*args, **kwargs)


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
    selected_df = select_representative_shared_scenarios(summary_df, n_representatives=representative_scenarios_count)
    selection_file = save_representative_scenario_selection(selected_df, output_path=selection_path)
    selected_scenarios = selected_df["scenario"].tolist()

    if apply_selection_to_modes:
        apply_selected_scenarios_to_mode_settings(selected_scenarios)

    if run_road:
        get_random_scenarios(
            start_year=start_year,
            end_year=end_year,
            num_of_scenarios=num_of_scenarios,
            use_cache=False,
            do_plot=do_plot,
            shared_components_path=saved_path,
        )
    if run_rail:
        get_rail_random_scenarios(
            start_year=start_year,
            end_year=end_year,
            num_of_scenarios=num_of_scenarios,
            use_cache=False,
            do_plot=do_plot,
            shared_components_path=saved_path,
        )
    if do_plot:
        plot_dir = os.path.join(os.path.dirname(summary_file) or integrated_paths.SCENARIO_CACHE_SHARED_DIR, "plots")
        plot_shared_scenario_components(components, summary_df, selected_df, plot_dir)

    return {
        "components_path": saved_path,
        "summary_path": summary_file,
        "selection_path": selection_file,
        "selected_scenarios": selected_scenarios,
        "selected_summary": selected_df,
    }


if __name__ == "__main__":
    generated = generate_and_apply_shared_scenarios(
        start_year=2018,
        end_year=2100,
        num_of_scenarios=int(integrated_settings.amount_of_scenarios),
        run_road=True,
        run_rail=True,
        do_plot=True,
    )
    print(f"Shared scenario components generated at: {generated['components_path']}")
    print(f"Shared scenario summary written to: {generated['summary_path']}")
    print(f"Representative scenarios written to: {generated['selection_path']}")
    print(f"Selected scenarios: {generated['selected_scenarios']}")
