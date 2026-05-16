import os
import pickle
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from joblib import Parallel, delayed
from scipy.stats import norm, qmc

from infraScan.infraScanRoad import settings as road_settings
from infraScan.infraScanRoad.scoring import GetCommuneShapes
from . import paths as integrated_paths
from . import settings as integrated_settings

DEFAULT_SHARED_COMPONENTS_PATH = integrated_paths.SHARED_COMPONENTS_PATH
DEFAULT_SHARED_SUMMARY_PATH = integrated_paths.SHARED_SUMMARY_PATH
DEFAULT_SHARED_SELECTION_PATH = integrated_paths.SHARED_SELECTION_PATH

ROAD_COMMUNE_TO_ZONE_MAPPING_PATH = os.path.join(integrated_paths.MAIN, "data", "infraScanRoad", "Scenario", "commune_to_zone_mapping.csv")
ROAD_VORONOI_TIF_PATH = os.path.join(integrated_paths.MAIN, "data", "infraScanRoad", "Network", "travel_time", "source_id_raster.tif")
ROAD_OD_MATRIX_PATH = os.path.join(integrated_paths.MAIN, "data", "infraScanRoad", "traffic_flow", "od", "od_matrix_20.csv")
ROAD_POPULATION_BY_COMMUNE_PATH = os.path.join(integrated_paths.MAIN, "data", "Scenario", "population_by_gemeinde_2018.csv")
ROAD_SCENARIO_CACHE_DIR = os.path.join(integrated_paths.MAIN, "data", "Scenario", "cache", "road", "random")
ROAD_POPULATION_RASTER_OUTPUT_DIR = os.path.join(integrated_paths.MAIN, "data", "independent_variable", "processed", "scenario")
RAIL_OD_MATRIX_PATH = os.path.join(integrated_paths.MAIN, "data", "infraScanRail", "traffic_flow", "od", "rail", "ktzh", "od_matrix_stations_ktzh_20.csv")
RAIL_COMMUNE_TO_STATION_PATH = os.path.join(integrated_paths.MAIN, "data", "infraScanRail", "Network", "processed", "Communes_to_railway_stations_ZH.xlsx")
RAIL_SCENARIO_CACHE_DIR = os.path.join(integrated_paths.MAIN, "data", "Scenario", "cache", "rail")


# Shared scenario core

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

def get_bezirk_population_scenarios():
    # Read the Swiss population scenario CSV with "," separator
    df_ch = pd.read_csv(integrated_paths.POPULATION_SCENARIO_CH_BFS_2055, sep=",")
    # Extract the relevant values for 2018 and 2050
    pop_2018 = df_ch.loc[df_ch['Jahr'] == 2018, 'Beobachtungen'].values
    pop_2050 = df_ch.loc[df_ch['Jahr'] == 2050, 'Referenzszenario A-00-2025'].values
    # Compute the growth factor: population_2050 / population_2018
    swiss_growth_factor_18_50 = pop_2050[0] / pop_2018[0]
    # Read the CSV file with ";" as separator
    df = pd.read_csv(integrated_paths.POPULATION_SCENARIO_CANTON_ZH_2050, sep=';', encoding='utf-8')
    # Step 1: Aggregate total population per district and year
    population_summary = (
        df.groupby(['bezirk', 'jahr'])['anzahl']
        .sum()
        .reset_index()
        .rename(columns={'anzahl': 'total_population'})
    )
    # Step 2: Create full grid for all districts and all years from 2011 to 2050
    all_years = pd.Series(range(2011, 2051), name='jahr')
    all_districts = population_summary['bezirk'].unique()
    full_index = pd.MultiIndex.from_product([all_districts, all_years], names=['bezirk', 'jahr'])
    # Reindex to ensure each district has all years, fill missing population with 0
    population_complete = (
        population_summary.set_index(['bezirk', 'jahr'])
        .reindex(full_index)
        .fillna({'total_population': 0})
        .reset_index()
    )
    # Step 3: Calculate year-over-year growth rate per district
    population_complete['growth_rate'] = (
        population_complete
        .sort_values(['bezirk', 'jahr'])
        .groupby('bezirk')['total_population']
        .pct_change()
    )
    # Step 4: Split the complete dataset into a dictionary by district
    district_tables = {}
    for district, group in population_complete.groupby('bezirk'):
        group = group.reset_index(drop=True)

        # Extract population for 2018 and 2050
        pop_2018 = group.loc[group['jahr'] == 2018, 'total_population'].values
        pop_2050 = group.loc[group['jahr'] == 2050, 'total_population'].values

        growth_factor_18_50 = pop_2050[0] / pop_2018[0]

        # Compute yearly relative growth factor vs. CH
        relative_growth = (growth_factor_18_50 - 1) / (swiss_growth_factor_18_50 - 1)
        yearly_growth_factor = relative_growth ** (
                    1 / 32)  # this factor is only applicable to yearly growth RATES in the form of 0.015 for example, not FACTORS 1.015!!!

        # Store in DataFrame attributes
        group.attrs['growth_factor_18_50'] = growth_factor_18_50
        group.attrs['yearly_growth_factor_district_to_CH'] = yearly_growth_factor

        district_tables[district] = group
    # Step 5: Read Swiss growth rates from Eurostat Excel file
    eurostat_df = pd.read_excel(integrated_paths.POPULATION_SCENARIO_CH_EUROSTAT_2100)
    # Convert all column names to strings FIRST (important!)
    eurostat_df.columns = eurostat_df.columns.map(str)
    # Filter for the row where unit == 'GROWTH_RATE'
    growth_rate_row = eurostat_df[eurostat_df['unit'] == 'GROWTH_RATE']
    # Define year columns as strings
    year_columns = [str(year) for year in range(2051, 2101)]
    # Extract growth rates from that row
    ch_growth_rates = growth_rate_row[year_columns].iloc[0].astype(float)
    # Step 6: Extend each district with projected growth rates and populations
    for district, df_district in district_tables.items():
        # Get the last known population (for 2050)
        last_population = df_district.loc[df_district['jahr'] == 2050, 'total_population'].values[0]

        # Get the district-specific yearly growth factor to scale national growth rates
        scaling_factor = df_district.attrs['yearly_growth_factor_district_to_CH']

        # Prepare data for years 2051–2100
        new_rows = []
        current_population = last_population

        for year in range(2051, 2101):
            base_growth_rate = ch_growth_rates[str(year)]  # national growth rate (e.g., 0.012)
            adjusted_growth_rate = base_growth_rate * scaling_factor

            current_population *= (1 + adjusted_growth_rate)

            new_rows.append({
                'bezirk': district,
                'jahr': year,
                'total_population': current_population,
                'growth_rate': adjusted_growth_rate
            })

        # Convert new rows to DataFrame and append
        extension_df = pd.DataFrame(new_rows)
        df_extended = pd.concat([df_district, extension_df], ignore_index=True)
        district_tables[district] = df_extended.reset_index(drop=True)
    return district_tables

def generate_population_scenarios(ref_df: pd.DataFrame,
                                  start_year: int,
                                  end_year: int,
                                  n_scenarios: int = 1000,
                                  start_std_dev: float = 0.01,
                                  end_std_dev: float = 0.03,
                                  std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic population scenarios using Latin Hypercube Sampling and a random walk process.
    The main growth rates are perturbed using LHS and a time-varying std dev. Random shocks are added separately.

    Parameters:
    - ref_df: DataFrame with columns "jahr", "total_population", "growth_rate"
              - Only the "total_population" value at start_year is used as the initial population.
              - "growth_rate" is used as the base deterministic growth.
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "population", "growth_rate"
    """
    # Filter and sort reference data
    ref_df = ref_df.sort_values("jahr")
    ref_df = ref_df[(ref_df["jahr"] >= start_year) & (ref_df["jahr"] <= end_year)].reset_index(drop=True)

    years = ref_df["jahr"].values
    ref_growth = ref_df["growth_rate"].values  # deterministic base growth per year
    n_years = len(years)
    initial_population = ref_df[ref_df["jahr"] == start_year]["total_population"].values[0]

    # Linearly interpolate std devs across years for growth rate variation
    growth_std_devs = np.linspace(start_std_dev, end_std_dev, n_years)

    # Latin Hypercube Sampling: growth rate perturbations
    sampler = qmc.LatinHypercube(d=n_years, seed = 42)
    lhs_samples = sampler.random(n=n_scenarios)  # shape: (n_scenarios, n_years)
    growth_perturbations = norm.ppf(lhs_samples) * growth_std_devs  # shape: (n_scenarios, n_years)

    # Perturbed growth rate: base + scenario-specific offset
    scenario_growth = ref_growth + growth_perturbations  # shape: (n_scenarios, n_years)
    # Setze Wachstumsrate für das erste Jahr auf 0 (kein Wachstum im ersten Jahr)
    scenario_growth[:, 0] = 0

    # Random shocks: et ~ N(0, std_dev_shocks)
    shock_sampler = qmc.LatinHypercube(d=n_years, seed = 43)
    lhs_shocks = shock_sampler.random(n=n_scenarios)
    et = norm.ppf(lhs_shocks) * std_dev_shocks
    # Setze Schocks für das erste Jahr auf 0
    et[:, 0] = 0

    # Cumulative shocks per scenario
    cumulative_shocks = np.cumsum(et, axis=1)  # shape: (n_scenarios, n_years)

    # Deterministic growth: cumulative product of (1 + growth_rate)
    deterministic_growth = np.cumprod(1 + scenario_growth, axis=1)  # shape: (n_scenarios, n_years)

    # Population index = deterministic path × stochastic shocks
    population_index = deterministic_growth + cumulative_shocks

    # Scale by initial population
    pop_scenarios = initial_population * population_index

    # Assemble output DataFrame
    scenario_data = []
    for i in range(n_scenarios):
        for t in range(n_years):
            # Berechne growth_index: 100 am Anfang und dann entsprechend der relativen Bevölkerungsentwicklung
            growth_index = 100 * (pop_scenarios[i, t] / initial_population)

            # Berechne die effektive Wachstumsrate inklusive Schocks
            if t == 0:
                # Für das erste Jahr ist die Wachstumsrate definitionsgemäß 0
                effective_growth_rate = 0.0
            else:
                # Berechne die prozentuale Änderung zur Bevölkerung des Vorjahres
                effective_growth_rate = (pop_scenarios[i, t] / pop_scenarios[i, t - 1]) - 1

            scenario_data.append({
                "scenario": i,
                "year": years[t],
                "population": pop_scenarios[i, t],
                "growth_rate": effective_growth_rate,
                "growth_index": growth_index
            })

    return pd.DataFrame(scenario_data)


def _finalize_modal_split_frame(
    scenarios_df: pd.DataFrame,
    start_year: int,
) -> pd.DataFrame:
    finalized = scenarios_df.copy()
    finalized["modal_split"] = pd.to_numeric(finalized["modal_split"], errors="coerce").clip(lower=0.0, upper=1.0)
    finalized = finalized.sort_values(["scenario", "year"]).reset_index(drop=True)
    finalized["growth_rate"] = (
        finalized.groupby("scenario")["modal_split"].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )
    start_values = (
        finalized[finalized["year"] == start_year][["scenario", "modal_split"]]
        .rename(columns={"modal_split": "start_modal_split"})
    )
    finalized = finalized.merge(start_values, on="scenario", how="left")
    # Keep the index calculation explicit so it behaves consistently across pandas versions.
    finalized["growth_index"] = np.where(
        finalized["start_modal_split"].fillna(0.0) > 0.0,
        100.0 * (finalized["modal_split"] / finalized["start_modal_split"]),
        100.0,
    )
    finalized = finalized.drop(columns=["start_modal_split"])
    return finalized


def generate_joint_modal_split_scenarios(
    start_year: int,
    end_year: int,
    n_scenarios: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate stochastic modal split scenarios for all three modes jointly using a logistic-normal approach.
    the three modes are generated together and always sum exactly to 1 after the softmax-style back-transformation.
    The latent process is calibrated on the residual simplex after injecting a minimum share, so the std dev settings are no longer direct share perturbations but are combined into the latent joint-process volatility.
    """
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)
    horizon = max(1, end_year - start_year)

    start_shares = np.array(
        [
            integrated_settings.rail_modal_split_start,
            integrated_settings.road_modal_split_start,
            integrated_settings.other_modal_split_start,
        ],
        dtype=float,
    )
    target_shares = np.array(
        [
            integrated_settings.rail_modal_split_start * ((1 + integrated_settings.rail_modal_split_avg_growth_rate) ** horizon),
            integrated_settings.road_modal_split_start * ((1 + integrated_settings.road_modal_split_avg_growth_rate) ** horizon),
            integrated_settings.other_modal_split_start * ((1 + integrated_settings.other_modal_split_avg_growth_rate) ** horizon),
        ],
        dtype=float,
    )
    target_shares = target_shares / target_shares.sum()
    min_share = integrated_settings.modal_split_min_share
    residual_weight = 1.0 - 3.0 * min_share

    # The latent process is calibrated on the residual simplex because min_share
    # is injected only in the final share space, not in latent space.
    start_residual = np.clip((start_shares - min_share) / residual_weight, 1e-9, None)
    start_residual = start_residual / start_residual.sum()
    target_residual = np.clip((target_shares - min_share) / residual_weight, 1e-9, None)
    target_residual = target_residual / target_residual.sum()

    start_latent = np.array(
        [
            np.log(start_residual[0] / start_residual[1]),
            np.log(start_residual[2] / start_residual[1]),
        ],
        dtype=float,
    )
    target_latent = np.array(
        [
            np.log(target_residual[0] / target_residual[1]),
            np.log(target_residual[2] / target_residual[1]),
        ],
        dtype=float,
    )
    latent_drift = (target_latent - start_latent) / horizon

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

    corr = np.array(
        [
            [1.0, integrated_settings.modal_split_latent_correlation],
            [integrated_settings.modal_split_latent_correlation, 1.0],
        ],
        dtype=float,
    )
    corr += np.eye(2) * 1e-9

    rng_growth = np.random.default_rng(42)
    rng_shock = np.random.default_rng(43)
    latent_paths = np.zeros((n_scenarios, n_years, 2), dtype=float)
    latent_paths[:, 0, :] = start_latent

    for year_idx in range(1, n_years):
        # The per-mode std dev settings are used as calibration inputs
        # for the latent log-ratio process rather than direct share perturbations.
        rail_scale = np.sqrt(rail_std_path[year_idx] ** 2 + road_std_path[year_idx] ** 2)
        other_scale = np.sqrt(other_std_path[year_idx] ** 2 + road_std_path[year_idx] ** 2)
        shock_rail_scale = np.sqrt(
            integrated_settings.rail_modal_split_std_dev_shocks ** 2
            + integrated_settings.road_modal_split_std_dev_shocks ** 2
        )
        shock_other_scale = np.sqrt(
            integrated_settings.other_modal_split_std_dev_shocks ** 2
            + integrated_settings.road_modal_split_std_dev_shocks ** 2
        )
        scale_factor = integrated_settings.modal_split_latent_std_scale
        rail_scale *= scale_factor
        other_scale *= scale_factor
        shock_rail_scale *= scale_factor
        shock_other_scale *= scale_factor
        # Fade in uncertainty smoothly so the first years do not jump
        warmup = max(1, int(integrated_settings.modal_split_warmup_years))
        warmup_factor = min(1.0, year_idx / warmup)
        warmup_factor = warmup_factor ** 1.5
        rail_scale *= warmup_factor
        other_scale *= warmup_factor
        shock_rail_scale *= warmup_factor
        shock_other_scale *= warmup_factor

        growth_cov = np.diag([rail_scale, other_scale]) @ corr @ np.diag([rail_scale, other_scale])
        shock_cov = np.diag([shock_rail_scale, shock_other_scale]) @ corr @ np.diag([shock_rail_scale, shock_other_scale])

        growth_innov = rng_growth.multivariate_normal(mean=np.zeros(2), cov=growth_cov, size=n_scenarios)
        shock_innov = rng_shock.multivariate_normal(mean=np.zeros(2), cov=shock_cov, size=n_scenarios)
        reference_prev = start_latent + latent_drift * (year_idx - 1)
        reversion = integrated_settings.modal_split_latent_reversion * (latent_paths[:, year_idx - 1, :] - reference_prev)
        latent_paths[:, year_idx, :] = latent_paths[:, year_idx - 1, :] + latent_drift + growth_innov + shock_innov - reversion

    latent_rail = latent_paths[:, :, 0]
    latent_other = latent_paths[:, :, 1]
    exp_rail = np.exp(np.clip(latent_rail, -20.0, 20.0))
    exp_other = np.exp(np.clip(latent_other, -20.0, 20.0))
    denominator = 1.0 + exp_rail + exp_other

    rail_share = min_share + residual_weight * (exp_rail / denominator)
    road_share = min_share + residual_weight * (1.0 / denominator)
    other_share = min_share + residual_weight * (exp_other / denominator)

    rail_share[:, 0] = integrated_settings.rail_modal_split_start
    road_share[:, 0] = integrated_settings.road_modal_split_start
    other_share[:, 0] = integrated_settings.other_modal_split_start

    def _shares_to_df(values: np.ndarray) -> pd.DataFrame:
        scenario_data = []
        for scenario_idx in range(n_scenarios):
            for year_idx, year in enumerate(years):
                scenario_data.append(
                    {
                        "scenario": scenario_idx,
                        "year": int(year),
                        "modal_split": float(values[scenario_idx, year_idx]),
                    }
                )
        return pd.DataFrame(scenario_data)

    rail_df = _finalize_modal_split_frame(_shares_to_df(rail_share), start_year=start_year)
    road_df = _finalize_modal_split_frame(_shares_to_df(road_share), start_year=start_year)
    other_df = _finalize_modal_split_frame(_shares_to_df(other_share), start_year=start_year)
    return rail_df, road_df, other_df


def generate_distance_per_person_scenarios(avg_growth_rate: float,
                                           start_value: float,
                                           start_year: int,
                                           end_year: int,
                                           n_scenarios: int = 1000,
                                           start_std_dev: float = 0.01,
                                           end_std_dev: float = 0.03,
                                           std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic trips per person scenarios using Latin Hypercube Sampling and a random walk process.

    Parameters:
    - avg_growth_rate: average annual growth rate to apply (can be positive or negative)
    - start_value: initial trips per person value at start_year
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "trips_per_person", "growth_rate", "growth_index"
    """
    # Distance per person still uses the single-series setup:
    # deterministic drift + LHS perturbations + cumulative shocks.
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)

    growth_factors = np.ones(n_years) * (1 + avg_growth_rate)
    growth_factors[0] = 1
    cumulative_growth = np.cumprod(growth_factors)
    distance_values = start_value * cumulative_growth

    growth_rates = np.zeros(n_years)
    growth_rates[1:] = avg_growth_rate

    ref_df = pd.DataFrame({
        "jahr": years,
        "total_population": distance_values,
        "growth_rate": growth_rates
    })

    scenarios_df = generate_population_scenarios(
        ref_df=ref_df,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
        start_std_dev=start_std_dev,
        end_std_dev=end_std_dev,
        std_dev_shocks=std_dev_shocks
    )

    scenarios_df = scenarios_df.rename(columns={"population": "distance_per_person"})

    return scenarios_df

# -----------------------------------------------------------------------------------
# Rail-only helpers
# The functions below are only needed when the integrated generator is asked to
# materialize rail scenarios, i.e. generate_and_apply_shared_scenarios(..., run_rail=True).
# -----------------------------------------------------------------------------------

def build_station_to_communes_mapping(
        communes_to_stations: pd.DataFrame
) -> Dict[str, List[int]]:
    """
    Baut ein Mapping: station_id -> List[Commune_BFS_code].
    """
    return communes_to_stations.groupby('ID_point')['Commune_BFS_code'] \
        .apply(list) \
        .to_dict()


def compute_growth_od_matrix_rail_optimized(
        initial_od: pd.DataFrame,
        station_communes: Dict[str, List[int]],
        communes_population: pd.DataFrame,
        population_scenarios: Dict[str, pd.DataFrame],
        scenario: int,
        year: int,
        start_year: int,
        station_commune_lookup: Dict[str, pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Optimierte Version von compute_growth_od_matrix
    """
    # --- 1) Struktur aufsetzen
    stations = [int(station) for station in initial_od.columns.tolist()]
    from_stations = [int(station) for station in initial_od.index.tolist()]

    # --- 2) Wachstumsindex für alle Stationen berechnen
    station_growth = {}

    # Wenn lookup nicht existiert, erstellen wir einen
    if station_commune_lookup is None:
        station_commune_lookup = {}

    for station in stations:
        # Lookup für diese Station verwenden wenn vorhanden
        if station not in station_commune_lookup:
            communes = station_communes.get(int(station), [])
            if not communes:
                station_growth[station] = 1.0
                continue

            # Vorfiltern der relevanten Gemeinden für diese Station
            station_data = []
            for commune in communes:
                row = communes_population[communes_population['gemeinde_bfs_nr'] == commune]
                if not row.empty:
                    district = row['bezirk'].iat[0]
                    pop_start_commune = row['anzahl'].iat[0]
                    station_data.append((commune, district, pop_start_commune))

            station_commune_lookup[station] = station_data

        sum_start = 0.0
        sum_curr = 0.0

        for commune, district, pop_start_commune in station_commune_lookup[station]:
            # Population im Szenario für Start- und Ziel-Jahr
            scen = population_scenarios[str(district) if str(district) in population_scenarios else district]

            # Effizienterer Zugriff mit vorgefilterten Daten
            scenario_data = scen[(scen['scenario'] == scenario)]
            pop_d_start = scenario_data[scenario_data['year'] == start_year]['population'].iloc[0]
            pop_d_curr = scenario_data[scenario_data['year'] == year]['population'].iloc[0]

            # Bezirksfaktor
            factor_d = (pop_d_curr / pop_d_start) if pop_d_start > 0 else 1.0

            sum_start += pop_start_commune
            sum_curr += pop_start_commune * factor_d

        station_growth[station] = (sum_curr / sum_start) if sum_start > 0 else 1.0

    # --- 3) OD-Matrix mit Wachstumsfaktoren erzeugen
    growth_od = pd.DataFrame(1.0, index=from_stations, columns=stations)

    # Vektorisierte Anwendung der Faktoren
    sqrt_factors = {station: np.sqrt(factor) for station, factor in station_growth.items()}

    # Zeilenweise Multiplikation
    for station in set(from_stations).intersection(sqrt_factors.keys()):
        growth_od.loc[station, :] *= sqrt_factors[station]

    # Spaltenweise Multiplikation
    for station in set(stations).intersection(sqrt_factors.keys()):
        growth_od.loc[:, station] *= sqrt_factors[station]

    return growth_od


# ----------------------------------------------------------------------------------------
# Road-only helpers
# The functions below are only needed when the integrated generator is asked to
# materialize road scenarios, i.e. generate_and_apply_shared_scenarios(..., run_road=True).
# ----------------------------------------------------------------------------------------

def build_commune_to_zone_mapping_from_raster_overlap(
    voronoi_tif_path: str,
    output_csv_path: str,
) -> pd.DataFrame:
    commune_raster, _ = GetCommuneShapes(raster_path=voronoi_tif_path)

    with rasterio.open(voronoi_tif_path) as src:
        zone_raster = src.read(1)

    zone_ids = np.sort(np.unique(zone_raster))
    commune_ids = np.sort(np.unique(commune_raster))

    records = []
    for zone_id in zone_ids:
        if zone_id <= 0:
            continue

        zone_mask = zone_raster == zone_id
        for commune_id in commune_ids:
            if commune_id <= 0:
                continue

            commune_mask = commune_raster == commune_id
            overlap = zone_mask & commune_mask
            overlap_cells = int(np.nansum(overlap))
            commune_cells = int(np.nansum(commune_mask))

            if overlap_cells > 0 and commune_cells > 0:
                records.append(
                    {
                        "zone_id": int(zone_id),
                        "commune_bfs": int(commune_id),
                        "weight": overlap_cells / commune_cells,
                    }
                )

    mapping = pd.DataFrame(records)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    mapping.to_csv(output_csv_path, index=False)
    return mapping


def load_or_create_commune_to_zone_mapping(mapping_path: str, voronoi_tif_path: str) -> pd.DataFrame:
    if os.path.exists(mapping_path):
        return pd.read_csv(mapping_path)

    if not os.path.exists(voronoi_tif_path):
        raise FileNotFoundError(
            f"Neither mapping file nor raster found. Missing: {mapping_path} and/or {voronoi_tif_path}"
        )

    return build_commune_to_zone_mapping_from_raster_overlap(
        voronoi_tif_path=voronoi_tif_path,
        output_csv_path=mapping_path,
    )


def build_zone_to_communes_mapping(communes_to_zones: pd.DataFrame) -> Dict[str, list]:
    communes_to_zones = communes_to_zones.copy()
    communes_to_zones["zone_id"] = communes_to_zones["zone_id"].astype(int).astype(str)
    communes_to_zones["commune_bfs"] = communes_to_zones["commune_bfs"].astype(int)

    zone_map: Dict[str, list] = {}
    for zone_id, group in communes_to_zones.groupby("zone_id"):
        zone_map[zone_id] = list(zip(group["commune_bfs"].tolist(), group["weight"].tolist()))
    return zone_map


def compute_growth_od_matrix_optimized(
        initial_od: pd.DataFrame,
        station_communes: Dict[str, List[int]],
        communes_population: pd.DataFrame,
        population_scenarios: Dict[str, pd.DataFrame],
        scenario: int,
        year: int,
        start_year: int,
        station_commune_lookup: Dict[str, pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Optimierte Version von compute_growth_od_matrix
    """
    # --- 1) Struktur aufsetzen
    stations = initial_od.columns.tolist()
    from_stations = initial_od.index.tolist()

    # --- 2) Wachstumsindex für alle Stationen berechnen
    station_growth = {}

    # Wenn lookup nicht existiert, erstellen wir einen
    if station_commune_lookup is None:
        station_commune_lookup = {}

    for station in stations:
        # Lookup für diese Station verwenden wenn vorhanden
        if station not in station_commune_lookup:
            communes = station_communes.get(int(station), [])
            if not communes:
                station_growth[station] = 1.0
                continue

            # Vorfiltern der relevanten Gemeinden für diese Station
            station_data = []
            for commune in communes:
                row = communes_population[communes_population['gemeinde_bfs_nr'] == commune]
                if not row.empty:
                    district = row['bezirk'].iat[0]
                    pop_start_commune = row['anzahl'].iat[0]
                    station_data.append((commune, district, pop_start_commune))

            station_commune_lookup[station] = station_data

        sum_start = 0.0
        sum_curr = 0.0

        for commune, district, pop_start_commune in station_commune_lookup[station]:
            # Population im Szenario für Start- und Ziel-Jahr
            scen = population_scenarios[str(district) if str(district) in population_scenarios else district]

            # Effizienterer Zugriff mit vorgefilterten Daten
            scenario_data = scen[(scen['scenario'] == scenario)]
            pop_d_start = scenario_data[scenario_data['year'] == start_year]['population'].iloc[0]
            pop_d_curr = scenario_data[scenario_data['year'] == year]['population'].iloc[0]

            # Bezirksfaktor
            factor_d = (pop_d_curr / pop_d_start) if pop_d_start > 0 else 1.0

            sum_start += pop_start_commune
            sum_curr += pop_start_commune * factor_d

        station_growth[station] = (sum_curr / sum_start) if sum_start > 0 else 1.0

    # --- 3) OD-Matrix mit Wachstumsfaktoren erzeugen
    growth_od = pd.DataFrame(1.0, index=from_stations, columns=stations)

    # Vektorisierte Anwendung der Faktoren
    sqrt_factors = {station: np.sqrt(factor) for station, factor in station_growth.items()}

    # Zeilenweise Multiplikation
    for station in set(list(map(str, from_stations))).intersection(sqrt_factors.keys()):
        growth_od.loc[station, :] *= sqrt_factors[station]

    # Spaltenweise Multiplikation
    for station in set(stations).intersection(sqrt_factors.keys()):
        growth_od.loc[:, station] *= sqrt_factors[station]

    return growth_od


def apply_modal_trips_optimized(
        initial_od: pd.DataFrame,
        growth_od: pd.DataFrame,
        modal_factors: Dict[tuple, float],
        distance_factors: Dict[tuple, float],
        scenario: int,
        start_year: int,
        year: int
) -> pd.DataFrame:
    """
    Optimierte Version von apply_modal_trips mit Lookup-Table
    """
    # Faktoren aus Lookup-Tabelle holen
    scenario_year_key = (scenario, year)
    m_factor = modal_factors.get(scenario_year_key, 1.0)
    d_factor = distance_factors.get(scenario_year_key, 1.0)

    # Einen Schritt berechnen
    return (initial_od * growth_od * m_factor * d_factor).astype('float32')


def precompute_modal_distance_factors(
        modal_df: pd.DataFrame,
        distance_df: pd.DataFrame,
        start_year: int
) -> tuple:
    """
    Vorausberechnung der Modal- und Distance-Faktoren
    """
    modal_factors = {}
    distance_factors = {}

    scenarios = modal_df['scenario'].unique()
    years = modal_df['year'].unique()

    # Modal split factors
    for s in scenarios:
        m_start = modal_df.loc[(modal_df['scenario'] == s) &
                               (modal_df['year'] == start_year), 'modal_split'].iat[0]

        for y in years:
            if y == start_year:
                modal_factors[(s, y)] = 1.0
                continue

            m_curr = modal_df.loc[(modal_df['scenario'] == s) &
                                  (modal_df['year'] == y), 'modal_split'].iat[0]
            m_factor = (m_curr / m_start) if m_start > 0 else 1.0
            modal_factors[(s, y)] = m_factor

    # Distance per person factors
    for s in scenarios:
        d_start = distance_df.loc[(distance_df['scenario'] == s) &
                                  (distance_df['year'] == start_year), 'distance_per_person'].iat[0]

        for y in years:
            if y == start_year:
                distance_factors[(s, y)] = 1.0
                continue

            d_curr = distance_df.loc[(distance_df['scenario'] == s) &
                                     (distance_df['year'] == y), 'distance_per_person'].iat[0]
            d_factor = (d_curr / d_start) if d_start > 0 else 1.0
            distance_factors[(s, y)] = d_factor

    return modal_factors, distance_factors


def generate_od_growth_scenarios(
    initial_od_matrix: pd.DataFrame,
    communes_to_zones: pd.DataFrame,
    communes_population: pd.DataFrame,
    start_year: int,
    end_year: int,
    num_of_scenarios: int,
    scenario_components: Dict[str, Any] | None = None,
    do_plot: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    initial_od_matrix = initial_od_matrix.rename(columns={"voronoi_id": "from_zone"}).copy()
    initial_od_matrix["from_zone"] = initial_od_matrix["from_zone"].astype(float).astype(int).astype(str)
    initial_od_matrix.columns = ["from_zone" if col == "from_zone" else str(int(float(col))) for col in initial_od_matrix.columns]
    initial_od_matrix = initial_od_matrix.set_index("from_zone")
    communes_population.columns = communes_population.columns.str.replace("\ufeff", "", regex=False)

    if scenario_components is not None:
        population_scenarios = scenario_components["population_scenarios"]
        modal_split_scenarios = scenario_components["modal_split_road"]
        distance_per_person_scenarios = scenario_components["distance_per_person"]
    else:
        generated_components = build_shared_scenario_components(
            start_year=start_year,
            end_year=end_year,
            num_of_scenarios=num_of_scenarios,
        )
        population_scenarios = generated_components["population_scenarios"]
        modal_split_scenarios = generated_components["modal_split_road"]
        distance_per_person_scenarios = generated_components["distance_per_person"]

    if do_plot:
        pass

    modal_factors, distance_factors = precompute_modal_distance_factors(
        modal_split_scenarios, distance_per_person_scenarios, start_year
    )

    zone_communes = build_zone_to_communes_mapping(communes_to_zones)
    zone_commune_lookup: Dict[str, pd.DataFrame] = {}

    def process_scenario(s: int):
        key = f"scenario_{s + 1}"
        results_s = {}
        for y in range(start_year, end_year + 1):
            pop_growth_od = compute_growth_od_matrix_optimized(
                initial_od_matrix,
                zone_communes,
                communes_population,
                population_scenarios,
                s,
                y,
                start_year,
                zone_commune_lookup,
            )
            results_s[y] = apply_modal_trips_optimized(
                initial_od_matrix,
                pop_growth_od,
                modal_factors,
                distance_factors,
                s,
                start_year,
                y,
            )
        return key, results_s

    try:
        scenario_results = Parallel(n_jobs=n_jobs, verbose=100)(delayed(process_scenario)(s) for s in range(num_of_scenarios))
    except PermissionError:
        # Fall back to serial execution when multiprocessing is blocked by the runtime.
        scenario_results = [process_scenario(s) for s in range(num_of_scenarios)]
    return dict(scenario_results)


def generate_rail_od_growth_scenarios(
    initial_od_matrix: pd.DataFrame,
    communes_to_stations: pd.DataFrame,
    communes_population: pd.DataFrame,
    start_year: int,
    end_year: int,
    num_of_scenarios: int,
    scenario_components: Dict[str, Any] | None,
    do_plot: bool = False,
    n_jobs: int = -1,
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Optimierte Version von generate_od_growth_scenarios mit Multiprocessing
    """
    initial_od_matrix = initial_od_matrix.copy()
    initial_od_matrix["from_station"] = initial_od_matrix["from_station"].astype(int)
    initial_od_matrix.columns = [
        "from_station" if col == "from_station" else int(col)
        for col in initial_od_matrix.columns
    ]
    initial_od_matrix = initial_od_matrix.set_index('from_station')
    if scenario_components is None:
        raise ValueError(
            "Rail OD generation in the integrated pipeline requires shared scenario components. "
            "Pass the saved shared_components_path from generate_and_apply_shared_scenarios()."
        )

    population_scenarios = scenario_components["population_scenarios"]
    modal_split_scenarios = scenario_components["modal_split_rail"]
    distance_per_person_scenarios = scenario_components["distance_per_person"]
    if do_plot:
        pass

    modal_factors, distance_factors = precompute_modal_distance_factors(
        modal_split_scenarios, distance_per_person_scenarios, start_year
    )

    station_communes = build_station_to_communes_mapping(communes_to_stations)
    station_commune_lookup = {}

    def process_scenario(s):
        key = f"scenario_{s + 1}"
        results_s = {}

        for y in range(start_year, end_year + 1):
            pop_growth_od = compute_growth_od_matrix_rail_optimized(
                initial_od_matrix,
                station_communes,
                communes_population,
                population_scenarios,
                s, y, start_year,
                station_commune_lookup
            )

            final_od = apply_modal_trips_optimized(
                initial_od_matrix,
                pop_growth_od,
                modal_factors,
                distance_factors,
                s, start_year, y
            )
            results_s[y] = final_od

        return key, results_s

    try:
        scenario_results = Parallel(n_jobs=n_jobs, verbose=100)(
            delayed(process_scenario)(s) for s in range(num_of_scenarios)
        )
    except PermissionError:
        # Fall back to serial execution when multiprocessing is blocked by the runtime.
        scenario_results = [process_scenario(s) for s in range(num_of_scenarios)]
    return dict(scenario_results)


def load_scenarios_from_cache(cache_dir: str):
    scenarios = {}
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith(".pkl"):
                scenario_name = file.replace(".pkl", "")
                with open(os.path.join(cache_dir, file), "rb") as handle:
                    scenarios[scenario_name] = pickle.load(handle)
    return scenarios


def export_generated_population_rasters(
    scenarios: dict,
    start_year: int,
    end_year: int,
    num_of_scenarios: int,
    valuation_year: int,
    scenario_components: Dict[str, Any] | None = None,
    output_dir: str = ROAD_POPULATION_RASTER_OUTPUT_DIR,
) -> int:
    del end_year, num_of_scenarios
    os.makedirs(output_dir, exist_ok=True)

    base_candidates = [
        os.path.join(output_dir, "pop20_corrected.tif"),
        os.path.join(output_dir, "pop20.tif"),
        os.path.join(integrated_paths.MAIN, "data", "independent_variable", "processed", "raw", "pop20.tif"),
    ]
    base_raster_path = next((p for p in base_candidates if os.path.exists(p)), None)
    if base_raster_path is None:
        raise FileNotFoundError("Cannot create generated population rasters: no base population raster found.")

    with rasterio.open(base_raster_path) as src:
        base_pop = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    commune_raster, _ = GetCommuneShapes(raster_path=base_raster_path)
    communes_population = pd.read_csv(ROAD_POPULATION_BY_COMMUNE_PATH)
    communes_population.columns = communes_population.columns.str.replace("\ufeff", "", regex=False)
    required_cols = {"gemeinde_bfs_nr", "bezirk", "anzahl"}
    if not required_cols.issubset(set(communes_population.columns)):
        raise KeyError("population_by_gemeinde_2018.csv must contain columns: gemeinde_bfs_nr, bezirk, anzahl")

    communes_population = communes_population[["gemeinde_bfs_nr", "bezirk", "anzahl"]].copy()
    communes_population["gemeinde_bfs_nr"] = pd.to_numeric(communes_population["gemeinde_bfs_nr"], errors="coerce")
    communes_population["anzahl"] = pd.to_numeric(communes_population["anzahl"], errors="coerce")
    communes_population = communes_population.dropna(subset=["gemeinde_bfs_nr", "bezirk", "anzahl"])
    communes_population["gemeinde_bfs_nr"] = communes_population["gemeinde_bfs_nr"].astype(int)

    if scenario_components is not None:
        population_scenarios = scenario_components["population_scenarios"]
    else:
        bezirk_pop_scenarios = get_bezirk_population_scenarios()
        population_scenarios = {
            bezirk: generate_population_scenarios(df, start_year, valuation_year, num_of_scenarios)
            for bezirk, df in bezirk_pop_scenarios.items()
        }

    scenario_indices = []
    for scenario_name in scenarios.keys():
        if scenario_name.startswith("scenario_"):
            try:
                scenario_indices.append(int(scenario_name.split("_")[-1]) - 1)
            except ValueError:
                continue
    if not scenario_indices:
        scenario_indices = list(range(num_of_scenarios))

    for filename in os.listdir(output_dir):
        if filename.startswith("scenario_") and "_pop" in filename and filename.endswith(".tif"):
            os.remove(os.path.join(output_dir, filename))

    profile.update(dtype="float32", count=1, nodata=0)
    written = 0
    for scenario_idx in sorted(set(scenario_indices)):
        if scenario_idx < 0:
            continue

        modified_raster = base_pop.copy()
        for row in communes_population.itertuples(index=False):
            bfs = int(row.gemeinde_bfs_nr)
            district = row.bezirk
            district_key = str(district) if str(district) in population_scenarios else district
            scenario_df = population_scenarios.get(district_key)
            if scenario_df is None:
                continue
            scen_slice = scenario_df[scenario_df["scenario"] == scenario_idx]
            if scen_slice.empty:
                continue
            pop_start_rows = scen_slice[scen_slice["year"] == start_year]
            pop_target_rows = scen_slice[scen_slice["year"] == valuation_year]
            if pop_start_rows.empty or pop_target_rows.empty:
                continue

            pop_start = float(pop_start_rows["population"].iloc[0])
            pop_target = float(pop_target_rows["population"].iloc[0])
            growth_factor = (pop_target / pop_start) if pop_start > 0 else 1.0

            commune_mask = commune_raster == bfs
            if np.any(commune_mask):
                modified_raster[commune_mask] *= growth_factor

        scen_name = f"scenario_{scenario_idx + 1}"
        out_year_path = os.path.join(output_dir, f"{scen_name}_pop_{valuation_year}.tif")
        out_alias_path = os.path.join(output_dir, f"{scen_name}_pop.tif")

        with rasterio.open(out_year_path, "w", **profile) as dst:
            dst.write(modified_raster, 1)
        with rasterio.open(out_alias_path, "w", **profile) as dst:
            dst.write(modified_raster, 1)

        written += 1

    return written


def get_random_scenarios(
    start_year: int = 2018,
    end_year: int = 2100,
    num_of_scenarios: int = 100,
    use_cache: bool = False,
    do_plot: bool = False,
    shared_components_path: str | None = None,
):
    cache_dir = ROAD_SCENARIO_CACHE_DIR

    if use_cache:
        scenarios = load_scenarios_from_cache(cache_dir)
        if scenarios:
            export_generated_population_rasters(
                scenarios=scenarios,
                start_year=start_year,
                end_year=end_year,
                num_of_scenarios=num_of_scenarios,
                valuation_year=road_settings.start_valuation_year,
                scenario_components=scenario_components,
            )
            return scenarios

    communes_to_zones = load_or_create_commune_to_zone_mapping(
        mapping_path=ROAD_COMMUNE_TO_ZONE_MAPPING_PATH,
        voronoi_tif_path=ROAD_VORONOI_TIF_PATH,
    )

    scenario_components = None
    if shared_components_path and os.path.exists(shared_components_path):
        with open(shared_components_path, "rb") as handle:
            scenario_components = pickle.load(handle)

    scenarios = generate_od_growth_scenarios(
        pd.read_csv(ROAD_OD_MATRIX_PATH),
        communes_to_zones,
        pd.read_csv(ROAD_POPULATION_BY_COMMUNE_PATH),
        start_year=start_year,
        end_year=end_year,
        num_of_scenarios=num_of_scenarios,
        scenario_components=scenario_components,
        do_plot=do_plot,
    )

    os.makedirs(cache_dir, exist_ok=True)
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pkl") and not filename.startswith("._"):
            os.remove(os.path.join(cache_dir, filename))
    for scenario_name, scenario_data in scenarios.items():
        with open(os.path.join(cache_dir, f"{scenario_name}.pkl"), "wb") as handle:
            pickle.dump(scenario_data, handle)

    export_generated_population_rasters(
        scenarios=scenarios,
        start_year=start_year,
        end_year=end_year,
        num_of_scenarios=num_of_scenarios,
        valuation_year=integrated_settings.start_valuation_year,
        scenario_components=scenario_components,
    )

    return scenarios


def get_rail_random_scenarios(
    start_year: int = 2018,
    end_year: int = 2100,
    num_of_scenarios: int = 100,
    use_cache: bool = False,
    do_plot: bool = False,
    shared_components_path: str | None = None,
):
    cache_dir = RAIL_SCENARIO_CACHE_DIR

    if use_cache:
        scenarios = load_scenarios_from_cache(cache_dir)
        if scenarios:
            return scenarios

    scenario_components = None
    if shared_components_path and os.path.exists(shared_components_path):
        with open(shared_components_path, "rb") as handle:
            scenario_components = pickle.load(handle)
    else:
        raise FileNotFoundError(
            "Integrated rail scenario generation requires a valid shared_components_path."
        )

    scenarios = generate_rail_od_growth_scenarios(
        pd.read_csv(RAIL_OD_MATRIX_PATH),
        pd.read_excel(RAIL_COMMUNE_TO_STATION_PATH),
        pd.read_csv(ROAD_POPULATION_BY_COMMUNE_PATH),
        start_year=start_year,
        end_year=end_year,
        num_of_scenarios=num_of_scenarios,
        scenario_components=scenario_components,
        do_plot=do_plot,
    )

    os.makedirs(cache_dir, exist_ok=True)
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pkl") and not filename.startswith("._"):
            os.remove(os.path.join(cache_dir, filename))
    for scenario_name, scenario_data in scenarios.items():
        scen_path = os.path.join(cache_dir, f"{scenario_name}.pkl")
        with open(scen_path, 'wb') as handle:
            pickle.dump(scenario_data, handle)

    return scenarios

# -----------------------------------------------------------------
# Shared outputs, selection and plotting
# -----------------------------------------------------------------

def build_shared_scenario_components(
    start_year: int,
    end_year: int,
    num_of_scenarios: int,
) -> Dict[str, Any]:
    bezirk_pop_scenarios = get_bezirk_population_scenarios()
    population_scenarios = {
        bezirk: generate_population_scenarios(
            df,
            start_year,
            end_year,
            num_of_scenarios,
        )
        for bezirk, df in bezirk_pop_scenarios.items()
    }

    # The integrated pipeline uses the joint logistic-normal generator so the
    # three modal split paths are produced together and remain internally coherent.
    modal_split_rail, modal_split_road, modal_split_other = generate_joint_modal_split_scenarios(
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
    """
    Determines a set of key indicators for all generated scenarios in the valuation year and compares them to the start year.
    Is used to select a representative subset of scenarios for the road and rail scenario generation steps.
    """
    meta = components["meta"]
    start_year = int(meta["start_year"])
    num_of_scenarios = int(meta["num_of_scenarios"])

    population_scenarios = components["population_scenarios"]
    modal_split_road = components["modal_split_road"]
    modal_split_rail = components["modal_split_rail"]
    modal_split_other = components.get("modal_split_other")
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
        other_modal = (
            _get_value_for_scenario_year(modal_split_other, scenario_idx, valuation_year, "modal_split")
            if modal_split_other is not None
            else max(0.0, 1.0 - road_modal - rail_modal)
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
        lower_q = 0.25
        upper_q = 0.75

        selected_positions = [
            round(
                (lower_q + idx * (upper_q - lower_q) / (n_representatives - 1))
                * (len(sorted_df) - 1)
            )
            for idx in range(n_representatives)
        ]

    selected_rows = []
    used_positions = set()

    for order, target_pos in enumerate(selected_positions, start=1):

        if target_pos not in used_positions:
            chosen_pos = target_pos
        else:
            candidate_positions = sorted(
                range(len(sorted_df)),
                key=lambda pos: (abs(pos - target_pos), pos),
            )
            chosen_pos = next(
                pos for pos in candidate_positions
                if pos not in used_positions
            )

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
            plot_base_dir = os.path.dirname(summary_file) or integrated_paths.SCENARIO_CACHE_SHARED_DIR
            plot_dir = os.path.join(plot_base_dir, "plots")
            plot_shared_scenario_components(components, summary_df, selected_df, plot_dir)

        return {
            "components_path": saved_path,
            "summary_path": summary_file,
            "selection_path": selection_file,
            "selected_scenarios": selected_scenarios,
            "selected_summary": selected_df,
        }
    finally:
        os.chdir(previous_cwd)


def _plot_scenario_band(df: pd.DataFrame, value_column: str, title: str, output_path: str) -> None:
    year_stats = (
        df.groupby("year")[value_column]
        .agg(
            min="min",
            max="max",
            mean="mean",
            std="std",
            q05=lambda x: x.quantile(0.05),
            q95=lambda x: x.quantile(0.95),
        )
        .reset_index()
    )
    std = year_stats["std"].fillna(0.0)
    year_stats["upper_195"] = year_stats["mean"] + 1.95 * std
    year_stats["lower_195"] = year_stats["mean"] - 1.95 * std

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.fill_between(year_stats["year"], year_stats["min"], year_stats["max"], alpha=0.22, color="#7f8c8d", label="Total range")
    ax.plot(year_stats["year"], year_stats["mean"], color="#2c3e50", linestyle="--", linewidth=1.8, label="Mean")
    ax.plot(year_stats["year"], year_stats["q95"], color="#c0392b", linewidth=1.2, label="95% quantile")
    ax.plot(year_stats["year"], year_stats["q05"], color="#c0392b", linewidth=1.2, label="5% quantile")
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
    _plot_scenario_band(
        components["modal_split_rail"],
        "modal_split",
        "Rail modal split scenarios",
        os.path.join(output_dir, "modal_split_rail.png"),
    )
    _plot_scenario_band(
        components["modal_split_road"],
        "modal_split",
        "Road modal split scenarios",
        os.path.join(output_dir, "modal_split_road.png"),
    )
    if "modal_split_other" in components:
        _plot_scenario_band(
            components["modal_split_other"],
            "modal_split",
            "Other modal split scenarios",
            os.path.join(output_dir, "modal_split_other.png"),
        )
    _plot_scenario_band(
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
