import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from tqdm import tqdm
from scipy.stats import norm, qmc
import numpy as np
from typing import Any, Dict, List, Optional
from joblib import Parallel, delayed
import pickle
import os
import rasterio

from .scoring import GetCommuneShapes
from . import settings
from infraScan.infraScanIntegrated import settings as integrated_settings
from infraScan.infraScanIntegrated import paths as integrated_paths

def get_bezirk_population_scenarios():
    # Read the Swiss population scenario CSV with "," separator
    df_ch = pd.read_csv("data/Scenario/pop_scenario_switzerland_2055.csv", sep=",")
    # Extract the relevant values for 2018 and 2050
    pop_2018 = df_ch.loc[df_ch['Jahr'] == 2018, 'Beobachtungen'].values
    pop_2050 = df_ch.loc[df_ch['Jahr'] == 2050, 'Referenzszenario A-00-2025'].values
    # Compute the growth factor: population_2050 / population_2018
    swiss_growth_factor_18_50 = pop_2050[0] / pop_2018[0]
    # Read the CSV file with ";" as separator
    df = pd.read_csv("data/Scenario/KTZH_00000705_00001741.csv", sep=';', encoding='utf-8')
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
    eurostat_df = pd.read_excel("data/Scenario/Eurostat_population_CH_2100.xlsx")
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


def generate_modal_split_scenarios(avg_growth_rate: float,
                                   start_value: float,
                                   start_year: int,
                                   end_year: int,
                                   n_scenarios: int = 1000,
                                   start_std_dev: float = 0.01,
                                   end_std_dev: float = 0.03,
                                   std_dev_shocks: float = 0.02) -> pd.DataFrame:
    """
    Generate stochastic modal split scenarios using Latin Hypercube Sampling and a random walk process.

    Parameters:
    - avg_growth_rate: average annual growth rate to apply (can be positive or negative)
    - start_value: initial modal split value at start_year
    - start_year: year to begin scenario generation
    - end_year: year to end scenario generation
    - n_scenarios: number of scenarios to generate
    - start_std_dev: starting std deviation applied to growth rate perturbation
    - end_std_dev: ending std deviation applied to growth rate perturbation
    - std_dev_shocks: std deviation of yearly additive shocks

    Returns:
    - DataFrame with columns: "scenario", "year", "modal_split", "growth_rate", "growth_index"
    """
    # Erstelle temporären Referenzdatensatz mit konstanter Wachstumsrate
    years = np.arange(start_year, end_year + 1)
    n_years = len(years)

    # Berechne die Werte mit konstanter Wachstumsrate
    growth_factors = np.ones(n_years) * (1 + avg_growth_rate)
    growth_factors[0] = 1  # Erster Faktor ist 1, da es der Startwert ist

    # Kumulatives Wachstum berechnen
    cumulative_growth = np.cumprod(growth_factors)
    modal_split_values = start_value * cumulative_growth

    # Erstelle Array von Wachstumsraten (erster Wert ist 0, danach konstant)
    growth_rates = np.zeros(n_years)
    growth_rates[1:] = avg_growth_rate  # Konstante Wachstumsrate für alle Jahre außer dem ersten

    # Erstelle temporären DataFrame
    ref_df = pd.DataFrame({
        "jahr": years,
        "total_population": modal_split_values,
        "growth_rate": growth_rates
    })

    # Verwende die bestehende Funktion
    modal_split_scenarios_df = generate_population_scenarios(
        ref_df=ref_df,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
        start_std_dev=start_std_dev,
        end_std_dev=end_std_dev,
        std_dev_shocks=std_dev_shocks
    )

    # Umbenennen der Spalte "population" in "modal_split"
    modal_split_scenarios_df = modal_split_scenarios_df.rename(columns={"population": "modal_split"})

    return modal_split_scenarios_df


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
    # Nutze die bestehende Funktion für Modal-Split-Szenarien
    scenarios_df = generate_modal_split_scenarios(
        avg_growth_rate=avg_growth_rate,
        start_value=start_value,
        start_year=start_year,
        end_year=end_year,
        n_scenarios=n_scenarios,
        start_std_dev=start_std_dev,
        end_std_dev=end_std_dev,
        std_dev_shocks=std_dev_shocks
    )

    # Benenne die Spalte "modal_split" in "trips_per_person" um
    scenarios_df = scenarios_df.rename(columns={"modal_split": "distance_per_person"})

    return scenarios_df


def plot_population_scenarios(scenarios_df: pd.DataFrame, n_to_plot: int = 10):
    """
    Plot a sample of population scenarios.

    Parameters:
    - scenarios_df: DataFrame with columns "scenario", "year", "population"
    - n_to_plot: number of scenarios to randomly plot
    """
    plt.figure(figsize=(10, 6),dpi=300)

    sample_ids = scenarios_df["scenario"].drop_duplicates().sample(n=min(n_to_plot, scenarios_df["scenario"].nunique()))
    sample_df = scenarios_df[scenarios_df["scenario"].isin(sample_ids)]

    for scenario_id in sample_df["scenario"].unique():
        data = sample_df[sample_df["scenario"] == scenario_id]
        plt.plot(data["year"], data["population"] / 1e3, label=f"Scenario {scenario_id}")

    plt.xlabel("Year")
    plt.ylabel("Population (thousands)")
    plt.title(f"Sample of {n_to_plot} Population Scenarios")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_scenarios_with_range(
        scenarios_df: pd.DataFrame,
        save_path,
        value_col: str = "population"
):
    """
    Plot the range of all scenarios for a given value column as a shaded area
    and a single example scenario, with automatically scaled SI-prefix axis.
    Also plots lines for +/- 1.65 standard deviations from the mean (contains 90% of all values).

    Parameters:
    - scenarios_df: DataFrame with columns "scenario", "year", and the specified value column
    - save_path: path where the plot will be saved
    - value_col: name of the column in scenarios_df containing the values to plot
    """
    # compute per-year stats
    year_stats = (
        scenarios_df
        .groupby("year")[value_col]
        .agg(min="min", max="max", mean="mean", std="std")
        .reset_index()
    )

    # Berechne +/- 1.65 Standardabweichungen (90% Konfidenzintervall)
    year_stats["mean_plus_1_65std"] = year_stats["mean"] + 1.65 * year_stats["std"]
    year_stats["mean_minus_1_65std"] = year_stats["mean"] - 1.65 * year_stats["std"]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    # shaded range
    ax.fill_between(
        year_stats["year"],
        year_stats["min"],
        year_stats["max"],
        color='grey', alpha=0.3,
        label="Total Range"
    )

    # +/- 1.65 Std. Abw. Linien (90% Konfidenzintervall)
    ax.plot(
        year_stats["year"],
        year_stats["mean_plus_1_65std"],
        color='red', linestyle='-', alpha=0.7,
        label="+1.65σ (95th percentile)"
    )

    ax.plot(
        year_stats["year"],
        year_stats["mean_minus_1_65std"],
        color='red', linestyle='-', alpha=0.7,
        label="-1.65σ (5th percentile)"
    )

    # mean line
    ax.plot(
        year_stats["year"],
        year_stats["mean"],
        color='grey', linestyle='--', alpha=0.8,
        label="Mean"
    )

    # pick a random scenario to highlight
    sample_id = scenarios_df["scenario"].drop_duplicates().sample(n=1).iloc[0]
    sample_df = scenarios_df[scenarios_df["scenario"] == sample_id]
    ax.plot(
        sample_df["year"],
        sample_df[value_col],
        color='blue', linewidth=2,
        label=f"Sample Scenario {sample_id}"
    )

    # apply automatic SI‐prefix scaling on the Y axis
    ax.yaxis.set_major_formatter(EngFormatter(unit='', places=2))

    # labels & styling
    col_title = value_col.replace('_', ' ').title()
    ax.set_xlabel("Year")
    ax.set_title(f"{col_title} Scenarios: Range, Mean and 90% Confidence Interval")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    # Save the plot, creating a filename based on the value column
    filename = f"{value_col.lower().replace(' ', '_')}_scenarios.png"
    full_path = os.path.join(save_path, filename)
    plt.savefig(full_path)
    # plt.show()
    plt.close(fig)


def build_commune_to_zone_mapping_from_raster_overlap(
        voronoi_tif_path: str,
        output_csv_path: str
) -> pd.DataFrame:
    """
    Build weighted commune -> zone mapping from raster overlap.
    Weight = share of commune cells lying in each zone.
    """
    commune_raster, commune_df = GetCommuneShapes(raster_path=voronoi_tif_path)

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
                records.append({
                    "zone_id": int(zone_id),
                    "commune_bfs": int(commune_id),
                    "weight": overlap_cells / commune_cells
                })

    mapping = pd.DataFrame(records)
    mapping.to_csv(output_csv_path, index=False)

    return mapping
   
def load_or_create_commune_to_zone_mapping(
        mapping_path: str,
        voronoi_tif_path: str
) -> pd.DataFrame:
    if os.path.exists(mapping_path):
        return pd.read_csv(mapping_path)

    if not os.path.exists(voronoi_tif_path):
        raise FileNotFoundError(
            f"Neither mapping file nor raster found. Missing: {mapping_path} and/or {voronoi_tif_path}"
        )


    return build_commune_to_zone_mapping_from_raster_overlap(voronoi_tif_path=voronoi_tif_path, output_csv_path=mapping_path)


def build_zone_to_communes_mapping(
        communes_to_zones: pd.DataFrame
) -> Dict[str, list]:
    """
    Baut ein Mapping: zone_id -> List[Tuple(Commune_BFS_code, weight)].
    """
    communes_to_zones = communes_to_zones.copy()
    communes_to_zones["zone_id"] = communes_to_zones["zone_id"].astype(int).astype(str)
    communes_to_zones["commune_bfs"] = communes_to_zones["commune_bfs"].astype(int)

    zone_map = {}
    for zone_id, group in communes_to_zones.groupby("zone_id"):
        zone_map[zone_id] = list(zip(
            group["commune_bfs"].tolist(),
            group["weight"].tolist()
        ))
    return zone_map


def compute_growth_od_matrix_optimized(
        initial_od: pd.DataFrame,
        zone_communes: Dict[str, List[int]],
        communes_population: pd.DataFrame,
        population_scenarios: Dict[str, pd.DataFrame],
        scenario: int,
        year: int,
        start_year: int,
        zone_commune_lookup: Dict[str, pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Optimierte Version von compute_growth_od_matrix
    """
    # --- 1) Struktur aufsetzen
    zones = initial_od.columns.tolist()
    from_zones = initial_od.index.tolist()

    # --- 2) Wachstumsindex für alle zoneen berechnen
    zone_growth = {}

    # Wenn lookup nicht existiert, erstellen wir einen
    if zone_commune_lookup is None:
        zone_commune_lookup = {}

    for zone in zones:
        # Lookup für diese zone verwenden wenn vorhanden
        if zone not in zone_commune_lookup:
            communes = zone_communes.get(zone, [])
            if not communes:
                zone_growth[zone] = 1.0
                continue

            # Vorfiltern der relevanten Gemeinden für diese zone
            zone_data = []
            for commune, weight in communes:
                row = communes_population[communes_population['gemeinde_bfs_nr'] == commune]
                if not row.empty:
                    district = row['bezirk'].iat[0]
                    pop_start_commune = row['anzahl'].iat[0]
                    zone_data.append((commune, district, pop_start_commune, weight))

            zone_commune_lookup[zone] = zone_data

        sum_start = 0.0
        sum_curr = 0.0

        for commune, district, pop_start_commune, weight in zone_commune_lookup[zone]:
            # Population im Szenario für Start- und Ziel-Jahr
            scen = population_scenarios[str(district) if str(district) in population_scenarios else district]

            # Effizienterer Zugriff mit vorgefilterten Daten
            scenario_data = scen[(scen['scenario'] == scenario)]
            pop_d_start = scenario_data[scenario_data['year'] == start_year]['population'].iloc[0]
            pop_d_curr = scenario_data[scenario_data['year'] == year]['population'].iloc[0]

            # Bezirksfaktor
            factor_d = (pop_d_curr / pop_d_start) if pop_d_start > 0 else 1.0

            weighted_base = pop_start_commune * weight
            sum_start += weighted_base
            sum_curr += weighted_base * factor_d

        zone_growth[zone] = (sum_curr / sum_start) if sum_start > 0 else 1.0

    # --- 3) OD-Matrix mit Wachstumsfaktoren erzeugen
    growth_od = pd.DataFrame(1.0, index=from_zones, columns=zones)

    # Vektorisierte Anwendung der Faktoren
    sqrt_factors = {zone: np.sqrt(factor) for zone, factor in zone_growth.items()}

    # Row-wise scaling
    for zone in map(str, from_zones):
        if zone in sqrt_factors:
            growth_od.loc[zone, :] *= sqrt_factors[zone]

    # Column-wise scaling
    for zone in zones:
        if zone in sqrt_factors:
            growth_od.loc[:, zone] *= sqrt_factors[zone]

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
    scenario_components: Optional[Dict[str, Any]] = None,
        do_plot: bool = False,
        n_jobs: int = -1
) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Optimierte Version von generate_od_growth_scenarios mit Multiprocessing
    """

    # first column = from_zone, all zone ids = stringified integers
    initial_od_matrix = initial_od_matrix.rename(columns={"voronoi_id": "from_zone"}).copy()

    initial_od_matrix["from_zone"] = initial_od_matrix["from_zone"].astype(float).astype(int).astype(str)

    initial_od_matrix.columns = [
        "from_zone" if col == "from_zone" else str(int(float(col)))
        for col in initial_od_matrix.columns
    ]

    initial_od_matrix = initial_od_matrix.set_index("from_zone")

    # Ensure commone columns are strings and remove any potential BOM characters
    communes_population.columns = communes_population.columns.str.replace("\ufeff", "", regex=False)

    if scenario_components is None:
        raise ValueError(
            "Road OD generation requires shared scenario components so the tessellated road demand "
            "uses the same integrated modal split and distance assumptions."
        )

    population_scenarios = scenario_components["population_scenarios"]
    modal_split_scenarios = scenario_components["modal_split_road"]
    distance_per_person_scenarios = scenario_components["distance_per_person"]
    if do_plot:
        first_three_bezirk = list(population_scenarios.keys())[:3]
        first_three_scenarios = {bezirk: population_scenarios[bezirk] for bezirk in first_three_bezirk}
        for pop_scenario in first_three_scenarios.values():
            plot_scenarios_with_range(pop_scenario, "plots/scenarios/plots/", 'population')
        plot_scenarios_with_range(modal_split_scenarios, "plots/scenarios/plots/", 'modal_split')
        plot_scenarios_with_range(distance_per_person_scenarios, "plots/scenarios/plots/", 'distance_per_person')

    # components = {
    #     "population_scenarios": population_scenarios,
    #     "modal_split_scenarios": modal_split_scenarios,
    #     "distance_per_person_scenarios": distance_per_person_scenarios
    # }


    # Speichere alle Komponenten in einer Datei
    #with open("scenario_data_for_plots.pkl", 'wb') as f:
    #    pickle.dump(components, f)
    # Vorauswertung aller Modal/Distance Faktoren
    print("Berechne Modal und Distance Faktoren...")

    modal_factors, distance_factors = precompute_modal_distance_factors(
        modal_split_scenarios, distance_per_person_scenarios, start_year
    )

    # 3) Zone→Commune-Mapping (einmalig)
    print("Erstelle Zone-Commune Mapping...")
    zone_communes = build_zone_to_communes_mapping(communes_to_zones)
    zone_commune_lookup = {}  # Cache für zone-commune Beziehungen

    # 4) Funktion für parallele Verarbeitung
    def process_scenario(s):
        key = f"scenario_{s + 1}"
        results_s = {}

        for y in range(start_year, end_year + 1):
            pop_growth_od = compute_growth_od_matrix_optimized(
                initial_od_matrix,
                zone_communes,
                communes_population,
                population_scenarios,
                s, y, start_year,
                zone_commune_lookup
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

    # 5) Parallele Verarbeitung mit Fortschrittsbalken
    print(f"Berechne {num_of_scenarios} Szenarien mit {n_jobs} Prozessen...")
    scenario_results = Parallel(n_jobs=n_jobs, verbose = 100 )( #backend="loky", max_nbytes=None
        delayed(process_scenario)(s) for s in range(num_of_scenarios)
    )
    #scenario_results = [process_scenario(s) for s in range(num_of_scenarios)]
    print("fertig")
    # 6) Ergebnisse zusammenführen
    results = dict(scenario_results)
    return results


# bezirk_pop_scen = get_bezirk_population_scenarios()
# affoltern_df = bezirk_pop_scen['Affoltern']
# pop_scenarios_df = generate_population_scenarios(affoltern_df, 2022, 2100,n_scenarios=100, start_std_dev=0.005, end_std_dev=0.01, std_dev_shocks=0.02)
#
# plot_population_scenarios(pop_scenarios_df, n_to_plot=100)
# ms_scenario_df = generate_modal_split_scenarios(0.0045, 0.209, 2022, 2100, n_scenarios=100, start_std_dev=0.002, end_std_dev=0.005, std_dev_shocks=0.01)#growth rate assumption from verkehrsperspektiven 2017 til 2060
# trips_per_person_scenario_df = generate_distance_per_person_scenarios(-0.0027, 39.79, 2022, 2100, n_scenarios=100, start_std_dev=0.002, end_std_dev=0.005, std_dev_shocks=0.01)#growth rate assumption from verkehrsperspektiven 2017 til 2050 via gesamte verkehrsleistung, computed with chatgpt
#
# plot_scenarios_with_range(pop_scenarios_df,'population')
# plot_scenarios_with_range(ms_scenario_df, 'modal_split')
# plot_scenarios_with_range(trips_per_person_scenario_df, 'distance_per_person')



def load_scenarios_from_cache(cache_dir):
    """
    Load scenarios from individual .pkl files in the cache directory.
    
    Parameters:
    - cache_dir: Directory containing scenario .pkl files
    
    Returns:
    - Dictionary of loaded scenarios
    """
    scenarios = {}
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl'):
                scenario_name = file.replace('.pkl', '')
                with open(os.path.join(cache_dir, file), 'rb') as f:
                    scenarios[scenario_name] = pickle.load(f)
        print(f"Loaded {len(scenarios)} scenarios from {cache_dir}")
    return scenarios

def export_generated_population_rasters(
        scenarios: dict,
        start_year: int,
        end_year: int,
        num_of_scenarios: int,
        valuation_year: int,
        scenario_components: Optional[Dict[str, Any]] = None,
        output_dir: str = "data/independent_variable/processed/scenario",
) -> int:
    """
    Create generated population rasters in the same spirit as `scenario_to_raster`:
    start from a base population raster and apply scenario-specific growth factors
    spatially via commune masks.

    Output naming:
      - scenario_X_pop_<valuation_year>.tif
      - scenario_X_pop.tif
    """
    os.makedirs(output_dir, exist_ok=True)

    base_candidates = [
        os.path.join(output_dir, "pop20_corrected.tif"),
        os.path.join(output_dir, "pop20.tif"),
        "data/independent_variable/processed/raw/pop20.tif",
    ]
    base_raster_path = next((p for p in base_candidates if os.path.exists(p)), None)
    if base_raster_path is None:
        raise FileNotFoundError(
            "Cannot create generated population rasters: no base population raster found "
            "(expected pop20_corrected.tif/pop20.tif/raw pop20.tif)."
        )

    with rasterio.open(base_raster_path) as src:
        base_pop = src.read(1).astype(np.float32)
        profile = src.profile.copy()

    # Commune raster on same grid as base raster
    commune_raster, _ = GetCommuneShapes(raster_path=base_raster_path)

    communes_population = pd.read_csv("data/Scenario/population_by_gemeinde_2018.csv")
    communes_population.columns = communes_population.columns.str.replace("\ufeff", "", regex=False)
    required_cols = {"gemeinde_bfs_nr", "bezirk", "anzahl"}
    if not required_cols.issubset(set(communes_population.columns)):
        raise KeyError(
            "population_by_gemeinde_2018.csv must contain columns: "
            "gemeinde_bfs_nr, bezirk, anzahl"
        )

    communes_population = communes_population[["gemeinde_bfs_nr", "bezirk", "anzahl"]].copy()
    communes_population["gemeinde_bfs_nr"] = pd.to_numeric(communes_population["gemeinde_bfs_nr"], errors="coerce")
    communes_population["anzahl"] = pd.to_numeric(communes_population["anzahl"], errors="coerce")
    communes_population = communes_population.dropna(subset=["gemeinde_bfs_nr", "bezirk", "anzahl"])
    communes_population["gemeinde_bfs_nr"] = communes_population["gemeinde_bfs_nr"].astype(int)

    if scenario_components is not None:
        population_scenarios = scenario_components["population_scenarios"]
    else:
        # Build district population scenarios (same stochastic engine as OD generation)
        bezirk_pop_scenarios = get_bezirk_population_scenarios()
        population_scenarios = {
            bezirk: generate_population_scenarios(df, start_year, end_year, num_of_scenarios)
            for bezirk, df in bezirk_pop_scenarios.items()
        }

    # Infer which scenario indices are present
    scenario_indices = []
    for scenario_name in scenarios.keys():
        if scenario_name.startswith("scenario_"):
            try:
                scenario_indices.append(int(scenario_name.split("_")[-1]) - 1)
            except ValueError:
                continue

    if not scenario_indices:
        scenario_indices = list(range(num_of_scenarios))

    # Cleanup previous generated population rasters
    for filename in os.listdir(output_dir):
        if filename.startswith("scenario_") and "_pop" in filename and filename.endswith(".tif"):
            os.remove(os.path.join(output_dir, filename))

    profile.update(dtype="float32", count=1, nodata=0)

    written = 0
    for scenario_idx in sorted(set(scenario_indices)):
        if scenario_idx < 0:
            continue

        # same approach as growth_to_tif: modify a copy of base raster per scenario
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

    print(
        f"Generated population rasters written={written} for valuation_year={valuation_year} "
        f"in {output_dir}"
    )

    return written



def get_random_scenarios(start_year=2018, end_year=2100, num_of_scenarios=100, use_cache=False, do_plot=False, shared_components_path=None):
    """
    Retrieve or generate random OD growth scenarios.

    Parameters:
    - start_year: The starting year for the scenarios.
    - end_year: The ending year for the scenarios.
    - num_of_scenarios: The number of scenarios to generate.
    - use_cache: If True, load scenarios from cache instead of regenerating.
    - do_plot: If True, plot the scenarios after generation.

    Returns:
    - scenarios: Dict of generated scenarios.
    """
    cache_dir = "data/Scenario/cache/road/random"
    resolved_shared_components_path = shared_components_path or integrated_paths.SHARED_COMPONENTS_PATH
    scenario_components = None

    if os.path.exists(resolved_shared_components_path):
        with open(resolved_shared_components_path, "rb") as f:
            scenario_components = pickle.load(f)
    else:
        from infraScan.infraScanIntegrated.random_scenarios import (
            build_shared_scenario_components,
            save_shared_scenario_components,
        )

        scenario_components = build_shared_scenario_components(
            start_year=start_year,
            end_year=end_year,
            num_of_scenarios=num_of_scenarios,
        )
        save_shared_scenario_components(
            scenario_components,
            output_path=resolved_shared_components_path,
        )

    if use_cache:
        scenarios = load_scenarios_from_cache(cache_dir)

        if scenarios:
            export_generated_population_rasters(
                scenarios=scenarios,
                start_year=start_year,
                end_year=end_year,
                num_of_scenarios=num_of_scenarios,
                valuation_year=settings.start_valuation_year,
                scenario_components=scenario_components,
            )
            return scenarios
        else:
            print("Scenario cache is empty; regenerating scenarios.")

    # Generate new scenarios
    communes_to_zones = load_or_create_commune_to_zone_mapping(
        mapping_path= "data/infraScanRoad/Scenario/commune_to_zone_mapping.csv",
        voronoi_tif_path= "data/infraScanRoad/Network/travel_time/source_id_raster.tif"
    )

    scenarios = generate_od_growth_scenarios(
        pd.read_csv("data/infraScanRoad/traffic_flow/od/od_matrix_20.csv"),
        communes_to_zones,
        pd.read_csv("data/Scenario/population_by_gemeinde_2018.csv"),
        start_year=start_year,
        end_year=end_year,
        num_of_scenarios=num_of_scenarios,
        scenario_components=scenario_components,
        do_plot=do_plot
    )

    # Save to cache
    os.makedirs(cache_dir, exist_ok=True)  # Ensure directory exists

    # Empty the directory first (only .pkl files)
    for filename in os.listdir(cache_dir):
        if filename.endswith(".pkl") and not filename.startswith("._"):
            os.remove(os.path.join(cache_dir, filename))
    # Save each scenario to a separate .pkl file
    for scenario_name, scenario_data in scenarios.items():
        scen_path = os.path.join(cache_dir, f"{scenario_name}.pkl")
        with open(scen_path, 'wb') as f:
            pickle.dump(scenario_data, f)
    print(f"Saved {len(scenarios)} scenarios to {cache_dir}")

    export_generated_population_rasters(
        scenarios=scenarios,
        start_year=start_year,
        end_year=end_year,
        num_of_scenarios=num_of_scenarios,
        valuation_year=settings.start_valuation_year,
        scenario_components=scenario_components,
    )

    return scenarios
