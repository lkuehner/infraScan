import math
import sys
import os
import zipfile
import timeit

os.environ['USE_PYGEOS'] = '0'
import geopandas as gpd
import pandas as pd
import numpy as np
import osmnx as ox
import scipy.io
from scipy.interpolate import griddata
from scipy.optimize import minimize, Bounds, least_squares
import rasterio
from rasterio.transform import from_origin
from rasterio.features import geometry_mask, shapes, rasterize
from shapely.geometry import Point, Polygon, box, shape, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
from rasterio.mask import mask
from tqdm import tqdm
import matplotlib.pyplot as plt
import re
import networkx as nx
from itertools import islice

from infraScan.infraScanRoad import settings
from infraScan.infraScanRoad.scoring import * 


def legacy_od_raster_tts(VTTS, duration):
    """ 
    Monetize raster-basis OD travel-time savings

    Main idea:
    1. Keep a stable raster basis (same raster cells in SQ and development).
    2. Keep the original commune-to-commune OD demand as the demand basis.
    3. Recompute commune-to-catchment shares from raster population/employment
       separately for the SQ network and for each development network.
    4. Use routed catchment-pair travel times from phase 6 as tt(c(a), c(b)).
    5. Weight those catchment-pair travel times with the redistributed commune
       OD demand to get total weighted travel time per scenario/development.

    Inputs:
        - data/infraScanRoad/traffic_flow/od/status_quo_od_tt.csv
        - data/infraScanRoad/traffic_flow/od/developments_od_tt.csv

    Outputs:
        - data/infraScanRoad/traffic_flow/od/od_tt_savings_raster_detailed.csv
        - data/infraScanRoad/costs/traveltime_savings_od_raster.csv
    """
    # 1) Legacy raster variant: stable raster basis, but no explicit access/egress breakdown.
    if settings.scenario_type != "STATIC":
        raise NotImplementedError(
            "monetize_tts_by_od_raster currently implements the validated "
            "STATIC raster-basis workflow only."
        )

    status_quo_path = "data/infraScanRoad/traffic_flow/od/status_quo_od_tt.csv"
    developments_path = "data/infraScanRoad/traffic_flow/od/developments_od_tt.csv"
    sq_source_path = "data/infraScanRoad/Network/travel_time/source_id_raster.tif"
    pop_raster_path = "data/independent_variable/processed/scenario/scen_pop.tif"
    empl_raster_path = "data/independent_variable/processed/scenario/scen_empl.tif"

    tt_status_quo = pd.read_csv(status_quo_path)
    tt_developments = pd.read_csv(developments_path)

    required_sq_cols = {"scenario", "origin", "destination", "travel_time"}
    required_dev_cols = {"development", "scenario", "origin", "destination", "travel_time"}
    missing_sq = required_sq_cols.difference(tt_status_quo.columns)
    missing_dev = required_dev_cols.difference(tt_developments.columns)
    if missing_sq:
        raise ValueError(
            f"Missing required columns in {status_quo_path}: {sorted(missing_sq)}"
        )
    if missing_dev:
        raise ValueError(
            f"Missing required columns in {developments_path}: {sorted(missing_dev)}"
        )

    # Keep numeric consistency in the routed catchment-pair TT tables.
    tt_status_quo["scenario"] = tt_status_quo["scenario"].astype(str)
    tt_developments["scenario"] = tt_developments["scenario"].astype(str)
    for col in ["origin", "destination", "travel_time"]:
        tt_status_quo[col] = pd.to_numeric(tt_status_quo[col], errors="coerce")
    for col in ["development", "origin", "destination", "travel_time"]:
        tt_developments[col] = pd.to_numeric(tt_developments[col], errors="coerce")

    # Stable raster basis:
    # - sq_source gives the SQ road catchment id for each raster cell.
    # - commune_raster gives the commune id for each raster cell on the same grid.
    with rasterio.open(sq_source_path) as src:
        sq_source = src.read(1)
    commune_raster, _ = GetCommuneShapes(raster_path=sq_source_path)

    # Original commune-to-commune demand basis.
    od = GetHighwayPHDemandPerCommune()
    odmat = GetODMatrix(od).astype(float)
    od_long = odmat.stack().rename("od_demand").reset_index()
    od_long.columns = ["origin_commune", "destination_commune", "od_demand"]
    od_long = od_long[
        (od_long["origin_commune"] != od_long["destination_commune"]) &
        (od_long["od_demand"] > 0)
    ].copy()

    # STATIC scenario rasters use fixed bands.
    band_map = {"medium": 1, "low": 2, "high": 3}
    scenarios = [scen for scen in ["low", "medium", "high"] if scen in set(tt_developments["scenario"])]

    def shares_by_commune_and_catchment(df):
        """
        Convert raster population/employment mass into commune->catchment shares.

        origin_share:
            share of a commune's origin mass (population) that lies in a given catchment
        dest_share:
            share of a commune's destination mass (employment) that lies in a given catchment
        """ 
        origin = (
            df.groupby(["commune_id", "catchment"], as_index=False)["pop"]
            .sum()
            .rename(columns={"catchment": "origin_catchment", "pop": "origin_mass"})
        )
        dest = (
            df.groupby(["commune_id", "catchment"], as_index=False)["empl"]
            .sum()
            .rename(columns={"catchment": "destination_catchment", "empl": "dest_mass"})
        )

        origin["origin_total"] = origin.groupby("commune_id")["origin_mass"].transform("sum")
        dest["dest_total"] = dest.groupby("commune_id")["dest_mass"].transform("sum")

        origin["origin_share"] = np.where(
            origin["origin_total"] > 0,
            origin["origin_mass"] / origin["origin_total"],
            0.0,
        )
        dest["dest_share"] = np.where(
            dest["dest_total"] > 0,
            dest["dest_mass"] / dest["dest_total"],
            0.0,
        )

        origin = origin.rename(columns={"commune_id": "origin_commune"})[
            ["origin_commune", "origin_catchment", "origin_share"]
        ]
        dest = dest.rename(columns={"commune_id": "destination_commune"})[
            ["destination_commune", "destination_catchment", "dest_share"]
        ]
        return origin, dest

    rows = []
    development_ids = sorted(
        int(dev) for dev in tt_developments["development"].dropna().astype(int).unique()
    )

    # 2) For each scenario, distribute commune OD demand to SQ catchment pairs.
    for scen in scenarios:
        band = band_map[scen]

        # Load the scenario-specific population and employment rasters on the stable grid.
        with rasterio.open(pop_raster_path) as src:
            pop = src.read(band).astype(float)
        with rasterio.open(empl_raster_path) as src:
            empl = src.read(band).astype(float)

        # SQ catchment-pair travel times for this scenario.
        sq_tt = (
            tt_status_quo[tt_status_quo["scenario"] == scen][["origin", "destination", "travel_time"]]
            .rename(
                columns={
                    "origin": "origin_catchment",
                    "destination": "destination_catchment",
                    "travel_time": "tt_sq",
                }
            )
        )

        # Build raster -> commune -> SQ catchment allocation.
        valid_sq = (
            (sq_source > 0) &
            (commune_raster > 0) &
            np.isfinite(pop) &
            np.isfinite(empl) &
            (pop >= 0) &
            (empl >= 0)
        )
        sq_df = pd.DataFrame(
            {
                "commune_id": commune_raster[valid_sq].astype(int),
                "catchment": sq_source[valid_sq].astype(int),
                "pop": pop[valid_sq],
                "empl": empl[valid_sq],
            }
        )
        sq_origin, sq_dest = shares_by_commune_and_catchment(sq_df)

        # Redistribute commune OD demand over SQ catchment pairs.
        sq_pairs = (
            od_long
            .merge(sq_origin, on="origin_commune", how="left")
            .merge(sq_dest, on="destination_commune", how="left")
            .merge(sq_tt, on=["origin_catchment", "destination_catchment"], how="left")
        )
        sq_pairs = sq_pairs.dropna(subset=["origin_share", "dest_share", "tt_sq"]).copy()
        sq_pairs["pair_demand"] = (
            sq_pairs["od_demand"] * sq_pairs["origin_share"] * sq_pairs["dest_share"]
        )
        sq_pairs["weighted_tt"] = sq_pairs["pair_demand"] * sq_pairs["tt_sq"]
        sq_total = float(sq_pairs["weighted_tt"].sum())

        # 3) Reassign the same demand to development catchments and compare routed TT only.
        for dev_id in development_ids:
            dev_source_path = (
                f"data/infraScanRoad/Network/travel_time/developments/dev{dev_id}_source_id_raster.tif"
            )
            if not os.path.exists(dev_source_path):
                print(f"Missing source_id raster for development {dev_id} - skipping")
                continue

            with rasterio.open(dev_source_path) as src:
                dev_source = src.read(1)

            # Development catchment-pair travel times for this development and scenario.
            dev_tt = (
                tt_developments[
                    (tt_developments["development"] == dev_id) &
                    (tt_developments["scenario"] == scen)
                ][["origin", "destination", "travel_time"]]
                .rename(
                    columns={
                        "origin": "origin_catchment",
                        "destination": "destination_catchment",
                        "travel_time": "tt_dev",
                    }
                )
            )
            if dev_tt.empty:
                print(f"No OD TT rows for development {dev_id} in scenario {scen} - skipping")
                continue

            # Build raster -> commune -> development catchment allocation on the same stable grid.
            valid_dev = (
                (dev_source > 0) &
                (commune_raster > 0) &
                np.isfinite(pop) &
                np.isfinite(empl) &
                (pop >= 0) &
                (empl >= 0)
            )
            dev_df = pd.DataFrame(
                {
                    "commune_id": commune_raster[valid_dev].astype(int),
                    "catchment": dev_source[valid_dev].astype(int),
                    "pop": pop[valid_dev],
                    "empl": empl[valid_dev],
                }
            )
            dev_origin, dev_dest = shares_by_commune_and_catchment(dev_df)

            # Redistribute the SAME commune OD demand over development catchment pairs.
            # Only the within-commune spatial split changes because raster cells may
            # belong to different catchments in the development.
            dev_pairs = (
                od_long
                .merge(dev_origin, on="origin_commune", how="left")
                .merge(dev_dest, on="destination_commune", how="left")
                .merge(dev_tt, on=["origin_catchment", "destination_catchment"], how="left")
            )
            dev_pairs = dev_pairs.dropna(subset=["origin_share", "dest_share", "tt_dev"]).copy()
            if dev_pairs.empty:
                print(f"No raster-weighted OD pairs for development {dev_id} in scenario {scen} - skipping")
                continue

            dev_pairs["pair_demand"] = (
                dev_pairs["od_demand"] * dev_pairs["origin_share"] * dev_pairs["dest_share"]
            )
            dev_pairs["weighted_tt"] = dev_pairs["pair_demand"] * dev_pairs["tt_dev"]
            dev_total = float(dev_pairs["weighted_tt"].sum())

            rows.append(
                {
                    "development": int(dev_id),
                    "scenario": scen,
                    "sq_total_weighted_tt": sq_total,
                    "dev_total_weighted_tt": dev_total,
                    "tt_savings_daily": sq_total - dev_total,
                    "sq_pair_rows": int(len(sq_pairs)),
                    "dev_pair_rows": int(len(dev_pairs)),
                }
            )

    merged = pd.DataFrame(rows)
    if merged.empty:
        raise ValueError("No raster-basis OD TT savings could be computed.")

    # 4) Monetize the routed TT differences and write them in the old raster output format.
    # Monetize the daily TT savings with the same factor used elsewhere in road.
    # TO DO: check
    mon_factor = VTTS * 2.5 * 250 * duration
    merged["monetized_savings"] = merged["tt_savings_daily"] * mon_factor

    detailed_out = "data/infraScanRoad/traffic_flow/od/od_tt_savings_raster_detailed.csv"
    merged.to_csv(detailed_out, index=False)

    split_base = "data/infraScanRoad/traffic_flow/od/by_scenario"
    for scen, scen_df in merged.groupby("scenario"):
        scen_dir = os.path.join(split_base, str(scen))
        os.makedirs(scen_dir, exist_ok=True)
        scen_df.to_csv(os.path.join(scen_dir, "od_tt_savings_raster_detailed.csv"), index=False)

    tt_wide = (
        merged.pivot(index="development", columns="scenario", values="monetized_savings")
        .reset_index()
    )
    tt_wide.columns.name = None
    tt_wide = tt_wide.rename(
        columns={col: f"tt_{col}" for col in tt_wide.columns if col != "development"}
    )

    method = "od_raster"
    wide_out = f"data/infraScanRoad/costs/traveltime_savings_{method}.csv"
    tt_wide.to_csv(wide_out, index=False)

    print(f"Saved detailed OD-raster TT savings to: {detailed_out}")
    print(f"Saved aggregated TT savings to: {wide_out}")

    return merged, tt_wide


def legacy_od_popraster_tts(VTTS, duration):
    """
    Monetize OD-level travel time savings using a population-only raster-based
    tessellation transfer.

    The procedure is:
    1. Read routed OD travel times from phase 6:
       - status quo catchment-pair travel times
       - development catchment-pair travel times
    2. Read the original commune-to-commune OD demand matrix
    3. Compute commune -> catchment population shares for:
       - the status-quo tessellation
       - each development tessellation
    4. Redistribute commune OD demand to catchment-pair OD demand using:
         flow(l, m | i, j) = T_ij * share_origin(i -> l) * share_dest(j -> m)
    5. Weight redistributed flows with routed catchment-pair travel times
    6. Monetize the resulting daily travel-time savings

    Inputs:
    - data/infraScanRoad/traffic_flow/od/status_quo_od_tt.csv
    - data/infraScanRoad/traffic_flow/od/developments_od_tt.csv
    - data/infraScanRoad/Network/travel_time/source_id_raster.tif
    - data/infraScanRoad/Network/travel_time/developments/dev*_source_id_raster.tif
    - population rasters:
        STATIC:
          data/independent_variable/processed/scenario/scen_pop.tif
        GENERATED:
          data/independent_variable/processed/scenario/scenario_X_pop.tif
          or scenario_X_pop_<valuation_year>.tif

    Outputs:
    - data/infraScanRoad/traffic_flow/od/od_tt_savings_detailed.csv
    - data/infraScanRoad/costs/traveltime_savings_od.csv
    """
    # 1) Legacy popraster variant: population-only tessellation transfer on a fixed commune OD basis.
    status_quo_path = "data/infraScanRoad/traffic_flow/od/status_quo_od_tt.csv"
    developments_path = "data/infraScanRoad/traffic_flow/od/developments_od_tt.csv"
    sq_source_path = "data/infraScanRoad/Network/travel_time/source_id_raster.tif"

    tt_status_quo = pd.read_csv(status_quo_path)
    tt_developments = pd.read_csv(developments_path)

    required_sq_cols = {"scenario", "origin", "destination", "travel_time"}
    required_dev_cols = {"development", "scenario", "origin", "destination", "travel_time"}

    missing_sq = required_sq_cols.difference(tt_status_quo.columns)
    missing_dev = required_dev_cols.difference(tt_developments.columns)

    if missing_sq:
        raise ValueError(f"Missing required columns in {status_quo_path}: {sorted(missing_sq)}")
    if missing_dev:
        raise ValueError(f"Missing required columns in {developments_path}: {sorted(missing_dev)}")

    # Keep datatypes stable for merging.
    tt_status_quo["scenario"] = tt_status_quo["scenario"].astype(str)
    tt_developments["scenario"] = tt_developments["scenario"].astype(str)

    for col in ["origin", "destination", "travel_time"]:
        tt_status_quo[col] = pd.to_numeric(tt_status_quo[col], errors="coerce")
    for col in ["development", "origin", "destination", "travel_time"]:
        tt_developments[col] = pd.to_numeric(tt_developments[col], errors="coerce")

    tt_status_quo = tt_status_quo.dropna(subset=["origin", "destination", "travel_time"])
    tt_developments = tt_developments.dropna(
        subset=["development", "origin", "destination", "travel_time"]
    )

    tt_status_quo["origin"] = tt_status_quo["origin"].astype(int)
    tt_status_quo["destination"] = tt_status_quo["destination"].astype(int)

    tt_developments["development"] = tt_developments["development"].astype(int)
    tt_developments["origin"] = tt_developments["origin"].astype(int)
    tt_developments["destination"] = tt_developments["destination"].astype(int)

    # Stable raster basis:
    # - sq_source gives the status-quo road catchment id per raster cell
    # - commune_raster gives the commune id per raster cell on the same grid
    with rasterio.open(sq_source_path) as src:
        sq_source = src.read(1)

    commune_raster, _ = GetCommuneShapes(raster_path=sq_source_path)

    # Original commune-to-commune OD matrix = stable demand basis.
    od = GetHighwayPHDemandPerCommune()
    odmat = GetODMatrix(od).astype(float)

    od_long = odmat.stack().rename("od_demand").reset_index()
    od_long.columns = ["origin_commune", "destination_commune", "od_demand"]
    od_long["origin_commune"] = pd.to_numeric(od_long["origin_commune"], errors="coerce")
    od_long["destination_commune"] = pd.to_numeric(od_long["destination_commune"], errors="coerce")
    od_long["od_demand"] = pd.to_numeric(od_long["od_demand"], errors="coerce")
    od_long = od_long.dropna(subset=["origin_commune", "destination_commune", "od_demand"])

    od_long["origin_commune"] = od_long["origin_commune"].astype(int)
    od_long["destination_commune"] = od_long["destination_commune"].astype(int)

    # Exclude intrazonal commune flows and zero-demand rows.
    od_long = od_long[
        (od_long["origin_commune"] != od_long["destination_commune"]) &
        (od_long["od_demand"] > 0)
    ].copy()

    # 2) Restrict to scenarios that exist in both routed OD outputs.
    # Only keep scenarios that are actually available in both routed outputs.
    available_sq_scenarios = set(tt_status_quo["scenario"].unique())
    available_dev_scenarios = set(tt_developments["scenario"].unique())
    scenarios = sorted(available_sq_scenarios.intersection(available_dev_scenarios))
    if not scenarios:
        raise ValueError("No common scenarios found between status_quo_od_tt and developments_od_tt.")

    def load_population_raster_for_scenario(scen):
        """
        Load the population raster for a given scenario.
        STATIC:
            uses the multi-band scen_pop.tif with fixed band mapping.
        GENERATED:
            uses scenario-specific rasters written by the generated scenario workflow.
        """
        if settings.scenario_type == "STATIC":
            pop_raster_path = "data/independent_variable/processed/scenario/scen_pop.tif"
            band_map = {"medium": 1, "low": 2, "high": 3}

            if scen not in band_map:
                raise ValueError(
                    f"Unsupported STATIC scenario '{scen}' for population raster loading."
                )

            with rasterio.open(pop_raster_path) as src:
                pop_raster = src.read(band_map[scen]).astype(float)
            return pop_raster

        if settings.scenario_type == "GENERATED":
            candidates = [
                os.path.join("data", "independent_variable", "processed", "scenario",f"{scen}_pop.tif",),
                os.path.join("data", "independent_variable", "processed", "scenario",
                    f"{scen}_pop_{settings.start_valuation_year}.tif",),
            ]
            pop_raster_path = next((p for p in candidates if os.path.exists(p)), None)
            if pop_raster_path is None:
                raise FileNotFoundError(
                    f"Missing generated population raster for scenario '{scen}'. "
                    f"Tried: {candidates}"
                )

            with rasterio.open(pop_raster_path) as src:
                pop_raster = src.read(1).astype(float)
            return pop_raster

        raise ValueError(f"Unsupported scenario_type: {settings.scenario_type}")

    def shares_by_commune_and_catchment(catchment_raster, pop_raster):
        """
        Compute commune -> catchment population shares.

        For each raster cell, we know:
        - which commune it belongs to
        - which catchment (Voronoi zone) it belongs to
        - how much population mass it carries

        We aggregate that to:
            pop_mass(commune, catchment)

        and normalize within each commune:
            share(commune -> catchment) =
                pop_mass(commune, catchment) / total_pop_mass(commune)
        """
        rows = []

        unique_communes = np.unique(commune_raster)
        for commune_id in unique_communes:
            if commune_id <= 0:
                continue

            commune_mask = commune_raster == commune_id
            if not np.any(commune_mask):
                continue

            catchments_in_commune = np.unique(catchment_raster[commune_mask])

            for catchment_id in catchments_in_commune:
                if catchment_id <= 0:
                    continue

                overlap_mask = commune_mask & (catchment_raster == catchment_id)
                pop_mass = float(np.nansum(pop_raster[overlap_mask]))

                if pop_mass <= 0:
                    continue

                rows.append(
                    {
                        "commune_id": int(commune_id),
                        "catchment_id": int(catchment_id),
                        "pop_mass": pop_mass,
                    }
                )

        shares = pd.DataFrame(rows)
        if shares.empty:
            return shares

        shares["commune_total_pop"] = shares.groupby("commune_id")["pop_mass"].transform("sum")
        shares = shares[shares["commune_total_pop"] > 0].copy()
        shares["share"] = shares["pop_mass"] / shares["commune_total_pop"]

        return shares[["commune_id", "catchment_id", "share"]]

    def restrict_and_renormalize_shares(shares, valid_catchment_ids):
        """
        Restrict commune->catchment shares to catchments that are actually present
        in the routed OD travel-time table and renormalize within each commune.

        This keeps the raster-based tessellation transfer aligned with the set of
        zones for which phase 6 produced travel times. Without this step, the
        redistribution can generate OD pairs for catchments that exist in the
        raster tessellation but were not routable in the network assignment.
        """
        if shares.empty:
            return shares

        valid_ids = {int(x) for x in valid_catchment_ids}
        filtered = shares[shares["catchment_id"].isin(valid_ids)].copy()
        if filtered.empty:
            return filtered

        filtered["commune_total_share"] = filtered.groupby("commune_id")["share"].transform("sum")
        filtered = filtered[filtered["commune_total_share"] > 0].copy()
        filtered["share"] = filtered["share"] / filtered["commune_total_share"]

        return filtered[["commune_id", "catchment_id", "share"]]

    def redistribute_commune_od(od_long_df, origin_shares, dest_shares):
        """
        Redistribute commune-to-commune OD demand to catchment-to-catchment OD demand
        using only population shares.

        For each original OD pair T_ij:
            flow(l, m | i, j) = T_ij * share_origin(i -> l) * share_dest(j -> m)

        Then sum over all commune OD pairs to obtain the new OD matrix on the
        catchment tessellation.
        """
        origin_map = origin_shares.rename(
            columns={
                "commune_id": "origin_commune",
                "catchment_id": "origin",
                "share": "origin_share",
            }
        )
        dest_map = dest_shares.rename(
            columns={
                "commune_id": "destination_commune",
                "catchment_id": "destination",
                "share": "destination_share",
            }
        )

        redistributed = od_long_df.merge(origin_map, on="origin_commune", how="inner")
        redistributed = redistributed.merge(dest_map, on="destination_commune", how="inner")

        redistributed["flow"] = (
            redistributed["od_demand"] *
            redistributed["origin_share"] *
            redistributed["destination_share"]
        )

        redistributed = (
            redistributed.groupby(["origin", "destination"], as_index=False)["flow"]
            .sum()
        )

        redistributed["origin"] = redistributed["origin"].astype(int)
        redistributed["destination"] = redistributed["destination"].astype(int)

        # Remove intra-catchment OD flows if present.
        redistributed = redistributed[redistributed["origin"] != redistributed["destination"]].copy()

        return redistributed

    detailed_rows = []
    summary_rows = []

    # Precompute status-quo shares per scenario once.
    sq_shares_by_scenario = {}
    for scen in scenarios:
        pop_raster = load_population_raster_for_scenario(scen)
        sq_shares_by_scenario[scen] = shares_by_commune_and_catchment(
            catchment_raster=sq_source,
            pop_raster=pop_raster,
        )

    developments = sorted(tt_developments["development"].unique().tolist())

    for scen in scenarios:
        pop_raster = load_population_raster_for_scenario(scen)

        sq_tt = tt_status_quo[tt_status_quo["scenario"] == scen].copy()
        if sq_tt.empty:
            continue

        valid_sq_ids = set(sq_tt["origin"]).union(set(sq_tt["destination"]))
        sq_shares = restrict_and_renormalize_shares(
            sq_shares_by_scenario[scen],
            valid_sq_ids,
        )
        if sq_shares.empty:
            print(f"No status-quo population shares found for scenario {scen} - skipping.")
            continue

        sq_flows = redistribute_commune_od(
            od_long_df=od_long,
            origin_shares=sq_shares,
            dest_shares=sq_shares,
        )

        sq_weighted = sq_flows.merge(
            sq_tt[["origin", "destination", "travel_time"]],
            on=["origin", "destination"],
            how="left",
        )
        missing_sq_tt = int(sq_weighted["travel_time"].isna().sum())
        if missing_sq_tt > 0:
            raise ValueError(
                f"Missing {missing_sq_tt} status-quo travel-time matches for scenario '{scen}' "
                f"after commune-to-catchment redistribution."
            )
        sq_weighted["weighted_tt"] = sq_weighted["flow"] * sq_weighted["travel_time"]
        sq_total_tt = float(sq_weighted["weighted_tt"].sum(skipna=True))

        for dev in developments:
            dev_tt = tt_developments[
                (tt_developments["development"] == dev) &
                (tt_developments["scenario"] == scen)
            ].copy()

            if dev_tt.empty:
                continue

            dev_raster_path = (
                f"data/infraScanRoad/Network/travel_time/developments/dev{dev}_source_id_raster.tif"
            )
            if not os.path.exists(dev_raster_path):
                print(f"Missing development raster for dev {dev} - skipping.")
                continue

            with rasterio.open(dev_raster_path) as src:
                dev_catchment_raster = src.read(1)

            dev_shares = shares_by_commune_and_catchment(
                catchment_raster=dev_catchment_raster,
                pop_raster=pop_raster,
            )
            valid_dev_ids = set(dev_tt["origin"]).union(set(dev_tt["destination"]))
            dev_shares = restrict_and_renormalize_shares(
                dev_shares,
                valid_dev_ids,
            )

            if dev_shares.empty:
                print(f"No development population shares found for dev {dev}, scenario {scen} - skipping.")
                continue

            dev_flows = redistribute_commune_od(
                od_long_df=od_long,
                origin_shares=dev_shares,
                dest_shares=dev_shares,
            )

            dev_weighted = dev_flows.merge(
                dev_tt[["origin", "destination", "travel_time"]],
                on=["origin", "destination"],
                how="left",
            )
            missing_dev_tt = int(dev_weighted["travel_time"].isna().sum())
            if missing_dev_tt > 0:
                raise ValueError(
                    f"Missing {missing_dev_tt} development travel-time matches for "
                    f"dev {dev}, scenario '{scen}' after commune-to-catchment redistribution."
                )
            dev_weighted["weighted_tt"] = dev_weighted["flow"] * dev_weighted["travel_time"]
            dev_total_tt = float(dev_weighted["weighted_tt"].sum(skipna=True))

            tt_savings_daily = sq_total_tt - dev_total_tt

            # Monetization factor consistent with the existing OD functions.
            mon_factor = VTTS * 2.5 * 250 * duration
            monetized_savings = tt_savings_daily * mon_factor

            detailed_rows.append(
                {
                    "development": int(dev),
                    "scenario": str(scen),
                    "status_quo_total_tt": sq_total_tt,
                    "development_total_tt": dev_total_tt,
                    "tt_savings_daily": tt_savings_daily,
                    "monetized_savings": monetized_savings,
                }
            )

            summary_rows.append(
                {
                    "development": int(dev),
                    "scenario": str(scen),
                    "monetized_savings": monetized_savings,
                }
            )

    if not detailed_rows:
        raise ValueError("No OD pop-raster travel-time savings could be computed.")

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(
        "data/infraScanRoad/traffic_flow/od/od_tt_savings_popraster_detailed.csv",
        index=False,
    )

    split_base = "data/infraScanRoad/traffic_flow/od/by_scenario"
    for scen, scen_df in detailed_df.groupby("scenario"):
        scen_dir = os.path.join(split_base, str(scen))
        os.makedirs(scen_dir, exist_ok=True)
        scen_df.to_csv(
            os.path.join(scen_dir, "od_tt_savings_popraster_detailed.csv"),
            index=False,
        )

    tt_wide = (
        pd.DataFrame(summary_rows)
        .pivot(index="development", columns="scenario", values="monetized_savings")
        .reset_index()
    )

    tt_wide.columns.name = None
    tt_wide = tt_wide.rename(
        columns={
            col: f"tt_{col}"
            for col in tt_wide.columns
            if col != "development"
        }
    )

    method = "od_popraster"
    tt_wide.to_csv(
        fr"data/infraScanRoad/costs/traveltime_savings_{method}.csv",
        index=False,
    )

    print(
        "Saved detailed OD TT savings to: "
        "data/infraScanRoad/traffic_flow/od/od_tt_savings_popraster_detailed.csv"
    )
    print(
        f"Saved aggregated TT savings to: "
        f"data/infraScanRoad/costs/traveltime_savings_{method}.csv"
    )

    return detailed_df, tt_wide

