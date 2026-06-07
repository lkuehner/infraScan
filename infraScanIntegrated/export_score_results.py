from __future__ import annotations

import argparse
import sqlite3
import warnings
from pathlib import Path

import pandas as pd
try:
    import geopandas as gpd
except Exception:  # pragma: no cover
    gpd = None

from . import scoring_registry as registry
from ..infraScanRail import paths as rail_paths


DATA_ROOT = Path(rail_paths.MAIN)

ROAD_COSTS_DIR_CANDIDATES = [
    DATA_ROOT / "euler" / "infraScanRoad_trust_2iter_alldev_10sce" / "costs",
]
ROAD_EXTERNALITIES_DETAIL_CANDIDATES = [
    DATA_ROOT / "euler" / "infraScanRoad_trust_2iter_alldev_10sce" / "traffic_flow" / "road_externalities_inputs" / "road_externalities_link_detail.csv",
    DATA_ROOT / "infraScan" / "infraScanIntegrated" / "outputs" / "road_externalities_inputs" / "road_externalities_link_detail.csv",
]
RAIL_COSTS_DIR_CANDIDATES = [
    DATA_ROOT / "data" / "infraScanRail" / "costs",
]
RAIL_TRAIN_KM_CANDIDATES = [
    DATA_ROOT / "data" / "infraScanRail" / "Network" / "processed" / "train_km.csv",
]
RAIL_DISCOUNTED_TTS_CANDIDATES = [
    DATA_ROOT / "data" / "infraScanRail" / "costs" / "costs_and_benefits_discounted.csv",
]
DEFAULT_OUTPUT_DIR = Path("infraScan/infraScanIntegrated/outputs/score_results")


def first_existing_path(candidates: list[Path], path_kind: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No existing {path_kind} found. Checked:\n" + "\n".join(str(path) for path in candidates)
    )


def first_existing_optional_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def is_readable_sqlite(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        con = sqlite3.connect(str(path))
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master LIMIT 1")
        cur.fetchall()
        con.close()
        return True
    except Exception:
        return False


def is_readable_geopackage(path: Path) -> bool:
    if not path.exists() or gpd is None:
        return False
    try:
        gpd.read_file(path, rows=1)
        return True
    except Exception:
        return False


def first_readable_gpkg(candidates: list[Path], path_kind: str) -> Path:
    for candidate in candidates:
        if is_readable_sqlite(candidate) or is_readable_geopackage(candidate):
            return candidate
    raise FileNotFoundError(
        f"No readable {path_kind} found. Checked:\n" + "\n".join(str(path) for path in candidates)
    )


def first_readable_optional_gpkg(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if is_readable_sqlite(candidate) or is_readable_geopackage(candidate):
            return candidate
    return None


def read_gpkg_table(path: Path, table: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            if table is None:
                contents = pd.read_sql("SELECT table_name, data_type FROM gpkg_contents", con)
                table = contents.loc[contents["data_type"].isin(["features", "attributes"]), "table_name"].iloc[0]
            query = f"SELECT * FROM '{table}'" if columns is None else f"SELECT {', '.join(columns)} FROM '{table}'"
            return pd.read_sql(query, con)
        finally:
            con.close()
    except Exception:
        if gpd is None:
            raise
        gdf = gpd.read_file(path)
        df = pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
        if columns is not None:
            keep_cols = [col for col in columns if col in df.columns]
            return df[keep_cols].copy()
        return df


def get_table_columns(path: Path, table: str | None = None) -> list[str]:
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            if table is None:
                contents = pd.read_sql("SELECT table_name, data_type FROM gpkg_contents", con)
                table = contents.loc[contents["data_type"].isin(["features", "attributes"]), "table_name"].iloc[0]
            return pd.read_sql(f"PRAGMA table_info('{table}')", con)["name"].tolist()
        finally:
            con.close()
    except Exception:
        if gpd is None:
            raise
        gdf = gpd.read_file(path, rows=1)
        return list(gdf.columns)


def normalize_development(values: pd.Series) -> pd.Series:
    normalized = values.astype(str)
    normalized = normalized.str.replace("Development_", "", regex=False)
    normalized = normalized.str.replace(r"\.0$", "", regex=True)
    return normalized


def scenario_number(scenario: str) -> int:
    parts = str(scenario).split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return -1


def sort_scenarios(scenarios: list[str]) -> list[str]:
    return sorted(scenarios, key=scenario_number)


def prepare_road_construction(construction_gpkg: Path) -> pd.DataFrame:
    columns = ["ID_new", "cost_path", "cost_bridge", "cost_tunnel", "building_costs"]
    available_columns = get_table_columns(construction_gpkg)
    if "cost_ramp" in available_columns:
        columns.append("cost_ramp")
    construction = read_gpkg_table(construction_gpkg, columns=columns)

    for col in ["cost_path", "cost_bridge", "cost_tunnel", "cost_ramp", "building_costs"]:
        if col in construction.columns:
            construction[col] = pd.to_numeric(construction[col], errors="coerce").abs()

    construction["cost_open_highway"] = construction["cost_path"]
    if "cost_ramp" not in construction.columns:
        construction["cost_ramp"] = (
            construction["building_costs"]
            - construction["cost_path"]
            - construction["cost_bridge"]
            - construction["cost_tunnel"]
        ).clip(lower=0.0)
    return construction


def build_road_results(
    road_costs_dir: Path,
    road_externalities_detail_path: Path | None,
) -> pd.DataFrame:
    construction_gpkg = first_readable_gpkg(
        [road_costs_dir / "total_costs_od.gpkg"],
        "road construction gpkg",
    )
    total_costs_csv = first_existing_path(
        [road_costs_dir / "total_costs_od.csv"],
        "road total costs csv",
    )
    tts_csv = first_existing_path(
        [
            road_costs_dir / "traveltime_savings_od_yearly.csv",
            road_costs_dir / "traveltime_savings_od.csv",
        ],
        "road travel time savings csv",
    )
    maintenance_gpkg = first_readable_optional_gpkg(
        [road_costs_dir / "maintenance.gpkg"],
    )

    construction = prepare_road_construction(construction_gpkg)
    road_tt_wide = pd.read_csv(tts_csv)
    total_costs_df = pd.read_csv(total_costs_csv)
    standalone_externality_cols = [
        "climate_cost",
        "land_realloc",
        "nature",
        "noise_s1",
    ]
    missing_standalone_cols = [
        col for col in standalone_externality_cols if col not in total_costs_df.columns
    ]
    if missing_standalone_cols:
        gpkg_cols = ["ID_new"] + missing_standalone_cols
        total_costs_gpkg = read_gpkg_table(construction_gpkg, columns=gpkg_cols)
        total_costs_df = total_costs_df.merge(
            total_costs_gpkg,
            on="ID_new",
            how="left",
        )

    if "ID_new" in road_tt_wide.columns and "development" not in road_tt_wide.columns:
        road_tt_wide = road_tt_wide.rename(columns={"ID_new": "development"})

    road_tt_scenarios = {
        col.removeprefix("tt_")
        for col in road_tt_wide.columns
        if col.startswith("tt_")
    }
    road_total_scenarios = {
        col.removeprefix("total_")
        for col in total_costs_df.columns
        if col.startswith("total_scenario_")
    }
    scenarios = sort_scenarios(list(road_tt_scenarios & road_total_scenarios))

    result_frames = [
        registry.build_road_construction_result_df(construction, scenarios),
        registry.build_road_tts_result_df(road_tt_wide, scenarios),
    ]

    if maintenance_gpkg is not None:
        maintenance_columns = ["ID_new"]
        available_columns = get_table_columns(maintenance_gpkg)
        if "maintenance_annual" in available_columns:
            maintenance_columns.append("maintenance_annual")
            maintenance_df = read_gpkg_table(maintenance_gpkg, columns=maintenance_columns)
            result_frames.append(
                registry.build_road_maint_result_df(maintenance_df, scenarios)
            )
        elif "maintenance" in available_columns:
            maintenance_df = read_gpkg_table(maintenance_gpkg, columns=["ID_new", "maintenance"])
            maintenance_df["maintenance_annual"] = pd.to_numeric(
                maintenance_df["maintenance"],
                errors="coerce",
            )
            result_frames.append(
                registry.build_road_maint_result_df(
                    maintenance_df[["ID_new", "maintenance_annual"]],
                    scenarios,
                )
            )
        else:
            warnings.warn("Road maintenance.gpkg has neither 'maintenance_annual' nor 'maintenance'. Skipping road maintenance.")
    else:
        warnings.warn("Road maintenance.gpkg is not readable in the current Euler run. Skipping road maintenance.")

    if road_externalities_detail_path is not None and road_externalities_detail_path.exists():
        detail_df = pd.read_csv(road_externalities_detail_path)
        result_frames.append(
            registry.build_road_externalities_result_df(
                detail_df=detail_df,
                total_costs_df=total_costs_df,
            )
        )
    else:
        warnings.warn("Road externalities detail file not found. Skipping integrated road externalities.")

    road_result = pd.concat(result_frames, ignore_index=True)
    road_result["mode"] = "Road"
    road_result["development"] = normalize_development(road_result["development"])
    road_result["scenario"] = road_result["scenario"].astype(str)
    return road_result[["mode"] + registry.RESULT_COLUMNS]


def build_rail_results(
    rail_costs_dir: Path,
    rail_train_km_path: Path | None,
) -> pd.DataFrame:
    construction_csv = rail_costs_dir / "construction_cost.csv"
    tts_csv = rail_costs_dir / "traveltime_savings.csv"
    discounted_tts_path = first_existing_optional_path(RAIL_DISCOUNTED_TTS_CANDIDATES)

    construction_raw = pd.read_csv(construction_csv)
    tts_df = pd.read_csv(tts_csv)
    discounted_tts_df = None
    if discounted_tts_path is not None and discounted_tts_path.exists():
        discounted_raw = pd.read_csv(discounted_tts_path, usecols=["development", "scenario", "benefit"])
        discounted_tts_df = (
            discounted_raw.groupby(["development", "scenario"], as_index=False)
            .agg(standalone_value=("benefit", "sum"))
        )
        discounted_tts_df["standalone_value"] = (
            discounted_tts_df["standalone_value"] / registry.STANDALONE_ANNUAL_YEARS
        )

    construction_registry_input = construction_raw.rename(
        columns={
            "Development": "ID_new",
            "TotalConstructionCost": "building_costs",
        }
    ).copy()
    static_registry_input = construction_raw.rename(columns={"Development": "ID_new"}).copy()

    scenarios = sort_scenarios(sorted(tts_df["scenario"].astype(str).unique().tolist()))

    result_frames = [
        registry.build_rail_construction_result_df(construction_registry_input, scenarios),
        registry.build_rail_maint_result_df(static_registry_input, scenarios),
        registry.build_rail_operation_result_df(static_registry_input, scenarios),
        registry.build_rail_tts_result_df(
            tts_df.copy(),
            old_discounted_tts_df=discounted_tts_df,
        ),
    ]

    if rail_train_km_path is not None and rail_train_km_path.exists():
        train_km_df = pd.read_csv(rail_train_km_path)
        result_frames.append(
            registry.build_rail_externalities_result_df(
                train_km_df=train_km_df,
                scenarios=scenarios,
            )
        )
    else:
        warnings.warn("Rail train_km.csv not found. Skipping rail externalities.")

    rail_result = pd.concat(result_frames, ignore_index=True)
    rail_result["mode"] = "Rail"
    rail_result["development"] = normalize_development(rail_result["development"])
    rail_result["scenario"] = rail_result["scenario"].astype(str)
    return rail_result[["mode"] + registry.RESULT_COLUMNS]


def build_tidy_values(score_long: pd.DataFrame) -> pd.DataFrame:
    tidy = score_long.melt(
        id_vars=["mode", "development", "scenario", "score_id"],
        value_vars=["standalone_value", "integrated_value"],
        var_name="value_mode",
        value_name="value",
    )
    tidy["value_mode"] = tidy["value_mode"].replace(
        {
            "standalone_value": "standalone",
            "integrated_value": "integrated",
        }
    )
    return tidy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export integrated road and rail score tables.")
    parser.add_argument(
        "--road-costs-dir",
        type=Path,
        default=first_existing_path(ROAD_COSTS_DIR_CANDIDATES, "road costs directory"),
    )
    parser.add_argument(
        "--rail-costs-dir",
        type=Path,
        default=first_existing_path(RAIL_COSTS_DIR_CANDIDATES, "rail costs directory"),
    )
    parser.add_argument(
        "--road-externalities-detail",
        type=Path,
        default=first_existing_optional_path(ROAD_EXTERNALITIES_DETAIL_CANDIDATES),
    )
    parser.add_argument(
        "--rail-train-km",
        type=Path,
        default=first_existing_optional_path(RAIL_TRAIN_KM_CANDIDATES),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    road_result = build_road_results(
        road_costs_dir=args.road_costs_dir,
        road_externalities_detail_path=args.road_externalities_detail,
    )
    rail_result = build_rail_results(
        rail_costs_dir=args.rail_costs_dir,
        rail_train_km_path=args.rail_train_km,
    )

    score_long = pd.concat([road_result, rail_result], ignore_index=True)
    score_long["scenario_sort"] = score_long["scenario"].map(scenario_number)
    score_long = score_long.sort_values(
        ["mode", "development", "scenario_sort", "score_id"]
    ).drop(columns=["scenario_sort"]).reset_index(drop=True)

    tidy = build_tidy_values(score_long)
    tidy["scenario_sort"] = tidy["scenario"].map(scenario_number)
    tidy = tidy.sort_values(
        ["mode", "development", "scenario_sort", "score_id", "value_mode"]
    ).drop(columns=["scenario_sort"]).reset_index(drop=True)

    score_long_path = args.output_dir / "score_results_long.csv"
    tidy_path = args.output_dir / "score_results_tidy.csv"

    score_long.to_csv(score_long_path, index=False)
    tidy.to_csv(tidy_path, index=False)

    print(f"Wrote {score_long_path}")
    print(f"Wrote {tidy_path}")


if __name__ == "__main__":
    main()
