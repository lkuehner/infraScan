import os
from typing import Optional

import pandas as pd


BASE_DIR = "data/infraScanRoad"


def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def _coerce_scalar(value):
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
        return float(stripped)
    return float(value)


def _aggregate_od_totals(df: pd.DataFrame, demand_col: str, tt_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["weighted_tt"] = tmp[demand_col] * tmp[tt_col]
    return (
        tmp.groupby(["development", "scenario"], as_index=False)
        .agg(
            total_weighted_tt=("weighted_tt", "sum"),
            total_demand=(demand_col, "sum"),
            od_pairs=(demand_col, "size"),
        )
    )


def compare_tts_methods_for_case(
    scenario: str,
    development: int,
    base_dir: str = BASE_DIR,
) -> pd.DataFrame:
    """
    Compare aggregate and OD total travel-time calculations for one
    (development, scenario) case using already written CSV outputs.

    Returns a one-row DataFrame with:
    - aggregate_status_quo_tt
    - aggregate_development_tt
    - aggregate_delta_tt
    - od_status_quo_weighted_tt
    - od_development_weighted_tt
    - od_delta_tt
    - total demand / OD pair counts on OD side
    """
    traffic_dir = os.path.join(base_dir, "traffic_flow")
    costs_dir = os.path.join(base_dir, "costs")

    agg_status_quo = _read_csv(os.path.join(traffic_dir, "travel_time_status_quo.csv"))
    agg_developments = _read_csv(os.path.join(traffic_dir, "travel_time.csv"))

    od_status_quo = _read_csv(os.path.join(traffic_dir, "od", "status_quo_od_tt.csv"))
    od_developments = _read_csv(os.path.join(traffic_dir, "od", "developments_od_tt.csv"))

    scenario = str(scenario)

    if scenario not in agg_status_quo.columns:
        raise KeyError(
            f"Scenario '{scenario}' missing in aggregate status quo file. "
            f"Available: {list(agg_status_quo.columns)}"
        )

    if scenario not in agg_developments.columns:
        raise KeyError(
            f"Scenario '{scenario}' missing in aggregate developments file. "
            f"Available: {list(agg_developments.columns)}"
        )

    agg_status_quo_tt = _coerce_scalar(agg_status_quo[scenario].iloc[0])

    agg_developments["development"] = agg_developments["development"].astype(int)
    agg_dev_row = agg_developments.loc[agg_developments["development"] == int(development)]
    if agg_dev_row.empty:
        raise KeyError(f"Development {development} missing in aggregate travel_time.csv")
    agg_development_tt = _coerce_scalar(agg_dev_row[scenario].iloc[0])

    od_status_quo["development"] = od_status_quo["development"].astype(int)
    od_developments["development"] = od_developments["development"].astype(int)
    od_status_quo["scenario"] = od_status_quo["scenario"].astype(str)
    od_developments["scenario"] = od_developments["scenario"].astype(str)

    od_status_quo_totals = _aggregate_od_totals(
        od_status_quo,
        demand_col="demand",
        tt_col="travel_time",
    )
    od_development_totals = _aggregate_od_totals(
        od_developments,
        demand_col="demand",
        tt_col="travel_time",
    )

    od_sq_row = od_status_quo_totals.loc[
        (od_status_quo_totals["development"] == int(development))
        & (od_status_quo_totals["scenario"] == scenario)
    ]
    if od_sq_row.empty:
        raise KeyError(
            f"No OD status-quo totals found for development={development}, scenario='{scenario}'"
        )

    od_dev_row = od_development_totals.loc[
        (od_development_totals["development"] == int(development))
        & (od_development_totals["scenario"] == scenario)
    ]
    if od_dev_row.empty:
        raise KeyError(
            f"No OD development totals found for development={development}, scenario='{scenario}'"
        )

    result = pd.DataFrame(
        [
            {
                "development": int(development),
                "scenario": scenario,
                "aggregate_status_quo_tt": agg_status_quo_tt,
                "aggregate_development_tt": agg_development_tt,
                "aggregate_delta_tt": agg_status_quo_tt - agg_development_tt,
                "od_status_quo_weighted_tt": float(od_sq_row["total_weighted_tt"].iloc[0]),
                "od_development_weighted_tt": float(od_dev_row["total_weighted_tt"].iloc[0]),
                "od_delta_tt": (
                    float(od_sq_row["total_weighted_tt"].iloc[0])
                    - float(od_dev_row["total_weighted_tt"].iloc[0])
                ),
                "od_status_quo_total_demand": float(od_sq_row["total_demand"].iloc[0]),
                "od_development_total_demand": float(od_dev_row["total_demand"].iloc[0]),
                "od_status_quo_pairs": int(od_sq_row["od_pairs"].iloc[0]),
                "od_development_pairs": int(od_dev_row["od_pairs"].iloc[0]),
            }
        ]
    )

    agg_path = os.path.join(costs_dir, "traveltime_savings_aggregate.csv")
    od_path = os.path.join(costs_dir, "traveltime_savings_od.csv")
    if os.path.exists(agg_path):
        agg_savings = _read_csv(agg_path)
        agg_savings["development"] = agg_savings["development"].astype(int)
        col = f"tt_{scenario}"
        if col in agg_savings.columns:
            match = agg_savings.loc[agg_savings["development"] == int(development), col]
            if not match.empty:
                result["aggregate_monetized_tts"] = float(match.iloc[0])

    if os.path.exists(od_path):
        od_savings = _read_csv(od_path)
        od_savings["development"] = od_savings["development"].astype(int)
        col = f"tt_{scenario}"
        if col in od_savings.columns:
            match = od_savings.loc[od_savings["development"] == int(development), col]
            if not match.empty:
                result["od_monetized_tts"] = float(match.iloc[0])

    return result


def compare_tts_methods_summary(
    scenario: Optional[str] = None,
    max_rows: int = 20,
    base_dir: str = BASE_DIR,
) -> pd.DataFrame:
    """
    Build a summary table for all overlapping development-scenario pairs.
    Useful to see whether differences are systematic or isolated.
    """
    traffic_dir = os.path.join(base_dir, "traffic_flow")

    agg_status_quo = _read_csv(os.path.join(traffic_dir, "travel_time_status_quo.csv"))
    agg_developments = _read_csv(os.path.join(traffic_dir, "travel_time.csv"))
    od_status_quo = _read_csv(os.path.join(traffic_dir, "od", "status_quo_od_tt.csv"))
    od_developments = _read_csv(os.path.join(traffic_dir, "od", "developments_od_tt.csv"))

    aggregate_scenarios = [col for col in agg_developments.columns if col != "development"]
    if scenario is not None:
        aggregate_scenarios = [str(scenario)]

    od_developments["development"] = od_developments["development"].astype(int)
    od_developments["scenario"] = od_developments["scenario"].astype(str)

    keys = (
        od_developments[["development", "scenario"]]
        .drop_duplicates()
        .sort_values(["scenario", "development"])
    )

    rows = []
    for row in keys.itertuples(index=False):
        if row.scenario not in aggregate_scenarios:
            continue
        try:
            comparison = compare_tts_methods_for_case(
                scenario=row.scenario,
                development=int(row.development),
                base_dir=base_dir,
            )
            rows.append(comparison)
        except Exception as exc:
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "development": int(row.development),
                            "scenario": str(row.scenario),
                            "error": str(exc),
                        }
                    ]
                )
            )

    if not rows:
        return pd.DataFrame()

    summary = pd.concat(rows, ignore_index=True)

    if "aggregate_delta_tt" in summary.columns and "od_delta_tt" in summary.columns:
        summary["delta_gap"] = summary["aggregate_delta_tt"] - summary["od_delta_tt"]
        summary["relative_gap_vs_od"] = summary["delta_gap"] / summary["od_delta_tt"].replace(0, pd.NA)

    return summary.head(max_rows)
