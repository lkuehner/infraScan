import os

import pandas as pd

from infraScan.infraScanRoad import settings as road_settings


def _safe_corr(series_a: pd.Series, series_b: pd.Series, method: str) -> float:
    merged = pd.concat([series_a, series_b], axis=1).dropna()
    if len(merged) < 2:
        return float("nan")
    return float(merged.iloc[:, 0].corr(merged.iloc[:, 1], method=method))


def run_compare() -> pd.DataFrame:
    os.chdir(road_settings.MAIN)

    costs_dir = "data/infraScanRoad/costs"
    od_path = os.path.join(costs_dir, "traveltime_savings_od.csv")
    agg_path = os.path.join(costs_dir, "tt_analysis_aggregate_same_basis_signed.csv")

    if not os.path.exists(od_path):
        raise FileNotFoundError(f"Missing OD results file: {od_path}")
    if not os.path.exists(agg_path):
        raise FileNotFoundError(
            f"Missing apples-to-apples aggregate file: {agg_path}. "
            f"Run tt_method_apples_to_apples_analysis first."
        )

    od = pd.read_csv(od_path)
    agg = pd.read_csv(agg_path)

    id_col_od = "development" if "development" in od.columns else "ID_new"
    id_col_agg = "development" if "development" in agg.columns else "ID_new"

    od = od.rename(columns={id_col_od: "development"}).copy()
    agg = agg.rename(columns={id_col_agg: "development"}).copy()

    od_tt_cols = [c for c in od.columns if c.startswith("tt_")]
    agg_tt_cols = [c for c in agg.columns if c.startswith("tt_")]
    common = sorted(set(od_tt_cols).intersection(agg_tt_cols))

    if not common:
        raise ValueError(
            "No common tt_* scenario columns between OD and apples-to-apples aggregate outputs."
        )

    merged = od[["development"] + common].merge(
        agg[["development"] + common],
        on="development",
        how="inner",
        suffixes=("_od", "_agg_same_basis"),
    )

    if merged.empty:
        raise ValueError("No overlapping developments between OD and apples-to-apples aggregate outputs.")

    rows = []
    for col in common:
        col_od = f"{col}_od"
        col_agg = f"{col}_agg_same_basis"

        delta = merged[col_agg] - merged[col_od]

        rows.append(
            {
                "scenario_col": col,
                "n_developments": int(merged[[col_od, col_agg]].dropna().shape[0]),
                "mean_od": float(merged[col_od].mean()),
                "mean_agg_same_basis": float(merged[col_agg].mean()),
                "mean_delta_agg_minus_od": float(delta.mean()),
                "median_delta_agg_minus_od": float(delta.median()),
                "mean_abs_delta": float(delta.abs().mean()),
                "pearson": _safe_corr(merged[col_od], merged[col_agg], method="pearson"),
                "spearman": _safe_corr(merged[col_od], merged[col_agg], method="spearman"),
            }
        )

    summary = pd.DataFrame(rows).sort_values("scenario_col").reset_index(drop=True)

    out_path = os.path.join(costs_dir, "tt_method_od_vs_agg_same_basis_summary.csv")
    summary.to_csv(out_path, index=False)

    print("OD vs apples-to-apples aggregate comparison summary:")
    print(summary.to_string(index=False))
    print(f"\nSaved summary CSV: {out_path}")

    return summary


if __name__ == "__main__":
    run_compare()