from __future__ import annotations

import argparse
import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, minimize


MSC_ROOT = Path(__file__).resolve().parents[3]
if str(MSC_ROOT) not in sys.path:
    sys.path.insert(0, str(MSC_ROOT))

from infraScan.infraScanRoad import scoring as road_scoring  # noqa: E402


DEFAULT_ROAD_BASE = MSC_ROOT / "data" / "infraScanRoad"

SOLVER_PRESETS: dict[str, dict[str, Any]] = {
    "legacy_fast": {"method": "SLSQP", "ftol": 1e5, "eps": 1e5, "maxiter": 3},
    "slsqp_pragmatic": {"method": "SLSQP", "ftol": 1e3, "eps": 1e3, "maxiter": 10},
    "slsqp_moderate": {"method": "SLSQP", "ftol": 1e-1, "eps": 1e-1, "maxiter": 25},
    "trust_light": {"method": "trust-constr", "maxiter": 3},
}


DEFAULT_CASES = [
    ("scenario_19", None),
    ("scenario_19", 557),
    ("scenario_19", 612),
    ("scenario_30", 557),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Road SUE optimizer settings on selected cases.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ROAD_BASE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "road_sue_optimizer_benchmark",
    )
    parser.add_argument("--presets", nargs="*", default=["legacy_fast", "slsqp_pragmatic", "trust_light"])
    parser.add_argument("--cases", nargs="*", default=None, help="Cases like sq:scenario_19 or 557:scenario_19")
    return parser.parse_args()


def parse_cases(values: list[str] | None) -> list[tuple[str, int | None]]:
    if not values:
        return DEFAULT_CASES
    cases: list[tuple[str, int | None]] = []
    for value in values:
        dev, scenario = value.split(":", 1)
        cases.append((scenario, None if dev in {"sq", "status_quo"} else int(dev)))
    return cases


def _count_nonfinite(values: Any) -> int:
    arr = np.asarray(values, dtype=float)
    return int(np.size(arr) - np.isfinite(arr).sum())


def _load_base_network(base_dir: Path) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    points = gpd.read_file(base_dir / "Network" / "processed" / "points_with_attribute.gpkg")
    points.index = points.index.astype(int)
    points = points.sort_index()

    edges = gpd.read_file(base_dir / "Network" / "processed" / "edges_with_attribute.gpkg")
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=["ID_edge"])
    return points, edges


def _build_development_network(
    base_dir: Path,
    development_id: int,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    links_developments = gpd.read_file(base_dir / "costs" / "construction.gpkg")
    points_developments = gpd.read_file(base_dir / "Network" / "processed" / "generated_nodes.gpkg")
    points_current, links_current = _load_base_network(base_dir)

    point_temp = points_developments[points_developments["ID_new"] == development_id]
    if point_temp.empty:
        raise ValueError(f"Development {development_id} missing in generated_nodes.gpkg")

    edge_temp = links_developments[links_developments["ID_new"] == development_id]
    if edge_temp.empty:
        raise ValueError(f"Development {development_id} missing in construction.gpkg")

    points = points_current.copy()
    new_point_row = {
        "intersection": 0,
        "ID_point": 9999,
        "geometry": point_temp.geometry.iloc[0],
        "open_ends": None,
        "within_corridor": True,
        "on_corridor_border": False,
        "generate_traffic": 0,
    }
    points = gpd.GeoDataFrame(pd.concat([points, pd.DataFrame(pd.Series(new_point_row)).T], ignore_index=True))
    points.index = points.index.astype(int)
    points = points.sort_index()
    points["id_dummy"] = points.index.values

    edges = links_current.copy()
    edge_id_max = edges["ID_edge"].astype(int).max()
    index_point_start = points[points["id_dummy"] == edge_temp["ID_current"].values[0]].index[0]
    new_edge_row = {
        "start": index_point_start,
        "end": 9999,
        "geometry": edge_temp["geometry"].iloc[0],
        "ffs": 120,
        "capacity": 2200,
        "start_access": False,
        "end_access": True,
        "polygon_border": False,
        "ID_edge": edge_id_max + 1,
    }
    edges = gpd.GeoDataFrame(pd.concat([edges, pd.DataFrame(pd.Series(new_edge_row)).T], ignore_index=True))
    edges["ID_edge"] = edges["ID_edge"].astype(int)
    edges = edges.sort_values(by=["ID_edge"])

    voronoi = gpd.read_file(base_dir / "Network" / "travel_time" / "developments" / f"dev{development_id}_Voronoi.gpkg")
    return points, edges, voronoi


def _load_case_inputs(
    base_dir: Path,
    scenario: str,
    development_id: int | None,
) -> tuple[pd.DataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, str]:
    if development_id is None:
        od_matrix = pd.read_csv(
            base_dir / "traffic_flow" / "od" / "scenarios_voronoi" / "generated" / f"od_matrix_{scenario}.csv",
            index_col=0,
        )
        points, edges = _load_base_network(base_dir)
        voronoi = gpd.read_file(base_dir / "Network" / "travel_time" / "Voronoi_statusquo.gpkg")
        dev_label = "status_quo"
    else:
        od_matrix = pd.read_csv(
            base_dir
            / "traffic_flow"
            / "od"
            / "development_voronoi"
            / "generated"
            / f"od_matrix_dev{development_id}_{scenario}.csv",
            index_col=0,
        )
        points, edges, voronoi = _build_development_network(base_dir, development_id)
        dev_label = str(development_id)

    return od_matrix, points, edges, voronoi, dev_label


def int_cost_fun_with_optional_xi2(Xi: np.ndarray, par: dict[str, np.ndarray], force_xi2: bool) -> np.ndarray:
    if force_xi2:
        Xi = np.full_like(np.asarray(Xi, dtype=float), 2.0)
    return road_scoring.IntCostFun(Xi, par)


@contextmanager
def patched_sue_runner(
    *,
    solver_options: dict[str, Any],
    force_xi2: bool,
):
    original = road_scoring.SUE_C_Logit
    diagnostics: dict[str, Any] = {}

    def benchmark_sue(nroutes, D_od, par, delta_ir, delta_odr, cf_r, theta):
        D_od = np.nan_to_num(np.asarray(D_od).flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        cf_r = np.nan_to_num(np.asarray(cf_r).flatten(), nan=0.0, posinf=0.0, neginf=0.0)
        delta_ir_local = np.nan_to_num(np.asarray(delta_ir), nan=0.0, posinf=0.0, neginf=0.0)
        delta_odr_local = np.nan_to_num(np.asarray(delta_odr), nan=0.0, posinf=0.0, neginf=0.0)

        def int_links_times(D_r):
            x_i = np.matmul(delta_ir_local, D_r)
            return int_cost_fun_with_optional_xi2(x_i, par, force_xi2=force_xi2)

        def objective(x):
            x_safe = x.copy()
            x_safe[x_safe <= 0] = 0.0001
            temp_log = np.log(x_safe)
            temp_log[np.isinf(temp_log)] = 0.1
            temp_log[np.isnan(temp_log)] = 0.1
            return np.sum(int_links_times(x_safe)) + np.sum(x_safe * temp_log) + np.sum(x_safe * cf_r)

        row_sums = np.sum(delta_odr_local, axis=1).astype(float)
        tt = np.divide(
            D_od.transpose(),
            row_sums,
            out=np.zeros_like(D_od, dtype=float).transpose(),
            where=row_sums != 0,
        ).transpose()
        tt = np.nan_to_num(tt, nan=0.0, posinf=0.0, neginf=0.0)

        D_r0 = np.matmul(delta_odr_local.transpose(), tt)
        D_r0 = np.nan_to_num(D_r0, nan=0.01, posinf=0.01, neginf=0.01)
        D_r0[D_r0 <= 0] = 0.01

        lower_bound = 0.01
        lb = np.zeros(np.shape(D_r0)).flatten() + lower_bound
        ub = (max(D_od) * np.ones(np.shape(D_r0))).flatten() * 5
        bounds = Bounds(lb, ub)

        eq_cons = {"type": "eq", "fun": lambda x: (np.matmul(delta_odr_local, x) - D_od).flatten()}
        ineq_cons = {"type": "ineq", "fun": lambda x: x}

        method = solver_options["method"]
        if method == "trust-constr":
            options = {
                "maxiter": solver_options.get("maxiter", 3),
                "verbose": 0,
                "disp": False,
            }
            for key in ["gtol", "xtol", "barrier_tol"]:
                if key in solver_options:
                    options[key] = solver_options[key]
        else:
            options = {
                "ftol": solver_options.get("ftol", 1e3),
                "eps": solver_options.get("eps", 1e3),
                "maxiter": solver_options.get("maxiter", 10),
                "disp": False,
            }

        started = time.perf_counter()
        res = minimize(
            objective,
            D_r0.flatten(),
            method=method,
            constraints=[eq_cons, ineq_cons],
            options=options,
            bounds=bounds,
        )
        elapsed = time.perf_counter() - started

        D_r = res.x.copy()
        D_r[D_r <= 0] = 0.001
        x_i = delta_ir_local * D_r
        intTrec_i = int_cost_fun_with_optional_xi2(x_i, par, force_xi2=force_xi2)

        eq_residual = np.matmul(delta_odr_local, D_r) - D_od
        link_flows = np.asarray(x_i).sum(axis=1) if np.asarray(x_i).ndim > 1 else np.asarray(x_i).flatten()
        link_tt = road_scoring.CostFun(link_flows.reshape(-1, 1), par).flatten()

        diagnostics.clear()
        diagnostics.update(
            {
                "success": bool(res.success),
                "status": int(res.status),
                "message": str(res.message),
                "nit": int(getattr(res, "nit", -1)),
                "nfev": int(getattr(res, "nfev", -1)),
                "fun": float(res.fun),
                "elapsed_sue_s": elapsed,
                "max_abs_eq_violation": float(np.max(np.abs(eq_residual))),
                "mean_abs_eq_violation": float(np.mean(np.abs(eq_residual))),
                "relative_eq_violation": float(np.sum(np.abs(eq_residual)) / np.sum(D_od)),
                "min_route_flow": float(np.min(D_r)),
                "share_at_lower_bound": float(np.mean(D_r <= lower_bound * 1.01)),
                "nroutes": int(nroutes),
                "n_od_pairs": int(len(D_od)),
                "zero_route_init_count": int(np.sum(row_sums == 0)),
                "nonfinite_D_od": _count_nonfinite(D_od),
                "nonfinite_cf_r": _count_nonfinite(cf_r),
                "nonfinite_delta_ir": _count_nonfinite(delta_ir_local),
                "nonfinite_delta_odr": _count_nonfinite(delta_odr_local),
                "max_flow_capacity_ratio": float(np.nanmax(link_flows.reshape(-1, 1) / par["Xmax_i"])),
                "max_link_tt_min": float(np.nanmax(link_tt) * 60.0),
            }
        )
        fval = float(objective(D_r))
        return [x_i, D_r, intTrec_i, fval]

    road_scoring.SUE_C_Logit = benchmark_sue
    try:
        yield diagnostics
    finally:
        road_scoring.SUE_C_Logit = original


def run_case(
    *,
    base_dir: Path,
    scenario: str,
    development_id: int | None,
    preset_name: str,
    solver_options: dict[str, Any],
    force_xi2: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    od_matrix, points, edges, voronoi, dev_label = _load_case_inputs(base_dir, scenario, development_id)

    started = time.perf_counter()
    with patched_sue_runner(solver_options=solver_options, force_xi2=force_xi2) as diagnostics:
        od_tt_df = road_scoring.travel_flow_optimization_by_od(
            OD_matrix=od_matrix,
            points=points,
            edges=edges,
            voronoi=voronoi,
            dev=dev_label,
            scen=scenario,
            export_link_flows=False,
            flow_output_path=None,
        )
    total_elapsed = time.perf_counter() - started

    demand = np.clip(od_tt_df["demand"].to_numpy(dtype=float), 0, None)
    tt = od_tt_df["travel_time"].to_numpy(dtype=float)
    diagnostics = dict(diagnostics)
    diagnostics.update(
        {
            "preset": preset_name,
            "xi2_objective": bool(force_xi2),
            "scenario": scenario,
            "development": "status_quo" if development_id is None else int(development_id),
            "elapsed_total_s": total_elapsed,
            "od_rows": int(len(od_tt_df)),
            "mean_tt_h": float(np.nanmean(tt)),
            "median_tt_h": float(np.nanmedian(tt)),
            "weighted_mean_tt_h": float(np.average(tt, weights=demand)) if demand.sum() > 0 else np.nan,
            "total_demand": float(demand.sum()),
        }
    )
    return od_tt_df, diagnostics


def compare_pair(rows: list[dict[str, Any]], outputs: dict[tuple[str, str, bool], pd.DataFrame]) -> pd.DataFrame:
    deltas = []
    row_df = pd.DataFrame(rows)
    for (scenario, dev, xi2), dev_group in row_df[row_df["development"] != "status_quo"].groupby(
        ["scenario", "development", "xi2_objective"]
    ):
        sq_group = row_df[
            (row_df["scenario"] == scenario)
            & (row_df["development"] == "status_quo")
            & (row_df["xi2_objective"] == xi2)
        ]
        for _, dev_row in dev_group.iterrows():
            sq_match = sq_group[sq_group["preset"] == dev_row["preset"]]
            if sq_match.empty:
                continue
            sq_row = sq_match.iloc[0]
            deltas.append(
                {
                    "preset": dev_row["preset"],
                    "xi2_objective": xi2,
                    "scenario": scenario,
                    "development": dev,
                    "delta_weighted_mean_tt_h": sq_row["weighted_mean_tt_h"] - dev_row["weighted_mean_tt_h"],
                    "delta_mean_tt_h": sq_row["mean_tt_h"] - dev_row["mean_tt_h"],
                    "sq_weighted_mean_tt_h": sq_row["weighted_mean_tt_h"],
                    "dev_weighted_mean_tt_h": dev_row["weighted_mean_tt_h"],
                }
            )
    return pd.DataFrame(deltas)


def main() -> None:
    args = parse_args()
    base_dir = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = parse_cases(args.cases)
    presets = {name: SOLVER_PRESETS[name] for name in args.presets}
    xi2_modes = [True] if args.only_xi2 else ([False, True] if args.include_xi2 else [False])

    rows: list[dict[str, Any]] = []
    outputs: dict[tuple[str, str, bool], pd.DataFrame] = {}
    results_path = output_dir / "optimizer_benchmark_results.csv"
    deltas_path = output_dir / "optimizer_benchmark_tt_deltas.csv"
    for scenario, development_id in cases:
        label = "status_quo" if development_id is None else f"dev{development_id}"
        for preset_name, solver_options in presets.items():
            for force_xi2 in xi2_modes:
                print(f"Running {label} {scenario} | {preset_name} | xi2={force_xi2}", flush=True)
                try:
                    od_tt_df, diagnostics = run_case(
                        base_dir=base_dir,
                        scenario=scenario,
                        development_id=development_id,
                        preset_name=preset_name,
                        solver_options=solver_options,
                        force_xi2=force_xi2,
                    )
                    outputs[(f"{label}_{scenario}", preset_name, force_xi2)] = od_tt_df
                    rows.append(diagnostics)
                    pd.DataFrame(rows).to_csv(results_path, index=False)
                    compare_pair(rows, outputs).to_csv(deltas_path, index=False)
                    print(
                        f"  done: success={diagnostics['success']} "
                        f"runtime={diagnostics['elapsed_total_s']:.1f}s "
                        f"rel_eq={diagnostics['relative_eq_violation']:.3g} "
                        f"weighted_tt={diagnostics['weighted_mean_tt_h']:.4f}h",
                        flush=True,
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "preset": preset_name,
                            "xi2_objective": bool(force_xi2),
                            "scenario": scenario,
                            "development": "status_quo" if development_id is None else int(development_id),
                            "success": False,
                            "status": -999,
                            "message": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    pd.DataFrame(rows).to_csv(results_path, index=False)
                    compare_pair(rows, outputs).to_csv(deltas_path, index=False)
                    print(f"  failed: {type(exc).__name__}: {exc}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False)
    deltas = compare_pair(rows, outputs)
    deltas.to_csv(deltas_path, index=False)

    print(f"\nSaved {results_path}")
    print(f"Saved {deltas_path}")
    print("\nSummary:")
    cols = [
        "preset",
        "xi2_objective",
        "scenario",
        "development",
        "success",
        "elapsed_total_s",
        "nit",
        "nfev",
        "relative_eq_violation",
        "weighted_mean_tt_h",
        "max_flow_capacity_ratio",
        "max_link_tt_min",
    ]
    print(results[[c for c in cols if c in results.columns]].to_string(index=False))
    if not deltas.empty:
        print("\nStatus quo - development weighted mean TT deltas:")
        print(deltas.to_string(index=False))


if __name__ == "__main__":
    main()
