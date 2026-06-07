from __future__ import annotations
import re
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


DATA_ROOT = Path("/Volumes/WD_Windows/MSc_Thesis/euler/infraScanRoad_trust_2iter_alldev_10sce")
LINK_FLOW_DIR = DATA_ROOT / "traffic_flow" / "od" / "link_flows"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "road_vkt_summary"
SELECTED_SCENARIOS = (
    "scenario_26",
    "scenario_70",
    "scenario_89",
    "scenario_100",
    "scenario_75",
    "scenario_96",
    "scenario_44",
    "scenario_19",
    "scenario_64",
    "scenario_78",
)
SELECTED_SCENARIO_SET = set(SELECTED_SCENARIOS)

STATUS_QUO_RE = re.compile(r"^status_quo_(scenario_\d+)\.csv$")
DEV_RE = re.compile(r"^dev(\d+)_(scenario_\d+)\.csv$")


def scenario_key(name: str) -> int:
    return int(name.split("_")[-1])


def compute_vkt(flow_path: Path) -> float:
    veh_km = 0.0
    with flow_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                length_m = float(row["length_m"])
                flow = float(row["flow"])
            except (KeyError, TypeError, ValueError):
                continue
            veh_km += (length_m / 1000.0) * flow
    return veh_km


def read_flow_rows(flow_path: Path) -> list[dict]:
    rows = []
    with flow_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        "ID_edge": int(row["ID_edge"]),
                        "length_m": float(row["length_m"]),
                        "flow": float(row["flow"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def build_summary() -> list[dict]:
    rows: list[dict] = []

    for flow_path in sorted(LINK_FLOW_DIR.glob("*.csv")):
        status_match = STATUS_QUO_RE.match(flow_path.name)
        if status_match:
            scenario = status_match.group(1)
            if scenario not in SELECTED_SCENARIO_SET:
                continue
            rows.append(
                {
                    "development": "status_quo",
                    "scenario": scenario,
                    "veh_km": compute_vkt(flow_path),
                    "case_type": "status_quo",
                }
            )
            continue

        dev_match = DEV_RE.match(flow_path.name)
        if not dev_match:
            continue

        development, scenario = dev_match.groups()
        if scenario not in SELECTED_SCENARIO_SET:
            continue
        rows.append(
            {
                "development": int(development),
                "scenario": scenario,
                "veh_km": compute_vkt(flow_path),
                "case_type": "development",
            }
        )

    if not rows:
        raise RuntimeError(f"No link flow files found in {LINK_FLOW_DIR}")

    return sorted(
        rows,
        key=lambda row: (
            scenario_key(row["scenario"]),
            row["case_type"],
            str(row["development"]),
        ),
    )


def build_link_split_rows() -> list[dict]:
    status_quo_edge_max = {}
    status_quo_total = {}

    for scenario in SELECTED_SCENARIOS:
        sq_path = LINK_FLOW_DIR / f"status_quo_{scenario}.csv"
        sq_rows = read_flow_rows(sq_path)
        status_quo_edge_max[scenario] = max(row["ID_edge"] for row in sq_rows)
        status_quo_total[scenario] = sum((row["length_m"] / 1000.0) * row["flow"] for row in sq_rows)

    out_rows = []
    for flow_path in sorted(LINK_FLOW_DIR.glob("dev*_scenario_*.csv")):
        dev_match = DEV_RE.match(flow_path.name)
        if not dev_match:
            continue
        development, scenario = dev_match.groups()
        if scenario not in SELECTED_SCENARIO_SET:
            continue

        flow_rows = read_flow_rows(flow_path)
        new_link_threshold = status_quo_edge_max[scenario]
        new_link_veh_km = sum(
            (row["length_m"] / 1000.0) * row["flow"]
            for row in flow_rows
            if row["ID_edge"] > new_link_threshold
        )
        rest_network_veh_km = sum(
            (row["length_m"] / 1000.0) * row["flow"]
            for row in flow_rows
            if row["ID_edge"] <= new_link_threshold
        )
        total_veh_km = new_link_veh_km + rest_network_veh_km
        status_quo_veh_km = status_quo_total[scenario]

        out_rows.append(
            {
                "development": int(development),
                "scenario": scenario,
                "total_veh_km": total_veh_km,
                "new_link_veh_km": new_link_veh_km,
                "rest_network_veh_km": rest_network_veh_km,
                "status_quo_veh_km": status_quo_veh_km,
                "delta_total_veh_km": total_veh_km - status_quo_veh_km,
                "delta_rest_network_veh_km": rest_network_veh_km - status_quo_veh_km,
            }
        )

    return sorted(
        out_rows,
        key=lambda row: (scenario_key(row["scenario"]), row["development"]),
    )


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_absolute_pivot(summary: list[dict]) -> tuple[list[str], list[dict]]:
    scenarios = sorted({row["scenario"] for row in summary}, key=scenario_key)
    grouped: dict[str, dict[str, float]] = defaultdict(dict)

    for row in summary:
        development = str(row["development"])
        grouped[development][row["scenario"]] = row["veh_km"]

    def development_sort_key(value: str) -> tuple[int, int | str]:
        if value == "status_quo":
            return (0, value)
        return (1, int(value))

    pivot_rows = []
    for development in sorted(grouped, key=development_sort_key):
        out_row: dict[str, str | float] = {"development": development}
        for scenario in scenarios:
            out_row[scenario] = grouped[development].get(scenario, "")
        pivot_rows.append(out_row)

    return scenarios, pivot_rows


def build_plot(scenario_stats: list[dict], output_path: Path) -> None:
    width = 1800
    height = 900
    margin_left = 110
    margin_right = 40
    margin_top = 60
    margin_bottom = 180
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    values = []
    for row in scenario_stats:
        values.extend([
            row["status_quo_veh_km"] / 1_000_000,
            row["development_mean_veh_km"] / 1_000_000,
            row["development_median_veh_km"] / 1_000_000,
        ])
    y_min = min(values)
    y_max = max(values)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else max(1.0, y_max * 0.08)
    y_min -= y_pad
    y_max += y_pad

    def x_pos(index: int) -> float:
        if len(scenario_stats) == 1:
            return margin_left + plot_width / 2
        return margin_left + (plot_width * index / (len(scenario_stats) - 1))

    def y_pos(value_mio: float) -> float:
        return margin_top + plot_height * (1.0 - (value_mio - y_min) / (y_max - y_min))

    def polyline(rows: list[dict], key: str, color: str) -> str:
        points = " ".join(
            f"{x_pos(idx):.1f},{y_pos(row[key] / 1_000_000):.1f}"
            for idx, row in enumerate(rows)
        )
        return f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />'

    grid = []
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = y_pos(value)
        grid.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="#d9d9d9" stroke-dasharray="4,4" />'
        )
        grid.append(
            f'<text x="{margin_left - 12}" y="{y + 5:.1f}" text-anchor="end" '
            f'font-size="18" fill="#333">{value:.1f}</text>'
        )

    labels = []
    points = []
    series = [
        ("status_quo_veh_km", "#c43c39", "Status quo"),
        ("development_mean_veh_km", "#2b6cb0", "Development mean"),
        ("development_median_veh_km", "#2f855a", "Development median"),
    ]
    legend = []
    for legend_idx, (_, color, label) in enumerate(series):
        ly = margin_top - 20 + legend_idx * 24
        legend.append(f'<line x1="{width - 260}" y1="{ly}" x2="{width - 230}" y2="{ly}" stroke="{color}" stroke-width="4" />')
        legend.append(f'<text x="{width - 220}" y="{ly + 6}" font-size="18" fill="#222">{label}</text>')

    for idx, row in enumerate(scenario_stats):
        x = x_pos(idx)
        labels.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 60}" transform="rotate(45 {x:.1f},{height - margin_bottom + 60})" '
            f'text-anchor="start" font-size="16" fill="#333">{row["scenario"]}</text>'
        )
        for key, color, _ in series:
            y = y_pos(row[key] / 1_000_000)
            points.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}" />')

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="34" text-anchor="middle" font-size="28" fill="#111">Road veh-km by scenario</text>
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="20" fill="#222">Scenario</text>
<text x="24" y="{height / 2}" transform="rotate(-90 24,{height / 2})" text-anchor="middle" font-size="20" fill="#222">veh-km [million]</text>
{''.join(grid)}
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#444" stroke-width="2" />
<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#444" stroke-width="2" />
{polyline(scenario_stats, "status_quo_veh_km", "#c43c39")}
{polyline(scenario_stats, "development_mean_veh_km", "#2b6cb0")}
{polyline(scenario_stats, "development_median_veh_km", "#2f855a")}
{''.join(points)}
{''.join(labels)}
{''.join(legend)}
</svg>
"""
    output_path.write_text(svg)


def build_absolute_heatmap(
    pivot_rows: list[dict],
    scenarios: list[str],
    output_path: Path,
) -> None:
    cell_w = 28
    cell_h = 16
    label_w = 110
    top_h = 140
    right_pad = 30
    bottom_pad = 30
    width = label_w + len(scenarios) * cell_w + right_pad
    height = top_h + len(pivot_rows) * cell_h + bottom_pad

    values = [
        float(row[scenario])
        for row in pivot_rows
        for scenario in scenarios
        if row[scenario] != ""
    ]
    vmin = min(values)
    vmax = max(values)

    def color_for(value: float) -> str:
        if vmax == vmin:
            ratio = 0.5
        else:
            ratio = (value - vmin) / (vmax - vmin)
        r = int(245 - ratio * 175)
        g = int(247 - ratio * 120)
        b = int(250 - ratio * 40)
        return f"rgb({r},{g},{b})"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white" />',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="22" fill="#111">Road veh-km absolute values by scenario and development</text>',
        f'<text x="{width / 2}" y="52" text-anchor="middle" font-size="15" fill="#444">Darker cells indicate higher absolute veh-km</text>',
    ]

    for col_idx, scenario in enumerate(scenarios):
        x = label_w + col_idx * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top_h - 12}" transform="rotate(-60 {x:.1f},{top_h - 12})" '
            f'text-anchor="end" font-size="12" fill="#333">{scenario}</text>'
        )

    for row_idx, row in enumerate(pivot_rows):
        y = top_h + row_idx * cell_h
        label = str(row["development"])
        font_weight = "700" if label == "status_quo" else "400"
        parts.append(
            f'<text x="{label_w - 8}" y="{y + 12}" text-anchor="end" font-size="11" '
            f'font-weight="{font_weight}" fill="#222">{label}</text>'
        )
        for col_idx, scenario in enumerate(scenarios):
            x = label_w + col_idx * cell_w
            value = row[scenario]
            fill = "#f3f4f6" if value == "" else color_for(float(value))
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{fill}" stroke="#ffffff" stroke-width="0.5" />'
            )

    legend_x = width - 220
    legend_y = 70
    legend_w = 160
    legend_h = 14
    for i in range(legend_w):
        ratio = i / max(legend_w - 1, 1)
        value = vmin + ratio * (vmax - vmin)
        parts.append(
            f'<rect x="{legend_x + i}" y="{legend_y}" width="1" height="{legend_h}" fill="{color_for(value)}" stroke="none" />'
        )
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_w}" height="{legend_h}" fill="none" stroke="#999" stroke-width="0.6" />')
    parts.append(f'<text x="{legend_x}" y="{legend_y - 6}" font-size="11" fill="#333">{vmin / 1_000_000:.2f} mio veh-km</text>')
    parts.append(f'<text x="{legend_x + legend_w}" y="{legend_y - 6}" text-anchor="end" font-size="11" fill="#333">{vmax / 1_000_000:.2f} mio veh-km</text>')

    parts.append("</svg>")
    output_path.write_text("".join(parts))


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def build_development_boxplot(summary: list[dict], output_path: Path) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in summary:
        grouped[str(row["development"])].append(float(row["veh_km"]))

    developments = sorted(
        grouped,
        key=lambda value: (-1 if value == "status_quo" else int(value)),
    )
    stats = []
    all_values = []
    for development in developments:
        values = sorted(grouped[development])
        q1 = percentile(values, 0.25)
        med = percentile(values, 0.50)
        q3 = percentile(values, 0.75)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        whisker_low = min(v for v in values if v >= lower_fence)
        whisker_high = max(v for v in values if v <= upper_fence)
        stats.append(
            {
                "development": development,
                "min": min(values),
                "q1": q1,
                "median": med,
                "q3": q3,
                "max": max(values),
                "whisker_low": whisker_low,
                "whisker_high": whisker_high,
            }
        )
        all_values.extend(values)

    width = max(1800, 120 + len(developments) * 18)
    height = 850
    margin_left = 100
    margin_right = 30
    margin_top = 70
    margin_bottom = 190
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    y_min = min(all_values)
    y_max = max(all_values)
    y_pad = (y_max - y_min) * 0.08 if y_max > y_min else max(1.0, y_max * 0.08)
    y_min -= y_pad
    y_max += y_pad

    def x_pos(index: int) -> float:
        if len(developments) == 1:
            return margin_left + plot_width / 2
        return margin_left + plot_width * index / (len(developments) - 1)

    def y_pos(value: float) -> float:
        return margin_top + plot_height * (1.0 - (value - y_min) / (y_max - y_min))

    grid = []
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = y_pos(value)
        grid.append(
            f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" '
            f'stroke="#d9d9d9" stroke-dasharray="4,4" />'
        )
        grid.append(
            f'<text x="{margin_left - 10}" y="{y + 5:.1f}" text-anchor="end" font-size="16" fill="#333">'
            f'{value / 1_000_000:.2f}</text>'
        )

    box_width = max(6, min(14, plot_width / max(len(developments), 1) * 0.6))
    shapes = []
    labels = []
    for idx, row in enumerate(stats):
        x = x_pos(idx)
        y_q1 = y_pos(row["q1"])
        y_q3 = y_pos(row["q3"])
        y_med = y_pos(row["median"])
        y_low = y_pos(row["whisker_low"])
        y_high = y_pos(row["whisker_high"])
        shapes.append(f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_q3:.1f}" stroke="#444" stroke-width="1.2" />')
        shapes.append(f'<line x1="{x:.1f}" y1="{y_q1:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" stroke="#444" stroke-width="1.2" />')
        shapes.append(f'<line x1="{x - box_width/2:.1f}" y1="{y_high:.1f}" x2="{x + box_width/2:.1f}" y2="{y_high:.1f}" stroke="#444" stroke-width="1.2" />')
        shapes.append(f'<line x1="{x - box_width/2:.1f}" y1="{y_low:.1f}" x2="{x + box_width/2:.1f}" y2="{y_low:.1f}" stroke="#444" stroke-width="1.2" />')
        shapes.append(
            f'<rect x="{x - box_width/2:.1f}" y="{y_q3:.1f}" width="{box_width:.1f}" height="{max(y_q1 - y_q3, 1):.1f}" '
            f'fill="#9ecae1" stroke="#2b6cb0" stroke-width="1.2" />'
        )
        shapes.append(f'<line x1="{x - box_width/2:.1f}" y1="{y_med:.1f}" x2="{x + box_width/2:.1f}" y2="{y_med:.1f}" stroke="#c43c39" stroke-width="1.6" />')
        labels.append(
            f'<text x="{x:.1f}" y="{height - margin_bottom + 60}" transform="rotate(45 {x:.1f},{height - margin_bottom + 60})" '
            f'text-anchor="start" font-size="11" fill="#333">{row["development"]}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
<rect width="100%" height="100%" fill="white" />
<text x="{width / 2}" y="30" text-anchor="middle" font-size="24" fill="#111">Road veh-km distribution by development</text>
<text x="{width / 2}" y="54" text-anchor="middle" font-size="15" fill="#444">Each box summarizes the 10 selected scenarios for one development, including status quo</text>
<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-size="18" fill="#222">Development</text>
<text x="24" y="{height / 2}" transform="rotate(-90 24,{height / 2})" text-anchor="middle" font-size="18" fill="#222">veh-km [million]</text>
{''.join(grid)}
<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height - margin_bottom}" stroke="#444" stroke-width="2" />
<line x1="{margin_left}" y1="{height - margin_bottom}" x2="{width - margin_right}" y2="{height - margin_bottom}" stroke="#444" stroke-width="2" />
{''.join(shapes)}
{''.join(labels)}
</svg>
"""
    output_path.write_text(svg)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = build_summary()
    link_split_rows = build_link_split_rows()
    sq_by_scenario = {
        row["scenario"]: row["veh_km"]
        for row in summary
        if row["case_type"] == "status_quo"
    }

    comparison = []
    grouped_devs: dict[str, list[float]] = defaultdict(list)
    for row in summary:
        if row["case_type"] != "development":
            continue
        status_quo_veh_km = sq_by_scenario[row["scenario"]]
        delta_veh_km = row["veh_km"] - status_quo_veh_km
        delta_pct = (delta_veh_km / status_quo_veh_km) * 100.0 if status_quo_veh_km else 0.0
        comparison.append(
            {
                "development": row["development"],
                "scenario": row["scenario"],
                "veh_km": row["veh_km"],
                "status_quo_veh_km": status_quo_veh_km,
                "delta_veh_km": delta_veh_km,
                "delta_pct": delta_pct,
            }
        )
        grouped_devs[row["scenario"]].append(row["veh_km"])

    scenario_stats = []
    for scenario in sorted(grouped_devs, key=scenario_key):
        values = grouped_devs[scenario]
        sq_value = sq_by_scenario[scenario]
        delta_values = [value - sq_value for value in values]
        delta_pct_values = [((value - sq_value) / sq_value) * 100.0 for value in values] if sq_value else [0.0]
        scenario_stats.append(
            {
                "scenario": scenario,
                "status_quo_veh_km": sq_value,
                "development_mean_veh_km": mean(values),
                "development_median_veh_km": median(values),
                "development_min_veh_km": min(values),
                "development_max_veh_km": max(values),
                "mean_delta_veh_km": mean(delta_values),
                "median_delta_veh_km": median(delta_values),
                "mean_delta_pct": mean(delta_pct_values),
                "median_delta_pct": median(delta_pct_values),
                "n_developments": len(values),
            }
        )

    write_csv(
        OUTPUT_DIR / "road_vkt_all_cases.csv",
        summary,
        ["development", "scenario", "veh_km", "case_type"],
    )
    scenarios, absolute_pivot = build_absolute_pivot(summary)
    write_csv(
        OUTPUT_DIR / "road_vkt_absolute_by_dev_and_scenario.csv",
        absolute_pivot,
        ["development", *scenarios],
    )
    write_csv(
        OUTPUT_DIR / "road_vkt_new_link_vs_rest_network.csv",
        link_split_rows,
        [
            "development",
            "scenario",
            "total_veh_km",
            "new_link_veh_km",
            "rest_network_veh_km",
            "status_quo_veh_km",
            "delta_total_veh_km",
            "delta_rest_network_veh_km",
        ],
    )
    write_csv(
        OUTPUT_DIR / "road_vkt_status_quo.csv",
        [
            {"scenario": scenario, "status_quo_veh_km": sq_by_scenario[scenario]}
            for scenario in sorted(sq_by_scenario, key=scenario_key)
        ],
        ["scenario", "status_quo_veh_km"],
    )
    write_csv(
        OUTPUT_DIR / "road_vkt_developments_vs_status_quo.csv",
        comparison,
        ["development", "scenario", "veh_km", "status_quo_veh_km", "delta_veh_km", "delta_pct"],
    )
    write_csv(
        OUTPUT_DIR / "road_vkt_scenario_stats.csv",
        scenario_stats,
        [
            "scenario",
            "status_quo_veh_km",
            "development_mean_veh_km",
            "development_median_veh_km",
            "development_min_veh_km",
            "development_max_veh_km",
            "mean_delta_veh_km",
            "median_delta_veh_km",
            "mean_delta_pct",
            "median_delta_pct",
            "n_developments",
        ],
    )

    build_plot(scenario_stats, OUTPUT_DIR / "road_vkt_quickplot.svg")
    build_absolute_heatmap(
        absolute_pivot,
        scenarios,
        OUTPUT_DIR / "road_vkt_absolute_heatmap.svg",
    )
    build_development_boxplot(
        summary,
        OUTPUT_DIR / "road_vkt_boxplot_by_development.svg",
    )


if __name__ == "__main__":
    main()
