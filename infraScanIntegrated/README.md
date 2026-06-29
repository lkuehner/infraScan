# infraScanIntegrated Quick Start

`infraScanIntegrated` orchestrates the joint evaluation of rail and road developments under one shared scenario and valuation framework. The integrated run is intended for comparative assessment across modes; legacy rail and legacy road runs remain available for module-specific replication.

## Module structure

The integrated workflow is organized in four modules:

1. `Module 0: Configuration, setup, and network preparation`
   Applies shared integrated settings, synchronizes rail and road defaults, and prepares common inputs.
2. `Module 1: Infrastructure developments`
   Generates candidate developments for road and rail.
3. `Module 2: Shared scenario generation`
   Creates one common scenario sample and selects representative scenarios for both modes.
4. `Module 3: Assessment and outputs`
   Runs rail evaluation, road evaluation, export of integrated score tables, and optional plots.

## Detailed phase mapping

The integrated modules call the legacy pipelines internally:

- `Module 0`
  Road phase 1-2, Rail phase 1-2
- `Module 1`
  Road phase 3, Rail phase 3-4
- `Module 2`
  Shared multimodal scenario generation
- `Module 3.1`
  Rail phase 5-6 and 9-12
- `Module 3.2`
  Road phase 5-7
- `Module 3.3`
  Optional legacy visualizations, integrated score export, integrated plots, runtime report

## Run modes

Three run modes are available in `main_integrated.py`:

- `integrated`
  Recommended default. Runs the full joint rail-road workflow.
- `legacy_rail`
  Runs the rail module with integrated valuation assumptions where applicable.
- `legacy_road`
  Runs the road module with its standalone logic.

## Core settings to choose

The central defaults are defined in `settings.py`. The most relevant settings are:

- `RUN_MODE`
  Selects `integrated`, `legacy_rail`, or `legacy_road`.
- `INCLUDE_STANDALONE`
  Adds legacy comparison outputs during integrated runs.
- `RUN_RAIL`, `RUN_ROAD`
  High-level mode toggles reserved for integrated orchestration logic.
- `PLOT_LEGACY_RAIL`, `PLOT_LEGACY_ROAD`, `PLOT_INTEGRATED`
  Controls output generation.
- `use_cache_rail`, `use_cache_road_checkpoints`, `use_cache_shared_scenarios`
  Reuses intermediate results to reduce runtime.

## Valuation settings

These parameters define the common appraisal framework:

- `start_valuation_year`
  Base year for valuation.
- `appraisal_years`
  Appraisal horizon.
- `discount_rate`
  Social discount rate.
- `rail_VTTS`, `road_VTTS`
  Mode-specific values of travel time savings.
- `real_wage_growth`
  Real annual wage growth for valuation scaling.

Most dynamization factors in the integrated valuation setup are calibrated to the default appraisal horizon of `40` years. Consequently, changes to `appraisal_years` should be interpreted cautiously, because not all scaling relationships were recalibrated for alternative horizons.

## Scenario settings

Integrated mode currently supports only:

- `scenario_type = "GENERATED"`

The main scenario controls are:

- `amount_of_scenarios`
  Total number of generated scenarios.
- `representative_scenarios_count`
  Number of retained representative scenarios.
- `start_year_scenario`, `end_year_scenario`
  Scenario horizon.

## Rail-specific integrated controls

The integrated wrapper exposes a small set of rail capacity options:

- `rail_visualization_mode`
  `manual`, `none`, or `all`
- `rail_grouping_strategy`
  `manual`, `conservative`, `baseline`, or `optimal`
- `rail_capacity_threshold`
  Capacity trigger threshold.
- `rail_max_enhancement_iterations`
  Upper bound for rail enhancement iterations.
- `rail_use_existing_capacity_prep`
  Reuses prepared rail capacity inputs.
- `rail_intervention_costs_reviewed`
  Uses the reviewed intervention cost set.

## Recommended default workflow

For a standard integrated run:

1. Keep `RUN_MODE = "integrated"`.
2. Keep `scenario_type = "GENERATED"`.
3. Set `start_valuation_year`, `appraisal_years`, and VTTS values.
4. Decide whether cached outputs should be reused.
5. Run `main_integrated.py`.

## Main outputs

The integrated run writes:

- integrated score tables (`long` and `tidy` exports),
- optional rail and road standalone comparison plots,
- integrated plots,
- a runtime and settings report.
