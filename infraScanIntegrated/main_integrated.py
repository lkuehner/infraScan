"""
TODO: EXPLAIN

what can be choosen
what has to be choosen
what is the default
"""



import warnings

from infraScan.infraScanIntegrated import pipeline_integrated as integrated_pipeline
from infraScan.infraScanIntegrated import paths as integrated_paths
from infraScan.infraScanIntegrated import settings as integrated_settings

from infraScan.infraScanRail import cost_parameters as rail_cost_parameters
from infraScan.infraScanRail import settings as rail_settings
from infraScan.infraScanRail.main_pipeline import infrascanrail_cap

from infraScan.infraScanRoad import cost_parameters as road_cost_parameters
from infraScan.infraScanRoad import settings as road_settings
from infraScan.infraScanRoad.main_pipeline import infrascanroad


def write_standalone_run_report(run_mode):
    if run_mode == "legacy_rail":
        report_path = integrated_paths.RAIL_STANDALONE_RUN_REPORT_PATH
        title = "INFRASCANRAIL STANDALONE RUN REPORT"
        settings_rows = [
            ("Run mode", "rail_standalone"),
            ("Valuation year", rail_settings.start_valuation_year),
            ("Appraisal period [years]", rail_cost_parameters.duration),
            ("Discount rate", f"{rail_cost_parameters.discount_rate:.2%}"),
            ("Scenario mode", rail_settings.scenario_type),
            ("VTTS [CHF/h]", rail_cost_parameters.VTTS),
        ]
    elif run_mode == "legacy_road":
        report_path = integrated_paths.ROAD_STANDALONE_RUN_REPORT_PATH
        title = "INFRASCANROAD STANDALONE RUN REPORT"
        settings_rows = [
            ("Run mode", "road_standalone"),
            ("Valuation year", road_settings.start_valuation_year),
            ("Appraisal period [years]", road_cost_parameters.duration),
            ("Discount rate", "not applied in road standalone static model"),
            ("Scenario mode", road_settings.scenario_type),
            ("Travel time method", road_settings.travel_time_savings_method),
            ("VTTS [CHF/h]", road_cost_parameters.VTTS),
        ]
    else:
        return None

    with open(report_path, "w") as file:
        file.write("=" * 80 + "\n")
        file.write(title + "\n")
        file.write("=" * 80 + "\n\n")
        file.write("SETTINGS\n")
        file.write("-" * 80 + "\n")
        for label, value in settings_rows:
            file.write(f"{label:.<40} {value}\n")
        file.write("=" * 80 + "\n")

    print(f"Standalone run report saved to: {report_path}")
    return report_path


def configure_integrated_run():
    print("INFRASCAN INTEGRATED CONFIGURATION")
    print("-" * 80)

    print("\n1. Run mode")
    print("   1) Integrated")
    print("   2) Legacy Rail")
    print("   3) Legacy Road")

    while True:
        mode_choice = input("\n   Select mode (1-3) [1]: ").strip() or "1"
        if mode_choice in {"1", "2", "3"}:
            break
        print("   Invalid selection. Please enter 1, 2, or 3.")

    integrated_settings.RUN_MODE = {
        "1": "integrated",
        "2": "legacy_rail",
        "3": "legacy_road",
    }[mode_choice]

    if integrated_settings.RUN_MODE == "integrated":
        standalone_choice = (
            input("   Include standalone comparison outputs? (y/n) [y]: ").strip().lower() or "y"
        )
        integrated_settings.INCLUDE_STANDALONE = standalone_choice == "y"
        valuation_choice = input(
            f"   Valuation year [{integrated_settings.start_valuation_year}]: "
        ).strip()
        if valuation_choice:
            integrated_settings.start_valuation_year = int(valuation_choice)
    else:
        integrated_settings.INCLUDE_STANDALONE = False

    print(f"\n   -> Run mode: {integrated_settings.RUN_MODE}")
    if integrated_settings.RUN_MODE == "integrated":
        print(f"   -> Valuation year: {integrated_settings.start_valuation_year}")
        print(f"   -> Appraisal period: {integrated_settings.appraisal_years} years")
        print(f"   -> Rail VTTS: {integrated_settings.rail_VTTS} CHF/h")
        print(f"   -> Road VTTS: {integrated_settings.road_VTTS} CHF/h")
        print(f"   -> Discount rate: {integrated_settings.discount_rate:.0%}")
        print(f"   -> Scenario mode: {integrated_settings.scenario_type}")
        #print(f"   -> Representative scenarios: {integrated_settings.representative_scenarios_count}")
    print("-" * 80)


def run_legacy_mode():
    if integrated_settings.RUN_MODE == "legacy_rail":
        print("LEGACY RAIL MODE")
        print(f"Using rail settings.py defaults with valuation year {rail_settings.start_valuation_year}.")
        print(f"Appraisal period: {rail_cost_parameters.duration} years; discount rate: {rail_cost_parameters.discount_rate:.0%}.")
        integrated_pipeline.sync_integrated_shared_settings()
        integrated_pipeline.apply_integrated_overrides_to_rail()
        result = infrascanrail_cap()
        write_standalone_run_report("legacy_rail")
        return result

    if integrated_settings.RUN_MODE == "legacy_road":
        print("LEGACY ROAD MODE")
        print(f"Using road settings.py defaults with valuation year {road_settings.start_valuation_year}.")
        print(f"Appraisal period: {road_cost_parameters.duration} years.")
        result = infrascanroad()
        write_standalone_run_report("legacy_road")
        return result

    return None


def infrascan_integrated():
    configure_integrated_run()
    warnings.filterwarnings("ignore")  # TODO: No warnings should be ignored

    if integrated_settings.RUN_MODE in ("legacy_rail", "legacy_road"):
        return run_legacy_mode()

    if integrated_settings.scenario_type != "GENERATED":
        raise ValueError(
            "Integrated mode only supports GENERATED scenarios. "
            "Use legacy rail or legacy road mode for specific scenarios."
        )

    return integrated_pipeline.run_integrated_pipeline()


if __name__ == "__main__":
    infrascan_integrated()
