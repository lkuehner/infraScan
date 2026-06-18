"""
TODO: EXPLAIN

what can be choosen
what has to be choosen
what is the default
"""



import warnings

from infraScan.infraScanIntegrated import pipeline_integrated as integrated_pipeline
from infraScan.infraScanIntegrated import settings as integrated_settings

from infraScan.infraScanRail import cost_parameters as rail_cost_parameters
from infraScan.infraScanRail import settings as rail_settings
from infraScan.infraScanRail.main_pipeline import infrascanrail_cap

from infraScan.infraScanRoad import cost_parameters as road_cost_parameters
from infraScan.infraScanRoad import settings as road_settings
from infraScan.infraScanRoad.main_pipeline import infrascanroad


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
        return infrascanrail_cap()

    if integrated_settings.RUN_MODE == "legacy_road":
        print("LEGACY ROAD MODE")
        print(f"Using road settings.py defaults with valuation year {road_settings.start_valuation_year}.")
        print(f"Appraisal period: {road_cost_parameters.duration} years.")
        return infrascanroad()

    return None


def infrascan_integrated():
    configure_integrated_run()
    warnings.filterwarnings("ignore")  # TODO: No warnings should be ignored

    if integrated_settings.RUN_MODE in ("legacy_rail", "legacy_road"):
        return run_legacy_mode()

    return integrated_pipeline.run_integrated_pipeline()


if __name__ == "__main__":
    infrascan_integrated()
