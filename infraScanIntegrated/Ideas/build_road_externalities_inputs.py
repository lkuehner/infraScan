from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..scoring_registry import monetize_road_externalities_detail
from ...infraScanRoad.externalities_comp import RAW_OUTPUT_DIR


DATA_ROOT = Path("/Volumes/WD_Windows/MSc_Thesis")
INTEGRATED_COSTS_DIR = DATA_ROOT / "data Kopie" / "infraScanIntegrated" / "costs"
RAW_DETAIL_CSV = RAW_OUTPUT_DIR / "link_flow_externalities_long.csv"
ANNUALIZATION_FACTOR = 2.5 * 250


def main() -> None:
    INTEGRATED_COSTS_DIR.mkdir(parents=True, exist_ok=True)
    detail_df = pd.read_csv(RAW_DETAIL_CSV)

    # Monetize the raw link-flow externalities table and aggregate it to development-scenario level.
    detail_df = monetize_road_externalities_detail(
        detail_df=detail_df,
        annualization_factor=ANNUALIZATION_FACTOR,
    )
    monetization_df = (
        detail_df.groupby(["development", "scenario"], as_index=False)
        .agg(
            delta_vkm_peak_hour=("delta_vkm", "sum"),
            delta_vkm_peak_hour_new_link=(
                "delta_vkm",
                lambda s: s[detail_df.loc[s.index, "link_role"] == "new_link"].sum(),
            ),
            delta_vkm_peak_hour_existing_network=(
                "delta_vkm",
                lambda s: s[detail_df.loc[s.index, "link_role"] == "existing_network"].sum(),
            ),
            delta_vkm_peak_hour_noise_relevant=(
                "delta_vkm_annualized_noise_relevant",
                lambda s: s.sum() / ANNUALIZATION_FACTOR,
            ),
            road_accident_cost_annual=("road_accident_cost_annual", "sum"),
            road_airpollution_cost_annual=("road_airpollution_cost_annual", "sum"),
            road_co2_cost_annual=("road_co2_cost_annual", "sum"),
            road_noise_cost_annual=("road_noise_cost_annual", "sum"),
            road_land_consumption_cost=("road_land_consumption_cost", "sum"),
            mean_surface_share=("surface_share", "mean"),
            mean_settlement_buffer_share=("settlement_buffer_share", "mean"),
            mean_noise_relevant_share=("noise_relevant_share", "mean"),
        )
    )
    monetization_df["delta_vkm_annualized"] = monetization_df["delta_vkm_peak_hour"] * ANNUALIZATION_FACTOR
    monetization_df["delta_vkm_annualized_noise_relevant"] = (
        monetization_df["delta_vkm_peak_hour_noise_relevant"] * ANNUALIZATION_FACTOR
    )

    output_path = INTEGRATED_COSTS_DIR / "road_externalities_costs_long.csv"
    monetization_df.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
