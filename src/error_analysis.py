
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

BASELINE_PATH = OUTPUT_DIR / "baseline_results_rolling.csv"
LIGHTGBM_PATH = OUTPUT_DIR / "lightgbm_results_rolling.csv"


def run_error_analysis():
    print("Loading backtesting results...")

    baseline = pd.read_csv(BASELINE_PATH)
    lightgbm = pd.read_csv(LIGHTGBM_PATH)

    baseline = baseline[
        baseline["model"] == "seasonal_average"
    ].copy()

    # Verify that both models were tested on the same
    # products and forecast windows.
    keys = ["fold", "item_id"]

    if baseline.duplicated(keys).any():
        raise ValueError("Duplicate baseline product/fold pairs.")

    if lightgbm.duplicated(keys).any():
        raise ValueError("Duplicate LightGBM product/fold pairs.")

    comparison = baseline.merge(
        lightgbm,
        on=keys,
        how="outer",
        suffixes=("_baseline", "_lightgbm"),
        validate="one_to_one",
        indicator=True,
    )

    if not comparison["_merge"].eq("both").all():
        raise ValueError(
            "Baseline and LightGBM test sets do not match."
        )

    comparison = comparison.drop(columns="_merge")

    for column in ["test_start", "test_end"]:
        left = comparison[f"{column}_baseline"]
        right = comparison[f"{column}_lightgbm"]

        if not left.equals(right):
            raise ValueError(
                f"Different test periods: {column}"
            )

    if not np.allclose(
        comparison["actual_units_baseline"],
        comparison["actual_units_lightgbm"],
    ):
        raise ValueError(
            "Actual sales differ between models."
        )

    comparison["mae_improvement"] = (
        comparison["mae_baseline"]
        - comparison["mae_lightgbm"]
    )

    comparison["absolute_error_improvement"] = (
        comparison["absolute_error_sum_baseline"]
        - comparison["absolute_error_sum_lightgbm"]
    )

    comparison["rmse_improvement"] = (
        comparison["rmse_baseline"]
        - comparison["rmse_lightgbm"]
    )

    comparison["lightgbm_wins"] = (
        comparison["mae_improvement"] > 0
    )

    print("\n=== PRODUCT-FOLD COMPARISON ===")
    print(f"Compared observations: {len(comparison)}")
    print(
        "LightGBM wins:",
        int(comparison["lightgbm_wins"].sum()),
    )
    print(
        "Seasonal Average wins:",
        int((comparison["mae_improvement"] < 0).sum()),
    )
    print(
        "Ties:",
        int((comparison["mae_improvement"] == 0).sum()),
    )

    product_summary = (
        comparison.groupby("item_id")
        .agg(
            baseline_absolute_error=(
                "absolute_error_sum_baseline", "sum"
            ),
            lightgbm_absolute_error=(
                "absolute_error_sum_lightgbm", "sum"
            ),
            actual_units=("actual_units_baseline", "sum"),
            lightgbm_winning_folds=(
                "lightgbm_wins", "sum"
            ),
        )
        .reset_index()
    )

    product_summary["absolute_error_improvement"] = (
        product_summary["baseline_absolute_error"]
        - product_summary["lightgbm_absolute_error"]
    )

    product_summary["baseline_wape"] = np.where(
        product_summary["actual_units"] > 0,
        product_summary["baseline_absolute_error"]
        / product_summary["actual_units"],
        np.nan,
    )

    product_summary["lightgbm_wape"] = np.where(
        product_summary["actual_units"] > 0,
        product_summary["lightgbm_absolute_error"]
        / product_summary["actual_units"],
        np.nan,
    )

    product_summary["wape_improvement_pp"] = (
        product_summary["baseline_wape"]
        - product_summary["lightgbm_wape"]
    ) * 100

    product_summary = product_summary.sort_values(
        "absolute_error_improvement",
        ascending=False,
    )

    print("\n=== TOP 10 PRODUCTS IMPROVED BY LIGHTGBM ===")
    print(
        product_summary[
            [
                "item_id",
                "actual_units",
                "absolute_error_improvement",
                "wape_improvement_pp",
            ]
        ].head(10).round(2).to_string(index=False)
    )

    print("\n=== TOP 10 PRODUCTS WORSENED BY LIGHTGBM ===")
    print(
        product_summary[
            [
                "item_id",
                "actual_units",
                "absolute_error_improvement",
                "wape_improvement_pp",
            ]
        ].tail(10).sort_values(
            "absolute_error_improvement"
        ).round(2).to_string(index=False)
    )

    fold_summary = (
        comparison.groupby("fold")
        .agg(
            baseline_error=("absolute_error_sum_baseline", "sum"),
            lightgbm_error=("absolute_error_sum_lightgbm", "sum"),
            actual_units=("actual_units_baseline", "sum"),
        )
    )

    fold_summary["baseline_wape"] = (
        fold_summary["baseline_error"]
        / fold_summary["actual_units"]
    )

    fold_summary["lightgbm_wape"] = (
        fold_summary["lightgbm_error"]
        / fold_summary["actual_units"]
    )

    fold_summary["improvement_pp"] = (
        fold_summary["baseline_wape"]
        - fold_summary["lightgbm_wape"]
    ) * 100

    print("\n=== WAPE BY FOLD ===")
    print(
        fold_summary[
            [
                "baseline_wape",
                "lightgbm_wape",
                "improvement_pp",
            ]
        ].round(4)
    )

    comparison.to_csv(
        OUTPUT_DIR / "model_error_comparison.csv",
        index=False,
    )

    product_summary.to_csv(
        OUTPUT_DIR / "product_error_analysis.csv",
        index=False,
    )

    fold_summary.to_csv(
        OUTPUT_DIR / "fold_error_analysis.csv",
    )

    print("\nError analysis completed.")
    print("Saved 3 CSV files to outputs/.")


if __name__ == "__main__":
    run_error_analysis()
