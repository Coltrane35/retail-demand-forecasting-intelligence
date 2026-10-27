
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_ca1_foods.parquet"
OUTPUT_DIR = ROOT / "outputs"

HORIZON = 28
N_FOLDS = 4


def classify_demand(sales_frequency):
    if sales_frequency >= 0.70:
        return "regular"
    if sales_frequency >= 0.20:
        return "intermittent"
    return "sparse"


def run_segmentation():
    print("Loading sales and backtesting results...")

    sales = pd.read_parquet(DATA_PATH)

    baseline = pd.read_csv(
        OUTPUT_DIR / "baseline_results_rolling.csv"
    )
    lightgbm = pd.read_csv(
        OUTPUT_DIR / "lightgbm_results_rolling.csv"
    )

    baseline = baseline[
        baseline["model"] == "seasonal_average"
    ].copy()

    max_date = sales["date"].max()
    first_cutoff = max_date - pd.Timedelta(
        days=HORIZON * N_FOLDS
    )

    # Segment products using only information available
    # before the first backtesting window.
    historical = sales[
        sales["date"] <= first_cutoff
    ].copy()

    segments = (
        historical.groupby("item_id")
        .agg(
            historical_days=("sales", "size"),
            selling_days=(
                "sales",
                lambda s: (s > 0).sum()
            ),
            historical_units=("sales", "sum"),
        )
        .reset_index()
    )

    segments["sales_frequency"] = (
        segments["selling_days"]
        / segments["historical_days"]
    )

    segments["demand_segment"] = (
        segments["sales_frequency"]
        .apply(classify_demand)
    )

    print("\n=== ALL PRODUCTS BY DEMAND SEGMENT ===")
    print(
        segments["demand_segment"]
        .value_counts()
    )

    results = pd.concat(
        [baseline, lightgbm],
        ignore_index=True,
    )

    results = results.merge(
        segments[
            [
                "item_id",
                "sales_frequency",
                "demand_segment",
            ]
        ],
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    if results["demand_segment"].isna().any():
        raise ValueError("Missing product segments.")

    summary = (
        results.groupby(
            ["demand_segment", "model"]
        )
        .agg(
            products=("item_id", "nunique"),
            absolute_error=(
                "absolute_error_sum", "sum"
            ),
            squared_error=(
                "squared_error_sum", "sum"
            ),
            actual_units=("actual_units", "sum"),
            observations=("observations", "sum"),
        )
        .reset_index()
    )

    summary["mae"] = (
        summary["absolute_error"]
        / summary["observations"]
    )

    summary["rmse"] = np.sqrt(
        summary["squared_error"]
        / summary["observations"]
    )

    summary["wape"] = np.where(
        summary["actual_units"] > 0,
        summary["absolute_error"]
        / summary["actual_units"],
        np.nan,
    )

    print("\n=== MODEL PERFORMANCE BY SEGMENT ===")
    print(
        summary[
            [
                "demand_segment",
                "model",
                "products",
                "mae",
                "rmse",
                "wape",
            ]
        ].round(4).to_string(index=False)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    segments.to_csv(
        OUTPUT_DIR / "demand_segments.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "demand_segment_performance.csv",
        index=False,
    )

    print("\nSegmentation completed.")


if __name__ == "__main__":
    run_segmentation()

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_ca1_foods.parquet"
OUTPUT_DIR = ROOT / "outputs"

HORIZON = 28
N_FOLDS = 4


def classify_demand(sales_frequency: float) -> str:
    """Classify products by historical selling frequency."""
    if sales_frequency >= 0.70:
        return "regular"
    elif sales_frequency >= 0.20:
        return "intermittent"
    else:
        return "sparse"


def run_segmentation() -> pd.DataFrame:
    print("Loading sales and backtesting results...")

    sales = pd.read_parquet(DATA_PATH)

    baseline = pd.read_csv(
        OUTPUT_DIR / "baseline_results_rolling.csv"
    )

    lightgbm = pd.read_csv(
        OUTPUT_DIR / "lightgbm_results_rolling.csv"
    )

    baseline = baseline[
        baseline["model"] == "seasonal_average"
    ].copy()

    max_date = sales["date"].max()

    first_cutoff = max_date - pd.Timedelta(
        days=HORIZON * N_FOLDS
    )

    # Use historical sales only.
    # No future data is used for segmentation.
    historical = sales[
        sales["date"] <= first_cutoff
    ].copy()

    segments = (
        historical.groupby("item_id")
        .agg(
            historical_days=("sales", "size"),
            selling_days=(
                "sales",
                lambda s: (s > 0).sum()
            ),
            historical_units=("sales", "sum"),
        )
        .reset_index()
    )

    segments["sales_frequency"] = (
        segments["selling_days"]
        / segments["historical_days"]
    )

    segments["demand_segment"] = (
        segments["sales_frequency"]
        .apply(classify_demand)
    )

    print("\n=== ALL PRODUCTS BY DEMAND SEGMENT ===")
    print(
        segments["demand_segment"].value_counts()
    )

    # Compare both forecasting models.
    results = pd.concat(
        [baseline, lightgbm],
        ignore_index=True,
    )

    results = results.merge(
        segments[
            [
                "item_id",
                "sales_frequency",
                "demand_segment",
            ]
        ],
        on="item_id",
        how="left",
        validate="many_to_one",
    )

    if results["demand_segment"].isna().any():
        raise ValueError("Missing product segments.")

    summary = (
        results.groupby(
            ["demand_segment", "model"]
        )
        .agg(
            products=("item_id", "nunique"),
            absolute_error=(
                "absolute_error_sum", "sum"
            ),
            squared_error=(
                "squared_error_sum", "sum"
            ),
            actual_units=("actual_units", "sum"),
            observations=("observations", "sum"),
        )
        .reset_index()
    )

    summary["mae"] = (
        summary["absolute_error"]
        / summary["observations"]
    )

    summary["rmse"] = np.sqrt(
        summary["squared_error"]
        / summary["observations"]
    )

    summary["wape"] = np.where(
        summary["actual_units"] > 0,
        summary["absolute_error"]
        / summary["actual_units"],
        np.nan,
    )

    print("\n=== MODEL PERFORMANCE BY SEGMENT ===")
    print(
        summary[
            [
                "demand_segment",
                "model",
                "products",
                "mae",
                "rmse",
                "wape",
            ]
        ].round(4).to_string(index=False)
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    segments.to_csv(
        OUTPUT_DIR / "demand_segments.csv",
        index=False,
    )

    summary.to_csv(
        OUTPUT_DIR / "demand_segment_performance.csv",
        index=False,
    )

    print("\nSegmentation completed.")
    print("Saved results to outputs/.")

    return summary


if __name__ == "__main__":
    run_segmentation()
