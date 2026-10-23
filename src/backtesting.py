
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting import HORIZON, forecast_baselines
from src.metrics import mae, rmse, wape


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_ca1_foods.parquet"
OUTPUT_DIR = ROOT / "outputs"


def run_baseline_backtest(
    top_n: int = 100,
    n_folds: int = 4,
) -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)

    max_date = df["date"].max()

    # Select products using data available before the first test window.
    first_cutoff = max_date - pd.Timedelta(days=HORIZON * n_folds)

    eligible = df.loc[df["date"] <= first_cutoff]

    top_products = (
        eligible.groupby("item_id")["sales"]
        .sum()
        .nlargest(top_n)
        .index
    )

    df = df[df["item_id"].isin(top_products)].copy()

    results = []

    for fold in range(n_folds):
        test_start = (
            max_date
            - pd.Timedelta(days=HORIZON * (n_folds - fold))
            + pd.Timedelta(days=1)
        )
        test_end = test_start + pd.Timedelta(days=HORIZON - 1)
        cutoff = test_start - pd.Timedelta(days=1)

        train_df = df[df["date"] <= cutoff]
        test_df = df[
            (df["date"] >= test_start)
            & (df["date"] <= test_end)
        ]

        print(
            f"\nFold {fold + 1}: "
            f"train <= {cutoff.date()}, "
            f"test {test_start.date()} to {test_end.date()}"
        )

        for item_id in top_products:
            train = (
                train_df[train_df["item_id"] == item_id]
                .sort_values("date")["sales"]
            )

            test = (
                test_df[test_df["item_id"] == item_id]
                .sort_values("date")["sales"]
            )

            if len(test) != HORIZON:
                raise ValueError(
                    f"Unexpected test length for {item_id}, fold {fold + 1}"
                )

            predictions = forecast_baselines(train, HORIZON)

            for model_name, forecast in predictions.items():
                actual = test.to_numpy(dtype=float)

                results.append({
                    "fold": fold + 1,
                    "test_start": test_start,
                    "test_end": test_end,
                    "item_id": item_id,
                    "model": model_name,
                    "mae": mae(actual, forecast),
                    "rmse": rmse(actual, forecast),
                    "wape": wape(actual, forecast),
                    "absolute_error_sum": float(
                        np.abs(actual - forecast).sum()
                    ),
                    "squared_error_sum": float(
                        np.square(actual - forecast).sum()
                    ),
                    "actual_units": float(actual.sum()),
                    "forecast_units": float(forecast.sum()),
                    "observations": len(actual),
                })

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(
        OUTPUT_DIR / "baseline_results_rolling.csv",
        index=False,
    )

    summary = (
        results_df.groupby("model")
        .agg(
            total_absolute_error=("absolute_error_sum", "sum"),
            total_squared_error=("squared_error_sum", "sum"),
            total_actual_units=("actual_units", "sum"),
            total_observations=("observations", "sum"),
        )
    )

    summary["global_mae"] = (
        summary["total_absolute_error"] / summary["total_observations"]
    )
    summary["global_rmse"] = np.sqrt(
        summary["total_squared_error"] / summary["total_observations"]
    )
    summary["global_wape"] = (
        summary["total_absolute_error"] / summary["total_actual_units"]
    )

    print("\n=== ROLLING BACKTEST SUMMARY ===")
    print(
        summary[
            ["global_mae", "global_rmse", "global_wape"]
        ].round(4)
    )

    summary.to_csv(OUTPUT_DIR / "baseline_summary_rolling.csv")

    print("\nSaved:")
    print(OUTPUT_DIR / "baseline_results_rolling.csv")
    print(OUTPUT_DIR / "baseline_summary_rolling.csv")

    return results_df


if __name__ == "__main__":
    run_baseline_backtest()

