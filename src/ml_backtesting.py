
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.forecasting import HORIZON
from src.metrics import mae, rmse, wape
from src.ml_forecasting import create_features, FEATURES


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_ca1_foods.parquet"
OUTPUT_DIR = ROOT / "outputs"

TOP_N = 100
N_FOLDS = 4


def run_ml_backtest():
    print("Loading sales data...")
    df = pd.read_parquet(DATA_PATH)

    max_date = df["date"].max()
    first_cutoff = max_date - pd.Timedelta(
        days=HORIZON * N_FOLDS
    )

    # Use only information available before the first fold.
    top_items = (
        df.loc[df["date"] <= first_cutoff]
        .groupby("item_id")["sales"]
        .sum()
        .nlargest(TOP_N)
        .index
    )

    df = df[df["item_id"].isin(top_items)].copy()

    print("Creating features...")
    df = create_features(df)

    results = []

    for fold in range(N_FOLDS):
        test_start = (
            max_date
            - pd.Timedelta(days=HORIZON * (N_FOLDS - fold))
            + pd.Timedelta(days=1)
        )
        test_end = test_start + pd.Timedelta(
            days=HORIZON - 1
        )
        cutoff = test_start - pd.Timedelta(days=1)

        print(f"\n=== FOLD {fold + 1}/{N_FOLDS} ===")
        print(f"Train cutoff: {cutoff.date()}")
        print(f"Test: {test_start.date()} to {test_end.date()}")

        # A training row is valid only when its target
        # would already be known at the fold cutoff.
        train = df[
            (df["date"] <= cutoff)
            & df[FEATURES].notna().all(axis=1)
        ].copy()

        test = df[
            (df["date"] >= test_start)
            & (df["date"] <= test_end)
            & df[FEATURES].notna().all(axis=1)
        ].copy()

        if len(test) != TOP_N * HORIZON:
            raise ValueError(
                f"Unexpected test size in fold {fold + 1}: "
                f"{len(test)}"
            )

        model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )

        print(f"Training rows: {len(train):,}")
        print(f"Test rows: {len(test):,}")
        print("Training LightGBM...")

        model.fit(train[FEATURES], train["sales"])

        predictions = np.maximum(
            model.predict(test[FEATURES]),
            0,
        )

        actual = test["sales"].to_numpy(dtype=float)

        print(f"MAE:  {mae(actual, predictions):.4f}")
        print(f"RMSE: {rmse(actual, predictions):.4f}")
        print(f"WAPE: {wape(actual, predictions):.4%}")

        test["prediction"] = predictions
        test["absolute_error"] = np.abs(
            test["sales"] - test["prediction"]
        )
        test["squared_error"] = (
            test["sales"] - test["prediction"]
        ) ** 2

        item_results = (
            test.groupby("item_id")
            .agg(
                absolute_error_sum=(
                    "absolute_error", "sum"
                ),
                squared_error_sum=(
                    "squared_error", "sum"
                ),
                actual_units=("sales", "sum"),
                forecast_units=("prediction", "sum"),
                observations=("sales", "size"),
            )
            .reset_index()
        )

        item_results["fold"] = fold + 1
        item_results["test_start"] = test_start
        item_results["test_end"] = test_end
        item_results["model"] = "lightgbm"

        results.append(item_results)

    results_df = pd.concat(results, ignore_index=True)

    results_df["mae"] = (
        results_df["absolute_error_sum"]
        / results_df["observations"]
    )
    results_df["rmse"] = np.sqrt(
        results_df["squared_error_sum"]
        / results_df["observations"]
    )
    results_df["wape"] = np.where(
        results_df["actual_units"] > 0,
        results_df["absolute_error_sum"]
        / results_df["actual_units"],
        np.nan,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(
        OUTPUT_DIR / "lightgbm_results_rolling.csv",
        index=False,
    )

    summary = (
        results_df.groupby("model")
        .agg(
            total_absolute_error=(
                "absolute_error_sum", "sum"
            ),
            total_squared_error=(
                "squared_error_sum", "sum"
            ),
            total_actual_units=("actual_units", "sum"),
            total_observations=("observations", "sum"),
        )
    )

    summary["global_mae"] = (
        summary["total_absolute_error"]
        / summary["total_observations"]
    )

    summary["global_rmse"] = np.sqrt(
        summary["total_squared_error"]
        / summary["total_observations"]
    )

    summary["global_wape"] = (
        summary["total_absolute_error"]
        / summary["total_actual_units"]
    )

    print("\n=== LIGHTGBM ROLLING SUMMARY ===")
    print(
        summary[
            ["global_mae", "global_rmse", "global_wape"]
        ].round(4)
    )

    summary.to_csv(
        OUTPUT_DIR / "lightgbm_summary_rolling.csv"
    )

    print("\nSaved results:")
    print(OUTPUT_DIR / "lightgbm_results_rolling.csv")
    print(OUTPUT_DIR / "lightgbm_summary_rolling.csv")

    return results_df


if __name__ == "__main__":
    run_ml_backtest()
