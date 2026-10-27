
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.forecasting import HORIZON
from src.metrics import mae, rmse, wape
from src.ml_forecasting import (
    DATA_PATH,
    FEATURES,
    FEATURES_WITH_CALENDAR,
    create_features,
    add_calendar_features,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

TOP_N = 100
N_FOLDS = 4


def run_ml_backtest(
    top_n: int = TOP_N,
    n_folds: int = N_FOLDS,
    use_calendar: bool = True,
) -> pd.DataFrame:

    print("Loading sales data...")
    df = pd.read_parquet(DATA_PATH)

    max_date = df["date"].max()

    first_cutoff = max_date - pd.Timedelta(
        days=HORIZON * n_folds
    )

    # Select products before the first test window.
    top_items = (
        df.loc[df["date"] <= first_cutoff]
        .groupby("item_id")["sales"]
        .sum()
        .nlargest(top_n)
        .index
    )

    df = df[df["item_id"].isin(top_items)].copy()

    print("Creating historical features...")
    df = create_features(df)

    feature_columns = FEATURES

    if use_calendar:
        print("Adding calendar features...")
        df = add_calendar_features(df)
        feature_columns = FEATURES_WITH_CALENDAR

    valid_features = df[feature_columns].notna().all(axis=1)

    results = []

    for fold in range(n_folds):
        test_start = (
            max_date
            - pd.Timedelta(
                days=HORIZON * (n_folds - fold)
            )
            + pd.Timedelta(days=1)
        )

        test_end = test_start + pd.Timedelta(
            days=HORIZON - 1
        )

        cutoff = test_start - pd.Timedelta(days=1)

        print(f"\n=== FOLD {fold + 1}/{n_folds} ===")
        print(f"Train cutoff: {cutoff.date()}")
        print(
            f"Test: {test_start.date()} "
            f"to {test_end.date()}"
        )

        train = df[
            (df["date"] <= cutoff) & valid_features
        ].copy()

        test = df[
            (df["date"] >= test_start)
            & (df["date"] <= test_end)
            & valid_features
        ].copy()

        if len(test) != top_n * HORIZON:
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

        model.fit(
            train[feature_columns],
            train["sales"],
        )

        predictions = np.maximum(
            model.predict(test[feature_columns]),
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

        item_results["model"] = (
            "lightgbm_calendar"
            if use_calendar
            else "lightgbm"
        )

        results.append(item_results)

    results_df = pd.concat(
        results,
        ignore_index=True,
    )

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

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    suffix = (
        "lightgbm_calendar"
        if use_calendar
        else "lightgbm"
    )

    results_path = (
        OUTPUT_DIR / f"{suffix}_results_rolling.csv"
    )

    summary_path = (
        OUTPUT_DIR / f"{suffix}_summary_rolling.csv"
    )

    results_df.to_csv(
        results_path,
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
            [
                "global_mae",
                "global_rmse",
                "global_wape",
            ]
        ].round(4)
    )

    summary.to_csv(summary_path)

    print("\nSaved results:")
    print(results_path)
    print(summary_path)

    return results_df


if __name__ == "__main__":
    run_ml_backtest(use_calendar=True)
