
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml_forecasting import (
    DATA_PATH,
    HORIZON,
    FEATURES,
    FEATURES_WITH_CALENDAR,
    FEATURES_WITH_CALENDAR_AND_PRICE,
    create_features,
    add_calendar_features,
)
from src.price_features import add_price_features
from src.metrics import mae, rmse, wape

import lightgbm as lgb


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

TOP_N = 100
N_FOLDS = 4


def run_ml_backtest(
    top_n: int = TOP_N,
    n_folds: int = N_FOLDS,
    use_calendar: bool = True,
    use_price: bool = True,
) -> pd.DataFrame:

    if use_price and not use_calendar:
        raise ValueError(
            "Price experiment requires calendar features."
        )

    if use_price:
        model_name = "lightgbm_calendar_price"
        feature_columns = FEATURES_WITH_CALENDAR_AND_PRICE

    elif use_calendar:
        model_name = "lightgbm_calendar"
        feature_columns = FEATURES_WITH_CALENDAR

    else:
        model_name = "lightgbm"
        feature_columns = FEATURES

    print(f"\nExperiment: {model_name}")
    print("Loading sales data...")

    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    max_date = df["date"].max()

    first_cutoff = (
        max_date
        - pd.Timedelta(days=HORIZON * n_folds)
    )

    # Select products using only historical sales
    # before the first test period.
    top_items = (
        df.loc[df["date"] <= first_cutoff]
        .groupby("item_id")["sales"]
        .sum()
        .nlargest(top_n)
        .index
    )

    if len(top_items) != top_n:
        raise ValueError(
            "Not enough products for backtesting."
        )

    df = df[
        df["item_id"].isin(top_items)
    ].copy()

    print("Creating historical features...")
    df = create_features(df)

    if use_calendar:
        print("Adding calendar features...")
        df = add_calendar_features(df)

    if use_price:
        print("Adding historical price features...")
        df = add_price_features(df)

    # Require complete sales features, but allow
    # NaN in historical price features.
    valid_features = (
        df[FEATURES].notna().all(axis=1)
    )

    results = []

    for fold in range(n_folds):

        test_start = (
            max_date
            - pd.Timedelta(
                days=HORIZON * (n_folds - fold)
            )
            + pd.Timedelta(days=1)
        )

        test_end = (
            test_start
            + pd.Timedelta(days=HORIZON - 1)
        )

        cutoff = (
            test_start
            - pd.Timedelta(days=1)
        )

        print(f"\n=== FOLD {fold + 1}/{n_folds} ===")
        print(f"Train cutoff: {cutoff.date()}")
        print(
            f"Test: {test_start.date()} "
            f"to {test_end.date()}"
        )

        train = df[
            (df["date"] <= cutoff)
            & valid_features
        ].copy()

        test = df[
            (df["date"] >= test_start)
            & (df["date"] <= test_end)
            & valid_features
        ].copy()

        expected_rows = top_n * HORIZON

        if len(test) != expected_rows:
            raise ValueError(
                f"Unexpected test size in fold "
                f"{fold + 1}: {len(test)} "
                f"(expected {expected_rows})"
            )

        if train.empty:
            raise ValueError(
                f"Empty training data in fold {fold + 1}"
            )

        # Calendar and sales features must be complete.
        # Historical price features can contain NaN.
        non_price_features = [
            col for col in feature_columns
            if not col.startswith("price_")
        ]

        if train[non_price_features].isna().any().any():
            raise ValueError(
                "Missing non-price training features."
            )

        if test[non_price_features].isna().any().any():
            raise ValueError(
                "Missing non-price test features."
            )

        print(f"Training rows: {len(train):,}")
        print(f"Test rows: {len(test):,}")

        if use_price:
            missing_prices = (
                test["price_lag_28"].isna().sum()
            )

            print(
                "Missing historical prices in test:",
                f"{missing_prices:,}",
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

        print("Training LightGBM...")

        model.fit(
            train[feature_columns],
            train["sales"],
        )

        predictions = np.maximum(
            model.predict(test[feature_columns]),
            0,
        )

        actual = test["sales"].to_numpy(
            dtype=float
        )

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
        item_results["model"] = model_name

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

    results_path = (
        OUTPUT_DIR
        / f"{model_name}_results_rolling.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / f"{model_name}_summary_rolling.csv"
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
        ].round(4).to_string()
    )

    summary.to_csv(summary_path)

    print("\nSaved results:")
    print(results_path)
    print(summary_path)

    return results_df


if __name__ == "__main__":
    run_ml_backtest(
        use_calendar=True,
        use_price=True,
    )
