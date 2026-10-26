
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import mae, rmse, wape


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "processed" / "sales_ca1_foods.parquet"

HORIZON = 28
LAGS = [28, 35, 42, 56]

FEATURES = [
    "item_code",
    "lag_28",
    "lag_35",
    "lag_42",
    "lag_56",
    "rolling_mean_28",
    "rolling_std_28",
    "day_of_week",
    "month",
    "day_of_month",
]


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features available at a 28-day forecast origin."""
    df = df.sort_values(["item_id", "date"]).copy()

    df["item_code"] = (
        df["item_id"].astype("category").cat.codes
    )

    grouped = df.groupby("item_id")["sales"]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Shift by 28 days to avoid using sales
    # from inside the forecast horizon.
    df["rolling_mean_28"] = grouped.transform(
        lambda s: s.shift(28).rolling(28).mean()
    )

    df["rolling_std_28"] = grouped.transform(
        lambda s: s.shift(28).rolling(28).std()
    )

    df["day_of_week"] = df["date"].dt.day_of_week
    df["month"] = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day

    return df


def run_lightgbm_experiment(top_n: int = 20) -> None:
    print("Loading prepared sales data...")
    df = pd.read_parquet(DATA_PATH)

    cutoff = df["date"].max() - pd.Timedelta(days=HORIZON)

    # Select products using training history only.
    top_items = (
        df.loc[df["date"] <= cutoff]
        .groupby("item_id")["sales"]
        .sum()
        .nlargest(top_n)
        .index
    )

    df = df[df["item_id"].isin(top_items)].copy()

    print("Creating leakage-safe features...")
    df = create_features(df)

    train = df[
        (df["date"] <= cutoff)
        & df[FEATURES].notna().all(axis=1)
    ].copy()

    test = df[
        (df["date"] > cutoff)
        & df[FEATURES].notna().all(axis=1)
    ].copy()

    if len(test) != top_n * HORIZON:
        raise ValueError(
            f"Unexpected test size: {len(test)}"
        )

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Cutoff: {cutoff.date()}")

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
    model.fit(train[FEATURES], train["sales"])

    predictions = model.predict(test[FEATURES])
    predictions = np.maximum(predictions, 0)

    actual = test["sales"].to_numpy(dtype=float)

    print("\n=== LIGHTGBM RESULTS ===")
    print(f"MAE:  {mae(actual, predictions):.4f}")
    print(f"RMSE: {rmse(actual, predictions):.4f}")
    print(f"WAPE: {wape(actual, predictions):.4%}")

    print("\n=== FEATURE IMPORTANCE ===")
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print(importance)

    print("\nExperiment completed.")


if __name__ == "__main__":
    run_lightgbm_experiment()
