
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import mae, rmse, wape


ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    ROOT / "data" / "processed" / "sales_ca1_foods.parquet"
)

CALENDAR_PATH = (
    ROOT / "data" / "processed" / "calendar_features.parquet"
)

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

CALENDAR_FEATURES = [
    "has_event",
    "is_sporting_event",
    "is_national_event",
    "is_religious_event",
    "is_cultural_event",
    "snap_CA",
]

FEATURES_WITH_CALENDAR = FEATURES + CALENDAR_FEATURES


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features safe for direct 28-day forecasting."""
    df = df.sort_values(["item_id", "date"]).copy()

    df["item_code"] = (
        df["item_id"].astype("category").cat.codes
    )

    grouped = df.groupby("item_id")["sales"]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # The earliest sales lag is 28 days.
    # All features are available at forecast origin.
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


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge calendar features using date."""
    calendar = pd.read_parquet(CALENDAR_PATH)

    result = df.merge(
        calendar,
        on="date",
        how="left",
        validate="many_to_one",
    )

    if result[CALENDAR_FEATURES].isna().any().any():
        raise ValueError(
            "Missing calendar features after merge."
        )

    return result


def run_lightgbm_experiment(
    top_n: int = 20,
    use_calendar: bool = False,
) -> None:
    """Run a single 28-day LightGBM experiment."""
    print("Loading prepared sales data...")

    df = pd.read_parquet(DATA_PATH)

    cutoff = df["date"].max() - pd.Timedelta(
        days=HORIZON
    )

    top_items = (
        df.loc[df["date"] <= cutoff]
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

    train = df[
        (df["date"] <= cutoff) & valid_features
    ].copy()

    test = df[
        (df["date"] > cutoff) & valid_features
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
    model.fit(train[feature_columns], train["sales"])

    predictions = np.maximum(
        model.predict(test[feature_columns]),
        0,
    )

    actual = test["sales"].to_numpy(dtype=float)

    print("\n=== LIGHTGBM RESULTS ===")
    print(f"Calendar features: {use_calendar}")
    print(f"MAE:  {mae(actual, predictions):.4f}")
    print(f"RMSE: {rmse(actual, predictions):.4f}")
    print(f"WAPE: {wape(actual, predictions):.4%}")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(importance)

    print("\nExperiment completed.")


if __name__ == "__main__":
    run_lightgbm_experiment()
