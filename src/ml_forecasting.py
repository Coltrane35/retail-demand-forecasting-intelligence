
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

PRICE_FEATURES = [
    "price_lag_28",
    "price_lag_56",
    "price_rolling_mean_28",
    "price_change_28",
    "price_available_28",
]

FEATURES_WITH_CALENDAR = (
    FEATURES + CALENDAR_FEATURES
)

FEATURES_WITH_CALENDAR_AND_PRICE = (
    FEATURES_WITH_CALENDAR + PRICE_FEATURES
)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sales and date features for a 28-day
    direct forecasting horizon.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["item_id", "date"]
    ).copy()

    df["item_code"] = (
        df["item_id"]
        .astype("category")
        .cat.codes
    )

    grouped = df.groupby(
        "item_id",
        sort=False
    )["sales"]

    for lag in LAGS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    df["rolling_mean_28"] = (
        grouped.transform(
            lambda s: (
                s.shift(HORIZON)
                .rolling(28)
                .mean()
            )
        )
    )

    df["rolling_std_28"] = (
        grouped.transform(
            lambda s: (
                s.shift(HORIZON)
                .rolling(28)
                .std()
            )
        )
    )

    df["day_of_week"] = (
        df["date"].dt.dayofweek
    )

    df["month"] = (
        df["date"].dt.month
    )

    df["day_of_month"] = (
        df["date"].dt.day
    )

    return df


def add_calendar_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Merge known calendar features by date."""

    calendar = pd.read_parquet(
        CALENDAR_PATH
    )

    calendar["date"] = pd.to_datetime(
        calendar["date"]
    )

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


def get_feature_columns(
    use_calendar: bool = False,
    use_price: bool = False,
) -> list[str]:

    if use_price and not use_calendar:
        raise ValueError(
            "Price experiment requires calendar features."
        )

    if use_price:
        return FEATURES_WITH_CALENDAR_AND_PRICE

    if use_calendar:
        return FEATURES_WITH_CALENDAR

    return FEATURES


def prepare_ml_data(
    df: pd.DataFrame,
    use_calendar: bool = False,
    use_price: bool = False,
) -> pd.DataFrame:
    """Prepare all selected features."""

    print("Creating historical sales features...")
    df = create_features(df)

    if use_calendar:
        print("Adding calendar features...")
        df = add_calendar_features(df)

    if use_price:
        print("Adding historical price features...")

        from src.price_features import add_price_features

        df = add_price_features(df)

    return df


def train_lightgbm(
    train: pd.DataFrame,
    feature_columns: list[str],
) -> lgb.LGBMRegressor:

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        train[feature_columns],
        train["sales"],
    )

    return model


def run_lightgbm_experiment(
    top_n: int = 20,
    use_calendar: bool = False,
    use_price: bool = False,
) -> None:

    print("Loading prepared sales data...")
    df = pd.read_parquet(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    cutoff = (
        df["date"].max()
        - pd.Timedelta(days=HORIZON)
    )

    top_items = (
        df.loc[df["date"] <= cutoff]
        .groupby("item_id")["sales"]
        .sum()
        .nlargest(top_n)
        .index
    )

    df = df[
        df["item_id"].isin(top_items)
    ].copy()

    df = prepare_ml_data(
        df,
        use_calendar=use_calendar,
        use_price=use_price,
    )

    feature_columns = get_feature_columns(
        use_calendar=use_calendar,
        use_price=use_price,
    )

    # Only historical sales features must be complete.
    # Missing price features are retained for LightGBM.
    valid_features = (
        df[FEATURES].notna().all(axis=1)
    )

    train = df[
        (df["date"] <= cutoff)
        & valid_features
    ].copy()

    test = df[
        (df["date"] > cutoff)
        & valid_features
    ].copy()

    if len(test) != top_n * HORIZON:
        raise ValueError(
            f"Unexpected test size: {len(test)}"
        )

    if train.empty:
        raise ValueError("Training data is empty.")

    print(f"Training rows: {len(train):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Cutoff: {cutoff.date()}")

    print("Training LightGBM...")

    model = train_lightgbm(
        train,
        feature_columns,
    )

    predictions = np.maximum(
        model.predict(test[feature_columns]),
        0,
    )

    actual = test["sales"].to_numpy(
        dtype=float
    )

    print("\n=== LIGHTGBM RESULTS ===")
    print(f"Calendar features: {use_calendar}")
    print(f"Price features: {use_price}")

    print(f"MAE:  {mae(actual, predictions):.4f}")
    print(f"RMSE: {rmse(actual, predictions):.4f}")
    print(f"WAPE: {wape(actual, predictions):.4%}")

    importance = pd.Series(
        model.feature_importances_,
        index=feature_columns,
    ).sort_values(ascending=False)

    print("\n=== FEATURE IMPORTANCE ===")
    print(importance.to_string())

    print("\nExperiment completed.")


if __name__ == "__main__":
    run_lightgbm_experiment()
