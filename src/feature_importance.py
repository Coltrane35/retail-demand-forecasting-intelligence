
"""
Sprint 09.1 - LightGBM Feature Importance

Calculates:
- Gain importance
- Split importance

Outputs:
- outputs/feature_importance/feature_importance.csv
- outputs/feature_importance/feature_importance_gain.png
- outputs/feature_importance/feature_importance_split.png

Uses the same feature engineering and LightGBM
hyperparameters as the rolling backtesting experiment.

The model is trained only on observations available
before the first rolling backtesting fold.
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from src.ml_forecasting import (
    DATA_PATH,
    HORIZON,
    FEATURES,
    FEATURES_WITH_CALENDAR_AND_PRICE,
    create_features,
    add_calendar_features,
)
from src.price_features import add_price_features


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "feature_importance"

TOP_N = 100
N_FOLDS = 4

FEATURE_COLUMNS = FEATURES_WITH_CALENDAR_AND_PRICE


def prepare_data() -> tuple[pd.DataFrame, pd.Timestamp]:
    """Prepare historical training data without test leakage."""

    print("Loading M5 sales data...")

    df = pd.read_parquet(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"])

    max_date = df["date"].max()

    first_cutoff = (
        max_date
        - pd.Timedelta(days=HORIZON * N_FOLDS)
    )

    print(f"First cutoff: {first_cutoff.date()}")

    top_items = (
        df.loc[df["date"] <= first_cutoff]
        .groupby("item_id")["sales"]
        .sum()
        .nlargest(TOP_N)
        .index
    )

    if len(top_items) != TOP_N:
        raise ValueError(
            f"Expected {TOP_N} products, "
            f"found {len(top_items)}."
        )

    df = df[df["item_id"].isin(top_items)].copy()

    print("Creating historical sales features...")
    df = create_features(df)

    print("Adding calendar features...")
    df = add_calendar_features(df)

    print("Adding historical price features...")
    df = add_price_features(df)

    # Require complete historical sales features.
    # LightGBM handles missing historical prices.
    valid_sales_features = (
        df[FEATURES].notna().all(axis=1)
    )

    train = df[
        (df["date"] <= first_cutoff)
        & valid_sales_features
    ].copy()

    if train.empty:
        raise ValueError("Training dataset is empty.")

    if train[FEATURE_COLUMNS].select_dtypes(
        include=["object", "string"]
    ).shape[1] > 0:
        raise ValueError(
            "Feature columns contain non-numeric data."
        )

    print(f"Training rows: {len(train):,}")
    print(f"Number of features: {len(FEATURE_COLUMNS)}")

    return train, first_cutoff


def train_model(train: pd.DataFrame) -> lgb.LGBMRegressor:
    """Train the same LightGBM configuration as Sprint 08."""

    print("Training LightGBM...")

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
        train[FEATURE_COLUMNS],
        train["sales"],
    )

    return model


def calculate_importance(
    model: lgb.LGBMRegressor,
) -> pd.DataFrame:
    """Extract Gain and Split importance from LightGBM."""

    booster = model.booster_

    gain = booster.feature_importance(
        importance_type="gain"
    )

    split = booster.feature_importance(
        importance_type="split"
    )

    result = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "gain": gain,
            "split": split,
        }
    )

    total_gain = result["gain"].sum()
    total_split = result["split"].sum()

    result["gain_pct"] = (
        100 * result["gain"] / total_gain
        if total_gain > 0
        else 0.0
    )

    result["split_pct"] = (
        100 * result["split"] / total_split
        if total_split > 0
        else 0.0
    )

    result["gain_rank"] = (
        result["gain"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    result["split_rank"] = (
        result["split"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    result = result.sort_values(
        "gain",
        ascending=False,
    ).reset_index(drop=True)

    return result


def save_plot(
    importance: pd.DataFrame,
    column: str,
    title: str,
    filename: str,
) -> None:
    """Save a high-resolution horizontal bar chart."""

    ordered = importance.sort_values(
        column,
        ascending=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    bars = ax.barh(
        ordered["feature"],
        ordered[column],
        color="#2878B5",
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    ax.set_xlabel(
        "Share of total importance (%)",
        fontsize=11,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    max_value = ordered[column].max()

    if max_value > 0:
        ax.set_xlim(0, max_value * 1.18)

    for bar, value in zip(
        bars,
        ordered[column],
    ):
        ax.text(
            bar.get_width() + max_value * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()

    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved chart: {path}")


def run_feature_importance() -> pd.DataFrame:
    """Run the complete feature importance analysis."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, cutoff = prepare_data()

    model = train_model(train)

    importance = calculate_importance(model)

    csv_path = (
        OUTPUT_DIR / "feature_importance.csv"
    )

    importance.to_csv(
        csv_path,
        index=False,
    )

    save_plot(
        importance=importance,
        column="gain_pct",
        title=(
            "LightGBM Feature Importance - Gain\n"
            f"Training data through {cutoff.date()}"
        ),
        filename="feature_importance_gain.png",
    )

    save_plot(
        importance=importance,
        column="split_pct",
        title=(
            "LightGBM Feature Importance - Split\n"
            f"Training data through {cutoff.date()}"
        ),
        filename="feature_importance_split.png",
    )

    print("\n=== FEATURE IMPORTANCE: GAIN ===")

    print(
        importance[
            [
                "feature",
                "gain_pct",
                "split_pct",
                "gain_rank",
                "split_rank",
            ]
        ].round(2).to_string(index=False)
    )

    print("\n=== FEATURE GROUP SUMMARY ===")

    def feature_group(feature: str) -> str:
        if feature.startswith("price_"):
            return "Price"

        if feature in {
            "has_event",
            "is_sporting_event",
            "is_national_event",
            "is_religious_event",
            "is_cultural_event",
            "snap_CA",
        }:
            return "Calendar"

        return "Historical sales / date"

    importance["group"] = (
        importance["feature"].map(feature_group)
    )

    group_summary = (
        importance.groupby("group")[
            ["gain_pct", "split_pct"]
        ]
        .sum()
        .sort_values(
            "gain_pct",
            ascending=False,
        )
    )

    print(
        group_summary.round(2).to_string()
    )

    group_path = (
        OUTPUT_DIR / "feature_group_importance.csv"
    )

    group_summary.to_csv(group_path)

    print("\nSaved CSV files:")
    print(csv_path)
    print(group_path)

    print("\nSprint 09.1 completed.")

    return importance


if __name__ == "__main__":
    run_feature_importance()
