
"""
Sprint 09.2 - Permutation Importance

Evaluate feature importance on a separate
28-day historical validation window.

Metrics:
- WAPE
- MAE
- RMSE

Outputs:
- permutation_importance.csv
- permutation_importance_raw.csv
- permutation_baseline_metrics.csv
- permutation_importance_wape.png
- permutation_importance_mae.png
- permutation_importance_rmse.png

Final rolling backtesting periods are excluded
from feature selection and importance evaluation.
"""

from pathlib import Path

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.ml_forecasting import (
    DATA_PATH,
    HORIZON,
    FEATURES,
    FEATURES_WITH_CALENDAR_AND_PRICE,
    create_features,
    add_calendar_features,
)
from src.price_features import add_price_features
from src.metrics import mae, rmse, wape


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    ROOT / "outputs" / "permutation_importance"
)

TOP_N = 100
N_FOLDS = 4
N_REPEATS = 10
RANDOM_STATE = 42

FEATURE_COLUMNS = FEATURES_WITH_CALENDAR_AND_PRICE


def prepare_data():
    """Prepare leakage-safe train and validation sets."""

    print("Loading M5 sales data...")

    df = pd.read_parquet(DATA_PATH)

    df["date"] = pd.to_datetime(df["date"])

    max_date = df["date"].max()

    validation_end = (
        max_date
        - pd.Timedelta(days=HORIZON * N_FOLDS)
    )

    validation_start = (
        validation_end
        - pd.Timedelta(days=HORIZON - 1)
    )

    train_cutoff = (
        validation_start
        - pd.Timedelta(days=1)
    )

    print(f"Train cutoff: {train_cutoff.date()}")

    print(
        f"Validation: {validation_start.date()} "
        f"to {validation_end.date()}"
    )

    # Exclude all final backtesting observations.
    df = df[
        df["date"] <= validation_end
    ].copy()

    # Select products using training data only.
    top_items = (
        df.loc[df["date"] <= train_cutoff]
        .groupby("item_id")["sales"]
        .sum()
        .nlargest(TOP_N)
        .index
    )

    if len(top_items) != TOP_N:
        raise ValueError(
            "Unexpected number of selected products."
        )

    df = df[
        df["item_id"].isin(top_items)
    ].copy()

    print("Creating historical sales features...")
    df = create_features(df)

    print("Adding calendar features...")
    df = add_calendar_features(df)

    print("Adding historical price features...")
    df = add_price_features(df)

    valid_features = (
        df[FEATURES].notna().all(axis=1)
    )

    train = df[
        (df["date"] <= train_cutoff)
        & valid_features
    ].copy()

    validation = df[
        (df["date"] >= validation_start)
        & (df["date"] <= validation_end)
        & valid_features
    ].copy()

    expected_rows = TOP_N * HORIZON

    if len(validation) != expected_rows:
        raise ValueError(
            f"Unexpected validation size: "
            f"{len(validation)} "
            f"(expected {expected_rows})"
        )

    if train.empty:
        raise ValueError("Training data is empty.")

    non_price_features = [
        feature
        for feature in FEATURE_COLUMNS
        if not feature.startswith("price_")
    ]

    if train[non_price_features].isna().any().any():
        raise ValueError(
            "Missing non-price training features."
        )

    if validation[
        non_price_features
    ].isna().any().any():
        raise ValueError(
            "Missing non-price validation features."
        )

    print(f"Training rows: {len(train):,}")

    print(
        f"Validation rows: {len(validation):,}"
    )

    print(
        f"Number of features: {len(FEATURE_COLUMNS)}"
    )

    print(
        f"Products: {validation['item_id'].nunique()}"
    )

    return train, validation


def train_model(
    train: pd.DataFrame,
) -> lgb.LGBMRegressor:
    """Train the existing LightGBM configuration."""

    print("\nTraining LightGBM...")

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        train[FEATURE_COLUMNS],
        train["sales"],
    )

    return model


def evaluate_model(
    model: lgb.LGBMRegressor,
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict:
    """Evaluate predictions using project metrics."""

    predictions = np.maximum(
        model.predict(X),
        0,
    )

    return {
        "mae": mae(y, predictions),
        "rmse": rmse(y, predictions),
        "wape": wape(y, predictions),
    }


def feature_group(feature: str) -> str:
    """Assign a feature to its original group."""

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


def calculate_permutation_importance(
    model: lgb.LGBMRegressor,
    validation: pd.DataFrame,
):
    """
    Shuffle one validation feature at a time.

    Positive metric changes indicate that
    permutation worsened forecast accuracy.
    """

    X = validation[
        FEATURE_COLUMNS
    ].copy()

    y = validation["sales"].to_numpy(
        dtype=float
    )

    baseline = evaluate_model(
        model,
        X,
        y,
    )

    print("\n=== BASELINE VALIDATION METRICS ===")

    print(f"MAE:  {baseline['mae']:.4f}")
    print(f"RMSE: {baseline['rmse']:.4f}")
    print(f"WAPE: {baseline['wape']:.4%}")

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    # Use the same permutation indices for
    # each feature to improve comparability.
    permutations = [
        rng.permutation(len(X))
        for _ in range(N_REPEATS)
    ]

    records = []

    total_features = len(FEATURE_COLUMNS)

    for index, feature in enumerate(
        FEATURE_COLUMNS,
        start=1,
    ):

        print(
            f"[{index}/{total_features}] "
            f"Permuting {feature}..."
        )

        original_values = (
            X[feature].to_numpy(copy=True)
        )

        for repeat, permutation in enumerate(
            permutations,
            start=1,
        ):

            X_permuted = X.copy()

            X_permuted[feature] = (
                original_values[permutation]
            )

            scores = evaluate_model(
                model,
                X_permuted,
                y,
            )

            records.append(
                {
                    "feature": feature,
                    "group": feature_group(feature),
                    "repeat": repeat,
                    "permuted_mae": scores["mae"],
                    "permuted_rmse": scores["rmse"],
                    "permuted_wape": scores["wape"],
                    "delta_mae": (
                        scores["mae"]
                        - baseline["mae"]
                    ),
                    "delta_rmse": (
                        scores["rmse"]
                        - baseline["rmse"]
                    ),
                    "delta_wape_pp": (
                        scores["wape"]
                        - baseline["wape"]
                    ) * 100,
                }
            )

    raw_results = pd.DataFrame(records)

    importance = (
        raw_results.groupby(
            ["feature", "group"],
            as_index=False,
        )
        .agg(
            delta_mae_mean=("delta_mae", "mean"),
            delta_mae_std=("delta_mae", "std"),
            delta_rmse_mean=("delta_rmse", "mean"),
            delta_rmse_std=("delta_rmse", "std"),
            delta_wape_pp_mean=(
                "delta_wape_pp",
                "mean",
            ),
            delta_wape_pp_std=(
                "delta_wape_pp",
                "std",
            ),
        )
    )

    importance["wape_rank"] = (
        importance["delta_wape_pp_mean"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    importance = importance.sort_values(
        "delta_wape_pp_mean",
        ascending=False,
    ).reset_index(drop=True)

    return importance, raw_results, baseline


def save_plot(
    importance: pd.DataFrame,
    mean_column: str,
    std_column: str,
    title: str,
    xlabel: str,
    filename: str,
) -> None:
    """Generate permutation importance chart."""

    ordered = importance.sort_values(
        mean_column,
        ascending=True,
    )

    colors = [
        "#2878B5" if value >= 0 else "#D97706"
        for value in ordered[mean_column]
    ]

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    ax.barh(
        ordered["feature"],
        ordered[mean_column],
        xerr=ordered[std_column],
        color=colors,
        alpha=0.85,
        capsize=3,
    )

    ax.axvline(
        0,
        color="black",
        linewidth=1,
    )

    ax.set_title(
        title,
        fontsize=16,
        fontweight="bold",
        pad=16,
    )

    ax.set_xlabel(
        xlabel,
        fontsize=11,
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    fig.tight_layout()

    path = OUTPUT_DIR / filename

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved chart: {path}")


def run_permutation_importance():
    """Run the complete permutation experiment."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train, validation = prepare_data()

    model = train_model(train)

    importance, raw_results, baseline = (
        calculate_permutation_importance(
            model,
            validation,
        )
    )

    importance.to_csv(
        OUTPUT_DIR / "permutation_importance.csv",
        index=False,
    )

    raw_results.to_csv(
        OUTPUT_DIR / "permutation_importance_raw.csv",
        index=False,
    )

    pd.DataFrame([baseline]).to_csv(
        OUTPUT_DIR / "permutation_baseline_metrics.csv",
        index=False,
    )

    save_plot(
        importance=importance,
        mean_column="delta_wape_pp_mean",
        std_column="delta_wape_pp_std",
        title="Permutation Importance - WAPE",
        xlabel="Increase in WAPE (percentage points)",
        filename="permutation_importance_wape.png",
    )

    save_plot(
        importance=importance,
        mean_column="delta_mae_mean",
        std_column="delta_mae_std",
        title="Permutation Importance - MAE",
        xlabel="Increase in MAE",
        filename="permutation_importance_mae.png",
    )

    save_plot(
        importance=importance,
        mean_column="delta_rmse_mean",
        std_column="delta_rmse_std",
        title="Permutation Importance - RMSE",
        xlabel="Increase in RMSE",
        filename="permutation_importance_rmse.png",
    )

    print("\n=== PERMUTATION IMPORTANCE RANKING ===")

    print(
        importance[
            [
                "feature",
                "delta_wape_pp_mean",
                "delta_wape_pp_std",
                "delta_mae_mean",
                "delta_rmse_mean",
                "wape_rank",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== EXPERIMENT SUMMARY ===")

    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Repeats per feature: {N_REPEATS}")

    print(
        f"Total permutations: {len(raw_results)}"
    )

    print(
        f"Baseline WAPE: {baseline['wape']:.4%}"
    )

    print(
        f"Most important feature: "
        f"{importance.iloc[0]['feature']}"
    )

    print(f"\nOutput directory: {OUTPUT_DIR}")

    print("\nSprint 09.2 completed.")

    return importance


if __name__ == "__main__":
    run_permutation_importance()
