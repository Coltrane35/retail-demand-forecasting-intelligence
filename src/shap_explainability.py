
"""
Sprint 09.3 - SHAP Explainability.

Explain LightGBM forecasts using TreeSHAP on a separate
historical validation window.

The experiment reuses the data preparation and model
configuration from Sprint 09.2 - Permutation Importance.

SHAP values explain raw model predictions before clipping
negative forecasts to zero.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.permutation_importance import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    prepare_data,
    train_model,
)
from src.metrics import mae, rmse, wape


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "shap_explainability"

TOP_DEPENDENCE_FEATURES = 4
PLOT_DPI = 300


def calculate_shap_values(model, validation):
    """Calculate SHAP values and verify additivity."""

    print("\nCalculating SHAP values...")

    X = validation[FEATURE_COLUMNS].copy()

    explainer = shap.TreeExplainer(model)

    explanation = explainer(X)

    raw_predictions = np.asarray(
        model.predict(X),
        dtype=float,
    )

    shap_values = np.asarray(
        explanation.values,
        dtype=float,
    )

    base_values = np.asarray(
        explanation.base_values,
        dtype=float,
    )

    if base_values.size == 1:
        base_values = np.full(
            len(X),
            base_values.item(),
        )
    else:
        base_values = base_values.reshape(-1)

    expected_shape = (
        len(X),
        len(FEATURE_COLUMNS),
    )

    if shap_values.shape != expected_shape:
        raise ValueError(
            f"Unexpected SHAP shape: "
            f"{shap_values.shape}. "
            f"Expected: {expected_shape}"
        )

    if not np.isfinite(shap_values).all():
        raise ValueError(
            "Non-finite SHAP values detected."
        )

    reconstructed_predictions = (
        base_values + shap_values.sum(axis=1)
    )

    additivity_error = np.abs(
        raw_predictions
        - reconstructed_predictions
    )

    np.testing.assert_allclose(
        raw_predictions,
        reconstructed_predictions,
        rtol=1e-5,
        atol=1e-6,
    )

    print("\n=== SHAP ADDITIVITY CHECK ===")

    print(
        f"SHAP values shape: "
        f"{shap_values.shape}"
    )

    print(
        f"Maximum additivity error: "
        f"{additivity_error.max():.12e}"
    )

    print(
        f"Mean additivity error: "
        f"{additivity_error.mean():.12e}"
    )

    print("SHAP additivity check: PASS")

    return (
        X,
        explanation,
        raw_predictions,
        base_values,
        additivity_error,
    )


def feature_group(feature):
    """Assign a feature to its analytical group."""

    if feature.startswith("price_"):
        return "Price"

    calendar_features = {
        "has_event",
        "is_sporting_event",
        "is_national_event",
        "is_religious_event",
        "is_cultural_event",
        "snap_CA",
    }

    if feature in calendar_features:
        return "Calendar"

    return "Historical sales / date"


def calculate_global_importance(explanation):
    """Calculate global SHAP feature importance."""

    values = np.asarray(explanation.values)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "mean_abs_shap": np.abs(
                values
            ).mean(axis=0),
            "mean_shap": values.mean(axis=0),
            "std_shap": values.std(axis=0),
            "mean_positive_shap": np.maximum(
                values, 0
            ).mean(axis=0),
            "mean_negative_shap": np.minimum(
                values, 0
            ).mean(axis=0),
        }
    )

    importance["group"] = (
        importance["feature"].map(feature_group)
    )

    total_importance = (
        importance["mean_abs_shap"].sum()
    )

    if total_importance <= 0:
        raise ValueError(
            "Total SHAP importance is zero."
        )

    importance["importance_pct"] = (
        importance["mean_abs_shap"]
        / total_importance
        * 100
    )

    importance["shap_rank"] = (
        importance["mean_abs_shap"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    importance = (
        importance.sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return importance


def calculate_group_importance(importance):
    """Aggregate absolute SHAP importance by group."""

    group_importance = (
        importance.groupby(
            "group",
            as_index=False,
        )
        .agg(
            feature_count=("feature", "size"),
            mean_abs_shap_sum=(
                "mean_abs_shap",
                "sum",
            ),
            importance_pct=(
                "importance_pct",
                "sum",
            ),
        )
        .sort_values(
            "importance_pct",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return group_importance


def save_shap_data(
    validation,
    explanation,
    raw_predictions,
    base_values,
    additivity_error,
):
    """Save observation-level SHAP values."""

    metadata = validation[
        ["item_id", "date", "sales"]
    ].reset_index(drop=True).copy()

    metadata = metadata.rename(
        columns={"sales": "actual_sales"}
    )

    metadata["raw_prediction"] = raw_predictions

    metadata["clipped_prediction"] = np.maximum(
        raw_predictions,
        0,
    )

    metadata["base_value"] = base_values

    metadata["additivity_error"] = (
        additivity_error
    )

    shap_data = pd.DataFrame(
        explanation.values,
        columns=[
            f"shap_{feature}"
            for feature in FEATURE_COLUMNS
        ],
    )

    result = pd.concat(
        [metadata, shap_data],
        axis=1,
    )

    output_path = (
        OUTPUT_DIR / "shap_values.csv"
    )

    result.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")

    return result


def save_figure(filename):
    """Save the active Matplotlib figure."""

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=PLOT_DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close("all")

    print(f"Saved chart: {output_path}")


def plot_beeswarm(explanation):
    """Generate SHAP beeswarm summary plot."""

    np.random.seed(RANDOM_STATE)

    shap.plots.beeswarm(
        explanation,
        max_display=len(FEATURE_COLUMNS),
        show=False,
        plot_size=(12, 10),
    )

    plt.title(
        "SHAP Summary - LightGBM Forecasting",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )

    save_figure(
        "shap_summary_beeswarm.png"
    )


def plot_global_importance(importance):
    """Generate global SHAP importance bar chart."""

    ordered = importance.sort_values(
        "mean_abs_shap",
        ascending=True,
    )

    group_colors = {
        "Historical sales / date": "#2878B5",
        "Price": "#D97706",
        "Calendar": "#159A80",
    }

    colors = ordered["group"].map(
        group_colors
    )

    fig, ax = plt.subplots(
        figsize=(12, 10)
    )

    bars = ax.barh(
        ordered["feature"],
        ordered["mean_abs_shap"],
        color=colors,
        alpha=0.9,
    )

    max_value = ordered[
        "mean_abs_shap"
    ].max()

    for bar, percentage in zip(
        bars,
        ordered["importance_pct"],
    ):
        ax.text(
            bar.get_width()
            + max_value * 0.015,
            bar.get_y()
            + bar.get_height() / 2,
            f"{percentage:.2f}%",
            va="center",
            fontsize=9,
        )

    ax.set_xlim(
        0,
        max_value * 1.20,
    )

    ax.set_title(
        "Global SHAP Feature Importance",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )

    ax.set_xlabel(
        "Mean absolute SHAP value "
        "(sales units)"
    )

    ax.grid(
        axis="x",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    from matplotlib.patches import Patch

    handles = [
        Patch(
            color=color,
            label=group,
        )
        for group, color
        in group_colors.items()
    ]

    ax.legend(
        handles=handles,
        loc="lower right",
    )

    fig.tight_layout()

    save_figure(
        "shap_feature_importance.png"
    )


def plot_dependence(
    explanation,
    importance,
):
    """Generate plots for the most important features."""

    top_features = (
        importance["feature"]
        .head(TOP_DEPENDENCE_FEATURES)
        .tolist()
    )

    for feature in top_features:

        shap.plots.scatter(
            explanation[:, feature],
            show=False,
        )

        plt.gcf().set_size_inches(
            10,
            6,
        )

        plt.title(
            f"SHAP Dependence - {feature}",
            fontsize=14,
            fontweight="bold",
            pad=16,
        )

        plt.grid(
            alpha=0.20,
            linestyle="--",
        )

        save_figure(
            f"shap_dependence_{feature}.png"
        )


def plot_waterfall(
    validation,
    explanation,
    raw_predictions,
):
    """Explain a representative individual forecast."""

    median_prediction = np.median(
        raw_predictions
    )

    row_index = int(
        np.argmin(
            np.abs(
                raw_predictions
                - median_prediction
            )
        )
    )

    observation = validation.iloc[
        row_index
    ]

    item_id = observation["item_id"]

    date = pd.Timestamp(
        observation["date"]
    ).strftime("%Y-%m-%d")

    actual = float(
        observation["sales"]
    )

    prediction = float(
        raw_predictions[row_index]
    )

    print("\n=== WATERFALL EXAMPLE ===")

    print(f"Product: {item_id}")
    print(f"Date: {date}")
    print(f"Actual sales: {actual:.2f}")
    print(f"Raw prediction: {prediction:.4f}")

    shap.plots.waterfall(
        explanation[row_index],
        max_display=15,
        show=False,
    )

    plt.gcf().set_size_inches(
        12,
        8,
    )

    plt.title(
        f"Forecast Explanation: {item_id}\n"
        f"Date: {date} | "
        f"Actual: {actual:.0f} | "
        f"Forecast: {prediction:.2f}",
        fontsize=12,
        pad=25,
    )

    save_figure(
        "shap_waterfall_example.png"
    )

    example = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "feature_value": (
                explanation.data[row_index]
            ),
            "shap_value": (
                explanation.values[row_index]
            ),
        }
    )

    example["item_id"] = item_id
    example["date"] = date
    example["actual_sales"] = actual
    example["raw_prediction"] = prediction

    example["base_value"] = float(
        np.asarray(
            explanation[row_index].base_values
        ).item()
    )

    example["abs_shap_value"] = np.abs(
        example["shap_value"]
    )

    example = example.sort_values(
        "abs_shap_value",
        ascending=False,
    )

    example.to_csv(
        OUTPUT_DIR
        / "shap_waterfall_example.csv",
        index=False,
    )


def compare_importance_methods(importance):
    """Compare SHAP with previous importance methods."""

    comparison = importance[
        [
            "feature",
            "group",
            "mean_abs_shap",
            "importance_pct",
            "shap_rank",
        ]
    ].copy()

    gain_path = (
        ROOT
        / "outputs"
        / "feature_importance"
        / "feature_importance.csv"
    )

    permutation_path = (
        ROOT
        / "outputs"
        / "permutation_importance"
        / "permutation_importance.csv"
    )

    if gain_path.exists():

        gain = pd.read_csv(gain_path)

        required = {
            "feature",
            "gain_rank",
            "split_rank",
        }

        if required.issubset(gain.columns):

            comparison = comparison.merge(
                gain[
                    [
                        "feature",
                        "gain_rank",
                        "split_rank",
                    ]
                ],
                on="feature",
                how="left",
                validate="one_to_one",
            )

    else:
        print(
            "Gain importance file not found."
        )

    if permutation_path.exists():

        permutation = pd.read_csv(
            permutation_path
        )

        required = {
            "feature",
            "wape_rank",
            "delta_wape_pp_mean",
        }

        if required.issubset(
            permutation.columns
        ):

            permutation = permutation[
                [
                    "feature",
                    "wape_rank",
                    "delta_wape_pp_mean",
                ]
            ].rename(
                columns={
                    "wape_rank":
                    "permutation_rank"
                }
            )

            comparison = comparison.merge(
                permutation,
                on="feature",
                how="left",
                validate="one_to_one",
            )

    else:
        print(
            "Permutation importance file "
            "not found."
        )

    comparison = comparison.sort_values(
        "shap_rank"
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "importance_methods_comparison.csv",
        index=False,
    )

    return comparison


def save_experiment_summary(
    train,
    validation,
    raw_predictions,
    additivity_error,
    importance,
):
    """Save model and SHAP experiment metrics."""

    actual = validation[
        "sales"
    ].to_numpy(dtype=float)

    predictions = np.maximum(
        raw_predictions,
        0,
    )

    summary = {
        "model": "lightgbm_calendar_price",
        "shap_version": shap.__version__,
        "train_cutoff": str(
            pd.Timestamp(
                train["date"].max()
            ).date()
        ),
        "validation_start": str(
            pd.Timestamp(
                validation["date"].min()
            ).date()
        ),
        "validation_end": str(
            pd.Timestamp(
                validation["date"].max()
            ).date()
        ),
        "training_rows": len(train),
        "validation_rows": len(validation),
        "products": validation[
            "item_id"
        ].nunique(),
        "features": len(FEATURE_COLUMNS),
        "shap_values_count": (
            len(validation)
            * len(FEATURE_COLUMNS)
        ),
        "mae": mae(actual, predictions),
        "rmse": rmse(actual, predictions),
        "wape": wape(actual, predictions),
        "negative_raw_predictions": int(
            (raw_predictions < 0).sum()
        ),
        "max_additivity_error": (
            additivity_error.max()
        ),
        "mean_additivity_error": (
            additivity_error.mean()
        ),
        "most_important_feature": (
            importance.iloc[0]["feature"]
        ),
    }

    pd.DataFrame([summary]).to_csv(
        OUTPUT_DIR
        / "shap_experiment_summary.csv",
        index=False,
    )

    return summary


def run_shap_explainability():
    """Run the complete SHAP experiment."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("SPRINT 09.3 - SHAP EXPLAINABILITY")
    print("=" * 65)

    print(f"SHAP version: {shap.__version__}")

    train, validation = prepare_data()

    model = train_model(train)

    (
        X,
        explanation,
        raw_predictions,
        base_values,
        additivity_error,
    ) = calculate_shap_values(
        model,
        validation,
    )

    importance = calculate_global_importance(
        explanation
    )

    group_importance = (
        calculate_group_importance(
            importance
        )
    )

    importance.to_csv(
        OUTPUT_DIR
        / "shap_feature_importance.csv",
        index=False,
    )

    group_importance.to_csv(
        OUTPUT_DIR
        / "shap_group_importance.csv",
        index=False,
    )

    save_shap_data(
        validation,
        explanation,
        raw_predictions,
        base_values,
        additivity_error,
    )

    print("\nGenerating SHAP visualizations...")

    plot_beeswarm(explanation)

    plot_global_importance(importance)

    plot_dependence(
        explanation,
        importance,
    )

    plot_waterfall(
        validation,
        explanation,
        raw_predictions,
    )

    comparison = compare_importance_methods(
        importance
    )

    summary = save_experiment_summary(
        train,
        validation,
        raw_predictions,
        additivity_error,
        importance,
    )

    print("\n=== GLOBAL SHAP IMPORTANCE ===")

    print(
        importance[
            [
                "feature",
                "mean_abs_shap",
                "importance_pct",
                "shap_rank",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\n=== SHAP GROUP IMPORTANCE ===")

    print(
        group_importance
        .round(4)
        .to_string(index=False)
    )

    print("\n=== IMPORTANCE METHODS COMPARISON ===")

    print(
        comparison
        .round(4)
        .to_string(index=False)
    )

    print("\n=== EXPERIMENT SUMMARY ===")

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    print(
        f"Validation rows: "
        f"{len(validation):,}"
    )

    print(
        f"Features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print(
        f"SHAP values calculated: "
        f"{explanation.values.size:,}"
    )

    print(
        f"MAE: "
        f"{summary['mae']:.4f}"
    )

    print(
        f"RMSE: "
        f"{summary['rmse']:.4f}"
    )

    print(
        f"WAPE: "
        f"{summary['wape']:.4%}"
    )

    print(
        f"Most important feature: "
        f"{summary['most_important_feature']}"
    )

    print(
        f"\nOutput directory: "
        f"{OUTPUT_DIR}"
    )

    print("\nSprint 09.3 completed.")

    return importance


if __name__ == "__main__":
    run_shap_explainability()
