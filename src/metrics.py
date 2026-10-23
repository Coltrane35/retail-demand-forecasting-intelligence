
import numpy as np


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean(
        (np.asarray(y_true) - np.asarray(y_pred)) ** 2
    )))


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return float("nan")

    return float(np.sum(np.abs(y_true - y_pred)) / denominator)
