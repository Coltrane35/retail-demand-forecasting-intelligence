
import numpy as np
import pandas as pd


HORIZON = 28


def seasonal_naive(train: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Repeat the last observed 7-day pattern."""
    if len(train) < 7:
        raise ValueError("At least 7 observations required.")

    pattern = train[-7:]
    return np.resize(pattern, horizon)


def moving_average(train: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Forecast using the mean of the last 28 observations."""
    if len(train) < 28:
        raise ValueError("At least 28 observations required.")

    mean_value = np.mean(train[-28:])
    return np.full(horizon, mean_value)


def seasonal_average(train: np.ndarray, horizon: int = HORIZON) -> np.ndarray:
    """Average the last 4 weekly observations for each weekday."""
    if len(train) < 28:
        raise ValueError("At least 28 observations required.")

    last_28 = train[-28:]
    weekday_pattern = last_28.reshape(4, 7).mean(axis=0)

    return np.resize(weekday_pattern, horizon)


def forecast_baselines(
    series: pd.Series,
    horizon: int = HORIZON,
) -> dict[str, np.ndarray]:
    """Generate baseline forecasts using historical data only."""
    train = series.to_numpy(dtype=float)

    return {
        "seasonal_naive": seasonal_naive(train, horizon),
        "moving_average": moving_average(train, horizon),
        "seasonal_average": seasonal_average(train, horizon),
    }
