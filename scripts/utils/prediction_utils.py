from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def inverse_transform_if_available(normalizer, values):
    if normalizer is None or not hasattr(normalizer, "inverse_transform"):
        return to_numpy(values)
    transformed = normalizer.inverse_transform(values)
    return to_numpy(transformed)


def denormalize_logged_close(close_normalizer, normalized_close_value: float) -> float:
    normalized_close = torch.tensor([[normalized_close_value]], dtype=torch.float32)
    denormalized_close = inverse_transform_if_available(close_normalizer, normalized_close)
    return float(np.expm1(denormalized_close)[0, 0])


def accumulate_price_path(last_close_price: float, relative_returns) -> np.ndarray:
    current_price = float(last_close_price)
    prices = []
    for relative_return in to_numpy(relative_returns).astype(float).reshape(-1):
        current_price = current_price * (1.0 + relative_return)
        prices.append(current_price)
    return np.asarray(prices, dtype=float)


def accumulate_quantile_price_paths(
    last_close_price: float,
    median_returns,
    lower_returns=None,
    upper_returns=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    median = accumulate_price_path(last_close_price, median_returns)
    lower = accumulate_price_path(last_close_price, lower_returns if lower_returns is not None else median_returns)
    upper = accumulate_price_path(last_close_price, upper_returns if upper_returns is not None else median_returns)
    return median, lower, upper


def price_path_to_step_returns(last_close_price: float, price_path) -> np.ndarray:
    prices = to_numpy(price_path).astype(float).reshape(-1)
    if prices.size == 0:
        return np.asarray([], dtype=float)
    previous_prices = np.concatenate(([float(last_close_price)], prices[:-1]))
    previous_prices = np.where(previous_prices == 0.0, 1e-12, previous_prices)
    return (prices / previous_prices) - 1.0


def compute_directional_accuracy(predicted_returns, actual_returns) -> float:
    predicted = to_numpy(predicted_returns).astype(float).reshape(-1)
    actual = to_numpy(actual_returns).astype(float).reshape(-1)
    if predicted.size == 0 or actual.size == 0:
        return 0.0
    horizon = min(predicted.size, actual.size)
    return float(np.mean(np.sign(predicted[:horizon]) == np.sign(actual[:horizon])) * 100.0)


def estimate_future_business_dates(last_date, periods: int) -> pd.DatetimeIndex:
    last_timestamp = pd.Timestamp(last_date).tz_localize(None)
    next_business_day = last_timestamp + pd.offsets.BusinessDay(1)
    return pd.bdate_range(start=next_business_day, periods=periods)
