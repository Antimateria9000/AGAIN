from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
Z_SCORE_10_90 = 1.2815515655446004


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
    """
    Construye una trayectoria mediana compuesta y una banda multi-step aproximada.

    La mediana se compone paso a paso en precio. La banda inferior/superior se
    aproxima en espacio de log-precio acumulando la incertidumbre marginal como
    una envolvente probabilistica multi-step. Esto evita el error de encadenar
    cuantiles marginales q10/q90 como si fueran percentiles pathwise coherentes.
    """
    median_returns_np = to_numpy(median_returns).astype(float).reshape(-1)
    lower_returns_np = to_numpy(lower_returns if lower_returns is not None else median_returns).astype(float).reshape(-1)
    upper_returns_np = to_numpy(upper_returns if upper_returns is not None else median_returns).astype(float).reshape(-1)

    if not (median_returns_np.shape == lower_returns_np.shape == upper_returns_np.shape):
        raise ValueError("Los cuantiles de retorno deben tener la misma forma")
    if median_returns_np.ndim != 1:
        raise ValueError("Los cuantiles de retorno deben ser vectores 1D")
    if not np.all(np.isfinite(median_returns_np)) or not np.all(np.isfinite(lower_returns_np)) or not np.all(np.isfinite(upper_returns_np)):
        raise ValueError("Los cuantiles de retorno contienen NaN o Inf")

    lower_ordered = np.minimum(np.minimum(lower_returns_np, median_returns_np), upper_returns_np)
    upper_ordered = np.maximum(np.maximum(lower_returns_np, median_returns_np), upper_returns_np)
    median_ordered = np.clip(median_returns_np, lower_ordered, upper_ordered)
    if not (
        np.allclose(lower_ordered, lower_returns_np)
        and np.allclose(median_ordered, median_returns_np)
        and np.allclose(upper_ordered, upper_returns_np)
    ):
        logger.warning("Se han detectado cruces cuantílicos; se reordena lower/median/upper localmente.")

    safe_lower = np.clip(lower_ordered, -0.999999, None)
    safe_median = np.clip(median_ordered, -0.999999, None)
    safe_upper = np.clip(upper_ordered, -0.999999, None)
    if not (
        np.allclose(safe_lower, lower_ordered)
        and np.allclose(safe_median, median_ordered)
        and np.allclose(safe_upper, upper_ordered)
    ):
        logger.warning("Se han recortado retornos <= -1 para mantener log1p finito en la acumulacion.")

    g_med = np.log1p(safe_median)
    g_low = np.log1p(safe_lower)
    g_up = np.log1p(safe_upper)

    sigma_low = np.maximum((g_med - g_low) / Z_SCORE_10_90, 0.0)
    sigma_up = np.maximum((g_up - g_med) / Z_SCORE_10_90, 0.0)
    sigma_t = np.maximum(0.5 * (sigma_low + sigma_up), 0.0)

    cum_mu = np.cumsum(g_med)
    cum_sigma = np.sqrt(np.cumsum(np.square(sigma_t)))
    base_log_price = float(np.log(max(float(last_close_price), 1e-12)))

    median = np.exp(base_log_price + cum_mu)
    lower = np.exp(base_log_price + cum_mu - Z_SCORE_10_90 * cum_sigma)
    upper = np.exp(base_log_price + cum_mu + Z_SCORE_10_90 * cum_sigma)

    median = np.asarray(median, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower = np.minimum(lower, median)
    upper = np.maximum(upper, median)

    if not (np.all(np.isfinite(median)) and np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
        raise ValueError("La trayectoria de precios acumulada contiene NaN o Inf")
    if np.any(median <= 0.0) or np.any(lower <= 0.0) or np.any(upper <= 0.0):
        raise ValueError("La trayectoria de precios acumulada contiene valores no positivos")

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
