from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import plotly.graph_objs as go

from scripts.utils.prediction_utils import estimate_future_business_dates

logger = logging.getLogger(__name__)


def _compute_robust_y_range(
    historical_values,
    median_values,
    lower_values=None,
    upper_values=None,
) -> list[float] | None:
    historical_np = np.asarray(historical_values if historical_values is not None else [], dtype=float).reshape(-1)
    median_np = np.asarray(median_values if median_values is not None else [], dtype=float).reshape(-1)
    lower_np = np.asarray(lower_values if lower_values is not None else [], dtype=float).reshape(-1)
    upper_np = np.asarray(upper_values if upper_values is not None else [], dtype=float).reshape(-1)

    base_values = np.concatenate([historical_np[np.isfinite(historical_np)], median_np[np.isfinite(median_np)]])
    if base_values.size == 0:
        return None

    base_min = float(np.min(base_values))
    base_max = float(np.max(base_values))
    base_span = max(base_max - base_min, abs(base_max) * 0.05, 1.0)

    lower_finite = lower_np[np.isfinite(lower_np)]
    upper_finite = upper_np[np.isfinite(upper_np)]
    robust_min = base_min
    robust_max = base_max

    if lower_finite.size:
        robust_min = min(robust_min, float(np.percentile(lower_finite, 15)))
    if upper_finite.size:
        robust_max = max(robust_max, float(np.percentile(upper_finite, 85)))

    robust_min = max(robust_min, base_min - 2.5 * base_span)
    robust_max = min(robust_max, base_max + 2.5 * base_span)

    visible_span = max(robust_max - robust_min, base_span * 0.25, 1.0)
    padding = visible_span * 0.08
    return [float(robust_min - padding), float(robust_max + padding)]


def create_base_plot(title, xaxis_title="Fecha", yaxis_title="Precio de cierre", split_date=None):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type="date"),
        legend=dict(itemclick="toggle", itemdoubleclick="toggleothers"),
    )
    if split_date is not None:
        fig.add_shape(type="line", x0=split_date, x1=split_date, y0=0, y1=1, xref="x", yref="paper", line=dict(color="red", width=2, dash="dash"))
        fig.add_annotation(x=split_date, y=1.05, xref="x", yref="paper", text="Inicio de la prediccion", showarrow=False, font=dict(size=12), align="center")
    return fig


def build_stock_plot(
    config,
    ticker_data,
    original_close,
    median,
    lower_bound,
    upper_bound,
    ticker,
    historical_close=None,
    forecast_dates=None,
    historical_period_days=365,
):
    max_prediction_length = config["model"]["max_prediction_length"]
    last_date = pd.Timestamp(ticker_data["Date"].iloc[-1]).tz_localize(None).to_pydatetime()
    historical_dates = ticker_data["Date"].dt.tz_localize(None).tolist()

    cutoff_date = pd.Timestamp(last_date) - pd.Timedelta(days=historical_period_days)
    historical_series = pd.Series(original_close.tolist(), index=pd.to_datetime(historical_dates))
    filtered_history = historical_series[historical_series.index >= cutoff_date]
    filtered_historical_dates = filtered_history.index.tolist()
    filtered_original_close = filtered_history.tolist()

    if historical_close is not None:
        historical_close = historical_close.copy()
        historical_close.index = pd.to_datetime(historical_close.index).tz_localize(None)
        future_actual = historical_close[historical_close.index > pd.Timestamp(last_date)].sort_index().iloc[: len(median)]
        if len(future_actual) != len(median):
            raise ValueError("No hay suficientes observaciones reales para alinear la comparacion historica")
        pred_dates = future_actual.index.tolist()
        combined_dates = filtered_historical_dates + pred_dates
        combined_close = filtered_original_close + future_actual.tolist()
        combined_pred_close = [None] * len(filtered_historical_dates) + list(median)
        plot_data = pd.DataFrame({"Date": combined_dates, "Close": combined_close, "Predicted_Close": combined_pred_close})

        fig = create_base_plot(f"Comparacion historica para {ticker}", split_date=pd.Timestamp(last_date).isoformat())
        fig.add_trace(go.Scatter(x=plot_data["Date"], y=plot_data["Close"], mode="lines", name="Cierre real", line=dict(color="#0B5FFF")))
        fig.add_trace(go.Scatter(x=plot_data["Date"], y=plot_data["Predicted_Close"], mode="lines", name="Cierre predicho", line=dict(color="#FF8C00", dash="dash")))
        y_range = _compute_robust_y_range(filtered_original_close, median)
        if y_range is not None:
            fig.update_yaxes(range=y_range)
        return fig, None

    if forecast_dates is not None:
        pred_dates = [pd.Timestamp(value).tz_localize(None) for value in forecast_dates]
    else:
        pred_dates = list(estimate_future_business_dates(last_date, len(median)))
    if len(pred_dates) != len(median):
        raise ValueError("Las fechas del forecast no coinciden con el horizonte previsto")
    combined_dates = filtered_historical_dates + list(pred_dates)
    combined_close = filtered_original_close + list(median)
    combined_lower_bound = [None] * len(filtered_historical_dates) + list(lower_bound)
    combined_upper_bound = [None] * len(filtered_historical_dates) + list(upper_bound)
    if not (len(combined_dates) == len(combined_close) == len(combined_lower_bound) == len(combined_upper_bound)):
        raise ValueError("Las series historicas y previstas no estan alineadas")

    plot_data = pd.DataFrame(
        {
            "Date": combined_dates,
            "Close": combined_close,
            "Lower_Bound": combined_lower_bound,
            "Upper_Bound": combined_upper_bound,
        }
    )
    fig = create_base_plot(f"Prediccion futura para {ticker}", split_date=pd.Timestamp(last_date).isoformat())
    fig.add_trace(go.Scatter(x=plot_data["Date"], y=plot_data["Close"], mode="lines", name="Cierre real y predicho", line=dict(color="#0B5FFF")))
    fig.add_trace(go.Scatter(x=plot_data["Date"], y=plot_data["Upper_Bound"], mode="lines", name="Cuantil superior (90%)", line=dict(color="rgba(11,95,255,0.35)", dash="dash")))
    fig.add_trace(go.Scatter(x=plot_data["Date"], y=plot_data["Lower_Bound"], mode="lines", name="Cuantil inferior (10%)", line=dict(color="rgba(11,95,255,0.35)", dash="dash"), fill="tonexty", fillcolor="rgba(11,95,255,0.1)"))
    y_range = _compute_robust_y_range(filtered_original_close, median, lower_bound, upper_bound)
    if y_range is not None:
        fig.update_yaxes(range=y_range)

    pred_df = pd.DataFrame(
        {
            "Fecha": list(pred_dates)[:max_prediction_length],
            "Precio previsto": list(median)[:max_prediction_length],
            "Cuantil inferior (10%)": list(lower_bound)[:max_prediction_length],
            "Cuantil superior (90%)": list(upper_bound)[:max_prediction_length],
        }
    )
    return fig, pred_df
