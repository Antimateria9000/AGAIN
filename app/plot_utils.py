import logging

import pandas as pd
import plotly.graph_objs as go
import streamlit as st

from scripts.utils.prediction_utils import estimate_future_business_dates

logger = logging.getLogger(__name__)


def create_base_plot(title, xaxis_title='Fecha', yaxis_title='Precio de cierre', split_date=None):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        showlegend=True,
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'),
    )
    if split_date is not None:
        fig.add_shape(type='line', x0=split_date, x1=split_date, y0=0, y1=1, xref='x', yref='paper', line=dict(color='red', width=2, dash='dash'))
        fig.add_annotation(x=split_date, y=1.05, xref='x', yref='paper', text='Inicio de la prediccion', showarrow=False, font=dict(size=12), align='center')
    return fig


def create_stock_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker, historical_close=None, historical_period_days=365):
    max_prediction_length = config['model']['max_prediction_length']
    last_date = pd.Timestamp(ticker_data['Date'].iloc[-1]).tz_localize(None).to_pydatetime()
    historical_dates = ticker_data['Date'].dt.tz_localize(None).tolist()

    cutoff_date = pd.Timestamp(last_date) - pd.Timedelta(days=historical_period_days)
    historical_series = pd.Series(original_close.tolist(), index=pd.to_datetime(historical_dates))
    filtered_history = historical_series[historical_series.index >= cutoff_date]
    filtered_historical_dates = filtered_history.index.tolist()
    filtered_original_close = filtered_history.tolist()

    if historical_close is not None:
        historical_close = historical_close.copy()
        historical_close.index = pd.to_datetime(historical_close.index).tz_localize(None)
        future_actual = historical_close[historical_close.index > pd.Timestamp(last_date)].sort_index().iloc[:len(median)]
        if len(future_actual) != len(median):
            raise ValueError('No hay suficientes observaciones reales para alinear la comparacion historica')
        pred_dates = future_actual.index.tolist()
        combined_dates = filtered_historical_dates + pred_dates
        combined_close = filtered_original_close + future_actual.tolist()
        combined_pred_close = [None] * len(filtered_historical_dates) + list(median)
        plot_data = pd.DataFrame({'Date': combined_dates, 'Close': combined_close, 'Predicted_Close': combined_pred_close})

        fig = create_base_plot(f'Comparacion historica para {ticker}', split_date=pd.Timestamp(last_date).isoformat())
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], mode='lines', name='Cierre real', line=dict(color='#0B5FFF')))
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Predicted_Close'], mode='lines', name='Cierre predicho', line=dict(color='#FF8C00', dash='dash')))
    else:
        pred_dates = estimate_future_business_dates(last_date, len(median))
        combined_dates = filtered_historical_dates + list(pred_dates)
        combined_close = filtered_original_close + list(median)
        combined_lower_bound = [None] * len(filtered_historical_dates) + list(lower_bound)
        combined_upper_bound = [None] * len(filtered_historical_dates) + list(upper_bound)
        if not (len(combined_dates) == len(combined_close) == len(combined_lower_bound) == len(combined_upper_bound)):
            raise ValueError('Las series historicas y previstas no estan alineadas')
        plot_data = pd.DataFrame({'Date': combined_dates, 'Close': combined_close, 'Lower_Bound': combined_lower_bound, 'Upper_Bound': combined_upper_bound})

        fig = create_base_plot(f'Prediccion futura para {ticker}', split_date=pd.Timestamp(last_date).isoformat())
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], mode='lines', name='Cierre real y predicho', line=dict(color='#0B5FFF')))
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Upper_Bound'], mode='lines', name='Cuantil superior (90%)', line=dict(color='rgba(11,95,255,0.35)', dash='dash')))
        fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Lower_Bound'], mode='lines', name='Cuantil inferior (10%)', line=dict(color='rgba(11,95,255,0.35)', dash='dash'), fill='tonexty', fillcolor='rgba(11,95,255,0.1)'))

    st.plotly_chart(fig, use_container_width=True)

    if historical_close is None:
        pred_df = pd.DataFrame({
            'Fecha': list(pred_dates)[:max_prediction_length],
            'Precio previsto': list(median)[:max_prediction_length],
            'Cuantil inferior (10%)': list(lower_bound)[:max_prediction_length],
            'Cuantil superior (90%)': list(upper_bound)[:max_prediction_length],
        })
        st.subheader('Precios previstos')
        st.dataframe(pred_df.style.format({'Fecha': '{:%Y-%m-%d}', 'Precio previsto': '{:.2f}', 'Cuantil inferior (10%)': '{:.2f}', 'Cuantil superior (90%)': '{:.2f}'}))
