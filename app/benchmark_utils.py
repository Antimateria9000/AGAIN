import asyncio
import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import streamlit as st
import torch

from scripts.data_fetcher import DataFetcher
from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data
from scripts.runtime_config import ConfigManager
from scripts.utils.prediction_utils import compute_directional_accuracy, price_path_to_step_returns

logger = logging.getLogger(__name__)


async def fetch_ticker_data(ticker, start_date, end_date):
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        fetcher = DataFetcher(config_manager, config['prediction']['years'])
        data = fetcher.fetch_stock_data_sync(ticker, start_date, end_date)
        if data.empty:
            logger.error(f"No hay datos para {ticker}")
            return ticker, None
        data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
        return ticker, data
    except Exception as e:
        logger.error(f"Error al descargar datos de {ticker}: {e}")
        return ticker, None


async def process_ticker(ticker, full_data, config, trim_date, dataset, normalizers, model):
    try:
        if full_data is None or full_data.empty:
            return ticker, None

        full_data = full_data[full_data['Ticker'] == ticker].copy()
        full_data['Date'] = pd.to_datetime(full_data['Date']).dt.tz_localize(None)
        full_data = full_data.sort_values('Date').set_index('Date')
        historical_close = full_data['Close']

        new_data = full_data[full_data.index <= trim_date].copy().reset_index()
        if new_data.empty:
            logger.error(f"No hay datos previos al recorte para {ticker}")
            return ticker, None

        ticker_data, original_close = preprocess_data(config, new_data, ticker, normalizers, historical_mode=True)
        with torch.no_grad():
            median, _, _, details = generate_predictions(config, dataset, model, ticker_data, return_details=True)

        future_actual = historical_close[historical_close.index > trim_date].sort_index().iloc[:len(median)]
        if len(future_actual) != len(median):
            logger.error(f"Horizonte insuficiente para {ticker}: pred={len(median)} real={len(future_actual)}")
            return ticker, None

        pred_dates = [pd.Timestamp(date).to_pydatetime() for date in future_actual.index]
        historical_pred_close = future_actual.to_numpy(dtype=float)
        median = np.asarray(median, dtype=float)
        differences = np.abs(median - historical_pred_close)
        denominators = np.where(historical_pred_close == 0.0, 1e-12, historical_pred_close)
        relative_diff = (differences / denominators) * 100.0
        mape = np.mean(relative_diff)
        mae = np.mean(differences)
        rmse = float(np.sqrt(np.mean(np.square(median - historical_pred_close))))
        last_observed_close = float(original_close.iloc[-1])
        actual_returns = price_path_to_step_returns(last_observed_close, historical_pred_close)
        directional_accuracy = compute_directional_accuracy(details['relative_returns_median'], actual_returns)

        return ticker, {
            'historical_dates': ticker_data['Date'].dt.tz_localize(None).tolist(),
            'historical_close': original_close.tolist(),
            'pred_dates': pred_dates,
            'predictions': median.tolist(),
            'historical_pred_close': historical_pred_close.tolist(),
            'metrics': {
                'MAPE': mape,
                'MAE': mae,
                'RMSE': rmse,
                'DirAcc': directional_accuracy,
            },
            'last_date': pd.Timestamp(ticker_data['Date'].iloc[-1]).to_pydatetime(),
        }
    except Exception as e:
        logger.error(f"Error al procesar {ticker}: {e}")
        return ticker, None


async def create_benchmark_plot(config, benchmark_tickers, historical_close_dict, years, historical_period_days=365):
    del historical_close_dict
    all_results = {}
    max_prediction_length = config['model']['max_prediction_length']
    start_date = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(days=years * 365)
    end_date = datetime.now().replace(tzinfo=None)

    logger.info('Descargando datos del benchmark...')
    tasks = [fetch_ticker_data(ticker, start_date, end_date) for ticker in benchmark_tickers]
    ticker_data_results = await asyncio.gather(*tasks)
    ticker_data_dict = {ticker: data for ticker, data in ticker_data_results if data is not None and not data.empty}
    if not ticker_data_dict:
        logger.error('No se ha podido descargar ningun ticker del benchmark')
        return all_results

    trim_candidates = []
    for data in ticker_data_dict.values():
        unique_dates = pd.Series(pd.to_datetime(data['Date']).dt.tz_localize(None).sort_values().unique())
        if len(unique_dates) > max_prediction_length:
            trim_candidates.append(unique_dates.iloc[-(max_prediction_length + 1)])
    if not trim_candidates:
        raise ValueError('No hay suficiente historial para construir el benchmark')
    trim_date = min(trim_candidates)

    first_ticker = next(iter(ticker_data_dict))
    first_data = ticker_data_dict[first_ticker].copy()
    _, dataset, normalizers, model = load_data_and_model(config, first_ticker, raw_data=first_data, historical_mode=True)

    try:
        tasks = [
            process_ticker(ticker, data, config, trim_date, dataset, normalizers, model)
            for ticker, data in ticker_data_dict.items()
        ]
        results = await asyncio.gather(*tasks)
        for ticker, result in results:
            if result is not None:
                all_results[ticker] = result

        fig = go.Figure()
        colors = ['#0B5FFF', '#00A36C', '#E74C3C', '#8E44AD', '#FF8C00', '#00B8D9', '#C2185B', '#B7950B', '#6D4C41', '#546E7A']
        split_date = pd.Timestamp(trim_date).isoformat()

        for idx, (ticker, data) in enumerate(all_results.items()):
            color = colors[idx % len(colors)]
            historical_dates = data['historical_dates']
            pred_dates = data['pred_dates']
            historical_close = data['historical_close']
            historical_pred_close = data['historical_pred_close']
            predictions = data['predictions']

            cutoff_date = pd.Timestamp(pred_dates[0]) - pd.Timedelta(days=historical_period_days)
            historical_series = pd.Series(historical_close, index=pd.to_datetime(historical_dates))
            filtered_history = historical_series[historical_series.index >= cutoff_date]

            combined_dates = filtered_history.index.tolist() + pred_dates
            combined_close = filtered_history.tolist() + historical_pred_close
            combined_pred_close = [None] * len(filtered_history) + predictions
            if not (len(combined_dates) == len(combined_close) == len(combined_pred_close)):
                logger.error(f"Longitudes inconsistentes en benchmark para {ticker}")
                continue

            plot_data = pd.DataFrame({
                'Date': combined_dates,
                'Close': combined_close,
                'Predicted_Close': combined_pred_close,
            })
            plot_data['Date'] = pd.to_datetime(plot_data['Date']).dt.tz_localize(None)

            fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Close'], mode='lines', name=f'{ticker} (real)', line=dict(color=color), legendgroup=ticker))
            fig.add_trace(go.Scatter(x=plot_data['Date'], y=plot_data['Predicted_Close'], mode='lines', name=f'{ticker} (prediccion)', line=dict(color=color, dash='dash'), legendgroup=ticker))

        fig.add_shape(type='line', x0=split_date, x1=split_date, y0=0, y1=1, xref='x', yref='paper', line=dict(color='red', width=2, dash='dash'))
        fig.add_annotation(x=split_date, y=1.05, xref='x', yref='paper', text='Inicio del horizonte de validacion', showarrow=False, font=dict(size=12), align='center')
        fig.update_layout(title='Benchmark historico de predicciones', xaxis_title='Fecha', yaxis_title='Precio de cierre', showlegend=True, xaxis=dict(rangeslider=dict(visible=True), type='date'), legend=dict(itemclick='toggle', itemdoubleclick='toggleothers'))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader('Metricas por ticker')
        metrics_rows = []
        for ticker, data in all_results.items():
            row = {'Ticker': ticker}
            row.update(data['metrics'])
            metrics_rows.append(row)
        metrics_df = pd.DataFrame(metrics_rows)
        if not metrics_df.empty:
            st.dataframe(metrics_df.style.format({'MAPE': '{:.2f}%', 'MAE': '{:.2f}', 'RMSE': '{:.2f}', 'DirAcc': '{:.2f}%'}))

        return all_results
    finally:
        if 'model' in locals():
            del model
        if 'dataset' in locals():
            del dataset
        if 'normalizers' in locals():
            del normalizers
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_benchmark_to_csv(benchmark_date, all_results, model_name):
    csv_file = 'data/benchmarks_history.csv'
    metrics = ['MAPE', 'MAE', 'RMSE', 'DirAcc']
    columns = ['Date', 'Model_Name']
    for ticker in all_results.keys():
        for metric in metrics:
            columns.append(f"{ticker}_{metric}")
    columns.extend(['Avg_' + metric for metric in metrics])

    metrics_data = {'Date': [benchmark_date], 'Model_Name': [model_name]}
    valid_metrics = {}
    for ticker, data in all_results.items():
        if isinstance(data, dict) and 'metrics' in data:
            valid_metrics[ticker] = data['metrics']
            for metric in metrics:
                metrics_data[f"{ticker}_{metric}"] = [data['metrics'].get(metric, 0.0)]
        else:
            for metric in metrics:
                metrics_data[f"{ticker}_{metric}"] = [0.0]

    if valid_metrics:
        for metric in metrics:
            values = [metric_dict.get(metric, 0.0) for metric_dict in valid_metrics.values() if metric_dict.get(metric, 0.0) != 0.0]
            metrics_data[f"Avg_{metric}"] = [np.mean(values) if values else 0.0]
    else:
        for metric in metrics:
            metrics_data[f"Avg_{metric}"] = [0.0]

    new_data = pd.DataFrame(metrics_data)
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file, dtype=str)
        updated_df = pd.concat([existing_df, new_data], ignore_index=True)
    else:
        updated_df = new_data
    updated_df.to_csv(csv_file, index=False)
    logger.info(f"Benchmark guardado en {csv_file}")


def load_benchmark_history(benchmark_tickers):
    csv_file = 'data/benchmarks_history.csv'
    metrics = ['MAPE', 'MAE', 'RMSE', 'DirAcc']
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, dtype=str)
        columns = ['Date', 'Model_Name']
        multi_columns = [('Basico', 'Date'), ('Basico', 'Model_Name')]
        for metric in metrics:
            columns.append(f"Avg_{metric}")
            multi_columns.append(('Media', metric))
        for ticker in benchmark_tickers:
            for metric in metrics:
                column_name = f"{ticker}_{metric}"
                if column_name not in df.columns:
                    df[column_name] = '0.0'
                columns.append(column_name)
                multi_columns.append((ticker, metric))
        for metric in metrics:
            average_column = f"Avg_{metric}"
            if average_column not in df.columns:
                df[average_column] = '0.0'
        df = df[columns].fillna('0.0')
        df.columns = pd.MultiIndex.from_tuples(multi_columns)
        for col in df.columns:
            if col[0] != 'Basico':
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
        return df

    multi_columns = pd.MultiIndex.from_tuples([('Basico', 'Date'), ('Basico', 'Model_Name')] + [('Media', metric) for metric in metrics] + [(ticker, metric) for ticker in benchmark_tickers for metric in metrics])
    return pd.DataFrame(columns=multi_columns).fillna('0.0')


def delete_benchmark_row(date_str: str, model_name: str) -> bool:
    csv_file = 'data/benchmarks_history.csv'
    try:
        if not os.path.exists(csv_file):
            logger.warning('No existe el historico de benchmark')
            return False
        df = pd.read_csv(csv_file, dtype=str)
        initial_len = len(df)
        df = df[~((df['Date'] == date_str) & (df['Model_Name'] == model_name))]
        if len(df) < initial_len:
            df.to_csv(csv_file, index=False)
            logger.info(f"Se elimino la fila Date={date_str}, Model_Name={model_name}")
            return True
        logger.warning(f"No existe la fila Date={date_str}, Model_Name={model_name}")
        return False
    except Exception as e:
        logger.error(f"Error al borrar fila del benchmark: {e}")
        return False
