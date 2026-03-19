import asyncio
import logging
import os
import sys
from datetime import datetime

import nest_asyncio
import pandas as pd
import streamlit as st
import torch

nest_asyncio.apply()
logging.getLogger('streamlit.watcher.local_sources_watcher').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.benchmark_utils import create_benchmark_plot, delete_benchmark_row, load_benchmark_history, save_benchmark_to_csv
from app.config_loader import load_benchmark_tickers, load_tickers_and_names
from app.plot_utils import create_stock_plot
from scripts.data_fetcher import DataFetcher
from scripts.prediction_engine import generate_predictions, load_data_and_model, preprocess_data
from scripts.runtime_config import ConfigManager
from scripts.utils.logging_utils import configure_logging

configure_logging()


class StockPredictor:
    def __init__(self, config, years):
        self.config = config
        self.years = years
        self.max_prediction_length = config['model']['max_prediction_length']
        self.fetcher = DataFetcher(ConfigManager(), years)

    def fetch_stock_data(self, ticker, start_date, end_date):
        return self.fetcher.fetch_stock_data_sync(ticker, start_date, end_date)

    def predict(self, ticker, start_date, end_date):
        new_data = self.fetch_stock_data(ticker, start_date, end_date)
        if new_data.empty:
            raise ValueError(f"No hay datos para {ticker}")
        new_data['Date'] = pd.to_datetime(new_data['Date']).dt.tz_localize(None)
        _, dataset, normalizers, model = load_data_and_model(self.config, ticker, raw_data=new_data)
        ticker_data, original_close = preprocess_data(self.config, new_data, ticker, normalizers)
        with torch.no_grad():
            median, lower_bound, upper_bound = generate_predictions(self.config, dataset, model, ticker_data)
        return ticker_data, original_close, median, lower_bound, upper_bound

    def predict_historical(self, ticker, start_date, end_date):
        full_data = self.fetch_stock_data(ticker, start_date, end_date)
        if full_data.empty:
            raise ValueError(f"No hay datos para {ticker}")

        full_data = full_data[full_data['Ticker'] == ticker].copy()
        full_data['Date'] = pd.to_datetime(full_data['Date']).dt.tz_localize(None)
        full_data = full_data.sort_values('Date')
        unique_dates = full_data['Date'].drop_duplicates().reset_index(drop=True)
        if len(unique_dates) <= self.max_prediction_length:
            raise ValueError('No hay suficiente historial para comparacion historica')

        trim_date = unique_dates.iloc[-(self.max_prediction_length + 1)]
        new_data = full_data[full_data['Date'] <= trim_date].copy()
        _, dataset, normalizers, model = load_data_and_model(self.config, ticker, raw_data=new_data, historical_mode=True)
        ticker_data, original_close = preprocess_data(self.config, new_data, ticker, normalizers, historical_mode=True)
        with torch.no_grad():
            median, lower_bound, upper_bound = generate_predictions(self.config, dataset, model, ticker_data)

        historical_close = full_data.set_index('Date')['Close']
        return ticker_data, original_close, median, lower_bound, upper_bound, historical_close


def main():
    st.set_page_config(page_title='Predictor bursatil', layout='wide')
    st.title('Predictor bursatil')

    config_manager = ConfigManager()
    config = config_manager.config
    years = config['prediction']['years']

    if 'historical_period_days' not in st.session_state:
        st.session_state.historical_period_days = 365

    historical_period_options = {
        '90 dias': 90,
        '180 dias': 180,
        '1 ano': 365,
        '2 anos': 730,
        'Todo el periodo': years * 365,
    }
    selected_period = st.sidebar.selectbox('Periodo historico visible', options=list(historical_period_options.keys()), index=list(historical_period_options.keys()).index('1 ano'))
    st.session_state.historical_period_days = historical_period_options[selected_period]

    predictor = StockPredictor(config, years)
    benchmark_tickers = load_benchmark_tickers(config)
    page = st.sidebar.selectbox('Seccion', ['Prediccion futura', 'Comparacion historica', 'Benchmark'])

    if page == 'Prediccion futura':
        ticker_options = load_tickers_and_names(config)
        default_ticker = 'AAPL' if 'AAPL' in ticker_options else (list(ticker_options.keys())[0] if ticker_options else 'AAPL')
        ticker_option = st.selectbox('Selecciona una empresa de la lista o escribela manualmente', options=['Escribir manualmente'] + list(ticker_options.values()), index=0 if default_ticker not in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1)
        ticker_input = default_ticker
        if ticker_option == 'Escribir manualmente':
            ticker_input = st.text_input('Ticker (por ejemplo AAPL o CDR.WA)', value=default_ticker)
        else:
            ticker_input = [k for k, v in ticker_options.items() if v == ticker_option][0]

        if st.button('Generar prediccion'):
            with st.spinner('Generando prediccion...'):
                try:
                    start_date = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(days=years * 365)
                    ticker_data, original_close, median, lower_bound, upper_bound = predictor.predict(ticker_input, start_date, datetime.now().replace(tzinfo=None))
                    create_stock_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input, historical_period_days=st.session_state.historical_period_days)
                except Exception as e:
                    st.error(f"Error al generar la prediccion para {ticker_input}: {e}")

    elif page == 'Comparacion historica':
        ticker_options = load_tickers_and_names(config)
        default_ticker = 'AAPL' if 'AAPL' in ticker_options else (list(ticker_options.keys())[0] if ticker_options else 'AAPL')
        ticker_option = st.selectbox('Selecciona una empresa de la lista o escribela manualmente', options=['Escribir manualmente'] + list(ticker_options.values()), index=0 if default_ticker not in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1)
        ticker_input = default_ticker
        if ticker_option == 'Escribir manualmente':
            ticker_input = st.text_input('Ticker (por ejemplo AAPL o CDR.WA)', value=default_ticker)
        else:
            ticker_input = [k for k, v in ticker_options.items() if v == ticker_option][0]

        if st.button('Comparar con historico'):
            with st.spinner('Generando comparacion historica...'):
                try:
                    start_date = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(days=years * 365)
                    ticker_data, original_close, median, lower_bound, upper_bound, historical_close = predictor.predict_historical(ticker_input, start_date, datetime.now().replace(tzinfo=None))
                    create_stock_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input, historical_close, historical_period_days=st.session_state.historical_period_days)
                except Exception as e:
                    st.error(f"Error al comparar con historico para {ticker_input}: {e}")

    else:
        st.write('Tickers del benchmark:', ' '.join(benchmark_tickers))
        if st.button('Generar benchmark'):
            with st.spinner('Generando benchmark...'):
                try:
                    all_results = asyncio.run(create_benchmark_plot(config, benchmark_tickers, {}, years, historical_period_days=st.session_state.historical_period_days))
                    benchmark_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    save_benchmark_to_csv(benchmark_date, all_results, config['model_name'])
                except Exception as e:
                    logger.error(f"Error al generar benchmark: {e}")
                    st.error(f"Error al generar benchmark: {e}")

        st.subheader('Historico de benchmarks')
        benchmark_history = load_benchmark_history(benchmark_tickers)
        format_dict = {('Basico', 'Date'): '{}', ('Basico', 'Model_Name'): '{}'}
        for ticker in benchmark_tickers:
            format_dict[(ticker, 'MAPE')] = '{:.2f}%'
            format_dict[(ticker, 'MAE')] = '{:.2f}'
            format_dict[(ticker, 'RMSE')] = '{:.2f}'
            format_dict[(ticker, 'DirAcc')] = '{:.2f}%'
        for metric in ['MAPE', 'DirAcc']:
            format_dict[('Media', metric)] = '{:.2f}%'
        format_dict[('Media', 'MAE')] = '{:.2f}'
        format_dict[('Media', 'RMSE')] = '{:.2f}'
        st.dataframe(benchmark_history.style.format(format_dict))

        st.subheader('Gestion del historico')
        csv_path = os.path.join(project_root, 'data', 'benchmarks_history.csv')
        if os.path.exists(csv_path):
            df_raw = pd.read_csv(csv_path, dtype=str)
            if df_raw.empty:
                st.info('No hay benchmark guardados.')
            else:
                if 'confirm_delete' not in st.session_state:
                    st.session_state.confirm_delete = None
                for i, row in df_raw.iterrows():
                    cols = st.columns([0.12, 0.44, 0.44])
                    with cols[0]:
                        if st.button('Borrar', key=f'del_row_{i}'):
                            st.session_state.confirm_delete = {'idx': int(i), 'Date': row['Date'], 'Model_Name': row['Model_Name']}
                    with cols[1]:
                        st.write(f"Fecha: {row.get('Date', '')}")
                    with cols[2]:
                        st.write(f"Modelo: {row.get('Model_Name', '')}")
                    if st.session_state.confirm_delete and st.session_state.confirm_delete.get('idx') == i:
                        confirm_cols = st.columns([0.12, 0.44, 0.44])
                        with confirm_cols[0]:
                            st.warning('Confirmar?')
                        with confirm_cols[1]:
                            if st.button('Si, borrar', key=f'confirm_yes_{i}'):
                                ok = delete_benchmark_row(st.session_state.confirm_delete['Date'], st.session_state.confirm_delete['Model_Name'])
                                st.session_state.confirm_delete = None
                                if ok:
                                    st.success('Fila borrada')
                                else:
                                    st.error('No se pudo borrar la fila')
                                try:
                                    st.rerun()
                                except Exception:
                                    st.experimental_rerun()
                        with confirm_cols[2]:
                            if st.button('Cancelar', key=f'confirm_no_{i}'):
                                st.session_state.confirm_delete = None
                                try:
                                    st.rerun()
                                except Exception:
                                    st.experimental_rerun()
        else:
            st.info('Todavia no existe historico de benchmark.')


if __name__ == '__main__':
    main()
