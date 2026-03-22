from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from app.benchmark_utils import delete_benchmark_row, load_benchmark_history, save_benchmark_to_store
from app.config_loader import load_benchmark_tickers, load_tickers_and_names
from app.plot_utils import build_stock_plot
from app.services import BenchmarkService, ForecastService
from scripts.runtime_config import ConfigManager
from scripts.utils.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger(__name__)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)


@st.cache_resource(show_spinner=False)
def _get_config_manager() -> ConfigManager:
    return ConfigManager()


@st.cache_resource(show_spinner=False)
def _get_forecast_service(years: int) -> ForecastService:
    return ForecastService(_get_config_manager(), years)


@st.cache_resource(show_spinner=False)
def _get_benchmark_service(years: int) -> BenchmarkService:
    return BenchmarkService(_get_config_manager(), years)


def _select_ticker_input(ticker_options: dict[str, str], default_ticker: str) -> str:
    labels = ["Escribir manualmente"] + list(ticker_options.values())
    default_index = 0 if default_ticker not in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
    ticker_option = st.selectbox(
        "Selecciona una empresa de la lista o escribela manualmente",
        options=labels,
        index=default_index,
    )
    if ticker_option == "Escribir manualmente":
        return st.text_input("Ticker (por ejemplo AAPL o CDR.WA)", value=default_ticker)
    return next(ticker for ticker, name in ticker_options.items() if name == ticker_option)


def _render_prediction_view(config: dict, forecast_service: ForecastService, years: int):
    readiness = forecast_service.get_model_readiness()
    if not readiness.ready:
        st.error(readiness.summary)
        st.write("Problemas detectados:")
        for issue in readiness.issues:
            st.write(f"- {issue}")
        st.info("Antes de predecir, ejecuta preparar_modelo.bat para generar artefactos canonicos del modelo.")
        return

    ticker_options = load_tickers_and_names(config)
    default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")
    ticker_input = _select_ticker_input(ticker_options, default_ticker)

    if st.button("Generar prediccion"):
        with st.spinner("Generando prediccion..."):
            start_date = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(days=years * 365)
            try:
                ticker_data, original_close, median, lower_bound, upper_bound = forecast_service.predict(ticker_input, start_date, datetime.now().replace(tzinfo=None))
                fig, pred_df = build_stock_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input, historical_period_days=st.session_state.historical_period_days)
                st.plotly_chart(fig, use_container_width=True)
                if pred_df is not None:
                    st.subheader("Precios previstos")
                    st.dataframe(pred_df.style.format({"Fecha": "{:%Y-%m-%d}", "Precio previsto": "{:.2f}", "Cuantil inferior (10%)": "{:.2f}", "Cuantil superior (90%)": "{:.2f}"}))
            except Exception as exc:
                logger.exception("Error al generar la prediccion para %s", ticker_input)
                st.error(f"Error al generar la prediccion para {ticker_input}: {exc}")


def _render_historical_view(config: dict, forecast_service: ForecastService, years: int):
    readiness = forecast_service.get_model_readiness()
    if not readiness.ready:
        st.error(readiness.summary)
        st.write("Problemas detectados:")
        for issue in readiness.issues:
            st.write(f"- {issue}")
        st.info("Antes de comparar con historico, ejecuta preparar_modelo.bat para generar artefactos canonicos del modelo.")
        return

    ticker_options = load_tickers_and_names(config)
    default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")
    ticker_input = _select_ticker_input(ticker_options, default_ticker)

    if st.button("Comparar con historico"):
        with st.spinner("Generando comparacion historica..."):
            start_date = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(days=years * 365)
            try:
                ticker_data, original_close, median, lower_bound, upper_bound, historical_close = forecast_service.predict_historical(ticker_input, start_date, datetime.now().replace(tzinfo=None))
                fig, _ = build_stock_plot(config, ticker_data, original_close, median, lower_bound, upper_bound, ticker_input, historical_close=historical_close, historical_period_days=st.session_state.historical_period_days)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                logger.exception("Error al comparar con historico para %s", ticker_input)
                st.error(f"Error al comparar con historico para {ticker_input}: {exc}")


def _render_benchmark_history(config: dict, benchmark_tickers: list[str]):
    st.subheader("Historico de benchmarks")
    benchmark_history, entries = load_benchmark_history(config, benchmark_tickers)
    format_dict = {("Basico", "Date"): "{}", ("Basico", "Model_Name"): "{}"}
    for ticker in benchmark_tickers:
        format_dict[(ticker, "MAPE")] = "{:.2f}%"
        format_dict[(ticker, "MAE")] = "{:.2f}"
        format_dict[(ticker, "RMSE")] = "{:.2f}"
        format_dict[(ticker, "DirAcc")] = "{:.2f}%"
    for metric in ["MAPE", "DirAcc"]:
        format_dict[("Media", metric)] = "{:.2f}%"
    format_dict[("Media", "MAE")] = "{:.2f}"
    format_dict[("Media", "RMSE")] = "{:.2f}"
    st.dataframe(benchmark_history.style.format(format_dict))

    st.subheader("Gestion del historico")
    if not entries:
        st.info("Todavia no existe historico de benchmark.")
        return

    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = None

    for entry in entries:
        row_id = int(entry["id"])
        cols = st.columns([0.12, 0.44, 0.44])
        with cols[0]:
            if st.button("Borrar", key=f"del_row_{row_id}"):
                st.session_state.confirm_delete = entry
        with cols[1]:
            st.write(f"Fecha: {entry['Date']}")
        with cols[2]:
            st.write(f"Modelo: {entry['Model_Name']}")

        if st.session_state.confirm_delete and int(st.session_state.confirm_delete.get("id", -1)) == row_id:
            confirm_cols = st.columns([0.12, 0.44, 0.44])
            with confirm_cols[0]:
                st.warning("Confirmar?")
            with confirm_cols[1]:
                if st.button("Si, borrar", key=f"confirm_yes_{row_id}"):
                    ok = delete_benchmark_row(config, row_id)
                    st.session_state.confirm_delete = None
                    if ok:
                        st.success("Fila borrada")
                    else:
                        st.error("No se pudo borrar la fila")
                    st.rerun()
            with confirm_cols[2]:
                if st.button("Cancelar", key=f"confirm_no_{row_id}"):
                    st.session_state.confirm_delete = None
                    st.rerun()


def _render_benchmark_view(config: dict, benchmark_service: BenchmarkService, benchmark_tickers: list[str]):
    readiness = benchmark_service.get_model_readiness()
    if not readiness.ready:
        st.error(readiness.summary)
        st.write("Problemas detectados:")
        for issue in readiness.issues:
            st.write(f"- {issue}")
        st.info("Antes de ejecutar el benchmark, ejecuta preparar_modelo.bat para generar artefactos canonicos del modelo.")
        return

    st.write("Tickers del benchmark:", " ".join(benchmark_tickers))
    if st.button("Generar benchmark"):
        with st.spinner("Generando benchmark..."):
            try:
                all_results, fig, metrics_df = benchmark_service.run(benchmark_tickers, historical_period_days=st.session_state.historical_period_days)
                benchmark_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                save_benchmark_to_store(config, benchmark_date, all_results, config["model_name"])
                st.plotly_chart(fig, use_container_width=True)
                if not metrics_df.empty:
                    st.subheader("Metricas por ticker")
                    st.dataframe(metrics_df.style.format({"MAPE": "{:.2f}%", "MAE": "{:.2f}", "RMSE": "{:.2f}", "DirAcc": "{:.2f}%"}))
            except Exception as exc:
                logger.exception("Error al generar benchmark")
                st.error(f"Error al generar benchmark: {exc}")

    _render_benchmark_history(config, benchmark_tickers)


def main():
    st.set_page_config(page_title="Predictor bursatil", layout="wide")
    st.title("Predictor bursatil")

    config_manager = _get_config_manager()
    config = config_manager.config
    years = config["prediction"]["years"]
    if "historical_period_days" not in st.session_state:
        st.session_state.historical_period_days = 365

    historical_period_options = {
        "90 dias": 90,
        "180 dias": 180,
        "1 ano": 365,
        "2 anos": 730,
        "Todo el periodo": years * 365,
    }
    selected_period = st.sidebar.selectbox(
        "Periodo historico visible",
        options=list(historical_period_options.keys()),
        index=list(historical_period_options.keys()).index("1 ano"),
    )
    st.session_state.historical_period_days = historical_period_options[selected_period]

    forecast_service = _get_forecast_service(years)
    benchmark_service = _get_benchmark_service(years)
    readiness = forecast_service.get_model_readiness()
    if readiness.ready:
        st.sidebar.success("Modelo preparado para inferencia")
    else:
        st.sidebar.error("Modelo no preparado")
    benchmark_tickers = load_benchmark_tickers(config)
    page = st.sidebar.selectbox("Seccion", ["Prediccion futura", "Comparacion historica", "Benchmark"])

    if page == "Prediccion futura":
        _render_prediction_view(config, forecast_service, years)
    elif page == "Comparacion historica":
        _render_historical_view(config, forecast_service, years)
    else:
        _render_benchmark_view(config, benchmark_service, benchmark_tickers)


if __name__ == "__main__":
    main()
