from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from app.benchmark_utils import delete_benchmark_row, load_benchmark_history, save_benchmark_to_store
from app.config_loader import get_default_config_path, load_benchmark_tickers, load_tickers_and_names, load_training_groups
from app.plot_utils import build_stock_plot
from app.services import BenchmarkService, ForecastService, TrainingService
from scripts.runtime_config import ConfigManager
from scripts.utils.logging_utils import configure_logging
from scripts.utils.model_registry import get_active_profile_path, set_active_profile_path
from scripts.utils.training_universe import build_runtime_training_config

configure_logging()
logger = logging.getLogger(__name__)
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)


@st.cache_resource(show_spinner=False)
def _get_config_manager(config_path: str | None = None) -> ConfigManager:
    return ConfigManager(config_path)


@st.cache_resource(show_spinner=False)
def _get_forecast_service(config_path: str | None, years: int) -> ForecastService:
    return ForecastService(_get_config_manager(config_path), years)


@st.cache_resource(show_spinner=False)
def _get_benchmark_service(config_path: str | None, years: int) -> BenchmarkService:
    return BenchmarkService(_get_config_manager(config_path), years)


@st.cache_resource(show_spinner=False)
def _get_training_service(base_config_path: str | None = None) -> TrainingService:
    return TrainingService(_get_config_manager(base_config_path))


def _select_ticker_input(ticker_options: dict[str, str], default_ticker: str) -> str:
    labels = ["Escribir manualmente"] + list(ticker_options.values())
    default_index = 0 if default_ticker not in ticker_options else list(ticker_options.values()).index(ticker_options[default_ticker]) + 1
    ticker_option = st.selectbox(
        "Selecciona una empresa de la lista o escribela manualmente",
        options=labels,
        index=default_index,
    )
    if ticker_option == "Escribir manualmente":
        return st.text_input("Ticker (por ejemplo AAPL, BBVA.MC o CDR.WA)", value=default_ticker)
    return next(ticker for ticker, name in ticker_options.items() if name == ticker_option)


def _render_model_not_ready(readiness):
    st.error(readiness.summary)
    st.write("Problemas detectados:")
    for issue in readiness.issues:
        st.write(f"- {issue}")
    st.info("Ve a la seccion Entrenamiento y genera un modelo canonico para el universo que quieras usar.")


def _render_prediction_view(config: dict, forecast_service: ForecastService, years: int):
    readiness = forecast_service.get_model_readiness()
    if not readiness.ready:
        _render_model_not_ready(readiness)
        return

    ticker_options = load_tickers_and_names(config)
    default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")
    ticker_input = _select_ticker_input(ticker_options, default_ticker)

    if st.button("Generar prediccion"):
        with st.spinner("Generando prediccion..."):
            start_date = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(days=years * 365)
            try:
                ticker_data, original_close, median, lower_bound, upper_bound = forecast_service.predict(
                    ticker_input,
                    start_date,
                    datetime.now().replace(tzinfo=None),
                )
                fig, pred_df = build_stock_plot(
                    config,
                    ticker_data,
                    original_close,
                    median,
                    lower_bound,
                    upper_bound,
                    ticker_input,
                    historical_period_days=st.session_state.historical_period_days,
                )
                st.plotly_chart(fig, use_container_width=True)
                if pred_df is not None:
                    st.subheader("Precios previstos")
                    st.dataframe(
                        pred_df.style.format(
                            {
                                "Fecha": "{:%Y-%m-%d}",
                                "Precio previsto": "{:.2f}",
                                "Cuantil inferior (10%)": "{:.2f}",
                                "Cuantil superior (90%)": "{:.2f}",
                            }
                        )
                    )
            except Exception as exc:
                logger.exception("Error al generar la prediccion para %s", ticker_input)
                st.error(f"Error al generar la prediccion para {ticker_input}: {exc}")


def _render_historical_view(config: dict, forecast_service: ForecastService, years: int):
    readiness = forecast_service.get_model_readiness()
    if not readiness.ready:
        _render_model_not_ready(readiness)
        return

    ticker_options = load_tickers_and_names(config)
    default_ticker = "AAPL" if "AAPL" in ticker_options else (list(ticker_options.keys())[0] if ticker_options else "AAPL")
    ticker_input = _select_ticker_input(ticker_options, default_ticker)

    if st.button("Comparar con historico"):
        with st.spinner("Generando comparacion historica..."):
            start_date = pd.Timestamp(datetime.now()).tz_localize(None) - pd.Timedelta(days=years * 365)
            try:
                ticker_data, original_close, median, lower_bound, upper_bound, historical_close = forecast_service.predict_historical(
                    ticker_input,
                    start_date,
                    datetime.now().replace(tzinfo=None),
                )
                fig, _ = build_stock_plot(
                    config,
                    ticker_data,
                    original_close,
                    median,
                    lower_bound,
                    upper_bound,
                    ticker_input,
                    historical_close=historical_close,
                    historical_period_days=st.session_state.historical_period_days,
                )
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
        _render_model_not_ready(readiness)
        return

    st.write("Tickers del benchmark:", " ".join(benchmark_tickers))
    if st.button("Generar benchmark"):
        with st.spinner("Generando benchmark..."):
            try:
                all_results, fig, metrics_df = benchmark_service.run(
                    benchmark_tickers,
                    historical_period_days=st.session_state.historical_period_days,
                )
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


def _resolve_profile_options(base_config: dict, training_service: TrainingService) -> tuple[list[dict[str, str]], str]:
    default_config_path = get_default_config_path(base_config)
    active_profile_path = training_service.normalize_profile_path(get_active_profile_path(base_config)) or default_config_path
    options = [
        {
            "label": f"Configuracion base [{base_config['model_name']}]",
            "path": default_config_path,
        }
    ]
    for profile_entry in training_service.list_registered_profiles():
        profile_path = training_service.normalize_profile_path(str(profile_entry.get("profile_path") or ""))
        if not profile_path:
            continue
        options.append(
            {
                "label": training_service.format_profile_label(profile_entry),
                "path": profile_path,
            }
        )

    available_paths = {option["path"] for option in options}
    if active_profile_path not in available_paths:
        active_profile_path = default_config_path
    return options, active_profile_path


def _render_active_model_summary(config: dict):
    st.sidebar.markdown("### Modelo activo")
    st.sidebar.write(f"Nombre: `{config['model_name']}`")
    training_run = config.get("training_run") or {}
    if training_run:
        st.sidebar.write(f"Modo: {training_run.get('mode', 'desconocido')}")
        if training_run.get("single_ticker_symbol"):
            st.sidebar.write(f"Ticker: {training_run['single_ticker_symbol']}")
        if training_run.get("predefined_group_name"):
            st.sidebar.write(f"Grupo: {training_run['predefined_group_name']}")
        final_tickers = training_run.get("final_tickers_used") or training_run.get("requested_tickers") or []
        if final_tickers:
            st.sidebar.write(f"Universo: {len(final_tickers)} tickers")


def _render_training_result_summary():
    summary = st.session_state.get("training_result_summary")
    if not summary:
        return
    st.success("Entrenamiento finalizado correctamente")
    st.write(f"Modelo generado: `{summary['model_name']}`")
    st.write(f"Modo: `{summary['universe_mode']}`")
    st.write("Tickers finales usados:", ", ".join(summary["final_tickers_used"]))
    if summary["discarded_tickers"]:
        st.warning(f"Tickers descartados: {', '.join(summary['discarded_tickers'])}")


def _render_training_view(base_config: dict, training_service: TrainingService):
    st.subheader("Configuracion de entrenamiento")
    _render_training_result_summary()

    mode_label = st.radio(
        "Modo de entrenamiento",
        options=["Ticker unico", "Grupo predefinido"],
        horizontal=True,
    )
    mode = "single_ticker" if mode_label == "Ticker unico" else "predefined_group"

    single_ticker_symbol = None
    predefined_group_name = None

    if mode == "single_ticker":
        single_ticker_symbol = st.text_input(
            "Ticker a entrenar",
            value="BBVA.MC",
            help="Ejemplos: BBVA.MC, MSFT, TSLA, BNP.PA",
        )
    else:
        groups = load_training_groups(base_config)
        if not groups:
            st.error("No hay grupos predefinidos habilitados en training_universes.yaml")
            return
        group_labels = {group["label"]: group for group in groups}
        selected_label = st.selectbox("Grupo predefinido", options=list(group_labels.keys()))
        selected_group = group_labels[selected_label]
        predefined_group_name = str(selected_group["name"])
        st.caption(selected_group.get("description") or "Sin descripcion")
        with st.expander("Ver tickers del grupo", expanded=True):
            st.write(", ".join(selected_group.get("tickers", [])))
            if selected_group.get("notes"):
                st.caption(str(selected_group["notes"]))

    years = st.number_input(
        "Anos historicos",
        min_value=3,
        max_value=15,
        value=int(base_config["prediction"]["years"]),
        step=1,
    )
    use_optuna = st.checkbox("Usar Optuna durante el entrenamiento", value=False)

    try:
        universe = training_service.preview_universe(
            mode=mode,
            single_ticker_symbol=single_ticker_symbol,
            predefined_group_name=predefined_group_name,
        )
        runtime_config = build_runtime_training_config(base_config, universe, int(years))
        st.markdown("### Universo que se va a entrenar")
        st.write(f"Etiqueta: {universe.label}")
        st.write(f"Modelo previsto: `{runtime_config['model_name']}`")
        st.write("Tickers solicitados:", ", ".join(universe.tickers))
    except Exception as exc:
        st.error(f"Configuracion de entrenamiento invalida: {exc}")
        return

    st.info("El entrenamiento desde Streamlit se lanza desde cero para evitar mezclar artefactos de universos distintos.")
    if st.button("Entrenar universo seleccionado", type="primary"):
        with st.spinner("Entrenando modelo..."):
            try:
                result = training_service.train(
                    mode=mode,
                    years=int(years),
                    use_optuna=bool(use_optuna),
                    single_ticker_symbol=single_ticker_symbol,
                    predefined_group_name=predefined_group_name,
                )
                st.session_state.training_result_summary = {
                    "model_name": result.model_name,
                    "universe_mode": result.universe_mode,
                    "final_tickers_used": list(result.final_tickers_used),
                    "discarded_tickers": list(result.discarded_tickers),
                }
                if result.profile_path:
                    set_active_profile_path(base_config, result.profile_path)
                    st.session_state.selected_config_path = str(Path(result.profile_path))
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:
                logger.exception("Error al entrenar el universo seleccionado")
                st.error(f"Error al entrenar el universo seleccionado: {exc}")


def main():
    st.set_page_config(page_title="Predictor bursatil", layout="wide")
    st.title("Predictor bursatil")

    base_config_manager = _get_config_manager(None)
    base_config = base_config_manager.config
    base_config_path = get_default_config_path(base_config)
    training_service = _get_training_service(base_config_path)

    profile_options, default_active_profile = _resolve_profile_options(base_config, training_service)
    option_labels = [option["label"] for option in profile_options]
    path_by_label = {option["label"]: option["path"] for option in profile_options}
    label_by_path = {option["path"]: option["label"] for option in profile_options}

    if "selected_config_path" not in st.session_state or st.session_state.selected_config_path not in label_by_path:
        st.session_state.selected_config_path = default_active_profile

    selected_label = st.sidebar.selectbox(
        "Perfil activo para inferencia",
        options=option_labels,
        index=option_labels.index(label_by_path[st.session_state.selected_config_path]),
    )
    selected_config_path = path_by_label[selected_label]
    if selected_config_path != st.session_state.selected_config_path:
        st.session_state.selected_config_path = selected_config_path
        if selected_config_path == base_config_path:
            set_active_profile_path(base_config, None)
        else:
            set_active_profile_path(base_config, selected_config_path)
        st.cache_resource.clear()
        st.rerun()

    config_manager = _get_config_manager(st.session_state.selected_config_path)
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

    _render_active_model_summary(config)

    forecast_service = _get_forecast_service(st.session_state.selected_config_path, years)
    benchmark_service = _get_benchmark_service(st.session_state.selected_config_path, years)
    readiness = forecast_service.get_model_readiness()
    if readiness.ready:
        st.sidebar.success("Modelo preparado para inferencia")
    else:
        st.sidebar.error("Modelo no preparado")

    benchmark_tickers = load_benchmark_tickers(config)
    page = st.sidebar.selectbox("Seccion", ["Entrenamiento", "Prediccion futura", "Comparacion historica", "Benchmark"])

    if page == "Entrenamiento":
        _render_training_view(base_config, training_service)
    elif page == "Prediccion futura":
        _render_prediction_view(config, forecast_service, years)
    elif page == "Comparacion historica":
        _render_historical_view(config, forecast_service, years)
    else:
        _render_benchmark_view(config, benchmark_service, benchmark_tickers)


if __name__ == "__main__":
    main()
