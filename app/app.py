from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import pandas as pd
import streamlit as st

from app.config_loader import get_default_config_path, load_tickers_and_names, load_training_groups
from app.plot_utils import build_benchmark_plot, build_stock_plot
from app.services import BenchmarkService, ForecastService, TrainingService
from scripts.runtime_config import ConfigManager
from scripts.utils.device_utils import resolve_execution_context
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
                ticker_data, original_close, median, lower_bound, upper_bound, details = forecast_service.predict(
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
                    forecast_dates=details.get("forecast_dates"),
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


def _serialize_for_ui(value):
    if is_dataclass(value):
        return {key: _serialize_for_ui(item) for key, item in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _serialize_for_ui(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_for_ui(item) for item in value]
    return value


def _combine_benchmark_datetime(selected_date, selected_time):
    return datetime.combine(selected_date, selected_time).replace(tzinfo=None)


def _render_benchmark_run_view(run_view: dict):
    manifest = run_view["manifest"]
    summary = run_view["summary"]
    ticker_results = run_view["ticker_results"]
    discarded_tickers = run_view["discarded_tickers"]
    artifact_audit = run_view["artifact_audit"]
    plot_payload = run_view["plot_payload"]

    st.subheader("Resumen de corrida")
    st.write(f"Run ID: `{manifest.run_id}`")
    st.write(f"Modo: `{manifest.mode.value}`")
    st.write(f"Estado de validacion: `{manifest.validation_state.value}`")
    st.write(f"Benchmark ID: `{manifest.benchmark_id}`")
    st.write(f"Modelo: `{manifest.model_name}`")
    st.write(f"Split date: `{manifest.split_date.isoformat()}`")
    st.write(f"Snapshot ID: `{manifest.snapshot_id or 'N/A'}`")
    summary_row = {
        "run_id": summary.run_id,
        "mode": summary.mode.value,
        "validation_state": summary.validation_state.value,
        "requested_tickers": len(summary.requested_tickers),
        "effective_universe": len(summary.effective_universe),
        "completed_tickers": len(summary.completed_tickers),
        "failed_tickers": len(summary.failed_tickers),
        "discarded_tickers": len(summary.discarded_tickers),
        **summary.metrics,
    }
    st.dataframe(pd.DataFrame([summary_row]))

    if discarded_tickers:
        st.subheader("Descartes y razones")
        st.dataframe(
            pd.DataFrame(
                [{"ticker": item.ticker, "reason": item.reason.value, "detail": item.detail} for item in discarded_tickers]
            )
        )

    if ticker_results:
        metrics_df = pd.DataFrame(
            [
                {
                    "Ticker": result.ticker,
                    "Observed_Points": len(result.actual_close),
                    **result.metrics,
                }
                for result in ticker_results
            ]
        )
        st.subheader("Metricas por ticker")
        st.dataframe(metrics_df)

        tickers = [result.ticker for result in ticker_results]
        selected_ticker = st.selectbox("Ticker de detalle", options=tickers, key=f"benchmark_ticker_{manifest.run_id}")
        fig = build_benchmark_plot(plot_payload, selected_ticker)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Manifest y trazabilidad", expanded=False):
        st.json(_serialize_for_ui(manifest))
        st.write("Checksums y artefactos auditables")
        st.json(_serialize_for_ui(artifact_audit))


def _render_legacy_benchmark_history(benchmark_service: BenchmarkService):
    with st.expander("Modo legacy / exploratorio", expanded=False):
        legacy = benchmark_service.load_legacy_history()
        entries = legacy.get("entries") or []
        if not entries:
            st.info("No existe historico legacy de benchmark.")
            return
        st.warning("El benchmark legacy se conserva solo como compatibilidad exploratoria. No es el benchmark oficial reproducible.")
        st.dataframe(pd.DataFrame(entries))


def _render_benchmark_view(benchmark_service: BenchmarkService):
    readiness = benchmark_service.get_model_readiness()
    if not readiness.ready:
        _render_model_not_ready(readiness)
        return

    benchmark_service.ensure_default_definition()
    definitions = benchmark_service.list_definitions()
    if not definitions:
        st.error("No hay definiciones de benchmark disponibles.")
        return

    definition_labels = {
        f"{entry.label} [{entry.benchmark_id}:{entry.definition_id}]": entry.definition_id
        for entry in definitions
    }
    selected_definition_label = st.selectbox("Definicion oficial", options=list(definition_labels.keys()))
    selected_definition_id = definition_labels[selected_definition_label]

    now_value = datetime.now().replace(microsecond=0, second=0)
    benchmark_date = st.date_input("As-of date", value=now_value.date(), key="benchmark_as_of_date")
    benchmark_time = st.time_input("As-of time", value=now_value.time(), key="benchmark_as_of_time")
    as_of_timestamp = _combine_benchmark_datetime(benchmark_date, benchmark_time)
    st.caption("La fecha/hora seleccionada se persistira en el snapshot y en el run manifest. El valor por defecto usa el reloj local solo como conveniencia de UI.")

    mode_label = st.radio(
        "Modo de benchmark",
        options=["Frozen oficial", "Live exploratorio"],
        horizontal=True,
    )
    if mode_label == "Frozen oficial":
        st.success("Modo oficial reproducible: crea snapshot inmutable, manifiesto completo y permite rerun exacto.")
    else:
        st.warning("Modo live exploratorio: usa descarga viva y no debe interpretarse como benchmark oficial reproducible.")

    run_button_label = "Ejecutar frozen benchmark" if mode_label == "Frozen oficial" else "Ejecutar live benchmark"
    if st.button(run_button_label, type="primary"):
        with st.spinner("Ejecutando benchmark..."):
            try:
                if mode_label == "Frozen oficial":
                    run_bundle = benchmark_service.run_frozen(selected_definition_id, as_of_timestamp)
                else:
                    run_bundle = benchmark_service.run_live(selected_definition_id, as_of_timestamp)
                st.session_state.last_benchmark_run_id = run_bundle.manifest.run_id
                st.success(f"Benchmark completado: {run_bundle.manifest.run_id}")
                _render_benchmark_run_view(benchmark_service.load_run_view(run_bundle.manifest.run_id))
            except Exception as exc:
                logger.exception("Error al ejecutar benchmark")
                st.error(f"Error al ejecutar benchmark: {exc}")

    st.subheader("Catalogo de runs")
    run_filters_cols = st.columns(5)
    with run_filters_cols[0]:
        filter_benchmark_id = st.text_input("Filtro benchmark_id", value="")
    with run_filters_cols[1]:
        filter_mode_label = st.selectbox("Filtro modo", options=["Todos", "frozen", "live"], index=0)
    with run_filters_cols[2]:
        filter_model_name = st.text_input("Filtro modelo", value="")
    with run_filters_cols[3]:
        filter_run_id = st.text_input("Filtro run_id", value="")
    with run_filters_cols[4]:
        filter_validation_label = st.selectbox(
            "Filtro validacion",
            options=["Todos", "frozen_validated", "live_exploratory"],
            index=0,
        )
    filter_date_text = st.text_input("Filtro fecha (YYYY-MM-DD)", value="")
    filter_universe_text = st.text_input("Filtro universo efectivo contiene", value="")

    mode_filter = None if filter_mode_label == "Todos" else filter_mode_label
    validation_filter = None if filter_validation_label == "Todos" else filter_validation_label
    runs = benchmark_service.list_runs(
        benchmark_id=filter_benchmark_id or None,
        mode=mode_filter,
        model_name=filter_model_name or None,
        run_id=filter_run_id or None,
        validation_state=validation_filter,
        effective_universe_contains=filter_universe_text or None,
    )
    runs_df = pd.DataFrame(runs)
    if not runs_df.empty and filter_date_text:
        runs_df = runs_df[runs_df["created_at"].astype(str).str.contains(filter_date_text, na=False)]
    st.dataframe(runs_df)

    available_run_ids = runs_df["run_id"].tolist() if not runs_df.empty else []
    if available_run_ids:
        default_run_id = st.session_state.get("last_benchmark_run_id", available_run_ids[0])
        if default_run_id not in available_run_ids:
            default_run_id = available_run_ids[0]
        selected_run_id = st.selectbox(
            "Run para inspeccion",
            options=available_run_ids,
            index=available_run_ids.index(default_run_id),
        )
        rerun_cols = st.columns(2)
        with rerun_cols[0]:
            if st.button("Cargar run seleccionado"):
                _render_benchmark_run_view(benchmark_service.load_run_view(selected_run_id))
        with rerun_cols[1]:
            if st.button("Rerun exacto del frozen seleccionado"):
                try:
                    rerun_bundle = benchmark_service.rerun_exact(selected_run_id)
                    st.success(f"Rerun exacto disponible: {rerun_bundle.manifest.run_id}")
                    _render_benchmark_run_view(benchmark_service.load_run_view(rerun_bundle.manifest.run_id))
                except Exception as exc:
                    logger.exception("Error al reejecutar benchmark frozen")
                    st.error(f"No se pudo reejecutar el frozen benchmark: {exc}")

        st.subheader("Comparacion entre corridas")
        comparison_cols = st.columns(2)
        with comparison_cols[0]:
            left_run_id = st.selectbox("Run A", options=available_run_ids, key="comparison_left_run")
        with comparison_cols[1]:
            right_run_id = st.selectbox(
                "Run B",
                options=available_run_ids,
                index=min(1, len(available_run_ids) - 1),
                key="comparison_right_run",
            )
        if left_run_id and right_run_id and left_run_id != right_run_id:
            try:
                comparison = benchmark_service.compare_runs(left_run_id, right_run_id)
                st.write("Delta de metricas globales")
                st.dataframe(pd.DataFrame([comparison.summary_delta]))
                if comparison.ticker_deltas:
                    st.write("Delta por ticker")
                    st.dataframe(
                        pd.DataFrame(
                            [
                                {
                                    "ticker": item["ticker"],
                                    **item["delta_metrics"],
                                }
                                for item in comparison.ticker_deltas
                            ]
                        )
                    )
            except Exception as exc:
                logger.exception("Error al comparar corridas de benchmark")
                st.error(f"No se pudieron comparar las corridas: {exc}")

    _render_legacy_benchmark_history(benchmark_service)


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


def _render_runtime_status(title: str, runtime_status: dict):
    st.markdown(f"### {title}")
    st.write(f"Dispositivo actual: **{runtime_status['backend']}**")
    st.write(f"CUDA disponible: **{'Si' if runtime_status['cuda_available'] else 'No'}**")
    st.write(f"GPU detectada: **{runtime_status['gpu_name']}**")
    st.write(f"Torch: **{runtime_status['torch_version']}**")
    st.write(f"CUDA segun torch: **{runtime_status['torch_cuda_version']}**")
    st.write(f"Precision efectiva: **{runtime_status['precision']}**")
    if runtime_status.get("fallback_reason"):
        st.warning(f"Motivo del fallback a CPU: {runtime_status['fallback_reason']}")


def _render_sidebar_runtime_status(config: dict):
    runtime = resolve_execution_context(config, purpose="predict").to_display_dict()
    st.sidebar.markdown("### Hardware de ejecucion")
    st.sidebar.write(f"Dispositivo: `{runtime['backend']}`")
    st.sidebar.write(f"GPU: `{runtime['gpu_name']}`")
    st.sidebar.write(f"torch: `{runtime['torch_version']}`")
    st.sidebar.write(f"CUDA torch: `{runtime['torch_cuda_version']}`")
    if runtime.get("fallback_reason"):
        st.sidebar.caption(runtime["fallback_reason"])


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
        _render_runtime_status("Hardware de entrenamiento", training_service.get_runtime_status(runtime_config))
    except Exception as exc:
        st.error(f"Configuracion de entrenamiento invalida: {exc}")
        return

    st.info("El entrenamiento desde Streamlit se lanza desde cero para evitar mezclar artefactos de universos distintos.")
    runtime_status = training_service.get_runtime_status(runtime_config)
    if runtime_status["backend"] == "CPU":
        st.warning("CUDA no esta disponible. El entrenamiento se ejecutara en CPU y puede ser muy lento.")
    else:
        st.success(f"El entrenamiento se lanzara usando GPU: {runtime_status['gpu_name']}")
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
    _render_sidebar_runtime_status(config)

    forecast_service = _get_forecast_service(st.session_state.selected_config_path, years)
    benchmark_service = _get_benchmark_service(st.session_state.selected_config_path, years)
    readiness = forecast_service.get_model_readiness()
    if readiness.ready:
        st.sidebar.success("Modelo preparado para inferencia")
    else:
        st.sidebar.error("Modelo no preparado")

    page = st.sidebar.selectbox("Seccion", ["Entrenamiento", "Prediccion futura", "Comparacion historica", "Benchmark"])

    if page == "Entrenamiento":
        _render_training_view(base_config, training_service)
    elif page == "Prediccion futura":
        _render_prediction_view(config, forecast_service, years)
    elif page == "Comparacion historica":
        _render_historical_view(config, forecast_service, years)
    else:
        _render_benchmark_view(benchmark_service)


if __name__ == "__main__":
    main()
