from __future__ import annotations

import argparse
import logging
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from scripts.data_fetcher import DataFetcher
from scripts.model import build_model
from scripts.preprocessor import DataPreprocessor
from scripts.runtime_config import ConfigManager
from scripts.train import train_model
from scripts.utils.artifact_utils import write_json_artifact
from scripts.utils.device_utils import log_runtime_context, resolve_execution_context
from scripts.utils.lightning_compat import seed_everything
from scripts.utils.logging_utils import configure_logging
from scripts.utils.model_registry import register_model_profile
from scripts.utils.repo_layout import apply_training_profile_layout, build_training_run_id, resolve_repo_path, serialize_repo_path
from scripts.utils.training_catalog import register_training_run, snapshot_training_run_artifacts
from scripts.utils.training_storage import build_training_profile_entry, mirror_training_logs
from scripts.utils.training_universe import (
    build_runtime_training_config,
    resolve_training_universe,
    save_runtime_profile,
)
from scripts.utils.transfer_weights import transfer_weights

configure_logging()
logger = logging.getLogger(__name__)


@dataclass
class TrainingExecutionResult:
    model: Any
    model_name: str
    run_id: str | None
    profile_path: str | None
    universe_mode: str
    single_ticker_symbol: str | None
    predefined_group_name: str | None
    requested_tickers: list[str]
    downloaded_tickers: list[str]
    discarded_tickers: list[str]
    final_tickers_used: list[str]
    years: int
    used_optuna: bool


def create_directories(config: dict):
    candidate_paths = [
        resolve_repo_path(config, config["paths"]["data_dir"]),
        resolve_repo_path(config, config["paths"]["artifacts_dir"]),
        resolve_repo_path(config, config["paths"]["training_artifacts_dir"]),
        resolve_repo_path(config, config["paths"]["cache_dir"]),
        resolve_repo_path(config, config["paths"]["tmp_dir"]),
        resolve_repo_path(config, config["paths"]["logs_root_dir"]),
        resolve_repo_path(config, config["paths"]["models_dir"]),
        resolve_repo_path(config, config["paths"]["normalizers_dir"]),
        resolve_repo_path(config, config["paths"]["config_dir"]),
        resolve_repo_path(config, config["paths"]["logs_dir"]),
        resolve_repo_path(config, config["paths"]["optuna_dir"]),
        resolve_repo_path(config, config["paths"]["runtime_profiles_dir"]),
        resolve_repo_path(config, config["paths"]["model_registry_path"]).parent,
        resolve_repo_path(config, config["paths"]["training_universes_path"]).parent,
        resolve_repo_path(config, config["paths"]["training_catalog_path"]).parent,
        resolve_repo_path(config, config["paths"]["benchmark_history_db_path"]).parent,
        resolve_repo_path(config, config["paths"]["benchmark_storage_dir"]),
        resolve_repo_path(config, config["paths"]["backtest_storage_dir"]),
        resolve_repo_path(config, config["data"]["raw_data_path"]).parent,
        resolve_repo_path(config, config["data"]["processed_data_path"]).parent,
        resolve_repo_path(config, config["data"]["train_processed_df_path"]).parent,
        resolve_repo_path(config, config["data"]["val_processed_df_path"]).parent,
    ]
    if config["paths"].get("training_profile_root"):
        candidate_paths.append(resolve_repo_path(config, config["paths"]["training_profile_root"]))
    if config["paths"].get("training_run_root"):
        candidate_paths.append(resolve_repo_path(config, config["paths"]["training_run_root"]))
    for directory in dict.fromkeys(path for path in candidate_paths if str(path) not in {"", "."}):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info("Directorio preparado: %s", directory)


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runtime = resolve_execution_context({"training": {"accelerator": "auto", "precision": "auto"}}, purpose="train")
    if runtime.hardware.cuda_available:
        torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True)
    logger.info("Semilla global fijada en %s", seed)


def _select_legacy_tickers(config: dict, regions: str, ticker_percentage: float) -> list[str]:
    regions_list = [region.strip().lower() for region in regions.split(",") if region.strip()]
    valid_regions = list(config["data"]["valid_regions"])
    selected_regions = [region for region in regions_list if region in valid_regions]
    if not selected_regions:
        logger.warning("Region invalida. Se usara 'global'.")
        selected_regions = ["global"]

    with open(config["data"]["tickers_file"], "r", encoding="utf-8") as handle:
        tickers_config = yaml.safe_load(handle) or {}

    if "all" in selected_regions:
        regions_to_iterate = list(tickers_config.get("tickers", {}).keys())
    else:
        regions_to_iterate = selected_regions

    all_tickers: list[str] = []
    for region in regions_to_iterate:
        region_tickers = list((tickers_config.get("tickers", {}).get(region) or {}).keys())
        if not region_tickers:
            continue
        num_to_select = max(1, int(len(region_tickers) * ticker_percentage))
        all_tickers.extend(random.sample(region_tickers, num_to_select))

    selected = list(dict.fromkeys(all_tickers))
    logger.info("Tickers seleccionados por modo legacy (%.1f%%): %s", ticker_percentage * 100.0, selected)
    return selected


def _persist_runtime_profile(config: dict) -> Path:
    profile_path = save_runtime_profile(config)
    logger.info("Perfil de runtime guardado en %s", profile_path)
    return profile_path


def _build_universe_report_path(config: dict) -> Path:
    raw_data_path = resolve_repo_path(config, config["data"]["raw_data_path"])
    return raw_data_path.with_name(f"{raw_data_path.stem}__integrity_report.json")


def _build_staged_raw_data_path(config: dict) -> Path:
    raw_data_path = resolve_repo_path(config, config["data"]["raw_data_path"])
    return raw_data_path.with_name(f"{raw_data_path.stem}__staging{raw_data_path.suffix}")


def _persist_universe_snapshot(config: dict, df, report) -> None:
    training_run = dict(config.get("training_run") or {})
    report_path = _build_universe_report_path(config)
    staged_raw_data_path = _build_staged_raw_data_path(config)
    payload = report.to_dict()
    payload["saved_at"] = training_run.get("trained_at")
    write_json_artifact(report_path, payload)
    training_run["universe_integrity_report_path"] = serialize_repo_path(config, report_path)

    if df is not None and not df.empty:
        staged_raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(staged_raw_data_path, index=False)
        training_run["raw_data_staging_path"] = serialize_repo_path(config, staged_raw_data_path)
        logger.info("Snapshot bruto del universo guardado en %s", staged_raw_data_path)

        if report.can_promote_canonical:
            canonical_path = Path(config["data"]["raw_data_path"])
            canonical_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(staged_raw_data_path, canonical_path)
            training_run["raw_data_promoted"] = True
            training_run["raw_data_artifact_path"] = str(canonical_path)
            logger.info("Dataset bruto promovido a ruta canonica: %s", canonical_path)
        else:
            training_run["raw_data_promoted"] = False
            training_run["raw_data_artifact_path"] = str(staged_raw_data_path)
            logger.warning(
                "El universo no se promociona como dataset canonico. decision=%s | summary=%s",
                report.decision,
                report.summary,
            )

    config["training_run"] = training_run


def _record_universe_report(config: dict, report, *, stage: str) -> None:
    training_run = dict(config.get("training_run") or {})
    report_payload = report.to_dict()
    training_run["universe_integrity"] = report_payload
    if stage == "download":
        training_run["download_universe_integrity"] = report_payload
    elif stage == "preprocessed":
        training_run["preprocessed_universe_integrity"] = report_payload
    training_run["downloaded_tickers"] = list(report.successful_tickers)
    training_run["discarded_tickers"] = list(report.discarded_tickers)
    training_run["discarded_details"] = dict(report.discarded_details)
    training_run["integrity_decision"] = report.decision
    training_run["integrity_summary"] = report.summary
    config["training_run"] = training_run


def _register_training_result(config: dict, profile_path: Path | None) -> None:
    resolved_profile_path = profile_path or Path(config.get("_meta", {}).get("config_path", ""))
    profile_entry = build_training_profile_entry(config, resolved_profile_path)
    register_model_profile(config, profile_entry, set_active=True)


def _prepare_training_run(config: dict) -> str:
    training_run = dict(config.get("training_run") or {})
    trained_at = datetime.now().replace(microsecond=0).isoformat()
    training_run["trained_at"] = trained_at
    run_id = build_training_run_id(config["model_name"], trained_at)
    training_run["run_id"] = run_id
    config["training_run"] = training_run
    apply_training_profile_layout(config, run_id=run_id)
    return run_id


def _resolve_training_config(
    config_path: str | None,
    years: int,
    regions: str,
    ticker_percentage: float,
    training_universe_mode: str | None,
    single_ticker_symbol: str | None,
    predefined_group_name: str | None,
) -> tuple[ConfigManager, dict, Path | None]:
    base_manager = ConfigManager(config_path)
    base_config = base_manager.config

    if training_universe_mode:
        resolved_universe = resolve_training_universe(
            base_config,
            mode=training_universe_mode,
            single_ticker_symbol=single_ticker_symbol,
            predefined_group_name=predefined_group_name,
        )
        runtime_config = build_runtime_training_config(base_config, resolved_universe, years)
        create_directories(runtime_config)
        profile_path = _persist_runtime_profile(runtime_config)
        manager = ConfigManager(str(profile_path))
        return manager, manager.config, profile_path

    config = base_config
    apply_training_profile_layout(config)
    config["data"]["years"] = years
    config["data"]["tickers"] = _select_legacy_tickers(config, regions, ticker_percentage)
    config["training_run"] = {
        "mode": "legacy_regions",
        "requested_tickers": list(config["data"]["tickers"]),
        "downloaded_tickers": [],
        "discarded_tickers": [],
        "discarded_details": {},
        "final_tickers_used": [],
        "dropped_after_preprocessing": [],
        "anchor_ticker": None,
        "universe_integrity": {},
        "download_universe_integrity": {},
        "preprocessed_universe_integrity": {},
        "universe_integrity_report_path": None,
        "raw_data_staging_path": None,
        "raw_data_promoted": False,
        "trained_at": None,
        "years": int(years),
        "prediction_horizon": int(config["model"]["max_prediction_length"]),
        "label": f"Legacy regions: {regions}",
        "description": "",
        "notes": "",
        "single_ticker_symbol": None,
        "predefined_group_name": None,
        "run_id": None,
        "profile_id": config["paths"].get("training_profile_id"),
        "profile_root": config["paths"].get("training_profile_root"),
        "active_root": config["paths"].get("training_active_root"),
    }
    create_directories(config)
    return base_manager, config, None


def _fetch_training_dataframe(config_manager: ConfigManager, config: dict, years: int) -> tuple[Any, list[str], list[str], Any]:
    fetcher = DataFetcher(config_manager, years=years)
    logger.info("Descargando datos de mercado...")

    df, report = fetcher.fetch_training_universe(config["data"]["tickers"])
    requested_tickers = report.requested_tickers
    downloaded_tickers = report.successful_tickers
    _record_universe_report(config, report, stage="download")
    config["data"]["tickers"] = list(downloaded_tickers)

    for ticker in report.discarded_tickers:
        detail = report.discarded_details.get(ticker, "Sin detalle")
        logger.warning("Ticker descartado durante la descarga: %s (%s)", ticker, detail)

    _persist_universe_snapshot(config, df, report)

    if not report.training_allowed:
        reasons = " | ".join(report.decision_reasons) if report.decision_reasons else report.summary
        raise ValueError(f"Universo de entrenamiento invalido: {report.decision}. {reasons}")
    if report.degraded:
        logger.warning("Se continuara con un universo degradado explicitamente permitido: %s", report.summary)

    return df, requested_tickers, downloaded_tickers, report


def start_training(
    config_path: str | None = None,
    regions: str = "global",
    years: int = 3,
    use_optuna: bool = False,
    continue_training: bool = True,
    use_transfer_learning: bool = False,
    old_model_filename: str | None = None,
    new_lr: float | None = None,
    ticker_percentage: float = 1.0,
    training_universe_mode: str | None = None,
    single_ticker_symbol: str | None = None,
    predefined_group_name: str | None = None,
):
    config_manager, config, profile_path = _resolve_training_config(
        config_path=config_path,
        years=years,
        regions=regions,
        ticker_percentage=ticker_percentage,
        training_universe_mode=training_universe_mode,
        single_ticker_symbol=single_ticker_symbol,
        predefined_group_name=predefined_group_name,
    )
    run_id = _prepare_training_run(config)
    create_directories(config)
    if profile_path is not None:
        profile_path = _persist_runtime_profile(config)
    set_global_seed(int(config["training"]["seed"]))
    training_runtime = resolve_execution_context(config, purpose="train")
    log_runtime_context(logger, "Inicio del entrenamiento", training_runtime)

    df, requested_tickers, downloaded_tickers, _ = _fetch_training_dataframe(config_manager, config, years)
    if df.empty:
        raise ValueError("No se han podido descargar datos de mercado")

    if profile_path is not None:
        _persist_runtime_profile(config)

    logger.info("Preprocesando datos...")
    model_name = config["model_name"]
    normalizers_base_dir = config["paths"].get("transfer_destination_normalizers_dir") or config["paths"]["normalizers_dir"]
    normalizers_path = resolve_repo_path(config, normalizers_base_dir) / f"{model_name}_normalizers.pkl"

    preprocessor = DataPreprocessor(config)
    dataset = preprocessor.process_data(mode="train", df=df)
    train_dataset, _ = dataset

    if profile_path is not None:
        _persist_runtime_profile(config)

    config["paths"]["model_save_path"] = str(Path(config["paths"]["models_dir"]) / f"{model_name}.pth")
    if use_transfer_learning and not continue_training:
        if not old_model_filename:
            raise ValueError("Falta old_model_filename para transfer learning")
        source_models_dir = config["paths"].get("transfer_source_models_dir") or config["paths"]["models_dir"]
        old_checkpoint_path = resolve_repo_path(config, source_models_dir) / old_model_filename
        if not old_checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint {old_checkpoint_path}")
        logger.info("Construyendo modelo para transfer learning...")
        new_model = build_model(train_dataset, config)
        transfer_weights(
            old_checkpoint_path=old_checkpoint_path,
            new_model=new_model,
            config=config,
            normalizers_path=normalizers_path,
            device=training_runtime.device_string,
        )
        continue_training = True

    logger.info("Entrenando modelo...")
    final_model = train_model(dataset, config, use_optuna=use_optuna, continue_training=continue_training, new_lr=new_lr)
    mirror_training_logs(config)

    run_manifest = None
    if profile_path is not None:
        profile_path = _persist_runtime_profile(config)
        run_manifest = snapshot_training_run_artifacts(config, profile_path=str(profile_path))
        register_training_run(config, run_manifest, active_profile_path=str(profile_path))
        _register_training_result(config, profile_path)

    logger.info("Entrenamiento finalizado. La aplicacion se lanza con `streamlit run streamlit_app.py`.")
    return TrainingExecutionResult(
        model=final_model,
        model_name=config["model_name"],
        run_id=run_id,
        profile_path=serialize_repo_path(config, profile_path) if profile_path else None,
        universe_mode=str(config.get("training_run", {}).get("mode", "legacy_regions")),
        single_ticker_symbol=config.get("training_run", {}).get("single_ticker_symbol"),
        predefined_group_name=config.get("training_run", {}).get("predefined_group_name"),
        requested_tickers=requested_tickers,
        downloaded_tickers=downloaded_tickers,
        discarded_tickers=list(config.get("training_run", {}).get("discarded_tickers", [])),
        final_tickers_used=list(config.get("training_run", {}).get("final_tickers_used", downloaded_tickers)),
        years=years,
        used_optuna=use_optuna,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entrena el predictor bursatil TFT")
    parser.add_argument("--config-path", default=None, help="Ruta alternativa al archivo de configuracion")
    parser.add_argument("--regions", default="global", help="Regiones separadas por coma")
    parser.add_argument("--years", type=int, default=3, help="Numero de anos historicos")
    parser.add_argument("--use-optuna", action="store_true", help="Activa Optuna")
    parser.add_argument("--from-scratch", action="store_true", help="No continuar desde checkpoint")
    parser.add_argument("--use-transfer-learning", action="store_true", help="Transfiere pesos desde otro checkpoint")
    parser.add_argument("--old-model-filename", default=None, help="Checkpoint origen para transfer learning")
    parser.add_argument("--new-lr", type=float, default=None, help="Nueva tasa de aprendizaje al continuar")
    parser.add_argument("--ticker-percentage", type=float, default=1.0, help="Porcentaje de tickers a usar en rango 0.3-1.0")
    parser.add_argument(
        "--training-universe-mode",
        default=None,
        choices=["single_ticker", "predefined_group"],
        help="Modo moderno de seleccion del universo de entrenamiento",
    )
    parser.add_argument("--single-ticker-symbol", default=None, help="Ticker unico para entrenar")
    parser.add_argument("--predefined-group-name", default=None, help="Nombre del grupo predefinido")
    return parser


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    ticker_percentage = min(max(args.ticker_percentage, 0.3), 1.0)
    if args.years < 3:
        parser.error("--years debe ser 3 o mayor")
    return start_training(
        config_path=args.config_path,
        regions=args.regions,
        years=args.years,
        use_optuna=args.use_optuna,
        continue_training=not args.from_scratch,
        use_transfer_learning=args.use_transfer_learning,
        old_model_filename=args.old_model_filename,
        new_lr=args.new_lr,
        ticker_percentage=ticker_percentage,
        training_universe_mode=args.training_universe_mode,
        single_ticker_symbol=args.single_ticker_symbol,
        predefined_group_name=args.predefined_group_name,
    )


if __name__ == "__main__":
    main()
