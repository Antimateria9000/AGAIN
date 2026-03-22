from __future__ import annotations

import argparse
import logging
import random
import shutil
from dataclasses import dataclass
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
from scripts.utils.device_utils import log_runtime_context, resolve_execution_context
from scripts.utils.lightning_compat import seed_everything
from scripts.utils.logging_utils import configure_logging
from scripts.utils.model_registry import register_model_profile
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
        Path(config["paths"]["data_dir"]),
        Path(config["paths"]["models_dir"]),
        Path(config["paths"]["normalizers_dir"]),
        Path(config["paths"]["config_dir"]),
        Path(config["paths"]["logs_dir"]),
        Path(config["paths"]["runtime_profiles_dir"]),
        Path(config["paths"]["model_registry_path"]).parent,
        Path(config["paths"]["training_universes_path"]).parent,
        Path(config["paths"]["benchmark_history_db_path"]).parent,
        Path(config["data"]["raw_data_path"]).parent,
        Path(config["data"]["processed_data_path"]).parent,
        Path(config["data"]["train_processed_df_path"]).parent,
        Path(config["data"]["val_processed_df_path"]).parent,
    ]
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


def _register_training_result(config: dict, profile_path: Path | None) -> None:
    training_run = dict(config.get("training_run") or {})
    profile_entry = {
        "model_name": config["model_name"],
        "base_model_name": config.get("base_model_name", config["model_name"]),
        "profile_path": str(profile_path) if profile_path else str(config.get("_meta", {}).get("config_path", "")),
        "label": training_run.get("label") or config["model_name"],
        "description": training_run.get("description", ""),
        "notes": training_run.get("notes", ""),
        "universe_mode": training_run.get("mode", "legacy_regions"),
        "single_ticker_symbol": training_run.get("single_ticker_symbol"),
        "predefined_group_name": training_run.get("predefined_group_name"),
        "requested_tickers": list(training_run.get("requested_tickers", [])),
        "downloaded_tickers": list(training_run.get("downloaded_tickers", [])),
        "discarded_tickers": list(training_run.get("discarded_tickers", [])),
        "final_tickers_used": list(training_run.get("final_tickers_used", [])),
        "created_at": training_run.get("trained_at"),
        "last_trained_at": training_run.get("trained_at"),
        "prediction_horizon": training_run.get("prediction_horizon"),
        "years": training_run.get("years"),
    }
    register_model_profile(config, profile_entry, set_active=True)


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
        "trained_at": None,
        "years": int(years),
        "prediction_horizon": int(config["model"]["max_prediction_length"]),
        "label": f"Legacy regions: {regions}",
        "description": "",
        "notes": "",
        "single_ticker_symbol": None,
        "predefined_group_name": None,
    }
    create_directories(config)
    return base_manager, config, None


def _fetch_training_dataframe(config_manager: ConfigManager, config: dict, years: int) -> tuple[Any, list[str], list[str]]:
    fetcher = DataFetcher(config_manager, years=years)
    logger.info("Descargando datos de mercado...")

    training_mode = config.get("training_run", {}).get("mode")
    if training_mode in {"single_ticker", "predefined_group"}:
        df, report = fetcher.fetch_training_universe(config["data"]["tickers"])
        requested_tickers = report.requested_tickers
        downloaded_tickers = report.successful_tickers
        discarded_tickers = report.discarded_tickers
        config["training_run"]["downloaded_tickers"] = list(downloaded_tickers)
        config["training_run"]["discarded_tickers"] = list(discarded_tickers)
        config["training_run"]["discarded_details"] = dict(report.discarded_details)
        config["data"]["tickers"] = list(downloaded_tickers)

        if training_mode == "single_ticker" and not downloaded_tickers:
            ticker = config["training_run"].get("single_ticker_symbol") or "ticker solicitado"
            raise ValueError(f"No se han podido descargar datos validos para {ticker}")

        if training_mode == "predefined_group":
            minimum_group_tickers = int(config["training_universe"]["minimum_group_tickers"])
            if len(downloaded_tickers) < minimum_group_tickers:
                raise ValueError(
                    f"El grupo seleccionado se queda sin universo suficiente tras la descarga: "
                    f"se han obtenido {len(downloaded_tickers)} tickers y se requieren al menos {minimum_group_tickers}"
                )
            for ticker in discarded_tickers:
                detail = report.discarded_details.get(ticker, "Sin detalle")
                logger.warning("Ticker descartado durante la descarga: %s (%s)", ticker, detail)

        return df, requested_tickers, downloaded_tickers

    df = fetcher.fetch_global_stocks(region=None)
    if df.empty:
        raise ValueError("No se han podido descargar datos de mercado")
    requested = list(config["data"]["tickers"])
    config["training_run"]["downloaded_tickers"] = requested
    return df, requested, requested


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
    set_global_seed(int(config["training"]["seed"]))
    training_runtime = resolve_execution_context(config, purpose="train")
    log_runtime_context(logger, "Inicio del entrenamiento", training_runtime)

    df, requested_tickers, downloaded_tickers = _fetch_training_dataframe(config_manager, config, years)
    if df.empty:
        raise ValueError("No se han podido descargar datos de mercado")

    if profile_path is not None:
        _persist_runtime_profile(config)

    data_path = Path(config["data"]["raw_data_path"])
    data_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_path, index=False)
    logger.info("Datos guardados en %s", data_path)

    logger.info("Preprocesando datos...")
    model_name = config["model_name"]
    normalizers_path = Path(config["paths"]["models_dir"]) / "normalizers" / f"{model_name}_normalizers.pkl"

    if use_transfer_learning and old_model_filename:
        old_model_name = old_model_filename.replace(".pth", "")
        old_normalizers_path = Path(config["paths"]["models_dir"]) / "normalizers" / f"{old_model_name}_normalizers.pkl"
        if old_normalizers_path.exists():
            shutil.copy2(old_normalizers_path, normalizers_path)
            old_checksum = old_normalizers_path.with_name(f"{old_normalizers_path.name}.sha256")
            if old_checksum.exists():
                shutil.copy2(old_checksum, normalizers_path.with_name(f"{normalizers_path.name}.sha256"))
        else:
            logger.warning("No existen normalizadores previos en %s; se regeneraran", old_normalizers_path)

    preprocessor = DataPreprocessor(config)
    dataset = preprocessor.process_data(mode="train", df=df)
    train_dataset, _ = dataset

    if profile_path is not None:
        _persist_runtime_profile(config)

    config["paths"]["model_save_path"] = str(Path(config["paths"]["models_dir"]) / f"{model_name}.pth")
    if use_transfer_learning and not continue_training:
        if not old_model_filename:
            raise ValueError("Falta old_model_filename para transfer learning")
        old_checkpoint_path = Path(config["paths"]["models_dir"]) / old_model_filename
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

    if profile_path is not None:
        _persist_runtime_profile(config)
        _register_training_result(config, profile_path)

    logger.info("Entrenamiento finalizado. La aplicacion se lanza con `streamlit run streamlit_app.py`.")
    return TrainingExecutionResult(
        model=final_model,
        model_name=config["model_name"],
        profile_path=str(profile_path) if profile_path else None,
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
