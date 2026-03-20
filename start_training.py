from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml

from scripts.data_fetcher import DataFetcher
from scripts.model import build_model
from scripts.preprocessor import DataPreprocessor
from scripts.runtime_config import ConfigManager
from scripts.train import train_model
from scripts.utils.lightning_compat import seed_everything
from scripts.utils.logging_utils import configure_logging
from scripts.utils.transfer_weights import transfer_weights

configure_logging()
logger = logging.getLogger(__name__)


def create_directories(config: dict):
    candidate_paths = [
        Path(config["paths"]["data_dir"]),
        Path(config["paths"]["models_dir"]),
        Path(config["paths"]["normalizers_dir"]),
        Path(config["paths"]["config_dir"]),
        Path(config["paths"]["logs_dir"]),
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    seed_everything(seed, workers=True)
    logger.info("Semilla global fijada en %s", seed)



def _select_tickers(config: dict, regions: str, ticker_percentage: float) -> list[str]:
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
    logger.info("Tickers seleccionados (%.1f%%): %s", ticker_percentage * 100.0, selected)
    return selected



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
):
    config_manager = ConfigManager(config_path)
    config = config_manager.config
    create_directories(config)
    set_global_seed(int(config["training"]["seed"]))

    config["data"]["years"] = years
    config["data"]["tickers"] = _select_tickers(config, regions, ticker_percentage)
    fetcher = DataFetcher(config_manager, years=years)
    logger.info("Descargando datos de mercado...")
    df = fetcher.fetch_global_stocks(region=None)
    if df.empty:
        raise ValueError("No se han podido descargar datos de mercado")

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
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        continue_training = True

    logger.info("Entrenando modelo...")
    final_model = train_model(dataset, config, use_optuna=use_optuna, continue_training=continue_training, new_lr=new_lr)
    logger.info("Entrenamiento finalizado. La aplicacion se lanza con `streamlit run app/app.py`.")
    return final_model



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
    )


if __name__ == "__main__":
    main()
