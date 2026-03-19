import asyncio
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import yaml

from scripts.data_fetcher import DataFetcher
from scripts.model import build_model
from scripts.preprocessor import DataPreprocessor
from scripts.runtime_config import ConfigManager
from scripts.train import train_model
from scripts.utils.logging_utils import configure_logging
from scripts.utils.transfer_weights import transfer_weights

configure_logging()
logger = logging.getLogger(__name__)



def create_directories(config: dict):
    """Crea directorios a partir de la configuracion real y no de una lista hardcodeada."""
    candidate_paths = [
        Path(config['paths']['data_dir']),
        Path(config['paths']['models_dir']),
        Path(config['paths']['normalizers_dir']),
        Path(config['paths']['config_dir']),
        Path(config['paths']['logs_dir']),
        Path(config['data']['raw_data_path']).parent,
        Path(config['data']['processed_data_path']).parent,
        Path(config['data']['train_processed_df_path']).parent,
        Path(config['data']['val_processed_df_path']).parent,
    ]
    for directory in dict.fromkeys(path for path in candidate_paths if str(path) not in {'', '.'}):
        directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Directorio preparado: {directory}")



def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed, workers=True)
    logger.info(f"Semilla global fijada en {seed}")


async def start_training(
    regions: str = 'global',
    years: int = 3,
    use_optuna: bool = False,
    continue_training: bool = True,
    use_transfer_learning: bool = False,
    old_model_filename: str = None,
    new_lr: float = None,
    ticker_percentage: float = 1.0,
):
    try:
        config_manager = ConfigManager()
        config = config_manager.config
        create_directories(config)
        set_global_seed(int(config['training']['seed']))
        config['data']['years'] = years
        logger.info(f"Anos de historico configurados: {years}")

        regions_list = [region.strip().lower() for region in regions.split(',')]
        valid_regions = list(config['data']['valid_regions'])
        selected_regions = [region for region in regions_list if region in valid_regions]
        if not selected_regions:
            logger.warning("Region invalida. Se usara 'global'.")
            selected_regions = ['global']
        logger.info(f"Regiones seleccionadas: {', '.join(selected_regions)}")

        fetcher = DataFetcher(config_manager, years=years)
        all_tickers = []
        with open(config['data']['tickers_file'], 'r', encoding='utf-8') as f:
            tickers_config = yaml.safe_load(f)

        if 'all' in selected_regions:
            regions_to_iterate = list(tickers_config['tickers'].keys())
        else:
            regions_to_iterate = selected_regions

        for region in regions_to_iterate:
            region_tickers = list(tickers_config['tickers'][region].keys())
            num_to_select = max(1, int(len(region_tickers) * ticker_percentage))
            selected = random.sample(region_tickers, num_to_select)
            all_tickers.extend(selected)

        all_tickers = list(dict.fromkeys(all_tickers))
        logger.info(f"Tickers seleccionados ({ticker_percentage * 100:.1f}%): {all_tickers}")
        config['data']['tickers'] = all_tickers

        logger.info('Descargando datos de mercado...')
        df = await fetcher.fetch_global_stocks(region=None)
        if df.empty:
            raise ValueError('No se han podido descargar datos de mercado')

        data_path = Path(config['data']['raw_data_path'])
        df.to_csv(data_path, index=False)
        logger.info(f"Datos guardados en {data_path}")

        logger.info('Preprocesando datos...')
        model_name = config['model_name']
        normalizers_path = Path(config['paths']['models_dir']) / 'normalizers' / f"{model_name}_normalizers.pkl"
        logger.info(f"Ruta de normalizadores: {normalizers_path}")

        if use_transfer_learning and old_model_filename:
            old_model_name = old_model_filename.replace('.pth', '')
            old_normalizers_path = Path(config['paths']['models_dir']) / 'normalizers' / f"{old_model_name}_normalizers.pkl"
            if old_normalizers_path.exists():
                shutil.copy2(old_normalizers_path, normalizers_path)
                logger.info(f"Normalizadores copiados de {old_normalizers_path} a {normalizers_path}")
            else:
                logger.warning(f"No existen normalizadores previos en {old_normalizers_path}; se regeneraran")

        preprocessor = DataPreprocessor(config)
        dataset = preprocessor.process_data(mode='train', df=df)
        train_dataset, _ = dataset

        if use_transfer_learning and not continue_training:
            models_dir = Path(config['paths']['models_dir'])
            old_checkpoint_path = models_dir / old_model_filename
            if not old_checkpoint_path.exists():
                raise FileNotFoundError(f"No existe el checkpoint {old_checkpoint_path}")

            logger.info('Construyendo modelo para transfer learning...')
            new_model = build_model(train_dataset, config)
            new_model, config = transfer_weights(
                old_checkpoint_path=old_checkpoint_path,
                new_model=new_model,
                config=config,
                normalizers_path=normalizers_path,
                device='cuda' if torch.cuda.is_available() else 'cpu',
            )
            logger.info(f"Pesos transferidos y guardados en {config['paths']['model_save_path']}")
            config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{model_name}.pth")
            final_model = train_model(dataset, config, use_optuna=use_optuna, continue_training=True, new_lr=new_lr)
        else:
            logger.info('Entrenando modelo...')
            config['paths']['model_save_path'] = str(Path(config['paths']['models_dir']) / f"{model_name}.pth")
            final_model = train_model(dataset, config, use_optuna=use_optuna, continue_training=continue_training, new_lr=new_lr)

        logger.info('Entrenamiento finalizado. La aplicacion se lanza con `streamlit run app/app.py`.')
        return final_model
    except Exception as e:
        logger.error(f"Error durante el entrenamiento: {e}")
        raise


if __name__ == '__main__':
    config_manager = ConfigManager()
    config = config_manager.config
    model_name = config['model_name']
    model_path = Path(config['paths']['models_dir']) / f"{model_name}.pth"

    current_lr = config['model']['learning_rate']
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            current_lr = checkpoint['hyperparams']['learning_rate']
            logger.info(f"Tasa de aprendizaje leida del checkpoint: {current_lr}")
        except Exception as e:
            logger.warning(f"No se ha podido leer la tasa de aprendizaje del checkpoint: {e}. Se usa la de config: {current_lr}")

    regions = input(f"Elige region o regiones ({', '.join(config['data']['valid_regions'])}) [por defecto: global]: ").lower() or 'global'
    ticker_percentage_input = input('Porcentaje de tickers a usar (30-100) [por defecto: 100]: ').lower() or '100'
    try:
        ticker_percentage = float(ticker_percentage_input) / 100.0
        if ticker_percentage < 0.3 or ticker_percentage > 1.0:
            logger.warning('El porcentaje debe estar entre 30 y 100. Se usara 100%.')
            ticker_percentage = 1.0
    except ValueError as e:
        logger.error(f"Valor de porcentaje invalido: {e}. Se usara 100%.")
        ticker_percentage = 1.0

    years_input = input('Numero de anos historicos [minimo: 3, por defecto: 3]: ').lower() or '3'
    try:
        years = int(years_input)
        if years < 3:
            logger.warning('El minimo son 3 anos. Se usara 3.')
            years = 3
    except ValueError as e:
        logger.error(f"Valor de anos invalido: {e}. Se usara 3.")
        years = 3

    use_optuna = (input('Usar Optuna? (si/no) [por defecto: no]: ').lower() or 'no') == 'si'
    continue_training = (input('Continuar desde checkpoint? (si/no) [por defecto: si]: ').lower() or 'si') != 'no'

    custom_lr = None
    if continue_training:
        change_lr_input = input(f"La tasa de aprendizaje actual es {current_lr}. Quieres cambiarla? (si/no) [por defecto: no]: ").lower() or 'no'
        if change_lr_input == 'si':
            new_lr_str = input('Nueva tasa de aprendizaje (por ejemplo 0.0001): ').strip()
            try:
                custom_lr = float(new_lr_str)
                logger.info(f"Nueva tasa de aprendizaje configurada: {custom_lr}")
            except ValueError:
                logger.error('Tasa de aprendizaje invalida. Se mantiene la actual.')

    use_transfer_learning = False
    old_model_filename = None
    if not continue_training:
        use_transfer_learning = (input('Usar transfer learning desde otro modelo? (si/no) [por defecto: no]: ').lower() or 'no') == 'si'
        if use_transfer_learning:
            old_model_filename = input('Nombre del checkpoint origen en models (por ejemplo model.pth): ').strip()
            if not old_model_filename:
                raise ValueError('El nombre del checkpoint origen no puede estar vacio')

    asyncio.run(start_training(regions, years, use_optuna, continue_training, use_transfer_learning, old_model_filename, new_lr=custom_lr, ticker_percentage=ticker_percentage))
