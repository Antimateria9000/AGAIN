import logging

import yaml

from scripts.runtime_config import ConfigManager

logger = logging.getLogger(__name__)


def load_config():
    return ConfigManager().config


def load_tickers_and_names(config):
    try:
        with open(config['data']['tickers_file'], 'r', encoding='utf-8') as f:
            tickers_config = yaml.safe_load(f)
        ticker_dict = {}
        for region in tickers_config['tickers']:
            for ticker, name in tickers_config['tickers'][region].items():
                ticker_dict[ticker] = name
        return ticker_dict
    except Exception as e:
        logger.error(f"Error al cargar tickers y nombres: {e}")
        return {}


def load_benchmark_tickers(config):
    try:
        with open(config['data']['benchmark_tickers_file'], 'r', encoding='utf-8') as f:
            tickers_config = yaml.safe_load(f)
        all_tickers = []
        for region in tickers_config['tickers'].values():
            all_tickers.extend(list(region.keys()))
        return list(dict.fromkeys(all_tickers))
    except KeyError as e:
        logger.error(f"Falta una clave de configuracion: {e}")
        raise ValueError(f"Configuracion invalida: falta la clave {e} en config.yaml")
    except Exception as e:
        logger.error(f"Error al cargar benchmark_tickers.yaml: {e}")
        raise


def save_history_range(config, history_range_days):
    """Guarda el rango historico visible en el archivo de configuracion activo."""
    try:
        config_path = None
        if isinstance(config, str):
            config_path = config
        config_manager = ConfigManager(config_path) if config_path else ConfigManager()
        config_manager.config['prediction']['history_range_days'] = history_range_days
        with open(config_manager.config_path, 'w', encoding='utf-8') as f:
            yaml.safe_dump(config_manager.config, f, sort_keys=False, allow_unicode=True)
        logger.info(f"history_range_days actualizado a {history_range_days}")
    except Exception as e:
        logger.error(f"Error al guardar history_range_days: {e}")
        raise
