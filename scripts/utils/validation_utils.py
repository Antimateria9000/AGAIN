import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from scripts.utils.data_schema import NUMERIC_FEATURES
from scripts.utils.prediction_utils import accumulate_quantile_price_paths, denormalize_logged_close

logger = logging.getLogger(__name__)


def convert_to_prices(y_hat_denorm, y_target_denorm, last_close_price, batch_idx):
    if y_hat_denorm.dim() == 3:
        pred_median, pred_lower, pred_upper = accumulate_quantile_price_paths(
            last_close_price,
            y_hat_denorm[0, :, 1],
            y_hat_denorm[0, :, 0],
            y_hat_denorm[0, :, 2],
        )
    else:
        pred_median = accumulate_quantile_price_paths(last_close_price, y_hat_denorm[0, :])[0]
        pred_lower = pred_median
        pred_upper = pred_median

    target_prices = accumulate_quantile_price_paths(last_close_price, y_target_denorm[0, :])[0]
    logger.info(
        "Validation batch %s - precios reconstruidos: mediana=%s lower=%s upper=%s target=%s",
        batch_idx,
        [f"{value:.2f}" for value in pred_median[:5]],
        [f"{value:.2f}" for value in pred_lower[:5]],
        [f"{value:.2f}" for value in pred_upper[:5]],
        [f"{value:.2f}" for value in target_prices[:5]],
    )


def create_validation_plot(y_hat_denorm, y_target_denorm, batch_idx, logs_dir, current_epoch):
    plt.figure(figsize=(10, 6))
    time_steps = np.arange(y_hat_denorm.shape[1])
    if y_hat_denorm.dim() == 3:
        y_hat_median = y_hat_denorm[0, :, 1].numpy()
        y_hat_lower = y_hat_denorm[0, :, 0].numpy()
        y_hat_upper = y_hat_denorm[0, :, 2].numpy()
    else:
        y_hat_median = y_hat_denorm[0, :].numpy()
        y_hat_lower = y_hat_median
        y_hat_upper = y_hat_median
    y_target = y_target_denorm[0, :].numpy()
    plt.plot(time_steps, y_target, label='Real', color='blue', marker='o')
    plt.plot(time_steps, y_hat_median, label='Prediccion mediana', color='red', linestyle='--', marker='x')
    plt.fill_between(time_steps, y_hat_lower, y_hat_upper, color='red', alpha=0.1, label='Banda 10-90')
    plt.title(f'Prediccion vs real - batch {batch_idx}')
    plt.xlabel('Paso temporal')
    plt.ylabel('Retorno relativo')
    plt.legend()
    plt.grid(True)
    os.makedirs(logs_dir, exist_ok=True)
    plot_path = os.path.join(logs_dir, f'ValPlot_Batch_{batch_idx}_epoch_{current_epoch}.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"Se ha guardado el grafico de validacion en {plot_path}")


def log_validation_details(x, y_hat, y_target, batch_idx, normalizers, dataset, save_plots, plot_count, max_plots_per_epoch, logs_dir, current_epoch):
    try:
        y_hat_denorm = y_hat.float().cpu()
        y_target_denorm = y_target.float().cpu()
        if 'encoder_cont' in x:
            encoder_cont = x['encoder_cont'][0].cpu()
            close_normalizer = normalizers.get('Close')
            feature_order = list(getattr(dataset, 'reals', NUMERIC_FEATURES))
            if close_normalizer is not None and 'Close' in feature_order:
                close_idx = feature_order.index('Close')
                last_close_norm = float(encoder_cont[-1, close_idx])
                last_close_price = denormalize_logged_close(close_normalizer, last_close_norm)
                logger.info(f"Ultimo cierre del batch: {last_close_price:.2f}")
                convert_to_prices(y_hat_denorm, y_target_denorm, last_close_price, batch_idx)
            else:
                logger.warning("No se puede localizar la columna Close en dataset.reals")
        else:
            logger.warning("El batch no incluye encoder_cont")
    except Exception as e:
        logger.error(f"Error durante la validacion detallada: {e}")
        return

    if save_plots and plot_count < max_plots_per_epoch:
        try:
            create_validation_plot(y_hat_denorm, y_target_denorm, batch_idx, logs_dir, current_epoch)
        except Exception as e:
            logger.error(f"Error al generar el grafico de validacion: {e}")
