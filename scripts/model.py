from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from pytorch_forecasting import TemporalFusionTransformer

from scripts.config_manager import ConfigManager
from scripts.utils.device_utils import resolve_execution_context
from scripts.utils.lightning_compat import LightningModule
from scripts.utils.model_config import HyperparamFactory, ModelConfig
from scripts.utils.validation_utils import create_validation_plot, log_validation_details

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision("medium")
if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
    torch.backends.cuda.matmul.allow_tf32 = True



def move_to_device(obj: Any, device: torch.device) -> Any:
    if isinstance(obj, torch.Tensor):
        if obj.device == device:
            return obj
        return obj.to(device, non_blocking=device.type == "cuda")
    if isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    return obj


class CustomTemporalFusionTransformer(LightningModule):
    def __init__(self, dataset, config: Dict[str, Any], hyperparams: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.model_config = ModelConfig(config)
        self.hyperparams = hyperparams if hyperparams else self.model_config.default_hyperparams
        self.model_name = config["model_name"]
        self.dataset = dataset
        self.config_manager = ConfigManager(config.get("_meta", {}).get("config_path"))
        self.batch_size = int(config["training"]["batch_size"])
        self.runtime = resolve_execution_context(config, purpose="train")
        self._load_normalizers()
        self._initialize_model(dataset)
        self._save_hyperparameters()
        self.val_batch_count = 0
        self.enable_detailed_validation = config["validation"]["enable_detailed_validation"]
        self.max_val_batches_to_log = config["validation"]["max_validation_batches_to_log"]
        self.save_plots = config["validation"]["save_plots"]
        self.max_plots_per_epoch = config["validation"]["max_plots_per_epoch"]
        self.logs_dir = config["paths"]["logs_dir"]
        self.plot_count = 0
        self.debug = config["validation"]["debug"]

    def _load_normalizers(self):
        try:
            self.normalizers = self.config_manager.load_normalizers(self.model_name)
            logger.info("Normalizadores cargados para %s", self.model_name)
        except Exception as exc:
            logger.error("Error al cargar normalizadores: %s", exc)
            self.normalizers = {}

    def _initialize_model(self, dataset):
        filtered_params = self.model_config.get_filtered_params(self.hyperparams)
        logger.info("Parametros usados por TemporalFusionTransformer: %s", filtered_params)
        self.model = TFTWithTransfer.from_dataset(dataset, **filtered_params)

    def _save_hyperparameters(self):
        hparams_to_save = {
            key: value
            for key, value in self.hyperparams.items()
            if key not in {"loss", "logging_metrics"}
        }
        self.save_hyperparameters(hparams_to_save)

    def on_fit_start(self):
        self.model.to(self.device)

    def forward(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = move_to_device(x, self.device)
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    def predict(self, data, **kwargs):
        start_time = time.time()
        self.eval()
        device = self.runtime.torch_device
        logger.info("Prediccion en dispositivo: %s", device)
        if isinstance(data, torch.utils.data.DataLoader):
            predictions = self.model.predict(data, **kwargs)
        else:
            predictions = self.model.predict(move_to_device(data, device), **kwargs)
        logger.info("Forma de prediccion: %s", predictions.output.shape)
        logger.info("Tiempo de prediccion: %.3fs", time.time() - start_time)
        return predictions

    def interpret_output(self, x: Dict[str, torch.Tensor], **kwargs) -> Dict[str, Any]:
        x = move_to_device(x, self.device)
        try:
            self.model.eval()
            with torch.no_grad():
                full_output = self.model(x)
                if isinstance(full_output, dict):
                    return self.model.interpret_output(full_output, **kwargs)
                if hasattr(self.model, "_forward_full"):
                    full_output = self.model._forward_full(x)
                else:
                    full_output = self.model.forward(x)
                return self.model.interpret_output(full_output, **kwargs)
        except Exception as exc:
            logger.error("Error en interpret_output: %s", exc)
            raise

    def _shared_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int, stage: str) -> torch.Tensor:
        x, y = batch
        x = move_to_device(x, self.device)
        y_target = move_to_device(y[0], self.device)

        if torch.isnan(y_target).any() or torch.isinf(y_target).any():
            raise ValueError(f"y_target contiene NaN o Inf en batch {batch_idx}")

        y_hat = self(x)
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            raise ValueError(f"y_hat contiene NaN o Inf en batch {batch_idx}")
        loss = self.model.loss(y_hat, y_target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss no finita en batch {batch_idx}: {loss}")

        if stage == "val":
            y_hat_median = y_hat[:, :, 1] if y_hat.dim() == 3 else y_hat
            mape = torch.mean(torch.abs((y_target - y_hat_median) / (y_target + 1e-10))) * 100
            self.log(f"{stage}_mape", mape, on_step=False, on_epoch=True, prog_bar=True, batch_size=x["encoder_cont"].size(0))
            direction_pred = torch.sign(y_hat_median)
            direction_true = torch.sign(y_target)
            directional_accuracy = (direction_pred == direction_true).float().mean() * 100
            self.log(f"{stage}_directional_accuracy", directional_accuracy, on_step=False, on_epoch=True, prog_bar=True, batch_size=x["encoder_cont"].size(0))

        batch_size = x["encoder_cont"].size(0)
        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)

        try:
            l2_norm = sum(parameter.pow(2).sum() for parameter in self.parameters() if parameter.requires_grad).sqrt().item()
            self.log(f"{stage}_l2_norm", l2_norm, on_step=True, on_epoch=True, prog_bar=False, batch_size=batch_size)
        except Exception as exc:
            logger.warning("No se puede calcular l2_norm: %s", exc)

        if stage == "val":
            if self.enable_detailed_validation and self.val_batch_count < self.max_val_batches_to_log:
                try:
                    log_validation_details(
                        x,
                        y_hat,
                        y_target,
                        batch_idx,
                        self.normalizers,
                        self.dataset,
                        self.save_plots,
                        self.plot_count,
                        self.max_plots_per_epoch,
                        self.logs_dir,
                        self.current_epoch,
                    )
                    self.val_batch_count += 1
                except Exception as exc:
                    logger.error("Error al registrar detalles de validacion: %s", exc)

            if self.save_plots and self.plot_count < self.max_plots_per_epoch:
                try:
                    relative_returns_normalizer = self.normalizers.get("Relative_Returns") or self.dataset.target_normalizer
                    if relative_returns_normalizer:
                        y_hat_denorm = relative_returns_normalizer.inverse_transform(y_hat.float().cpu())
                        y_target_denorm = relative_returns_normalizer.inverse_transform(y_target.float().cpu())
                        create_validation_plot(y_hat_denorm, y_target_denorm, batch_idx, self.logs_dir, self.current_epoch)
                        self.plot_count += 1
                    else:
                        logger.warning("No existe normalizador para Relative_Returns")
                except Exception as exc:
                    logger.error("Error al crear grafico de validacion: %s", exc)

        return loss

    def on_validation_epoch_end(self) -> None:
        val_l2_norm = self.trainer.callback_metrics.get("val_l2_norm")
        if val_l2_norm is not None:
            logger.info("Validation epoch end: val_l2_norm = %.4f", val_l2_norm)

        optimizer = self.optimizers()
        if optimizer is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.info("Validation epoch end: learning_rate = %.6f", current_lr)
        self.val_batch_count = 0
        self.plot_count = 0

    def configure_optimizers(self) -> Dict[str, Any]:
        learning_rate = self.hyperparams.get("learning_rate", self.model_config.config["model"]["learning_rate"])
        weight_decay = self.model_config.config["training"]["weight_decay"]
        optimizer = torch.optim.AdamW(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=self.model_config.config["training"]["reduce_lr_patience"],
            factor=self.model_config.config["training"]["reduce_lr_factor"],
            mode="min",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }

    def training_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="train")

    def validation_step(self, batch: Tuple[Dict[str, torch.Tensor], List[torch.Tensor]], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, batch_idx, stage="val")


class TFTWithTransfer(TemporalFusionTransformer):
    def transfer_batch_to_device(self, batch, device, dataloader_idx=0):
        return move_to_device(batch, device)



def build_model(dataset, config: Dict[str, Any], trial=None, hyperparams: Optional[Dict[str, Any]] = None) -> CustomTemporalFusionTransformer:
    model_config = ModelConfig(config)
    if trial:
        hyperparams = HyperparamFactory.from_trial(trial, model_config)
    elif hyperparams:
        hyperparams = HyperparamFactory.from_checkpoint(hyperparams, model_config)
    else:
        hyperparams = model_config.default_hyperparams
    return CustomTemporalFusionTransformer(dataset, config, hyperparams)
