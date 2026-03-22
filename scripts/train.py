from __future__ import annotations

import logging
import shutil
from pathlib import Path

import optuna
import torch
import torch.backends.cudnn as cudnn
from pytorch_forecasting import TimeSeriesDataSet

from scripts.model import build_model
from scripts.utils.artifact_utils import ensure_relative_to, verify_checksum, write_checksum, write_metadata
from scripts.utils.batch_size_estimator import estimate_batch_size
from scripts.utils.data_schema import build_artifact_metadata, build_schema_hash
from scripts.utils.device_utils import log_runtime_context, resolve_execution_context
from scripts.utils.lightning_compat import CSVLogger, EarlyStopping, pl

logger = logging.getLogger(__name__)
cudnn.benchmark = False
cudnn.deterministic = True


class CustomModelCheckpoint(pl.callbacks.Callback):
    def __init__(self, monitor: str, save_path: str, mode: str = "min"):
        super().__init__()
        self.monitor = monitor
        self.save_path = Path(save_path)
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        is_better = (self.mode == "min" and current_score < self.best_score) or (self.mode == "max" and current_score > self.best_score)
        if not is_better:
            return
        self.best_score = current_score
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = _build_checkpoint_payload(pl_module, pl_module.model_config.config)
        torch.save(payload, self.save_path)
        write_metadata(self.save_path, payload["metadata"])
        write_checksum(self.save_path)
        logger.info("Nuevo mejor checkpoint en %s con %s=%s", self.save_path, self.monitor, current_score)



def _build_checkpoint_payload(model, config: dict) -> dict:
    metadata = build_artifact_metadata(config)
    return {
        "state_dict": model.state_dict(),
        "hyperparams": dict(model.hparams),
        "metadata": metadata,
    }



def _validate_checkpoint_metadata(config: dict, checkpoint: dict, checkpoint_path: Path):
    metadata = checkpoint.get("metadata")
    if metadata is None:
        logger.warning("El checkpoint %s no tiene metadatos de esquema.", checkpoint_path)
        return
    expected_hash = build_schema_hash(config, metadata.get("numeric_features"), metadata.get("known_categoricals"))
    if metadata.get("schema_hash") != expected_hash:
        raise ValueError(f"El checkpoint {checkpoint_path} no es compatible con la configuracion actual")



def _build_checkpoint_paths(config: dict) -> tuple[Path, Path, Path]:
    canonical_path = Path(config["paths"]["model_save_path"])
    stem = canonical_path.stem
    suffix = canonical_path.suffix or ".pth"
    best_path = canonical_path.with_name(f"{stem}_best{suffix}")
    last_path = canonical_path.with_name(f"{stem}_last{suffix}")
    return canonical_path, best_path, last_path

def _create_trainer(config: dict, checkpoint_path: Path) -> pl.Trainer:
    runtime = resolve_execution_context(config, purpose="train")
    log_runtime_context(logger, "Entrenamiento TFT", runtime)
    deterministic_mode = "warn" if runtime.uses_cuda else True
    if runtime.uses_cuda:
        logger.warning(
            "En GPU se usa deterministic='warn' para evitar fallos con operaciones CUDA sin implementacion determinista estricta."
        )
    return pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=runtime.accelerator,
        devices=1,
        precision=runtime.precision,
        deterministic=deterministic_mode,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=config["training"]["early_stopping_patience"]),
            CustomModelCheckpoint(monitor="val_loss", save_path=str(checkpoint_path), mode="min"),
        ],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir=config["paths"]["logs_dir"]),
        enable_checkpointing=False,
    )



def _to_dataloader(dataset: TimeSeriesDataSet, train: bool, config: dict):
    num_workers = int(config["training"]["num_workers"])
    runtime = resolve_execution_context(config, purpose="train")
    return dataset.to_dataloader(
        train=train,
        batch_size=int(config["training"]["batch_size"]),
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=runtime.pin_memory,
        prefetch_factor=config["training"]["prefetch_factor"] if num_workers > 0 else None,
    )



def objective(trial, train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet, config: dict):
    trial_config = {**config, "training": dict(config["training"])}
    model = build_model(train_dataset, trial_config, trial)
    batch_size = estimate_batch_size(model, train_dataset, trial_config)
    trial_config["training"]["batch_size"] = batch_size
    logger.info("Batch size estimado para trial %s: %s", trial.number, batch_size)

    trial_dir = Path(trial_config["paths"]["models_dir"]) / "optuna" / trial_config["model_name"]
    trial_dir.mkdir(parents=True, exist_ok=True)
    trainer = _create_trainer(trial_config, trial_dir / f"trial_{trial.number}.pth")
    trainer.fit(
        model,
        train_dataloaders=_to_dataloader(train_dataset, True, trial_config),
        val_dataloaders=_to_dataloader(val_dataset, False, trial_config),
    )
    return trainer.callback_metrics["val_loss"].item()



def _save_checkpoint(path: Path, model, config: dict) -> None:
    payload = _build_checkpoint_payload(model, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    write_metadata(path, payload["metadata"])
    write_checksum(path)



def train_model(dataset: tuple, config: dict, use_optuna: bool = True, continue_training: bool = False, new_lr: float | None = None):
    logger.info("Inicio de train_model")
    train_dataset, val_dataset = dataset
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Datasets vacios: train={len(train_dataset)}, val={len(val_dataset)}")

    runtime = resolve_execution_context(config, purpose="train")
    device = runtime.torch_device
    model_save_path, best_model_path, last_model_path = _build_checkpoint_paths(config)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    if use_optuna and not continue_training:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, config), n_trials=config["training"]["optuna_trials"])
        best_params = study.best_params
        logger.info("Mejores hiperparametros de Optuna: %s", best_params)
    else:
        best_params = None
        logger.info("Optuna omitida; se usaran hiperparametros por defecto o de checkpoint")

    if continue_training and model_save_path.exists():
        ensure_relative_to(model_save_path, Path(config["paths"]["models_dir"]))
        verify_checksum(model_save_path, required=config["artifacts"]["require_hash_validation"])
        checkpoint = torch.load(model_save_path, map_location=torch.device("cpu"), weights_only=False)
        _validate_checkpoint_metadata(config, checkpoint, model_save_path)
        hyperparams = checkpoint["hyperparams"]
        if new_lr is not None:
            hyperparams["learning_rate"] = new_lr
        final_model = build_model(train_dataset, config, hyperparams=hyperparams)
        final_model.load_state_dict(checkpoint["state_dict"])
        final_model.to(device)
        logger.info("Modelo cargado desde %s", model_save_path)
    else:
        final_model = build_model(train_dataset, config, hyperparams=best_params)
        logger.info("Entrenamiento desde cero")

    batch_size = estimate_batch_size(final_model, train_dataset, config)
    config["training"]["batch_size"] = batch_size
    logger.info("Batch size definitivo: %s", batch_size)
    log_runtime_context(
        logger,
        "Configuracion final de entrenamiento",
        runtime,
        batch_size=batch_size,
        num_workers=int(config["training"]["num_workers"]),
        prefetch_factor=int(config["training"]["prefetch_factor"]) if int(config["training"]["num_workers"]) > 0 else None,
    )

    trainer = _create_trainer(config, best_model_path)
    trainer.fit(
        model=final_model,
        train_dataloaders=_to_dataloader(train_dataset, True, config),
        val_dataloaders=_to_dataloader(val_dataset, False, config),
    )

    _save_checkpoint(last_model_path, final_model, config)
    logger.info("Ultimo estado guardado en %s", last_model_path)

    if best_model_path.exists():
        shutil.copy2(best_model_path, model_save_path)
        if best_model_path.with_name(f"{best_model_path.name}.sha256").exists():
            shutil.copy2(best_model_path.with_name(f"{best_model_path.name}.sha256"), model_save_path.with_name(f"{model_save_path.name}.sha256"))
        if best_model_path.with_name(f"{best_model_path.name}.meta.json").exists():
            shutil.copy2(best_model_path.with_name(f"{best_model_path.name}.meta.json"), model_save_path.with_name(f"{model_save_path.name}.meta.json"))
        logger.info("Checkpoint canonico actualizado con el mejor modelo: %s", model_save_path)
    else:
        shutil.copy2(last_model_path, model_save_path)
        shutil.copy2(last_model_path.with_name(f"{last_model_path.name}.sha256"), model_save_path.with_name(f"{model_save_path.name}.sha256"))
        shutil.copy2(last_model_path.with_name(f"{last_model_path.name}.meta.json"), model_save_path.with_name(f"{model_save_path.name}.meta.json"))
        logger.warning("No se encontro checkpoint mejor; se usa el ultimo estado en %s", model_save_path)

    return final_model
