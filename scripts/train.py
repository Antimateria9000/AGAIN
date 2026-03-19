import logging
import shutil
from pathlib import Path

import optuna
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from scripts.model import build_model
from scripts.utils.batch_size_estimator import estimate_batch_size
from scripts.utils.data_schema import build_artifact_metadata, build_schema_hash

logger = logging.getLogger(__name__)

cudnn.benchmark = True


class CustomModelCheckpoint(pl.callbacks.Callback):
    def __init__(self, monitor: str, save_path: str, mode: str = 'min'):
        super().__init__()
        self.monitor = monitor
        self.save_path = Path(save_path)
        self.mode = mode
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def on_validation_end(self, trainer, pl_module):
        current_score = trainer.callback_metrics.get(self.monitor)
        if current_score is None:
            return
        is_better = (self.mode == 'min' and current_score < self.best_score) or (self.mode == 'max' and current_score > self.best_score)
        if is_better:
            self.best_score = current_score
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(_build_checkpoint_payload(pl_module, pl_module.model_config.config), self.save_path)
            logger.info(f"Nuevo mejor checkpoint en {self.save_path} con {self.monitor}={current_score}")



def _build_checkpoint_payload(model, config: dict) -> dict:
    metadata = build_artifact_metadata(config)
    return {
        'state_dict': model.state_dict(),
        'hyperparams': dict(model.hparams),
        'metadata': metadata,
    }



def _validate_checkpoint_metadata(config: dict, checkpoint: dict, checkpoint_path: Path):
    metadata = checkpoint.get('metadata')
    if metadata is None:
        logger.warning(f"El checkpoint {checkpoint_path} no tiene metadatos de esquema.")
        return
    expected_hash = build_schema_hash(config, metadata.get('numeric_features'), metadata.get('known_categoricals'))
    if metadata.get('schema_hash') != expected_hash:
        raise ValueError(f"El checkpoint {checkpoint_path} no es compatible con la configuracion actual")



def _build_checkpoint_paths(config: dict) -> tuple[Path, Path, Path]:
    canonical_path = Path(config['paths']['model_save_path'])
    stem = canonical_path.stem
    suffix = canonical_path.suffix or '.pth'
    best_path = canonical_path.with_name(f"{stem}_best{suffix}")
    last_path = canonical_path.with_name(f"{stem}_last{suffix}")
    return canonical_path, best_path, last_path



def _create_trainer(config: dict, checkpoint_path: Path) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='16-mixed' if torch.cuda.is_available() else '32-true',
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=config['training']['early_stopping_patience']),
            CustomModelCheckpoint(monitor='val_loss', save_path=str(checkpoint_path), mode='min'),
        ],
        enable_progress_bar=True,
        logger=CSVLogger(save_dir='logs/'),
    )



def _to_dataloader(dataset: TimeSeriesDataSet, train: bool, config: dict):
    num_workers = config['training']['num_workers']
    return dataset.to_dataloader(
        train=train,
        batch_size=config['training']['batch_size'],
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=config['training']['prefetch_factor'] if num_workers > 0 else None,
    )



def objective(trial, train_dataset: TimeSeriesDataSet, val_dataset: TimeSeriesDataSet, config: dict):
    model = build_model(train_dataset, config, trial)
    batch_size = estimate_batch_size(model, train_dataset, config)
    config['training']['batch_size'] = batch_size
    logger.info(f"Batch size estimado para trial {trial.number}: {batch_size}")

    trial_dir = Path(config['paths']['models_dir']) / 'optuna' / config['model_name']
    trial_dir.mkdir(parents=True, exist_ok=True)
    trainer = _create_trainer(config, trial_dir / f"trial_{trial.number}.pth")
    trainer.fit(
        model,
        train_dataloaders=_to_dataloader(train_dataset, True, config),
        val_dataloaders=_to_dataloader(val_dataset, False, config),
    )
    return trainer.callback_metrics['val_loss'].item()



def train_model(dataset: tuple, config: dict, use_optuna: bool = True, continue_training: bool = False, new_lr: float = None):
    logger.info('Inicio de train_model')
    train_dataset, val_dataset = dataset
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError(f"Datasets vacios: train={len(train_dataset)}, val={len(val_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_save_path, best_model_path, last_model_path = _build_checkpoint_paths(config)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    if use_optuna and not continue_training:
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, config), n_trials=config['training']['optuna_trials'])
        best_params = study.best_params
        logger.info(f"Mejores hiperparametros de Optuna: {best_params}")
    else:
        best_params = None
        logger.info('Optuna omitida; se usaran hiperparametros por defecto o de checkpoint')

    if continue_training and model_save_path.exists():
        checkpoint = torch.load(model_save_path, map_location=torch.device('cpu'), weights_only=False)
        _validate_checkpoint_metadata(config, checkpoint, model_save_path)
        hyperparams = checkpoint['hyperparams']
        if new_lr is not None:
            hyperparams['learning_rate'] = new_lr
        final_model = build_model(train_dataset, config, hyperparams=hyperparams)
        final_model.load_state_dict(checkpoint['state_dict'])
        final_model.to(device)
        logger.info(f"Modelo cargado desde {model_save_path}")
    else:
        final_model = build_model(train_dataset, config, hyperparams=best_params)
        logger.info('Entrenamiento desde cero')

    batch_size = estimate_batch_size(final_model, train_dataset, config)
    config['training']['batch_size'] = batch_size
    logger.info(f"Batch size definitivo: {batch_size}")

    trainer = _create_trainer(config, best_model_path)
    trainer.fit(
        model=final_model,
        train_dataloaders=_to_dataloader(train_dataset, True, config),
        val_dataloaders=_to_dataloader(val_dataset, False, config),
    )

    torch.save(_build_checkpoint_payload(final_model, config), last_model_path)
    logger.info(f"Ultimo estado guardado en {last_model_path}")

    if best_model_path.exists():
        shutil.copy2(best_model_path, model_save_path)
        logger.info(f"Checkpoint canonico actualizado con el mejor modelo: {model_save_path}")
    else:
        shutil.copy2(last_model_path, model_save_path)
        logger.warning(f"No se encontro checkpoint mejor; se usa el ultimo estado en {model_save_path}")

    return final_model
