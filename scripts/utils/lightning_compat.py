from __future__ import annotations

try:
    import lightning.pytorch as pl
    from lightning.pytorch import LightningModule
    from lightning.pytorch.callbacks import EarlyStopping
    from lightning.pytorch.loggers import CSVLogger
except ImportError:  # pragma: no cover - compatibilidad con entornos antiguos
    import pytorch_lightning as pl
    from pytorch_lightning import LightningModule
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import CSVLogger


seed_everything = pl.seed_everything
LIGHTNING_LOGGER_NAME = "lightning.pytorch"

