"""Train the model."""

import os
import random
import time

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from athleticspose.plmodules.data_module import PoseDataModule
from athleticspose.plmodules.linghtning_module import LightningPose3D


def create_work_dir(work_dir: str) -> None:
    """Create the work directory.

    Args:
        work_dir (str): Work directory.

    """
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)


def init_wandb(cfg, work_dir: str) -> WandbLogger:
    """Initialize wandb.

    Args:
        cfg: Config object.
        work_dir (str): Work directory.

    """
    timestamp = time.strftime("%Y%m%d%H%M%S")
    wandb.init(project=cfg.wandb.project, name=f"{cfg.exp_name}-{timestamp}")
    wandb_logger = WandbLogger(project=cfg.wandb.project, name=f"{cfg.exp_name}-{timestamp}", save_dir=work_dir)
    return wandb_logger


def set_seed(seed: int = 42) -> None:
    """Set seed for reproducibility.

    Args:
        seed (int): Seed.

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pl.seed_everything(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.mps.deterministic = True
    torch.backends.mps.benchmark = False


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg):
    """Execute the training script."""
    set_seed(cfg.seed)
    pl.seed_everything(cfg.seed)
    work_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    wandb_logger = init_wandb(cfg, work_dir)

    model = LightningPose3D(cfg)
    data_module = PoseDataModule(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/epoch/mpjpe",
        dirpath=work_dir,
        filename="best",
        save_top_k=3,
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(dirpath=work_dir, filename="last", save_last=True)
    early_stopping_callback = EarlyStopping(
        monitor="val/epoch/mpjpe",
        patience=5,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            last_checkpoint_callback,
            early_stopping_callback,
        ],
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
