"""Train the model."""

import os
import random
import time

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import wandb
from athleticspose.plmodules.data_module import PoseDataModule
from athleticspose.plmodules.linghtning_module import LightningPose3D


def create_work_dir(work_dir: str) -> None:
    """Create the work directory.

    Args:
        work_dir (str): Work directory.

    """
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)


def init_wandb(work_dir: str) -> WandbLogger:
    """Initialize wandb.

    Args:
        work_dir (str): Work directory.

    """
    run_name = work_dir.replace("work_dir/", "").replace("/", "-")
    wandb.init(project="AthleticsPose", name=run_name)
    wandb_logger = WandbLogger(project="AthleticsPose", name=run_name)
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


def main():
    """Execute the training script."""
    set_seed()
    work_dir = os.path.join("work_dir", time.strftime("%Y%m%d_%H%M%S"))
    create_work_dir(work_dir)
    wandb_logger = init_wandb(work_dir)

    model = LightningPose3D()
    data_module = PoseDataModule()

    checkpoint_callback = ModelCheckpoint(
        monitor="val/epoch/mpjpe",
        dirpath=work_dir,
        filename="best",
        save_top_k=1,
        mode="min",
    )
    last_checkpoint_callback = ModelCheckpoint(dirpath=work_dir, filename="last", save_last=True)
    early_stopping_callback = EarlyStopping(
        monitor="val/epoch/mpjpe",
        patience=5,
        mode="min",
    )

    trainer = pl.Trainer(
        accelerator="mps",
        devices=1,
        precision="32",
        max_epochs=60,
        min_epochs=40,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            last_checkpoint_callback,
            early_stopping_callback,
        ],
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)

    wandb.finish()


if __name__ == "__main__":
    main()
