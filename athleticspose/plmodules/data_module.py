"""Data module for the pose estimation task."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from athleticspose.datasets.dynamic_dataset import DynamicMotionDataset3D
from athleticspose.utils import flip_data


class PoseDataModule(pl.LightningDataModule):
    """Data module for the pose estimation task."""

    def __init__(self, cfg):
        """Initialize the data module."""
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.datamodule.batch_size
        self.train_transform = flip_data

    def setup(self, stage: str) -> None:
        """Set the data module."""
        if stage == "fit" or stage is None:
            self.train_dataset = DynamicMotionDataset3D(
                self.cfg.data,
                split="train",
                transform=self.train_transform,
                flip=True,
            )
            self.val_dataset = DynamicMotionDataset3D(self.cfg.data, split="test", transform=None, flip=False)

        if stage == "validate":
            self.val_dataset = DynamicMotionDataset3D(self.cfg.data, split="test", transform=None, flip=False)

        if stage == "test":
            self.test_dataset = DynamicMotionDataset3D(self.cfg.data, split="test", transform=None, flip=False)

    def train_dataloader(self):
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            persistent_workers=True,
            pin_memory=True,
            multiprocessing_context="forkserver",
        )

    def val_dataloader(self):
        """Validate dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            multiprocessing_context="forkserver",
        )

    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            multiprocessing_context="forkserver",
        )
