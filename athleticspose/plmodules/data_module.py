"""Data module for the pose estimation task."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from athleticspose.datasets.ap_dataset import MotionDataset3D, flip_data


class PoseDataModule(pl.LightningDataModule):
    """Data module for the pose estimation task."""

    def __init__(self, cfg):
        """Initialize the data module."""
        super().__init__()
        self.batch_size = cfg.datamodule.batch_size
        self.train_transform = flip_data
        self.train_dir = cfg.data.train_dir
        self.test_dir = cfg.data.test_dir

    def setup(self, stage: str) -> None:
        """Set the data module."""
        if stage == "fit" or stage is None:
            self.train_dataset = MotionDataset3D(self.train_dir, transform=self.train_transform)
            self.val_dataset = MotionDataset3D(self.test_dir)

        if stage == "validate":
            self.val_dataset = MotionDataset3D(self.test_dir)

        if stage == "test":
            self.test_dataset = MotionDataset3D(self.test_dir)

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
