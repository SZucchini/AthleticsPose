"""Lightning module for the Pose3D model."""

import numpy as np
import pytorch_lightning as pl
import torch
from torch import optim

from athleticspose.loss import calc_mpjpe, loss_mpjpe, loss_velocity, n_mpjpe, p_mpjpe
from athleticspose.models.MotionAGFormer.model import MotionAGFormer


def decay_lr_exponentially(lr: float, lr_decay: float, optimizer: optim.Optimizer) -> float:
    """Decay the learning rate exponentially.

    Args:
        lr (float): Learning rate.
        lr_decay (float): Learning rate decay.
        optimizer (optim.Optimizer): Optimizer.

    Returns:
        lr (float): Decayed learning rate.

    """
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group["lr"] *= lr_decay
    return lr


class LightningPose3D(pl.LightningModule):
    """Lightning module for the Pose3D model."""

    def __init__(self):
        """Initialize the LightningPose3D module."""
        super().__init__()
        self.model = MotionAGFormer(
            n_layers=16,
            dim_in=3,
            dim_feat=128,
            num_heads=8,
            neighbour_num=2,
            n_frames=81,
        )
        self.lr = 0.0005
        self.lr_decay = 0.99
        self.weight_decay = 0.01
        self.train_epoch_loss = 0.0
        self.train_step_cnt = 0
        self.val_epoch_mpjpe = 0
        self.val_epoch_pa_mpjpe = 0
        self.val_step_cnt = 0

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor. Shape: (B, T, J, 3).

        Returns:
            torch.Tensor: Output tensor. Shape: (B, T, J, 3).

        """
        return self.model(x)

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step."""
        x, y, _, _ = batch
        pred = self.model(x)
        loss_3d_pos = loss_mpjpe(pred, y)
        loss_3d_scale = n_mpjpe(pred, y)
        loss_3d_velocity = loss_velocity(pred, y)
        loss = loss_3d_pos + loss_3d_scale * 0.5 + loss_3d_velocity * 20
        self.log("train/batch/loss", loss)

        self.train_epoch_loss += loss.item()
        self.train_step_cnt += 1
        return loss

    def on_train_epoch_end(self):
        """On train epoch end."""
        self.lr = decay_lr_exponentially(self.lr, self.lr_decay, self.optimizers())
        loss_epoch = self.train_epoch_loss / self.train_step_cnt
        self.log("train/epoch/loss", loss_epoch)
        self.train_epoch_loss = 0.0
        self.train_step_cnt = 0

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        """Calculate validation metrics."""
        x, y, p2mm, norm_scale = batch
        # batch_input_flip = flip_data(x)
        pred = self.model(x)
        # pred_flip = self.model(batch_input_flip)
        # pred_2 = flip_data(pred_flip)
        # pred = (pred_1 + pred_2) / 2
        mpjpe, pa_mpjpe = self.calculate_metrics(pred, y, p2mm, norm_scale)
        self.val_epoch_mpjpe += mpjpe.item()
        self.val_epoch_pa_mpjpe += pa_mpjpe.item()
        self.val_step_cnt += 1
        return {"mpjpe": mpjpe, "pa_mpjpe": pa_mpjpe}

    def on_validation_epoch_end(self):
        """On validation epoch end."""
        self.val_epoch_mpjpe /= self.val_step_cnt
        self.val_epoch_pa_mpjpe /= self.val_step_cnt
        self.log("val/epoch/mpjpe", self.val_epoch_mpjpe)
        self.log("val/epoch/pa_mpjpe", self.val_epoch_pa_mpjpe)
        self.val_epoch_mpjpe = 0.0
        self.val_epoch_pa_mpjpe = 0.0
        self.val_step_cnt = 0

    def calculate_metrics(self, pred, y, p2mm, norm_scale):
        """Calculate validation metrics."""
        pred[:, :, 0, :] = 0
        pred = pred.cpu().numpy()
        y = y.cpu().numpy()
        p2mm = p2mm.cpu().numpy()
        norm_scale = norm_scale.cpu().numpy()

        pred_denom = np.zeros_like(pred)
        y_denom = np.zeros_like(y)
        for i in range(pred.shape[0]):
            pred_denom[i, :, :, :] = pred[i] / norm_scale[i]
            y_denom[i, :, :, :] = y[i] / norm_scale[i]
        pred_denom = pred_denom / p2mm[:, :, None, None]
        y_denom = y_denom / p2mm[:, :, None, None]

        pa_mpjpe_list = []
        mpjpe = np.mean(calc_mpjpe(pred_denom, y_denom))
        for i in range(pred_denom.shape[0]):
            pa_mpjpe = p_mpjpe(pred_denom[i], y_denom[i])
            pa_mpjpe_list.append(pa_mpjpe)
        pa_mpjpe = np.mean(pa_mpjpe_list)
        return mpjpe, pa_mpjpe
