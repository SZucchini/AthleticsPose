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

    def __init__(self, cfg):
        """Initialize the LightningPose3D module."""
        super().__init__()
        self.model = MotionAGFormer(
            n_layers=cfg.model.n_layers,
            dim_in=cfg.model.dim_in,
            dim_feat=cfg.model.dim_feat,
            num_heads=cfg.model.num_heads,
            neighbour_num=cfg.model.neighbour_num,
            n_frames=cfg.model.n_frames,
        )
        self.lr = cfg.train.lr
        self.lr_decay = cfg.train.lr_decay
        self.weight_decay = cfg.train.weight_decay

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
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        x, y, _, _, _ = batch
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
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Calculate validation metrics."""
        x, y, p2mm, norm_scale, valid_length = batch
        # batch_input_flip = flip_data(x)
        pred = self.model(x)
        # pred_flip = self.model(batch_input_flip)
        # pred_2 = flip_data(pred_flip)
        # pred = (pred_1 + pred_2) / 2
        mpjpe, pa_mpjpe = self.calculate_metrics(pred, y, p2mm, norm_scale, valid_length)
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

    def calculate_metrics(self, pred, y, p2mm, norm_scale, valid_length):
        """Calculate validation metrics with padding mask."""
        pred[:, :, 0, :] = 0
        pred = pred.to(torch.float32).cpu().numpy()
        y = y.to(torch.float32).cpu().numpy()
        p2mm = p2mm.to(torch.float32).cpu().numpy()
        norm_scale = norm_scale.to(torch.float32).cpu().numpy()
        valid_length = valid_length.to(torch.int32).cpu().numpy()

        # Process each sample individually to avoid zero division in padding
        mpjpe_list = []
        pa_mpjpe_list = []
        for i in range(pred.shape[0]):
            valid_len = valid_length[i, 0]  # Extract scalar from array

            # Extract only valid frames (before padding) for this sample
            pred_sample = pred[i, :valid_len]
            y_sample = y[i, :valid_len]
            p2mm_sample = p2mm[i, :valid_len]

            # Apply normalization scale
            pred_sample = pred_sample * norm_scale[i]
            y_sample = y_sample * norm_scale[i]

            # Apply pixel-to-mm conversion (only on valid frames, no zero division)
            pred_sample = pred_sample / p2mm_sample[:, None, None]
            y_sample = y_sample / p2mm_sample[:, None, None]

            # Calculate MPJPE for valid frames only
            mpjpe_sample = calc_mpjpe(pred_sample[None], y_sample[None])  # Add batch dim
            mpjpe_list.append(np.mean(mpjpe_sample))

            # Calculate PA-MPJPE for valid frames only
            pa_mpjpe_sample = p_mpjpe(pred_sample, y_sample)
            pa_mpjpe_list.append(np.mean(pa_mpjpe_sample))

        mpjpe = np.mean(mpjpe_list)
        pa_mpjpe = np.mean(pa_mpjpe_list)
        return mpjpe, pa_mpjpe
