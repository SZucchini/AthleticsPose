"""AthleticsPose inference module."""

from typing import Optional

import numpy as np
import torch

from athleticspose.plmodules.linghtning_module import LightningPose3D
from athleticspose.preprocess.utils import normalize_kpts


class Pose3DInferencer:
    """3D pose inference class."""

    def __init__(self, checkpoint_path: str):
        """Initialize 3D pose inferencer.

        Args:
            checkpoint_path (str): Path to model checkpoint.

        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(checkpoint_path)
        self.clip_length = 81

    def _load_model(self, checkpoint_path: str) -> LightningPose3D:
        """Load model from checkpoint.

        Args:
            checkpoint_path (str): Path to model checkpoint.

        Returns:
            LightningPose3D: Loaded model.

        """
        model = LightningPose3D.load_from_checkpoint(checkpoint_path)
        model.eval()
        model.to(self.device)
        return model

    def _split_into_clips(self, poses_2d: np.ndarray) -> list[np.ndarray]:
        """Split poses into fixed-length clips with zero padding if needed.

        Args:
            poses_2d (np.ndarray): 2D poses. Shape: (T, J, 3).

        Returns:
            list[np.ndarray]: List of clips, each with shape (81, J, 3).

        """
        total_frames = poses_2d.shape[0]
        clips = []

        for start_idx in range(0, total_frames, self.clip_length):
            end_idx = start_idx + self.clip_length
            # Extract clip
            if end_idx <= total_frames:
                clip = poses_2d[start_idx:end_idx]
            else:
                # Create zero-padded clip for the last segment
                clip = np.zeros((self.clip_length, *poses_2d.shape[1:]))
                clip[: total_frames - start_idx] = poses_2d[start_idx:]
            clips.append(clip)

        return clips

    def preprocess(self, poses_2d: np.ndarray) -> tuple[torch.Tensor, list[tuple[float, np.ndarray]]]:
        """Preprocess 2D poses for model input.

        Args:
            poses_2d (np.ndarray): 2D poses. Shape: (T, J, 3).

        Returns:
            batch (torch.Tensor): Processed batch of clips. Shape: (N, 81, J, 3).

        """
        # Split into clips first
        clips = self._split_into_clips(poses_2d)

        # Process each clip
        processed_clips = []
        for clip in clips:
            scores = clip[..., 2].copy()
            clip_coords = clip
            clip_norm, norm_scale = normalize_kpts(clip_coords)

            # Restore scores
            clip_norm = clip_norm[..., :2]
            clip_with_scores = np.concatenate([clip_norm, scores[..., None]], axis=-1)
            processed_clips.append(clip_with_scores)

        # Stack clips into batch
        batch = np.stack(processed_clips)
        batch = torch.from_numpy(batch).float().to(self.device)
        return batch

    def postprocess(
        self,
        predictions: np.ndarray,
        original_length: int,
    ) -> np.ndarray:
        """Postprocess predictions by restoring original scale and combining clips.

        Args:
            predictions (np.ndarray): Model predictions. Shape: (N, 81, J, 3).
            original_length (int): Original sequence length.

        Returns:
            np.ndarray: Processed predictions. Shape: (T, J, 3).

        """
        final_predictions = []
        current_frame = 0

        for pred in predictions:
            remaining_frames = original_length - current_frame
            valid_frames = min(self.clip_length, remaining_frames)

            if valid_frames > 0:
                final_predictions.append(pred[:valid_frames])
                current_frame += valid_frames

            if current_frame >= original_length:
                break

        return np.concatenate(final_predictions)

    @torch.no_grad()
    def predict(self, poses_2d: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Predict 3D poses from 2D poses.

        Args:
            poses_2d (np.ndarray): 2D poses. Shape: (T, J, 3).
            batch_size (Optional[int], optional): Batch size for inference.
                If None, process all clips at once. Defaults to None.

        Returns:
            np.ndarray: Predicted 3D poses. Shape: (T, J, 3).

        """
        original_length = poses_2d.shape[0]
        batch = self.preprocess(poses_2d)

        if batch_size is not None:
            predictions = []
            if len(batch) > 0:
                for i in range(0, len(batch), batch_size):
                    batch_predictions = self.model(batch[i : i + batch_size])
                    predictions.append(batch_predictions.cpu().numpy())
                if len(predictions) == 1:
                    predictions = np.array(predictions)
                else:
                    predictions = np.concatenate(predictions, axis=0)
            else:
                raise ValueError("No clips to process.")
        else:
            predictions = self.model(batch).cpu().numpy()

        poses_3d = self.postprocess(predictions, original_length)
        return poses_3d
