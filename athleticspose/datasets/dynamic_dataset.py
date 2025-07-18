"""Dynamic dataset for AthleticsPose that generates clips on-the-fly."""

import glob
import logging
import os
import random
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset

from athleticspose.utils import normalize_kpts, split_clips

# Set up logger
logger = logging.getLogger(__name__)


class DynamicMotionDataset3D(Dataset):
    """3D motion dataset with dynamic clip generation."""

    def __init__(
        self,
        cfg,
        split: str = "train",
        transform: Callable | None = None,
        flip: bool = True,
    ) -> None:
        """Initialize the dynamic dataset.

        Args:
            cfg: Hydra configuration object containing data settings.
            split: Dataset split ("train" or "test").
            transform: Transform to apply to the data (e.g., flip_data).
            flip: Flag to enable data augmentation flipping.

        """
        self.cfg = cfg
        self.split = split
        self.transform = transform
        self.flip = flip

        # Validate configuration
        self._validate_config()

        # Generate filtered clip index
        self.clips = self._create_filtered_index()

        # Log dataset initialization summary
        logger.info(
            f"DynamicMotionDataset3D initialized: "
            f"split={split}, clips={len(self.clips)}, "
            f"input_2d_type={getattr(self.cfg, 'input_2d_type', 'gt')}"
        )

    def _validate_config(self) -> None:
        """Validate dataset configuration."""
        # Check input_2d_type
        valid_input_types = {"gt", "det"}
        if hasattr(self.cfg, "input_2d_type") and self.cfg.input_2d_type not in valid_input_types:
            raise ValueError(f"input_2d_type must be one of {valid_input_types}, got {self.cfg.input_2d_type}")

        # Check det_model_type
        valid_det_models = {"pretrained", "ft"}
        if hasattr(self.cfg, "det_model_type") and self.cfg.det_model_type not in valid_det_models:
            raise ValueError(f"det_model_type must be one of {valid_det_models}, got {self.cfg.det_model_type}")

    def _get_2d_markers_dir(self) -> str:
        """Get 2D markers directory based on configuration.

        Returns:
            Directory name for 2D marker files.

        """
        input_2d_type = getattr(self.cfg, "input_2d_type", "gt")

        if input_2d_type == "gt":
            return "gt_markers2d_by_cam"
        elif input_2d_type == "det":
            det_model_type = getattr(self.cfg, "det_model_type", "pretrained")
            if det_model_type == "pretrained":
                return "det_markers2d_by_cam_coco"
            elif det_model_type == "ft":
                return "det_markers2d_by_cam_ft"

        raise ValueError(f"Invalid configuration: input_2d_type={input_2d_type}")

    def _parse_file_info(self, source_file: str) -> tuple[str, str, str, int]:
        """Parse file path to extract action, subject, date, and camera index.

        Args:
            source_file: Path to .npz file in gt_markers3d_by_cam.
                Example: "data/AthleticsPoseDataset/gt_markers3d_by_cam/hurdle/S01/20250125_00_0.npz"

        Returns:
            Tuple of (action, subject, date, camera_idx).

        """
        path_parts = source_file.split(os.sep)
        # Find gt_markers3d_by_cam directory and extract info
        markers_idx = path_parts.index("gt_markers3d_by_cam")
        action = path_parts[markers_idx + 1]  # e.g., "hurdle"
        subject = path_parts[markers_idx + 2]  # e.g., "S01"
        filename = os.path.basename(source_file)  # e.g., "20250125_00_0.npz"

        # Parse filename: "20250125_00_0.npz" -> date="20250125", camera_idx=0
        name_parts = filename.split("_")
        date = name_parts[0]  # e.g., "20250125"
        camera_idx_str = name_parts[2].split(".")[0]  # e.g., "0" from "0.npz"
        camera_idx = int(camera_idx_str)  # Convert to 0-indexed integer

        return action, subject, date, camera_idx

    def _create_filtered_index(self) -> list[dict]:
        """Generate filtered clip index based on Hydra configuration.

        Returns:
            List of clip information dictionaries.

        """
        # Generate full index first
        all_clips = self._create_full_index()

        # Apply filtering
        filtered_clips = []
        for clip in all_clips:
            # Action filter
            if (
                hasattr(self.cfg, "filter_actions")
                and self.cfg.filter_actions is not None
                and clip["action"] not in self.cfg.filter_actions
            ):
                continue
            # Subject filter
            if (
                hasattr(self.cfg, "filter_subjects")
                and self.cfg.filter_subjects is not None
                and clip["subject"] not in self.cfg.filter_subjects
            ):
                continue
            # Camera filter
            if (
                hasattr(self.cfg, "filter_cameras")
                and self.cfg.filter_cameras is not None
                and clip["camera_idx"] not in self.cfg.filter_cameras
            ):
                continue
            filtered_clips.append(clip)

        return filtered_clips

    def _create_full_index(self) -> list[dict]:
        """Generate full clip index for all available data.

        Returns:
            List of all clip information dictionaries.

        """
        clips = []
        stride = self.cfg.stride_train if self.split == "train" else self.cfg.stride_test

        # Scan all .npz files in gt_markers3d_by_cam
        marker_pattern = os.path.join(self.cfg.data_root, "gt_markers3d_by_cam", "**", "*.npz")
        for source_file in glob.glob(marker_pattern, recursive=True):
            # Parse file information (now includes camera_idx)
            action, subject, date, camera_idx = self._parse_file_info(source_file)

            # Train/test split based on test_subjects
            is_test_subject = subject in self.cfg.test_subjects
            if (self.split == "test") != is_test_subject:
                continue

            # Check if corresponding 2D marker file exists
            markers_2d_dir = self._get_2d_markers_dir()
            # Build 2D marker filename using same format as 3D files but with .npy extension
            base_filename = os.path.basename(source_file).replace(".npz", ".npy")
            markers_2d_file = os.path.join(self.cfg.data_root, markers_2d_dir, action, subject, base_filename)
            if not os.path.exists(markers_2d_file):
                logger.warning(f"Skipping {source_file}: corresponding 2D marker file not found: {markers_2d_file}")
                continue

            # Get total frame count from 3D markers
            try:
                with np.load(source_file) as data:
                    if "markers_h36m" not in data:
                        logger.error(f"Missing 'markers_h36m' key in {source_file}")
                        continue
                    markers_h36m = data["markers_h36m"]
                    total_frames = markers_h36m.shape[0]

                    # Validate markers_h36m shape
                    if len(markers_h36m.shape) != 3 or markers_h36m.shape[1] != 17 or markers_h36m.shape[2] != 3:
                        logger.error(
                            f"Invalid markers_h36m shape in {source_file}: "
                            f"expected (T, 17, 3), got {markers_h36m.shape}"
                        )
                        continue

            except Exception as e:
                logger.error(f"Failed to load {source_file}: {e}")
                continue

            # Generate clips (no loop over cameras since camera_idx is already parsed)
            if self.split == "train":
                # Train: use existing split_clips (with potential frame duplication)
                clip_indices_list = split_clips(total_frames, self.cfg.clip_length, stride)
                for frame_indices in clip_indices_list:
                    clips.append(
                        {
                            "source_file": source_file,
                            "markers_2d_file": markers_2d_file,
                            "camera_idx": camera_idx,
                            "frame_indices": list(frame_indices),
                            "action": action,
                            "subject": subject,
                            "date": date,
                            "needs_padding": False,
                        }
                    )
            else:
                # Test: handle frame shortage with zero padding
                if total_frames >= self.cfg.clip_length:
                    # Sufficient frames available
                    clip_indices_list = split_clips(total_frames, self.cfg.clip_length, stride)
                    for frame_indices in clip_indices_list:
                        clips.append(
                            {
                                "source_file": source_file,
                                "markers_2d_file": markers_2d_file,
                                "camera_idx": camera_idx,
                                "frame_indices": list(frame_indices),
                                "action": action,
                                "subject": subject,
                                "date": date,
                                "needs_padding": False,
                            }
                        )
                else:
                    # Frame shortage: use all frames and pad with zeros
                    clips.append(
                        {
                            "source_file": source_file,
                            "markers_2d_file": markers_2d_file,
                            "camera_idx": camera_idx,
                            "frame_indices": list(range(total_frames)),
                            "action": action,
                            "subject": subject,
                            "date": date,
                            "needs_padding": True,
                            "target_length": self.cfg.clip_length,
                        }
                    )

        return clips

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            Number of clips in the dataset.

        """
        return len(self.clips)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, float, torch.LongTensor]:
        """Get item at given index with dynamic data processing.

        Args:
            idx: Index of the clip.

        Returns:
            Tuple of (input2d, label3d, p2mm, norm_scale, valid_length).
            - input2d: 2D motion data with Z=1. Shape: (clip_length, 17, 3)
            - label3d: Normalized 3D motion data. Shape: (clip_length, 17, 3)
            - p2mm: Pixel to mm scale factors. Shape: (clip_length,)
            - norm_scale: Normalization scale factor (scalar).
            - valid_length: Number of valid (non-padded) frames (scalar).

        """
        if idx < 0 or idx >= len(self.clips):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.clips)} clips")

        clip_info = self.clips[idx]

        try:
            # Load 3D markers and p2mm from npz file
            with np.load(clip_info["source_file"]) as data:
                if "markers_h36m" not in data:
                    raise KeyError(f"'markers_h36m' not found in {clip_info['source_file']}")
                if "p2mm" not in data:
                    raise KeyError(f"'p2mm' not found in {clip_info['source_file']}")

                markers_h36m = data["markers_h36m"]  # Already in H36M format
                p2mm_per_frame = data["p2mm"]  # Pixel to mm conversion factor

            # Load corresponding 2D markers
            if not os.path.exists(clip_info["markers_2d_file"]):
                raise FileNotFoundError(f"2D marker file not found: {clip_info['markers_2d_file']}")

            markers_2d = np.load(clip_info["markers_2d_file"])

            # Validate 2D marker shape
            if len(markers_2d.shape) != 3 or markers_2d.shape[1] != 17 or markers_2d.shape[2] != 3:
                raise ValueError(
                    f"Invalid 2D marker shape in {clip_info['markers_2d_file']}: "
                    f"expected (T, 17, 3), got {markers_2d.shape}"
                )

        except Exception as e:
            logger.error(f"Failed to load data for clip {idx}: {e}")
            raise RuntimeError(f"Data loading failed for clip {idx}") from e

        # Extract clip using frame indices
        clipped_3d = markers_h36m[clip_info["frame_indices"]]
        clipped_2d = markers_2d[clip_info["frame_indices"]]

        # Validate p2mm format - should be per-frame array, not scalar
        if np.isscalar(p2mm_per_frame) or p2mm_per_frame.ndim == 0:
            raise ValueError(
                f"p2mm should be a per-frame array with shape ({len(markers_h36m)},), "
                f"but got scalar value: {p2mm_per_frame}. "
                f"This indicates a problem in the preprocessing step."
            )

        # Use frame-specific p2mm values
        clipped_p2mm = p2mm_per_frame[clip_info["frame_indices"]]

        # Track valid length before padding
        valid_length = len(clipped_3d)

        # Zero padding for test data if needed
        if clip_info.get("needs_padding", False):
            target_length = clip_info["target_length"]
            current_length = len(clipped_3d)
            if current_length < target_length:
                # Pad 3D keypoint data with zeros
                pad_width = ((0, target_length - current_length), (0, 0), (0, 0))
                clipped_3d = np.pad(clipped_3d, pad_width, mode="constant", constant_values=0)
                # Pad 2D keypoint data with zeros
                clipped_2d = np.pad(clipped_2d, pad_width, mode="constant", constant_values=0)
                # Pad p2mm data with zeros
                pad_width_scale = (0, target_length - current_length)
                clipped_p2mm = np.pad(clipped_p2mm, pad_width_scale, mode="constant", constant_values=0)

        # Normalize 3D keypoints
        label3d_norm, norm_scale = normalize_kpts(clipped_3d)

        score2d = clipped_2d.copy()[:, :, 2]
        input2d, _ = normalize_kpts(clipped_2d)
        input2d[:, :, 2] = score2d

        # Data augmentation (flip processing)
        if self.transform is not None and self.flip and random.random() > 0.5:
            input2d = self.transform(input2d)
            label3d_norm = self.transform(label3d_norm)

        # Convert to tensors
        input2d = torch.FloatTensor(input2d).to(torch.float32)
        label3d = torch.FloatTensor(label3d_norm).to(torch.float32)
        p2mm = torch.FloatTensor(clipped_p2mm).to(torch.float32)
        norm_scale = np.float32(norm_scale)
        valid_length = torch.LongTensor([valid_length])

        return input2d, label3d, p2mm, norm_scale, valid_length

    def get_clip_info(self, idx: int) -> dict:
        """Get clip information for given index.

        Args:
            idx: Index of the clip.

        Returns:
            Dictionary containing clip information.

        Raises:
            IndexError: If idx is out of range.

        """
        if idx < 0 or idx >= len(self.clips):
            raise IndexError(f"Index {idx} is out of range for dataset of size {len(self.clips)}")
        return self.clips[idx].copy()  # Return a copy to prevent modification

    def get_unique_actions(self) -> list[str]:
        """Get unique actions in the dataset.

        Returns:
            List of unique action names sorted alphabetically.

        """
        return sorted(set(clip["action"] for clip in self.clips))

    def get_unique_subjects(self) -> list[str]:
        """Get unique subjects in the dataset.

        Returns:
            List of unique subject names sorted alphabetically.

        """
        return sorted(set(clip["subject"] for clip in self.clips))
