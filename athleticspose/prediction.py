"""2D marker-based 3D pose prediction for AthleticsPose."""

import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from athleticspose.plmodules.linghtning_module import LightningPose3D
from athleticspose.utils import denormalize_kpts, normalize_kpts


class From2DMarkersPredictor:
    """2D marker-based 3D pose predictor."""

    def __init__(self, cfg):
        """Initialize 2D marker predictor.

        Args:
            cfg: Hydra configuration object

        """
        self.cfg = cfg
        self.device = self._setup_device()
        self.model = self._load_model()
        self.clip_length = 81

    def _setup_device(self) -> torch.device:
        """Setup computation device."""
        if self.cfg.prediction.processing.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.cfg.prediction.processing.device)
        return device

    def _load_model(self) -> LightningPose3D:
        """Load model from checkpoint."""
        checkpoint_path = self.cfg.prediction.model.checkpoint_path
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            model = LightningPose3D.load_from_checkpoint(checkpoint_path, cfg=self.cfg)
            model.eval()
            model.to(self.device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {str(e)}") from e

    def _get_input_dir(self) -> str:
        """Get input directory based on marker type."""
        marker_type = self.cfg.prediction.input.marker_type
        data_root = self.cfg.prediction.input.data_root

        if marker_type == "gt":
            return os.path.join(data_root, "gt_markers2d_by_cam")
        elif marker_type == "det_coco":
            return os.path.join(data_root, "det_markers2d_by_cam_coco")
        elif marker_type == "det_ft":
            return os.path.join(data_root, "det_markers2d_by_cam_ft")
        else:
            raise ValueError(f"Invalid marker_type: {marker_type}")

    def _get_gt_dir(self) -> str:
        """Get GT directory for 3D markers."""
        data_root = self.cfg.prediction.input.data_root
        return os.path.join(data_root, "gt_markers3d_by_cam")

    def _get_output_dir(self) -> str:
        """Get output directory based on marker type."""
        marker_type = self.cfg.prediction.input.marker_type
        output_root = self.cfg.prediction.output.output_dir

        if marker_type == "gt":
            return os.path.join(output_root, "pred_from_gt")
        elif marker_type == "det_coco":
            return os.path.join(output_root, "pred_from_det_coco")
        elif marker_type == "det_ft":
            return os.path.join(output_root, "pred_from_det_ft")
        else:
            raise ValueError(f"Invalid marker_type: {marker_type}")

    def _collect_input_files(self) -> List[str]:
        """Collect input files from specified directory."""
        input_dir = self._get_input_dir()
        test_subjects = self.cfg.prediction.test_subjects

        input_files = []
        for action in os.listdir(input_dir):
            action_dir = os.path.join(input_dir, action)
            if not os.path.isdir(action_dir):
                continue

            for subject in os.listdir(action_dir):
                if subject not in test_subjects:
                    continue

                subject_dir = os.path.join(action_dir, subject)
                if not os.path.isdir(subject_dir):
                    continue

                for filename in os.listdir(subject_dir):
                    if filename.endswith(".npy"):
                        file_path = os.path.join(subject_dir, filename)
                        input_files.append(file_path)

        return sorted(input_files)

    def _normalize_2d_markers(self, markers_2d: np.ndarray) -> Tuple[np.ndarray, float]:
        """Normalize 2D markers preserving confidence scores.

        Args:
            markers_2d: 2D markers with shape (T, 17, 3) - (x, y, confidence)

        Returns:
            Tuple of (normalized_markers, norm_scale)

        """
        # Extract coordinates and confidence
        coords = markers_2d[:, :, :2]  # (T, 17, 2)
        confidence = markers_2d[:, :, 2:3]  # (T, 17, 1)

        # Create 3D coords with z=0 for normalization
        coords_3d = np.concatenate([coords, np.zeros_like(coords[:, :, :1])], axis=-1)

        # Normalize using existing function
        coords_norm, norm_scale = normalize_kpts(coords_3d)

        # Combine normalized coords with original confidence (don't normalize confidence)
        normalized_markers = np.concatenate([coords_norm[:, :, :2], confidence], axis=-1)

        return normalized_markers, norm_scale

    def _normalize_2d_markers_with_gt(
        self, markers_2d: np.ndarray, gt_markers_3d: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Normalize 2D markers normally but get GT norm_scale for denormalization.

        Args:
            markers_2d: 2D markers with shape (T, 17, 3) - (x, y, confidence)
            gt_markers_3d: GT 3D markers with shape (T, 17, 3)

        Returns:
            Tuple of (normalized_markers, gt_norm_scale)

        """
        # Normalize 2D markers using the original method
        normalized_markers, _ = self._normalize_2d_markers(markers_2d)

        # Calculate GT norm_scale for denormalization
        _, gt_norm_scale = normalize_kpts(gt_markers_3d)

        return normalized_markers, gt_norm_scale

    def _split_into_clips(self, data: np.ndarray, gt_data: np.ndarray) -> List[Tuple[np.ndarray, int, int, float]]:
        """Split data into clips of 81 frames, normalize 2D markers, and get GT norm_scale.

        Args:
            data: Input 2D marker data with shape (T, ...)
            gt_data: GT 3D marker data with shape (T, 17, 3)

        Returns:
            List of (normalized_clip_data, start_idx, valid_length, gt_norm_scale) tuples

        """
        total_frames = data.shape[0]
        clips = []

        if total_frames <= self.clip_length:
            # Single clip with padding if needed
            if total_frames < self.clip_length:
                # Pad with zeros
                pad_width = [(0, self.clip_length - total_frames)] + [(0, 0)] * (data.ndim - 1)
                padded_data = np.pad(data, pad_width, mode="constant", constant_values=0)
                # Normalize only the valid part using GT normalization
                valid_data = padded_data[:total_frames]
                valid_gt = gt_data[:total_frames]
                normalized_valid, gt_norm_scale = self._normalize_2d_markers_with_gt(valid_data, valid_gt)

                # Reconstruct padded data with normalized valid part
                normalized_clip = np.zeros_like(padded_data)
                normalized_clip[:total_frames] = normalized_valid
                clips.append((normalized_clip, 0, total_frames, gt_norm_scale))
            else:
                normalized_clip, gt_norm_scale = self._normalize_2d_markers_with_gt(data, gt_data)
                clips.append((normalized_clip, 0, total_frames, gt_norm_scale))
        else:
            # Multiple clips
            start_idx = 0
            while start_idx < total_frames:
                end_idx = min(start_idx + self.clip_length, total_frames)
                clip_data = data[start_idx:end_idx]
                clip_gt = gt_data[start_idx:end_idx]

                if clip_data.shape[0] < self.clip_length:
                    # Pad last clip
                    pad_width = [(0, self.clip_length - clip_data.shape[0])] + [(0, 0)] * (data.ndim - 1)
                    padded_clip = np.pad(clip_data, pad_width, mode="constant", constant_values=0)
                    # Normalize only the valid part using GT normalization
                    normalized_valid, gt_norm_scale = self._normalize_2d_markers_with_gt(clip_data, clip_gt)

                    # Reconstruct padded data with normalized valid part
                    normalized_clip = np.zeros_like(padded_clip)
                    normalized_clip[: clip_data.shape[0]] = normalized_valid
                    clips.append((normalized_clip, start_idx, end_idx - start_idx, gt_norm_scale))
                else:
                    normalized_clip, gt_norm_scale = self._normalize_2d_markers_with_gt(clip_data, clip_gt)
                    clips.append((normalized_clip, start_idx, self.clip_length, gt_norm_scale))

                start_idx += self.clip_length

        return clips

    def _predict_clips_batch(self, clips: List[np.ndarray]) -> List[np.ndarray]:
        """Predict 3D poses for multiple clips in batch.

        Args:
            clips: List of normalized 2D marker clips

        Returns:
            List of predicted 3D poses

        """
        batch_size = self.cfg.prediction.processing.batch_size
        predictions = []

        for i in range(0, len(clips), batch_size):
            batch_clips = clips[i : i + batch_size]
            batch_tensor = torch.FloatTensor(np.stack(batch_clips)).to(self.device)

            with torch.no_grad():
                batch_pred = self.model(batch_tensor)
                batch_pred[:, :, 0, :] = 0
                batch_pred_np = batch_pred.cpu().numpy()

            for pred in batch_pred_np:
                predictions.append(pred)

        return predictions

    def _reconstruct_sequence(
        self,
        clip_predictions: List[np.ndarray],
        clip_info: List[Tuple[int, int, float]],
    ) -> np.ndarray:
        """Reconstruct full sequence from clip predictions with per-clip denormalization.

        Args:
            clip_predictions: List of predicted clips
            clip_info: List of (start_idx, valid_length, norm_scale) for each clip

        Returns:
            Reconstructed full sequence

        """
        if len(clip_predictions) == 1:
            # Single clip - extract valid frames and denormalize
            _, valid_length, norm_scale = clip_info[0]
            valid_pred = clip_predictions[0][:valid_length]
            return denormalize_kpts(valid_pred, norm_scale)

        # Multiple clips - denormalize each clip and concatenate
        full_sequence = []
        for pred, (_, valid_length, norm_scale) in zip(clip_predictions, clip_info, strict=False):
            valid_pred = pred[:valid_length]
            denorm_pred = denormalize_kpts(valid_pred, norm_scale)
            full_sequence.append(denorm_pred)

        return np.concatenate(full_sequence, axis=0)

    def _load_gt_file(self, input_file: str) -> np.ndarray:
        """Load corresponding GT 3D markers file.

        Args:
            input_file: Path to input 2D marker file

        Returns:
            GT 3D markers array with shape (T, 17, 3)

        """
        # Convert input file path to GT file path
        input_dir = self._get_input_dir()
        gt_dir = self._get_gt_dir()

        # Get relative path from input directory
        rel_path = os.path.relpath(input_file, input_dir)

        # Create GT file path (change extension from .npy to .npz)
        gt_file_path = os.path.join(gt_dir, rel_path.replace(".npy", ".npz"))

        if not os.path.exists(gt_file_path):
            raise FileNotFoundError(f"GT file not found: {gt_file_path}")

        # Load GT data
        gt_data = np.load(gt_file_path)
        return gt_data["markers_h36m"]  # (T, 17, 3)

    def predict_single_file(self, input_file: str) -> Dict[str, Any]:
        """Predict 3D pose from single 2D marker file.

        Args:
            input_file: Path to input 2D marker file

        Returns:
            Dictionary containing prediction results

        """
        start_time = time.time()

        try:
            # Load 2D markers
            markers_2d = np.load(input_file)  # (T, 17, 3)

            # Load corresponding GT 3D markers
            gt_markers_3d = self._load_gt_file(input_file)  # (T, 17, 3)

            # Split into clips and normalize each clip using GT normalization
            clips_data = self._split_into_clips(markers_2d, gt_markers_3d)
            clips = [clip_data for clip_data, _, _, _ in clips_data]
            clip_info = [
                (start_idx, valid_length, gt_norm_scale) for _, start_idx, valid_length, gt_norm_scale in clips_data
            ]

            # Predict in batches
            clip_predictions = self._predict_clips_batch(clips)

            # Reconstruct full sequence with per-clip GT denormalization
            full_prediction = self._reconstruct_sequence(clip_predictions, clip_info)

            processing_time = time.time() - start_time

            return {
                "predictions": full_prediction,
                "processing_time": processing_time,
                "input_file": input_file,
                "success": True,
            }

        except Exception as e:
            processing_time = time.time() - start_time
            return {
                "error": str(e),
                "processing_time": processing_time,
                "input_file": input_file,
                "success": False,
            }

    def predict_all_files(self) -> List[Dict[str, Any]]:
        """Predict 3D poses for all input files."""
        input_files = self._collect_input_files()
        print(f"Found {len(input_files)} input files to process")

        results = []
        for i, input_file in enumerate(input_files, 1):
            print(f"Processing {i}/{len(input_files)}: {os.path.basename(input_file)}")
            result = self.predict_single_file(input_file)
            results.append(result)

        return results

    def save_predictions(self, results: List[Dict[str, Any]]) -> None:
        """Save prediction results to output directory.

        Args:
            results: List of prediction results

        """
        output_dir = self._get_output_dir()
        input_dir = self._get_input_dir()

        for result in results:
            if not result["success"]:
                print(f"Skipping failed prediction: {result['input_file']}")
                continue

            # Get relative path and create output path
            rel_path = os.path.relpath(result["input_file"], input_dir)
            output_path = os.path.join(output_dir, rel_path)

            # Create output directory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save predictions
            np.save(output_path, result["predictions"])
            print(f"Saved: {output_path}")
