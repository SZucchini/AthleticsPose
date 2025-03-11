"""Pose format conversion utilities."""

from typing import Tuple

import numpy as np


class PoseFormatConverter:
    """Convert between different pose formats."""

    # H36M and COCO joint mapping
    H36M_COCO_ORDER = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
    COCO_ORDER = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    SPPLE_KEYPOINTS = [10, 8, 0, 7]  # head, thorax, pelvis, spine

    @classmethod
    def coco_to_h36m(cls, keypoints: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Convert COCO format keypoints to H36M format.

        Args:
            keypoints (np.ndarray): Keypoints in COCO format. Shape: (T, 17, 2).
            scores (np.ndarray): Confidence scores. Shape: (T, 17).

        Returns:
            tuple[np.ndarray, np.ndarray]: H36M format keypoints and scores.
                - keypoints: Shape (T, 17, 2)
                - scores: Shape (T, 17)

        """
        temporal = keypoints.shape[0]
        kpts_h36m = np.zeros_like(keypoints, dtype=np.float32)
        htps_kpts = np.zeros((temporal, 4, 2), dtype=np.float32)

        # Calculate HTPS (head, thorax, pelvis, spine) keypoints
        # Head: mean of eyes and ears
        htps_kpts[:, 0, 0] = np.mean(keypoints[:, 1:5, 0], axis=1, dtype=np.float32)
        htps_kpts[:, 0, 1] = np.sum(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]

        # Thorax: mean of shoulders + neck adjustment
        htps_kpts[:, 1, :] = np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)
        htps_kpts[:, 1, :] += (keypoints[:, 0, :] - htps_kpts[:, 1, :]) / 3

        # Pelvis: mean of hips
        htps_kpts[:, 2, :] = np.mean(keypoints[:, 11:13, :], axis=1, dtype=np.float32)

        # Spine: mean of shoulders and hips
        htps_kpts[:, 3, :] = np.mean(keypoints[:, [5, 6, 11, 12], :], axis=1, dtype=np.float32)

        # Assign HTPS keypoints
        kpts_h36m[:, cls.SPPLE_KEYPOINTS, :] = htps_kpts

        # Map COCO keypoints to H36M
        kpts_h36m[:, cls.H36M_COCO_ORDER, :] = keypoints[:, cls.COCO_ORDER, :]

        # Adjust neck position
        kpts_h36m[:, 9, :] -= (kpts_h36m[:, 9, :] - np.mean(keypoints[:, 5:7, :], axis=1, dtype=np.float32)) / 4

        # Adjust spine X coordinate
        kpts_h36m[:, 7, 0] += 2 * (kpts_h36m[:, 7, 0] - np.mean(kpts_h36m[:, [0, 8], 0], axis=1, dtype=np.float32))

        # Adjust thorax Y coordinate
        kpts_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]) * 2 / 3

        # Convert scores to H36M format
        h36m_scores = np.zeros_like(scores, dtype=np.float32)
        h36m_scores[:, cls.H36M_COCO_ORDER] = scores[:, cls.COCO_ORDER]
        h36m_scores[:, 0] = np.mean(scores[:, [11, 12]], axis=1, dtype=np.float32)  # pelvis
        h36m_scores[:, 8] = np.mean(scores[:, [5, 6]], axis=1, dtype=np.float32)  # thorax
        h36m_scores[:, 7] = np.mean(h36m_scores[:, [0, 8]], axis=1, dtype=np.float32)  # spine
        h36m_scores[:, 10] = np.mean(scores[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)  # head

        return kpts_h36m, h36m_scores
