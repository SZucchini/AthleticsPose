"""Pipeline for pose estimation from video input."""

import os
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from athleticspose.inferencer import Pose3DInferencer
from athleticspose.pipeline.video_estimator import VideoEstimator
from athleticspose.preprocess.pose_format import PoseFormatConverter

DEFAULT_MMDET_CONFIG = "athleticspose/mmpose/mmdet_cfg/rtmdet.py"
DEFAULT_MMPOSE_CONFIG = "athleticspose/mmpose/config/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
DEFAULT_DET_WEIGHTS = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)
DEFAULT_POSE_WEIGHTS = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"


class VideoPosePipeline:
    """Pipeline for video pose estimation."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        max_duration: float = 3.0,
        target_resolution: Tuple[int, int] = (1280, 720),
    ):
        """Initialize pipeline.

        Args:
            checkpoint_path (str): Path to 3D pose model checkpoint
            device (str, optional): Device to run models on. Defaults to "cuda:0".
            max_duration (float, optional): Maximum video duration in seconds. Defaults to 3.0.
            target_resolution (tuple[int, int], optional): Target resolution (width, height).
                Defaults to (1280, 720).

        """
        self.device = device
        self.max_duration = max_duration
        self.target_resolution = target_resolution

        # Initialize 2D pose estimator
        self.video_estimator = VideoEstimator(
            mmdet_config=DEFAULT_MMDET_CONFIG,
            mmpose_config=DEFAULT_MMPOSE_CONFIG,
            det_weights=DEFAULT_DET_WEIGHTS,
            pose_weights=DEFAULT_POSE_WEIGHTS,
            device=device,
        )

        # Initialize 3D pose inferencer
        self.pose_inferencer = Pose3DInferencer(checkpoint_path=checkpoint_path)

    def preprocess_video(self, video_path: str) -> str:
        """Preprocess video to match requirements.

        Args:
            video_path (str): Path to input video

        Returns:
            str: Path to preprocessed video

        Raises:
            ValueError: If video duration exceeds max_duration

        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        # Check duration
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        if duration > self.max_duration:
            raise ValueError(
                f"Video duration ({duration:.1f}s) exceeds maximum allowed duration ({self.max_duration}s)"
            )

        # Get original resolution
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Check if resizing is needed
        needs_resize = (width, height) != self.target_resolution

        if needs_resize:
            # Create temporary file for resized video
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "resized_video.mp4")

            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                self.target_resolution,
            )

            # Process frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame
                resized_frame = cv2.resize(frame, self.target_resolution)
                out.write(resized_frame)

            out.release()
            cap.release()
            return output_path

        cap.release()
        return video_path

    def process_video(self, video_path: str, batch_size: Optional[int] = None) -> np.ndarray:
        """Process video and extract 3D poses.

        Args:
            video_path (str): Path to input video
            batch_size (int, optional): Batch size for 3D pose inference.
                Defaults to None.

        Returns:
            np.ndarray: 3D pose predictions. Shape: (T, J, 3).

        """
        # Preprocess video
        processed_video = self.preprocess_video(video_path)

        # Extract 2D poses
        keypoints_2d, scores = self.video_estimator.process_video(processed_video)

        # Convert COCO format to H36M format
        keypoints_h36m, scores_h36m = PoseFormatConverter.coco_to_h36m(keypoints_2d, scores)

        # Stack keypoints and scores
        poses_2d = np.concatenate([keypoints_h36m, scores_h36m[..., None]], axis=-1)

        # Predict 3D poses
        poses_3d = self.pose_inferencer.predict(poses_2d, batch_size=batch_size)

        # Clean up temporary file if created
        if processed_video != video_path:
            os.unlink(processed_video)
            os.rmdir(os.path.dirname(processed_video))

        return poses_3d
