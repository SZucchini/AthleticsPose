"""Test video pose estimation pipeline."""

import cv2
import numpy as np
import pytest

from athleticspose.pipeline.video_estimator import VideoEstimator

# Model URLs provided
DET_MODEL = "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_m_8xb32-300e_coco/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth"
POSE_MODEL = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"


@pytest.fixture
def video_estimator():
    """Create VideoEstimator instance."""
    estimator = VideoEstimator(
        mmdet_config="athleticspose/mmpose/mmdet_cfg/rtmdet.py",
        mmpose_config="athleticspose/mmpose/config/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py",
        det_weights=DET_MODEL,
        pose_weights=POSE_MODEL,
        device="cuda:0",  # テスト環境ではCUDAが利用可能と仮定
    )
    return estimator


def test_process_frame(video_estimator):
    """Test processing a single frame from sample video."""
    # Load first frame from sample video
    cap = cv2.VideoCapture("tests/sample_data/test_small.mp4")
    ret, frame = cap.read()
    cap.release()
    assert ret, "Failed to read sample video frame"

    # Process frame
    keypoints, scores = video_estimator.process_frame(frame)

    # Basic shape and type checks
    assert keypoints is not None
    assert scores is not None
    assert isinstance(keypoints, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert keypoints.shape == (17, 2)  # 17 joints, x-y coordinates
    assert scores.shape == (17,)  # confidence score for each joint


def test_process_video(video_estimator):
    """Test processing entire sample video."""
    keypoints, scores = video_estimator.process_video("tests/sample_data/test_small.mp4")

    # Check output shapes and types
    assert isinstance(keypoints, np.ndarray)
    assert isinstance(scores, np.ndarray)
    assert keypoints.ndim == 3  # (T, 17, 2)
    assert scores.ndim == 2  # (T, 17)
    assert keypoints.shape[1] == 17
    assert keypoints.shape[2] == 2
    assert scores.shape[1] == 17

    # Check if any valid poses were detected
    assert len(keypoints) > 0
    assert len(scores) > 0


def test_process_video_invalid_file(video_estimator):
    """Test handling invalid video file."""
    with pytest.raises(ValueError, match="Failed to open video file"):
        video_estimator.process_video("nonexistent.mp4")
