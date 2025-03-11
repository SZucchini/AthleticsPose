"""Test pose format conversion utilities."""

import numpy as np
import pytest

from athleticspose.preprocess.pose_format import PoseFormatConverter


@pytest.fixture
def sample_coco_data():
    """Create sample COCO format keypoints data.

    Returns:
        tuple: (keypoints, scores)
            - keypoints: Sample keypoints in COCO format (T=2, J=17, 2)
            - scores: Sample confidence scores (T=2, J=17)

    """
    # Create simple triangle-like pose for testing
    keypoints = np.zeros((2, 17, 2), dtype=np.float32)
    scores = np.ones((2, 17), dtype=np.float32)

    # Frame 1
    # Nose (0)
    keypoints[0, 0] = [50, 30]
    # Left/Right eye (1,2)
    keypoints[0, 1] = [45, 25]
    keypoints[0, 2] = [55, 25]
    # Left/Right ear (3,4)
    keypoints[0, 3] = [40, 27]
    keypoints[0, 4] = [60, 27]
    # Left/Right shoulder (5,6)
    keypoints[0, 5] = [30, 50]
    keypoints[0, 6] = [70, 50]
    # Left/Right elbow (7,8)
    keypoints[0, 7] = [20, 70]
    keypoints[0, 8] = [80, 70]
    # Left/Right wrist (9,10)
    keypoints[0, 9] = [15, 90]
    keypoints[0, 10] = [85, 90]
    # Left/Right hip (11,12)
    keypoints[0, 11] = [40, 100]
    keypoints[0, 12] = [60, 100]
    # Left/Right knee (13,14)
    keypoints[0, 13] = [35, 130]
    keypoints[0, 14] = [65, 130]
    # Left/Right ankle (15,16)
    keypoints[0, 15] = [30, 160]
    keypoints[0, 16] = [70, 160]

    # Frame 2 - Shift frame 1 slightly to simulate movement
    keypoints[1] = keypoints[0] + 5

    return keypoints, scores


def test_coco_to_h36m_shape(sample_coco_data):
    """Test if the output shapes are correct."""
    keypoints, scores = sample_coco_data
    h36m_kpts, h36m_scores = PoseFormatConverter.coco_to_h36m(keypoints, scores)

    assert h36m_kpts.shape == keypoints.shape
    assert h36m_scores.shape == scores.shape


def test_coco_to_h36m_conversion(sample_coco_data):
    """Test specific joint conversions and relationships."""
    keypoints, scores = sample_coco_data
    h36m_kpts, h36m_scores = PoseFormatConverter.coco_to_h36m(keypoints, scores)

    # Test pelvis position (mean of hips)
    expected_pelvis = np.mean(keypoints[:, 11:13, :], axis=1)
    np.testing.assert_allclose(h36m_kpts[:, 0, :], expected_pelvis)

    # Test thorax position
    # First calculate base position (mean of shoulders with neck adjustment)
    shoulders_mean = np.mean(keypoints[:, 5:7, :], axis=1)
    expected_thorax = shoulders_mean + (keypoints[:, 0, :] - shoulders_mean) / 3
    # Then adjust Y coordinate
    expected_thorax[:, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1) - keypoints[:, 0, 1]) * 2 / 3
    np.testing.assert_allclose(h36m_kpts[:, 8, :], expected_thorax, rtol=1e-5)

    # Test head position
    # X coordinate: mean of eyes and ears
    expected_head_x = np.mean(keypoints[:, 1:5, 0], axis=1)
    # Y coordinate: sum of eye heights - nose height
    expected_head_y = np.sum(keypoints[:, 1:3, 1], axis=1) - keypoints[:, 0, 1]
    np.testing.assert_allclose(h36m_kpts[:, 10, 0], expected_head_x)
    np.testing.assert_allclose(h36m_kpts[:, 10, 1], expected_head_y)


def test_coco_to_h36m_scores(sample_coco_data):
    """Test if confidence scores are correctly converted."""
    keypoints, scores = sample_coco_data
    _, h36m_scores = PoseFormatConverter.coco_to_h36m(keypoints, scores)

    # Test pelvis score (mean of hip scores)
    expected_pelvis_score = np.mean(scores[:, [11, 12]], axis=1)
    np.testing.assert_allclose(h36m_scores[:, 0], expected_pelvis_score)

    # Test thorax score (mean of shoulder scores)
    expected_thorax_score = np.mean(scores[:, [5, 6]], axis=1)
    np.testing.assert_allclose(h36m_scores[:, 8], expected_thorax_score)

    # Test head score (mean of eye and ear scores)
    expected_head_score = np.mean(scores[:, [1, 2, 3, 4]], axis=1)
    np.testing.assert_allclose(h36m_scores[:, 10], expected_head_score)
