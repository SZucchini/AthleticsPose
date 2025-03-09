"""Test inference functionality."""

import numpy as np
import pytest
import torch

from athleticspose.inference import Pose3DInference

CHECKPOINT_PATH = "work_dir/20250302_110906/best.ckpt"


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Force CPU device for all tests."""

    def mock_cuda_available():
        return False

    monkeypatch.setattr(torch.cuda, "is_available", mock_cuda_available)


def test_split_into_clips():
    """Test splitting poses into clips."""
    inferencer = Pose3DInference(CHECKPOINT_PATH)

    # Test exact division
    poses = np.random.rand(81, 17, 3)
    clips = inferencer._split_into_clips(poses)
    assert len(clips) == 1
    assert clips[0].shape == (81, 17, 3)
    assert np.array_equal(clips[0], poses)

    # Test with remainder
    poses = np.random.rand(100, 17, 3)
    clips = inferencer._split_into_clips(poses)
    assert len(clips) == 2
    assert clips[0].shape == (81, 17, 3)
    assert clips[1].shape == (81, 17, 3)
    # Verify first clip
    assert np.array_equal(clips[0], poses[0:81])
    # Verify padded clip
    assert np.array_equal(clips[1][:19], poses[81:])
    assert np.all(clips[1][19:] == 0)


def test_preprocess():
    """Test preprocessing function."""
    inferencer = Pose3DInference(CHECKPOINT_PATH)

    # Create test data with both coordinates and scores
    poses_2d = np.zeros((100, 17, 3))
    coords = np.random.rand(100, 17, 2) * 100  # Random coordinates
    scores = np.random.rand(100, 17)  # Random scores
    poses_2d[..., :2] = coords
    poses_2d[..., 2] = scores

    # Add offset to test normalization
    root_offset = np.random.rand(1, 1, 2) * 1000
    poses_2d[..., :2] += root_offset

    # Preprocess
    batch, norm_info = inferencer.preprocess(poses_2d)

    # Check basic shapes
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (2, 81, 17, 3)  # 2 clips for 100 frames
    assert batch.device == torch.device("cpu")  # Verify CPU device

    # Verify normalization (root should be at origin)
    assert torch.allclose(batch[..., 0, :2], torch.zeros(2))

    # Verify scores are preserved
    for clip_idx, (_, clip_scores) in enumerate(norm_info):
        start_idx = clip_idx * 81
        end_idx = min(start_idx + 81, poses_2d.shape[0])
        valid_frames = end_idx - start_idx
        assert np.array_equal(clip_scores[:valid_frames], scores[start_idx:end_idx])


def test_postprocess():
    """Test postprocessing function."""
    inferencer = Pose3DInference(CHECKPOINT_PATH)

    # Create test predictions and normalization info
    original_length = 100
    predictions = np.random.rand(2, 81, 17, 3)  # 2 clips
    norm_scales = [2.5, 2.0]  # Random scales
    scores = [np.random.rand(81, 17), np.random.rand(81, 17)]
    norm_info = list(zip(norm_scales, scores, strict=False))

    # Run postprocessing
    processed = inferencer.postprocess(predictions, original_length, norm_info)

    # Check output shape
    assert processed.shape == (original_length, 17, 3)

    # Verify scale restoration
    first_clip = predictions[0] * norm_scales[0]
    assert np.array_equal(processed[:81], first_clip)

    second_clip_valid_frames = original_length - 81
    second_clip = predictions[1] * norm_scales[1]
    assert np.array_equal(processed[81:], second_clip[:second_clip_valid_frames])


def test_end_to_end():
    """Test complete inference pipeline."""
    # Create test data
    poses_2d = np.random.rand(200, 17, 3)  # 200 frames
    coords = np.random.rand(200, 17, 2) * 100
    scores = np.random.rand(200, 17)
    poses_2d[..., :2] = coords
    poses_2d[..., 2] = scores

    # Initialize inferencer
    inferencer = Pose3DInference(CHECKPOINT_PATH)

    # Verify using CPU
    assert inferencer.device == torch.device("cpu")

    # Run inference
    poses_3d = inferencer.predict(poses_2d)

    # Verify output
    assert poses_3d.shape == (200, 17, 3)  # Same time length as input
    assert not np.any(np.isnan(poses_3d))  # No NaN values
    assert np.all(np.isfinite(poses_3d))  # No infinite values
