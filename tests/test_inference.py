"""Test inference functionality."""

import numpy as np
import pytest
import torch

from athleticspose.inferencer import Pose3DInferencer

CHECKPOINT_PATH = "work_dir/20250302_110906/best.ckpt"


@pytest.fixture(autouse=True)
def force_cpu(monkeypatch):
    """Force CPU device for all tests."""

    def mock_cuda_available():
        return False

    monkeypatch.setattr(torch.cuda, "is_available", mock_cuda_available)


def test_split_into_clips():
    """Test splitting poses into clips."""
    inferencer = Pose3DInferencer(CHECKPOINT_PATH)

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
    inferencer = Pose3DInferencer(CHECKPOINT_PATH)

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
    batch = inferencer.preprocess(poses_2d)

    # Check basic shapes
    assert isinstance(batch, torch.Tensor)
    assert batch.shape == (2, 81, 17, 3)  # 2 clips for 100 frames
    assert batch.device == torch.device("cpu")  # Verify CPU device

    # Verify normalization (root joint should be roughly at origin)
    root_positions = batch[..., 0, :2].cpu().numpy()
    assert np.allclose(root_positions, 0, atol=1e-6)

    # Verify scores are preserved
    original_scores = poses_2d[..., 2]
    for clip_idx in range(batch.shape[0]):
        start_idx = clip_idx * 81
        end_idx = min(start_idx + 81, poses_2d.shape[0])
        valid_frames = end_idx - start_idx
        assert np.allclose(
            batch[clip_idx, :valid_frames, :, 2].cpu().numpy(),
            original_scores[start_idx:end_idx],
        )


def test_postprocess():
    """Test postprocessing function."""
    inferencer = Pose3DInferencer(CHECKPOINT_PATH)

    # Create test predictions
    original_length = 100
    predictions = np.random.rand(2, 81, 17, 3)  # 2 clips

    # Run postprocessing
    processed = inferencer.postprocess(predictions, original_length)

    # Check output shape
    assert processed.shape == (original_length, 17, 3)

    # Verify correct frame selection
    assert np.array_equal(processed[:81], predictions[0][:81])
    assert np.array_equal(processed[81:], predictions[1][:19])


def test_predict():
    """Test complete inference pipeline."""
    # Create test data
    poses_2d = np.random.rand(200, 17, 3)  # 200 frames
    coords = np.random.rand(200, 17, 2) * 100
    scores = np.random.rand(200, 17)
    poses_2d[..., :2] = coords
    poses_2d[..., 2] = scores

    # Initialize inferencer
    inferencer = Pose3DInferencer(CHECKPOINT_PATH)

    # Verify using CPU
    assert inferencer.device == torch.device("cpu")

    # Test with default batch size
    poses_3d = inferencer.predict(poses_2d)
    assert poses_3d.shape == (200, 17, 3)  # Same time length as input
    assert not np.any(np.isnan(poses_3d))  # No NaN values
    assert np.all(np.isfinite(poses_3d))  # No infinite values

    # Test with custom batch size
    poses_3d_batched = inferencer.predict(poses_2d, batch_size=1)
    assert poses_3d_batched.shape == (200, 17, 3)
    assert not np.any(np.isnan(poses_3d_batched))
    assert np.all(np.isfinite(poses_3d_batched))
