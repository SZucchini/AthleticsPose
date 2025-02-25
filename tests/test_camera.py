"""Test the Camera class."""

import numpy as np
import pytest

from athleticspose.camera.camera import Camera


@pytest.fixture
def camera():
    """Camera instance."""
    return Camera("tests/sample_data/sample_camera_params.json")


def test_camera_initialization(camera: Camera):
    """Initialize the Camera class."""
    assert camera is not None
    assert hasattr(camera, "camera_params")
    assert isinstance(camera.camera_params, dict)


def test_world_to_camera(camera: Camera):
    """Test for the world_to_camera method."""
    frames, joints = 2, 3
    test_points = np.random.rand(frames, joints, 3)

    cam_idx = 0
    camera_coords = camera.world_to_camera(test_points, cam_idx)

    assert camera_coords.shape == (frames, joints, 3)
    assert not np.allclose(camera_coords, test_points)
    assert not np.any(np.isnan(camera_coords))
    assert not np.any(np.isinf(camera_coords))


def test_world_to_image(camera: Camera):
    """Test for the world_to_image method."""
    frames, joints = 2, 3
    test_points = np.random.rand(frames, joints, 3)

    cam_idx = 0
    image_coords = camera.world_to_image(test_points, cam_idx)

    assert image_coords.shape == (frames, joints, 2)
    assert not np.any(np.isnan(image_coords))
    assert not np.any(np.isinf(image_coords))


def test_camera_parameter_loading(camera: Camera):
    """Test for the camera parameter loading."""
    cam_idx = 0
    cam_params = camera.camera_params[cam_idx]

    required_keys = ["affine_intrinsics_matrix", "distortion", "extrinsic_matrix", "xyz"]
    for key in required_keys:
        assert key in cam_params

    assert np.array(cam_params["affine_intrinsics_matrix"]).shape == (3, 3)
    assert len(cam_params["distortion"]) == 5
    assert np.array(cam_params["extrinsic_matrix"]).shape == (3, 3)
    assert len(cam_params["xyz"]) == 3


def test_invalid_camera_file():
    """Test for an invalid camera file."""
    with pytest.raises(FileNotFoundError):
        Camera("non_existent_file.json")


def test_invalid_camera_index(camera: Camera):
    """Test for an invalid camera index."""
    test_points = np.zeros((2, 3, 3))
    invalid_cam_idx = 999

    with pytest.raises(ValueError):
        camera.world_to_image(test_points, invalid_cam_idx)


def test_camera_to_image(camera: Camera):
    """Test for the camera_to_image method."""
    frames, joints = 2, 3
    test_points = np.random.rand(frames, joints, 3)

    cam_idx = 0
    kpts_image = camera.camera_to_image(test_points, cam_idx)

    assert kpts_image.shape == (frames, joints, 2)
    assert not np.any(np.isnan(kpts_image))
    assert not np.any(np.isinf(kpts_image))


def test_camera_to_pixel3d(camera: Camera):
    """Test for the camera_to_pixel3d method."""
    frames, joints = 2, 3
    test_points = np.random.rand(frames, joints, 3)

    cam_idx = 0
    kpts_pixel, scales = camera.camera_to_pixel3d(test_points, cam_idx)
    kpts_image = camera.camera_to_image(test_points, cam_idx)

    assert kpts_pixel.shape == (frames, joints, 3)
    assert scales.shape == (frames,)
    assert np.allclose(kpts_pixel[:, :, :2], kpts_image)
