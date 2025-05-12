"""Utility functions for the preprocess module."""

import numpy as np

from athleticspose.statics.joints import joint_index_h36m_to_hj_mocap, joint_index_h36m_to_mocap


def mocap_to_h36m(mocap_markers: np.ndarray, hj_mocap: bool = False) -> np.ndarray:
    """Convert mocap markers to Human3.6M format.

    Args:
        mocap_markers: The mocap markers to convert. The shape should be (T, J, 3).
        hj_mocap: If True, convert to Human3.6M format with HJ markers. Defaults to False.

    Returns:
        h36m_markers: The Human3.6M markers. The shape is (T, 17, 3).

    """
    h36m_markers = np.zeros((mocap_markers.shape[0], 17, 3))
    if hj_mocap:
        joint_index = joint_index_h36m_to_hj_mocap
    else:
        joint_index = joint_index_h36m_to_mocap
    for i, joint_idx in joint_index.items():
        h36m_markers[:, i, :] = np.mean(mocap_markers[:, joint_idx, :], axis=1)
    return h36m_markers


def normalize_kpts(kpts: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize the keypoints to the range [-1, 1].

    Args:
        kpts: The keypoints to normalize. The shape should be (T, 17, 3).

    Returns:
        kpts_norm: The normalized keypoints. The shape is (T, 17, 3).
        norm_scale: The normalization scale used for denormalization.

    """
    kpts = kpts - kpts[:, 0:1, :]
    norm_scale = np.max(np.abs(kpts))
    kpts_norm = kpts / norm_scale
    return kpts_norm, norm_scale


def denormalize_kpts(kpts_norm: np.ndarray, norm_scale: float) -> np.ndarray:
    """Denormalize the keypoints scale.

    Args:
        kpts_norm: The normalized keypoints. The shape is (T, 17, 3).
        norm_scale: The normalization scale used for denormalization.

    Returns:
        kpts_denom: The denormalized keypoints. The shape is (T, 17, 3).

    """
    kpts_denom = kpts_norm * norm_scale
    return kpts_denom


def create_frame_indices(
    sequence_length: int,
    target_length: int,
    replay: bool = False,
    random_sampling: bool = True,
) -> np.ndarray:
    """Create indices for frame sampling with specified length.

    Args:
        sequence_length (int): Length of the original sequence.
        target_length (int): Desired length of the sampled sequence.
        replay (bool): If True, creates continuous frame indices that can be replayed.
        random_sampling (bool): If True, applies random sampling within intervals.

    Returns:
        indices (np.ndarray): Array of frame indices for sampling.

    """
    rng = np.random.default_rng()
    if replay:
        if sequence_length > target_length:
            start_idx = rng.integers(sequence_length - target_length)
            return np.array(range(start_idx, start_idx + target_length))
        else:
            return np.array(range(target_length)) % sequence_length
    else:
        if random_sampling:
            sample_points = np.linspace(0, sequence_length, num=target_length, endpoint=False)
            if sequence_length < target_length:
                floor_values = np.floor(sample_points)
                ceil_values = np.ceil(sample_points)
                random_choice = rng.integers(2, size=sample_points.shape)
                indices = np.sort(random_choice * floor_values + (1 - random_choice) * ceil_values)
            else:
                interval = sample_points[1] - sample_points[0]
                indices = rng.random(sample_points.shape) * interval + sample_points
            indices = np.clip(indices, a_min=0, a_max=sequence_length - 1).astype(np.uint32)
        else:
            indices = np.linspace(0, sequence_length, num=target_length, endpoint=False, dtype=int)
        return indices


def split_clips(
    total_frames: int,
    clip_length: int = 81,
    stride: int = 27,
) -> list[list[int]]:
    """Split a sequence into overlapping clips of specified length.

    Args:
        total_frames (int): Total number of frames in the sequence.
        clip_length (int): Number of frames in each clip.
        stride (int): Number of frames to move forward for each new clip.

    Returns:
        clips (list[list[int]]): List of frame indices for each clip.

    """
    assert stride > 0, "Stride must be greater than 0."

    clips = []
    clip_start = 0

    while clip_start < total_frames:
        if total_frames - clip_start < clip_length // 2:
            break
        if clip_start + clip_length > total_frames:
            clip_indices = create_frame_indices(total_frames - clip_start, clip_length) + clip_start
            clips.append(clip_indices)
            break
        else:
            clips.append(range(clip_start, clip_start + clip_length))
            clip_start += stride

    return clips


def sampling_camera_and_target_positions(
    kpts_world: np.ndarray,
    num_cameras: int = 8,
    fov_deg: float = 60.0,
    distance_factor: float = 2.0,
    target_noise_scale: float = 0.05,
    min_polar_angle_deg: float = 30.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample camera and target positions.

    Args:
        kpts_world (np.ndarray): 3D keypoints in world coordinates. (T, J, 3)
        num_cameras (int): Number of cameras to sample.
        fov_deg (float): Field of view in degrees.
        distance_factor (float): Distance factor for camera placement.
        target_noise_scale (float): Standard deviation of noise for target positions.
        min_polar_angle_deg (float): Minimum polar angle in degrees.
        rng (np.random.Generator | None): Random number generator.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of camera positions and target positions.
            Each of shape (num_cameras, 3).

    """
    if rng is None:
        rng = np.random.default_rng()

    pts = np.asarray(kpts_world, dtype=np.float64).reshape(-1, 3)
    if pts.size == 0:
        raise ValueError("kpts_world must contain at least one point.")

    min_xyz, max_xyz = pts.min(axis=0), pts.max(axis=0)
    center = 0.5 * (min_xyz + max_xyz)
    radius = np.linalg.norm(pts - center, axis=1).max()
    if radius == 0.0:
        radius = 1e-6

    fov_rad = np.deg2rad(fov_deg)
    d_min = radius / np.tan(fov_rad * 0.5)
    distance = max(distance_factor * radius, d_min * 1.05)

    theta_min = np.deg2rad(min_polar_angle_deg)
    if not (0.0 <= min_polar_angle_deg <= 90.0):
        raise ValueError("min_polar_angle_deg must be in [0, 90]")
    cos_theta_max = np.cos(theta_min)

    phi = rng.uniform(0.0, 2.0 * np.pi, size=num_cameras)
    cos_theta = rng.uniform(0.0, cos_theta_max, num_cameras)
    sin_theta = np.sqrt(1.0 - cos_theta**2)

    dirs = np.column_stack(
        (
            sin_theta * np.cos(phi),
            sin_theta * np.sin(phi),
            cos_theta,
        )
    )
    camera_positions = center + distance * dirs

    noise_sigma = target_noise_scale * radius
    target_positions = center + rng.normal(
        loc=0.0,
        scale=noise_sigma,
        size=(num_cameras, 3),
    )

    return camera_positions.astype(np.float32), target_positions.astype(np.float32)
