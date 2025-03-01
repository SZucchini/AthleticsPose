"""Loss functions for the Pose3D model."""

import numpy as np
import torch


def calc_mpjpe(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean per-joint position error (i.e. mean Euclidean distance).

    Args:
        predicted (np.ndarray): Predicted keypoints. shape: (B, T, J, 3)
        target (np.ndarray): Target keypoints. shape: (B, T, J, 3)

    Returns:
        np.ndarray: Mean per-joint position error.

    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape) - 1), axis=1)


def p_mpjpe(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pose error: MPJPE after rigid alignment (scale, rotation, and translation).

    Args:
        predicted (np.ndarray): Predicted keypoints. shape: (T, J, 3)
        target (np.ndarray): Target keypoints. shape: (T, J, 3)

    Returns:
        np.ndarray: PA-MPJPE.

    """
    assert predicted.shape == target.shape
    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)

    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0**2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0**2, axis=(1, 2), keepdims=True))

    x0 /= norm_x
    y0 /= norm_y

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, v_t = np.linalg.svd(h)
    v = v_t.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * norm_x / norm_y  # Scale
    t = mu_x - a * np.matmul(mu_y, r)  # Translation
    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, r) + t
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape) - 1), axis=1)


def loss_mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean per-joint position error (i.e. mean Euclidean distance).

    Args:
        predicted (torch.Tensor): Predicted keypoints. shape: (B, T, J, 3)
        target (torch.Tensor): Target keypoints. shape: (B, T, J, 3)

    Returns:
        torch.Tensor: Mean per-joint position error.

    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def n_mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate Normalized MPJPE (scale only).

    Args:
        predicted (torch.Tensor): Predicted keypoints. shape: (B, T, J, 3)
        target (torch.Tensor): Target keypoints. shape: (B, T, J, 3)

    Returns:
        torch.Tensor: Normalized MPJPE.

    """
    assert predicted.shape == target.shape
    norm_predicted = torch.mean(torch.sum(predicted**2, dim=3, keepdim=True), dim=2, keepdim=True)
    norm_target = torch.mean(torch.sum(target * predicted, dim=3, keepdim=True), dim=2, keepdim=True)
    scale = norm_target / norm_predicted
    return loss_mpjpe(scale * predicted, target)


def loss_velocity(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Calculate mean per-joint velocity error (i.e. mean Euclidean distance of the 1st derivative).

    Args:
        predicted (torch.Tensor): Predicted keypoints. shape: (B, T, J, 3)
        target (torch.Tensor): Target keypoints. shape: (B, T, J, 3)

    Returns:
        torch.Tensor: Mean per-joint velocity error.

    """
    assert predicted.shape == target.shape
    if predicted.shape[1] <= 1:
        return torch.FloatTensor(1).fill_(0.0)[0].to(predicted.device)
    velocity_predicted = predicted[:, 1:] - predicted[:, :-1]
    velocity_target = target[:, 1:] - target[:, :-1]
    return torch.mean(torch.norm(velocity_predicted - velocity_target, dim=-1))
