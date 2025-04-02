"""Visualization functions."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.mplot3d import art3d

from athleticspose.statics.bones import h36m_bones


def align_poses(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Align predicted pose to target pose using Procrustes Analysis.

    Args:
        predicted (np.ndarray): Predicted keypoints. shape: (T, J, 3)
        target (np.ndarray): Target keypoints. shape: (T, J, 3)

    Returns:
        np.ndarray: Aligned predicted keypoints.

    """
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

    # Avoid improper rotations (reflections)
    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))  # Rotation

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * norm_x / norm_y  # Scale
    t = mu_x - a * np.matmul(mu_y, r)  # Translation

    # Apply rigid transformation
    predicted_aligned = a * np.matmul(predicted, r) + t
    return predicted_aligned


def set_lines(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, bones: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Set lines for 3D visualization.

    Args:
        x (np.ndarray): X coordinates.
        y (np.ndarray): Y coordinates.
        z (np.ndarray): Z coordinates.
        bones (np.ndarray): Bone connections.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Line coordinates for X, Y, Z.

    """
    line_x, line_y, line_z = [], [], []
    for bone in bones:
        line_x.append([x[bone[0]], x[bone[1]]])
        line_y.append([y[bone[0]], y[bone[1]]])
        line_z.append([z[bone[0]], z[bone[1]]])
    return np.array(line_x), np.array(line_y), np.array(line_z)


def visualize_pose_comparison(
    pred: np.ndarray,
    gt: np.ndarray,
    output_path: str,
    title: Optional[str] = None,
    is_aligned: bool = False,
) -> None:
    """Visualize prediction and ground truth in 3D.

    Args:
        pred: Prediction. shape: (T, J, 3)
        gt: Ground truth 3D pose. shape: (T, J, 3)
        output_path: Output path for visualization
        title: Optional title for the plot
        is_aligned: Whether the prediction is aligned to ground truth

    """
    fig = plt.figure(figsize=(10, 10))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax = fig.add_subplot(111, projection="3d")

    def draw_skeleton(ax: plt.Axes, kpts3d_pred: np.ndarray, kpts3d_gt: np.ndarray) -> None:
        """Draw skeleton."""
        ax.clear()
        if title is not None:
            ax.set_title(title)
        ax.view_init(elev=100, azim=90)
        ax.set_xlim(-1000, 1000)
        ax.set_ylim(-1000, 1000)
        ax.set_zlim(-1000, 1000)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # Draw ground truth (black)
        x0_gt, y0_gt, z0_gt = kpts3d_gt[:, 0], kpts3d_gt[:, 1], kpts3d_gt[:, 2]
        ax.plot(x0_gt, y0_gt, z0_gt, "k.", label="GT", markersize=18)
        x_bone_gt, y_bone_gt, z_bone_gt = set_lines(x0_gt, y0_gt, z0_gt, h36m_bones)
        for x, y, z in zip(x_bone_gt, y_bone_gt, z_bone_gt, strict=False):
            line = art3d.Line3D(x, y, z, color="black", linewidth=4.5)
            ax.add_line(line)

        # Draw prediction (red) with transparency
        x0_pred, y0_pred, z0_pred = kpts3d_pred[:, 0], kpts3d_pred[:, 1], kpts3d_pred[:, 2]
        ax.plot(x0_pred, y0_pred, z0_pred, "r.", label="Pred", markersize=18, alpha=0.8)
        x_bone_pred, y_bone_pred, z_bone_pred = set_lines(x0_pred, y0_pred, z0_pred, h36m_bones)
        for x, y, z in zip(x_bone_pred, y_bone_pred, z_bone_pred, strict=False):
            line = art3d.Line3D(x, y, z, color="red", linewidth=4.5, alpha=0.8)
            ax.add_line(line)

        ax.invert_xaxis()
        ax.invert_zaxis()

    def update_frame(fc: int) -> None:
        """Update frame."""
        draw_skeleton(ax, pred[fc], gt[fc])

    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=pred.shape[0],
        interval=30,
        repeat=False,
    )

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add suffix based on alignment status
    suffix = "_aligned" if is_aligned else "_raw"
    output_path = output_path.parent / f"{output_path.stem}{suffix}"

    # Save as MP4
    ani.save(f"{output_path}.mp4", fps=30)

    # Save as GIF with high quality
    ani.save(f"{output_path}.gif", fps=30, writer="pillow", dpi=200)
    plt.close()
