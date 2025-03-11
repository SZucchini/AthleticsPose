"""Visualization module for 3D pose animation."""

import os
import tempfile
from typing import Optional

import ffmpeg
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import art3d

# Human3.6M joint keypoints definition
H36M_KEY = {
    "MidHip": 0,
    "RHip": 1,
    "RKnee": 2,
    "RAnkle": 3,
    "LHip": 4,
    "LKnee": 5,
    "LAnkle": 6,
    "Spine": 7,
    "Thorax": 8,
    "Nose": 9,
    "Head": 10,
    "LShoulder": 11,
    "LElbow": 12,
    "LWrist": 13,
    "RShoulder": 14,
    "RElbow": 15,
    "RWrist": 16,
}

# Bone connections for visualization
H36M_BONE = np.array(
    [
        [H36M_KEY["Head"], H36M_KEY["Nose"]],
        [H36M_KEY["Nose"], H36M_KEY["Thorax"]],
        [H36M_KEY["Thorax"], H36M_KEY["Spine"]],
        [H36M_KEY["Thorax"], H36M_KEY["RShoulder"]],
        [H36M_KEY["Thorax"], H36M_KEY["LShoulder"]],
        [H36M_KEY["RShoulder"], H36M_KEY["RElbow"]],
        [H36M_KEY["LShoulder"], H36M_KEY["LElbow"]],
        [H36M_KEY["RWrist"], H36M_KEY["RElbow"]],
        [H36M_KEY["LWrist"], H36M_KEY["LElbow"]],
        [H36M_KEY["Spine"], H36M_KEY["MidHip"]],
        [H36M_KEY["RHip"], H36M_KEY["MidHip"]],
        [H36M_KEY["LHip"], H36M_KEY["MidHip"]],
        [H36M_KEY["RHip"], H36M_KEY["RKnee"]],
        [H36M_KEY["RKnee"], H36M_KEY["RAnkle"]],
        [H36M_KEY["LHip"], H36M_KEY["LKnee"]],
        [H36M_KEY["LKnee"], H36M_KEY["LAnkle"]],
    ],
    dtype=np.int64,
)

# Color scheme for bones
BONE_COLORS = []
for i in range(len(H36M_BONE)):
    if i in [3, 5, 7, 10, 12, 13]:
        BONE_COLORS.append("blue")
    elif i in [4, 6, 8, 11, 14, 15]:
        BONE_COLORS.append("red")
    else:
        BONE_COLORS.append("black")


class PoseVisualizer:
    """Class for visualizing 3D pose data."""

    def __init__(self, output_fps: int = 30):
        """Initialize visualizer.

        Args:
            output_fps (int, optional): Output video FPS. Defaults to 30.

        """
        self.output_fps = output_fps

    def _set_lines(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set lines for 3D visualization.

        Args:
            x (np.ndarray): X coordinates.
            y (np.ndarray): Y coordinates.
            z (np.ndarray): Z coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Line coordinates
                - line_x: X coordinates for lines
                - line_y: Y coordinates for lines
                - line_z: Z coordinates for lines

        """
        line_x = []
        line_y = []
        line_z = []

        for bone in H36M_BONE:
            line_x.append([x[bone[0]], x[bone[1]]])
            line_y.append([y[bone[0]], y[bone[1]]])
            line_z.append([z[bone[0]], z[bone[1]]])

        return np.array(line_x), np.array(line_y), np.array(line_z)

    def _draw_skeleton(self, ax: Axes, kpts3d: np.ndarray) -> None:
        """Draw skeleton on 3D axes.

        Args:
            ax (Axes): Matplotlib 3D axes
            kpts3d (np.ndarray): 3D keypoints. Shape: (J, 3)

        """
        ax.clear()
        ax.view_init(elev=100, azim=90)

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        x0 = kpts3d[:, 0]
        y0 = kpts3d[:, 1]
        z0 = kpts3d[:, 2]
        ax.scatter(x0, y0, z0, c="k", marker=".")

        x_bone, y_bone, z_bone = self._set_lines(x0, y0, z0)
        for x, y, z, color in zip(x_bone, y_bone, z_bone, BONE_COLORS, strict=False):
            line = art3d.Line3D(x, y, z, color=color)
            ax.add_line(line)

        # Invert axes for better visualization
        ax.invert_xaxis()
        ax.invert_zaxis()

        # Remove axis labels and ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")

    def create_animation(self, kpts3d: np.ndarray, output_path: Optional[str] = None) -> Optional[str]:
        """Create animation from 3D keypoints.

        Args:
            kpts3d (np.ndarray): 3D keypoints. Shape: (F, J, 3).
            output_path (Optional[str], optional): Output video path.
                If None, create temporary file. Defaults to None.

        Returns:
            Optional[str]: Path to output video if successful, None otherwise

        """
        # Create figure
        fig = plt.figure(figsize=(10, 10))
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax = fig.add_subplot(111, projection="3d")

        def update_frame(fc: int) -> None:
            self._draw_skeleton(ax, kpts3d[fc, :, :])

        # Create animation
        ani = animation.FuncAnimation(
            fig,
            update_frame,
            frames=kpts3d.shape[0],
            interval=1000 // self.output_fps,
            repeat=False,
        )

        # Save animation
        if output_path is None:
            temp_dir = tempfile.mkdtemp()
            output_path = os.path.join(temp_dir, "animation.mp4")

        # Save animation as temporary file
        temp_path = output_path + ".temp.mp4"
        ani.save(temp_path, writer="ffmpeg", fps=self.output_fps)

        # Convert to web-compatible format
        stream = ffmpeg.input(temp_path)
        stream = ffmpeg.output(
            stream,
            output_path,
            vcodec="h264",
            acodec="aac",
            vf="format=yuv420p",
            movflags="+faststart",
        )
        ffmpeg.run(stream, overwrite_output=True, capture_stdout=True, capture_stderr=True)

        # Clean up
        plt.close(fig)
        os.unlink(temp_path)

        return output_path
