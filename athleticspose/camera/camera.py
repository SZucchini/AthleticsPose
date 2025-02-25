"""Camera class for loading and transforming 3D keypoints."""

import json

import numpy as np


class Camera:
    """Camera class."""

    def __init__(self, camera_file: str) -> None:
        """Initialize the Camera class.

        Args:
            camera_file (str): Path to the camera parameters json file.

        """
        self.camera_params = self._load_camera_params(camera_file)

    def _load_camera_params(self, camera_file: str) -> dict:
        """Load camera parameters from a JSON file.

        Args:
            camera_file (str): Path to the camera parameters file.

        Returns:
            dict: A dictionary containing camera parameters.

        """
        camera_params = {}
        with open(camera_file, "r") as f:
            camera_parameters = json.load(f)["Cameras"]

        for cam_idx, cam in enumerate(camera_parameters):
            camera_params[cam_idx] = self._parse_camera_params_from_json(cam)
        return camera_params

    def _parse_camera_params_from_json(self, cam: dict) -> dict:
        """Parse camera parameters from a JSON dictionary.

        Args:
            cam (dict): A dictionary containing camera parameters.

        Returns:
            dict : A dictionary containing parsed camera parameters.

        """
        scale_u = (cam["Intrinsic"]["SensorMaxU"] - cam["Intrinsic"]["SensorMinU"]) / (
            cam["FovVideoMax"]["Right"] - cam["FovVideoMax"]["Left"]
        )
        scale_v = (cam["Intrinsic"]["SensorMaxV"] - cam["Intrinsic"]["SensorMinV"]) / (
            cam["FovVideoMax"]["Bottom"] - cam["FovVideoMax"]["Top"]
        )

        return {
            "affine_intrinsics_matrix": np.array(
                [
                    [
                        cam["Intrinsic"]["FocalLengthU"] / scale_u,
                        cam["Intrinsic"]["Skew"],
                        cam["Intrinsic"]["CenterPointU"] / scale_u - cam["FovVideo"]["Left"],
                    ],
                    [
                        0.0,
                        cam["Intrinsic"]["FocalLengthV"] / scale_v,
                        cam["Intrinsic"]["CenterPointV"] / scale_v - cam["FovVideo"]["Top"],
                    ],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "distortion": np.array(
                [
                    cam["Intrinsic"]["RadialDistortion1"],
                    cam["Intrinsic"]["RadialDistortion2"],
                    cam["Intrinsic"]["RadialDistortion3"],
                    cam["Intrinsic"]["TangentalDistortion1"],
                    cam["Intrinsic"]["TangentalDistortion2"],
                ]
            ),
            "extrinsic_matrix": np.array(
                [
                    [cam["Transform"]["r11"], cam["Transform"]["r12"], cam["Transform"]["r13"]],
                    [cam["Transform"]["r21"], cam["Transform"]["r22"], cam["Transform"]["r23"]],
                    [cam["Transform"]["r31"], cam["Transform"]["r32"], cam["Transform"]["r33"]],
                ]
            ),
            "xyz": np.array([cam["Transform"]["x"], cam["Transform"]["y"], cam["Transform"]["z"]]),
        }

    def world_to_image(self, kpts_world: np.ndarray, cam_idx: int) -> np.ndarray:
        """Transform 3D keypoints from world coordinates to image coordinates.

        Args:
            kpts_world (np.ndarray): World coordinates of keypoints. The shape is (T, J, 3),
            cam_idx (int): The index of the camera to use.

        Returns:
            kpts_image (np.ndarray): Projected 2D keypoints. The shape is (T, J, 2),

        """
        if cam_idx not in self.camera_params:
            raise ValueError(f"Invalid camera index: {cam_idx}")

        intrinsics = self.camera_params[cam_idx]["affine_intrinsics_matrix"]
        rot_mat = self.camera_params[cam_idx]["extrinsic_matrix"].copy()
        rot_mat[1:] *= -1
        camera_position = self.camera_params[cam_idx]["xyz"]

        frames, joints, _ = kpts_world.shape
        translated_kpts = kpts_world.reshape(-1, 3) - camera_position
        kpts_cam = translated_kpts @ rot_mat.T

        kpts_image = np.zeros((frames * joints, 2))
        kpts_image[:, 0] = intrinsics[0, 0] * (kpts_cam[:, 0] / kpts_cam[:, 2]) + intrinsics[0, 2]
        kpts_image[:, 1] = intrinsics[1, 1] * (kpts_cam[:, 1] / kpts_cam[:, 2]) + intrinsics[1, 2]
        return kpts_image.reshape(frames, joints, 2)

    def world_to_camera(self, kpts_world: np.ndarray, cam_idx: int) -> np.ndarray:
        """Transform 3D keypoints from world coordinates to camera coordinates.

        Args:
            kpts_world (np.ndarray): World coordinates of keypoints with shape (T, J, 3).
            cam_idx (int): The index of the camera to use.

        Returns:
            kpts_cam (np.ndarray): Camera coordinates of keypoints with shape (T, J, 3).

        """
        if cam_idx not in self.camera_params:
            raise ValueError(f"Invalid camera index: {cam_idx}")

        rot_mat = self.camera_params[cam_idx]["extrinsic_matrix"].copy()
        rot_mat[1:] *= -1
        camera_position = self.camera_params[cam_idx]["xyz"]

        frames, joints, _ = kpts_world.shape
        translated_kpts = kpts_world.reshape(-1, 3) - camera_position
        kpts_cam = translated_kpts @ rot_mat.T
        return kpts_cam.reshape(frames, joints, 3)

    def camera_to_image(self, kpts_cam: np.ndarray, cam_idx: int) -> np.ndarray:
        """Transform 3D keypoints from camera coordinates to image coordinates.

        Args:
            kpts_cam (np.ndarray): Camera coordinates of keypoints with shape (T, J, 3).
            cam_idx (int): The index of the camera to use.

        Returns:
            kpts_image (np.ndarray): Projected 2D keypoints. The shape is (T, J, 2).

        """
        if cam_idx not in self.camera_params:
            raise ValueError(f"Invalid camera index: {cam_idx}")

        intrinsics = self.camera_params[cam_idx]["affine_intrinsics_matrix"]
        frames, joints, _ = kpts_cam.shape
        kpts_cam_reshaped = kpts_cam.reshape(-1, 3)
        kpts_image = np.zeros((frames * joints, 2))

        x = kpts_cam_reshaped[:, 0] / kpts_cam_reshaped[:, 2]
        y = kpts_cam_reshaped[:, 1] / kpts_cam_reshaped[:, 2]

        kpts_image[:, 0] = intrinsics[0, 0] * x + intrinsics[0, 2]
        kpts_image[:, 1] = intrinsics[1, 1] * y + intrinsics[1, 2]

        return kpts_image.reshape(frames, joints, 2)

    def camera_to_pixel3d(
        self, kpts_cam: np.ndarray, cam_idx: int, root_idx: int = 0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Transform 3D keypoints from camera coordinates to pixel coordinates with depth.

        Args:
            kpts_cam (np.ndarray): Camera coordinates of keypoints with shape (T, J, 3).
            cam_idx (int): The index of the camera to use.
            root_idx (int): Index of the root joint.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - kpts_pixel (np.ndarray): Pixel coordinates with scaled depth. Shape is (T, J, 3).
                - scales (np.ndarray): Scale factors for each frame. Shape is (T,).

        """
        if cam_idx not in self.camera_params:
            raise ValueError(f"Invalid camera index: {cam_idx}")

        frames, _, _ = kpts_cam.shape
        kpts_pixel = np.zeros_like(kpts_cam)
        scales = np.zeros(frames)

        intrinsics = self.camera_params[cam_idx]["affine_intrinsics_matrix"]
        kpts_pixel[..., 0] = intrinsics[0, 0] * (kpts_cam[..., 0] / kpts_cam[..., 2]) + intrinsics[0, 2]
        kpts_pixel[..., 1] = intrinsics[1, 1] * (kpts_cam[..., 1] / kpts_cam[..., 2]) + intrinsics[1, 2]

        root_joints = kpts_cam[:, root_idx]
        for i in range(frames):
            ref_near = root_joints[i] - np.array([1000, 0, 0])
            ref_far = root_joints[i] + np.array([1000, 0, 0])
            x_near = intrinsics[0, 0] * (ref_near[0] / ref_near[2]) + intrinsics[0, 2]
            x_far = intrinsics[0, 0] * (ref_far[0] / ref_far[2]) + intrinsics[0, 2]
            scales[i] = (x_far - x_near) / 2000
            kpts_pixel[i, :, 2] = (kpts_cam[i, :, 2] - root_joints[i, 2]) * scales[i]

        return kpts_pixel, scales
