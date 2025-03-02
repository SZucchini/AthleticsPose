"""Dataset for the AthleticPose dataset."""

import copy
import glob
import pickle
import random
from typing import Callable

import numpy as np
import torch
from natsort import natsorted
from torch.utils.data import Dataset


def read_pkl(data_path: str) -> dict:
    """Read a pickle file.

    Args:
        data_path (str): Path to the pickle file.

    Returns:
        data (dict): Data from the pickle file.

    """
    with open(data_path, "rb") as file:
        data = pickle.load(file)
    return data


def flip_data(
    data: np.ndarray,
    left: list[int] | None = None,
    right: list[int] | None = None,
) -> np.ndarray:
    """Flip the keypoints data.

    Args:
        data (np.ndarray): Keypoints data.
        left (list[int], optional): Indices of the left keypoints. Defaults to None.
        right (list[int], optional): Indices of the right keypoints. Defaults to None.

    Returns:
        np.ndarray: Flipped keypoints data.

    """
    if left is None:
        left = [1, 2, 3, 14, 15, 16]
    if right is None:
        right = [4, 5, 6, 11, 12, 13]

    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1
    flipped_data[..., left + right, :] = flipped_data[..., right + left, :]
    return flipped_data


class MotionDataset3D(Dataset):
    """3D motion dataset."""

    def __init__(self, data_dir: str, transform: Callable | None = None, flip: bool = True) -> None:
        """Initialize the dataset class.

        Args:
            data_dir (str): Data directory.
            transform (Callable, optional): Transform to apply to the data. Defaults to None.
            flip (bool, optional): Flag to flip the data. Defaults to True.

        """
        self.data_dir = data_dir
        self.transform = transform
        self.flip = flip
        self.file_list = self._generate_file_list()

    def _generate_file_list(self) -> list[str]:
        """Generate a list of files in the data root directory.

        Returns:
            file_list (list): List of files in the data root directory.

        """
        files = glob.glob(self.data_dir + "/*.pkl")
        file_list = natsorted(files)
        return file_list

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.

        """
        return len(self.file_list)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
        torch.FloatTensor,
    ]:
        """Get the item at the given index.

        Args:
            idx (int): Index of the item.

        Returns:
            input2d (torch.FloatTensor): 2D motion data.
            label3d (torch.FloatTensor): 3D motion data.
            p2mm (np.ndarray): Scale factors pixel coordinates to camera coordinates.
            norm_scale (np.ndarray): Normalization scale.

        """
        file_path = self.file_list[idx]
        motion_file = read_pkl(file_path)

        input2d = motion_file["input2d"]
        label3d = motion_file["label3d"]
        p2mm = motion_file["pixel_to_mm_scale"]
        norm_scale = motion_file["norm_scale"]

        if self.transform is not None:
            if self.flip and random.random() > 0.5:
                input2d = self.transform(input2d)
                label3d = self.transform(label3d)

        input2d = torch.FloatTensor(input2d).to(torch.float32)
        label3d = torch.FloatTensor(label3d).to(torch.float32)
        p2mm = torch.FloatTensor(p2mm).to(torch.float32)
        norm_scale = np.float32(norm_scale)

        return (
            input2d,
            label3d,
            p2mm,
            norm_scale,
        )
