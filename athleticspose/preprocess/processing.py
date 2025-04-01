"""Process the AthleticPose dataset to the format required by the 3D pose estimation model."""

import os
import pickle

import numpy as np

from athleticspose.camera.camera import Camera
from athleticspose.preprocess.utils import mocap_to_h36m, normalize_kpts, split_clips


def process_files(files: list[str], output_dir: str, clip_length: int, split: str):
    """Process the files and save them to the output directory.

    Args:
        files: List of file paths to process.
        output_dir: Directory to save the processed files.
        clip_length: Length of the clip to process.
        split: Split to process.

    """
    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    if split == "train":
        stride = clip_length // 3
    else:
        stride = clip_length

    data_cnt = 0
    camera_dir = "./data/AthleticsPoseDataset/camera_params"
    for file in files:
        date = os.path.basename(file).split("_")[0]
        action = file.split("/")[-3]
        camera_file = os.path.join(camera_dir, f"{date}_{action}.json")
        camera = Camera(camera_file)
        num_cameras = len(camera.camera_params)

        kpts_world = np.load(file)
        kpts_world = mocap_to_h36m(kpts_world)
        for cam_idx in range(num_cameras):
            kpts_cam = camera.world_to_camera(kpts_world, cam_idx)
            kpts_pixel, scales = camera.camera_to_pixel3d(kpts_cam, cam_idx)

            clips = split_clips(kpts_pixel.shape[0], clip_length, stride)
            kpts_clips = kpts_pixel[clips]
            scales_clips = scales[clips]
            for i in range(len(clips)):
                kpts_clip = kpts_clips[i]
                scale = scales_clips[i]
                kpts_clip_norm, norm_scale = normalize_kpts(kpts_clip)
                label3d = kpts_clip_norm
                input2d = np.ones_like(kpts_clip_norm)
                input2d[:, :, :2] = kpts_clip_norm[:, :, :2]

                data = {
                    "input2d": input2d,
                    "label3d": label3d,
                    "pixel_to_mm_scale": scale,
                    "norm_scale": norm_scale,
                }
                output_file = os.path.join(output_dir, f"{data_cnt:05d}.pkl")
                with open(output_file, "wb") as f:
                    pickle.dump(data, f)
                data_cnt += 1

    print(f"Processed {data_cnt} files for {split} split.")


def process_files_for_multiview(files: list[str], output_dir: str, split: str):
    """Process the files and save them to the output directory.

    Args:
        files: List of file paths to process.
        output_dir: Directory to save the processed files.
        clip_length: Length of the clip to process.
        split: Split to process.

    """
    output_dir = os.path.join(output_dir, split)
    os.makedirs(output_dir, exist_ok=True)

    data_cnt = 0
    camera_dir = "./data/AthleticsPoseDataset/camera_params"
    for file in files:
        video_file_base_name = file.replace("markers", "videos").split(".npy")[0]
        date = os.path.basename(file).split("_")[0]
        action = file.split("/")[-3]
        camera_file = os.path.join(camera_dir, f"{date}_{action}.json")
        camera = Camera(camera_file)
        num_cameras = len(camera.camera_params)

        kpts_world = np.load(file)
        kpts_world = mocap_to_h36m(kpts_world)

        data_all_cam = []
        for cam_idx in range(num_cameras):
            video_file = f"{video_file_base_name}_{cam_idx + 1}.mp4"
            intrinsics = camera.camera_params[cam_idx]["affine_intrinsics_matrix"]
            kpts_cam = camera.world_to_camera(kpts_world, cam_idx)
            kpts_pixel, _ = camera.camera_to_pixel3d(kpts_cam, cam_idx)
            kpts_image = kpts_pixel[:, :, :2]

            data_all_cam.append(
                {
                    "kpts_image": kpts_image,
                    "kpts_world": kpts_world,
                    "intrinsics": intrinsics,
                    "video_file": video_file,
                }
            )

        output_file = os.path.join(output_dir, f"{data_cnt:04d}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump(data_all_cam, f)
        data_cnt += 1

    print(f"Processed {data_cnt} files for {split} split.")
