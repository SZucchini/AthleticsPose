"""Script for evaluating 3D poses from multi-view video."""

import argparse
import glob
import os
import pickle
from pathlib import Path

import numpy as np
from natsort import natsorted

from athleticspose.inferencer import Pose3DInferencer
from athleticspose.loss import p_mpjpe
from athleticspose.multiview.calib import calibrate
from athleticspose.pipeline.video_estimator import VideoEstimator
from athleticspose.preprocess.pose_format import PoseFormatConverter

DEFAULT_CHECKPOINT = "work_dir/20250302_110906/best.ckpt"
DEFAULT_MMDET_CONFIG = "athleticspose/mmpose/mmdet_cfg/rtmdet.py"
DEFAULT_MMPOSE_CONFIG = "athleticspose/mmpose/config/td-hm_ViTPose-base_8xb64-210e_coco-256x192.py"
DEFAULT_DET_WEIGHTS = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"
)
DEFAULT_POSE_WEIGHTS = "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth"


def parse_args():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments

    """
    parser = argparse.ArgumentParser(description="Evaluate 3D poses from multi-view video")
    parser.add_argument("data_root", type=str, help="Path to input multi-view data")
    parser.add_argument("camera_num", type=int, default=3, help="Using camera number")
    parser.add_argument("--use_gt", action="store_true", help="Use ground truth 2D keypoints")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to 3D pose model checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for 3D pose inference. If None, process all clips at once.",
    )
    # MMDetection and MMPose configs
    parser.add_argument(
        "--mmdet_config",
        type=str,
        default=DEFAULT_MMDET_CONFIG,
        help="Path to MMDetection config file",
    )
    parser.add_argument(
        "--mmpose_config",
        type=str,
        default=DEFAULT_MMPOSE_CONFIG,
        help="Path to MMPose config file",
    )
    parser.add_argument(
        "--det_weights",
        type=str,
        default=DEFAULT_DET_WEIGHTS,
        help="Path or URL to detection model weights",
    )
    parser.add_argument(
        "--pose_weights",
        type=str,
        default=DEFAULT_POSE_WEIGHTS,
        help="Path or URL to pose estimation model weights",
    )
    return parser.parse_args()


def build_data_for_calibration(video_estimator, inferencer, input_data, camera_num, batch_size=None, use_gt=False):
    intrinsics_list = []
    kpts2d_list = []
    scores2d_list = []
    kpts3d_list = []
    scores3d_list = []
    for cam in range(len(input_data)):
        if cam >= camera_num:
            break

        intrinsics = input_data[cam]["intrinsics"]

        if use_gt:
            kpts2d = input_data[cam]["kpts_image"]
            scores2d = np.ones((kpts2d.shape[0], kpts2d.shape[1]))
        else:
            video_file = input_data[cam]["video_file"]
            keypoints_2d, scores = video_estimator.process_video(video_file)
            kpts2d, scores2d = PoseFormatConverter.coco_to_h36m(keypoints_2d, scores)

        inputs2d = np.concatenate([kpts2d, scores2d[..., None]], axis=-1)
        kpts3d = inferencer.predict(inputs2d, batch_size=batch_size)
        scores3d = np.ones((kpts3d.shape[0], kpts3d.shape[1]))

        intrinsics_list.append(intrinsics)
        kpts2d_list.append(kpts2d)
        scores2d_list.append(scores2d)
        kpts3d_list.append(kpts3d)
        scores3d_list.append(scores3d)

    for i in range(len(kpts2d_list)):
        if len(kpts2d_list[i]) != len(kpts2d_list[0]):
            return [], [], [], [], []

    intrinsics_list = np.array(intrinsics_list)
    kpts2d_list = np.array(kpts2d_list)
    scores2d_list = np.array(scores2d_list)
    kpts3d_list = np.array(kpts3d_list)
    scores3d_list = np.array(scores3d_list)

    return intrinsics_list, kpts2d_list, scores2d_list, kpts3d_list, scores3d_list


def main():
    """Inference keypoints from video."""
    args = parse_args()

    # Validate paths
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint_path}")

    # Initialize 2D pose estimator
    video_estimator = VideoEstimator(
        mmdet_config=args.mmdet_config,
        mmpose_config=args.mmpose_config,
        det_weights=args.det_weights,
        pose_weights=args.pose_weights,
        device=args.device,
    )
    # Initialize 3D pose inferencer
    inferencer = Pose3DInferencer(checkpoint_path=args.checkpoint_path)

    file_cnt, avg_p_mpjpe = 0, 0
    pickle_files = natsorted((glob.glob(os.path.join(args.data_root, "*.pkl"))))
    for pickle_file in pickle_files:
        with open(pickle_file, "rb") as f:
            input_data = pickle.load(f)
        if not isinstance(input_data, list):
            raise ValueError("Input data should be a list of dictionaries.")

        gt_kpts = input_data[0]["kpts_world"]

        intrinsics_list, kpts2d_list, scores2d_list, kpts3d_list, scores3d_list = build_data_for_calibration(
            video_estimator,
            inferencer,
            input_data,
            camera_num=args.camera_num,
            batch_size=args.batch_size,
            use_gt=args.use_gt,
        )

        if len(kpts2d_list) == 0:
            print(f"Skipping {pickle_file} due to inconsistent keypoint lengths.")
            continue

        _, _, triangulated_kpts3d = calibrate(
            kpts2d=kpts2d_list,
            score2d=scores2d_list,
            kpts3d=kpts3d_list,
            score3d=scores3d_list,
            intrinsics=intrinsics_list,
            ransac=False,
        )

        p_mpjpe_score = np.mean(p_mpjpe(triangulated_kpts3d, gt_kpts))
        file_cnt += 1
        avg_p_mpjpe += p_mpjpe_score
        print(f"Processed {pickle_file} with PA-MPJPE: {p_mpjpe_score:.2f} mm")

    if file_cnt > 0:
        avg_p_mpjpe /= file_cnt
        print(f"Average PA-MPJPE: {avg_p_mpjpe:.2f} mm")


if __name__ == "__main__":
    main()
