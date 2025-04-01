"""Script for inferring 3D poses from multi-view video."""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from athleticspose.inferencer import Pose3DInferencer
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
    parser = argparse.ArgumentParser(description="Infer 3D poses from multi-view videos")
    parser.add_argument("input_data", type=str, help="Path to input multi-view data")
    parser.add_argument("camera_num", type=int, default=3, help="Using camera number")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to 3D pose model checkpoint",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save 3D pose predictions (will save as .npy file)",
    )
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


def main():
    """Inference keypoints from video."""
    args = parse_args()

    # Validate paths
    if not os.path.exists(args.input_data):
        raise FileNotFoundError(f"Input data not found: {args.input_data}")
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

    with open(args.input_data, "rb") as f:
        input_data = pickle.load(f)
    if not isinstance(input_data, list):
        raise ValueError("Input data should be a list of dictionaries.")

    intrinsics_list = []
    kpts2d_list = []
    gt_kpts2d_list = []
    scores2d_list = []
    gt_scores2d_list = []
    kpts3d_list = []
    kpts3d_from_gt_list = []
    scores3d_list = []
    for i in range(len(input_data)):
        if i >= args.camera_num:
            break
        video_file = input_data[i]["video_file"]
        intrinsics = input_data[i]["intrinsics"]
        gt_kpts2d = input_data[i]["kpts_image"]

        keypoints_2d, scores = video_estimator.process_video(video_file)
        keypoints_h36m, scores_h36m = PoseFormatConverter.coco_to_h36m(keypoints_2d, scores)

        inputs2d = np.concatenate([keypoints_h36m, scores_h36m[..., None]], axis=-1)
        kpts3d = inferencer.predict(inputs2d, batch_size=args.batch_size)

        scores = np.ones((kpts3d.shape[0], kpts3d.shape[1]))
        inputs2d = np.concatenate([gt_kpts2d, scores[..., None]], axis=-1)
        kpts3d_from_gt = inferencer.predict(inputs2d, batch_size=args.batch_size)

        intrinsics_list.append(intrinsics)
        kpts2d_list.append(keypoints_h36m)
        gt_kpts2d_list.append(gt_kpts2d)
        gt_scores2d_list.append(np.ones((gt_kpts2d.shape[0], 17)))
        scores2d_list.append(scores_h36m)
        kpts3d_list.append(kpts3d)
        kpts3d_from_gt_list.append(kpts3d_from_gt)
        scores3d_list.append(np.ones((kpts3d.shape[0], kpts3d.shape[1])))

    kpts2d_list = np.array(kpts2d_list)
    scores2d_list = np.array(scores2d_list)
    gt_kpts2d_list = np.array(gt_kpts2d_list)
    gt_scores2d_list = np.array(gt_scores2d_list)
    kpts3d_list = np.array(kpts3d_list)
    scores3d_list = np.array(scores3d_list)
    intrinsics_list = np.array(intrinsics_list)

    _, _, triangulated_kpts3d = calibrate(
        kpts2d=kpts2d_list,
        score2d=scores2d_list,
        kpts3d=kpts3d_list,
        score3d=scores3d_list,
        intrinsics=intrinsics_list,
    )

    # Save predictions if output path provided
    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        if not str(output_path).endswith(".npy"):
            output_path = output_path.with_suffix(".npy")
        np.save(output_path, triangulated_kpts3d)
        print(f"Saved predictions to {output_path}")

    _, _, triangulated_kpts3d_from_gt_kpts2d = calibrate(
        kpts2d=gt_kpts2d_list,
        score2d=gt_scores2d_list,
        kpts3d=kpts3d_from_gt_list,
        score3d=scores3d_list,
        intrinsics=intrinsics_list,
    )

    if args.output_path:
        # fixed output path with gt
        output_path = Path(args.output_path).with_name("gt_" + Path(args.output_path).name)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        if not str(output_path).endswith(".npy"):
            output_path = output_path.with_suffix(".npy")
        np.save(output_path, triangulated_kpts3d_from_gt_kpts2d)
        print(f"Saved predictions to {output_path}")

    print(
        f"Inference complete! Predicted 3D poses shape: {triangulated_kpts3d.shape}"
        f"\n- Number of frames: {triangulated_kpts3d.shape[0]}"
        f"\n- Number of joints: {triangulated_kpts3d.shape[1]}"
    )


if __name__ == "__main__":
    main()
