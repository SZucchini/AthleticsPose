"""Script for inferring 3D poses from video."""

import argparse
import os
from pathlib import Path

import numpy as np

from athleticspose.inferencer import Pose3DInferencer
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
    parser = argparse.ArgumentParser(description="Infer 3D poses from video")
    parser.add_argument("video_path", type=str, help="Path to input video file")
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
    if not os.path.exists(args.video_path):
        raise FileNotFoundError(f"Video file not found: {args.video_path}")
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

    print("Extracting 2D poses from video...")
    keypoints_2d, scores = video_estimator.process_video(args.video_path)
    print(f"Extracted poses from {len(keypoints_2d)} frames")

    # Convert COCO format to H36M format
    print("Converting to H36M format...")
    keypoints_h36m, scores_h36m = PoseFormatConverter.coco_to_h36m(keypoints_2d, scores)

    # Stack keypoints and scores
    poses_2d = np.concatenate([keypoints_h36m, scores_h36m[..., None]], axis=-1)

    # Initialize 3D pose inferencer
    print("Initializing 3D pose inferencer...")
    inferencer = Pose3DInferencer(checkpoint_path=args.checkpoint_path)

    # Predict 3D poses
    print("Predicting 3D poses...")
    poses_3d = inferencer.predict(poses_2d, batch_size=args.batch_size)

    # Save predictions if output path provided
    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.parent.exists():
            output_path.parent.mkdir(parents=True)
        if not str(output_path).endswith(".npy"):
            output_path = output_path.with_suffix(".npy")
        np.save(output_path, poses_3d)
        print(f"Saved predictions to {output_path}")

    print(
        f"Inference complete! Predicted 3D poses shape: {poses_3d.shape}"
        f"\n- Number of frames: {poses_3d.shape[0]}"
        f"\n- Number of joints: {poses_3d.shape[1]}"
    )


if __name__ == "__main__":
    main()
