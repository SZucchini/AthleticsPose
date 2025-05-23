"""Script for evaluating a single pose estimation result."""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch

from athleticspose.datasets.ap_dataset import read_pkl
from athleticspose.loss import calc_mpjpe, p_mpjpe
from athleticspose.plmodules.linghtning_module import LightningPose3D
from athleticspose.visualize import align_poses, visualize_pose_comparison


def calculate_metrics(
    pred: torch.Tensor,
    y: torch.Tensor,
    p2mm: torch.Tensor,
    norm_scale: np.ndarray,
) -> Tuple[float, float]:
    """Calculate MPJPE and PA-MPJPE metrics.

    Args:
        pred (torch.Tensor): Predicted 3D pose.
        y (torch.Tensor): Ground truth 3D pose.
        p2mm (torch.Tensor): Scale factors pixel coordinates to camera coordinates.
        norm_scale (np.ndarray): Normalization scale.

    Returns:
        Tuple[float, float]: MPJPE and PA-MPJPE.

    """
    # Zero out root joint
    pred[:, :, 0, :] = 0
    pred = pred.cpu().numpy()
    y = y.cpu().numpy()
    p2mm = p2mm.cpu().numpy()

    # De-normalize predictions and ground truth
    pred_denom = np.zeros_like(pred)
    y_denom = np.zeros_like(y)
    for i in range(pred.shape[0]):
        pred_denom[i, :, :, :] = pred[i] * norm_scale[i]
        y_denom[i, :, :, :] = y[i] * norm_scale[i]
    pred_denom = pred_denom / p2mm[:, :, None, None]
    y_denom = y_denom / p2mm[:, :, None, None]

    # Calculate metrics
    pa_mpjpe_list = []
    mpjpe = np.mean(calc_mpjpe(pred_denom, y_denom))
    for i in range(pred_denom.shape[0]):
        pa_mpjpe = p_mpjpe(pred_denom[i], y_denom[i])
        pa_mpjpe_list.append(pa_mpjpe)
    pa_mpjpe = np.mean(pa_mpjpe_list)

    return mpjpe, pa_mpjpe


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Evaluate single pose estimation result.")
    parser.add_argument("input_file", type=str, help="Path to input pickle file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="work_dir/20250302_110906/best.ckpt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/output",
        help="Directory to save visualization results",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="If set, visualize the prediction and ground truth",
    )
    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LightningPose3D.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(device)

    # Load data
    data = read_pkl(args.input_file)
    action = data["action"]

    # Convert data to tensors
    input2d = torch.FloatTensor(data["input2d"]).to(device)
    label3d = torch.FloatTensor(data["label3d"]).to(device)
    p2mm = torch.FloatTensor(data["pixel_to_mm_scale"]).to(device)
    norm_scale = data["norm_scale"]

    # Add batch dimension if needed
    if len(input2d.shape) == 3:
        input2d = input2d.unsqueeze(0)
        label3d = label3d.unsqueeze(0)
        p2mm = p2mm.unsqueeze(0)
        norm_scale = np.array([norm_scale])

    # Inference
    with torch.no_grad():
        pred = model(input2d)

    # Calculate metrics
    mpjpe, pa_mpjpe = calculate_metrics(pred, label3d, p2mm, norm_scale)

    # Print results
    print(f"File: {Path(args.input_file).name}")
    print(f"Action: {action}")
    print(f"MPJPE: {mpjpe:.2f} mm")
    print(f"PA-MPJPE: {pa_mpjpe:.2f} mm")

    # Visualize if requested
    if args.visualize:
        # Convert prediction and label to numpy and denormalize
        pred = pred.cpu().numpy()
        label3d = label3d.cpu().numpy()

        pred_denom = np.zeros_like(pred)
        label3d_denom = np.zeros_like(label3d)
        for i in range(pred.shape[0]):
            pred_denom[i, :, :, :] = pred[i] * norm_scale[i]
            label3d_denom[i, :, :, :] = label3d[i] * norm_scale[i]
        pred_denom = pred_denom / p2mm.cpu().numpy()[:, :, None, None]
        label3d_denom = label3d_denom / p2mm.cpu().numpy()[:, :, None, None]

        # Prepare output path
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        output_base = output_dir / Path(args.input_file).stem

        # Visualize raw poses
        visualize_pose_comparison(
            pred_denom[0],  # Remove batch dimension
            label3d_denom[0],  # Remove batch dimension
            str(output_base),
            title=f"Action: {action}",
            is_aligned=False,
        )

        # Align prediction to ground truth and visualize
        pred_aligned = align_poses(pred_denom[0], label3d_denom[0])
        visualize_pose_comparison(
            pred_aligned,  # Already batch dimension removed
            label3d_denom[0],  # Remove batch dimension
            str(output_base),
            title=f"Action: {action}",
            is_aligned=True,
        )


if __name__ == "__main__":
    main()
