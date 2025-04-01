"""Evaluate the model by action."""

import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import torch
from natsort import natsorted

from athleticspose.datasets.ap_dataset import read_pkl
from athleticspose.loss import calc_mpjpe, p_mpjpe
from athleticspose.plmodules.linghtning_module import LightningPose3D


def load_checkpoint(model: LightningPose3D, ckpt_path: str) -> None:
    """Load checkpoint.

    Args:
        model (LightningPose3D): Model to load checkpoint.
        ckpt_path (str): Path to checkpoint.

    """
    state_dict = torch.load(ckpt_path, map_location="cpu")["state_dict"]
    model.load_state_dict(state_dict)


def calculate_metrics(pred, y, p2mm, norm_scale) -> Tuple[float, float]:
    """Calculate validation metrics.

    Args:
        pred (torch.Tensor): Predicted 3D pose.
        y (torch.Tensor): Ground truth 3D pose.
        p2mm (torch.Tensor): Scale factors pixel coordinates to camera coordinates.
        norm_scale (torch.Tensor): Normalization scale.

    Returns:
        Tuple[float, float]: MPJPE and PA-MPJPE.

    """
    pred[:, :, 0, :] = 0
    pred = pred.cpu().numpy()
    y = y.cpu().numpy()
    p2mm = p2mm.cpu().numpy()
    norm_scale = norm_scale

    pred_denom = np.zeros_like(pred)
    y_denom = np.zeros_like(y)
    for i in range(pred.shape[0]):
        pred_denom[i, :, :, :] = pred[i] * norm_scale[i]
        y_denom[i, :, :, :] = y[i] * norm_scale[i]
    pred_denom = pred_denom / p2mm[:, :, None, None]
    y_denom = y_denom / p2mm[:, :, None, None]

    pa_mpjpe_list = []
    mpjpe = np.mean(calc_mpjpe(pred_denom, y_denom))
    for i in range(pred_denom.shape[0]):
        pa_mpjpe = p_mpjpe(pred_denom[i], y_denom[i])
        pa_mpjpe_list.append(pa_mpjpe)
    pa_mpjpe = np.mean(pa_mpjpe_list)
    return mpjpe, pa_mpjpe


def evaluate_by_action(
    model: LightningPose3D,
    test_dir: str,
    device: torch.device,
) -> Dict[str, Dict[str, List[float]]]:
    """Evaluate the model by action.

    Args:
        model (LightningPose3D): Model to evaluate.
        test_dir (str): Directory containing test data.
        device (torch.device): Device to run evaluation on.

    Returns:
        Dict[str, Dict[str, List[float]]]: Results by action.

    """
    model.eval()
    model.to(device)

    # Get all test files
    test_files = glob.glob(os.path.join(test_dir, "*.pkl"))
    test_files = natsorted(test_files)

    # Dictionary to store results by action
    action_results: Dict[str, Dict[str, List[float]]] = {}

    with torch.no_grad():
        for file in test_files:
            data = read_pkl(file)
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

            # Forward pass
            pred = model(input2d)

            # Calculate metrics
            mpjpe, pa_mpjpe = calculate_metrics(pred, label3d, p2mm, norm_scale)

            # Store results by action
            if action not in action_results:
                action_results[action] = {"mpjpe": [], "pa_mpjpe": []}
            action_results[action]["mpjpe"].append(mpjpe)
            action_results[action]["pa_mpjpe"].append(pa_mpjpe)

    return action_results


def main():
    """Evaluate the model by action."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = LightningPose3D.load_from_checkpoint("work_dir/20250302_110906/best.ckpt")

    # Set test directory
    test_dir = "data/processed/AP_81/test"

    # Evaluate
    results = evaluate_by_action(model, test_dir, device)

    # Print results
    print("\nResults by action:")
    print("-" * 50)
    for action, metrics in results.items():
        avg_mpjpe = np.mean(metrics["mpjpe"])
        avg_pa_mpjpe = np.mean(metrics["pa_mpjpe"])
        print(f"\nAction: {action}")
        print(f"  Average MPJPE: {avg_mpjpe:.2f} mm")
        print(f"  Average PA-MPJPE: {avg_pa_mpjpe:.2f} mm")
        print(f"  Number of samples: {len(metrics['mpjpe'])}")

    # Print overall results
    all_mpjpe = []
    all_pa_mpjpe = []
    for metrics in results.values():
        all_mpjpe.extend(metrics["mpjpe"])
        all_pa_mpjpe.extend(metrics["pa_mpjpe"])

    print("\nOverall Results:")
    print("-" * 50)
    print(f"Average MPJPE: {np.mean(all_mpjpe):.2f} mm")
    print(f"Average PA-MPJPE: {np.mean(all_pa_mpjpe):.2f} mm")
    print(f"Total number of samples: {len(all_mpjpe)}")


if __name__ == "__main__":
    main()
