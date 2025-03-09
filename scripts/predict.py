"""Prediction script for AthleticsPose."""

import argparse

import numpy as np

from athleticspose.inference import Pose3DInference


def main():
    """Predict 3D poses from 2D poses."""
    parser = argparse.ArgumentParser(description="Predict 3D poses from 2D poses")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input .npy file containing 2D poses (T, J, 3)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output .npy file for saving 3D poses",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for inference (optional)",
    )
    args = parser.parse_args()

    # Load input data
    poses_2d = np.load(args.input)

    # Initialize inferencer
    inferencer = Pose3DInference(args.checkpoint)

    # Run inference
    poses_3d = inferencer.predict(poses_2d, args.batch_size)

    # Save results
    np.save(args.output, poses_3d)


if __name__ == "__main__":
    main()
