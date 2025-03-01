"""Preprocess the AthleticPose dataset for the 3D pose estimation."""

import argparse
import glob
import os

from natsort import natsorted

from athleticspose.preprocess.processing import process_files


def main():
    """Convert the AthleticPose dataset to the format required by the 3D pose estimation model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="data/processed")
    parser.add_argument("--clip_length", type=int, default=81)
    args = parser.parse_args()

    output_dir = os.path.join(args.output_root, f"AP_{args.clip_length}")
    os.makedirs(output_dir, exist_ok=True)

    test_subjects = [
        "S00",
        "S04",
        "S05",
        "S13",
        "S20",
    ]

    train_files = []
    test_files = []
    action_dirs = glob.glob(os.path.join(args.data_root, "markers", "*"))
    action_dirs = natsorted(action_dirs)
    for action_dir in action_dirs:
        subject_dirs = glob.glob(os.path.join(action_dir, "*"))
        subject_dirs = natsorted(subject_dirs)
        for subject_dir in subject_dirs:
            subject = os.path.basename(subject_dir)
            marker_files = glob.glob(os.path.join(subject_dir, "*.npy"))
            marker_files = natsorted(marker_files)
            if subject in test_subjects:
                test_files.extend(marker_files)
            else:
                train_files.extend(marker_files)

    process_files(train_files, output_dir, args.clip_length, "train")
    process_files(test_files, output_dir, args.clip_length, "test")


if __name__ == "__main__":
    main()
