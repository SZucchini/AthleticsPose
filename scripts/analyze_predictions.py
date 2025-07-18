"""Analyze 3D pose predictions in detail."""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from athleticspose.utils import denormalize_kpts, normalize_kpts

"""Analysis utilities for 3D pose prediction evaluation."""


def calc_mpjpe_per_frame(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate MPJPE per frame.

    Args:
        predicted (np.ndarray): Predicted keypoints. shape: (T, J, 3)
        target (np.ndarray): Target keypoints. shape: (T, J, 3)

    Returns:
        np.ndarray: MPJPE per frame. shape: (T,)

    """
    assert predicted.shape == target.shape
    return np.mean(np.linalg.norm(predicted - target, axis=2), axis=1)


def calc_p_mpjpe_per_frame(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Calculate P-MPJPE per frame (Procrustes-aligned MPJPE).

    Args:
        predicted (np.ndarray): Predicted keypoints. shape: (T, J, 3)
        target (np.ndarray): Target keypoints. shape: (T, J, 3)

    Returns:
        np.ndarray: P-MPJPE per frame. shape: (T,)

    """
    assert predicted.shape == target.shape
    mu_x = np.mean(target, axis=1, keepdims=True)
    mu_y = np.mean(predicted, axis=1, keepdims=True)

    x0 = target - mu_x
    y0 = predicted - mu_y

    norm_x = np.sqrt(np.sum(x0**2, axis=(1, 2), keepdims=True))
    norm_y = np.sqrt(np.sum(y0**2, axis=(1, 2), keepdims=True))

    x0 /= norm_x
    y0 /= norm_y

    h = np.matmul(x0.transpose(0, 2, 1), y0)
    u, s, v_t = np.linalg.svd(h)
    v = v_t.transpose(0, 2, 1)
    r = np.matmul(v, u.transpose(0, 2, 1))

    # Avoid improper rotations (reflections), i.e. rotations with det(R) = -1
    sign_det_r = np.sign(np.expand_dims(np.linalg.det(r), axis=1))
    v[:, :, -1] *= sign_det_r
    s[:, -1] *= sign_det_r.flatten()
    r = np.matmul(v, u.transpose(0, 2, 1))  # Rotation
    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * norm_x / norm_y  # Scale
    t = mu_x - a * np.matmul(mu_y, r)  # Translation

    # Perform rigid transformation on the input
    predicted_aligned = a * np.matmul(predicted, r) + t
    return np.mean(np.linalg.norm(predicted_aligned - target, axis=2), axis=1)


def classify_bbox_scale(bbox: np.ndarray, clip_length: int = 81) -> np.ndarray:
    """Classify bbox scale based on height, processing in clips.

    Args:
        bbox (np.ndarray): BBOX information. shape: (T, 4) in format (x_min, y_min, w, h)
        clip_length (int): Length of each clip for processing. Default: 81

    Returns:
        np.ndarray: Scale classification per frame. shape: (T,)
                   0: small (scale < 0.5)
                   1: medium (0.5 <= scale < 0.75)
                   2: large (0.75 <= scale <= 1.0)

    """
    heights = bbox[:, 3]  # Extract height values
    total_frames = len(heights)
    scale_classes = np.zeros(total_frames, dtype=int)

    # Process in clips of clip_length frames
    for start_idx in range(0, total_frames, clip_length):
        end_idx = min(start_idx + clip_length, total_frames)
        clip_heights = heights[start_idx:end_idx]

        # Calculate scales for this clip
        clip_max_height = np.max(clip_heights)
        if clip_max_height > 0:  # Avoid division by zero
            clip_scales = clip_heights / clip_max_height
        else:
            print(
                f"Warning: Division by zero in bbox scale classification. "
                f"Max height is 0 for frames {start_idx}-{end_idx - 1}"
            )
            clip_scales = np.ones_like(clip_heights)

        # Classify scales for this clip
        clip_scale_classes = np.zeros(len(clip_scales), dtype=int)
        clip_scale_classes[clip_scales < 0.33] = 0  # small
        clip_scale_classes[(clip_scales >= 0.33) & (clip_scales < 0.67)] = 1  # medium
        clip_scale_classes[clip_scales >= 0.67] = 2  # large

        # Store results
        scale_classes[start_idx:end_idx] = clip_scale_classes

    return scale_classes


def classify_camera_view(date: str, cam_idx: int) -> str:
    """Classify camera view based on date and camera index.

    Args:
        date (str): Date in format 'yyyymmdd'
        cam_idx (int): Camera index (0-7)

    Returns:
        str: Camera view classification ('side' or 'front_or_back')

    """
    camera_mapping = {
        "20250125": {"side": [1, 2, 6, 7], "front_or_back": [0, 3, 4, 5]},
        "20250126": {"side": [0, 3, 4, 7], "front_or_back": [1, 2, 5, 6]},
        "20250215": {"side": [2, 3, 4, 5], "front_or_back": [0, 1, 6, 7]},
        "20250216": {"side": [0, 2, 6, 7], "front_or_back": [1, 3, 4, 5]},
    }

    if date not in camera_mapping:
        raise ValueError(f"Unknown date: {date}")

    if cam_idx in camera_mapping[date]["side"]:
        return "side"
    elif cam_idx in camera_mapping[date]["front_or_back"]:
        return "front_or_back"
    else:
        raise ValueError(f"Unknown camera index {cam_idx} for date {date}")


def get_scale_name(scale_class: int) -> str:
    """Get scale name from scale class.

    Args:
        scale_class (int): Scale class (0: small, 1: medium, 2: large)

    Returns:
        str: Scale name

    """
    scale_names = {0: "small", 1: "medium", 2: "large"}
    return scale_names[scale_class]


def aggregate_metrics(mpjpe_values: List[float], p_mpjpe_values: List[float]) -> Dict[str, float]:
    """Aggregate MPJPE and P-MPJPE values.

    Args:
        mpjpe_values (List[float]): List of MPJPE values
        p_mpjpe_values (List[float]): List of P-MPJPE values

    Returns:
        Dict[str, float]: Aggregated metrics

    """
    if len(mpjpe_values) == 0:
        return {"mpjpe": 0.0, "p_mpjpe": 0.0, "count": 0}

    return {
        "mpjpe": np.mean(mpjpe_values),
        "p_mpjpe": np.mean(p_mpjpe_values),
        "count": len(mpjpe_values),
    }


def parse_filename(filename: str) -> Tuple[str, str, str, int]:
    """Parse filename to extract date, data_num, and cam_idx.

    Args:
        filename (str): Filename in format 'yyyymmdd_dd_c.npy'

    Returns:
        Tuple[str, str, str, int]: (date, data_num, cam_idx_str, cam_idx)

    """
    basename = filename.replace(".npy", "").replace(".npz", "")
    parts = basename.split("_")

    if len(parts) != 3:
        raise ValueError(f"Invalid filename format: {filename}")

    date = parts[0]
    data_num = parts[1]
    cam_idx_str = parts[2]
    cam_idx = int(cam_idx_str)

    return date, data_num, cam_idx_str, cam_idx


def find_prediction_directories(base_dir: Path) -> List[Path]:
    """Find all prediction directories.

    Args:
        base_dir (Path): Base directory containing prediction results

    Returns:
        List[Path]: List of prediction directories

    """
    prediction_dirs = []
    for pred_dir in ["pred_from_gt", "pred_from_det_coco", "pred_from_det_ft"]:
        pred_path = base_dir / pred_dir
        if pred_path.exists():
            prediction_dirs.append(pred_path)
    return prediction_dirs


def find_prediction_files(pred_dir: Path) -> List[Tuple[Path, str, str]]:
    """Find all prediction files in a directory.

    Args:
        pred_dir (Path): Prediction directory

    Returns:
        List[Tuple[Path, str, str]]: List of (file_path, action, subject)

    """
    files = []
    target_actions = ["hurdle", "sd", "sprint", "racewalk", "running"]

    for action in target_actions:
        action_dir = pred_dir / action
        if not action_dir.exists():
            continue

        for subject_dir in action_dir.iterdir():
            if not subject_dir.is_dir():
                continue

            subject = subject_dir.name
            for file_path in subject_dir.glob("*.npy"):
                files.append((file_path, action, subject))

    return files


def load_gt_markers(gt_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load GT markers and p2mm from npz file.

    Args:
        gt_file (Path): Path to GT markers file

    Returns:
        Tuple[np.ndarray, np.ndarray]: (markers_h36m, p2mm)

    """
    data = np.load(gt_file)
    markers_h36m = data["markers_h36m"]
    p2mm = data["p2mm"]
    return markers_h36m, p2mm


def load_bbox_info(bbox_file: Path) -> np.ndarray:
    """Load 2D bbox information from npy file.

    Args:
        bbox_file (Path): Path to bbox file

    Returns:
        np.ndarray: Bbox information. shape: (T, 4)

    """
    return np.load(bbox_file)


def process_single_file(
    pred_file: Path,
    action: str,
    subject: str,
    gt_base_dir: Path,
    bbox_base_dir: Path,
    verbose: bool = False,
) -> Dict:
    """Process a single prediction file.

    Args:
        pred_file (Path): Path to prediction file
        action (str): Action name
        subject (str): Subject name
        gt_base_dir (Path): Base directory for GT files
        bbox_base_dir (Path): Base directory for bbox files
        verbose (bool): Whether to print verbose output

    Returns:
        Dict: Processing results

    """
    try:
        # Parse filename
        filename = pred_file.name
        date, data_num, cam_idx_str, cam_idx = parse_filename(filename)

        # Load prediction
        pred_data = np.load(pred_file)
        if pred_data.shape[1] != 17 or pred_data.shape[2] != 3:
            if verbose:
                print(f"Warning: Unexpected prediction shape {pred_data.shape} for {pred_file}")
            return None

        # Construct GT and bbox file paths
        gt_file = gt_base_dir / action / subject / filename.replace(".npy", ".npz")
        bbox_file = bbox_base_dir / action / subject / filename

        # Check if GT and bbox files exist
        if not gt_file.exists():
            if verbose:
                print(f"Warning: GT file not found: {gt_file}")
            return None

        if not bbox_file.exists():
            if verbose:
                print(f"Warning: Bbox file not found: {bbox_file}")
            return None

        # Load GT markers and p2mm
        gt_markers, p2mm = load_gt_markers(gt_file)
        if gt_markers.shape != pred_data.shape:
            if verbose:
                print(
                    f"Warning: Shape mismatch between prediction {pred_data.shape} "
                    f"and GT {gt_markers.shape} for {pred_file}"
                )
            return None

        # Load bbox info
        bbox_info = load_bbox_info(bbox_file)
        if bbox_info.shape[0] != pred_data.shape[0]:
            if verbose:
                print(
                    f"Warning: Frame count mismatch between prediction {pred_data.shape[0]} "
                    f"and bbox {bbox_info.shape[0]} for {pred_file}"
                )
            return None

        # Convert to root-relative coordinates
        gt_markers, norm_scale = normalize_kpts(gt_markers)
        gt_markers = denormalize_kpts(gt_markers, norm_scale)

        # Convert from pixel to mm coordinates
        pred_data = pred_data / p2mm[:, None, None]
        gt_markers = gt_markers / p2mm[:, None, None]

        # Calculate per-frame metrics
        mpjpe_per_frame = calc_mpjpe_per_frame(pred_data, gt_markers)
        p_mpjpe_per_frame = calc_p_mpjpe_per_frame(pred_data, gt_markers)

        # Classify camera view and scale
        camera_view = classify_camera_view(date, cam_idx)
        scale_classes = classify_bbox_scale(bbox_info)

        return {
            "action": action,
            "subject": subject,
            "date": date,
            "data_num": data_num,
            "cam_idx": cam_idx,
            "camera_view": camera_view,
            "mpjpe_per_frame": mpjpe_per_frame,
            "p_mpjpe_per_frame": p_mpjpe_per_frame,
            "scale_classes": scale_classes,
            "filename": filename,
        }

    except Exception as e:
        if verbose:
            print(f"Error processing {pred_file}: {str(e)}")
        return None


def analyze_prediction_directory(
    pred_dir: Path, gt_base_dir: Path, bbox_base_dir: Path, verbose: bool = False
) -> Dict:
    """Analyze all files in a prediction directory.

    Args:
        pred_dir (Path): Prediction directory
        gt_base_dir (Path): Base directory for GT files
        bbox_base_dir (Path): Base directory for bbox files
        verbose (bool): Whether to print verbose output

    Returns:
        Dict: Analysis results

    """
    print(f"Analyzing {pred_dir.name}...")

    # Find all prediction files
    pred_files = find_prediction_files(pred_dir)
    print(f"Found {len(pred_files)} prediction files")

    # Process each file
    results = []
    for pred_file, action, subject in pred_files:
        result = process_single_file(pred_file, action, subject, gt_base_dir, bbox_base_dir, verbose)
        if result is not None:
            results.append(result)

    print(f"Successfully processed {len(results)} files")

    # Aggregate results
    aggregated_results = aggregate_results(results)

    return {
        "directory": pred_dir.name,
        "total_files": len(pred_files),
        "processed_files": len(results),
        "results": aggregated_results,
    }


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate analysis results.

    Args:
        results (List[Dict]): List of processing results

    Returns:
        Dict: Aggregated results

    """
    # Initialize aggregation containers
    overall_mpjpe = []
    overall_p_mpjpe = []

    # Camera view aggregation
    camera_view_mpjpe = defaultdict(list)
    camera_view_p_mpjpe = defaultdict(list)

    # Scale aggregation
    scale_mpjpe = defaultdict(list)
    scale_p_mpjpe = defaultdict(list)

    # Action aggregation
    action_mpjpe = defaultdict(list)
    action_p_mpjpe = defaultdict(list)

    # Action × Camera view aggregation
    action_camera_mpjpe = defaultdict(list)
    action_camera_p_mpjpe = defaultdict(list)

    # Action × Scale aggregation
    action_scale_mpjpe = defaultdict(list)
    action_scale_p_mpjpe = defaultdict(list)

    # Process each result
    for result in results:
        action = result["action"]
        camera_view = result["camera_view"]
        mpjpe_frames = result["mpjpe_per_frame"]
        p_mpjpe_frames = result["p_mpjpe_per_frame"]
        scale_classes = result["scale_classes"]

        # Overall aggregation
        overall_mpjpe.extend(mpjpe_frames)
        overall_p_mpjpe.extend(p_mpjpe_frames)

        # Camera view aggregation
        camera_view_mpjpe[camera_view].extend(mpjpe_frames)
        camera_view_p_mpjpe[camera_view].extend(p_mpjpe_frames)

        # Action aggregation
        action_mpjpe[action].extend(mpjpe_frames)
        action_p_mpjpe[action].extend(p_mpjpe_frames)

        # Action × Camera view aggregation
        action_camera_key = f"{action}_{camera_view}"
        action_camera_mpjpe[action_camera_key].extend(mpjpe_frames)
        action_camera_p_mpjpe[action_camera_key].extend(p_mpjpe_frames)

        # Scale and Action × Scale aggregation (for all cameras)
        for frame_idx, scale_class in enumerate(scale_classes):
            scale_name = get_scale_name(scale_class)
            scale_mpjpe[scale_name].append(mpjpe_frames[frame_idx])
            scale_p_mpjpe[scale_name].append(p_mpjpe_frames[frame_idx])

            action_scale_key = f"{action}_{scale_name}"
            action_scale_mpjpe[action_scale_key].append(mpjpe_frames[frame_idx])
            action_scale_p_mpjpe[action_scale_key].append(p_mpjpe_frames[frame_idx])

    # Create aggregated results
    aggregated = {
        "overall": aggregate_metrics(overall_mpjpe, overall_p_mpjpe),
        "by_camera_view": {},
        "by_scale": {},
        "by_action": {},
        "by_action_camera": {},
        "by_action_scale": {},
    }

    # Aggregate by camera view
    for camera_view in camera_view_mpjpe:
        aggregated["by_camera_view"][camera_view] = aggregate_metrics(
            camera_view_mpjpe[camera_view], camera_view_p_mpjpe[camera_view]
        )

    # Aggregate by scale
    for scale in scale_mpjpe:
        aggregated["by_scale"][scale] = aggregate_metrics(scale_mpjpe[scale], scale_p_mpjpe[scale])

    # Aggregate by action
    for action in action_mpjpe:
        aggregated["by_action"][action] = aggregate_metrics(action_mpjpe[action], action_p_mpjpe[action])

    # Aggregate by action × camera view
    for key in action_camera_mpjpe:
        aggregated["by_action_camera"][key] = aggregate_metrics(action_camera_mpjpe[key], action_camera_p_mpjpe[key])

    # Aggregate by action × scale
    for key in action_scale_mpjpe:
        aggregated["by_action_scale"][key] = aggregate_metrics(action_scale_mpjpe[key], action_scale_p_mpjpe[key])

    return aggregated


def print_results(results: Dict):
    """Print analysis results to console.

    Args:
        results (Dict): Analysis results

    """
    print(f"\n{'=' * 60}")
    print(f"Analysis Results: {results['directory']}")
    print(f"{'=' * 60}")
    print(f"Total files: {results['total_files']}")
    print(f"Processed files: {results['processed_files']}")

    agg_results = results["results"]

    # Overall results
    print("\nOverall Results:")
    overall = agg_results["overall"]
    print(f"  MPJPE: {overall['mpjpe']:.2f} mm")
    print(f"  P-MPJPE: {overall['p_mpjpe']:.2f} mm")
    print(f"  Frame count: {overall['count']}")

    # By camera view
    print("\nBy Camera View:")
    for camera_view, metrics in agg_results["by_camera_view"].items():
        print(f"  {camera_view}:")
        print(f"    MPJPE: {metrics['mpjpe']:.2f} mm")
        print(f"    P-MPJPE: {metrics['p_mpjpe']:.2f} mm")
        print(f"    Frame count: {metrics['count']}")

    # By scale
    print("\nBy Scale:")
    for scale, metrics in agg_results["by_scale"].items():
        print(f"  {scale}:")
        print(f"    MPJPE: {metrics['mpjpe']:.2f} mm")
        print(f"    P-MPJPE: {metrics['p_mpjpe']:.2f} mm")
        print(f"    Frame count: {metrics['count']}")

    # By action
    print("\nBy Action:")
    for action, metrics in agg_results["by_action"].items():
        print(f"  {action}:")
        print(f"    MPJPE: {metrics['mpjpe']:.2f} mm")
        print(f"    P-MPJPE: {metrics['p_mpjpe']:.2f} mm")
        print(f"    Frame count: {metrics['count']}")


def save_results(results: Dict, output_dir: Path):
    """Save analysis results to files.

    Args:
        results (Dict): Analysis results
        output_dir (Path): Output directory

    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    json_file = output_dir / f"{results['directory']}_results.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    # Save as text
    text_file = output_dir / f"{results['directory']}_results.txt"
    with open(text_file, "w") as f:
        f.write(f"{'=' * 60}\n")
        f.write(f"Analysis Results: {results['directory']}\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"Total files: {results['total_files']}\n")
        f.write(f"Processed files: {results['processed_files']}\n")

        agg_results = results["results"]

        # Overall results
        f.write("\nOverall Results:\n")
        overall = agg_results["overall"]
        f.write(f"  MPJPE: {overall['mpjpe']:.2f} mm\n")
        f.write(f"  P-MPJPE: {overall['p_mpjpe']:.2f} mm\n")
        f.write(f"  Frame count: {overall['count']}\n")

        # By camera view
        f.write("\nBy Camera View:\n")
        for camera_view, metrics in agg_results["by_camera_view"].items():
            f.write(f"  {camera_view}:\n")
            f.write(f"    MPJPE: {metrics['mpjpe']:.2f} mm\n")
            f.write(f"    P-MPJPE: {metrics['p_mpjpe']:.2f} mm\n")
            f.write(f"    Frame count: {metrics['count']}\n")

        # By scale
        f.write("\nBy Scale:\n")
        for scale, metrics in agg_results["by_scale"].items():
            f.write(f"  {scale}:\n")
            f.write(f"    MPJPE: {metrics['mpjpe']:.2f} mm\n")
            f.write(f"    P-MPJPE: {metrics['p_mpjpe']:.2f} mm\n")
            f.write(f"    Frame count: {metrics['count']}\n")

        # By action
        f.write("\nBy Action:\n")
        for action, metrics in agg_results["by_action"].items():
            f.write(f"  {action}:\n")
            f.write(f"    MPJPE: {metrics['mpjpe']:.2f} mm\n")
            f.write(f"    P-MPJPE: {metrics['p_mpjpe']:.2f} mm\n")
            f.write(f"    Frame count: {metrics['count']}\n")

        # By action × camera view
        f.write("\nBy Action × Camera View:\n")
        for key, metrics in agg_results["by_action_camera"].items():
            f.write(f"  {key}:\n")
            f.write(f"    MPJPE: {metrics['mpjpe']:.2f} mm\n")
            f.write(f"    P-MPJPE: {metrics['p_mpjpe']:.2f} mm\n")
            f.write(f"    Frame count: {metrics['count']}\n")

        # By action × scale
        f.write("\nBy Action × Scale:\n")
        for key, metrics in agg_results["by_action_scale"].items():
            f.write(f"  {key}:\n")
            f.write(f"    MPJPE: {metrics['mpjpe']:.2f} mm\n")
            f.write(f"    P-MPJPE: {metrics['p_mpjpe']:.2f} mm\n")
            f.write(f"    Frame count: {metrics['count']}\n")

    # Save as CSV
    csv_file = output_dir / f"{results['directory']}_results.csv"
    csv_data = []

    agg_results = results["results"]

    # Overall
    csv_data.append(
        {
            "category": "overall",
            "subcategory": "all",
            "mpjpe": agg_results["overall"]["mpjpe"],
            "p_mpjpe": agg_results["overall"]["p_mpjpe"],
            "count": agg_results["overall"]["count"],
        }
    )

    # By camera view
    for camera_view, metrics in agg_results["by_camera_view"].items():
        csv_data.append(
            {
                "category": "camera_view",
                "subcategory": camera_view,
                "mpjpe": metrics["mpjpe"],
                "p_mpjpe": metrics["p_mpjpe"],
                "count": metrics["count"],
            }
        )

    # By scale
    for scale, metrics in agg_results["by_scale"].items():
        csv_data.append(
            {
                "category": "scale",
                "subcategory": scale,
                "mpjpe": metrics["mpjpe"],
                "p_mpjpe": metrics["p_mpjpe"],
                "count": metrics["count"],
            }
        )

    # By action
    for action, metrics in agg_results["by_action"].items():
        csv_data.append(
            {
                "category": "action",
                "subcategory": action,
                "mpjpe": metrics["mpjpe"],
                "p_mpjpe": metrics["p_mpjpe"],
                "count": metrics["count"],
            }
        )

    # By action × camera view
    for key, metrics in agg_results["by_action_camera"].items():
        csv_data.append(
            {
                "category": "action_camera",
                "subcategory": key,
                "mpjpe": metrics["mpjpe"],
                "p_mpjpe": metrics["p_mpjpe"],
                "count": metrics["count"],
            }
        )

    # By action × scale
    for key, metrics in agg_results["by_action_scale"].items():
        csv_data.append(
            {
                "category": "action_scale",
                "subcategory": key,
                "mpjpe": metrics["mpjpe"],
                "p_mpjpe": metrics["p_mpjpe"],
                "count": metrics["count"],
            }
        )

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_file, index=False)

    print(f"Results saved to {json_file}, {text_file}, and {csv_file}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Analyze 3D pose predictions")
    parser.add_argument("predictions_dir", type=str, help="Directory containing prediction results")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="work_dir/analysis_results",
        help="Output directory for results",
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")

    args = parser.parse_args()

    # Setup paths
    predictions_dir = Path(args.predictions_dir)
    output_dir = Path(args.output_dir)

    # Base directories for GT and bbox data
    gt_base_dir = predictions_dir.parent / "gt_markers3d_by_cam"
    bbox_base_dir = predictions_dir.parent / "gt_bboxes2d_by_cam"

    # Check if base directories exist
    if not gt_base_dir.exists():
        raise FileNotFoundError(f"GT markers directory not found: {gt_base_dir}")
    if not bbox_base_dir.exists():
        raise FileNotFoundError(f"Bbox directory not found: {bbox_base_dir}")

    # Find prediction directories
    pred_dirs = find_prediction_directories(predictions_dir)
    if not pred_dirs:
        raise FileNotFoundError(f"No prediction directories found in {predictions_dir}")

    print(f"Found {len(pred_dirs)} prediction directories")

    # Analyze each directory
    all_results = []
    for pred_dir in pred_dirs:
        try:
            result = analyze_prediction_directory(pred_dir, gt_base_dir, bbox_base_dir, args.verbose)
            all_results.append(result)
            print_results(result)
            save_results(result, output_dir)
        except Exception as e:
            print(f"Error analyzing {pred_dir}: {str(e)}")
            if args.verbose:
                import traceback

                traceback.print_exc()

    print(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
