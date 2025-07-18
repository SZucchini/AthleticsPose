"""Unified evaluation script for AthleticsPose."""

from typing import Any, Callable, Dict, List

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from athleticspose.datasets.dynamic_dataset import DynamicMotionDataset3D
from athleticspose.loss import calc_mpjpe, loss_velocity, n_mpjpe, p_mpjpe
from athleticspose.plmodules.linghtning_module import LightningPose3D
from athleticspose.statics.joints import h36m_joints_name_to_index


def calculate_sample_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    p2mm: torch.Tensor,
    norm_scale: torch.Tensor,
    valid_length: torch.Tensor,
    cfg: OmegaConf,
) -> Dict[str, float]:
    """Calculate metrics for a single sample using valid_length for accurate frame masking.

    Args:
        pred: Predicted 3D pose. Shape: (batch_size, frames, joints, 3)
        target: Ground truth 3D pose. Shape: (batch_size, frames, joints, 3)
        p2mm: Scale factors pixel to mm. Shape: (batch_size, frames)
        norm_scale: Normalization scale factors. Shape: (batch_size,)
        valid_length: Number of valid frames per sample. Shape: (batch_size,)
        cfg: Evaluation configuration

    Returns:
        Dictionary of calculated metrics

    """
    # Root joint zero setting (consistent with evaluate_by_action.py)
    pred = pred.clone()
    pred[:, :, 0, :] = 0

    # Convert to numpy for metric calculations
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    p2mm_np = p2mm.cpu().numpy()
    norm_scale_np = norm_scale.cpu().numpy() if isinstance(norm_scale, torch.Tensor) else norm_scale
    valid_length_np = valid_length.cpu().numpy()

    # Denormalization (following evaluate_by_action.py approach)
    pred_denom = pred_np * norm_scale_np[:, None, None, None]
    target_denom = target_np * norm_scale_np[:, None, None, None]

    # Safe division avoiding zero division for padded frames
    with np.errstate(divide="ignore", invalid="ignore"):
        pred_denom = pred_denom / p2mm_np[:, :, None, None]
        target_denom = target_denom / p2mm_np[:, :, None, None]

    # Extract only valid frames using valid_length
    batch_size = pred_denom.shape[0]
    pred_valid_list = []
    target_valid_list = []

    for i in range(batch_size):
        valid_len = int(valid_length_np[i].item())
        if valid_len > 0:
            pred_valid_list.append(pred_denom[i, :valid_len])
            target_valid_list.append(target_denom[i, :valid_len])

    if not pred_valid_list:
        # No valid frames in any sample
        return {
            metric: 0.0
            for metric in ["mpjpe", "pa_mpjpe", "n_mpjpe", "velocity_error"]
            if getattr(cfg.evaluation.metrics, f"enable_{metric}", False)
        }

    # Concatenate all valid frames for metric calculation
    pred_valid = np.concatenate(pred_valid_list, axis=0)  # Shape: (total_valid_frames, joints, 3)
    target_valid = np.concatenate(target_valid_list, axis=0)

    # Reshape for calc_mpjpe compatibility: (1, frames, joints, 3)
    pred_for_calc = pred_valid[None, ...]
    target_for_calc = target_valid[None, ...]

    metrics = {}

    # MPJPE calculation (following evaluate_by_action.py)
    if cfg.evaluation.metrics.enable_mpjpe:
        mpjpe_values = calc_mpjpe(pred_for_calc, target_for_calc)
        metrics["mpjpe"] = float(np.mean(mpjpe_values))

        # Per-joint MPJPE calculation
        if cfg.evaluation.output.get("print_per_joint", False):
            # calc_mpjpe returns (B, J) shape - we have B=1 so take [0]
            per_joint_mpjpe = mpjpe_values[0]  # Shape: (17,)
            metrics["mpjpe_per_joint"] = per_joint_mpjpe.tolist()

    # PA-MPJPE calculation (following evaluate_by_action.py approach)
    if cfg.evaluation.metrics.enable_pa_mpjpe:
        try:
            pa_mpjpe_val = p_mpjpe(pred_valid, target_valid)
            metrics["pa_mpjpe"] = float(np.mean(pa_mpjpe_val))
        except np.linalg.LinAlgError as e:
            # SVD convergence error - raise error instead of skip
            raise RuntimeError("PA-MPJPE calculation failed due to SVD convergence error") from e

    # N-MPJPE and Velocity Error (if enabled)
    if cfg.evaluation.metrics.enable_n_mpjpe:
        pred_tensor = torch.from_numpy(pred_for_calc)
        target_tensor = torch.from_numpy(target_for_calc)
        metrics["n_mpjpe"] = float(n_mpjpe(pred_tensor, target_tensor).item())

    if cfg.evaluation.metrics.enable_velocity_error:
        pred_tensor = torch.from_numpy(pred_for_calc)
        target_tensor = torch.from_numpy(target_for_calc)
        metrics["velocity_error"] = float(loss_velocity(pred_tensor, target_tensor).item())

    return metrics


def aggregate_results_by_key(
    results: List[Dict[str, Any]], key_func: Callable[[Dict[str, Any]], str]
) -> Dict[str, Dict[str, Any]]:
    """Aggregate evaluation results by a given key function.

    Args:
        results: List of result dictionaries
        key_func: Function to extract grouping key from result dict

    Returns:
        Dictionary mapping keys to aggregated metrics

    """
    grouped = {}
    for result in results:
        key = key_func(result)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(result)

    # Calculate averages for each group
    aggregated = {}
    for key, group_results in grouped.items():
        metric_values = {}
        for result in group_results:
            for metric_name, value in result["metrics"].items():
                if metric_name not in metric_values:
                    metric_values[metric_name] = []

                # Handle per-joint metrics separately
                if metric_name.endswith("_per_joint"):
                    metric_values[metric_name].append(value)
                else:
                    metric_values[metric_name].append(value)

        # Calculate mean for each metric
        aggregated[key] = {}
        for metric, values in metric_values.items():
            if metric.endswith("_per_joint"):
                # For per-joint metrics, calculate mean across samples for each joint
                if values:  # Check if we have any values
                    try:
                        # Check if all samples have the same number of joints
                        first_length = len(values[0]) if values else 0
                        if all(len(v) == first_length for v in values):
                            joint_values = np.array(values)  # Shape: (num_samples, num_joints)
                            aggregated[key][metric] = np.mean(joint_values, axis=0).tolist()
                        else:
                            # Handle inconsistent joint counts by taking the minimum common length
                            min_length = min(len(v) for v in values)
                            truncated_values = [v[:min_length] for v in values]
                            joint_values = np.array(truncated_values)
                            aggregated[key][metric] = np.mean(joint_values, axis=0).tolist()
                    except (ValueError, TypeError):
                        # Skip problematic per-joint metrics
                        continue
            else:
                aggregated[key][metric] = float(np.mean(values))

        aggregated[key]["sample_count"] = len(group_results)

    return aggregated


def aggregate_per_joint_results(
    results: List[Dict[str, Any]],
) -> Dict[str, List[float]]:
    """Aggregate per-joint results across all samples.

    Args:
        results: List of result dictionaries containing per-joint metrics

    Returns:
        Dictionary mapping metric names to per-joint averages

    """
    per_joint_metrics = {}

    for result in results:
        for metric_name, value in result["metrics"].items():
            if metric_name.endswith("_per_joint"):
                if metric_name not in per_joint_metrics:
                    per_joint_metrics[metric_name] = []
                per_joint_metrics[metric_name].append(value)

    # Calculate mean across samples for each joint
    aggregated_per_joint = {}
    for metric_name, values in per_joint_metrics.items():
        if values:
            try:
                # Check if all samples have the same number of joints
                first_length = len(values[0]) if values else 0
                if all(len(v) == first_length for v in values):
                    joint_values = np.array(values)  # Shape: (num_samples, num_joints)
                    aggregated_per_joint[metric_name] = np.mean(joint_values, axis=0).tolist()
                else:
                    # Handle inconsistent joint counts by taking the minimum common length
                    min_length = min(len(v) for v in values)
                    truncated_values = [v[:min_length] for v in values]
                    joint_values = np.array(truncated_values)
                    aggregated_per_joint[metric_name] = np.mean(joint_values, axis=0).tolist()
            except (ValueError, TypeError) as e:
                print(f"Warning: Could not aggregate {metric_name}: {e}")
                continue

    return aggregated_per_joint


def evaluate_model(cfg: OmegaConf) -> Dict[str, Any]:
    """Evaluate the model using the provided configuration.

    Args:
        cfg: Hydra configuration object

    Returns:
        Dictionary containing evaluation results and metadata

    """
    # Device auto-selection
    if cfg.evaluation.runtime.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(cfg.evaluation.runtime.device)

    if cfg.evaluation.output.verbose:
        print(f"Using device: {device}")

    # Model loading
    checkpoint_path = cfg.evaluation.checkpoint.path

    # Check if we need to load only pretrained weights or a full Lightning checkpoint
    if checkpoint_path is None or checkpoint_path == "null":
        # Load only pretrained weights (no Lightning checkpoint)
        if not (hasattr(cfg.evaluation.checkpoint, "h36m_ckpt") and cfg.evaluation.checkpoint.h36m_ckpt):
            raise RuntimeError("Either checkpoint.path or checkpoint.h36m_ckpt must be provided")

        h36m_ckpt_path = cfg.evaluation.checkpoint.h36m_ckpt
        if cfg.evaluation.output.verbose:
            print(f"Loading model with pretrained weights only: {h36m_ckpt_path}")

        try:
            # Initialize model with configuration
            model = LightningPose3D(cfg=cfg)
            model.eval()
            model.to(device)

            # Load pretrained weights
            try:
                state_dict = torch.load(h36m_ckpt_path, map_location=device)
                if "module." in list(state_dict.keys())[0]:
                    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.model.load_state_dict(state_dict)
                if cfg.evaluation.output.verbose:
                    print("Pretrained weights loaded successfully")
            except FileNotFoundError as e:
                raise RuntimeError(f"Pretrained checkpoint file not found: {h36m_ckpt_path}") from e
            except Exception as e:
                raise RuntimeError(f"Failed to load pretrained checkpoint from {h36m_ckpt_path}: {str(e)}") from e

        except Exception as e:
            raise RuntimeError(f"Failed to initialize model with pretrained weights: {str(e)}") from e

    else:
        # Load full Lightning checkpoint
        if cfg.evaluation.output.verbose:
            print(f"Loading Lightning checkpoint: {checkpoint_path}")

        try:
            model = LightningPose3D.load_from_checkpoint(checkpoint_path, cfg=cfg)
            model.eval()
            model.to(device)

            # Load additional H36M pretrained weights if specified
            if hasattr(cfg.evaluation.checkpoint, "h36m_ckpt") and cfg.evaluation.checkpoint.h36m_ckpt:
                h36m_ckpt_path = cfg.evaluation.checkpoint.h36m_ckpt
                if cfg.evaluation.output.verbose:
                    print(f"Loading additional H36M pretrained weights: {h36m_ckpt_path}")

                try:
                    h36m_state_dict = torch.load(h36m_ckpt_path, map_location=device)
                    if "module." in list(h36m_state_dict.keys())[0]:
                        h36m_state_dict = {k.replace("module.", ""): v for k, v in h36m_state_dict.items()}
                    model.model.load_state_dict(h36m_state_dict)
                    if cfg.evaluation.output.verbose:
                        print("Additional H36M pretrained weights loaded successfully")
                except FileNotFoundError as e:
                    raise RuntimeError(f"H36M checkpoint file not found: {h36m_ckpt_path}") from e
                except Exception as e:
                    raise RuntimeError(f"Failed to load H36M checkpoint from {h36m_ckpt_path}: {str(e)}") from e

        except Exception as e:
            raise RuntimeError(f"Failed to load model from {checkpoint_path}: {str(e)}") from e

    # Dataset creation
    if cfg.evaluation.output.verbose:
        print("Creating dataset with filters:")
        print(f"  Actions: {cfg.data.filter_actions}")
        print(f"  Subjects: {cfg.data.filter_subjects}")
        print(f"  Cameras: {cfg.data.filter_cameras}")

    try:
        dataset = DynamicMotionDataset3D(
            cfg=cfg.data,
            split="test",
            transform=None,
            flip=False,  # No augmentation during evaluation
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create dataset: {str(e)}") from e

    # DataLoader creation
    try:
        dataloader = DataLoader(
            dataset,
            batch_size=cfg.evaluation.runtime.batch_size,
            shuffle=False,
            num_workers=cfg.evaluation.runtime.num_workers,
            pin_memory=(device.type == "cuda"),  # Optimize for GPU transfer
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create dataloader: {str(e)}") from e

    if cfg.evaluation.output.verbose:
        print(f"Dataset size: {len(dataset)} clips")

    # Get dataset metadata
    unique_actions = dataset.get_unique_actions()
    unique_subjects = dataset.get_unique_subjects()

    if cfg.evaluation.output.verbose:
        print(f"Actions in dataset: {unique_actions}")
        print(f"Subjects in dataset: {unique_subjects}")
        print("\nRunning evaluation...")

    # Evaluation execution
    results = []
    len(dataloader)

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Handle both 4-tuple and 5-tuple batch formats
            if len(batch) == 5:
                input2d, label3d, p2mm, norm_scale, valid_length = batch
            else:
                input2d, label3d, p2mm, norm_scale = batch
                valid_length = None

            # Move to device
            input2d = input2d.to(device, non_blocking=True)
            label3d = label3d.to(device, non_blocking=True)
            p2mm = p2mm.to(device, non_blocking=True)

            # Model inference
            try:
                pred = model(input2d)
            except Exception as e:
                raise RuntimeError(f"Model inference failed on batch {batch_idx}: {str(e)}") from e

            # Process each sample in the batch
            batch_size = input2d.shape[0]
            for i in range(batch_size):
                sample_idx = batch_idx * cfg.evaluation.runtime.batch_size + i
                if sample_idx >= len(dataset):
                    break

                # Get clip information
                try:
                    clip_info = dataset.get_clip_info(sample_idx)
                except Exception as e:
                    raise RuntimeError(f"Failed to get clip info for sample {sample_idx}: {str(e)}") from e

                # Extract single sample tensors
                sample_pred = pred[i : i + 1]  # Keep batch dimension
                sample_label = label3d[i : i + 1]
                sample_p2mm = p2mm[i : i + 1]
                sample_norm_scale = (
                    norm_scale[i : i + 1] if isinstance(norm_scale, torch.Tensor) else norm_scale[i : i + 1]
                )

                # Extract valid_length for this sample
                if valid_length is not None:
                    sample_valid_length = valid_length[i : i + 1]
                else:
                    # Fallback for datasets not returning valid_length
                    sample_valid_length = torch.tensor([sample_pred.shape[1]], dtype=torch.long)

                # Calculate metrics
                try:
                    metrics = calculate_sample_metrics(
                        sample_pred,
                        sample_label,
                        sample_p2mm,
                        sample_norm_scale,
                        sample_valid_length,
                        cfg,
                    )
                except Exception as e:
                    raise RuntimeError(f"Metrics calculation failed for sample {sample_idx}: {str(e)}") from e

                # Store result
                results.append(
                    {
                        "action": clip_info["action"],
                        "subject": clip_info["subject"],
                        "camera_idx": clip_info["camera_idx"],
                        "sample_idx": sample_idx,
                        "metrics": metrics,
                    }
                )

            # Progress reporting
            if cfg.evaluation.output.show_progress and batch_idx % 10 == 0:
                processed_samples = min((batch_idx + 1) * cfg.evaluation.runtime.batch_size, len(dataset))
                print(f"Processed {processed_samples}/{len(dataset)} samples...")

    if cfg.evaluation.output.verbose:
        print(f"Evaluation completed. Total samples: {len(results)}")

    return {
        "results": results,
        "unique_actions": unique_actions,
        "unique_subjects": unique_subjects,
        "dataset_size": len(dataset),
        "processed_samples": len(results),
        "device": str(device),
    }


def print_evaluation_results(
    results: List[Dict],
    unique_actions: List[str],
    unique_subjects: List[str],
    cfg: OmegaConf,
):
    """Print comprehensive evaluation results with configurable display options.

    Args:
        results: List of evaluation results from evaluate_model
        unique_actions: List of unique actions in dataset
        unique_subjects: List of unique subjects in dataset
        cfg: Hydra configuration object

    """
    if not results:
        print("No evaluation results to display.")
        return

    decimal_places = cfg.evaluation.output.decimal_places
    show_per_joint = cfg.evaluation.output.get("print_per_joint", False)

    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)

    # Calculate overall statistics (exclude per-joint metrics from overall stats)
    all_metrics = {}
    for result in results:
        for metric_name, value in result["metrics"].items():
            # Skip per-joint metrics for overall statistics
            if metric_name.endswith("_per_joint"):
                continue
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Print overall results
    print("\nOVERALL RESULTS:")
    print("-" * 40)
    for metric_name, values in all_metrics.items():
        mean_val = np.mean(values)
        std_val = np.std(values)
        unit = "mm" if "mpjpe" in metric_name.lower() else ("mm" if "velocity" in metric_name.lower() else "")
        print(
            f"{metric_name:<15}: {mean_val:>{decimal_places + 6}.{decimal_places}f} ± "
            f"{std_val:>{decimal_places + 3}.{decimal_places}f} {unit}"
        )
    print(f"{'Total samples':<15}: {len(results):>6}")

    # Print per-joint results (if enabled)
    if show_per_joint:
        print("\nPER-JOINT RESULTS:")
        print("-" * 40)

        per_joint_aggregated = aggregate_per_joint_results(results)

        # Create joint names list in order
        joint_names = list(h36m_joints_name_to_index.keys())

        for metric_name, joint_values in per_joint_aggregated.items():
            base_metric = metric_name.replace("_per_joint", "")
            unit = "mm" if "mpjpe" in base_metric.lower() else ("mm" if "velocity" in base_metric.lower() else "")

            print(f"\n{base_metric.upper()} by joint:")
            for _, (joint_name, value) in enumerate(zip(joint_names, joint_values, strict=False)):
                print(f"  {joint_name:<12}: {value:>{decimal_places + 6}.{decimal_places}f} {unit}")

    # Print results by action (only if multiple actions exist and enabled)
    if len(unique_actions) > 1 and cfg.evaluation.output.print_per_action:
        print("\nRESULTS BY ACTION:")
        print("-" * 40)

        action_aggregated = aggregate_results_by_key(results, lambda x: x["action"])
        for action in sorted(action_aggregated.keys()):
            metrics = action_aggregated[action]
            sample_count = metrics.pop("sample_count")

            print(f"\n{action}:")
            for metric_name, value in metrics.items():
                # Skip per-joint metrics and sample_count for action/subject display
                if metric_name.endswith("_per_joint") or metric_name == "sample_count":
                    continue
                unit = "mm" if "mpjpe" in metric_name.lower() else ("mm" if "velocity" in metric_name.lower() else "")
                print(f"  {metric_name:<13}: {value:>{decimal_places + 6}.{decimal_places}f} {unit}")
            print(f"  {'samples':<13}: {sample_count:>6}")

            # Print per-joint results for this action (if enabled)
            if show_per_joint:
                action_results = [r for r in results if r["action"] == action]
                if action_results:
                    action_per_joint = aggregate_per_joint_results(action_results)
                    joint_names = list(h36m_joints_name_to_index.keys())

                    for metric_name, joint_values in action_per_joint.items():
                        base_metric = metric_name.replace("_per_joint", "")
                        unit = (
                            "mm"
                            if "mpjpe" in base_metric.lower()
                            else ("mm" if "velocity" in base_metric.lower() else "")
                        )

                        print(f"\n  {base_metric.upper()} by joint for {action}:")
                        for joint_name, value in zip(joint_names, joint_values, strict=False):
                            print(f"    {joint_name:<12}: {value:>{decimal_places + 6}.{decimal_places}f} {unit}")

    # Print results by subject (only if multiple subjects exist and enabled)
    if len(unique_subjects) > 1 and cfg.evaluation.output.print_per_subject:
        print("\nRESULTS BY SUBJECT:")
        print("-" * 40)

        subject_aggregated = aggregate_results_by_key(results, lambda x: x["subject"])
        for subject in sorted(subject_aggregated.keys()):
            metrics = subject_aggregated[subject]
            sample_count = metrics.pop("sample_count")

            print(f"\n{subject}:")
            for metric_name, value in metrics.items():
                # Skip per-joint metrics for subject display
                if metric_name.endswith("_per_joint"):
                    continue
                unit = "mm" if "mpjpe" in metric_name.lower() else ("mm" if "velocity" in metric_name.lower() else "")
                print(f"  {metric_name:<13}: {value:>{decimal_places + 6}.{decimal_places}f} {unit}")
            print(f"  {'samples':<13}: {sample_count:>6}")

    # Print dataset summary information
    if cfg.evaluation.output.verbose:
        print("\nDATASET SUMMARY:")
        print("-" * 40)
        print(f"Actions: {unique_actions}")
        print(f"Subjects: {unique_subjects}")
        print(f"Total clips processed: {len(results)}")

        # Camera distribution
        camera_counts = {}
        for result in results:
            camera_idx = result["camera_idx"]
            camera_counts[camera_idx] = camera_counts.get(camera_idx, 0) + 1
        if camera_counts:
            camera_list = [f"Cam{idx}({count})" for idx, count in sorted(camera_counts.items())]
            print(f"Camera distribution: {', '.join(camera_list)}")

    print("=" * 80)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: OmegaConf) -> None:
    """Main entry point for evaluation script.

    Args:
        cfg: Hydra configuration object containing all evaluation settings

    """
    try:
        # Configuration validation
        _validate_config(cfg)

        # Print configuration summary if verbose
        if cfg.evaluation.output.verbose:
            _print_config_summary(cfg)

        # Execute evaluation
        evaluation_results = evaluate_model(cfg)

        # Display results
        print_evaluation_results(
            results=evaluation_results["results"],
            unique_actions=evaluation_results["unique_actions"],
            unique_subjects=evaluation_results["unique_subjects"],
            cfg=cfg,
        )

        # Print final summary
        if cfg.evaluation.output.verbose:
            print(f"\nEvaluation completed successfully on {evaluation_results['device']}")
            print(f"Processed {evaluation_results['processed_samples']}/{evaluation_results['dataset_size']} samples")

    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\nEvaluation failed: {str(e)}")
        if cfg.evaluation.output.verbose:
            import traceback

            traceback.print_exc()
        exit(1)


def _validate_config(cfg: OmegaConf) -> None:
    """Validate configuration parameters with detailed error messages.

    Args:
        cfg: Hydra configuration object

    Raises:
        RuntimeError: If configuration is invalid

    """
    # Check evaluation configuration exists
    if not hasattr(cfg, "evaluation"):
        raise RuntimeError("Missing 'evaluation' configuration section")

    # Validate checkpoint configuration
    if not hasattr(cfg.evaluation, "checkpoint"):
        raise RuntimeError("Missing 'evaluation.checkpoint' configuration section")

    # Either checkpoint.path or checkpoint.h36m_ckpt must be provided
    has_checkpoint_path = (
        hasattr(cfg.evaluation.checkpoint, "path")
        and cfg.evaluation.checkpoint.path
        and cfg.evaluation.checkpoint.path != "null"
    )
    has_h36m_ckpt = hasattr(cfg.evaluation.checkpoint, "h36m_ckpt") and cfg.evaluation.checkpoint.h36m_ckpt

    if not has_checkpoint_path and not has_h36m_ckpt:
        raise RuntimeError("Either evaluation.checkpoint.path or evaluation.checkpoint.h36m_ckpt must be provided")

    # Validate runtime settings
    if not hasattr(cfg.evaluation, "runtime"):
        raise RuntimeError("Missing 'evaluation.runtime' configuration section")

    runtime = cfg.evaluation.runtime
    if runtime.batch_size <= 0:
        raise RuntimeError(f"Invalid batch_size: {runtime.batch_size}. Must be > 0")

    if runtime.num_workers < 0:
        raise RuntimeError(f"Invalid num_workers: {runtime.num_workers}. Must be >= 0")

    # Validate device setting
    valid_devices = ["auto", "cpu", "cuda"]
    if runtime.device not in valid_devices:
        raise RuntimeError(f"Invalid device: {runtime.device}. Must be one of {valid_devices}")

    # Validate metrics configuration
    if not hasattr(cfg.evaluation, "metrics"):
        raise RuntimeError("Missing 'evaluation.metrics' configuration section")

    # Check at least one metric is enabled
    metrics = cfg.evaluation.metrics
    enabled_metrics = [
        metrics.enable_mpjpe,
        metrics.enable_pa_mpjpe,
        metrics.enable_n_mpjpe,
        metrics.enable_velocity_error,
    ]
    if not any(enabled_metrics):
        raise RuntimeError("At least one metric must be enabled")

    # Validate output settings
    if not hasattr(cfg.evaluation, "output"):
        raise RuntimeError("Missing 'evaluation.output' configuration section")

    output = cfg.evaluation.output
    if output.decimal_places < 0 or output.decimal_places > 10:
        raise RuntimeError(f"Invalid decimal_places: {output.decimal_places}. Must be 0-10")

    # Validate data configuration
    if not hasattr(cfg, "data"):
        raise RuntimeError("Missing 'data' configuration section")

    # Check data directory existence
    if hasattr(cfg.data, "test_dir") and cfg.data.test_dir:
        import os

        if not os.path.exists(cfg.data.test_dir):
            raise RuntimeError(f"Test data directory not found: {cfg.data.test_dir}")


def _print_config_summary(cfg: OmegaConf) -> None:
    """Print configuration summary for transparency.

    Args:
        cfg: Hydra configuration object

    """
    print("=" * 80)
    print("EVALUATION CONFIGURATION SUMMARY")
    print("=" * 80)

    # Checkpoint info
    print(f"Checkpoint path: {cfg.evaluation.checkpoint.path}")

    # Runtime settings
    runtime = cfg.evaluation.runtime
    print("Runtime settings:")
    print(f"  - Batch size: {runtime.batch_size}")
    print(f"  - Num workers: {runtime.num_workers}")
    print(f"  - Device: {runtime.device}")
    print(f"  - Precision: {runtime.precision}")

    # Enabled metrics
    metrics = cfg.evaluation.metrics
    enabled = []
    if metrics.enable_mpjpe:
        enabled.append("MPJPE")
    if metrics.enable_pa_mpjpe:
        enabled.append("PA-MPJPE")
    if metrics.enable_n_mpjpe:
        enabled.append("N-MPJPE")
    if metrics.enable_velocity_error:
        enabled.append("Velocity Error")
    print(f"Enabled metrics: {', '.join(enabled)}")

    # Data filters
    print("Data filters:")
    print(f"  - Actions: {cfg.data.filter_actions}")
    print(f"  - Subjects: {cfg.data.filter_subjects}")
    print(f"  - Cameras: {cfg.data.filter_cameras}")

    # Output settings
    output = cfg.evaluation.output
    print("Output settings:")
    print(f"  - Verbose: {output.verbose}")
    print(f"  - Show progress: {output.show_progress}")
    print(f"  - Per-action results: {output.print_per_action}")
    print(f"  - Per-subject results: {output.print_per_subject}")
    print(f"  - Decimal places: {output.decimal_places}")

    print("=" * 80)


if __name__ == "__main__":
    main()
