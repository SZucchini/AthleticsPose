"""2D marker-based 3D pose prediction script for AthleticsPose."""

import os
import time

import hydra
from omegaconf import OmegaConf

from athleticspose.prediction import From2DMarkersPredictor


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: OmegaConf) -> None:
    """Main function for 2D marker-based prediction.

    Args:
        cfg: Hydra configuration object

    """
    start_time = time.time()

    print("=" * 80)
    print("2D Marker-based 3D Pose Prediction")
    print("=" * 80)

    # Print configuration
    marker_type = cfg.prediction.input.marker_type
    checkpoint_path = cfg.prediction.model.checkpoint_path
    print(f"Marker type: {marker_type}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test subjects: {cfg.prediction.test_subjects}")
    print()

    try:
        # Initialize predictor
        print("Initializing predictor...")
        predictor = From2DMarkersPredictor(cfg)
        print("✓ Predictor initialized successfully")
        print()

        # Run predictions
        print("Running predictions...")
        results = predictor.predict_all_files()
        print()

        # Save results
        print("Saving predictions...")
        predictor.save_predictions(results)
        print()

        # Print summary
        successful_results = [r for r in results if r["success"]]
        failed_results = [r for r in results if not r["success"]]

        total_time = time.time() - start_time
        avg_time = (
            sum(r["processing_time"] for r in successful_results) / len(successful_results)
            if successful_results
            else 0
        )

        print("=" * 80)
        print("PREDICTION SUMMARY")
        print("=" * 80)
        print(f"Total files processed: {len(results)}")
        print(f"Successful predictions: {len(successful_results)}")
        print(f"Failed predictions: {len(failed_results)}")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Average time per file: {avg_time:.2f}s")

        if failed_results:
            print("\nFailed files:")
            for result in failed_results:
                print(f"  - {os.path.basename(result['input_file'])}: {result['error']}")

        print("=" * 80)
        print("✓ Prediction completed successfully!")

    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
