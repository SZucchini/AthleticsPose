"""Gradio application for 3D pose estimation."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import gradio as gr

from demo.pipeline import VideoPosePipeline
from demo.visualization import PoseVisualizer

DEFAULT_CHECKPOINT = "work_dir/20250302_110906/best.ckpt"


def create_pose_demo(checkpoint_path: Optional[str] = None) -> gr.Blocks:
    """Create pose estimation demo.

    Args:
        checkpoint_path (Optional[str], optional): Path to model checkpoint.
            If None, use default checkpoint. Defaults to None.

    Returns:
        gr.Blocks: Gradio interface

    """
    if checkpoint_path is None:
        checkpoint_path = DEFAULT_CHECKPOINT

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Initialize pipeline and visualizer
    pipeline = VideoPosePipeline(checkpoint_path=checkpoint_path)
    visualizer = PoseVisualizer()

    def process_video(video_path: str, progress: gr.Progress) -> str:
        """Process video and create visualization.

        Args:
            video_path (str): Path to input video
            progress (gr.Progress): Progress indicator

        Returns:
            str: Path to output video

        """
        # Update progress
        progress(0, desc="Estimating poses...")

        # Process video
        poses_3d = pipeline.process_video(video_path)

        # Update progress
        progress(0.5, desc="Creating visualization...")

        # Create animation
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "output.mp4")
        output_video = visualizer.create_animation(poses_3d, output_path)

        # Update progress
        progress(1.0, desc="Done!")

        return output_video

    # Create Gradio interface
    with gr.Blocks(title="3D Pose Estimation") as demo:
        gr.Markdown(
            """
            # 3D Pose Estimation Demo
            Upload a video to estimate 3D poses.

            **Note:**
            - Maximum video duration: 3 seconds
            - Recommended resolution: HD (1280x720)
            """
        )

        with gr.Row():
            with gr.Column():
                input_video = gr.Video(
                    label="Input Video",
                    format="mp4",
                    sources=["upload"],
                    type="filepath",
                )
                submit_btn = gr.Button("Estimate Poses", variant="primary")

            with gr.Column():
                output_video = gr.Video(
                    label="3D Pose Animation",
                    format="mp4",
                    interactive=False,
                )

        # Set up event handler
        submit_btn.click(
            fn=process_video,
            inputs=[input_video],
            outputs=[output_video],
        )

    return demo


def main():
    """Run the Gradio application."""
    demo = create_pose_demo()
    demo.launch()


if __name__ == "__main__":
    main()
