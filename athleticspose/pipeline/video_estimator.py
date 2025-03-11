"""Video pose estimation pipeline."""

from typing import Optional, Tuple

import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.utils import adapt_mmdet_pipeline


class VideoEstimator:
    """Video pose estimation pipeline using MMPose and MMDetection."""

    def __init__(
        self,
        mmdet_config: str,
        mmpose_config: str,
        det_weights: str,
        pose_weights: str,
        device: str = "cuda:0",
    ):
        """Initialize video pose estimator.

        Args:
            mmdet_config (str): MMDetection config file path
            mmpose_config (str): MMPose config file path
            det_weights (str): Detection model weights path
            pose_weights (str): Pose estimation model weights path
            device (str, optional): Device to run models on. Defaults to "cuda:0".

        """
        self.device = device
        self._init_models(mmdet_config, mmpose_config, det_weights, pose_weights)

    def _init_models(self, mmdet_config: str, mmpose_config: str, det_weights: str, pose_weights: str) -> None:
        """Initialize detection and pose models.

        Args:
            mmdet_config (str): MMDetection config file path
            mmpose_config (str): MMPose config file path
            det_weights (str): Detection model weights path
            pose_weights (str): Pose estimation model weights path

        """
        # Initialize detector
        self.detector = init_detector(mmdet_config, det_weights, device=self.device)

        # Initialize pose estimator
        self.pose_estimator = init_pose_estimator(mmpose_config, pose_weights, device=self.device)

        # Adapt detection model for pose estimation
        self.detector.cfg = adapt_mmdet_pipeline(self.detector.cfg)

    def _get_max_person_bbox(self, det_results) -> Optional[np.ndarray]:
        """Get bounding box for the person with largest area.

        Args:
            det_results: Detection results from MMDetection

        Returns:
            Optional[np.ndarray]: Bounding box with format [x1, y1, x2, y2]
                or None if no person detected

        """
        # Get instance predictions
        pred_instance = det_results.pred_instances.cpu().numpy()
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        person_bboxes = bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.5)]
        person_bboxes = person_bboxes[nms(person_bboxes, 0.3), :]

        # Extract person bounding boxes (assuming person is category 0)
        if len(person_bboxes) == 0:
            return None

        # Get bbox with maximum score
        max_idx = np.argmax(person_bboxes[:, -1])
        return person_bboxes[max_idx, :4]

    def process_frame(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Process a single frame.

        Args:
            frame (np.ndarray): Input frame in BGR format

        Returns:
            tuple[Optional[np.ndarray], Optional[np.ndarray]]: Keypoints and scores
                - keypoints: Shape (17, 2) or None if no person detected
                - scores: Shape (17,) or None if no person detected

        """
        # Detect person
        det_result = inference_detector(self.detector, frame)
        person_bbox = self._get_max_person_bbox(det_result)
        if person_bbox is None:
            return None, None

        # Convert to list format for pose estimation
        pose_results = inference_topdown(self.pose_estimator, frame, [person_bbox])
        if not pose_results:
            return None, None

        # Extract keypoints and scores from pose results
        pred_instances = pose_results[0].pred_instances
        keypoints = pred_instances.keypoints  # (1, 17, 2) numpy array
        scores = pred_instances.keypoint_scores  # (1, 17) numpy array

        # Remove batch dimension to match expected shape
        keypoints = keypoints.squeeze(0)  # (17, 2)
        scores = scores.squeeze(0)  # (17,)

        return keypoints, scores

    def process_video(self, video_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Process video file and extract pose keypoints.

        Args:
            video_path (str): Path to video file

        Returns:
            tuple[np.ndarray, np.ndarray]: Keypoints and scores for all frames
                - keypoints: Shape (T, 17, 2)
                - scores: Shape (T, 17)

        Raises:
            ValueError: If video file cannot be opened

        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        keypoints_list = []
        scores_list = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = self.process_frame(frame)
            if keypoints is not None:
                keypoints_list.append(keypoints)
                scores_list.append(scores)

        cap.release()

        if not keypoints_list:
            raise ValueError("No valid pose detected in video")

        return np.stack(keypoints_list), np.stack(scores_list)
