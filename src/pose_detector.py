from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class PoseResult:
    keypoints: np.ndarray
    confidences: np.ndarray
    bbox: tuple[int, int, int, int]
    annotated_frame: np.ndarray


class PoseDetector:
    """Runs YOLO pose estimation and returns the most confident person."""

    def __init__(self, model_name: str = "yolo11n-pose.pt", confidence: float = 0.35):
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, frame: np.ndarray) -> Optional[PoseResult]:
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )

        result = results[0]
        annotated = result.plot()

        if result.keypoints is None or result.boxes is None:
            return None

        if len(result.keypoints.xy) == 0:
            return None

        boxes_conf = result.boxes.conf.cpu().numpy()
        best_index = int(np.argmax(boxes_conf))

        keypoints = result.keypoints.xy[best_index].cpu().numpy()
        if result.keypoints.conf is not None:
            confidences = result.keypoints.conf[best_index].cpu().numpy()
        else:
            confidences = np.ones(len(keypoints), dtype=np.float32)

        x1, y1, x2, y2 = result.boxes.xyxy[best_index].cpu().numpy().astype(int)

        return PoseResult(
            keypoints=keypoints,
            confidences=confidences,
            bbox=(x1, y1, x2, y2),
            annotated_frame=annotated
        )
