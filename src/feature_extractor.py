from __future__ import annotations

import math
import numpy as np


# COCO keypoint indexes used by YOLO pose:
# 5 left shoulder, 6 right shoulder
# 11 left hip, 12 right hip
# 13 left knee, 14 right knee
# 15 left ankle, 16 right ankle


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a + b) / 2.0


def safe_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def extract_features(
    keypoints: np.ndarray,
    confidences: np.ndarray,
    bbox: tuple[int, int, int, int],
    frame_shape: tuple[int, int, int]
) -> np.ndarray:
    """
    Returns one normalized feature vector for a single frame.
    """
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = bbox

    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_hip, right_hip = keypoints[11], keypoints[12]
    left_knee, right_knee = keypoints[13], keypoints[14]
    left_ankle, right_ankle = keypoints[15], keypoints[16]

    shoulder_center = midpoint(left_shoulder, right_shoulder)
    hip_center = midpoint(left_hip, right_hip)
    knee_center = midpoint(left_knee, right_knee)
    ankle_center = midpoint(left_ankle, right_ankle)

    torso_vector = shoulder_center - hip_center
    torso_angle = math.degrees(
        math.atan2(abs(float(torso_vector[0])), abs(float(torso_vector[1])) + 1e-6)
    )

    bbox_width = max(x2 - x1, 1)
    bbox_height = max(y2 - y1, 1)
    aspect_ratio = bbox_width / bbox_height

    body_center = (shoulder_center + hip_center) / 2.0

    selected_confidence = np.mean(
        confidences[[5, 6, 11, 12, 13, 14, 15, 16]]
    )

    features = np.array([
        shoulder_center[0] / width,
        shoulder_center[1] / height,
        hip_center[0] / width,
        hip_center[1] / height,
        knee_center[0] / width,
        knee_center[1] / height,
        ankle_center[0] / width,
        ankle_center[1] / height,
        body_center[0] / width,
        body_center[1] / height,
        torso_angle / 90.0,
        aspect_ratio,
        bbox_width / width,
        bbox_height / height,
        safe_distance(shoulder_center, hip_center) / height,
        safe_distance(hip_center, ankle_center) / height,
        float(selected_confidence),
    ], dtype=np.float32)

    return features
