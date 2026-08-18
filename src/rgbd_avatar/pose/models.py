"""Data structures shared by pose-estimation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .halpe26 import HALPE26_NAMES


@dataclass
class Pose2D:
    """A single detected person's Halpe26 pose in image coordinates."""

    keypoints: np.ndarray
    scores: np.ndarray
    bbox_xyxy: np.ndarray
    bbox_score: float

    def __post_init__(self) -> None:
        self.keypoints = np.asarray(self.keypoints, dtype=np.float32)
        self.scores = np.asarray(self.scores, dtype=np.float32)
        self.bbox_xyxy = np.asarray(self.bbox_xyxy, dtype=np.float32)
        self.bbox_score = float(self.bbox_score)

        expected_keypoints = (len(HALPE26_NAMES), 2)
        if self.keypoints.shape != expected_keypoints:
            raise ValueError(
                "Expected Halpe26 keypoints with shape "
                f"{expected_keypoints}, got {self.keypoints.shape}."
            )
        if self.scores.shape != (len(HALPE26_NAMES),):
            raise ValueError(
                f"Expected {len(HALPE26_NAMES)} scores, got {self.scores.shape}."
            )
        if self.bbox_xyxy.shape != (4,):
            raise ValueError(
                f"Expected bbox shape (4,), got {self.bbox_xyxy.shape}."
            )

    @property
    def mean_score(self) -> float:
        return float(np.mean(self.scores))

    def to_dict(self, score_threshold: float = 0.0) -> dict[str, Any]:
        keypoints = []
        for index, name in enumerate(HALPE26_NAMES):
            score = float(self.scores[index])
            keypoints.append(
                {
                    "id": index,
                    "name": name,
                    "x": float(self.keypoints[index, 0]),
                    "y": float(self.keypoints[index, 1]),
                    "confidence": score,
                    "valid": score >= score_threshold,
                }
            )

        return {
            "keypoint_format": "halpe26",
            "bbox_xyxy": self.bbox_xyxy.tolist(),
            "bbox_score": self.bbox_score,
            "mean_keypoint_score": self.mean_score,
            "keypoints": keypoints,
        }


@dataclass
class Pose3D:
    """A Halpe26 pose in the right-handed metric camera coordinate system."""

    joints_m: np.ndarray
    confidence: np.ndarray
    valid: np.ndarray
    depth_m: np.ndarray
    depth_confidence: np.ndarray

    def __post_init__(self) -> None:
        count = len(HALPE26_NAMES)
        self.joints_m = np.asarray(self.joints_m, dtype=np.float32)
        self.confidence = np.asarray(self.confidence, dtype=np.float32)
        self.valid = np.asarray(self.valid, dtype=bool)
        self.depth_m = np.asarray(self.depth_m, dtype=np.float32)
        self.depth_confidence = np.asarray(
            self.depth_confidence, dtype=np.float32
        )

        if self.joints_m.shape != (count, 3):
            raise ValueError(
                f"Expected joints shape {(count, 3)}, got {self.joints_m.shape}."
            )
        for name, values in (
            ("confidence", self.confidence),
            ("valid", self.valid),
            ("depth_m", self.depth_m),
            ("depth_confidence", self.depth_confidence),
        ):
            if values.shape != (count,):
                raise ValueError(
                    f"Expected {name} shape {(count,)}, got {values.shape}."
                )

    def to_dict(self) -> dict[str, Any]:
        joints = []
        for index, name in enumerate(HALPE26_NAMES):
            xyz = (
                self.joints_m[index].tolist()
                if self.valid[index]
                else None
            )
            joints.append(
                {
                    "id": index,
                    "name": name,
                    "xyz_m": xyz,
                    "depth_m": (
                        float(self.depth_m[index])
                        if self.valid[index]
                        else None
                    ),
                    "confidence": float(self.confidence[index]),
                    "depth_confidence": float(
                        self.depth_confidence[index]
                    ),
                    "valid": bool(self.valid[index]),
                }
            )
        return {
            "keypoint_format": "halpe26",
            "coordinate_system": {
                "handedness": "right",
                "x": "right",
                "y": "down",
                "z": "forward",
                "unit": "meter",
            },
            "valid_joint_count": int(np.count_nonzero(self.valid)),
            "joints": joints,
        }
