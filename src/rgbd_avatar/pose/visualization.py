"""OpenCV visualization for Halpe26 poses."""

from __future__ import annotations

import cv2
import numpy as np

from .halpe26 import HALPE26_LINKS
from .models import Pose2D


def draw_pose(
    image_bgr: np.ndarray,
    pose: Pose2D,
    score_threshold: float = 0.3,
) -> np.ndarray:
    canvas = image_bgr.copy()

    for start, end in HALPE26_LINKS:
        if (
            pose.scores[start] < score_threshold
            or pose.scores[end] < score_threshold
        ):
            continue
        p1 = tuple(np.rint(pose.keypoints[start]).astype(int))
        p2 = tuple(np.rint(pose.keypoints[end]).astype(int))
        cv2.line(canvas, p1, p2, (0, 220, 255), 2, cv2.LINE_AA)

    for index, point in enumerate(pose.keypoints):
        if pose.scores[index] < score_threshold:
            continue
        center = tuple(np.rint(point).astype(int))
        cv2.circle(canvas, center, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (center[0] + 4, center[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    x1, y1, x2, y2 = np.rint(pose.bbox_xyxy).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (80, 255, 80), 2)
    cv2.putText(
        canvas,
        f"Halpe26 mean={pose.mean_score:.3f}",
        (x1, max(18, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )
    return canvas
