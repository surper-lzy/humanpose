"""Convert validated Hand21 observations into viewer geometry."""

from __future__ import annotations

from typing import Any

import numpy as np

from rgbd_avatar.pose import HAND21_LINKS, hand_observation_quality
from rgbd_avatar.visualization.sequence3d import (
    SkeletonDisplayArrays,
    transform_camera_points,
)


LEFT_HAND_COLOR = np.array([0.10, 0.80, 1.00], dtype=np.float64)
RIGHT_HAND_COLOR = np.array([1.00, 0.78, 0.10], dtype=np.float64)
REJECTED_HAND_COLOR = np.array([1.00, 0.12, 0.12], dtype=np.float64)


def build_hand_skeleton_display_arrays(
    record: dict[str, Any] | None,
    *,
    camera_to_display_transform: np.ndarray | None,
) -> tuple[SkeletonDisplayArrays, tuple[str, ...]]:
    """Build compact Hand21 geometry and retain rejected detections in red."""

    all_points: list[np.ndarray] = []
    point_colors: list[np.ndarray] = []
    all_lines: list[tuple[int, int]] = []
    line_colors: list[np.ndarray] = []
    original_indices: list[int] = []
    rejected: list[str] = []
    hands = record.get("hands") if isinstance(record, dict) else None
    if not isinstance(hands, dict):
        hands = {}
    for side_index, side in enumerate(("left", "right")):
        hand = hands.get(side)
        pose3d = hand.get("pose3d") if isinstance(hand, dict) else None
        joints_payload = pose3d.get("joints") if isinstance(pose3d, dict) else None
        if not isinstance(joints_payload, list) or len(joints_payload) != 21:
            continue
        joints = np.full((21, 3), np.nan, dtype=np.float64)
        valid = np.zeros(21, dtype=bool)
        for index, joint in enumerate(joints_payload):
            xyz = joint.get("xyz_m")
            if bool(joint.get("valid")) and isinstance(xyz, list) and len(xyz) == 3:
                joints[index] = np.asarray(xyz, dtype=np.float64)
                valid[index] = np.isfinite(joints[index]).all()
        quality_ok, reason, _ = hand_observation_quality(joints, valid)
        if not quality_ok:
            rejected.append(f"{side}:{reason}")
        color = (
            LEFT_HAND_COLOR
            if quality_ok and side == "left"
            else RIGHT_HAND_COLOR
            if quality_ok
            else REJECTED_HAND_COLOR
        )
        kept = np.flatnonzero(valid)
        compact = np.full(21, -1, dtype=np.int64)
        compact[kept] = np.arange(len(kept)) + len(all_points)
        if len(kept):
            display_points = transform_camera_points(
                joints[kept], camera_to_display_transform
            )
            all_points.extend(display_points)
            point_colors.extend([color] * len(kept))
            original_indices.extend((side_index * 21 + kept).astype(int).tolist())
        for start, end in HAND21_LINKS:
            if valid[start] and valid[end]:
                all_lines.append((int(compact[start]), int(compact[end])))
                line_colors.append(color)
    return (
        SkeletonDisplayArrays(
            points=np.asarray(all_points, dtype=np.float64).reshape(-1, 3),
            point_colors=np.asarray(point_colors, dtype=np.float64).reshape(-1, 3),
            lines=np.asarray(all_lines, dtype=np.int32).reshape(-1, 2),
            line_colors=np.asarray(line_colors, dtype=np.float64).reshape(-1, 3),
            original_joint_indices=np.asarray(original_indices, dtype=np.int32),
        ),
        tuple(rejected),
    )


__all__ = ["build_hand_skeleton_display_arrays"]
