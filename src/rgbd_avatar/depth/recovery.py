"""Recover metric Halpe26 joints from a 2D pose and aligned depth."""

from __future__ import annotations

import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.pose.models import Pose2D, Pose3D

from .deprojection import deproject_pixel
from .sampling import sample_joint_depth


def recover_pose3d(
    pose2d: Pose2D,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    keypoint_threshold: float = 0.3,
    radius: int = 3,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    expected_depths_m: np.ndarray | None = None,
    max_expected_depth_delta_m: float = 0.45,
) -> Pose3D:
    count = pose2d.keypoints.shape[0]
    if expected_depths_m is None:
        expected_depths = np.full(count, np.nan, dtype=np.float32)
    else:
        expected_depths = np.asarray(expected_depths_m, dtype=np.float32)
        if expected_depths.shape != (count,):
            raise ValueError(
                f"expected_depths_m must have shape {(count,)}, "
                f"got {expected_depths.shape}."
            )
    joints = np.full((count, 3), np.nan, dtype=np.float32)
    confidence = np.zeros(count, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    sampled_depth = np.full(count, np.nan, dtype=np.float32)
    depth_confidence = np.zeros(count, dtype=np.float32)

    for index, ((u, v), pose_score) in enumerate(
        zip(pose2d.keypoints, pose2d.scores, strict=True)
    ):
        if pose_score < keypoint_threshold:
            continue
        sample = sample_joint_depth(
            depth_m=depth_m,
            u=float(u),
            v=float(v),
            radius=radius,
            min_depth_m=min_depth_m,
            max_depth_m=max_depth_m,
            expected_depth_m=(
                float(expected_depths[index])
                if np.isfinite(expected_depths[index])
                else None
            ),
            max_expected_depth_delta_m=max_expected_depth_delta_m,
        )
        if sample is None:
            continue

        joints[index] = deproject_pixel(
            u=float(u),
            v=float(v),
            z_m=sample.depth_m,
            intrinsics=intrinsics,
        )
        sampled_depth[index] = sample.depth_m
        depth_confidence[index] = sample.confidence
        confidence[index] = float(pose_score) * sample.confidence
        valid[index] = True

    return Pose3D(
        joints_m=joints,
        confidence=confidence,
        valid=valid,
        depth_m=sampled_depth,
        depth_confidence=depth_confidence,
    )
