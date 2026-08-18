"""Single-person live RGB-D pose processing without renderer dependencies."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol

import numpy as np

from rgbd_avatar.depth import (
    PointCloudRecoveryConfig,
    depth_to_organized_point_cloud,
    recover_pose3d,
    recover_pose3d_from_point_cloud,
)
from rgbd_avatar.pose import HALPE26_NAMES, Pose2D, Pose3D
from rgbd_avatar.tracking import (
    BoneLengthCalibrator,
    BoneLengthConstraint,
    FramePresenceDecision,
    PersonFramePresenceGate,
    Pose3DTemporalFilter,
    TemporalPose3D,
)

from .extrinsics import ApplicationExtrinsics
from .models import RGBDFrame


class PoseBackend(Protocol):
    def infer(self, image: np.ndarray) -> list[Pose2D]: ...


@dataclass(frozen=True)
class LivePoseResult:
    """One processed frame ready for a standalone avatar renderer."""

    frame_number: int
    timestamp_ns: int
    source_id: str
    rgb_bgr: np.ndarray
    status: str
    pose2d: Pose2D | None
    pose3d_raw: Pose3D | None
    pose3d_output: TemporalPose3D
    corrected: np.ndarray
    joints_application_m: np.ndarray
    presence: FramePresenceDecision
    timing_ms: dict[str, float]

    def __post_init__(self) -> None:
        count = len(HALPE26_NAMES)
        rgb = np.asarray(self.rgb_bgr, dtype=np.uint8)
        corrected = np.asarray(self.corrected, dtype=bool)
        joints = np.asarray(self.joints_application_m, dtype=np.float32)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb_bgr must have shape HxWx3.")
        if corrected.shape != (count,):
            raise ValueError(f"corrected must have shape {(count,)}.")
        if joints.shape != (count, 3):
            raise ValueError(f"joints_application_m must have shape {(count, 3)}.")
        if not np.isfinite(joints[self.pose3d_output.usable]).all():
            raise ValueError("Every usable application joint must be finite.")
        if int(self.frame_number) < 0 or int(self.timestamp_ns) < 0:
            raise ValueError("Frame number and timestamp must be non-negative.")
        object.__setattr__(self, "frame_number", int(self.frame_number))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "rgb_bgr", rgb)
        object.__setattr__(self, "corrected", corrected)
        object.__setattr__(self, "joints_application_m", joints)


class LivePoseProcessor:
    """Run the existing 2D, metric lifting, temporal, and bone stages."""

    def __init__(
        self,
        *,
        backend: PoseBackend,
        extrinsics: ApplicationExtrinsics,
        temporal_filter: Pose3DTemporalFilter,
        presence_gate: PersonFramePresenceGate,
        keypoint_threshold: float,
        min_depth_m: float,
        max_depth_m: float,
        depth_window_radius: int,
        recovery_method: str = "pointcloud_cluster",
        pointcloud_config: PointCloudRecoveryConfig | None = None,
        bone_calibrator: BoneLengthCalibrator | None = None,
        bone_constraint: BoneLengthConstraint | None = None,
    ) -> None:
        if keypoint_threshold < 0:
            raise ValueError("keypoint_threshold must be non-negative.")
        if min_depth_m <= 0 or max_depth_m <= min_depth_m:
            raise ValueError("Live depth limits are invalid.")
        if depth_window_radius < 0:
            raise ValueError("depth_window_radius must be non-negative.")
        if recovery_method not in ("window_median", "pointcloud_cluster"):
            raise ValueError(
                "recovery_method must be window_median or pointcloud_cluster."
            )
        if (bone_calibrator is None) != (bone_constraint is None):
            raise ValueError(
                "bone_calibrator and bone_constraint must be supplied together."
            )
        self.backend = backend
        self.extrinsics = extrinsics
        self.temporal_filter = temporal_filter
        self.presence_gate = presence_gate
        self.keypoint_threshold = float(keypoint_threshold)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.depth_window_radius = int(depth_window_radius)
        self.recovery_method = recovery_method
        self.pointcloud_config = pointcloud_config or PointCloudRecoveryConfig()
        self.bone_calibrator = bone_calibrator
        self.bone_constraint = bone_constraint
        self._bone_reset_pending = False

    def _recover_pose3d(
        self,
        pose2d: Pose2D,
        frame: RGBDFrame,
    ) -> Pose3D:
        if self.recovery_method == "window_median":
            return recover_pose3d(
                pose2d=pose2d,
                depth_m=frame.depth_m,
                intrinsics=frame.intrinsics,
                keypoint_threshold=self.keypoint_threshold,
                radius=self.depth_window_radius,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            )
        organized_points = depth_to_organized_point_cloud(
            depth_m=frame.depth_m,
            intrinsics=frame.intrinsics,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
        )
        return recover_pose3d_from_point_cloud(
            pose2d=pose2d,
            organized_points_m=organized_points,
            intrinsics=frame.intrinsics,
            keypoint_threshold=self.keypoint_threshold,
            config=self.pointcloud_config,
        ).pose3d

    def process(self, frame: RGBDFrame) -> LivePoseResult:
        started = time.perf_counter()
        inference_started = time.perf_counter()
        poses = self.backend.infer(frame.rgb_bgr)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        pose2d = poses[0] if poses else None
        presence = self.presence_gate.evaluate(
            pose2d,
            image_width=frame.intrinsics.width,
            image_height=frame.intrinsics.height,
            keypoint_threshold=self.keypoint_threshold,
        )
        status = "ok"
        if pose2d is None:
            status = "no_person"
        elif not presence.accepted:
            status = presence.reason
            pose2d = None
        elif self._bone_reset_pending:
            if self.bone_calibrator is not None:
                self.bone_calibrator.reset()
            self._bone_reset_pending = False

        recovery_started = time.perf_counter()
        pose3d: Pose3D | None = None
        if pose2d is not None:
            pose3d = self._recover_pose3d(pose2d, frame)
            if not np.any(pose3d.valid):
                status = "no_valid_3d_joints"
        recovery_ms = (time.perf_counter() - recovery_started) * 1000.0

        timestamp_s = frame.timestamp_ns * 1e-9
        if presence.track_reset_required:
            temporal_pose = self.temporal_filter.terminate_track(timestamp_s)
            self._bone_reset_pending = True
        else:
            temporal_pose = self.temporal_filter.update(timestamp_s, pose3d)

        constraint_started = time.perf_counter()
        corrected = np.zeros(len(HALPE26_NAMES), dtype=bool)
        output_pose = temporal_pose
        if self.bone_calibrator is not None and self.bone_constraint is not None:
            if pose3d is not None and pose2d is not None:
                self.bone_calibrator.update(pose3d, pose2d.scores)
            constrained = self.bone_constraint.apply(
                temporal_pose,
                self.bone_calibrator.prior(),
            )
            output_pose = constrained.pose
            corrected = constrained.corrected
        constraint_ms = (time.perf_counter() - constraint_started) * 1000.0

        joints_application = self.extrinsics.transform_points(
            output_pose.joints_m
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        return LivePoseResult(
            frame_number=frame.frame_number,
            timestamp_ns=frame.timestamp_ns,
            source_id=frame.source_id,
            rgb_bgr=frame.rgb_bgr,
            status=status,
            pose2d=pose2d,
            pose3d_raw=pose3d,
            pose3d_output=output_pose,
            corrected=corrected,
            joints_application_m=joints_application,
            presence=presence,
            timing_ms={
                "inference": inference_ms,
                "recovery": recovery_ms,
                "bone_constraint": constraint_ms,
                "total": total_ms,
            },
        )
