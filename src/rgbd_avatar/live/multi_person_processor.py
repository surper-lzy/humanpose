"""Multi-person RGB-D pose processing and selectable identity tracking."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import logging
import math
import time

import numpy as np
from scipy.optimize import linear_sum_assignment

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import (
    PointCloudRecoveryConfig,
    depth_to_organized_point_cloud,
    recover_pose3d,
    recover_pose3d_from_depth_connected,
    recover_pose3d_from_point_cloud,
)
from rgbd_avatar.pose import HALPE26_NAMES, Pose2D, Pose3D
from rgbd_avatar.tracking import (
    BoneLengthCalibrator,
    BoneLengthConstraint,
    BoneLengthPrior,
    FramePresenceDecision,
    PersonFramePresenceGate,
    Pose3DTemporalFilter,
    TemporalPose3D,
)
from rgbd_avatar.tracking.shadow_identity import (
    ShadowIdentityConfig,
    ShadowIdentityObservation,
    ShadowRGBDIdentityTracker,
    extract_upper_body_hsv_descriptor,
)

from .extrinsics import ApplicationExtrinsics
from .models import RGBDFrame
from .processor import PoseBackend


TemporalFilterFactory = Callable[[], Pose3DTemporalFilter]
PresenceGateFactory = Callable[[], PersonFramePresenceGate]
BoneComponentsFactory = Callable[
    [], tuple[BoneLengthCalibrator | None, BoneLengthConstraint | None]
]

_TORSO_INDICES = np.asarray((18, 19, 5, 6, 11, 12), dtype=np.int64)
# Head/face and both arms are the joints most likely to select background or
# another person's depth. The local multi-person hybrid mode spends the robust
# point-cloud search on this subset and uses the fast median path elsewhere.
_HYBRID_POINTCLOUD_INDICES = frozenset((*range(0, 11), 17, 18))
_ADAPTIVE_HYBRID_GROUPS: tuple[frozenset[int], ...] = (
    # Recover the complete face anchor group together so the point-cloud
    # face/ear topology gates remain available.
    frozenset((0, 1, 2, 3, 4, 17, 18)),
    # Arm groups retain their shoulder and neck anchors.  Recovering only an
    # isolated wrist would save work but would remove useful body context.
    frozenset((5, 7, 9, 18)),
    frozenset((6, 8, 10, 18)),
)
_GUIDED_MAX_DEPTH_DELTA_M = 0.45
_DEPTH_GUIDANCE_REACQUIRE_FRAMES = 3
_UNMATCHABLE_COST = 1_000_000.0
LOGGER = logging.getLogger("multi_person_processor")

# A bad parent depth invalidates its dependent limb. Keeping descendants as
# isolated usable points would still draw floating joints in the front end.
_QUALITY_JOINT_BRANCHES: dict[int, tuple[int, ...]] = {
    18: (18, 17, 0, 1, 2, 3, 4, 5, 7, 9, 6, 8, 10),
    17: (17, 0, 1, 2, 3, 4),
    5: (5, 7, 9),
    7: (7, 9),
    6: (6, 8, 10),
    8: (8, 10),
    11: (11, 13, 15, 20, 22, 24),
    13: (13, 15, 20, 22, 24),
    15: (15, 20, 22, 24),
    12: (12, 14, 16, 21, 23, 25),
    14: (14, 16, 21, 23, 25),
    16: (16, 21, 23, 25),
}

# A root-outward tree covering all Halpe26 joints. The calibrated constraint
# links cover the metric body; the extra links let the live renderer retain
# face and foot leaves when their depth samples are unavailable.
_SKELETON_COMPLETION_LINKS: tuple[tuple[int, int], ...] = (
    (19, 11),
    (11, 13),
    (13, 15),
    (15, 20),
    (15, 22),
    (15, 24),
    (19, 12),
    (12, 14),
    (14, 16),
    (16, 21),
    (16, 23),
    (16, 25),
    (19, 18),
    (18, 17),
    (17, 0),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (18, 5),
    (5, 7),
    (7, 9),
    (18, 6),
    (6, 8),
    (8, 10),
)

# Absolute last-resort guards for links that are not part of the configurable
# core-body quality checks.
_EXTRA_COMPLETION_LINK_MAX_M: dict[tuple[int, int], float] = {
    (15, 20): 0.30,
    (15, 22): 0.30,
    (15, 24): 0.22,
    (16, 21): 0.30,
    (16, 23): 0.30,
    (16, 25): 0.22,
    (17, 0): 0.25,
    (0, 1): 0.18,
    (0, 2): 0.18,
    (1, 3): 0.20,
    (2, 4): 0.20,
}


@dataclass(frozen=True)
class LocalMultiPersonConfig:
    """Association and lifecycle thresholds for the local experiment."""

    max_persons: int = 2
    max_missing_s: float = 0.35
    minimum_bbox_iou: float = 0.01
    max_center_distance_ratio: float = 1.25
    max_keypoint_distance_ratio: float = 0.60
    max_root_distance_m: float = 0.80
    max_match_cost: float = 1.40

    def __post_init__(self) -> None:
        if self.max_persons <= 0:
            raise ValueError("max_persons must be positive.")
        positive = (
            self.max_missing_s,
            self.max_center_distance_ratio,
            self.max_keypoint_distance_ratio,
            self.max_root_distance_m,
            self.max_match_cost,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Multi-person metric thresholds must be positive.")
        if not 0 <= self.minimum_bbox_iou <= 1:
            raise ValueError("minimum_bbox_iou must be in [0, 1].")


@dataclass(frozen=True)
class AdaptiveHybridConfig:
    """Quality gates that decide which hybrid groups need robust recovery."""

    min_depth_confidence: float = 0.55
    max_torso_depth_delta_m: float = 0.45
    head_neck_max_length_m: float = 0.40
    upper_arm_max_length_m: float = 0.50
    forearm_max_length_m: float = 0.45

    def __post_init__(self) -> None:
        if not 0 <= self.min_depth_confidence <= 1:
            raise ValueError("min_depth_confidence must be in [0, 1].")
        positive = (
            self.max_torso_depth_delta_m,
            self.head_neck_max_length_m,
            self.upper_arm_max_length_m,
            self.forearm_max_length_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Adaptive hybrid metric thresholds must be positive.")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object] | None,
    ) -> AdaptiveHybridConfig:
        return cls(**dict(mapping or {}))


@dataclass(frozen=True)
class Pose3DQualityConfig:
    """Fail-closed gates for plausible person-level metric geometry."""

    enabled: bool = True
    min_valid_joint_count: int = 8
    min_valid_torso_joint_count: int = 3
    max_torso_depth_span_m: float = 0.55
    depth_jump_base_m: float = 0.20
    max_depth_speed_m_s: float = 2.50
    max_depth_jump_m: float = 0.65
    reject_depth_jump_joint_count: int = 5
    reject_bone_violation_count: int = 2
    prior_length_ratio: float = 1.65
    max_spine_projection_ratio: float = 1.35
    spine_projection_slack_m: float = 0.10
    hip_offset_max_m: float = 0.30
    thigh_max_m: float = 0.75
    shin_max_m: float = 0.75
    spine_max_m: float = 0.85
    head_neck_max_m: float = 0.40
    shoulder_offset_max_m: float = 0.38
    upper_arm_max_m: float = 0.55
    forearm_max_m: float = 0.50

    def __post_init__(self) -> None:
        positive_integers = (
            self.min_valid_joint_count,
            self.min_valid_torso_joint_count,
            self.reject_depth_jump_joint_count,
            self.reject_bone_violation_count,
        )
        if any(value <= 0 for value in positive_integers):
            raise ValueError("Pose3D quality counts must be positive.")
        if self.min_valid_joint_count > len(HALPE26_NAMES):
            raise ValueError("min_valid_joint_count exceeds Halpe26 size.")
        if self.min_valid_torso_joint_count > len(_TORSO_INDICES):
            raise ValueError(
                "min_valid_torso_joint_count exceeds torso joint count."
            )
        positive = (
            self.max_torso_depth_span_m,
            self.depth_jump_base_m,
            self.max_depth_speed_m_s,
            self.max_depth_jump_m,
            self.hip_offset_max_m,
            self.thigh_max_m,
            self.shin_max_m,
            self.spine_max_m,
            self.head_neck_max_m,
            self.shoulder_offset_max_m,
            self.upper_arm_max_m,
            self.forearm_max_m,
            self.max_spine_projection_ratio,
            self.spine_projection_slack_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Pose3D quality metric thresholds must be positive.")
        if (
            not math.isfinite(self.prior_length_ratio)
            or self.prior_length_ratio <= 1
        ):
            raise ValueError("prior_length_ratio must exceed 1.")
        if self.max_spine_projection_ratio < 1:
            raise ValueError("max_spine_projection_ratio must be at least 1.")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object] | None,
    ) -> Pose3DQualityConfig:
        return cls(**dict(mapping or {}))


@dataclass(frozen=True)
class KinematicFallbackConfig:
    """Short, bone-safe recovery for incomplete established tracks."""

    enabled: bool = True
    max_age_s: float = 0.25
    min_keypoint_confidence: float = 0.30
    confidence_scale: float = 0.35
    max_joint_speed_m_s: float = 3.0
    displacement_slack_m: float = 0.05
    complete_skeleton: bool = True
    reconstruct_from_current_2d: bool = True
    min_core_2d_joint_count: int = 2
    min_core_3d_joint_count: int = 1
    min_history_joint_count: int = 8
    max_torso_rotation_deg: float = 45.0

    def __post_init__(self) -> None:
        positive = (
            self.max_age_s,
            self.max_joint_speed_m_s,
            self.displacement_slack_m,
            self.max_torso_rotation_deg,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("Kinematic fallback metric limits must be positive.")
        if not 0 <= self.min_keypoint_confidence <= 1:
            raise ValueError("min_keypoint_confidence must be in [0, 1].")
        if not 0 < self.confidence_scale <= 1:
            raise ValueError("confidence_scale must be in (0, 1].")
        if not 1 <= self.min_core_2d_joint_count <= len(_TORSO_INDICES):
            raise ValueError(
                "min_core_2d_joint_count must fit the torso joint set."
            )
        if not 1 <= self.min_core_3d_joint_count <= len(_TORSO_INDICES):
            raise ValueError(
                "min_core_3d_joint_count must fit the torso joint set."
            )
        if not 1 <= self.min_history_joint_count <= len(HALPE26_NAMES):
            raise ValueError(
                "min_history_joint_count must fit the Halpe26 joint set."
            )
        if self.max_torso_rotation_deg > 180:
            raise ValueError("max_torso_rotation_deg must not exceed 180.")

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object] | None,
    ) -> KinematicFallbackConfig:
        return cls(**dict(mapping or {}))


@dataclass(frozen=True)
class LocalPersonPoseResult:
    """One locally tracked person."""

    track_id: int
    status: str
    observed_in_frame: bool
    pose2d: Pose2D | None
    pose3d_raw: Pose3D | None
    pose3d_output: TemporalPose3D
    corrected: np.ndarray
    joints_application_m: np.ndarray
    presence: FramePresenceDecision
    match_cost: float | None
    kinematic_fallback: np.ndarray
    joint_sources: tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(HALPE26_NAMES)
        corrected = np.asarray(self.corrected, dtype=bool)
        kinematic_fallback = np.asarray(self.kinematic_fallback, dtype=bool)
        joints = np.asarray(self.joints_application_m, dtype=np.float32)
        if self.track_id <= 0:
            raise ValueError("track_id must be positive.")
        if corrected.shape != (count,):
            raise ValueError(f"corrected must have shape {(count,)}.")
        if kinematic_fallback.shape != (count,):
            raise ValueError(f"kinematic_fallback must have shape {(count,)}.")
        if len(self.joint_sources) != count:
            raise ValueError(f"joint_sources must contain {count} values.")
        if joints.shape != (count, 3):
            raise ValueError(f"joints_application_m must have shape {(count, 3)}.")
        if not np.isfinite(joints[self.pose3d_output.usable]).all():
            raise ValueError("Every usable application joint must be finite.")
        if self.match_cost is not None and not np.isfinite(self.match_cost):
            raise ValueError("match_cost must be finite or None.")
        object.__setattr__(self, "corrected", corrected)
        object.__setattr__(self, "kinematic_fallback", kinematic_fallback)
        object.__setattr__(self, "joint_sources", tuple(self.joint_sources))
        object.__setattr__(self, "joints_application_m", joints)


@dataclass(frozen=True)
class LocalMultiPersonPoseResult:
    """One local frame containing zero or more independently tracked people."""

    frame_number: int
    timestamp_ns: int
    source_id: str
    rgb_bgr: np.ndarray
    status: str
    detected_person_count: int
    persons: tuple[LocalPersonPoseResult, ...]
    timing_ms: dict[str, float]
    identity_method: str = "geometry"
    identity_fallback: bool = False
    recovery_stats: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rgb = np.asarray(self.rgb_bgr, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rgb_bgr must have shape HxWx3.")
        if self.frame_number < 0 or self.timestamp_ns < 0:
            raise ValueError("Frame number and timestamp must be non-negative.")
        if self.detected_person_count < 0:
            raise ValueError("detected_person_count must be non-negative.")
        if self.identity_method not in ("geometry", "shadow"):
            raise ValueError("identity_method must be geometry or shadow.")
        track_ids = [person.track_id for person in self.persons]
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("A local multi-person frame cannot repeat track_id.")
        object.__setattr__(self, "rgb_bgr", rgb)
        object.__setattr__(self, "persons", tuple(self.persons))
        object.__setattr__(
            self,
            "recovery_stats",
            {str(key): int(value) for key, value in self.recovery_stats.items()},
        )


@dataclass(frozen=True)
class _DetectionObservation:
    pose2d: Pose2D
    pose3d: Pose3D
    root_camera_m: np.ndarray | None
    joint_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.joint_reasons and len(self.joint_reasons) != len(HALPE26_NAMES):
            raise ValueError("joint_reasons must be empty or contain 26 values.")


@dataclass
class _TrackState:
    track_id: int
    temporal_filter: Pose3DTemporalFilter
    presence_gate: PersonFramePresenceGate
    bone_calibrator: BoneLengthCalibrator | None
    bone_constraint: BoneLengthConstraint | None
    last_pose2d: Pose2D
    last_root_camera_m: np.ndarray | None
    last_detection_s: float
    missing_since_s: float | None = None
    bone_reset_pending: bool = False
    depth_guidance_joints_m: np.ndarray | None = None
    depth_guidance_failure_counts: np.ndarray = field(
        default_factory=lambda: np.zeros(len(HALPE26_NAMES), dtype=np.int32)
    )
    depth_guidance_observed_s: np.ndarray = field(
        default_factory=lambda: np.full(len(HALPE26_NAMES), np.nan)
    )


def _bbox_iou(first: np.ndarray, second: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = np.asarray(first, dtype=np.float64)
    bx1, by1, bx2, by2 = np.asarray(second, dtype=np.float64)
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = width * height
    first_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    second_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _bbox_center_and_diagonal(bbox: np.ndarray) -> tuple[np.ndarray, float]:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float64)
    size = np.maximum(np.array([x2 - x1, y2 - y1]), 1.0)
    return np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5]), float(
        np.linalg.norm(size)
    )


def _keypoint_distance_ratio(
    first: Pose2D,
    second: Pose2D,
    threshold: float,
) -> float | None:
    common = (
        (first.scores >= threshold)
        & (second.scores >= threshold)
        & np.isfinite(first.keypoints).all(axis=1)
        & np.isfinite(second.keypoints).all(axis=1)
    )
    if np.count_nonzero(common) < 4:
        return None
    _, first_diagonal = _bbox_center_and_diagonal(first.bbox_xyxy)
    _, second_diagonal = _bbox_center_and_diagonal(second.bbox_xyxy)
    scale = max(1.0, 0.5 * (first_diagonal + second_diagonal))
    distances = np.linalg.norm(
        first.keypoints[common] - second.keypoints[common], axis=1
    )
    return float(np.median(distances) / scale)


def _root_camera_m(pose3d: Pose3D) -> np.ndarray | None:
    if pose3d.valid[19] and np.isfinite(pose3d.joints_m[19]).all():
        return pose3d.joints_m[19].astype(np.float64, copy=True)
    valid_torso = _TORSO_INDICES[
        pose3d.valid[_TORSO_INDICES]
        & np.isfinite(pose3d.joints_m[_TORSO_INDICES]).all(axis=1)
    ]
    if not len(valid_torso):
        return None
    return np.median(pose3d.joints_m[valid_torso], axis=0).astype(np.float64)


def _has_valid_3d_observation(observation: _DetectionObservation) -> bool:
    pose3d = observation.pose3d
    return bool(
        np.any(pose3d.valid & np.isfinite(pose3d.joints_m).all(axis=1))
    )


def _empty_pose3d() -> Pose3D:
    count = len(HALPE26_NAMES)
    return Pose3D(
        joints_m=np.full((count, 3), np.nan, dtype=np.float32),
        confidence=np.zeros(count, dtype=np.float32),
        valid=np.zeros(count, dtype=bool),
        depth_m=np.full(count, np.nan, dtype=np.float32),
        depth_confidence=np.zeros(count, dtype=np.float32),
    )


def _hide_whole_person_prediction(pose: TemporalPose3D) -> TemporalPose3D:
    """Keep filter state internal while making a missing person non-renderable."""

    count = len(HALPE26_NAMES)
    return TemporalPose3D(
        joints_m=pose.joints_m.copy(),
        confidence=np.zeros(count, dtype=np.float32),
        usable=np.zeros(count, dtype=bool),
        observed=np.zeros(count, dtype=bool),
        predicted=np.zeros(count, dtype=bool),
        age_s=pose.age_s.copy(),
        reset_occurred=pose.reset_occurred,
    )


def _pose3d_with_invalidated_joints(
    pose: Pose3D,
    joint_indices: set[int] | np.ndarray,
) -> Pose3D:
    invalid = np.asarray(sorted(joint_indices), dtype=np.int64)
    if not invalid.size:
        return pose
    joints = pose.joints_m.copy()
    confidence = pose.confidence.copy()
    valid = pose.valid.copy()
    depth_m = pose.depth_m.copy()
    depth_confidence = pose.depth_confidence.copy()
    joints[invalid] = np.nan
    confidence[invalid] = 0.0
    valid[invalid] = False
    depth_m[invalid] = np.nan
    depth_confidence[invalid] = 0.0
    return Pose3D(
        joints_m=joints,
        confidence=confidence,
        valid=valid,
        depth_m=depth_m,
        depth_confidence=depth_confidence,
    )


def _camera_ray(
    pixel: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray | None:
    u, v = np.asarray(pixel, dtype=np.float64)
    if (
        not np.isfinite((u, v)).all()
        or u < 0
        or u >= intrinsics.width
        or v < 0
        or v >= intrinsics.height
    ):
        return None
    ray = np.asarray(
        (
            (u - intrinsics.cx) / intrinsics.fx,
            (v - intrinsics.cy) / intrinsics.fy,
            1.0,
        ),
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(ray))
    return ray / norm if np.isfinite(norm) and norm > 1e-9 else None


def _ray_bone_candidate(
    *,
    parent_m: np.ndarray,
    previous_parent_m: np.ndarray,
    previous_child_m: np.ndarray,
    pixel: np.ndarray,
    bone_length_m: float,
    intrinsics: CameraIntrinsics,
    min_depth_m: float,
    max_depth_m: float,
    max_displacement_m: float,
) -> np.ndarray | None:
    """Place a child on its current camera ray while preserving bone length."""

    parent = np.asarray(parent_m, dtype=np.float64)
    previous_parent = np.asarray(previous_parent_m, dtype=np.float64)
    previous_child = np.asarray(previous_child_m, dtype=np.float64)
    ray = _camera_ray(pixel, intrinsics)
    if (
        ray is None
        or not np.isfinite(parent).all()
        or not np.isfinite(previous_parent).all()
        or not np.isfinite(previous_child).all()
        or not np.isfinite(bone_length_m)
        or bone_length_m <= 0
    ):
        return None

    ray_candidates: list[np.ndarray] = []
    projection = float(np.dot(parent, ray))
    discriminant = projection**2 - (
        float(np.dot(parent, parent)) - bone_length_m**2
    )
    if discriminant >= 0:
        root = math.sqrt(discriminant)
        for distance in (projection - root, projection + root):
            if distance <= 0:
                continue
            candidate = distance * ray
            if min_depth_m <= candidate[2] <= max_depth_m:
                ray_candidates.append(candidate)

    if ray_candidates:
        ray_candidates.sort(
            key=lambda value: float(np.linalg.norm(value - previous_child))
        )
        candidate = ray_candidates[0]
        if (
            float(np.linalg.norm(candidate - previous_child))
            <= max_displacement_m
        ):
            return candidate.astype(np.float32)

    # When noisy 2D and filtered parent coordinates make the ray miss the
    # exact sphere, preserve the last bone direction and translate it with the
    # current parent. This is stable and cannot stretch the segment.
    previous_delta = previous_child - previous_parent
    previous_norm = float(np.linalg.norm(previous_delta))
    if np.isfinite(previous_norm) and previous_norm > 1e-9:
        propagated = parent + bone_length_m * previous_delta / previous_norm
        if (
            min_depth_m <= propagated[2] <= max_depth_m
            and float(np.linalg.norm(propagated - previous_child))
            <= max_displacement_m
        ):
            return propagated.astype(np.float32)
    return None


def _rigid_transform_from_anchors(
    source_m: np.ndarray,
    target_m: np.ndarray,
    *,
    max_rotation_deg: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Estimate a conservative previous-to-current torso transform."""

    source = np.asarray(source_m, dtype=np.float64)
    target = np.asarray(target_m, dtype=np.float64)
    if (
        source.ndim != 2
        or source.shape != target.shape
        or source.shape[1:] != (3,)
        or not len(source)
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        return None

    rotation = np.eye(3, dtype=np.float64)
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    if (
        len(source) >= 3
        and np.linalg.matrix_rank(source_centered) >= 2
        and np.linalg.matrix_rank(target_centered) >= 2
    ):
        covariance = source_centered.T @ target_centered
        left, _singular, right_t = np.linalg.svd(covariance)
        candidate_rotation = right_t.T @ left.T
        if np.linalg.det(candidate_rotation) < 0:
            right_t[-1] *= -1
            candidate_rotation = right_t.T @ left.T
        cosine = float(
            np.clip((np.trace(candidate_rotation) - 1.0) * 0.5, -1.0, 1.0)
        )
        angle_deg = math.degrees(math.acos(cosine))
        if np.isfinite(angle_deg) and angle_deg <= max_rotation_deg:
            rotation = candidate_rotation

    translation = target_center - rotation @ source_center
    return rotation, translation


def _propagated_bone_candidate(
    *,
    parent_m: np.ndarray,
    previous_parent_m: np.ndarray,
    previous_child_m: np.ndarray,
    rotation: np.ndarray,
    bone_length_m: float,
    min_depth_m: float,
    max_depth_m: float,
    max_displacement_m: float,
) -> np.ndarray | None:
    """Move a previous bone with the torso while preserving its length."""

    parent = np.asarray(parent_m, dtype=np.float64)
    previous_parent = np.asarray(previous_parent_m, dtype=np.float64)
    previous_child = np.asarray(previous_child_m, dtype=np.float64)
    previous_delta = previous_child - previous_parent
    previous_norm = float(np.linalg.norm(previous_delta))
    if (
        not np.isfinite(parent).all()
        or not np.isfinite(previous_delta).all()
        or not np.isfinite(previous_norm)
        or previous_norm <= 1e-9
        or not np.isfinite(bone_length_m)
        or bone_length_m <= 0
    ):
        return None
    direction = np.asarray(rotation, dtype=np.float64) @ (
        previous_delta / previous_norm
    )
    candidate = parent + bone_length_m * direction
    if (
        min_depth_m <= candidate[2] <= max_depth_m
        and float(np.linalg.norm(candidate - previous_child))
        <= max_displacement_m
    ):
        return candidate.astype(np.float32)
    return None


def _current_2d_bone_candidate(
    *,
    parent_m: np.ndarray,
    child_pixel: np.ndarray,
    intrinsics: CameraIntrinsics,
    preferred_length_m: float | None,
    max_length_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray | None:
    """Build a bounded bone in the parent's depth plane from current 2D."""

    parent = np.asarray(parent_m, dtype=np.float64)
    pixel = np.asarray(child_pixel, dtype=np.float64)
    if (
        not np.isfinite(parent).all()
        or not np.isfinite(pixel).all()
        or not np.isfinite(max_length_m)
        or max_length_m <= 0
        or not min_depth_m <= parent[2] <= max_depth_m
        or pixel[0] < 0
        or pixel[0] >= intrinsics.width
        or pixel[1] < 0
        or pixel[1] >= intrinsics.height
    ):
        return None

    planar = np.asarray(
        (
            (pixel[0] - intrinsics.cx) * parent[2] / intrinsics.fx,
            (pixel[1] - intrinsics.cy) * parent[2] / intrinsics.fy,
            parent[2],
        ),
        dtype=np.float64,
    )
    direction = planar - parent
    projected_length_m = float(np.linalg.norm(direction))
    if not np.isfinite(projected_length_m) or projected_length_m <= 1e-6:
        return None

    requested_length_m = (
        float(preferred_length_m)
        if preferred_length_m is not None
        and np.isfinite(preferred_length_m)
        and preferred_length_m > 0
        else projected_length_m
    )
    resolved_length_m = min(requested_length_m, max_length_m)
    candidate = parent + resolved_length_m * direction / projected_length_m
    if not min_depth_m <= candidate[2] <= max_depth_m:
        return None
    return candidate.astype(np.float32)


def _projected_link_length_at_parent_depth(
    *,
    parent_m: np.ndarray,
    child_pixel: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> float | None:
    """Return the metric 2D link length in the parent's depth plane."""

    parent = np.asarray(parent_m, dtype=np.float64)
    pixel = np.asarray(child_pixel, dtype=np.float64)
    if (
        not np.isfinite(parent).all()
        or not np.isfinite(pixel).all()
        or parent[2] <= 0
        or pixel[0] < 0
        or pixel[0] >= intrinsics.width
        or pixel[1] < 0
        or pixel[1] >= intrinsics.height
    ):
        return None
    planar = np.asarray(
        (
            (pixel[0] - intrinsics.cx) * parent[2] / intrinsics.fx,
            (pixel[1] - intrinsics.cy) * parent[2] / intrinsics.fy,
            parent[2],
        ),
        dtype=np.float64,
    )
    length_m = float(np.linalg.norm(planar - parent))
    return length_m if np.isfinite(length_m) and length_m > 1e-4 else None


def _body_axis_head_candidate(
    *,
    neck_m: np.ndarray,
    hip_m: np.ndarray,
    preferred_length_m: float | None,
    max_length_m: float,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray | None:
    """Infer the top-of-head point for a back view without face landmarks."""

    neck = np.asarray(neck_m, dtype=np.float64)
    hip = np.asarray(hip_m, dtype=np.float64)
    axis = neck - hip
    torso_length_m = float(np.linalg.norm(axis))
    if (
        not np.isfinite(neck).all()
        or not np.isfinite(hip).all()
        or not np.isfinite(torso_length_m)
        or torso_length_m <= 1e-6
        or not np.isfinite(max_length_m)
        or max_length_m <= 0
    ):
        return None
    requested_length_m = (
        float(preferred_length_m)
        if preferred_length_m is not None
        and np.isfinite(preferred_length_m)
        and preferred_length_m > 0
        else float(np.clip(0.45 * torso_length_m, 0.18, 0.30))
    )
    candidate = neck + min(requested_length_m, max_length_m) * (
        axis / torso_length_m
    )
    if not min_depth_m <= candidate[2] <= max_depth_m:
        return None
    return candidate.astype(np.float32)


def _merge_hybrid_pose3d(
    fast: Pose3D,
    robust: Pose3D,
    joint_indices: frozenset[int] = _HYBRID_POINTCLOUD_INDICES,
) -> Pose3D:
    """Replace high-risk joints with fail-closed point-cloud estimates."""
    indices = np.fromiter(
        sorted(joint_indices),
        dtype=np.int64,
    )
    joints = fast.joints_m.copy()
    confidence = fast.confidence.copy()
    valid = fast.valid.copy()
    depth_m = fast.depth_m.copy()
    depth_confidence = fast.depth_confidence.copy()
    joints[indices] = robust.joints_m[indices]
    confidence[indices] = robust.confidence[indices]
    valid[indices] = robust.valid[indices]
    depth_m[indices] = robust.depth_m[indices]
    depth_confidence[indices] = robust.depth_confidence[indices]
    return Pose3D(
        joints_m=joints,
        confidence=confidence,
        valid=valid,
        depth_m=depth_m,
        depth_confidence=depth_confidence,
    )


def _torso_depth_hint_m(pose3d: Pose3D) -> float | None:
    valid = _TORSO_INDICES[
        pose3d.valid[_TORSO_INDICES]
        & np.isfinite(pose3d.joints_m[_TORSO_INDICES]).all(axis=1)
    ]
    if len(valid) < 2:
        return None
    return float(np.median(pose3d.joints_m[valid, 2]))


def _link_exceeds(
    pose3d: Pose3D,
    first: int,
    second: int,
    max_length_m: float,
) -> bool:
    if not pose3d.valid[first] or not pose3d.valid[second]:
        return False
    return bool(
        np.linalg.norm(pose3d.joints_m[first] - pose3d.joints_m[second])
        > max_length_m
    )


def _adaptive_hybrid_joint_indices(
    pose2d: Pose2D,
    fast_pose3d: Pose3D,
    *,
    keypoint_threshold: float,
    config: AdaptiveHybridConfig,
    pointcloud_config: PointCloudRecoveryConfig,
) -> frozenset[int]:
    """Select complete high-risk groups whose fast depth is suspicious."""

    eligible = pose2d.scores >= keypoint_threshold
    suspicious = np.zeros(len(HALPE26_NAMES), dtype=bool)
    for index in _HYBRID_POINTCLOUD_INDICES:
        if not eligible[index]:
            continue
        suspicious[index] = (
            not fast_pose3d.valid[index]
            or fast_pose3d.depth_confidence[index]
            < config.min_depth_confidence
        )

    torso_depth_m = _torso_depth_hint_m(fast_pose3d)
    if torso_depth_m is not None:
        high_risk = np.fromiter(
            sorted(_HYBRID_POINTCLOUD_INDICES),
            dtype=np.int64,
        )
        comparable = (
            eligible[high_risk]
            & fast_pose3d.valid[high_risk]
            & np.isfinite(fast_pose3d.depth_m[high_risk])
        )
        residuals = np.abs(fast_pose3d.depth_m[high_risk] - torso_depth_m)
        suspicious[high_risk] |= (
            comparable & (residuals > config.max_torso_depth_delta_m)
        )

    metric_links = (
        (18, 17, config.head_neck_max_length_m),
        (18, 5, pointcloud_config.shoulder_neck_max_length_m),
        (18, 6, pointcloud_config.shoulder_neck_max_length_m),
        (5, 7, config.upper_arm_max_length_m),
        (6, 8, config.upper_arm_max_length_m),
        (7, 9, config.forearm_max_length_m),
        (8, 10, config.forearm_max_length_m),
    )
    for first, second, max_length_m in metric_links:
        if _link_exceeds(fast_pose3d, first, second, max_length_m):
            suspicious[first] = bool(eligible[first])
            suspicious[second] = bool(eligible[second])

    selected: set[int] = set()
    for group in _ADAPTIVE_HYBRID_GROUPS:
        group_indices = np.fromiter(sorted(group), dtype=np.int64)
        if np.any(suspicious[group_indices]):
            selected.update(
                index for index in group if bool(eligible[index])
            )
    return frozenset(selected)


class LocalMultiPersonPoseProcessor:
    """Detect and track several people with a selectable identity backend."""

    def __init__(
        self,
        *,
        backend: PoseBackend,
        extrinsics: ApplicationExtrinsics,
        temporal_filter_factory: TemporalFilterFactory,
        presence_gate_factory: PresenceGateFactory,
        bone_components_factory: BoneComponentsFactory,
        keypoint_threshold: float,
        min_depth_m: float,
        max_depth_m: float,
        depth_window_radius: int,
        recovery_method: str = "hybrid",
        pointcloud_config: PointCloudRecoveryConfig | None = None,
        adaptive_hybrid_config: AdaptiveHybridConfig | None = None,
        pose3d_quality_config: Pose3DQualityConfig | None = None,
        kinematic_fallback_config: KinematicFallbackConfig | None = None,
        multi_person_config: LocalMultiPersonConfig | None = None,
        identity_tracker: str = "geometry",
        shadow_identity_config: ShadowIdentityConfig | None = None,
        depth_connected_refresh_interval: int = 1,
    ) -> None:
        if not 0 <= keypoint_threshold <= 1:
            raise ValueError("keypoint_threshold must be in [0, 1].")
        if min_depth_m <= 0 or max_depth_m <= min_depth_m:
            raise ValueError("Live depth limits are invalid.")
        if depth_window_radius < 0:
            raise ValueError("depth_window_radius must be non-negative.")
        if recovery_method not in (
            "window_median",
            "guided_window",
            "depth_connected",
            "pointcloud_cluster",
            "hybrid",
            "adaptive_hybrid",
        ):
            raise ValueError(
                "recovery_method must be window_median, guided_window, "
                "depth_connected, pointcloud_cluster, hybrid, or "
                "adaptive_hybrid."
            )
        if identity_tracker not in ("geometry", "shadow"):
            raise ValueError("identity_tracker must be geometry or shadow.")
        if depth_connected_refresh_interval <= 0:
            raise ValueError(
                "depth_connected_refresh_interval must be positive."
            )
        self.backend = backend
        self.extrinsics = extrinsics
        self.temporal_filter_factory = temporal_filter_factory
        self.presence_gate_factory = presence_gate_factory
        self.bone_components_factory = bone_components_factory
        self.keypoint_threshold = float(keypoint_threshold)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.depth_window_radius = int(depth_window_radius)
        self.recovery_method = recovery_method
        self.pointcloud_config = pointcloud_config or PointCloudRecoveryConfig()
        self.adaptive_hybrid_config = (
            adaptive_hybrid_config or AdaptiveHybridConfig()
        )
        self.pose3d_quality_config = (
            pose3d_quality_config or Pose3DQualityConfig()
        )
        self.kinematic_fallback_config = (
            kinematic_fallback_config or KinematicFallbackConfig()
        )
        self.config = multi_person_config or LocalMultiPersonConfig()
        self.identity_tracker = identity_tracker
        self.depth_connected_refresh_interval = int(
            depth_connected_refresh_interval
        )
        self._shadow_tracker = (
            ShadowRGBDIdentityTracker(shadow_identity_config)
            if identity_tracker == "shadow"
            else None
        )
        self._shadow_failed = False
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1

    @property
    def active_track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._tracks))

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._shadow_failed = False
        if self._shadow_tracker is not None:
            self._shadow_tracker.reset()

    def _make_observation(
        self,
        pose2d: Pose2D,
        pose3d: Pose3D,
    ) -> _DetectionObservation:
        finite_valid = pose3d.valid & np.isfinite(pose3d.joints_m).all(axis=1)
        reasons: list[str] = []
        for index in range(len(HALPE26_NAMES)):
            if finite_valid[index]:
                reasons.append("observed")
            elif (
                pose2d.scores[index] < self.keypoint_threshold
                or not np.isfinite(pose2d.keypoints[index]).all()
            ):
                reasons.append("low_2d_confidence")
            else:
                reasons.append("no_depth_candidate")
        return _DetectionObservation(
            pose2d=pose2d,
            pose3d=pose3d,
            root_camera_m=_root_camera_m(pose3d),
            joint_reasons=tuple(reasons),
        )

    def _can_complete_sparse_track(
        self,
        track: _TrackState | None,
        observation: _DetectionObservation,
        timestamp_s: float,
    ) -> bool:
        """Allow sparse geometry only after this track has reliable history."""

        config = self.kinematic_fallback_config
        if (
            track is None
            or not config.enabled
            or not config.complete_skeleton
            or track.depth_guidance_joints_m is None
            or timestamp_s - track.last_detection_s > config.max_age_s
        ):
            return False
        history_valid = np.isfinite(
            track.depth_guidance_joints_m
        ).all(axis=1)
        if (
            int(np.count_nonzero(history_valid))
            < config.min_history_joint_count
        ):
            return False
        minimum_score = max(
            self.keypoint_threshold,
            config.min_keypoint_confidence,
        )
        core_2d = (
            observation.pose2d.scores[_TORSO_INDICES] >= minimum_score
        ) & np.isfinite(
            observation.pose2d.keypoints[_TORSO_INDICES]
        ).all(axis=1)
        finite_valid = observation.pose3d.valid & np.isfinite(
            observation.pose3d.joints_m
        ).all(axis=1)
        return bool(
            np.count_nonzero(core_2d) >= config.min_core_2d_joint_count
            and np.count_nonzero(finite_valid[_TORSO_INDICES])
            >= config.min_core_3d_joint_count
        )

    def _completion_link_max_length_m(
        self,
        link: tuple[int, int],
    ) -> float:
        config = self.pose3d_quality_config
        limits = {
            (19, 11): config.hip_offset_max_m,
            (11, 13): config.thigh_max_m,
            (13, 15): config.shin_max_m,
            (19, 12): config.hip_offset_max_m,
            (12, 14): config.thigh_max_m,
            (14, 16): config.shin_max_m,
            (19, 18): config.spine_max_m,
            (18, 17): config.head_neck_max_m,
            (18, 5): config.shoulder_offset_max_m,
            (5, 7): config.upper_arm_max_m,
            (7, 9): config.forearm_max_m,
            (18, 6): config.shoulder_offset_max_m,
            (6, 8): config.upper_arm_max_m,
            (8, 10): config.forearm_max_m,
        }
        if link in limits:
            return float(limits[link])
        return float(_EXTRA_COMPLETION_LINK_MAX_M[link])

    def _apply_kinematic_fallback(
        self,
        track: _TrackState,
        pose: TemporalPose3D,
        pose2d: Pose2D,
        intrinsics: CameraIntrinsics,
        timestamp_s: float,
        prior: BoneLengthPrior,
    ) -> tuple[TemporalPose3D, np.ndarray, np.ndarray]:
        config = self.kinematic_fallback_config
        fallback = np.zeros(len(HALPE26_NAMES), dtype=bool)
        skeleton_completion = np.zeros(len(HALPE26_NAMES), dtype=bool)
        history = track.depth_guidance_joints_m
        if not config.enabled or history is None:
            return pose, fallback, skeleton_completion

        joints = pose.joints_m.astype(np.float64, copy=True)
        confidence = pose.confidence.astype(np.float64, copy=True)
        usable = pose.usable.copy()
        observed = pose.observed.copy()
        predicted = pose.predicted.copy()
        age_s = pose.age_s.astype(np.float64, copy=True)
        minimum_score = max(
            self.keypoint_threshold,
            config.min_keypoint_confidence,
        )
        history_valid = np.isfinite(history).all(axis=1)
        core_2d = (
            pose2d.scores[_TORSO_INDICES] >= minimum_score
        ) & np.isfinite(pose2d.keypoints[_TORSO_INDICES]).all(axis=1)
        anchor_mask = (
            usable[_TORSO_INDICES]
            & observed[_TORSO_INDICES]
            & history_valid[_TORSO_INDICES]
        )
        anchor_indices = _TORSO_INDICES[anchor_mask]
        transform = _rigid_transform_from_anchors(
            history[anchor_indices],
            joints[anchor_indices],
            max_rotation_deg=config.max_torso_rotation_deg,
        )
        completion_allowed = bool(
            config.complete_skeleton
            and np.count_nonzero(core_2d)
            >= config.min_core_2d_joint_count
            and len(anchor_indices) >= config.min_core_3d_joint_count
            and np.count_nonzero(history_valid)
            >= config.min_history_joint_count
            and transform is not None
        )
        rotation, translation = (
            transform
            if transform is not None
            else (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        )
        anchor_confidence = (
            float(np.median(confidence[anchor_indices]))
            if len(anchor_indices)
            else 0.0
        )

        prior_indices = {
            link: bone_index for bone_index, link in enumerate(prior.links)
        }

        # The completion tree is ordered from the hip toward every leaf, so a
        # recovered elbow/knee can immediately anchor its child. Current 2D
        # can synthesize a bounded planar bone even when that child has never
        # had a valid depth sample; this is the important distinction from a
        # short history hold.
        for parent_index, child_index in _SKELETON_COMPLETION_LINKS:
            if usable[child_index]:
                continue
            if (
                not usable[parent_index]
                or not np.isfinite(joints[parent_index]).all()
            ):
                continue
            has_current_2d = bool(
                pose2d.scores[child_index] >= minimum_score
                and np.isfinite(pose2d.keypoints[child_index]).all()
            )
            history_pair_valid = bool(
                history_valid[parent_index] and history_valid[child_index]
            )
            last_observed_s = float(
                track.depth_guidance_observed_s[child_index]
            )
            missing_age_s = (
                timestamp_s - last_observed_s
                if np.isfinite(last_observed_s)
                else np.inf
            )
            history_recent = bool(
                history_pair_valid
                and 0 < missing_age_s <= config.max_age_s
            )
            previous_length_m: float | None = None
            if history_pair_valid:
                candidate_length_m = float(
                    np.linalg.norm(
                        history[child_index] - history[parent_index]
                    )
                )
                if np.isfinite(candidate_length_m) and candidate_length_m > 0:
                    previous_length_m = candidate_length_m
            bone_index = prior_indices.get((parent_index, child_index))
            calibrated_length_m: float | None = None
            if bone_index is not None:
                candidate_length_m = float(
                    prior.target_lengths_m[bone_index]
                )
                if np.isfinite(candidate_length_m) and candidate_length_m > 0:
                    calibrated_length_m = candidate_length_m
            preferred_length_m = calibrated_length_m or previous_length_m
            max_length_m = self._completion_link_max_length_m(
                (parent_index, child_index)
            )
            if (parent_index, child_index) == (19, 18) and has_current_2d:
                projected_length_m = _projected_link_length_at_parent_depth(
                    parent_m=joints[parent_index],
                    child_pixel=pose2d.keypoints[child_index],
                    intrinsics=intrinsics,
                )
                if projected_length_m is not None:
                    max_length_m = min(
                        max_length_m,
                        projected_length_m
                        * self.pose3d_quality_config.max_spine_projection_ratio
                        + self.pose3d_quality_config.spine_projection_slack_m,
                    )
            candidate = None
            if (
                has_current_2d
                and history_recent
                and preferred_length_m is not None
            ):
                max_displacement_m = (
                    config.displacement_slack_m
                    + config.max_joint_speed_m_s * missing_age_s
                )
                candidate = _ray_bone_candidate(
                    parent_m=joints[parent_index],
                    previous_parent_m=history[parent_index],
                    previous_child_m=history[child_index],
                    pixel=pose2d.keypoints[child_index],
                    bone_length_m=preferred_length_m,
                    intrinsics=intrinsics,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                    max_displacement_m=max_displacement_m,
                )
            used_completion = False
            if (
                candidate is None
                and completion_allowed
                and history_recent
                and preferred_length_m is not None
            ):
                max_displacement_m = (
                    config.displacement_slack_m
                    + config.max_joint_speed_m_s * missing_age_s
                )
                candidate = _propagated_bone_candidate(
                    parent_m=joints[parent_index],
                    previous_parent_m=history[parent_index],
                    previous_child_m=history[child_index],
                    rotation=rotation,
                    bone_length_m=preferred_length_m,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                    max_displacement_m=max_displacement_m,
                )
                used_completion = candidate is not None
            current_2d_completion = False
            if (
                candidate is None
                and completion_allowed
                and config.reconstruct_from_current_2d
                and has_current_2d
            ):
                candidate = _current_2d_bone_candidate(
                    parent_m=joints[parent_index],
                    child_pixel=pose2d.keypoints[child_index],
                    intrinsics=intrinsics,
                    preferred_length_m=preferred_length_m,
                    max_length_m=max_length_m,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                )
                current_2d_completion = candidate is not None
                used_completion = current_2d_completion
            if (
                candidate is None
                and completion_allowed
                and (parent_index, child_index) == (18, 17)
                and usable[19]
            ):
                candidate = _body_axis_head_candidate(
                    neck_m=joints[18],
                    hip_m=joints[19],
                    preferred_length_m=preferred_length_m,
                    max_length_m=max_length_m,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                )
                current_2d_completion = candidate is not None
                used_completion = current_2d_completion
            if candidate is None:
                continue
            remaining = (
                1.0
                if current_2d_completion
                else max(0.0, 1.0 - missing_age_s / config.max_age_s)
            )
            joints[child_index] = candidate
            confidence[child_index] = (
                (
                    float(pose2d.scores[child_index])
                    if has_current_2d
                    else anchor_confidence
                )
                * config.confidence_scale
                * remaining
            )
            usable[child_index] = True
            observed[child_index] = False
            predicted[child_index] = True
            age_s[child_index] = 0.0 if current_2d_completion else missing_age_s
            fallback[child_index] = True
            skeleton_completion[child_index] = used_completion

        # Face details, feet and any branch whose parent was unavailable are
        # not part of the calibrated constraint tree. Move their last real
        # coordinates rigidly with the current torso for a very short hold.
        if completion_allowed:
            for joint_index in range(len(HALPE26_NAMES)):
                if usable[joint_index] or not history_valid[joint_index]:
                    continue
                last_observed_s = float(
                    track.depth_guidance_observed_s[joint_index]
                )
                if not np.isfinite(last_observed_s):
                    continue
                missing_age_s = timestamp_s - last_observed_s
                if missing_age_s <= 0 or missing_age_s > config.max_age_s:
                    continue
                candidate = rotation @ history[joint_index] + translation
                max_displacement_m = (
                    config.displacement_slack_m
                    + config.max_joint_speed_m_s * missing_age_s
                )
                if (
                    not self.min_depth_m <= candidate[2] <= self.max_depth_m
                    or float(np.linalg.norm(candidate - history[joint_index]))
                    > max_displacement_m
                ):
                    continue
                remaining = max(
                    0.0,
                    1.0 - missing_age_s / config.max_age_s,
                )
                joints[joint_index] = candidate
                confidence[joint_index] = (
                    anchor_confidence * config.confidence_scale * remaining
                )
                usable[joint_index] = True
                observed[joint_index] = False
                predicted[joint_index] = True
                age_s[joint_index] = missing_age_s
                fallback[joint_index] = True
                skeleton_completion[joint_index] = True

        if not np.any(fallback):
            return pose, fallback, skeleton_completion
        return (
            TemporalPose3D(
                joints_m=joints,
                confidence=confidence,
                usable=usable,
                observed=observed,
                predicted=predicted,
                age_s=age_s,
                reset_occurred=pose.reset_occurred,
            ),
            fallback,
            skeleton_completion,
        )

    @staticmethod
    def _joint_sources(
        *,
        output_pose: TemporalPose3D,
        kinematic_fallback: np.ndarray,
        skeleton_completion: np.ndarray,
        observation: _DetectionObservation | None,
        status: str,
    ) -> tuple[str, ...]:
        sources: list[str] = []
        reasons = observation.joint_reasons if observation is not None else ()
        for index in range(len(HALPE26_NAMES)):
            if output_pose.observed[index]:
                sources.append("observed")
            elif skeleton_completion[index]:
                sources.append("skeleton_completion")
            elif kinematic_fallback[index]:
                sources.append("kinematic_fallback")
            elif output_pose.predicted[index]:
                sources.append("velocity_prediction")
            elif reasons and reasons[index] != "observed":
                sources.append(reasons[index])
            elif observation is None:
                sources.append(status)
            elif observation.pose3d.valid[index]:
                sources.append("low_3d_confidence")
            else:
                sources.append("prediction_expired")
        return tuple(sources)

    def _recover_observations(
        self,
        poses: list[Pose2D],
        frame: RGBDFrame,
    ) -> tuple[
        list[_DetectionObservation],
        dict[str, float],
        dict[str, int],
    ]:
        started = time.perf_counter()
        timing_ms = {
            "recovery_fast": 0.0,
            "recovery_cloud_build": 0.0,
            "recovery_robust": 0.0,
            "recovery_refine": 0.0,
            "recovery_refine_connected": 0.0,
            "recovery_refine_guided": 0.0,
        }
        recovery_stats = {
            "candidate_person_count": len(poses),
            "robust_joint_count": 0,
            "depth_connected_full_person_count": 0,
            "depth_connected_guided_person_count": 0,
        }
        organized_points: np.ndarray | None = None
        if self.recovery_method in ("pointcloud_cluster", "hybrid") and poses:
            cloud_started = time.perf_counter()
            organized_points = depth_to_organized_point_cloud(
                depth_m=frame.depth_m,
                intrinsics=frame.intrinsics,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            )
            timing_ms["recovery_cloud_build"] += (
                time.perf_counter() - cloud_started
            ) * 1000.0
        observations: list[_DetectionObservation] = []
        adaptive_requests: list[frozenset[int]] = []
        for pose2d in poses:
            if self.recovery_method == "depth_connected":
                pose3d = _empty_pose3d()
                observations.append(self._make_observation(pose2d, pose3d))
                continue
            needs_fast = self.recovery_method != "pointcloud_cluster"
            if needs_fast:
                fast_started = time.perf_counter()
                fast_pose3d = recover_pose3d(
                    pose2d=pose2d,
                    depth_m=frame.depth_m,
                    intrinsics=frame.intrinsics,
                    keypoint_threshold=self.keypoint_threshold,
                    radius=self.depth_window_radius,
                    min_depth_m=self.min_depth_m,
                    max_depth_m=self.max_depth_m,
                )
                timing_ms["recovery_fast"] += (
                    time.perf_counter() - fast_started
                ) * 1000.0
            if self.recovery_method == "adaptive_hybrid":
                requested = _adaptive_hybrid_joint_indices(
                    pose2d,
                    fast_pose3d,
                    keypoint_threshold=self.keypoint_threshold,
                    config=self.adaptive_hybrid_config,
                    pointcloud_config=self.pointcloud_config,
                )
                adaptive_requests.append(requested)
                pose3d = fast_pose3d
            elif organized_points is None:
                pose3d = fast_pose3d
            elif self.recovery_method == "pointcloud_cluster":
                robust_started = time.perf_counter()
                pose3d = recover_pose3d_from_point_cloud(
                    pose2d=pose2d,
                    organized_points_m=organized_points,
                    intrinsics=frame.intrinsics,
                    keypoint_threshold=self.keypoint_threshold,
                    config=self.pointcloud_config,
                ).pose3d
                timing_ms["recovery_robust"] += (
                    time.perf_counter() - robust_started
                ) * 1000.0
                recovery_stats["robust_joint_count"] += len(HALPE26_NAMES)
            else:
                robust_started = time.perf_counter()
                robust_pose3d = recover_pose3d_from_point_cloud(
                    pose2d=pose2d,
                    organized_points_m=organized_points,
                    intrinsics=frame.intrinsics,
                    keypoint_threshold=self.keypoint_threshold,
                    config=self.pointcloud_config,
                    joint_indices=_HYBRID_POINTCLOUD_INDICES,
                    person_depth_hint_m=_torso_depth_hint_m(fast_pose3d),
                ).pose3d
                pose3d = _merge_hybrid_pose3d(fast_pose3d, robust_pose3d)
                timing_ms["recovery_robust"] += (
                    time.perf_counter() - robust_started
                ) * 1000.0
                recovery_stats["robust_joint_count"] += len(
                    _HYBRID_POINTCLOUD_INDICES
                )
            observations.append(self._make_observation(pose2d, pose3d))

        if any(adaptive_requests):
            cloud_started = time.perf_counter()
            organized_points = depth_to_organized_point_cloud(
                depth_m=frame.depth_m,
                intrinsics=frame.intrinsics,
                min_depth_m=self.min_depth_m,
                max_depth_m=self.max_depth_m,
            )
            timing_ms["recovery_cloud_build"] += (
                time.perf_counter() - cloud_started
            ) * 1000.0
            for index, requested in enumerate(adaptive_requests):
                if not requested:
                    continue
                observation = observations[index]
                robust_started = time.perf_counter()
                robust_pose3d = recover_pose3d_from_point_cloud(
                    pose2d=observation.pose2d,
                    organized_points_m=organized_points,
                    intrinsics=frame.intrinsics,
                    keypoint_threshold=self.keypoint_threshold,
                    config=self.pointcloud_config,
                    joint_indices=requested,
                    person_depth_hint_m=_torso_depth_hint_m(
                        observation.pose3d
                    ),
                ).pose3d
                timing_ms["recovery_robust"] += (
                    time.perf_counter() - robust_started
                ) * 1000.0
                recovery_stats["robust_joint_count"] += len(requested)
                pose3d = _merge_hybrid_pose3d(
                    observation.pose3d,
                    robust_pose3d,
                    requested,
                )
                observations[index] = self._make_observation(
                    observation.pose2d,
                    pose3d,
                )

        timing_ms["recovery"] = (
            time.perf_counter() - started
        ) * 1000.0
        return observations, timing_ms, recovery_stats

    def _guided_observation(
        self,
        track: _TrackState,
        observation: _DetectionObservation,
        frame: RGBDFrame,
    ) -> _DetectionObservation:
        """Re-select local depth modes using this track's previous pose."""
        expected_depths = self._expected_depths(track)
        if expected_depths is None:
            return observation
        valid_expected = (
            np.isfinite(expected_depths)
            & (expected_depths >= self.min_depth_m)
            & (expected_depths <= self.max_depth_m)
        )
        valid_torso = _TORSO_INDICES[valid_expected[_TORSO_INDICES]]
        if len(valid_torso) >= 2:
            torso_depth = float(np.median(expected_depths[valid_torso]))
            expected_depths[~valid_expected] = torso_depth
        elif not np.any(valid_expected):
            return observation

        pose3d = recover_pose3d(
            pose2d=observation.pose2d,
            depth_m=frame.depth_m,
            intrinsics=frame.intrinsics,
            keypoint_threshold=self.keypoint_threshold,
            radius=self.depth_window_radius,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            expected_depths_m=expected_depths,
            max_expected_depth_delta_m=_GUIDED_MAX_DEPTH_DELTA_M,
        )
        return self._make_observation(observation.pose2d, pose3d)

    @staticmethod
    def _expected_depths(track: _TrackState) -> np.ndarray | None:
        guidance = track.depth_guidance_joints_m
        if guidance is None:
            return None
        expected = np.asarray(guidance[:, 2], dtype=np.float32).copy()
        expected[
            track.depth_guidance_failure_counts
            >= _DEPTH_GUIDANCE_REACQUIRE_FRAMES
        ] = np.nan
        return expected

    def _depth_connected_observation(
        self,
        track: _TrackState | None,
        observation: _DetectionObservation,
        frame: RGBDFrame,
    ) -> _DetectionObservation:
        expected_depths = (
            self._expected_depths(track) if track is not None else None
        )
        person_depth_hint_m: float | None = None
        if expected_depths is not None:
            torso_depths = expected_depths[_TORSO_INDICES]
            torso_depths = torso_depths[np.isfinite(torso_depths)]
            if len(torso_depths) >= 2:
                person_depth_hint_m = float(np.median(torso_depths))
        pose3d = recover_pose3d_from_depth_connected(
            pose2d=observation.pose2d,
            depth_m=frame.depth_m,
            intrinsics=frame.intrinsics,
            keypoint_threshold=self.keypoint_threshold,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            config=self.pointcloud_config,
            expected_depths_m=expected_depths,
            person_depth_hint_m=person_depth_hint_m,
        )
        return self._make_observation(observation.pose2d, pose3d)

    def _depth_connected_refresh_due(
        self,
        track: _TrackState,
        frame_number: int,
    ) -> bool:
        """Stagger exact connected-depth refreshes across established tracks."""

        if self.depth_connected_refresh_interval == 1:
            return True
        if self._expected_depths(track) is None:
            return True
        return (
            int(frame_number) + int(track.track_id)
        ) % self.depth_connected_refresh_interval == 0

    @staticmethod
    def _remember_depth_guidance(
        track: _TrackState,
        result: LocalPersonPoseResult,
        timestamp_s: float,
    ) -> None:
        if result.presence.track_reset_required:
            track.depth_guidance_joints_m = None
            track.depth_guidance_failure_counts.fill(0)
            track.depth_guidance_observed_s.fill(np.nan)
            return
        if track.depth_guidance_joints_m is None:
            if not result.observed_in_frame or not result.presence.accepted:
                return
            track.depth_guidance_joints_m = np.full_like(
                result.pose3d_output.joints_m,
                np.nan,
            )
        if not result.observed_in_frame or not result.presence.accepted:
            track.depth_guidance_failure_counts[:] = np.minimum(
                track.depth_guidance_failure_counts + 1,
                np.iinfo(np.int32).max,
            )
            return
        raw = result.pose3d_raw
        if raw is None:
            return
        observed = (
            raw.valid
            & result.pose3d_output.usable
            & np.isfinite(result.pose3d_output.joints_m).all(axis=1)
        )
        missing = ~observed
        track.depth_guidance_failure_counts[observed] = 0
        track.depth_guidance_failure_counts[missing] = np.minimum(
            track.depth_guidance_failure_counts[missing] + 1,
            np.iinfo(np.int32).max,
        )
        track.depth_guidance_joints_m[observed] = (
            result.pose3d_output.joints_m[observed]
        )
        track.depth_guidance_observed_s[observed] = timestamp_s

    def _quality_checked_observation(
        self,
        observation: _DetectionObservation,
        *,
        track: _TrackState | None,
        timestamp_s: float,
        intrinsics: CameraIntrinsics,
    ) -> tuple[_DetectionObservation, bool, int]:
        """Reject incoherent people or invalidate one contaminated branch."""

        config = self.pose3d_quality_config
        pose = observation.pose3d
        finite_valid = pose.valid & np.isfinite(pose.joints_m).all(axis=1)
        original_valid_count = int(np.count_nonzero(finite_valid))
        if not config.enabled or original_valid_count == 0:
            return observation, False, 0

        sparse_track_completion = self._can_complete_sparse_track(
            track,
            observation,
            timestamp_s,
        )
        minimum_valid_count = (
            self.kinematic_fallback_config.min_core_3d_joint_count
            if sparse_track_completion
            else config.min_valid_joint_count
        )
        minimum_torso_count = (
            self.kinematic_fallback_config.min_core_3d_joint_count
            if sparse_track_completion
            else config.min_valid_torso_joint_count
        )

        reject_whole = original_valid_count < minimum_valid_count
        torso_valid = _TORSO_INDICES[finite_valid[_TORSO_INDICES]]
        if len(torso_valid) < minimum_torso_count:
            reject_whole = True
        elif (
            float(np.ptp(pose.joints_m[torso_valid, 2]))
            > config.max_torso_depth_span_m
        ):
            reject_whole = True

        invalidated: set[int] = set()
        invalidated_reasons: dict[int, str] = {}

        def invalidate_branch(joint_id: int, reason: str) -> None:
            for index in _QUALITY_JOINT_BRANCHES.get(joint_id, (joint_id,)):
                invalidated.add(index)
                invalidated_reasons.setdefault(index, reason)

        minimum_score = max(
            self.keypoint_threshold,
            self.kinematic_fallback_config.min_keypoint_confidence,
        )
        if (
            not reject_whole
            and finite_valid[19]
            and finite_valid[18]
            and observation.pose2d.scores[18] >= minimum_score
        ):
            projected_spine_m = _projected_link_length_at_parent_depth(
                parent_m=pose.joints_m[19],
                child_pixel=observation.pose2d.keypoints[18],
                intrinsics=intrinsics,
            )
            spine_length_m = float(
                np.linalg.norm(pose.joints_m[18] - pose.joints_m[19])
            )
            if (
                projected_spine_m is not None
                and spine_length_m
                > projected_spine_m * config.max_spine_projection_ratio
                + config.spine_projection_slack_m
            ):
                invalidate_branch(18, "spine_projection_violation")

        if (
            not reject_whole
            and track is not None
            and track.depth_guidance_joints_m is not None
        ):
            previous = track.depth_guidance_joints_m
            history_valid = np.isfinite(previous).all(axis=1)
            common = finite_valid & history_valid
            delta_s = max(0.0, timestamp_s - track.last_detection_s)
            jump_threshold_m = min(
                config.max_depth_jump_m,
                config.depth_jump_base_m
                + config.max_depth_speed_m_s * delta_s,
            )
            depth_jump = common & (
                np.abs(pose.joints_m[:, 2] - previous[:, 2])
                > jump_threshold_m
            )
            jump_ids = set(int(index) for index in np.flatnonzero(depth_jump))
            if (
                len(jump_ids) >= config.reject_depth_jump_joint_count
                or bool(jump_ids & {18, 19})
            ):
                reject_whole = True
            else:
                for joint_id in jump_ids:
                    invalidate_branch(joint_id, "temporal_depth_jump")

        link_limits = (
            ((19, 11), config.hip_offset_max_m),
            ((11, 13), config.thigh_max_m),
            ((13, 15), config.shin_max_m),
            ((19, 12), config.hip_offset_max_m),
            ((12, 14), config.thigh_max_m),
            ((14, 16), config.shin_max_m),
            ((19, 18), config.spine_max_m),
            ((18, 17), config.head_neck_max_m),
            ((18, 5), config.shoulder_offset_max_m),
            ((5, 7), config.upper_arm_max_m),
            ((7, 9), config.forearm_max_m),
            ((18, 6), config.shoulder_offset_max_m),
            ((6, 8), config.upper_arm_max_m),
            ((8, 10), config.forearm_max_m),
        )
        prior_by_link: dict[tuple[int, int], tuple[float, bool]] = {}
        if track is not None and track.bone_calibrator is not None:
            prior = track.bone_calibrator.prior()
            prior_by_link = {
                link: (
                    float(prior.target_lengths_m[index]),
                    bool(prior.ready[index]),
                )
                for index, link in enumerate(prior.links)
            }

        violations: list[tuple[int, int]] = []
        for (start, end), absolute_max_m in link_limits:
            if (
                not finite_valid[start]
                or not finite_valid[end]
                or start in invalidated
                or end in invalidated
            ):
                continue
            length_m = float(
                np.linalg.norm(pose.joints_m[end] - pose.joints_m[start])
            )
            violates = not np.isfinite(length_m) or length_m > absolute_max_m
            target_m, prior_ready = prior_by_link.get(
                (start, end),
                (np.nan, False),
            )
            if prior_ready and np.isfinite(target_m) and target_m > 0:
                violates = violates or not (
                    target_m / config.prior_length_ratio
                    <= length_m
                    <= target_m * config.prior_length_ratio
                )
            if violates:
                violations.append((start, end))

        if (
            (19, 18) in violations
            or len(violations) >= config.reject_bone_violation_count
        ):
            reject_whole = True
        elif violations:
            for _start, end in violations:
                invalidate_branch(end, "bone_length_violation")

        face_anchor_index = (
            17
            if finite_valid[17] and 17 not in invalidated
            else 18
        )
        if finite_valid[face_anchor_index]:
            face_anchor = pose.joints_m[face_anchor_index]
            for face_index in range(5):
                if not finite_valid[face_index] or face_index in invalidated:
                    continue
                face_offset_m = float(
                    np.linalg.norm(pose.joints_m[face_index] - face_anchor)
                )
                if (
                    not np.isfinite(face_offset_m)
                    or face_offset_m > config.head_neck_max_m
                ):
                    invalidated.add(face_index)
                    invalidated_reasons.setdefault(
                        face_index,
                        "face_geometry_violation",
                    )

        if reject_whole:
            invalidated = set(
                int(index) for index in np.flatnonzero(finite_valid)
            )
            invalidated_reasons = {
                index: "person_quality_rejected" for index in invalidated
            }
        else:
            remaining = finite_valid.copy()
            if invalidated:
                remaining[list(invalidated)] = False
            if (
                np.count_nonzero(remaining) < minimum_valid_count
                or np.count_nonzero(remaining[_TORSO_INDICES])
                < minimum_torso_count
            ):
                reject_whole = True
                invalidated = set(
                    int(index) for index in np.flatnonzero(finite_valid)
                )
                invalidated_reasons = {
                    index: "person_quality_rejected" for index in invalidated
                }

        sanitized = _pose3d_with_invalidated_joints(pose, invalidated)
        reasons = list(observation.joint_reasons)
        if not reasons:
            reasons = [
                "observed" if finite_valid[index] else "no_depth_candidate"
                for index in range(len(HALPE26_NAMES))
            ]
        for index in invalidated:
            reasons[index] = invalidated_reasons.get(
                index,
                "person_quality_rejected",
            )
        return (
            _DetectionObservation(
                pose2d=observation.pose2d,
                pose3d=sanitized,
                root_camera_m=_root_camera_m(sanitized),
                joint_reasons=tuple(reasons),
            ),
            reject_whole,
            len(invalidated),
        )

    def _association_cost(
        self,
        track: _TrackState,
        observation: _DetectionObservation,
    ) -> float:
        previous = track.last_pose2d
        current = observation.pose2d
        iou = _bbox_iou(previous.bbox_xyxy, current.bbox_xyxy)
        previous_center, previous_diagonal = _bbox_center_and_diagonal(
            previous.bbox_xyxy
        )
        current_center, current_diagonal = _bbox_center_and_diagonal(
            current.bbox_xyxy
        )
        center_ratio = float(
            np.linalg.norm(current_center - previous_center)
            / max(1.0, 0.5 * (previous_diagonal + current_diagonal))
        )
        keypoint_ratio = _keypoint_distance_ratio(
            previous,
            current,
            self.keypoint_threshold,
        )
        root_distance: float | None = None
        if (
            track.last_root_camera_m is not None
            and observation.root_camera_m is not None
        ):
            root_distance = float(
                np.linalg.norm(
                    observation.root_camera_m - track.last_root_camera_m
                )
            )

        spatially_close = (
            iou >= self.config.minimum_bbox_iou
            or center_ratio <= self.config.max_center_distance_ratio
            or (
                keypoint_ratio is not None
                and keypoint_ratio <= self.config.max_keypoint_distance_ratio
            )
        )
        if not spatially_close:
            return _UNMATCHABLE_COST
        if (
            root_distance is not None
            and root_distance > self.config.max_root_distance_m
            and iou < self.config.minimum_bbox_iou
        ):
            return _UNMATCHABLE_COST

        center_term = min(
            2.0, center_ratio / self.config.max_center_distance_ratio
        )
        keypoint_term = (
            min(
                2.0,
                keypoint_ratio / self.config.max_keypoint_distance_ratio,
            )
            if keypoint_ratio is not None
            else center_term
        )
        root_term = (
            min(2.0, root_distance / self.config.max_root_distance_m)
            if root_distance is not None
            else 0.5
        )
        return float(
            0.35 * (1.0 - iou)
            + 0.25 * center_term
            + 0.25 * keypoint_term
            + 0.15 * root_term
        )

    def _associate(
        self,
        observations: list[_DetectionObservation],
    ) -> tuple[dict[int, tuple[int, float]], set[int], set[int]]:
        track_ids = sorted(self._tracks)
        detection_ids = [
            index
            for index, observation in enumerate(observations)
            if self.recovery_method == "depth_connected"
            or _has_valid_3d_observation(observation)
        ]
        if not track_ids or not detection_ids:
            return {}, set(track_ids), set(detection_ids)
        costs = np.full(
            (len(track_ids), len(detection_ids)),
            _UNMATCHABLE_COST,
            dtype=np.float64,
        )
        for row, track_id in enumerate(track_ids):
            track = self._tracks[track_id]
            for column, detection_index in enumerate(detection_ids):
                observation = observations[detection_index]
                costs[row, column] = self._association_cost(track, observation)

        rows, columns = linear_sum_assignment(costs)
        matches: dict[int, tuple[int, float]] = {}
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        for row, column in zip(rows, columns, strict=True):
            cost = float(costs[row, column])
            if cost >= _UNMATCHABLE_COST or cost > self.config.max_match_cost:
                continue
            track_id = track_ids[int(row)]
            detection_index = detection_ids[int(column)]
            matches[track_id] = (detection_index, cost)
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
        return (
            matches,
            set(track_ids) - matched_tracks,
            set(detection_ids) - matched_detections,
        )

    def _expire_stale_tracks(self, timestamp_s: float) -> None:
        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missing_since_s is not None
            and timestamp_s - track.missing_since_s > self.config.max_missing_s
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

    def _new_track(
        self,
        observation: _DetectionObservation,
        timestamp_s: float,
        *,
        track_id: int | None = None,
    ) -> _TrackState:
        calibrator, constraint = self.bone_components_factory()
        if (calibrator is None) != (constraint is None):
            raise ValueError(
                "bone_components_factory must return both components or neither."
            )
        resolved_track_id = self._next_track_id if track_id is None else track_id
        if resolved_track_id <= 0 or resolved_track_id in self._tracks:
            raise ValueError("New track_id must be positive and unique.")
        track = _TrackState(
            track_id=resolved_track_id,
            temporal_filter=self.temporal_filter_factory(),
            presence_gate=self.presence_gate_factory(),
            bone_calibrator=calibrator,
            bone_constraint=constraint,
            last_pose2d=observation.pose2d,
            last_root_camera_m=observation.root_camera_m,
            last_detection_s=timestamp_s,
        )
        self._tracks[track.track_id] = track
        self._next_track_id = max(self._next_track_id, track.track_id + 1)
        return track

    def _associate_shadow(
        self,
        observations: list[_DetectionObservation],
        rgb_bgr: np.ndarray,
        timestamp_s: float,
    ) -> tuple[
        dict[int, tuple[int, float]],
        set[int],
        dict[int, int],
    ]:
        """Return existing matches, missing tracks and new detection IDs."""

        if self._shadow_tracker is None:
            raise RuntimeError("Shadow identity tracker is not initialized.")
        identity_observations = [
            ShadowIdentityObservation(
                observation_id=index,
                pose2d=observation.pose2d,
                root_camera_m=observation.root_camera_m,
                appearance=extract_upper_body_hsv_descriptor(
                    rgb_bgr,
                    observation.pose2d,
                ),
            )
            for index, observation in enumerate(observations)
            if self.recovery_method == "depth_connected"
            or _has_valid_3d_observation(observation)
        ]
        identity_frame = self._shadow_tracker.update(
            identity_observations,
            timestamp_s,
        )
        for track_id in identity_frame.removed_shadow_ids:
            self._tracks.pop(track_id, None)

        matches: dict[int, tuple[int, float]] = {}
        new_detection_track_ids: dict[int, int] = {}
        for assignment in identity_frame.assignments:
            if assignment.shadow_id in self._tracks:
                matches[assignment.shadow_id] = (
                    assignment.observation_id,
                    assignment.match_cost or 0.0,
                )
            else:
                new_detection_track_ids[
                    assignment.observation_id
                ] = assignment.shadow_id
        missing_track_ids = set(self._tracks) - set(matches)
        return matches, missing_track_ids, new_detection_track_ids

    def _apply_track_state(
        self,
        track: _TrackState,
        *,
        timestamp_s: float,
        intrinsics: CameraIntrinsics,
        observation: _DetectionObservation | None,
        match_cost: float | None,
    ) -> LocalPersonPoseResult:
        pose2d = observation.pose2d if observation is not None else None
        pose3d = observation.pose3d if observation is not None else None
        presence = track.presence_gate.evaluate(
            pose2d,
            image_width=intrinsics.width,
            image_height=intrinsics.height,
            keypoint_threshold=self.keypoint_threshold,
        )
        status = "ok"
        accepted_pose2d = pose2d
        accepted_pose3d = pose3d
        if observation is None:
            status = "temporarily_missing"
        elif not presence.accepted:
            status = presence.reason
            accepted_pose2d = None
            accepted_pose3d = None
        elif not _has_valid_3d_observation(observation):
            status = "no_valid_3d_joints"
            accepted_pose2d = None
            accepted_pose3d = None
        elif track.bone_reset_pending:
            if track.bone_calibrator is not None:
                track.bone_calibrator.reset()
            track.bone_reset_pending = False
        observation_accepted = (
            observation is not None
            and presence.accepted
            and accepted_pose3d is not None
        )
        whole_person_missing = accepted_pose3d is None

        if presence.track_reset_required:
            temporal_pose = track.temporal_filter.terminate_track(timestamp_s)
            track.bone_reset_pending = True
        else:
            temporal_pose = track.temporal_filter.update(
                timestamp_s,
                accepted_pose3d,
            )

        corrected = np.zeros(len(HALPE26_NAMES), dtype=bool)
        kinematic_fallback = np.zeros(len(HALPE26_NAMES), dtype=bool)
        skeleton_completion = np.zeros(len(HALPE26_NAMES), dtype=bool)
        output_pose = temporal_pose
        if (
            not whole_person_missing
            and track.bone_calibrator is not None
            and track.bone_constraint is not None
        ):
            if accepted_pose3d is not None and accepted_pose2d is not None:
                track.bone_calibrator.update(
                    accepted_pose3d,
                    accepted_pose2d.scores,
                )
            prior = track.bone_calibrator.prior()
            if accepted_pose2d is not None:
                (
                    temporal_pose,
                    kinematic_fallback,
                    skeleton_completion,
                ) = (
                    self._apply_kinematic_fallback(
                        track,
                        temporal_pose,
                        accepted_pose2d,
                        intrinsics,
                        timestamp_s,
                        prior,
                    )
                )
            constrained = track.bone_constraint.apply(
                temporal_pose,
                prior,
            )
            output_pose = constrained.pose
            corrected = constrained.corrected
        if whole_person_missing:
            output_pose = _hide_whole_person_prediction(temporal_pose)

        joints_application = self.extrinsics.transform_points(
            output_pose.joints_m
        )
        joint_sources = self._joint_sources(
            output_pose=output_pose,
            kinematic_fallback=kinematic_fallback,
            skeleton_completion=skeleton_completion,
            observation=observation,
            status=status,
        )
        return LocalPersonPoseResult(
            track_id=track.track_id,
            status=status,
            observed_in_frame=observation_accepted,
            pose2d=pose2d,
            pose3d_raw=pose3d,
            pose3d_output=output_pose,
            corrected=corrected,
            joints_application_m=joints_application,
            presence=presence,
            match_cost=match_cost,
            kinematic_fallback=kinematic_fallback,
            joint_sources=joint_sources,
        )

    def process(self, frame: RGBDFrame) -> LocalMultiPersonPoseResult:
        started = time.perf_counter()
        inference_started = time.perf_counter()
        all_poses = self.backend.infer(frame.rgb_bgr)
        inference_ms = (time.perf_counter() - inference_started) * 1000.0
        poses = all_poses[: self.config.max_persons]
        observations, recovery_timing_ms, recovery_stats = (
            self._recover_observations(poses, frame)
        )
        recovery_ms = recovery_timing_ms["recovery"]
        timestamp_s = frame.timestamp_ns * 1e-9
        quality_ms = 0.0
        recovery_stats["quality_rejected_person_count"] = 0
        recovery_stats["quality_invalidated_joint_count"] = 0

        # Absolute person geometry is checked before either identity backend,
        # so a malformed but numerically valid pose cannot create a track.
        quality_started = time.perf_counter()
        for index, observation in enumerate(observations):
            checked, rejected, invalidated_count = (
                self._quality_checked_observation(
                    observation,
                    track=None,
                    timestamp_s=timestamp_s,
                    intrinsics=frame.intrinsics,
                )
            )
            observations[index] = checked
            recovery_stats["quality_rejected_person_count"] += int(rejected)
            recovery_stats["quality_invalidated_joint_count"] += (
                invalidated_count
            )
        quality_ms += (time.perf_counter() - quality_started) * 1000.0

        matching_started = time.perf_counter()
        identity_method = "geometry"
        identity_fallback = False
        new_detection_track_ids: dict[int, int] = {}
        if self._shadow_tracker is not None and not self._shadow_failed:
            try:
                (
                    matches,
                    unmatched_track_ids,
                    new_detection_track_ids,
                ) = self._associate_shadow(
                    observations,
                    frame.rgb_bgr,
                    timestamp_s,
                )
                unmatched_detection_ids = set(new_detection_track_ids)
                identity_method = "shadow"
            except Exception:
                self._shadow_failed = True
                identity_fallback = True
                LOGGER.exception(
                    "Shadow identity failed; using geometry for this run"
                )
        if identity_method == "geometry":
            identity_fallback = self.identity_tracker == "shadow"
            self._expire_stale_tracks(timestamp_s)
            (
                matches,
                unmatched_track_ids,
                unmatched_detection_ids,
            ) = self._associate(observations)
        matching_ms = (time.perf_counter() - matching_started) * 1000.0

        if self.recovery_method in ("guided_window", "depth_connected"):
            refinement_started = time.perf_counter()
            connected_refinement_ms = 0.0
            guided_refinement_ms = 0.0
            for track_id, (detection_index, _cost) in matches.items():
                if self.recovery_method == "guided_window":
                    observations[detection_index] = self._guided_observation(
                        self._tracks[track_id],
                        observations[detection_index],
                        frame,
                    )
                else:
                    track = self._tracks[track_id]
                    observation = observations[detection_index]
                    if self._depth_connected_refresh_due(
                        track,
                        frame.frame_number,
                    ):
                        connected_started = time.perf_counter()
                        observation = self._depth_connected_observation(
                            track,
                            observation,
                            frame,
                        )
                        connected_refinement_ms += (
                            time.perf_counter() - connected_started
                        ) * 1000.0
                        recovery_stats[
                            "depth_connected_full_person_count"
                        ] += 1
                    else:
                        guided_started = time.perf_counter()
                        guided = self._guided_observation(
                            track,
                            observation,
                            frame,
                        )
                        guided_refinement_ms += (
                            time.perf_counter() - guided_started
                        ) * 1000.0
                        if _has_valid_3d_observation(guided):
                            observation = guided
                            recovery_stats[
                                "depth_connected_guided_person_count"
                            ] += 1
                        else:
                            connected_started = time.perf_counter()
                            observation = self._depth_connected_observation(
                                track,
                                observation,
                                frame,
                            )
                            connected_refinement_ms += (
                                time.perf_counter() - connected_started
                            ) * 1000.0
                            recovery_stats[
                                "depth_connected_full_person_count"
                            ] += 1
                    observations[detection_index] = observation
            if self.recovery_method == "depth_connected":
                for detection_index in unmatched_detection_ids:
                    connected_started = time.perf_counter()
                    observations[detection_index] = self._depth_connected_observation(
                        None,
                        observations[detection_index],
                        frame,
                    )
                    connected_refinement_ms += (
                        time.perf_counter() - connected_started
                    ) * 1000.0
                    recovery_stats[
                        "depth_connected_full_person_count"
                    ] += 1
            refinement_ms = (
                time.perf_counter() - refinement_started
            ) * 1000.0
            recovery_ms += refinement_ms
            recovery_timing_ms["recovery_refine"] = refinement_ms
            recovery_timing_ms[
                "recovery_refine_connected"
            ] = connected_refinement_ms
            recovery_timing_ms[
                "recovery_refine_guided"
            ] = guided_refinement_ms
            recovery_timing_ms["recovery"] = recovery_ms

        # Once association supplies a track history, reject coherent jumps to
        # a background/door surface and use calibrated bone lengths when ready.
        quality_started = time.perf_counter()
        track_by_detection = {
            detection_index: self._tracks[track_id]
            for track_id, (detection_index, _cost) in matches.items()
        }
        for index, observation in enumerate(observations):
            checked, rejected, invalidated_count = (
                self._quality_checked_observation(
                    observation,
                    track=track_by_detection.get(index),
                    timestamp_s=timestamp_s,
                    intrinsics=frame.intrinsics,
                )
            )
            observations[index] = checked
            recovery_stats["quality_rejected_person_count"] += int(rejected)
            recovery_stats["quality_invalidated_joint_count"] += (
                invalidated_count
            )
        quality_ms += (time.perf_counter() - quality_started) * 1000.0

        # A guided/depth-connected refinement can invalidate every joint even
        # when the fast observation was initially matchable. Such a detection
        # must neither renew an existing track nor create a new one.
        invalid_detection_ids = {
            index
            for index, observation in enumerate(observations)
            if not _has_valid_3d_observation(observation)
        }
        for track_id, (detection_index, _cost) in list(matches.items()):
            if detection_index not in invalid_detection_ids:
                continue
            del matches[track_id]
            unmatched_track_ids.add(track_id)
        unmatched_detection_ids.difference_update(invalid_detection_ids)
        for detection_index in invalid_detection_ids:
            new_detection_track_ids.pop(detection_index, None)

        state_started = time.perf_counter()
        results: list[LocalPersonPoseResult] = []
        for track_id, (detection_index, cost) in matches.items():
            track = self._tracks[track_id]
            observation = observations[detection_index]
            track.last_pose2d = observation.pose2d
            if observation.root_camera_m is not None:
                track.last_root_camera_m = observation.root_camera_m
            track.last_detection_s = timestamp_s
            track.missing_since_s = None
            result = self._apply_track_state(
                track,
                timestamp_s=timestamp_s,
                intrinsics=frame.intrinsics,
                observation=observation,
                match_cost=cost,
            )
            self._remember_depth_guidance(track, result, timestamp_s)
            results.append(result)

        for detection_index in sorted(unmatched_detection_ids):
            observation = observations[detection_index]
            track = self._new_track(
                observation,
                timestamp_s,
                track_id=new_detection_track_ids.get(detection_index),
            )
            result = self._apply_track_state(
                track,
                timestamp_s=timestamp_s,
                intrinsics=frame.intrinsics,
                observation=observation,
                match_cost=None,
            )
            self._remember_depth_guidance(track, result, timestamp_s)
            results.append(result)

        for track_id in sorted(unmatched_track_ids):
            track = self._tracks[track_id]
            if track.missing_since_s is None:
                track.missing_since_s = timestamp_s
            result = self._apply_track_state(
                track,
                timestamp_s=timestamp_s,
                intrinsics=frame.intrinsics,
                observation=None,
                match_cost=None,
            )
            self._remember_depth_guidance(track, result, timestamp_s)
            results.append(result)

        state_ms = (time.perf_counter() - state_started) * 1000.0
        results.sort(key=lambda person: person.track_id)
        recovery_stats["kinematic_fallback_joint_count"] = sum(
            int(np.count_nonzero(person.kinematic_fallback))
            for person in results
            if person.observed_in_frame
        )
        recovery_stats["skeleton_completion_joint_count"] = sum(
            sum(source == "skeleton_completion" for source in person.joint_sources)
            for person in results
            if person.observed_in_frame
        )
        recovery_stats["missing_output_joint_count"] = sum(
            int(np.count_nonzero(~person.pose3d_output.usable))
            for person in results
            if person.observed_in_frame
        )
        total_ms = (time.perf_counter() - started) * 1000.0
        observed_count = sum(person.observed_in_frame for person in results)
        if observed_count:
            status = "ok"
        elif results:
            status = "tracks_predicted"
        else:
            status = "no_person"
        return LocalMultiPersonPoseResult(
            frame_number=frame.frame_number,
            timestamp_ns=frame.timestamp_ns,
            source_id=frame.source_id,
            rgb_bgr=frame.rgb_bgr,
            status=status,
            detected_person_count=len(all_poses),
            persons=tuple(results),
            identity_method=identity_method,
            identity_fallback=identity_fallback,
            recovery_stats=recovery_stats,
            timing_ms={
                "inference": inference_ms,
                "recovery": recovery_ms,
                "matching": matching_ms,
                "quality": quality_ms,
                "track_state": state_ms,
                "total": total_ms,
                **recovery_timing_ms,
            },
        )
