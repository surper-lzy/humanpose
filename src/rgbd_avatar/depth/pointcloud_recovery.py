"""Recover 3D joints from local surfaces in an organized point cloud."""

from __future__ import annotations

from collections import Counter, deque
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.pose.halpe26 import HALPE26_NAMES
from rgbd_avatar.pose.models import Pose2D, Pose3D

from .deprojection import deproject_pixel
from .topology import (
    TopologyCandidate,
    bilateral_length_outliers,
    select_face_core_candidates,
    select_foot_group_candidates,
)


TORSO_KEYPOINT_INDICES: tuple[int, ...] = (18, 19, 5, 6, 11, 12)
FACE_KEYPOINT_INDICES: frozenset[int] = frozenset((0, 1, 2, 3, 4, 17))
TORSO_JOINT_INDICES: frozenset[int] = frozenset((5, 6, 11, 12, 18, 19))
FOOT_KEYPOINT_INDICES: frozenset[int] = frozenset((20, 21, 22, 23, 24, 25))
EAR_TO_EYE_ANCHOR: dict[int, int] = {3: 1, 4: 2}
FACE_DEPTH_ANCHOR_INDICES: tuple[int, ...] = (0, 1, 2)
FOOT_GROUPS: tuple[tuple[int, int, int, int, int], ...] = (
    (13, 15, 20, 22, 24),
    (14, 16, 21, 23, 25),
)
SELF_OCCLUSION_GROUPS: tuple[
    tuple[str, int, int, int],
    ...,
] = (
    ("shoulder", 18, 5, 6),
    ("hip", 19, 11, 12),
)


@dataclass(frozen=True)
class PointCloudRecoveryConfig:
    """Parameters for local organized-point-cloud surface selection."""

    radius_scale_bbox_height: float = 0.025
    min_radius_px: int = 4
    max_radius_px: int = 9
    expansion_factor: float = 1.75
    expanded_max_radius_px: int = 14
    bbox_padding_ratio: float = 0.05
    min_cluster_points: int = 3
    support_target_points: int = 8
    depth_edge_abs_m: float = 0.025
    depth_edge_relative: float = 0.012
    max_depth_edge_m: float = 0.070
    cluster_depth_mad_scale_m: float = 0.050
    torso_keypoint_threshold: float = 0.5
    torso_group_gap_abs_m: float = 0.20
    torso_group_gap_relative: float = 0.08
    min_torso_seed_count: int = 2
    person_depth_near_tolerance_m: float = 0.75
    person_depth_near_tolerance_ratio: float = 0.35
    person_depth_far_tolerance_m: float = 0.60
    person_depth_far_tolerance_ratio: float = 0.25
    person_depth_sigma_m: float = 0.30
    person_depth_sigma_ratio: float = 0.12
    ambiguity_relative_margin: float = 0.02
    ambiguity_depth_gap_m: float = 0.08
    reject_ambiguous_clusters: bool = True
    face_group_gate_enabled: bool = True
    face_candidate_limit: int = 5
    face_core_depth_tolerance_m: float = 0.10
    face_core_depth_tolerance_ratio: float = 0.04
    nose_eye_max_length_m: float = 0.15
    eye_eye_max_length_m: float = 0.18
    face_neck_far_tolerance_m: float = 0.15
    face_missing_score: float = 0.25
    ear_topology_gate_enabled: bool = True
    ear_face_depth_tolerance_m: float = 0.20
    ear_face_depth_tolerance_ratio: float = 0.08
    eye_ear_max_length_m: float = 0.25
    min_face_anchor_count: int = 2
    foot_topology_gate_enabled: bool = True
    foot_candidate_limit: int = 6
    foot_ankle_min_score_ratio: float = 0.75
    foot_min_person_quality: float = 0.10
    foot_missing_score: float = 0.25
    foot_compactness_weight: float = 0.15
    knee_ankle_max_length_m: float = 0.75
    ankle_toe_max_length_m: float = 0.30
    ankle_heel_max_length_m: float = 0.22
    toe_pair_max_length_m: float = 0.20
    self_occlusion_gate_enabled: bool = True
    shoulder_neck_max_length_m: float = 0.32
    hip_center_max_length_m: float = 0.22
    self_occlusion_asymmetry_ratio: float = 1.60

    def __post_init__(self) -> None:
        if self.radius_scale_bbox_height <= 0:
            raise ValueError("radius_scale_bbox_height must be positive.")
        if self.min_radius_px < 0:
            raise ValueError("min_radius_px must be non-negative.")
        if self.max_radius_px < self.min_radius_px:
            raise ValueError("max_radius_px must be at least min_radius_px.")
        if self.expansion_factor < 1.0:
            raise ValueError("expansion_factor must be at least 1.")
        if self.expanded_max_radius_px < self.max_radius_px:
            raise ValueError(
                "expanded_max_radius_px must be at least max_radius_px."
            )
        if self.bbox_padding_ratio < 0:
            raise ValueError("bbox_padding_ratio must be non-negative.")
        if self.min_cluster_points <= 0:
            raise ValueError("min_cluster_points must be positive.")
        if self.support_target_points < self.min_cluster_points:
            raise ValueError(
                "support_target_points must be at least min_cluster_points."
            )
        if (
            self.depth_edge_abs_m <= 0
            or self.depth_edge_relative < 0
            or self.max_depth_edge_m < self.depth_edge_abs_m
        ):
            raise ValueError("Invalid 3D connectivity thresholds.")
        if self.cluster_depth_mad_scale_m <= 0:
            raise ValueError("cluster_depth_mad_scale_m must be positive.")
        if not 0 <= self.torso_keypoint_threshold <= 1:
            raise ValueError("torso_keypoint_threshold must be in [0, 1].")
        if self.min_torso_seed_count <= 0:
            raise ValueError("min_torso_seed_count must be positive.")
        positive_values = (
            self.torso_group_gap_abs_m,
            self.person_depth_near_tolerance_m,
            self.person_depth_far_tolerance_m,
            self.person_depth_sigma_m,
            self.ambiguity_depth_gap_m,
        )
        if any(value <= 0 for value in positive_values):
            raise ValueError("Metric point-cloud thresholds must be positive.")
        nonnegative_values = (
            self.torso_group_gap_relative,
            self.person_depth_near_tolerance_ratio,
            self.person_depth_far_tolerance_ratio,
            self.person_depth_sigma_ratio,
            self.ambiguity_relative_margin,
        )
        if any(value < 0 for value in nonnegative_values):
            raise ValueError("Relative point-cloud thresholds cannot be negative.")
        if self.ear_face_depth_tolerance_m <= 0:
            raise ValueError(
                "ear_face_depth_tolerance_m must be positive."
            )
        if self.ear_face_depth_tolerance_ratio < 0:
            raise ValueError(
                "ear_face_depth_tolerance_ratio must be non-negative."
            )
        if self.eye_ear_max_length_m <= 0:
            raise ValueError("eye_ear_max_length_m must be positive.")
        if self.min_face_anchor_count <= 0:
            raise ValueError("min_face_anchor_count must be positive.")
        if self.min_face_anchor_count > len(FACE_DEPTH_ANCHOR_INDICES):
            raise ValueError(
                "min_face_anchor_count exceeds available face anchors."
            )
        positive_integer_values = (
            self.face_candidate_limit,
            self.foot_candidate_limit,
        )
        if any(value <= 0 for value in positive_integer_values):
            raise ValueError("Topology candidate limits must be positive.")
        topology_positive_values = (
            self.face_core_depth_tolerance_m,
            self.nose_eye_max_length_m,
            self.eye_eye_max_length_m,
            self.face_neck_far_tolerance_m,
            self.knee_ankle_max_length_m,
            self.ankle_toe_max_length_m,
            self.ankle_heel_max_length_m,
            self.toe_pair_max_length_m,
            self.shoulder_neck_max_length_m,
            self.hip_center_max_length_m,
        )
        if any(value <= 0 for value in topology_positive_values):
            raise ValueError("Metric topology thresholds must be positive.")
        if self.face_core_depth_tolerance_ratio < 0:
            raise ValueError(
                "face_core_depth_tolerance_ratio must be non-negative."
            )
        if (
            self.face_missing_score < 0
            or self.foot_missing_score < 0
            or self.foot_compactness_weight < 0
        ):
            raise ValueError(
                "Topology missing scores and weights must be non-negative."
            )
        if not 0 < self.foot_ankle_min_score_ratio <= 1:
            raise ValueError(
                "foot_ankle_min_score_ratio must be in (0, 1]."
            )
        if not 0 <= self.foot_min_person_quality <= 1:
            raise ValueError(
                "foot_min_person_quality must be in [0, 1]."
            )
        if self.self_occlusion_asymmetry_ratio <= 1:
            raise ValueError(
                "self_occlusion_asymmetry_ratio must exceed 1."
            )

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> PointCloudRecoveryConfig:
        """Build a validated config while rejecting misspelled keys."""
        if values is None:
            return cls()
        valid_names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - valid_names)
        if unknown:
            raise ValueError(
                "Unknown pointcloud_cluster config key(s): "
                + ", ".join(unknown)
            )
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PointCloudRecoveryResult:
    pose3d: Pose3D
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _Patch:
    x1: int
    y1: int
    points: np.ndarray
    candidate: np.ndarray
    pixel_distance: np.ndarray


@dataclass(frozen=True)
class _Cluster:
    local_pixels: np.ndarray
    point_count: int
    depth_m: float
    depth_mad_m: float
    min_pixel_distance_px: float
    median_pixel_distance_px: float
    surface_medoid_xyz_m: np.ndarray
    surface_medoid_uv: np.ndarray
    q_center: float
    q_support: float
    q_compact: float
    q_dominance: float
    q_person: float | None
    selection_score: float


@dataclass(frozen=True)
class _JointSearch:
    u: float
    v: float
    pose_score: float
    radius: int
    patch: _Patch
    clusters: tuple[_Cluster, ...]


def _clip_bbox(
    bbox_xyxy: np.ndarray,
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError("Person bbox must contain four finite values.")
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Person bbox has no positive area: {bbox.tolist()}.")
    padding_x = (x2 - x1) * padding_ratio
    padding_y = (y2 - y1) * padding_ratio
    clipped_x1 = max(0, int(np.floor(x1 - padding_x)))
    clipped_y1 = max(0, int(np.floor(y1 - padding_y)))
    clipped_x2 = min(width, int(np.ceil(x2 + padding_x)))
    clipped_y2 = min(height, int(np.ceil(y2 + padding_y)))
    if clipped_x2 <= clipped_x1 or clipped_y2 <= clipped_y1:
        raise ValueError("Person bbox does not intersect the image.")
    return clipped_x1, clipped_y1, clipped_x2, clipped_y2


def _joint_radius(
    index: int,
    bbox_height_px: int,
    config: PointCloudRecoveryConfig,
) -> int:
    base = int(round(config.radius_scale_bbox_height * bbox_height_px))
    base = int(np.clip(base, config.min_radius_px, config.max_radius_px))
    if index in FACE_KEYPOINT_INDICES:
        scale = 0.75
    elif index in TORSO_JOINT_INDICES or index in FOOT_KEYPOINT_INDICES:
        scale = 1.25
    else:
        scale = 1.0
    return max(1, int(round(base * scale)))


def _make_patch(
    points_m: np.ndarray,
    candidate_mask: np.ndarray,
    u: float,
    v: float,
    radius: int,
) -> _Patch | None:
    height, width = candidate_mask.shape
    x1 = max(0, int(np.floor(u - radius)))
    x2 = min(width, int(np.ceil(u + radius)) + 1)
    y1 = max(0, int(np.floor(v - radius)))
    y2 = min(height, int(np.ceil(v + radius)) + 1)
    if x2 <= x1 or y2 <= y1:
        return None

    rows, columns = np.indices((y2 - y1, x2 - x1), dtype=np.float32)
    rows += y1
    columns += x1
    distance = np.sqrt(np.square(columns - u) + np.square(rows - v))
    disk = distance <= float(radius)
    local_candidate = candidate_mask[y1:y2, x1:x2] & disk
    return _Patch(
        x1=x1,
        y1=y1,
        points=points_m[y1:y2, x1:x2],
        candidate=local_candidate,
        pixel_distance=distance,
    )


def _edge_threshold_m(
    first: np.ndarray,
    second: np.ndarray,
    config: PointCloudRecoveryConfig,
) -> float:
    mean_depth = 0.5 * float(first[2] + second[2])
    adaptive = max(
        config.depth_edge_abs_m,
        config.depth_edge_relative * mean_depth,
    )
    return min(adaptive, config.max_depth_edge_m)


def _connected_components(
    patch: _Patch,
    config: PointCloudRecoveryConfig,
) -> list[np.ndarray]:
    candidate = patch.candidate
    visited = np.zeros(candidate.shape, dtype=bool)
    components: list[np.ndarray] = []
    height, width = candidate.shape

    for start_y, start_x in np.argwhere(candidate):
        start_y = int(start_y)
        start_x = int(start_x)
        if visited[start_y, start_x]:
            continue
        visited[start_y, start_x] = True
        queue: deque[tuple[int, int]] = deque(((start_y, start_x),))
        pixels: list[tuple[int, int]] = []

        while queue:
            current_y, current_x = queue.popleft()
            pixels.append((current_y, current_x))
            current_point = patch.points[current_y, current_x]
            for offset_y in (-1, 0, 1):
                for offset_x in (-1, 0, 1):
                    if offset_x == 0 and offset_y == 0:
                        continue
                    neighbor_y = current_y + offset_y
                    neighbor_x = current_x + offset_x
                    if (
                        neighbor_y < 0
                        or neighbor_y >= height
                        or neighbor_x < 0
                        or neighbor_x >= width
                        or visited[neighbor_y, neighbor_x]
                        or not candidate[neighbor_y, neighbor_x]
                    ):
                        continue
                    neighbor_point = patch.points[neighbor_y, neighbor_x]
                    threshold_m = _edge_threshold_m(
                        current_point,
                        neighbor_point,
                        config,
                    )
                    if (
                        float(np.linalg.norm(current_point - neighbor_point))
                        <= threshold_m
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))

        if len(pixels) >= config.min_cluster_points:
            components.append(np.asarray(pixels, dtype=np.int32))
    return components


def _cluster_from_component(
    component: np.ndarray,
    patch: _Patch,
    valid_point_count: int,
    radius: int,
    person_depth_m: float | None,
    config: PointCloudRecoveryConfig,
) -> _Cluster:
    local_y = component[:, 0]
    local_x = component[:, 1]
    points = patch.points[local_y, local_x]
    distances = patch.pixel_distance[local_y, local_x]
    depths = points[:, 2]
    depth_m = float(np.median(depths))
    depth_mad_m = float(np.median(np.abs(depths - depth_m)))
    median_xyz = np.median(points, axis=0)
    medoid_index = int(
        np.argmin(np.linalg.norm(points - median_xyz[None, :], axis=1))
    )
    surface_medoid_xyz_m = points[medoid_index].astype(np.float32)
    surface_medoid_uv = np.array(
        [
            patch.x1 + int(local_x[medoid_index]),
            patch.y1 + int(local_y[medoid_index]),
        ],
        dtype=np.int32,
    )
    min_distance = float(np.min(distances))
    median_distance = float(np.median(distances))
    center_sigma = max(1.0, 0.35 * radius)
    q_center = float(
        np.exp(-0.5 * np.square(min_distance / center_sigma))
    )
    q_support = float(
        min(1.0, len(component) / config.support_target_points)
    )
    q_compact = float(
        np.clip(
            1.0 - depth_mad_m / config.cluster_depth_mad_scale_m,
            0.0,
            1.0,
        )
    )
    q_dominance = float(len(component) / max(valid_point_count, 1))

    weighted_terms = [
        (0.35, q_center),
        (0.20, q_support),
        (0.20, q_compact),
        (0.10, q_dominance),
    ]
    q_person: float | None = None
    if person_depth_m is not None:
        sigma = max(
            config.person_depth_sigma_m,
            config.person_depth_sigma_ratio * person_depth_m,
        )
        q_person = float(
            np.exp(-0.5 * np.square((depth_m - person_depth_m) / sigma))
        )
        weighted_terms.append((0.15, q_person))
    weight_sum = sum(weight for weight, _ in weighted_terms)
    selection_score = sum(
        weight * value for weight, value in weighted_terms
    ) / weight_sum
    return _Cluster(
        local_pixels=component,
        point_count=len(component),
        depth_m=depth_m,
        depth_mad_m=depth_mad_m,
        min_pixel_distance_px=min_distance,
        median_pixel_distance_px=median_distance,
        surface_medoid_xyz_m=surface_medoid_xyz_m,
        surface_medoid_uv=surface_medoid_uv,
        q_center=q_center,
        q_support=q_support,
        q_compact=q_compact,
        q_dominance=q_dominance,
        q_person=q_person,
        selection_score=float(selection_score),
    )


def _clusters_for_patch(
    patch: _Patch,
    radius: int,
    person_depth_m: float | None,
    config: PointCloudRecoveryConfig,
) -> list[_Cluster]:
    valid_point_count = int(np.count_nonzero(patch.candidate))
    clusters = [
        _cluster_from_component(
            component,
            patch,
            valid_point_count,
            radius,
            person_depth_m,
            config,
        )
        for component in _connected_components(patch, config)
    ]
    return sorted(
        clusters,
        key=lambda cluster: cluster.selection_score,
        reverse=True,
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cutoff = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), cutoff, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _selected_depth(
    cluster: _Cluster,
    patch: _Patch,
    radius: int,
) -> float:
    local_y = cluster.local_pixels[:, 0]
    local_x = cluster.local_pixels[:, 1]
    depths = patch.points[local_y, local_x, 2]
    distances = patch.pixel_distance[local_y, local_x]
    sigma = max(1.0, 0.5 * radius)
    weights = np.exp(-0.5 * np.square(distances / sigma))
    return _weighted_median(depths, weights)


def _selection_margin(clusters: list[_Cluster]) -> float:
    if len(clusters) < 2:
        return 1.0
    best = clusters[0].selection_score
    second = clusters[1].selection_score
    return float(max(0.0, (best - second) / max(abs(best), 1e-12)))


def _filter_clusters_with_joint_topology(
    *,
    index: int,
    u: float,
    v: float,
    clusters: list[_Cluster],
    joints_m: np.ndarray,
    valid: np.ndarray,
    intrinsics: CameraIntrinsics,
    config: PointCloudRecoveryConfig,
) -> tuple[list[_Cluster], dict[str, Any] | None]:
    """Reject ear candidates inconsistent with already recovered face anchors."""

    if (
        not config.ear_topology_gate_enabled
        or index not in EAR_TO_EYE_ANCHOR
        or not clusters
    ):
        return clusters, None

    face_anchor_ids = [
        anchor_index
        for anchor_index in FACE_DEPTH_ANCHOR_INDICES
        if valid[anchor_index]
        and np.isfinite(joints_m[anchor_index]).all()
    ]
    eye_anchor_index = EAR_TO_EYE_ANCHOR[index]
    if (
        len(face_anchor_ids) < config.min_face_anchor_count
        or not valid[eye_anchor_index]
        or not np.isfinite(joints_m[eye_anchor_index]).all()
    ):
        return clusters, {
            "type": "ear_face_depth_eye_bone",
            "applied": False,
            "reason": "insufficient_face_anchors",
            "face_anchor_ids": face_anchor_ids,
            "eye_anchor_id": eye_anchor_index,
            "input_cluster_count": len(clusters),
            "feasible_cluster_count": len(clusters),
            "rejected_cluster_count": 0,
            "candidates": [],
        }

    face_reference_depth_m = float(
        np.median(joints_m[face_anchor_ids, 2])
    )
    depth_tolerance_m = max(
        config.ear_face_depth_tolerance_m,
        config.ear_face_depth_tolerance_ratio * face_reference_depth_m,
    )
    feasible: list[_Cluster] = []
    candidates: list[dict[str, Any]] = []
    eye_anchor = joints_m[eye_anchor_index]
    for cluster in clusters:
        candidate_joint = deproject_pixel(
            u,
            v,
            cluster.depth_m,
            intrinsics,
        )
        depth_residual_m = float(
            abs(cluster.depth_m - face_reference_depth_m)
        )
        eye_ear_length_m = float(
            np.linalg.norm(candidate_joint - eye_anchor)
        )
        rejection_reasons: list[str] = []
        if depth_residual_m > depth_tolerance_m:
            rejection_reasons.append("face_depth_residual")
        if eye_ear_length_m > config.eye_ear_max_length_m:
            rejection_reasons.append("eye_ear_length")
        accepted = not rejection_reasons
        if accepted:
            feasible.append(cluster)
        candidates.append(
            {
                "depth_m": cluster.depth_m,
                "selection_score": cluster.selection_score,
                "point_count": cluster.point_count,
                "face_depth_residual_m": depth_residual_m,
                "eye_ear_length_m": eye_ear_length_m,
                "accepted": accepted,
                "rejection_reasons": rejection_reasons,
            }
        )

    return feasible, {
        "type": "ear_face_depth_eye_bone",
        "applied": True,
        "face_anchor_ids": face_anchor_ids,
        "face_reference_depth_m": face_reference_depth_m,
        "depth_tolerance_m": depth_tolerance_m,
        "eye_anchor_id": eye_anchor_index,
        "eye_ear_max_length_m": config.eye_ear_max_length_m,
        "input_cluster_count": len(clusters),
        "feasible_cluster_count": len(feasible),
        "rejected_cluster_count": len(clusters) - len(feasible),
        "candidates": candidates,
    }


def _seed_depth_record(
    index: int,
    pose2d: Pose2D,
    points_m: np.ndarray,
    bbox_candidate: np.ndarray,
    radius: int,
    config: PointCloudRecoveryConfig,
) -> dict[str, Any] | None:
    u, v = pose2d.keypoints[index]
    patch = _make_patch(
        points_m,
        bbox_candidate,
        float(u),
        float(v),
        radius,
    )
    if patch is None:
        return None
    clusters = _clusters_for_patch(
        patch,
        radius,
        person_depth_m=None,
        config=config,
    )
    if not clusters:
        return None
    selected = clusters[0]
    return {
        "joint_id": index,
        "depth_m": selected.depth_m,
        "pose_score": float(np.clip(pose2d.scores[index], 0.0, 1.0)),
        "cluster_points": selected.point_count,
    }


def _person_depth_prior(
    pose2d: Pose2D,
    points_m: np.ndarray,
    bbox_candidate: np.ndarray,
    bbox_height_px: int,
    config: PointCloudRecoveryConfig,
) -> tuple[float | None, list[dict[str, Any]], list[int]]:
    seeds: list[dict[str, Any]] = []
    for index in TORSO_KEYPOINT_INDICES:
        if pose2d.scores[index] < config.torso_keypoint_threshold:
            continue
        seed = _seed_depth_record(
            index=index,
            pose2d=pose2d,
            points_m=points_m,
            bbox_candidate=bbox_candidate,
            radius=_joint_radius(index, bbox_height_px, config),
            config=config,
        )
        if seed is not None:
            seeds.append(seed)

    if len(seeds) < config.min_torso_seed_count:
        return None, seeds, []

    seed_depths = np.asarray(
        [seed["depth_m"] for seed in seeds], dtype=np.float64
    )
    order = np.argsort(seed_depths)
    sorted_depths = seed_depths[order]
    gap_m = max(
        config.torso_group_gap_abs_m,
        config.torso_group_gap_relative * float(np.median(sorted_depths)),
    )
    split_indices = np.flatnonzero(np.diff(sorted_depths) >= gap_m) + 1
    groups = np.split(order, split_indices)
    selected_group = max(
        groups,
        key=lambda group: (
            len(group),
            sum(seeds[int(item)]["pose_score"] for item in group),
            -float(np.median(seed_depths[group])),
        ),
    )
    if len(selected_group) < config.min_torso_seed_count:
        return None, seeds, []

    selected_depths = seed_depths[selected_group]
    selected_weights = np.asarray(
        [seeds[int(item)]["pose_score"] for item in selected_group],
        dtype=np.float64,
    )
    prior_m = _weighted_median(selected_depths, selected_weights)
    selected_ids = [
        int(seeds[int(item)]["joint_id"]) for item in selected_group
    ]
    return prior_m, seeds, selected_ids


def _depth_confidence(
    selected: _Cluster,
    margin: float,
    config: PointCloudRecoveryConfig,
) -> float:
    q_margin = (
        1.0
        if config.ambiguity_relative_margin == 0
        else float(
            np.clip(
                margin / config.ambiguity_relative_margin,
                0.0,
                1.0,
            )
        )
    )
    terms = [
        (0.25, selected.q_center),
        (0.20, selected.q_support),
        (0.20, selected.q_compact),
        (0.10, selected.q_dominance),
        (0.15, q_margin),
    ]
    if selected.q_person is not None:
        terms.append((0.10, selected.q_person))
    weight_sum = sum(weight for weight, _ in terms)
    return float(
        np.clip(
            sum(weight * value for weight, value in terms) / weight_sum,
            0.0,
            1.0,
        )
    )


def _topology_candidates(
    search: _JointSearch,
    intrinsics: CameraIntrinsics,
) -> list[TopologyCandidate]:
    candidates: list[TopologyCandidate] = []
    for token, cluster in enumerate(search.clusters):
        depth_m = _selected_depth(cluster, search.patch, search.radius)
        candidates.append(
            TopologyCandidate(
                token=token,
                xyz_m=deproject_pixel(
                    search.u,
                    search.v,
                    depth_m,
                    intrinsics,
                ),
                score=cluster.selection_score,
                person_quality=cluster.q_person,
            )
        )
    return candidates


def _clear_recovered_joint(
    index: int,
    *,
    status: str,
    joints: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    sampled_depth: np.ndarray,
    depth_confidence: np.ndarray,
    diagnostic: dict[str, Any],
) -> None:
    joints[index] = np.nan
    confidence[index] = 0.0
    valid[index] = False
    sampled_depth[index] = np.nan
    depth_confidence[index] = 0.0
    diagnostic["status"] = status
    diagnostic["depth_confidence"] = 0.0


def _apply_topology_cluster(
    index: int,
    token: int,
    *,
    search: _JointSearch,
    status: str,
    intrinsics: CameraIntrinsics,
    config: PointCloudRecoveryConfig,
    joints: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    sampled_depth: np.ndarray,
    depth_confidence: np.ndarray,
    diagnostic: dict[str, Any],
) -> None:
    cluster = search.clusters[token]
    depth_m = _selected_depth(cluster, search.patch, search.radius)
    joint = deproject_pixel(search.u, search.v, depth_m, intrinsics)
    topology_depth_confidence = _depth_confidence(
        cluster,
        margin=1.0,
        config=config,
    )
    previous_depth = (
        float(sampled_depth[index]) if valid[index] else None
    )
    diagnostic.setdefault(
        "pre_topology_selected_depth_m",
        previous_depth,
    )
    diagnostic.setdefault(
        "pre_topology_selection_margin",
        diagnostic.get("selection_margin"),
    )
    joints[index] = joint
    confidence[index] = (
        float(np.clip(search.pose_score, 0.0, 1.0))
        * topology_depth_confidence
    )
    valid[index] = True
    sampled_depth[index] = depth_m
    depth_confidence[index] = topology_depth_confidence
    diagnostic.update(
        {
            "status": status,
            "radius_px": search.radius,
            "selected_candidate_rank": token,
            "selected_point_count": cluster.point_count,
            "selected_depth_m": depth_m,
            "depth_mad_m": cluster.depth_mad_m,
            "center_distance_px": cluster.min_pixel_distance_px,
            "surface_medoid_uv": cluster.surface_medoid_uv.tolist(),
            "surface_medoid_xyz_m": (
                cluster.surface_medoid_xyz_m.tolist()
            ),
            "selection_score": cluster.selection_score,
            "selection_margin": 1.0,
            "depth_confidence": topology_depth_confidence,
        }
    )


def _apply_face_group_gate(
    *,
    searches: list[_JointSearch | None],
    joints: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    sampled_depth: np.ndarray,
    depth_confidence: np.ndarray,
    diagnostics: list[dict[str, Any]],
    intrinsics: CameraIntrinsics,
    config: PointCloudRecoveryConfig,
) -> None:
    if not config.face_group_gate_enabled:
        return
    candidate_map = {
        index: _topology_candidates(search, intrinsics)
        for index in FACE_DEPTH_ANCHOR_INDICES
        if (search := searches[index]) is not None
    }
    if len(candidate_map) < config.min_face_anchor_count:
        return
    neck_depth_m = (
        float(joints[18, 2])
        if valid[18] and np.isfinite(joints[18]).all()
        else None
    )
    selection = select_face_core_candidates(
        candidate_map,
        candidate_limit=config.face_candidate_limit,
        min_present=config.min_face_anchor_count,
        missing_score=config.face_missing_score,
        depth_tolerance_m=config.face_core_depth_tolerance_m,
        depth_tolerance_ratio=config.face_core_depth_tolerance_ratio,
        nose_eye_max_length_m=config.nose_eye_max_length_m,
        eye_eye_max_length_m=config.eye_eye_max_length_m,
        neck_depth_m=neck_depth_m,
        neck_far_tolerance_m=config.face_neck_far_tolerance_m,
    )
    if selection is None:
        return

    common = {
        "type": "nose_eye_joint_surface_selection",
        "applied": True,
        "objective": selection.objective,
        "evaluated_combination_count": (
            selection.evaluated_combination_count
        ),
        "feasible_combination_count": (
            selection.feasible_combination_count
        ),
        "neck_depth_m": neck_depth_m,
    }
    for index in FACE_DEPTH_ANCHOR_INDICES:
        search = searches[index]
        token = selection.selected_tokens.get(index)
        diagnostic = diagnostics[index]
        diagnostic["face_group_gate"] = {
            **common,
            "selected_candidate_rank": token,
        }
        if search is None:
            continue
        if token is None:
            _clear_recovered_joint(
                index,
                status="face_consistency_rejected",
                joints=joints,
                confidence=confidence,
                valid=valid,
                sampled_depth=sampled_depth,
                depth_confidence=depth_confidence,
                diagnostic=diagnostic,
            )
            continue
        _apply_topology_cluster(
            index,
            token,
            search=search,
            status="selected_face_group",
            intrinsics=intrinsics,
            config=config,
            joints=joints,
            confidence=confidence,
            valid=valid,
            sampled_depth=sampled_depth,
            depth_confidence=depth_confidence,
            diagnostic=diagnostic,
        )

    # Ear selection depends on the final nose/eye consensus, so rerun it even
    # when the first sequential pass had already accepted an ear surface.
    for index in EAR_TO_EYE_ANCHOR:
        search = searches[index]
        if search is None:
            continue
        feasible, gate = _filter_clusters_with_joint_topology(
            index=index,
            u=search.u,
            v=search.v,
            clusters=list(search.clusters),
            joints_m=joints,
            valid=valid,
            intrinsics=intrinsics,
            config=config,
        )
        diagnostics[index]["topology_gate"] = gate
        if not feasible:
            _clear_recovered_joint(
                index,
                status="joint_topology_rejected",
                joints=joints,
                confidence=confidence,
                valid=valid,
                sampled_depth=sampled_depth,
                depth_confidence=depth_confidence,
                diagnostic=diagnostics[index],
            )
            continue
        selected = feasible[0]
        token = next(
            token
            for token, candidate in enumerate(search.clusters)
            if candidate is selected
        )
        _apply_topology_cluster(
            index,
            token,
            search=search,
            status="selected_face_group",
            intrinsics=intrinsics,
            config=config,
            joints=joints,
            confidence=confidence,
            valid=valid,
            sampled_depth=sampled_depth,
            depth_confidence=depth_confidence,
            diagnostic=diagnostics[index],
        )
        if gate is not None and gate.get("applied"):
            face_reference_depth_m = float(
                gate["face_reference_depth_m"]
            )
            eye_anchor_id = int(gate["eye_anchor_id"])
            gate["selected_depth_residual_m"] = float(
                abs(joints[index, 2] - face_reference_depth_m)
            )
            gate["selected_eye_ear_length_m"] = float(
                np.linalg.norm(joints[index] - joints[eye_anchor_id])
            )


def _apply_foot_group_gate(
    *,
    searches: list[_JointSearch | None],
    joints: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    sampled_depth: np.ndarray,
    depth_confidence: np.ndarray,
    diagnostics: list[dict[str, Any]],
    intrinsics: CameraIntrinsics,
    config: PointCloudRecoveryConfig,
) -> None:
    if not config.foot_topology_gate_enabled:
        return
    for knee_id, ankle_id, big_id, small_id, heel_id in FOOT_GROUPS:
        group_ids = (ankle_id, big_id, small_id, heel_id)
        candidate_map = {
            index: _topology_candidates(search, intrinsics)
            for index in group_ids
            if (search := searches[index]) is not None
        }
        if ankle_id not in candidate_map:
            continue
        knee_xyz = (
            joints[knee_id].copy()
            if valid[knee_id] and np.isfinite(joints[knee_id]).all()
            else None
        )
        selection = select_foot_group_candidates(
            candidate_map,
            ankle_id=ankle_id,
            big_toe_id=big_id,
            small_toe_id=small_id,
            heel_id=heel_id,
            knee_xyz_m=knee_xyz,
            candidate_limit=config.foot_candidate_limit,
            ankle_min_score_ratio=config.foot_ankle_min_score_ratio,
            min_person_quality=config.foot_min_person_quality,
            missing_score=config.foot_missing_score,
            compactness_weight=config.foot_compactness_weight,
            knee_ankle_max_length_m=config.knee_ankle_max_length_m,
            ankle_toe_max_length_m=config.ankle_toe_max_length_m,
            ankle_heel_max_length_m=config.ankle_heel_max_length_m,
            toe_pair_max_length_m=config.toe_pair_max_length_m,
        )
        if selection is None:
            for index in (big_id, small_id, heel_id):
                search = searches[index]
                if search is None:
                    continue
                diagnostics[index]["foot_group_gate"] = {
                    "type": "foot_topology_ground_rejection",
                    "applied": True,
                    "reason": "no_coherent_group",
                }
                _clear_recovered_joint(
                    index,
                    status="foot_topology_ground_rejected",
                    joints=joints,
                    confidence=confidence,
                    valid=valid,
                    sampled_depth=sampled_depth,
                    depth_confidence=depth_confidence,
                    diagnostic=diagnostics[index],
                )
            continue

        common = {
            "type": "foot_topology_ground_rejection",
            "applied": True,
            "objective": selection.objective,
            "evaluated_combination_count": (
                selection.evaluated_combination_count
            ),
            "feasible_combination_count": (
                selection.feasible_combination_count
            ),
            "knee_id": knee_id,
            "ankle_id": ankle_id,
        }
        for index in group_ids:
            search = searches[index]
            token = selection.selected_tokens.get(index)
            diagnostic = diagnostics[index]
            diagnostic["foot_group_gate"] = {
                **common,
                "selected_candidate_rank": token,
            }
            if search is None:
                continue
            if token is None:
                _clear_recovered_joint(
                    index,
                    status="foot_topology_ground_rejected",
                    joints=joints,
                    confidence=confidence,
                    valid=valid,
                    sampled_depth=sampled_depth,
                    depth_confidence=depth_confidence,
                    diagnostic=diagnostic,
                )
                continue
            _apply_topology_cluster(
                index,
                token,
                search=search,
                status="selected_foot_group",
                intrinsics=intrinsics,
                config=config,
                joints=joints,
                confidence=confidence,
                valid=valid,
                sampled_depth=sampled_depth,
                depth_confidence=depth_confidence,
                diagnostic=diagnostic,
            )


def _apply_self_occlusion_gate(
    *,
    joints: np.ndarray,
    confidence: np.ndarray,
    valid: np.ndarray,
    sampled_depth: np.ndarray,
    depth_confidence: np.ndarray,
    diagnostics: list[dict[str, Any]],
    config: PointCloudRecoveryConfig,
) -> None:
    if not config.self_occlusion_gate_enabled:
        return
    for group_name, center_id, left_id, right_id in SELF_OCCLUSION_GROUPS:
        if not valid[center_id]:
            continue
        max_length_m = (
            config.shoulder_neck_max_length_m
            if group_name == "shoulder"
            else config.hip_center_max_length_m
        )
        left_outlier, right_outlier, lengths = bilateral_length_outliers(
            center_xyz_m=joints[center_id],
            left_xyz_m=joints[left_id] if valid[left_id] else None,
            right_xyz_m=joints[right_id] if valid[right_id] else None,
            max_length_m=max_length_m,
            asymmetry_ratio=config.self_occlusion_asymmetry_ratio,
        )
        for index, is_outlier in (
            (left_id, left_outlier),
            (right_id, right_outlier),
        ):
            if not is_outlier:
                continue
            diagnostics[index]["self_occlusion_gate"] = {
                "type": "bilateral_bone_asymmetry",
                "applied": True,
                "group": group_name,
                "center_id": center_id,
                "max_length_m": max_length_m,
                "asymmetry_ratio": (
                    config.self_occlusion_asymmetry_ratio
                ),
                **lengths,
            }
            _clear_recovered_joint(
                index,
                status="self_occlusion_topology_rejected",
                joints=joints,
                confidence=confidence,
                valid=valid,
                sampled_depth=sampled_depth,
                depth_confidence=depth_confidence,
                diagnostic=diagnostics[index],
            )


def recover_pose3d_from_point_cloud(
    pose2d: Pose2D,
    organized_points_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    keypoint_threshold: float = 0.3,
    config: PointCloudRecoveryConfig | None = None,
    person_mask: np.ndarray | None = None,
    joint_indices: Collection[int] | None = None,
    person_depth_hint_m: float | None = None,
) -> PointCloudRecoveryResult:
    """Lift a 2D pose using local 3D surface clusters.

    The selected point-cloud surface supplies a robust depth. The final joint
    is deprojected on the original 2D keypoint ray, so projecting the recovered
    XYZ through the same intrinsics returns the RTMPose pixel.

    Without ``person_mask``, the first implementation uses the padded detector
    bbox plus a broad torso-derived depth band as a proxy person mask. It is
    intentionally not presented as instance segmentation.

    ``joint_indices`` optionally limits the expensive local-surface search.
    The torso depth prior is still computed from all available torso seeds, so
    a caller can robustly recover a high-risk subset and merge it into a fast
    full-body result. Unrequested joints are returned invalid with a
    ``not_requested`` diagnostic.

    ``person_depth_hint_m`` can reuse a caller's already computed torso depth
    and skips the six point-cloud torso seed searches. This is intended for a
    hybrid recovery path; full point-cloud recovery should normally leave it
    unset.
    """
    recovery_config = config or PointCloudRecoveryConfig()
    points_m = np.asarray(organized_points_m, dtype=np.float32)
    expected_shape = (intrinsics.height, intrinsics.width, 3)
    if points_m.shape != expected_shape:
        raise ValueError(
            f"Expected organized point cloud shape {expected_shape}, "
            f"got {points_m.shape}."
        )
    if not 0 <= keypoint_threshold <= 1:
        raise ValueError("keypoint_threshold must be in [0, 1].")
    if person_depth_hint_m is not None and (
        not np.isfinite(person_depth_hint_m) or person_depth_hint_m <= 0
    ):
        raise ValueError("person_depth_hint_m must be finite and positive.")

    supplied_mask: np.ndarray | None = None
    if person_mask is not None:
        supplied_mask = np.asarray(person_mask, dtype=bool)
        if supplied_mask.shape != expected_shape[:2]:
            raise ValueError(
                "person_mask must match the organized point-cloud image grid: "
                f"expected {expected_shape[:2]}, got {supplied_mask.shape}."
            )

    height, width = expected_shape[:2]
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = _clip_bbox(
        pose2d.bbox_xyxy,
        width,
        height,
        recovery_config.bbox_padding_ratio,
    )
    bbox_mask = np.zeros((height, width), dtype=bool)
    bbox_mask[bbox_y1:bbox_y2, bbox_x1:bbox_x2] = True
    finite_points = np.isfinite(points_m).all(axis=2) & (points_m[:, :, 2] > 0)
    bbox_candidate = finite_points & bbox_mask
    if supplied_mask is not None:
        bbox_candidate &= supplied_mask

    bbox_height_px = bbox_y2 - bbox_y1
    if person_depth_hint_m is None:
        person_depth_m, seed_records, selected_seed_ids = _person_depth_prior(
            pose2d,
            points_m,
            bbox_candidate,
            bbox_height_px,
            recovery_config,
        )
        person_depth_source = "pointcloud_torso_seeds"
    else:
        person_depth_m = float(person_depth_hint_m)
        seed_records = []
        selected_seed_ids = []
        person_depth_source = "provided_hint"

    depth_band = np.ones((height, width), dtype=bool)
    near_tolerance_m: float | None = None
    far_tolerance_m: float | None = None
    if person_depth_m is not None:
        near_tolerance_m = max(
            recovery_config.person_depth_near_tolerance_m,
            recovery_config.person_depth_near_tolerance_ratio
            * person_depth_m,
        )
        far_tolerance_m = max(
            recovery_config.person_depth_far_tolerance_m,
            recovery_config.person_depth_far_tolerance_ratio
            * person_depth_m,
        )
        depth_band = (
            points_m[:, :, 2] >= person_depth_m - near_tolerance_m
        ) & (points_m[:, :, 2] <= person_depth_m + far_tolerance_m)

    proxy_mask = finite_points & bbox_mask & depth_band
    fallback_mask = finite_points & depth_band
    if supplied_mask is not None:
        proxy_mask &= supplied_mask
        fallback_mask &= supplied_mask

    count = pose2d.keypoints.shape[0]
    if joint_indices is None:
        requested_indices = frozenset(range(count))
    else:
        raw_indices = tuple(joint_indices)
        if any(
            not isinstance(index, (int, np.integer))
            for index in raw_indices
        ):
            raise ValueError("joint_indices must contain integer indices.")
        requested_indices = frozenset(int(index) for index in raw_indices)
        if any(index < 0 or index >= count for index in requested_indices):
            raise ValueError(
                f"joint_indices must be within [0, {count - 1}]."
            )
    joints = np.full((count, 3), np.nan, dtype=np.float32)
    confidence = np.zeros(count, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    sampled_depth = np.full(count, np.nan, dtype=np.float32)
    depth_confidence = np.zeros(count, dtype=np.float32)
    joint_diagnostics: list[dict[str, Any]] = []
    searches: list[_JointSearch | None] = [None] * count

    for index, ((u_value, v_value), pose_score_value) in enumerate(
        zip(pose2d.keypoints, pose2d.scores, strict=True)
    ):
        u = float(u_value)
        v = float(v_value)
        pose_score = float(pose_score_value)
        diagnostic: dict[str, Any] = {
            "id": index,
            "name": HALPE26_NAMES[index],
            "status": None,
            "radius_px": None,
            "expanded_radius": False,
            "bbox_mismatch": False,
            "candidate_point_count": 0,
            "cluster_count": 0,
            "feasible_cluster_count": 0,
            "selected_point_count": 0,
            "selected_depth_m": None,
            "depth_mad_m": None,
            "center_distance_px": None,
            "surface_medoid_uv": None,
            "surface_medoid_xyz_m": None,
            "selection_score": None,
            "selection_margin": None,
            "depth_confidence": 0.0,
            "topology_gate": None,
            "face_group_gate": None,
            "foot_group_gate": None,
            "self_occlusion_gate": None,
        }
        if index not in requested_indices:
            diagnostic["status"] = "not_requested"
            joint_diagnostics.append(diagnostic)
            continue
        if pose_score < keypoint_threshold:
            diagnostic["status"] = "low_pose_score"
            joint_diagnostics.append(diagnostic)
            continue
        if u < 0 or u >= width or v < 0 or v >= height:
            diagnostic["status"] = "outside_image"
            joint_diagnostics.append(diagnostic)
            continue

        bbox_mismatch = not (
            bbox_x1 <= u < bbox_x2 and bbox_y1 <= v < bbox_y2
        )
        diagnostic["bbox_mismatch"] = bbox_mismatch
        candidate_mask = fallback_mask if bbox_mismatch else proxy_mask
        base_radius = _joint_radius(
            index,
            bbox_height_px,
            recovery_config,
        )
        expanded_radius = min(
            recovery_config.expanded_max_radius_px,
            max(
                base_radius,
                int(np.ceil(base_radius * recovery_config.expansion_factor)),
            ),
        )
        radii = (
            [base_radius]
            if expanded_radius == base_radius
            else [base_radius, expanded_radius]
        )

        selected: _Cluster | None = None
        selected_patch: _Patch | None = None
        selected_clusters: list[_Cluster] = []
        selected_radius = base_radius
        saw_candidate = False
        saw_supported_cluster = False
        saw_topology_rejection = False
        for radius in radii:
            patch = _make_patch(points_m, candidate_mask, u, v, radius)
            if patch is None:
                continue
            candidate_count = int(np.count_nonzero(patch.candidate))
            saw_candidate |= candidate_count > 0
            raw_clusters = _clusters_for_patch(
                patch,
                radius,
                person_depth_m,
                recovery_config,
            )
            if raw_clusters and searches[index] is None:
                searches[index] = _JointSearch(
                    u=u,
                    v=v,
                    pose_score=pose_score,
                    radius=radius,
                    patch=patch,
                    clusters=tuple(raw_clusters),
                )
            saw_supported_cluster |= bool(raw_clusters)
            clusters, topology_gate = _filter_clusters_with_joint_topology(
                index=index,
                u=u,
                v=v,
                clusters=raw_clusters,
                joints_m=joints,
                valid=valid,
                intrinsics=intrinsics,
                config=recovery_config,
            )
            if topology_gate is not None:
                diagnostic["topology_gate"] = topology_gate
                saw_topology_rejection |= bool(
                    topology_gate["rejected_cluster_count"]
                )
            diagnostic["candidate_point_count"] = candidate_count
            diagnostic["cluster_count"] = len(raw_clusters)
            diagnostic["feasible_cluster_count"] = len(clusters)
            diagnostic["radius_px"] = radius
            if clusters:
                searches[index] = _JointSearch(
                    u=u,
                    v=v,
                    pose_score=pose_score,
                    radius=radius,
                    patch=patch,
                    clusters=tuple(raw_clusters),
                )
                selected = clusters[0]
                selected_patch = patch
                selected_clusters = clusters
                selected_radius = radius
                diagnostic["expanded_radius"] = radius != base_radius
                break

        if selected is None or selected_patch is None:
            if saw_supported_cluster and saw_topology_rejection:
                diagnostic["status"] = "joint_topology_rejected"
            else:
                diagnostic["status"] = (
                    "no_supported_cluster"
                    if saw_candidate
                    else "no_valid_depth"
                )
            joint_diagnostics.append(diagnostic)
            continue

        margin = _selection_margin(selected_clusters)
        ambiguous = (
            len(selected_clusters) >= 2
            and margin < recovery_config.ambiguity_relative_margin
            and abs(
                selected.depth_m - selected_clusters[1].depth_m
            )
            >= recovery_config.ambiguity_depth_gap_m
        )
        diagnostic.update(
            {
                "selected_point_count": selected.point_count,
                "selected_depth_m": selected.depth_m,
                "depth_mad_m": selected.depth_mad_m,
                "center_distance_px": selected.min_pixel_distance_px,
                "surface_medoid_uv": selected.surface_medoid_uv.tolist(),
                "surface_medoid_xyz_m": (
                    selected.surface_medoid_xyz_m.tolist()
                ),
                "selection_score": selected.selection_score,
                "selection_margin": margin,
            }
        )
        if ambiguous and recovery_config.reject_ambiguous_clusters:
            diagnostic["status"] = "ambiguous_clusters"
            joint_diagnostics.append(diagnostic)
            continue

        z_m = _selected_depth(selected, selected_patch, selected_radius)
        joint = deproject_pixel(u, v, z_m, intrinsics)
        topology_gate = diagnostic["topology_gate"]
        if topology_gate is not None and topology_gate["applied"]:
            final_depth_residual_m = float(
                abs(
                    z_m
                    - float(topology_gate["face_reference_depth_m"])
                )
            )
            eye_anchor_index = int(topology_gate["eye_anchor_id"])
            final_eye_ear_length_m = float(
                np.linalg.norm(joint - joints[eye_anchor_index])
            )
            topology_gate["selected_depth_residual_m"] = (
                final_depth_residual_m
            )
            topology_gate["selected_eye_ear_length_m"] = (
                final_eye_ear_length_m
            )
            if (
                final_depth_residual_m
                > float(topology_gate["depth_tolerance_m"])
                or final_eye_ear_length_m
                > recovery_config.eye_ear_max_length_m
            ):
                diagnostic["status"] = "joint_topology_rejected"
                joint_diagnostics.append(diagnostic)
                continue
        joint_depth_confidence = _depth_confidence(
            selected,
            margin,
            recovery_config,
        )
        joints[index] = joint
        sampled_depth[index] = z_m
        depth_confidence[index] = joint_depth_confidence
        confidence[index] = (
            float(np.clip(pose_score, 0.0, 1.0))
            * joint_depth_confidence
        )
        valid[index] = True
        diagnostic.update(
            {
                "status": (
                    "selected_ambiguous" if ambiguous else "selected"
                ),
                "selected_depth_m": z_m,
                "depth_confidence": joint_depth_confidence,
            }
        )
        joint_diagnostics.append(diagnostic)

    _apply_face_group_gate(
        searches=searches,
        joints=joints,
        confidence=confidence,
        valid=valid,
        sampled_depth=sampled_depth,
        depth_confidence=depth_confidence,
        diagnostics=joint_diagnostics,
        intrinsics=intrinsics,
        config=recovery_config,
    )
    _apply_foot_group_gate(
        searches=searches,
        joints=joints,
        confidence=confidence,
        valid=valid,
        sampled_depth=sampled_depth,
        depth_confidence=depth_confidence,
        diagnostics=joint_diagnostics,
        intrinsics=intrinsics,
        config=recovery_config,
    )
    _apply_self_occlusion_gate(
        joints=joints,
        confidence=confidence,
        valid=valid,
        sampled_depth=sampled_depth,
        depth_confidence=depth_confidence,
        diagnostics=joint_diagnostics,
        config=recovery_config,
    )

    status_counts = Counter(
        str(diagnostic["status"]) for diagnostic in joint_diagnostics
    )
    pose3d = Pose3D(
        joints_m=joints,
        confidence=confidence,
        valid=valid,
        depth_m=sampled_depth,
        depth_confidence=depth_confidence,
    )
    diagnostics = {
        "method": "pointcloud_cluster",
        "requested_joint_indices": sorted(requested_indices),
        "parameters": recovery_config.to_dict(),
        "person_filter": {
            "type": (
                "provided_mask_bbox_depth_band"
                if supplied_mask is not None
                else "bbox_depth_proxy"
            ),
            "external_mask_supplied": supplied_mask is not None,
            "padded_bbox_xyxy": [
                bbox_x1,
                bbox_y1,
                bbox_x2,
                bbox_y2,
            ],
            "person_depth_prior_m": person_depth_m,
            "person_depth_source": person_depth_source,
            "near_tolerance_m": near_tolerance_m,
            "far_tolerance_m": far_tolerance_m,
            "torso_seed_count": len(seed_records),
            "selected_torso_seed_ids": selected_seed_ids,
            "candidate_point_count": int(np.count_nonzero(proxy_mask)),
        },
        "valid_joint_count": int(np.count_nonzero(valid)),
        "joint_status_counts": dict(sorted(status_counts.items())),
        "joints": joint_diagnostics,
    }
    return PointCloudRecoveryResult(
        pose3d=pose3d,
        diagnostics=diagnostics,
    )
