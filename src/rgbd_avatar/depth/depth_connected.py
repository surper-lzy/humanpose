"""Lift 2D joints with local connected components in aligned depth images."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components as graph_components

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.pose.models import Pose2D, Pose3D

from .deprojection import deproject_pixel
from .pointcloud_recovery import PointCloudRecoveryConfig
from .topology import TopologyCandidate, bilateral_length_outliers
from .topology import select_face_core_candidates


_BODY_ORDER: tuple[tuple[int, int | None, float | None], ...] = (
    (19, None, None),
    (18, 19, 0.85),
    (11, 19, 0.22),
    (12, 19, 0.22),
    (5, 18, 0.32),
    (6, 18, 0.32),
    (13, 11, 0.75),
    (14, 12, 0.75),
    (7, 5, 0.55),
    (8, 6, 0.55),
    (15, 13, 0.75),
    (16, 14, 0.75),
    (9, 7, 0.55),
    (10, 8, 0.55),
    (17, 18, 0.40),
)
_FACE_CORE = (0, 1, 2)
_EAR_TO_EYE = ((3, 1), (4, 2))
_FOOT_LEAVES = (
    (20, 15, 0.30),
    (22, 15, 0.30),
    (24, 15, 0.22),
    (21, 16, 0.30),
    (23, 16, 0.30),
    (25, 16, 0.22),
)
_FACE_INDICES = frozenset((0, 1, 2, 3, 4, 17))
_TORSO_INDICES = frozenset((5, 6, 11, 12, 18, 19))
_FOOT_INDICES = frozenset((20, 21, 22, 23, 24, 25))


@dataclass(frozen=True)
class _DepthCandidate:
    token: int
    depth_m: float
    xyz_m: np.ndarray
    score: float
    person_quality: float | None


@dataclass(frozen=True)
class _SharedDepthWorkspace:
    """One connected-component labeling shared by all joints of a person."""

    x1: int
    y1: int
    depth_m: np.ndarray
    candidate: np.ndarray
    labels: np.ndarray
    bbox: tuple[int, int, int, int]
    bbox_height_px: int
    person_depth_hint_m: float | None
    person_sigma_m: float | None


def _clip_bbox(
    bbox_xyxy: np.ndarray,
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = np.asarray(bbox_xyxy, dtype=np.float64)
    padding_x = max(0.0, x2 - x1) * padding_ratio
    padding_y = max(0.0, y2 - y1) * padding_ratio
    return (
        max(0, int(np.floor(x1 - padding_x))),
        max(0, int(np.floor(y1 - padding_y))),
        min(width, int(np.ceil(x2 + padding_x))),
        min(height, int(np.ceil(y2 + padding_y))),
    )


def _joint_radius(
    index: int,
    bbox_height_px: int,
    config: PointCloudRecoveryConfig,
) -> int:
    radius = int(round(config.radius_scale_bbox_height * bbox_height_px))
    radius = int(np.clip(radius, config.min_radius_px, config.max_radius_px))
    if index in _FACE_INDICES:
        scale = 0.75
    elif index in _TORSO_INDICES or index in _FOOT_INDICES:
        scale = 1.25
    else:
        scale = 1.0
    return max(1, int(round(radius * scale)))


def _edge_threshold_m(
    first_depth_m: float,
    second_depth_m: float,
    config: PointCloudRecoveryConfig,
) -> float:
    mean_depth_m = 0.5 * (first_depth_m + second_depth_m)
    adaptive = max(
        config.depth_edge_abs_m,
        config.depth_edge_relative * mean_depth_m,
    )
    return min(adaptive, config.max_depth_edge_m)


def _connected_components(
    depths: np.ndarray,
    candidate: np.ndarray,
    config: PointCloudRecoveryConfig,
) -> list[np.ndarray]:
    """Return depth-aware 8-connected components without a Python BFS."""

    coordinates = np.argwhere(candidate).astype(np.int32, copy=False)
    point_count = len(coordinates)
    if point_count == 0:
        return []
    pixel_ids = np.full(candidate.shape, -1, dtype=np.int32)
    pixel_ids[coordinates[:, 0], coordinates[:, 1]] = np.arange(
        point_count,
        dtype=np.int32,
    )
    edge_starts: list[np.ndarray] = []
    edge_ends: list[np.ndarray] = []
    # Four unique directions are enough because the graph is undirected.
    for offset_y, offset_x in ((0, 1), (1, -1), (1, 0), (1, 1)):
        if offset_y == 0:
            first_slice = (slice(None), slice(0, -1))
            second_slice = (slice(None), slice(1, None))
        elif offset_x == -1:
            first_slice = (slice(0, -1), slice(1, None))
            second_slice = (slice(1, None), slice(0, -1))
        elif offset_x == 0:
            first_slice = (slice(0, -1), slice(None))
            second_slice = (slice(1, None), slice(None))
        else:
            first_slice = (slice(0, -1), slice(0, -1))
            second_slice = (slice(1, None), slice(1, None))
        first_candidate = candidate[first_slice]
        second_candidate = candidate[second_slice]
        first_depth = depths[first_slice]
        second_depth = depths[second_slice]
        mean_depth = 0.5 * (first_depth + second_depth)
        threshold = np.minimum(
            np.maximum(
                config.depth_edge_abs_m,
                config.depth_edge_relative * mean_depth,
            ),
            config.max_depth_edge_m,
        )
        connected = (
            first_candidate
            & second_candidate
            & (np.abs(first_depth - second_depth) <= threshold)
        )
        if not np.any(connected):
            continue
        edge_starts.append(pixel_ids[first_slice][connected])
        edge_ends.append(pixel_ids[second_slice][connected])

    if edge_starts:
        starts = np.concatenate(edge_starts)
        ends = np.concatenate(edge_ends)
        rows = np.concatenate((starts, ends))
        columns = np.concatenate((ends, starts))
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
            shape=(point_count, point_count),
        ).tocsr()
        _count, component_ids = graph_components(
            graph,
            directed=False,
            return_labels=True,
        )
    else:
        component_ids = np.arange(point_count, dtype=np.int32)
    sizes = np.bincount(component_ids)
    return [
        coordinates[component_ids == component_id]
        for component_id, size in enumerate(sizes)
        if size >= config.min_cluster_points
    ]


def _joint_candidates(
    *,
    index: int,
    u: float,
    v: float,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    bbox: tuple[int, int, int, int],
    bbox_height_px: int,
    min_depth_m: float,
    max_depth_m: float,
    person_depth_hint_m: float | None,
    expected_depth_m: float | None,
    config: PointCloudRecoveryConfig,
) -> list[_DepthCandidate]:
    base_radius = _joint_radius(index, bbox_height_px, config)
    expanded_radius = min(
        config.expanded_max_radius_px,
        max(base_radius, int(np.ceil(base_radius * config.expansion_factor))),
    )
    radii = (
        (base_radius,)
        if expanded_radius == base_radius
        else (base_radius, expanded_radius)
    )
    image_height, image_width = depth_m.shape
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
    center_inside_bbox = bbox_x1 <= u < bbox_x2 and bbox_y1 <= v < bbox_y2

    for radius in radii:
        x1 = max(0, int(np.floor(u - radius)))
        x2 = min(image_width, int(np.ceil(u + radius)) + 1)
        y1 = max(0, int(np.floor(v - radius)))
        y2 = min(image_height, int(np.ceil(v + radius)) + 1)
        if x2 <= x1 or y2 <= y1:
            continue
        patch = depth_m[y1:y2, x1:x2]
        rows, columns = np.indices(patch.shape, dtype=np.float32)
        global_rows = rows + y1
        global_columns = columns + x1
        pixel_distance = np.sqrt(
            np.square(global_columns - u) + np.square(global_rows - v)
        )
        candidate = (
            np.isfinite(patch)
            & (patch >= min_depth_m)
            & (patch <= max_depth_m)
            & (pixel_distance <= float(radius))
        )
        if center_inside_bbox:
            candidate &= (
                (global_columns >= bbox_x1)
                & (global_columns < bbox_x2)
                & (global_rows >= bbox_y1)
                & (global_rows < bbox_y2)
            )

        person_sigma_m: float | None = None
        if person_depth_hint_m is not None:
            near_tolerance_m = max(
                config.person_depth_near_tolerance_m,
                config.person_depth_near_tolerance_ratio
                * person_depth_hint_m,
            )
            far_tolerance_m = max(
                config.person_depth_far_tolerance_m,
                config.person_depth_far_tolerance_ratio
                * person_depth_hint_m,
            )
            candidate &= (
                (patch >= person_depth_hint_m - near_tolerance_m)
                & (patch <= person_depth_hint_m + far_tolerance_m)
            )
            person_sigma_m = max(
                config.person_depth_sigma_m,
                config.person_depth_sigma_ratio * person_depth_hint_m,
            )

        valid_point_count = int(np.count_nonzero(candidate))
        if valid_point_count == 0:
            continue
        components = _connected_components(patch, candidate, config)
        candidates: list[_DepthCandidate] = []
        for token, component in enumerate(components):
            local_y = component[:, 0]
            local_x = component[:, 1]
            values = patch[local_y, local_x]
            cluster_depth_m = float(np.median(values))
            depth_mad_m = float(
                np.median(np.abs(values - cluster_depth_m))
            )
            min_distance_px = float(
                np.min(pixel_distance[local_y, local_x])
            )
            center_sigma_px = max(1.0, 0.35 * radius)
            q_center = float(
                np.exp(-0.5 * np.square(min_distance_px / center_sigma_px))
            )
            q_support = float(
                min(1.0, len(component) / config.support_target_points)
            )
            q_compact = float(
                np.clip(
                    1.0
                    - depth_mad_m / config.cluster_depth_mad_scale_m,
                    0.0,
                    1.0,
                )
            )
            q_dominance = float(len(component) / valid_point_count)
            weighted_terms = [
                (0.30, q_center),
                (0.20, q_support),
                (0.15, q_compact),
                (0.10, q_dominance),
            ]
            q_person: float | None = None
            if person_sigma_m is not None and person_depth_hint_m is not None:
                q_person = float(
                    np.exp(
                        -0.5
                        * np.square(
                            (cluster_depth_m - person_depth_hint_m)
                            / person_sigma_m
                        )
                    )
                )
                weighted_terms.append((0.10, q_person))
            if expected_depth_m is not None:
                history_sigma_m = max(0.18, 0.08 * expected_depth_m)
                q_history = float(
                    np.exp(
                        -0.5
                        * np.square(
                            (cluster_depth_m - expected_depth_m)
                            / history_sigma_m
                        )
                    )
                )
                # History is deliberately a soft term. A spatially coherent
                # current surface can therefore recover from bad history.
                weighted_terms.append((0.15, q_history))
            score = sum(
                weight * quality for weight, quality in weighted_terms
            ) / sum(weight for weight, _quality in weighted_terms)
            candidates.append(
                _DepthCandidate(
                    token=token,
                    depth_m=cluster_depth_m,
                    xyz_m=deproject_pixel(
                        u,
                        v,
                        cluster_depth_m,
                        intrinsics,
                    ),
                    score=float(score),
                    person_quality=q_person,
                )
            )
        if candidates:
            return sorted(candidates, key=lambda value: value.score, reverse=True)
    return []


def _build_shared_workspace(
    *,
    pose2d: Pose2D,
    depth_m: np.ndarray,
    bbox: tuple[int, int, int, int],
    keypoint_threshold: float,
    min_depth_m: float,
    max_depth_m: float,
    person_depth_hint_m: float | None,
    config: PointCloudRecoveryConfig,
) -> _SharedDepthWorkspace:
    """Label the union of all expanded joint windows exactly once."""

    image_height, image_width = depth_m.shape
    bbox_height_px = max(1, bbox[3] - bbox[1])
    joint_windows: list[
        tuple[float, float, int, int, int, int, int]
    ] = []
    for index, ((u, v), pose_score) in enumerate(
        zip(pose2d.keypoints, pose2d.scores, strict=True)
    ):
        if (
            float(pose_score) < keypoint_threshold
            or not np.isfinite((u, v)).all()
            or u < 0
            or u >= image_width
            or v < 0
            or v >= image_height
        ):
            continue
        base_radius = _joint_radius(index, bbox_height_px, config)
        radius = min(
            config.expanded_max_radius_px,
            max(
                base_radius,
                int(np.ceil(base_radius * config.expansion_factor)),
            ),
        )
        x1 = max(0, int(np.floor(float(u) - radius)))
        x2 = min(image_width, int(np.ceil(float(u) + radius)) + 1)
        y1 = max(0, int(np.floor(float(v) - radius)))
        y2 = min(image_height, int(np.ceil(float(v) + radius)) + 1)
        if bbox[0] <= u < bbox[2] and bbox[1] <= v < bbox[3]:
            x1 = max(x1, bbox[0])
            x2 = min(x2, bbox[2])
            y1 = max(y1, bbox[1])
            y2 = min(y2, bbox[3])
        if x2 > x1 and y2 > y1:
            joint_windows.append((float(u), float(v), radius, x1, y1, x2, y2))

    if not joint_windows:
        empty_float = np.empty((0, 0), dtype=np.float32)
        empty_bool = np.empty((0, 0), dtype=bool)
        empty_labels = np.empty((0, 0), dtype=np.int32)
        return _SharedDepthWorkspace(
            x1=0,
            y1=0,
            depth_m=empty_float,
            candidate=empty_bool,
            labels=empty_labels,
            bbox=bbox,
            bbox_height_px=bbox_height_px,
            person_depth_hint_m=person_depth_hint_m,
            person_sigma_m=None,
        )

    roi_x1 = min(window[3] for window in joint_windows)
    roi_y1 = min(window[4] for window in joint_windows)
    roi_x2 = max(window[5] for window in joint_windows)
    roi_y2 = max(window[6] for window in joint_windows)
    roi_depth = depth_m[roi_y1:roi_y2, roi_x1:roi_x2]
    union = np.zeros(roi_depth.shape, dtype=bool)
    for u, v, radius, x1, y1, x2, y2 in joint_windows:
        rows, columns = np.ogrid[y1:y2, x1:x2]
        disk = (
            np.square(columns.astype(np.float32) - u)
            + np.square(rows.astype(np.float32) - v)
            <= float(radius * radius)
        )
        union[
            y1 - roi_y1 : y2 - roi_y1,
            x1 - roi_x1 : x2 - roi_x1,
        ] |= disk

    candidate = (
        union
        & np.isfinite(roi_depth)
        & (roi_depth >= min_depth_m)
        & (roi_depth <= max_depth_m)
    )
    person_sigma_m: float | None = None
    if person_depth_hint_m is not None:
        near_tolerance_m = max(
            config.person_depth_near_tolerance_m,
            config.person_depth_near_tolerance_ratio * person_depth_hint_m,
        )
        far_tolerance_m = max(
            config.person_depth_far_tolerance_m,
            config.person_depth_far_tolerance_ratio * person_depth_hint_m,
        )
        candidate &= (
            (roi_depth >= person_depth_hint_m - near_tolerance_m)
            & (roi_depth <= person_depth_hint_m + far_tolerance_m)
        )
        person_sigma_m = max(
            config.person_depth_sigma_m,
            config.person_depth_sigma_ratio * person_depth_hint_m,
        )

    labels = np.full(candidate.shape, -1, dtype=np.int32)
    for token, component in enumerate(
        _connected_components(roi_depth, candidate, config)
    ):
        labels[component[:, 0], component[:, 1]] = token
    return _SharedDepthWorkspace(
        x1=roi_x1,
        y1=roi_y1,
        depth_m=roi_depth,
        candidate=candidate,
        labels=labels,
        bbox=bbox,
        bbox_height_px=bbox_height_px,
        person_depth_hint_m=person_depth_hint_m,
        person_sigma_m=person_sigma_m,
    )


def _joint_candidates_from_workspace(
    *,
    index: int,
    u: float,
    v: float,
    intrinsics: CameraIntrinsics,
    workspace: _SharedDepthWorkspace,
    expected_depth_m: float | None,
    config: PointCloudRecoveryConfig,
) -> list[_DepthCandidate]:
    """Score shared components inside one joint's local support disk."""

    if not workspace.depth_m.size:
        return []
    base_radius = _joint_radius(index, workspace.bbox_height_px, config)
    expanded_radius = min(
        config.expanded_max_radius_px,
        max(base_radius, int(np.ceil(base_radius * config.expansion_factor))),
    )
    radii = (
        (base_radius,)
        if expanded_radius == base_radius
        else (base_radius, expanded_radius)
    )
    roi_height, roi_width = workspace.depth_m.shape
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = workspace.bbox
    center_inside_bbox = bbox_x1 <= u < bbox_x2 and bbox_y1 <= v < bbox_y2
    for radius in radii:
        local_x1 = max(0, int(np.floor(u - radius)) - workspace.x1)
        local_x2 = min(
            roi_width,
            int(np.ceil(u + radius)) + 1 - workspace.x1,
        )
        local_y1 = max(0, int(np.floor(v - radius)) - workspace.y1)
        local_y2 = min(
            roi_height,
            int(np.ceil(v + radius)) + 1 - workspace.y1,
        )
        if local_x2 <= local_x1 or local_y2 <= local_y1:
            continue
        patch = workspace.depth_m[local_y1:local_y2, local_x1:local_x2]
        patch_candidate = workspace.candidate[
            local_y1:local_y2,
            local_x1:local_x2,
        ].copy()
        patch_labels = workspace.labels[
            local_y1:local_y2,
            local_x1:local_x2,
        ]
        rows, columns = np.indices(patch.shape, dtype=np.float32)
        global_rows = rows + workspace.y1 + local_y1
        global_columns = columns + workspace.x1 + local_x1
        pixel_distance = np.sqrt(
            np.square(global_columns - u) + np.square(global_rows - v)
        )
        patch_candidate &= pixel_distance <= float(radius)
        if center_inside_bbox:
            patch_candidate &= (
                (global_columns >= bbox_x1)
                & (global_columns < bbox_x2)
                & (global_rows >= bbox_y1)
                & (global_rows < bbox_y2)
            )
        valid_point_count = int(np.count_nonzero(patch_candidate))
        if valid_point_count == 0:
            continue
        tokens = np.unique(patch_labels[patch_candidate])
        tokens = tokens[tokens >= 0]
        candidates: list[_DepthCandidate] = []
        for token_value in tokens:
            token = int(token_value)
            component = patch_candidate & (patch_labels == token)
            point_count = int(np.count_nonzero(component))
            if point_count < config.min_cluster_points:
                continue
            values = patch[component]
            cluster_depth_m = float(np.median(values))
            depth_mad_m = float(np.median(np.abs(values - cluster_depth_m)))
            min_distance_px = float(np.min(pixel_distance[component]))
            center_sigma_px = max(1.0, 0.35 * radius)
            q_center = float(
                np.exp(-0.5 * np.square(min_distance_px / center_sigma_px))
            )
            q_support = float(
                min(1.0, point_count / config.support_target_points)
            )
            q_compact = float(
                np.clip(
                    1.0 - depth_mad_m / config.cluster_depth_mad_scale_m,
                    0.0,
                    1.0,
                )
            )
            q_dominance = float(point_count / valid_point_count)
            weighted_terms = [
                (0.30, q_center),
                (0.20, q_support),
                (0.15, q_compact),
                (0.10, q_dominance),
            ]
            q_person: float | None = None
            if (
                workspace.person_sigma_m is not None
                and workspace.person_depth_hint_m is not None
            ):
                q_person = float(
                    np.exp(
                        -0.5
                        * np.square(
                            (
                                cluster_depth_m
                                - workspace.person_depth_hint_m
                            )
                            / workspace.person_sigma_m
                        )
                    )
                )
                weighted_terms.append((0.10, q_person))
            if expected_depth_m is not None:
                history_sigma_m = max(0.18, 0.08 * expected_depth_m)
                q_history = float(
                    np.exp(
                        -0.5
                        * np.square(
                            (cluster_depth_m - expected_depth_m)
                            / history_sigma_m
                        )
                    )
                )
                weighted_terms.append((0.15, q_history))
            score = sum(
                weight * quality for weight, quality in weighted_terms
            ) / sum(weight for weight, _quality in weighted_terms)
            candidates.append(
                _DepthCandidate(
                    token=token,
                    depth_m=cluster_depth_m,
                    xyz_m=deproject_pixel(u, v, cluster_depth_m, intrinsics),
                    score=float(score),
                    person_quality=q_person,
                )
            )
        if candidates:
            return sorted(
                candidates,
                key=lambda value: value.score,
                reverse=True,
            )
    return []


def _choose_candidate(
    candidates: list[_DepthCandidate],
    *,
    parent_xyz_m: np.ndarray | None,
    max_length_m: float | None,
    config: PointCloudRecoveryConfig,
) -> _DepthCandidate | None:
    feasible = candidates
    if parent_xyz_m is not None and max_length_m is not None:
        feasible = [
            candidate
            for candidate in candidates
            if float(np.linalg.norm(candidate.xyz_m - parent_xyz_m))
            <= max_length_m
        ]
    if not feasible:
        return None
    feasible = sorted(feasible, key=lambda value: value.score, reverse=True)
    if len(feasible) >= 2:
        margin = (feasible[0].score - feasible[1].score) / max(
            abs(feasible[0].score),
            1e-12,
        )
        depth_gap_m = abs(feasible[0].depth_m - feasible[1].depth_m)
        if (
            config.reject_ambiguous_clusters
            and margin < config.ambiguity_relative_margin
            and depth_gap_m >= config.ambiguity_depth_gap_m
        ):
            return None
    return feasible[0]


def recover_pose3d_from_depth_connected(
    pose2d: Pose2D,
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    *,
    keypoint_threshold: float = 0.3,
    min_depth_m: float = 0.2,
    max_depth_m: float = 8.0,
    config: PointCloudRecoveryConfig | None = None,
    expected_depths_m: np.ndarray | None = None,
    person_depth_hint_m: float | None = None,
) -> Pose3D:
    """Recover a pose without constructing an organized XYZ point cloud."""
    recovery_config = config or PointCloudRecoveryConfig()
    depth = np.asarray(depth_m, dtype=np.float32)
    expected_shape = (intrinsics.height, intrinsics.width)
    if depth.shape != expected_shape:
        raise ValueError(
            f"Expected depth shape {expected_shape}, got {depth.shape}."
        )
    if not 0 <= keypoint_threshold <= 1:
        raise ValueError("keypoint_threshold must be in [0, 1].")
    if min_depth_m <= 0 or max_depth_m <= min_depth_m:
        raise ValueError("Invalid metric depth range.")
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
    if person_depth_hint_m is not None:
        person_depth_hint_m = float(person_depth_hint_m)
        if not np.isfinite(person_depth_hint_m) or person_depth_hint_m <= 0:
            raise ValueError(
                "person_depth_hint_m must be finite, positive, or None."
            )

    bbox = _clip_bbox(
        pose2d.bbox_xyxy,
        intrinsics.width,
        intrinsics.height,
        recovery_config.bbox_padding_ratio,
    )
    workspace = _build_shared_workspace(
        pose2d=pose2d,
        depth_m=depth,
        bbox=bbox,
        keypoint_threshold=keypoint_threshold,
        min_depth_m=min_depth_m,
        max_depth_m=max_depth_m,
        person_depth_hint_m=person_depth_hint_m,
        config=recovery_config,
    )
    candidates_by_joint: list[list[_DepthCandidate]] = []
    for index, ((u, v), pose_score) in enumerate(
        zip(pose2d.keypoints, pose2d.scores, strict=True)
    ):
        if (
            float(pose_score) < keypoint_threshold
            or not np.isfinite((u, v)).all()
            or u < 0
            or u >= intrinsics.width
            or v < 0
            or v >= intrinsics.height
        ):
            candidates_by_joint.append([])
            continue
        expected = (
            float(expected_depths[index])
            if np.isfinite(expected_depths[index])
            else None
        )
        candidates_by_joint.append(
            _joint_candidates_from_workspace(
                index=index,
                u=float(u),
                v=float(v),
                intrinsics=intrinsics,
                workspace=workspace,
                expected_depth_m=expected,
                config=recovery_config,
            )
        )

    joints = np.full((count, 3), np.nan, dtype=np.float32)
    confidence = np.zeros(count, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    sampled_depth = np.full(count, np.nan, dtype=np.float32)
    depth_confidence = np.zeros(count, dtype=np.float32)

    def assign(index: int, candidate: _DepthCandidate | None) -> None:
        if candidate is None:
            return
        joints[index] = candidate.xyz_m
        sampled_depth[index] = candidate.depth_m
        depth_confidence[index] = candidate.score
        confidence[index] = float(pose2d.scores[index]) * candidate.score
        valid[index] = True

    def clear(index: int) -> None:
        joints[index] = np.nan
        sampled_depth[index] = np.nan
        depth_confidence[index] = 0.0
        confidence[index] = 0.0
        valid[index] = False

    for index, parent_index, max_length_m in _BODY_ORDER:
        parent = (
            joints[parent_index]
            if parent_index is not None and valid[parent_index]
            else None
        )
        assign(
            index,
            _choose_candidate(
                candidates_by_joint[index],
                parent_xyz_m=parent,
                max_length_m=max_length_m,
                config=recovery_config,
            ),
        )

    if recovery_config.face_group_gate_enabled:
        face_candidates = {
            index: [
                TopologyCandidate(
                    token=candidate.token,
                    xyz_m=candidate.xyz_m,
                    score=candidate.score,
                    person_quality=candidate.person_quality,
                )
                for candidate in candidates_by_joint[index]
            ]
            for index in _FACE_CORE
        }
        selection = select_face_core_candidates(
            face_candidates,
            candidate_limit=recovery_config.face_candidate_limit,
            min_present=recovery_config.min_face_anchor_count,
            missing_score=recovery_config.face_missing_score,
            depth_tolerance_m=(
                recovery_config.face_core_depth_tolerance_m
            ),
            depth_tolerance_ratio=(
                recovery_config.face_core_depth_tolerance_ratio
            ),
            nose_eye_max_length_m=recovery_config.nose_eye_max_length_m,
            eye_eye_max_length_m=recovery_config.eye_eye_max_length_m,
            neck_depth_m=float(joints[18, 2]) if valid[18] else None,
            neck_far_tolerance_m=recovery_config.face_neck_far_tolerance_m,
        )
        if selection is not None:
            for index in _FACE_CORE:
                token = selection.selected_tokens.get(index)
                candidate = next(
                    (
                        value
                        for value in candidates_by_joint[index]
                        if value.token == token
                    ),
                    None,
                )
                assign(index, candidate)
    else:
        for index in _FACE_CORE:
            assign(
                index,
                _choose_candidate(
                    candidates_by_joint[index],
                    parent_xyz_m=joints[18] if valid[18] else None,
                    max_length_m=0.40,
                    config=recovery_config,
                ),
            )

    for ear_index, eye_index in _EAR_TO_EYE:
        eye = joints[eye_index] if valid[eye_index] else None
        candidates = candidates_by_joint[ear_index]
        if eye is not None:
            face_depths = sampled_depth[
                np.asarray(_FACE_CORE, dtype=np.int64)
            ]
            face_depths = face_depths[np.isfinite(face_depths)]
            reference_depth = (
                float(np.median(face_depths))
                if len(face_depths)
                else float(eye[2])
            )
            tolerance = max(
                recovery_config.ear_face_depth_tolerance_m,
                recovery_config.ear_face_depth_tolerance_ratio
                * reference_depth,
            )
            candidates = [
                candidate
                for candidate in candidates
                if abs(candidate.depth_m - reference_depth) <= tolerance
            ]
        assign(
            ear_index,
            _choose_candidate(
                candidates,
                parent_xyz_m=eye,
                max_length_m=recovery_config.eye_ear_max_length_m,
                config=recovery_config,
            ),
        )

    for index, ankle_index, max_length_m in _FOOT_LEAVES:
        assign(
            index,
            _choose_candidate(
                candidates_by_joint[index],
                parent_xyz_m=(
                    joints[ankle_index] if valid[ankle_index] else None
                ),
                max_length_m=max_length_m,
                config=recovery_config,
            ),
        )
    for first, second in ((20, 22), (21, 23)):
        if (
            valid[first]
            and valid[second]
            and float(np.linalg.norm(joints[first] - joints[second]))
            > recovery_config.toe_pair_max_length_m
        ):
            clear(
                first
                if depth_confidence[first] < depth_confidence[second]
                else second
            )

    if recovery_config.self_occlusion_gate_enabled:
        for center, left, right, max_length in (
            (18, 5, 6, recovery_config.shoulder_neck_max_length_m),
            (19, 11, 12, recovery_config.hip_center_max_length_m),
        ):
            if not valid[center]:
                continue
            left_outlier, right_outlier, _lengths = bilateral_length_outliers(
                center_xyz_m=joints[center],
                left_xyz_m=joints[left] if valid[left] else None,
                right_xyz_m=joints[right] if valid[right] else None,
                max_length_m=max_length,
                asymmetry_ratio=(
                    recovery_config.self_occlusion_asymmetry_ratio
                ),
            )
            if left_outlier:
                clear(left)
            if right_outlier:
                clear(right)

    return Pose3D(
        joints_m=joints,
        confidence=confidence,
        valid=valid,
        depth_m=sampled_depth,
        depth_confidence=depth_confidence,
    )


__all__ = ["recover_pose3d_from_depth_connected"]
