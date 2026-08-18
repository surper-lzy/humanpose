"""Metric-avatar to unrelated-static-scene placement.

The RGB-D avatar world ``W`` is right handed, measured in metres, and uses
``X right, Y forward, Z up``.  A separately reconstructed 3DGS world ``G`` has
no shared origin or guaranteed metric scale.  This module stores the explicit
similarity transform

``point_g = scale_g_per_m * rotation_g_from_w @ point_w + translation_g``.

For an unrelated scene this is an authored placement, not a recovered sensor
extrinsic.  The placement is constructed from a known scene length, a fitted
floor, a spawn point, and a forward point.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.io import atomic_write_json, load_json_mapping


_EPSILON = 1e-10
_KNOWN_LENGTH_DIRECTIONS = {"any", "vertical", "horizontal"}
_MAX_GROUND_RESIDUAL_M = 0.05
_MAX_PLACEMENT_POINT_DISTANCE_M = 0.15
_MAX_KNOWN_LENGTH_ANGLE_DEG = 20.0


def _finite_vector(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite XYZ vector.")
    return array


def _finite_points(
    value: Any,
    *,
    name: str,
    minimum_count: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 3
        or len(array) < minimum_count
        or not np.isfinite(array).all()
    ):
        raise ValueError(
            f"{name} must contain at least {minimum_count} finite XYZ points."
        )
    return array


def _unit_vector(value: Any, *, name: str) -> np.ndarray:
    vector = _finite_vector(value, name=name)
    norm = float(np.linalg.norm(vector))
    if norm <= _EPSILON:
        raise ValueError(f"{name} must be non-zero.")
    return vector / norm


def _proper_rotation(value: Any, *, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name} must be a finite 3x3 matrix.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
        raise ValueError(f"{name} must be orthonormal.")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must be a proper rotation with det=+1.")
    return rotation


def fit_ground_plane(
    points_g: Any,
    *,
    up_reference_g: Any,
) -> tuple[np.ndarray, float, np.ndarray]:
    """Fit ``normal_g dot point_g + offset_g = 0`` using SVD.

    ``up_reference_g`` is a point visibly above the selected floor.  It
    resolves the otherwise ambiguous sign of the plane normal.
    """

    points = _finite_points(
        points_g,
        name="ground_points_g",
        minimum_count=3,
    )
    reference = _finite_vector(up_reference_g, name="up_reference_g")
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, singular_values, right_vectors = np.linalg.svd(
        centered,
        full_matrices=False,
    )
    if singular_values[1] <= _EPSILON:
        raise ValueError("ground_points_g must not be collinear.")
    normal = _unit_vector(right_vectors[-1], name="ground normal")
    if float(np.dot(normal, reference - centroid)) < 0.0:
        normal = -normal
    separation = float(np.dot(normal, reference - centroid))
    if separation <= _EPSILON:
        raise ValueError("up_reference_g must be visibly above the ground plane.")
    offset = -float(np.dot(normal, centroid))
    residuals = np.abs(points @ normal + offset)
    return normal, offset, residuals


def fit_ground_plane_robust(
    points_g: Any,
    *,
    up_reference_g: Any,
    scale_g_per_m: float,
    maximum_residual_m: float = _MAX_GROUND_RESIDUAL_M,
    minimum_inlier_fraction: float = 0.8,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit a floor while allowing sparse expected-depth outliers.

    At least five picks are required before an outlier may be excluded, and
    at least 80 percent (with a minimum of four points) must agree. Residuals
    are returned for every input point, alongside the selected inlier mask.
    """

    points = _finite_points(
        points_g,
        name="ground_points_g",
        minimum_count=3,
    )
    scale = float(scale_g_per_m)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale_g_per_m must be finite and positive.")
    threshold = float(maximum_residual_m)
    if not math.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("maximum_residual_m must be finite and positive.")
    fraction = float(minimum_inlier_fraction)
    if not math.isfinite(fraction) or not 0.5 < fraction <= 1.0:
        raise ValueError("minimum_inlier_fraction must lie in (0.5,1].")

    normal, offset, residuals = fit_ground_plane(
        points,
        up_reference_g=up_reference_g,
    )
    residuals_m = residuals / scale
    if float(np.max(residuals_m)) <= threshold:
        return normal, offset, residuals, np.ones(len(points), dtype=bool)

    minimum_inliers = max(
        4,
        math.ceil(fraction * len(points)),
        len(points) - 1,
    )
    best: tuple[
        tuple[int, float, float],
        np.ndarray,
        float,
        np.ndarray,
        np.ndarray,
    ] | None = None
    if len(points) >= 5:
        for indices in combinations(range(len(points)), minimum_inliers):
            try:
                candidate_normal, candidate_offset, _ = fit_ground_plane(
                    points[list(indices)],
                    up_reference_g=up_reference_g,
                )
            except ValueError:
                continue
            candidate_residuals = np.abs(
                points @ candidate_normal + candidate_offset
            )
            candidate_inliers = candidate_residuals / scale <= threshold
            if int(np.count_nonzero(candidate_inliers)) < minimum_inliers:
                continue
            refined_normal, refined_offset, _ = fit_ground_plane(
                points[candidate_inliers],
                up_reference_g=up_reference_g,
            )
            refined_residuals = np.abs(points @ refined_normal + refined_offset)
            refined_inliers = refined_residuals / scale <= threshold
            count = int(np.count_nonzero(refined_inliers))
            if count < minimum_inliers:
                continue
            inlier_residuals_m = refined_residuals[refined_inliers] / scale
            score = (
                count,
                -float(np.median(inlier_residuals_m)),
                -float(np.mean(inlier_residuals_m)),
            )
            if best is None or score > best[0]:
                best = (
                    score,
                    refined_normal,
                    refined_offset,
                    refined_residuals,
                    refined_inliers,
                )
    if best is not None:
        _, normal, offset, residuals, inliers = best
        return normal, offset, residuals, inliers

    worst_ground_index = int(np.argmax(residuals_m))
    residual_summary = ", ".join(
        f"G{index + 1}={residual:.3f}m"
        for index, residual in enumerate(residuals_m)
    )
    raise ValueError(
        "Selected ground points do not describe one floor plane: maximum "
        f"residual is at G{worst_ground_index + 1} "
        f"({residuals_m[worst_ground_index]:.3f} m), allowed "
        f"{threshold:.3f} m. Select dispersed points only on the visible "
        f"floor. Residuals: {residual_summary}."
    )


@dataclass(frozen=True)
class ManualScenePlacement:
    """Measurements used to place an avatar in an unrelated 3DGS scene."""

    known_point_a_g: np.ndarray
    known_point_b_g: np.ndarray
    known_distance_m: float
    ground_points_g: np.ndarray
    up_reference_g: np.ndarray
    spawn_point_g: np.ndarray
    forward_point_g: np.ndarray
    avatar_anchor_w_m: np.ndarray
    known_length_direction: str = "any"
    description: str = ""

    def __post_init__(self) -> None:
        point_a = _finite_vector(self.known_point_a_g, name="known_point_a_g")
        point_b = _finite_vector(self.known_point_b_g, name="known_point_b_g")
        ground = _finite_points(
            self.ground_points_g,
            name="ground_points_g",
            minimum_count=3,
        )
        up_reference = _finite_vector(
            self.up_reference_g,
            name="up_reference_g",
        )
        spawn = _finite_vector(self.spawn_point_g, name="spawn_point_g")
        forward = _finite_vector(self.forward_point_g, name="forward_point_g")
        anchor = _finite_vector(
            self.avatar_anchor_w_m,
            name="avatar_anchor_w_m",
        )
        distance = float(self.known_distance_m)
        if not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("known_distance_m must be finite and positive.")
        if float(np.linalg.norm(point_b - point_a)) <= _EPSILON:
            raise ValueError("Known-length endpoints must be distinct.")
        direction = str(self.known_length_direction).lower()
        if direction not in _KNOWN_LENGTH_DIRECTIONS:
            raise ValueError(
                "known_length_direction must be 'any', 'vertical', or "
                "'horizontal'."
            )
        if not isinstance(self.description, str):
            raise ValueError("description must be a string.")
        object.__setattr__(self, "known_point_a_g", point_a)
        object.__setattr__(self, "known_point_b_g", point_b)
        object.__setattr__(self, "known_distance_m", distance)
        object.__setattr__(self, "ground_points_g", ground)
        object.__setattr__(self, "up_reference_g", up_reference)
        object.__setattr__(self, "spawn_point_g", spawn)
        object.__setattr__(self, "forward_point_g", forward)
        object.__setattr__(self, "avatar_anchor_w_m", anchor)
        object.__setattr__(self, "known_length_direction", direction)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ManualScenePlacement":
        if not isinstance(payload, Mapping):
            raise ValueError("Manual scene placement must be a JSON object.")
        known = payload.get("known_length")
        if not isinstance(known, Mapping):
            raise ValueError("Placement requires a known_length object.")
        return cls(
            known_point_a_g=np.asarray(known["point_a_g"]),
            known_point_b_g=np.asarray(known["point_b_g"]),
            known_distance_m=float(known["distance_m"]),
            ground_points_g=np.asarray(payload["ground_points_g"]),
            up_reference_g=np.asarray(payload["up_reference_g"]),
            spawn_point_g=np.asarray(payload["spawn_point_g"]),
            forward_point_g=np.asarray(payload["forward_point_g"]),
            avatar_anchor_w_m=np.asarray(payload["avatar_anchor_w_m"]),
            known_length_direction=str(known.get("direction", "any")),
            description=str(payload.get("description", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "known_length": {
                "point_a_g": self.known_point_a_g.tolist(),
                "point_b_g": self.known_point_b_g.tolist(),
                "distance_m": self.known_distance_m,
                "direction": self.known_length_direction,
            },
            "ground_points_g": self.ground_points_g.tolist(),
            "up_reference_g": self.up_reference_g.tolist(),
            "spawn_point_g": self.spawn_point_g.tolist(),
            "forward_point_g": self.forward_point_g.tolist(),
            "avatar_anchor_w_m": self.avatar_anchor_w_m.tolist(),
        }


@dataclass(frozen=True)
class SceneAlignment:
    """Validated similarity transform from metric avatar world to 3DGS."""

    scale_g_per_m: float
    rotation_g_from_w: np.ndarray
    translation_g_from_w: np.ndarray
    ground_normal_g: np.ndarray
    ground_offset_g: float
    spawn_point_g: np.ndarray
    avatar_anchor_w_m: np.ndarray
    method: str = "external_scene_manual_placement"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        scale = float(self.scale_g_per_m)
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("scale_g_per_m must be finite and positive.")
        rotation = _proper_rotation(
            self.rotation_g_from_w,
            name="rotation_g_from_w",
        )
        translation = _finite_vector(
            self.translation_g_from_w,
            name="translation_g_from_w",
        )
        normal = _unit_vector(self.ground_normal_g, name="ground_normal_g")
        offset = float(self.ground_offset_g)
        if not math.isfinite(offset):
            raise ValueError("ground_offset_g must be finite.")
        spawn = _finite_vector(self.spawn_point_g, name="spawn_point_g")
        anchor = _finite_vector(
            self.avatar_anchor_w_m,
            name="avatar_anchor_w_m",
        )
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string.")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary.")
        if not np.allclose(rotation[:, 2], normal, atol=1e-6):
            raise ValueError(
                "rotation_g_from_w must map W +Z to ground_normal_g."
            )
        if abs(float(np.dot(normal, spawn) + offset)) > 1e-5:
            raise ValueError("spawn_point_g must lie on the saved ground plane.")
        mapped_anchor = scale * (rotation @ anchor) + translation
        if not np.allclose(mapped_anchor, spawn, atol=1e-6):
            raise ValueError(
                "translation_g_from_w must map avatar_anchor_w_m to spawn_point_g."
            )
        object.__setattr__(self, "scale_g_per_m", scale)
        object.__setattr__(self, "rotation_g_from_w", rotation)
        object.__setattr__(self, "translation_g_from_w", translation)
        object.__setattr__(self, "ground_normal_g", normal)
        object.__setattr__(self, "ground_offset_g", offset)
        object.__setattr__(self, "spawn_point_g", spawn)
        object.__setattr__(self, "avatar_anchor_w_m", anchor)

    @property
    def transform_g_from_w(self) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.scale_g_per_m * self.rotation_g_from_w
        transform[:3, 3] = self.translation_g_from_w
        return transform

    @property
    def forward_g(self) -> np.ndarray:
        return self.rotation_g_from_w[:, 1].copy()

    @property
    def right_g(self) -> np.ndarray:
        return self.rotation_g_from_w[:, 0].copy()

    def transform_points_w_to_g(self, points_w_m: Any) -> np.ndarray:
        points = np.asarray(points_w_m, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("Avatar points must end in XYZ.")
        finite = np.isfinite(points).all(axis=-1)
        result = np.full(points.shape, np.nan, dtype=np.float64)
        if np.any(finite):
            result[finite] = (
                self.scale_g_per_m
                * (points[finite] @ self.rotation_g_from_w.T)
                + self.translation_g_from_w
            )
        return result

    def transform_points_g_to_w(self, points_g: Any) -> np.ndarray:
        points = np.asarray(points_g, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("Scene points must end in XYZ.")
        finite = np.isfinite(points).all(axis=-1)
        result = np.full(points.shape, np.nan, dtype=np.float64)
        if np.any(finite):
            result[finite] = (
                (points[finite] - self.translation_g_from_w)
                @ self.rotation_g_from_w
                / self.scale_g_per_m
            )
        return result

    def transform_directions_w_to_g(self, directions_w: Any) -> np.ndarray:
        directions = np.asarray(directions_w, dtype=np.float64)
        if directions.shape[-1] != 3 or not np.isfinite(directions).all():
            raise ValueError("Directions must be finite and end in XYZ.")
        return directions @ self.rotation_g_from_w.T

    def camera_to_world_w_to_g(self, camera_to_world_w: Any) -> np.ndarray:
        """Move a rigid camera pose into G without scaling its rotation."""

        pose = np.asarray(camera_to_world_w, dtype=np.float64)
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError("camera_to_world_w must be a finite 4x4 matrix.")
        if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("camera_to_world_w must be homogeneous.")
        source_rotation = _proper_rotation(
            pose[:3, :3],
            name="camera_to_world_w rotation",
        )
        output = np.eye(4, dtype=np.float64)
        output[:3, :3] = self.rotation_g_from_w @ source_rotation
        output[:3, 3] = self.transform_points_w_to_g(pose[:3, 3])
        return output

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "source_coordinate_system": "display_x_right_y_forward_z_up_m",
            "target_coordinate_system": "3dgs_world",
            "method": self.method,
            "scale_g_per_m": self.scale_g_per_m,
            "rotation_g_from_w": self.rotation_g_from_w.tolist(),
            "translation_g_from_w": self.translation_g_from_w.tolist(),
            "transform_g_from_w": self.transform_g_from_w.tolist(),
            "avatar_anchor_w_m": self.avatar_anchor_w_m.tolist(),
            "spawn_point_g": self.spawn_point_g.tolist(),
            "ground_plane_g": {
                "normal": self.ground_normal_g.tolist(),
                "offset": self.ground_offset_g,
                "equation": "normal dot point_g + offset = 0",
            },
            "axes_g": {
                "right": self.right_g.tolist(),
                "forward": self.forward_g.tolist(),
                "up": self.ground_normal_g.tolist(),
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "SceneAlignment":
        if not isinstance(payload, Mapping):
            raise ValueError("Scene alignment must be a JSON object.")
        if int(payload.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported scene alignment schema_version.")
        if payload.get("source_coordinate_system") != (
            "display_x_right_y_forward_z_up_m"
        ):
            raise ValueError("Scene alignment source coordinate system differs.")
        if payload.get("target_coordinate_system") != "3dgs_world":
            raise ValueError("Scene alignment target coordinate system differs.")
        ground = payload.get("ground_plane_g")
        if not isinstance(ground, Mapping):
            raise ValueError("Scene alignment requires ground_plane_g.")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError("Scene alignment metadata must be an object.")
        return cls(
            scale_g_per_m=float(payload["scale_g_per_m"]),
            rotation_g_from_w=np.asarray(payload["rotation_g_from_w"]),
            translation_g_from_w=np.asarray(payload["translation_g_from_w"]),
            ground_normal_g=np.asarray(ground["normal"]),
            ground_offset_g=float(ground["offset"]),
            spawn_point_g=np.asarray(payload["spawn_point_g"]),
            avatar_anchor_w_m=np.asarray(payload["avatar_anchor_w_m"]),
            method=str(payload.get("method", "external_scene_manual_placement")),
            metadata=metadata,
        )

    @classmethod
    def load(cls, path: str | Path) -> "SceneAlignment":
        return cls.from_mapping(load_json_mapping(path))

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def build_manual_scene_alignment(
    placement: ManualScenePlacement,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> SceneAlignment:
    """Construct a case-one placement from explicit scene measurements."""

    if not isinstance(placement, ManualScenePlacement):
        raise TypeError("placement must be a ManualScenePlacement.")
    scene_length_g = float(
        np.linalg.norm(placement.known_point_b_g - placement.known_point_a_g)
    )
    scale = scene_length_g / placement.known_distance_m
    up, ground_offset, residuals, ground_inliers = fit_ground_plane_robust(
        placement.ground_points_g,
        up_reference_g=placement.up_reference_g,
        scale_g_per_m=scale,
    )
    ground_residuals_m = residuals / scale
    ground_residual_max_m = float(np.max(ground_residuals_m[ground_inliers]))

    length_direction = _unit_vector(
        placement.known_point_b_g - placement.known_point_a_g,
        name="known-length direction",
    )
    normal_component = float(
        np.clip(abs(np.dot(length_direction, up)), 0.0, 1.0)
    )
    if placement.known_length_direction == "vertical":
        known_length_angle_deg = math.degrees(math.acos(normal_component))
    elif placement.known_length_direction == "horizontal":
        known_length_angle_deg = math.degrees(math.asin(normal_component))
    else:
        known_length_angle_deg = None
    if (
        known_length_angle_deg is not None
        and known_length_angle_deg > _MAX_KNOWN_LENGTH_ANGLE_DEG
    ):
        raise ValueError(
            f"Known {placement.known_length_direction} length is "
            f"{known_length_angle_deg:.1f} degrees from its expected direction; "
            f"allowed {_MAX_KNOWN_LENGTH_ANGLE_DEG:.1f} degrees. Check both "
            "endpoints and the selected ground points."
        )

    spawn_signed_distance = float(
        np.dot(up, placement.spawn_point_g) + ground_offset
    )
    spawn_distance_m = abs(spawn_signed_distance) / scale
    if spawn_distance_m > _MAX_PLACEMENT_POINT_DISTANCE_M:
        raise ValueError(
            f"Spawn pick is {spawn_distance_m:.3f} m from the fitted floor; "
            f"allowed {_MAX_PLACEMENT_POINT_DISTANCE_M:.3f} m. Select the "
            "visible floor rather than an object surface."
        )
    spawn = placement.spawn_point_g - spawn_signed_distance * up
    forward_signed_distance = float(
        np.dot(up, placement.forward_point_g) + ground_offset
    )
    forward_distance_m = abs(forward_signed_distance) / scale
    if forward_distance_m > _MAX_PLACEMENT_POINT_DISTANCE_M:
        raise ValueError(
            f"Forward pick is {forward_distance_m:.3f} m from the fitted floor; "
            f"allowed {_MAX_PLACEMENT_POINT_DISTANCE_M:.3f} m. Select the "
            "visible floor rather than an object surface."
        )
    forward = placement.forward_point_g - spawn
    forward = forward - float(np.dot(forward, up)) * up
    forward = _unit_vector(forward, name="ground-projected forward direction")
    right = _unit_vector(np.cross(forward, up), name="scene right direction")
    forward = _unit_vector(np.cross(up, right), name="scene forward direction")
    rotation = _proper_rotation(
        np.column_stack((right, forward, up)),
        name="rotation_g_from_w",
    )
    translation = spawn - scale * (rotation @ placement.avatar_anchor_w_m)

    merged_metadata = dict(metadata or {})
    merged_metadata.update(
        {
            "placement": placement.to_dict(),
            "known_length_g": scene_length_g,
            "ground_fit_residual_median_g": float(np.median(residuals)),
            "ground_fit_residual_max_g": float(np.max(residuals)),
            "ground_fit_residual_median_m": float(np.median(residuals) / scale),
            "ground_fit_residual_max_m": ground_residual_max_m,
            "ground_fit_residual_all_max_m": float(
                np.max(ground_residuals_m)
            ),
            "ground_fit_residuals_m": ground_residuals_m.tolist(),
            "ground_point_inliers": ground_inliers.tolist(),
            "spawn_projection_distance_g": spawn_signed_distance,
            "spawn_projection_distance_m": spawn_signed_distance / scale,
            "forward_projection_distance_m": forward_signed_distance / scale,
            "known_length_angle_error_deg": known_length_angle_deg,
            "validation_limits": {
                "ground_residual_max_m": _MAX_GROUND_RESIDUAL_M,
                "placement_point_distance_max_m": (
                    _MAX_PLACEMENT_POINT_DISTANCE_M
                ),
                "known_length_angle_max_deg": _MAX_KNOWN_LENGTH_ANGLE_DEG,
            },
        }
    )
    return SceneAlignment(
        scale_g_per_m=scale,
        rotation_g_from_w=rotation,
        translation_g_from_w=translation,
        ground_normal_g=up,
        ground_offset_g=ground_offset,
        spawn_point_g=spawn,
        avatar_anchor_w_m=placement.avatar_anchor_w_m,
        metadata=merged_metadata,
    )


def first_avatar_ground_anchor(
    cache: SMPLSequenceCache,
    *,
    mode: str = "feet",
) -> np.ndarray:
    """Return a stable W-space ground anchor from the first fitted frame."""

    if not isinstance(cache, SMPLSequenceCache):
        raise TypeError("cache must be an SMPLSequenceCache.")
    present_indices = np.flatnonzero(cache.present)
    if len(present_indices) == 0:
        raise ValueError("SMPL cache contains no fitted frame.")
    index = int(present_indices[0])
    if mode == "feet":
        anchor = np.mean(cache.joints_display_m[index, [10, 11]], axis=0)
    elif mode == "pelvis":
        anchor = cache.joints_display_m[index, 0].astype(np.float64)
    elif mode == "origin":
        return np.zeros(3, dtype=np.float64)
    else:
        raise ValueError("Anchor mode must be 'feet', 'pelvis', or 'origin'.")
    anchor = np.asarray(anchor, dtype=np.float64)
    anchor[2] = 0.0
    return anchor
