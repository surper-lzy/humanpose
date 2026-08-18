"""Robust ground-plane estimation and camera-to-gravity alignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping

import numpy as np


CAMERA_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)
CAMERA_RIGHT = np.array([1.0, 0.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class GroundPlaneConfig:
    """Sampling and RANSAC thresholds for an indoor RGB-D floor."""

    lower_image_fraction: float = 0.52
    pixel_stride: int = 6
    bbox_exclusion_margin_px: int = 20
    max_points_per_frame: int = 4000
    max_total_points: int = 50000
    min_depth_m: float = 0.3
    max_depth_m: float = 6.0
    ransac_iterations: int = 800
    inlier_distance_m: float = 0.025
    max_tilt_from_camera_up_deg: float = 45.0
    min_camera_height_m: float = 0.5
    max_camera_height_m: float = 3.0
    min_inlier_count: int = 500
    random_seed: int = 20260731

    def __post_init__(self) -> None:
        if not 0 <= self.lower_image_fraction < 1:
            raise ValueError("lower_image_fraction must be in [0, 1).")
        for name, value in (
            ("pixel_stride", self.pixel_stride),
            ("max_points_per_frame", self.max_points_per_frame),
            ("max_total_points", self.max_total_points),
            ("ransac_iterations", self.ransac_iterations),
            ("min_inlier_count", self.min_inlier_count),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if self.bbox_exclusion_margin_px < 0:
            raise ValueError(
                "bbox_exclusion_margin_px must be non-negative."
            )
        if not 0 < self.min_depth_m < self.max_depth_m:
            raise ValueError("Ground depth limits are invalid.")
        if self.inlier_distance_m <= 0:
            raise ValueError("inlier_distance_m must be positive.")
        if not 0 < self.max_tilt_from_camera_up_deg < 90:
            raise ValueError(
                "max_tilt_from_camera_up_deg must be in (0, 90)."
            )
        if not 0 < self.min_camera_height_m < self.max_camera_height_m:
            raise ValueError("Camera-height limits are invalid.")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "GroundPlaneConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("ground must be a YAML mapping.")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                "Unknown ground configuration keys: "
                + ", ".join(unknown)
            )
        return cls(**dict(values))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundPlaneEstimate:
    """Plane ``normal_camera · point + offset_m = 0`` with upward normal."""

    normal_camera: np.ndarray
    offset_m: float
    inlier_count: int
    candidate_count: int
    inlier_ratio: float
    residual_median_m: float
    residual_p95_m: float
    residual_rms_m: float
    tilt_from_camera_up_deg: float
    source_frame_count: int = 0

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal_camera, dtype=np.float64)
        if normal.shape != (3,) or not np.isfinite(normal).all():
            raise ValueError("Ground normal must be finite shape (3,).")
        norm = float(np.linalg.norm(normal))
        if norm <= 1e-12:
            raise ValueError("Ground normal must be non-zero.")
        normal = normal / norm
        offset = float(self.offset_m) / norm
        if float(np.dot(normal, CAMERA_UP)) < 0:
            normal = -normal
            offset = -offset
        if not math.isfinite(offset) or offset <= 0:
            raise ValueError(
                "Ground offset must place the camera above the plane."
            )
        if self.inlier_count < 0 or self.candidate_count < 0:
            raise ValueError("Ground point counts must be non-negative.")
        object.__setattr__(self, "normal_camera", normal)
        object.__setattr__(self, "offset_m", offset)

    @property
    def camera_height_m(self) -> float:
        return self.offset_m

    def signed_distance_m(self, points_camera_m: np.ndarray) -> np.ndarray:
        points = np.asarray(points_camera_m, dtype=np.float64)
        if points.shape[-1] != 3:
            raise ValueError("Ground distances require points ending in XYZ.")
        return points @ self.normal_camera + self.offset_m

    def camera_to_ground_transform(self) -> np.ndarray:
        """Return a proper 4x4 transform with ground at world ``Z=0``."""

        world_z_camera = self.normal_camera
        world_x_camera = CAMERA_RIGHT - (
            np.dot(CAMERA_RIGHT, world_z_camera) * world_z_camera
        )
        x_norm = float(np.linalg.norm(world_x_camera))
        if x_norm <= 1e-8:
            raise ValueError("Ground normal is degenerate with camera right.")
        world_x_camera /= x_norm
        world_y_camera = np.cross(
            world_z_camera,
            world_x_camera,
        )
        world_y_camera /= np.linalg.norm(world_y_camera)
        rotation = np.stack(
            (world_x_camera, world_y_camera, world_z_camera),
            axis=0,
        )
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8):
            raise ValueError("Ground alignment must be a proper rotation.")
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = rotation
        transform[2, 3] = self.offset_m
        return transform

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "coordinate_system": "right_handed_camera_xyz_m",
            "plane_equation": "normal_camera dot point_camera_m + offset_m = 0",
            "normal_camera": self.normal_camera.tolist(),
            "offset_m": self.offset_m,
            "camera_height_m": self.camera_height_m,
            "inlier_count": self.inlier_count,
            "candidate_count": self.candidate_count,
            "inlier_ratio": self.inlier_ratio,
            "residual_median_m": self.residual_median_m,
            "residual_p95_m": self.residual_p95_m,
            "residual_rms_m": self.residual_rms_m,
            "tilt_from_camera_up_deg": self.tilt_from_camera_up_deg,
            "source_frame_count": self.source_frame_count,
            "camera_to_ground_transform": (
                self.camera_to_ground_transform().tolist()
            ),
        }

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any],
    ) -> "GroundPlaneEstimate":
        return cls(
            normal_camera=np.asarray(values["normal_camera"]),
            offset_m=float(values["offset_m"]),
            inlier_count=int(values.get("inlier_count", 0)),
            candidate_count=int(values.get("candidate_count", 0)),
            inlier_ratio=float(values.get("inlier_ratio", 0.0)),
            residual_median_m=float(
                values.get("residual_median_m", 0.0)
            ),
            residual_p95_m=float(values.get("residual_p95_m", 0.0)),
            residual_rms_m=float(values.get("residual_rms_m", 0.0)),
            tilt_from_camera_up_deg=float(
                values.get("tilt_from_camera_up_deg", 0.0)
            ),
            source_frame_count=int(values.get("source_frame_count", 0)),
        )


def sample_ground_candidates(
    organized_points_camera_m: np.ndarray,
    config: GroundPlaneConfig,
    *,
    person_bbox_xyxy: np.ndarray | None = None,
) -> np.ndarray:
    """Sample lower-image points while excluding the detected person."""

    points = np.asarray(organized_points_camera_m, dtype=np.float64)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(
            "Ground sampling expects an HxWx3 organized point cloud."
        )
    height, width = points.shape[:2]
    row_start = int(math.floor(height * config.lower_image_fraction))
    rows = np.arange(row_start, height, config.pixel_stride)
    columns = np.arange(0, width, config.pixel_stride)
    sampled = points[np.ix_(rows, columns)]
    keep = (
        np.isfinite(sampled).all(axis=2)
        & (sampled[:, :, 2] >= config.min_depth_m)
        & (sampled[:, :, 2] <= config.max_depth_m)
    )

    if person_bbox_xyxy is not None:
        bbox = np.asarray(person_bbox_xyxy, dtype=np.float64)
        if bbox.shape != (4,) or not np.isfinite(bbox).all():
            raise ValueError("person_bbox_xyxy must be finite shape (4,).")
        margin = config.bbox_exclusion_margin_px
        x1, y1, x2, y2 = bbox
        inside_x = (
            (columns >= math.floor(x1) - margin)
            & (columns <= math.ceil(x2) + margin)
        )
        inside_y = (
            (rows >= math.floor(y1) - margin)
            & (rows <= math.ceil(y2) + margin)
        )
        keep &= ~(inside_y[:, None] & inside_x[None, :])

    candidates = sampled[keep]
    if len(candidates) > config.max_points_per_frame:
        indices = np.linspace(
            0,
            len(candidates) - 1,
            config.max_points_per_frame,
            dtype=np.int64,
        )
        candidates = candidates[indices]
    return candidates


def _plane_from_three_points(
    points: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    first, second, third = points
    normal = np.cross(second - first, third - first)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-10:
        return None
    normal /= norm
    if float(np.dot(normal, CAMERA_UP)) < 0:
        normal = -normal
    offset = -float(np.dot(normal, first))
    return normal, offset


def _refine_plane(
    points: np.ndarray,
) -> tuple[np.ndarray, float]:
    center = np.mean(points, axis=0)
    _, _, right_vectors = np.linalg.svd(
        points - center,
        full_matrices=False,
    )
    normal = right_vectors[-1]
    normal /= np.linalg.norm(normal)
    if float(np.dot(normal, CAMERA_UP)) < 0:
        normal = -normal
    return normal, -float(np.dot(normal, center))


def fit_ground_plane_ransac(
    candidate_points_camera_m: np.ndarray,
    config: GroundPlaneConfig,
    *,
    source_frame_count: int = 0,
) -> GroundPlaneEstimate:
    """Fit the dominant floor-like plane with a deterministic RANSAC."""

    points = np.asarray(candidate_points_camera_m, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Ground RANSAC expects Nx3 candidate points.")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > config.max_total_points:
        indices = np.linspace(
            0,
            len(points) - 1,
            config.max_total_points,
            dtype=np.int64,
        )
        points = points[indices]
    if len(points) < max(3, config.min_inlier_count):
        raise ValueError(
            f"Insufficient ground candidates: {len(points)} points."
        )

    rng = np.random.default_rng(config.random_seed)
    minimum_up_dot = math.cos(
        math.radians(config.max_tilt_from_camera_up_deg)
    )
    best_inliers: np.ndarray | None = None
    best_key: tuple[int, float] | None = None
    for _ in range(config.ransac_iterations):
        sample_indices = rng.choice(len(points), size=3, replace=False)
        model = _plane_from_three_points(points[sample_indices])
        if model is None:
            continue
        normal, offset = model
        if float(np.dot(normal, CAMERA_UP)) < minimum_up_dot:
            continue
        if not (
            config.min_camera_height_m
            <= offset
            <= config.max_camera_height_m
        ):
            continue
        residuals = np.abs(points @ normal + offset)
        inliers = residuals <= config.inlier_distance_m
        count = int(np.count_nonzero(inliers))
        if count < config.min_inlier_count:
            continue
        median = float(np.median(residuals[inliers]))
        key = (count, -median)
        if best_key is None or key > best_key:
            best_key = key
            best_inliers = inliers

    if best_inliers is None:
        raise RuntimeError(
            "RANSAC found no floor-like plane satisfying the configured "
            "normal, height, and support thresholds."
        )

    normal, offset = _refine_plane(points[best_inliers])
    for _ in range(2):
        residuals = np.abs(points @ normal + offset)
        refined_inliers = residuals <= config.inlier_distance_m
        if np.count_nonzero(refined_inliers) < config.min_inlier_count:
            break
        normal, offset = _refine_plane(points[refined_inliers])
        best_inliers = refined_inliers

    residuals = np.abs(points @ normal + offset)
    best_inliers = residuals <= config.inlier_distance_m
    inlier_residuals = residuals[best_inliers]
    inlier_count = int(np.count_nonzero(best_inliers))
    if inlier_count < config.min_inlier_count:
        raise RuntimeError("Refined ground plane lost required support.")
    tilt = math.degrees(
        math.acos(
            float(np.clip(np.dot(normal, CAMERA_UP), -1.0, 1.0))
        )
    )
    return GroundPlaneEstimate(
        normal_camera=normal,
        offset_m=offset,
        inlier_count=inlier_count,
        candidate_count=len(points),
        inlier_ratio=inlier_count / len(points),
        residual_median_m=float(np.median(inlier_residuals)),
        residual_p95_m=float(np.percentile(inlier_residuals, 95)),
        residual_rms_m=float(
            np.sqrt(np.mean(np.square(inlier_residuals)))
        ),
        tilt_from_camera_up_deg=tilt,
        source_frame_count=source_frame_count,
    )
