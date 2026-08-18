"""Startup calibration for aligned RGB-D intrinsics and application axes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import (
    GroundPlaneConfig,
    GroundPlaneEstimate,
    depth_to_organized_point_cloud,
    fit_ground_plane_ransac,
    sample_ground_candidates,
)

from .extrinsics import ApplicationExtrinsics
from .models import RGBDSource
from .processor import PoseBackend


@dataclass(frozen=True)
class LiveAutoCalibrationConfig:
    """Bounded startup calibration; the accepted transform is then frozen."""

    enabled: bool = False
    sample_frame_count: int = 12
    max_attempt_frame_count: int = 30
    exclude_detected_people: bool = True
    min_inlier_ratio: float = 0.35
    max_residual_p95_m: float = 0.04
    fallback_to_config: bool = True
    output_path: str = "outputs/calibration/live_camera_calibration.json"

    def __post_init__(self) -> None:
        if self.sample_frame_count <= 0:
            raise ValueError("sample_frame_count must be positive.")
        if self.max_attempt_frame_count < self.sample_frame_count:
            raise ValueError(
                "max_attempt_frame_count must be >= sample_frame_count."
            )
        if not 0 < self.min_inlier_ratio <= 1:
            raise ValueError("min_inlier_ratio must be in (0, 1].")
        if self.max_residual_p95_m <= 0:
            raise ValueError("max_residual_p95_m must be positive.")
        if not self.output_path.strip():
            raise ValueError("output_path must not be empty.")

    @classmethod
    def from_mapping(
        cls,
        values: Mapping[str, Any] | None,
    ) -> "LiveAutoCalibrationConfig":
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("live.auto_calibration must be a mapping.")
        known = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(
                "Unknown live auto-calibration keys: " + ", ".join(unknown)
            )
        return cls(**dict(values))


@dataclass(frozen=True)
class LiveCalibrationResult:
    intrinsics: CameraIntrinsics
    ground_plane: GroundPlaneEstimate
    extrinsics: ApplicationExtrinsics
    sampled_frame_count: int
    attempted_frame_count: int


def application_extrinsics_from_ground_plane(
    ground_plane: GroundPlaneEstimate,
    heading_reference: ApplicationExtrinsics,
) -> ApplicationExtrinsics:
    """Level roll/pitch and height while preserving configured scene yaw."""

    world_z_camera = ground_plane.normal_camera
    reference_x_camera = (
        heading_reference.rotation_application_from_camera[0]
    )
    world_x_camera = reference_x_camera - (
        np.dot(reference_x_camera, world_z_camera) * world_z_camera
    )
    x_norm = float(np.linalg.norm(world_x_camera))
    if x_norm <= 1e-8:
        raise ValueError(
            "Configured application X heading is parallel to ground normal."
        )
    world_x_camera /= x_norm
    world_y_camera = np.cross(world_z_camera, world_x_camera)
    world_y_camera /= np.linalg.norm(world_y_camera)
    rotation = np.stack(
        (world_x_camera, world_y_camera, world_z_camera),
        axis=0,
    )

    translation = heading_reference.translation_m.copy()
    # The plane equation is n·p + offset = 0, so t_z=offset maps it to Z=0.
    translation[2] = ground_plane.offset_m
    return ApplicationExtrinsics.from_rotation_translation(
        rotation,
        translation,
    )


def calibrate_live_camera(
    source: RGBDSource,
    backend: PoseBackend,
    *,
    heading_reference: ApplicationExtrinsics,
    config: LiveAutoCalibrationConfig,
    ground_config: GroundPlaneConfig,
    read_timeout_ms: int,
    output_path: Path | None = None,
    source_already_started: bool = False,
) -> LiveCalibrationResult:
    """Read startup frames, fit the floor, and freeze camera-to-app axes."""

    if not source_already_started:
        source.start()
    candidates: list[np.ndarray] = []
    sampled_frames = 0
    attempted_frames = 0
    reference_intrinsics: CameraIntrinsics | None = None

    while (
        attempted_frames < config.max_attempt_frame_count
        and sampled_frames < config.sample_frame_count
    ):
        frame = source.read(timeout_ms=read_timeout_ms)
        attempted_frames += 1
        if reference_intrinsics is None:
            reference_intrinsics = frame.intrinsics
        else:
            _require_matching_intrinsics(reference_intrinsics, frame.intrinsics)

        person_bbox = None
        if config.exclude_detected_people:
            poses = backend.infer(frame.rgb_bgr)
            finite_boxes = [
                pose.bbox_xyxy
                for pose in poses
                if np.isfinite(pose.bbox_xyxy).all()
            ]
            if finite_boxes:
                boxes = np.asarray(finite_boxes, dtype=np.float64)
                person_bbox = np.array(
                    [
                        np.min(boxes[:, 0]),
                        np.min(boxes[:, 1]),
                        np.max(boxes[:, 2]),
                        np.max(boxes[:, 3]),
                    ],
                    dtype=np.float64,
                )

        organized = depth_to_organized_point_cloud(
            frame.depth_m,
            frame.intrinsics,
        )
        frame_candidates = sample_ground_candidates(
            organized,
            ground_config,
            person_bbox_xyxy=person_bbox,
        )
        if len(frame_candidates) == 0:
            continue
        candidates.append(frame_candidates)
        sampled_frames += 1

    if reference_intrinsics is None:
        raise RuntimeError("Auto-calibration received no RGB-D frame.")
    if sampled_frames < config.sample_frame_count:
        raise RuntimeError(
            "Auto-calibration collected only "
            f"{sampled_frames}/{config.sample_frame_count} usable frames."
        )

    ground_plane = fit_ground_plane_ransac(
        np.concatenate(candidates, axis=0),
        ground_config,
        source_frame_count=sampled_frames,
    )
    if ground_plane.inlier_ratio < config.min_inlier_ratio:
        raise RuntimeError(
            "Ground calibration rejected: inlier ratio "
            f"{ground_plane.inlier_ratio:.3f} < {config.min_inlier_ratio:.3f}."
        )
    if ground_plane.residual_p95_m > config.max_residual_p95_m:
        raise RuntimeError(
            "Ground calibration rejected: p95 residual "
            f"{ground_plane.residual_p95_m:.4f} m > "
            f"{config.max_residual_p95_m:.4f} m."
        )

    extrinsics = application_extrinsics_from_ground_plane(
        ground_plane,
        heading_reference,
    )
    result = LiveCalibrationResult(
        intrinsics=reference_intrinsics,
        ground_plane=ground_plane,
        extrinsics=extrinsics,
        sampled_frame_count=sampled_frames,
        attempted_frame_count=attempted_frames,
    )
    if output_path is not None:
        write_live_calibration(output_path, source.source_id, result)
    return result


def write_live_calibration(
    path: Path,
    source_id: str,
    result: LiveCalibrationResult,
) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_id": source_id,
        "intrinsics_source": "camera_sdk_aligned_rgb_stream",
        "intrinsics": asdict(result.intrinsics),
        "ground_plane": result.ground_plane.to_dict(),
        "application_extrinsics": result.extrinsics.to_mapping(),
        "application_from_camera_matrix": result.extrinsics.matrix.tolist(),
        "sampled_frame_count": result.sampled_frame_count,
        "attempted_frame_count": result.attempted_frame_count,
    }
    resolved.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _require_matching_intrinsics(
    reference: CameraIntrinsics,
    current: CameraIntrinsics,
) -> None:
    if (reference.width, reference.height) != (current.width, current.height):
        raise RuntimeError("Camera resolution changed during auto-calibration.")
    reference_values = np.array(
        [reference.fx, reference.fy, reference.cx, reference.cy],
        dtype=np.float64,
    )
    current_values = np.array(
        [current.fx, current.fy, current.cx, current.cy],
        dtype=np.float64,
    )
    if not np.allclose(reference_values, current_values, rtol=0.0, atol=1e-5):
        raise RuntimeError("Camera intrinsics changed during auto-calibration.")
