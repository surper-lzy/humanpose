"""Pure data preparation for an interactive RGB-D pose sequence viewer."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from rgbd_avatar.data import load_pose_records
from rgbd_avatar.pose import HALPE26_LINKS, HALPE26_NAMES


PoseLayer = Literal["raw", "temporal", "constrained"]
CloudScope = Literal["bbox", "full", "none"]

# Camera coordinates are +X right, +Y down, +Z forward.  This is a proper
# rotation (determinant +1) used only by the viewer:
#
#   display +X = camera +X (right)
#   display +Y = camera +Z (forward)
#   display +Z = camera -Y (up)
DISPLAY_FROM_CAMERA = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

RAW_COLOR = np.array([0.95, 0.12, 0.10], dtype=np.float64)
OBSERVED_COLOR = np.array([0.10, 0.85, 0.20], dtype=np.float64)
PREDICTED_COLOR = np.array([1.00, 0.55, 0.05], dtype=np.float64)
CORRECTED_COLOR = np.array([0.90, 0.05, 1.00], dtype=np.float64)


@dataclass(frozen=True)
class PoseDisplayData:
    """One Halpe26 pose layer with observation provenance."""

    requested_layer: PoseLayer
    resolved_layer: PoseLayer
    joints_camera_m: np.ndarray
    confidence: np.ndarray
    usable: np.ndarray
    observed: np.ndarray
    predicted: np.ndarray
    corrected: np.ndarray


@dataclass(frozen=True)
class SkeletonDisplayArrays:
    """Compact, finite Open3D-ready skeleton arrays."""

    points: np.ndarray
    point_colors: np.ndarray
    lines: np.ndarray
    line_colors: np.ndarray
    original_joint_indices: np.ndarray


@dataclass(frozen=True)
class CloudDisplayArrays:
    """Finite colored point-cloud arrays in display coordinates."""

    points: np.ndarray
    colors: np.ndarray
    resolved_scope: CloudScope


@dataclass(frozen=True)
class GroundGridDisplayArrays:
    """Metric ground-reference grid in viewer coordinates."""

    points: np.ndarray
    lines: np.ndarray
    colors: np.ndarray
    floor_height_m: float
    center_xy_m: np.ndarray


def _empty_pose(layer: PoseLayer, resolved_layer: PoseLayer) -> PoseDisplayData:
    count = len(HALPE26_NAMES)
    return PoseDisplayData(
        requested_layer=layer,
        resolved_layer=resolved_layer,
        joints_camera_m=np.full((count, 3), np.nan, dtype=np.float64),
        confidence=np.zeros(count, dtype=np.float64),
        usable=np.zeros(count, dtype=bool),
        observed=np.zeros(count, dtype=bool),
        predicted=np.zeros(count, dtype=bool),
        corrected=np.zeros(count, dtype=bool),
    )


def parse_pose_layer(
    record: dict[str, Any],
    layer: PoseLayer,
) -> PoseDisplayData:
    """Extract raw, temporal, or constrained joints from one frame record.

    When the constraint stage was disabled, ``constrained`` falls back to the
    temporal layer.  A missing raw detection produces an empty pose instead of
    dropping the frame, so point-cloud playback remains continuous.
    """

    if layer not in ("raw", "temporal", "constrained"):
        raise ValueError(f"Unsupported pose layer: {layer!r}")

    payload_key = {
        "raw": "pose3d_raw",
        "temporal": "pose3d_temporal",
        "constrained": "pose3d_constrained",
    }[layer]
    payload = record.get(payload_key)
    resolved_layer: PoseLayer = layer
    if layer == "constrained" and payload is None:
        payload = record.get("pose3d_temporal")
        resolved_layer = "temporal"
    if payload is None:
        return _empty_pose(layer, resolved_layer)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Frame {record.get('frame_index')} has a non-object "
            f"{payload_key}."
        )

    joints_payload = payload.get("joints")
    if not isinstance(joints_payload, list) or len(joints_payload) != len(
        HALPE26_NAMES
    ):
        raise ValueError(
            f"Frame {record.get('frame_index')} layer {resolved_layer} must "
            f"contain {len(HALPE26_NAMES)} joints."
        )

    count = len(HALPE26_NAMES)
    joints_m = np.full((count, 3), np.nan, dtype=np.float64)
    confidence = np.zeros(count, dtype=np.float64)
    usable = np.zeros(count, dtype=bool)
    observed = np.zeros(count, dtype=bool)
    predicted = np.zeros(count, dtype=bool)
    corrected = np.zeros(count, dtype=bool)

    for index, (name, joint) in enumerate(
        zip(HALPE26_NAMES, joints_payload)
    ):
        if not isinstance(joint, dict):
            raise ValueError(
                f"Frame {record.get('frame_index')} joint {index} is not "
                "an object."
            )
        if joint.get("id") != index or joint.get("name") != name:
            raise ValueError(
                f"Frame {record.get('frame_index')} has an invalid Halpe26 "
                f"joint at index {index}."
            )

        xyz = joint.get("xyz_m")
        finite_xyz = False
        if isinstance(xyz, (list, tuple)) and len(xyz) == 3:
            point = np.asarray(xyz, dtype=np.float64)
            finite_xyz = bool(np.isfinite(point).all())
            if finite_xyz:
                joints_m[index] = point

        score = float(joint.get("confidence", 0.0))
        confidence[index] = score if math.isfinite(score) else 0.0
        if resolved_layer == "raw":
            is_usable = bool(joint.get("valid", False)) and finite_xyz
            usable[index] = is_usable
            observed[index] = is_usable
        else:
            is_usable = bool(joint.get("usable", False)) and finite_xyz
            is_observed = bool(joint.get("observed", False))
            is_predicted = bool(joint.get("predicted", False))
            if is_observed and is_predicted:
                raise ValueError(
                    f"Frame {record.get('frame_index')} joint {index} cannot "
                    "be both observed and predicted."
                )
            usable[index] = is_usable
            if is_usable:
                # Old or manually created records may omit provenance.  Treat
                # an otherwise usable joint as observed.
                observed[index] = is_observed or not is_predicted
                predicted[index] = is_predicted
                corrected[index] = bool(joint.get("corrected", False))

    joints_m[~usable] = np.nan
    return PoseDisplayData(
        requested_layer=layer,
        resolved_layer=resolved_layer,
        joints_camera_m=joints_m,
        confidence=confidence,
        usable=usable,
        observed=observed,
        predicted=predicted,
        corrected=corrected,
    )


def camera_to_display(points_xyz: np.ndarray) -> np.ndarray:
    """Rotate camera XYZ to viewer XYZ without modifying the input."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}.")
    return points @ DISPLAY_FROM_CAMERA.T


def transform_camera_points(
    points_xyz: np.ndarray,
    camera_to_display_transform: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the default view rotation or a supplied rigid 4x4 transform."""

    points = np.asarray(points_xyz, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 points, got {points.shape}.")
    if camera_to_display_transform is None:
        return camera_to_display(points)
    transform = np.asarray(
        camera_to_display_transform,
        dtype=np.float64,
    )
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(
            "camera_to_display_transform must be finite shape (4, 4)."
        )
    rotation = transform[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-7):
        raise ValueError("Display transform rotation must be orthonormal.")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError("Display transform rotation must be proper.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("Display transform must have a rigid last row.")
    return points @ rotation.T + transform[:3, 3]


def _joint_color(pose: PoseDisplayData, index: int) -> np.ndarray:
    if pose.corrected[index]:
        return CORRECTED_COLOR
    if pose.predicted[index]:
        return PREDICTED_COLOR
    if pose.resolved_layer == "raw":
        return RAW_COLOR
    return OBSERVED_COLOR


def build_skeleton_display_arrays(
    pose: PoseDisplayData,
    links: Sequence[tuple[int, int]] = HALPE26_LINKS,
    *,
    camera_to_display_transform: np.ndarray | None = None,
) -> SkeletonDisplayArrays:
    """Drop missing joints and remap Halpe26 links to compact indices."""

    finite = np.isfinite(pose.joints_camera_m).all(axis=1)
    keep = pose.usable & finite
    original_indices = np.flatnonzero(keep)
    compact_index = np.full(len(HALPE26_NAMES), -1, dtype=np.int64)
    compact_index[original_indices] = np.arange(len(original_indices))

    camera_points = pose.joints_camera_m[original_indices]
    points = transform_camera_points(
        camera_points,
        camera_to_display_transform,
    )
    point_colors = np.asarray(
        [_joint_color(pose, int(index)) for index in original_indices],
        dtype=np.float64,
    ).reshape(-1, 3)

    compact_lines: list[tuple[int, int]] = []
    line_colors: list[np.ndarray] = []
    for start, end in links:
        if not keep[start] or not keep[end]:
            continue
        compact_lines.append(
            (int(compact_index[start]), int(compact_index[end]))
        )
        if pose.corrected[start] or pose.corrected[end]:
            color = CORRECTED_COLOR
        elif pose.predicted[start] or pose.predicted[end]:
            color = PREDICTED_COLOR
        elif pose.resolved_layer == "raw":
            color = RAW_COLOR
        else:
            color = OBSERVED_COLOR
        line_colors.append(color)

    return SkeletonDisplayArrays(
        points=points,
        point_colors=point_colors,
        lines=np.asarray(compact_lines, dtype=np.int32).reshape(-1, 2),
        line_colors=np.asarray(line_colors, dtype=np.float64).reshape(-1, 3),
        original_joint_indices=original_indices.astype(np.int32),
    )


def build_ground_grid_display_arrays(
    records: Sequence[dict[str, Any]],
    *,
    layer: PoseLayer = "constrained",
    spacing_m: float = 0.25,
    margin_m: float = 0.50,
    minimum_span_m: float = 2.0,
    camera_to_display_transform: np.ndarray | None = None,
    floor_height_m: float | None = None,
) -> GroundGridDisplayArrays:
    """Build a fixed grid around a sequence's metric skeleton trajectory."""

    if not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ValueError("spacing_m must be finite and positive.")
    if not math.isfinite(margin_m) or margin_m < 0:
        raise ValueError("margin_m must be finite and non-negative.")
    if not math.isfinite(minimum_span_m) or minimum_span_m <= 0:
        raise ValueError("minimum_span_m must be finite and positive.")

    all_points: list[np.ndarray] = []
    frame_floor_candidates: list[float] = []
    foot_ids = np.asarray((20, 21, 22, 23, 24, 25), dtype=np.int64)
    for record in records:
        pose = parse_pose_layer(record, layer)
        valid_ids = np.flatnonzero(
            pose.usable & np.isfinite(pose.joints_camera_m).all(axis=1)
        )
        if valid_ids.size == 0:
            continue
        display_points = transform_camera_points(
            pose.joints_camera_m[valid_ids],
            camera_to_display_transform,
        )
        all_points.append(display_points)
        valid_foot_ids = foot_ids[pose.usable[foot_ids]]
        if valid_foot_ids.size:
            foot_points = transform_camera_points(
                pose.joints_camera_m[valid_foot_ids],
                camera_to_display_transform,
            )
            finite_feet = foot_points[
                np.isfinite(foot_points).all(axis=1)
            ]
            if finite_feet.size:
                frame_floor_candidates.append(
                    float(np.min(finite_feet[:, 2]))
                )

    if all_points:
        trajectory = np.concatenate(all_points, axis=0)
        x_low, x_high = np.percentile(trajectory[:, 0], [2.0, 98.0])
        y_low, y_high = np.percentile(trajectory[:, 1], [2.0, 98.0])
        fallback_floor = float(np.percentile(trajectory[:, 2], 2.0))
    else:
        x_low = y_low = -minimum_span_m / 2.0
        x_high = y_high = minimum_span_m / 2.0
        fallback_floor = 0.0

    if floor_height_m is not None and not math.isfinite(floor_height_m):
        raise ValueError("floor_height_m must be finite when provided.")
    floor_height = (
        float(floor_height_m)
        if floor_height_m is not None
        else (
            float(np.median(frame_floor_candidates))
            if frame_floor_candidates
            else fallback_floor
        )
    )
    center_x = float((x_low + x_high) / 2.0)
    center_y = float((y_low + y_high) / 2.0)
    half_x = max(
        minimum_span_m / 2.0,
        float((x_high - x_low) / 2.0 + margin_m),
    )
    half_y = max(
        minimum_span_m / 2.0,
        float((y_high - y_low) / 2.0 + margin_m),
    )
    x_start = spacing_m * math.floor((center_x - half_x) / spacing_m)
    x_end = spacing_m * math.ceil((center_x + half_x) / spacing_m)
    y_start = spacing_m * math.floor((center_y - half_y) / spacing_m)
    y_end = spacing_m * math.ceil((center_y + half_y) / spacing_m)
    x_values = np.arange(
        x_start,
        x_end + spacing_m * 0.5,
        spacing_m,
        dtype=np.float64,
    )
    y_values = np.arange(
        y_start,
        y_end + spacing_m * 0.5,
        spacing_m,
        dtype=np.float64,
    )

    points: list[tuple[float, float, float]] = []
    lines: list[tuple[int, int]] = []
    colors: list[tuple[float, float, float]] = []
    minor_color = (0.24, 0.27, 0.31)
    major_color = (0.42, 0.46, 0.52)
    for x in x_values:
        start = len(points)
        points.extend(
            (
                (float(x), float(y_start), floor_height),
                (float(x), float(y_end), floor_height),
            )
        )
        lines.append((start, start + 1))
        colors.append(
            major_color
            if abs(x / spacing_m - round(x / spacing_m)) < 1e-8
            and round(x / spacing_m) % 4 == 0
            else minor_color
        )
    for y in y_values:
        start = len(points)
        points.extend(
            (
                (float(x_start), float(y), floor_height),
                (float(x_end), float(y), floor_height),
            )
        )
        lines.append((start, start + 1))
        colors.append(
            major_color
            if abs(y / spacing_m - round(y / spacing_m)) < 1e-8
            and round(y / spacing_m) % 4 == 0
            else minor_color
        )

    return GroundGridDisplayArrays(
        points=np.asarray(points, dtype=np.float64).reshape(-1, 3),
        lines=np.asarray(lines, dtype=np.int32).reshape(-1, 2),
        colors=np.asarray(colors, dtype=np.float64).reshape(-1, 3),
        floor_height_m=floor_height,
        center_xy_m=np.asarray((center_x, center_y), dtype=np.float64),
    )


def empty_cloud_display_arrays() -> CloudDisplayArrays:
    """Return a valid cloud payload without touching RGB-D source files."""

    return CloudDisplayArrays(
        points=np.empty((0, 3), dtype=np.float64),
        colors=np.empty((0, 3), dtype=np.float64),
        resolved_scope="none",
    )


def _crop_bounds(
    height: int,
    width: int,
    bbox_xyxy: np.ndarray | None,
    margin_px: int,
    scope: CloudScope,
) -> tuple[int, int, int, int, CloudScope]:
    if scope == "full" or bbox_xyxy is None:
        return 0, 0, width, height, "full"
    bbox = np.asarray(bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return 0, 0, width, height, "full"
    x1, y1, x2, y2 = bbox
    left = max(0, int(math.floor(x1)) - margin_px)
    top = max(0, int(math.floor(y1)) - margin_px)
    right = min(width, int(math.ceil(x2)) + margin_px + 1)
    bottom = min(height, int(math.ceil(y2)) + margin_px + 1)
    if left >= right or top >= bottom:
        return 0, 0, width, height, "full"
    return left, top, right, bottom, "bbox"


def build_cloud_display_arrays(
    organized_points_camera_m: np.ndarray,
    rgb: np.ndarray,
    *,
    stride: int = 3,
    scope: CloudScope = "bbox",
    bbox_xyxy: np.ndarray | None = None,
    bbox_margin_px: int = 30,
    camera_to_display_transform: np.ndarray | None = None,
) -> CloudDisplayArrays:
    """Crop, decimate, color, and rotate an organized point cloud."""

    points = np.asarray(organized_points_camera_m)
    colors = np.asarray(rgb)
    if points.ndim != 3 or points.shape[2] != 3:
        raise ValueError(
            f"Expected an HxWx3 organized point cloud, got {points.shape}."
        )
    if colors.shape != points.shape:
        raise ValueError(
            "RGB and organized point-cloud shapes must match, got "
            f"{colors.shape} and {points.shape}."
        )
    if stride <= 0:
        raise ValueError("Point-cloud stride must be positive.")
    if bbox_margin_px < 0:
        raise ValueError("bbox_margin_px must be non-negative.")
    if scope not in ("bbox", "full"):
        raise ValueError(f"Unsupported cloud scope: {scope!r}")

    height, width = points.shape[:2]
    left, top, right, bottom, resolved_scope = _crop_bounds(
        height,
        width,
        bbox_xyxy,
        bbox_margin_px,
        scope,
    )
    sampled_points = points[top:bottom:stride, left:right:stride].reshape(
        -1, 3
    )
    sampled_colors = colors[
        top:bottom:stride, left:right:stride
    ].reshape(-1, 3)
    valid = np.isfinite(sampled_points).all(axis=1)
    sampled_points = sampled_points[valid]
    sampled_colors = sampled_colors[valid].astype(np.float64) / 255.0
    sampled_colors = np.clip(sampled_colors, 0.0, 1.0)

    return CloudDisplayArrays(
        points=transform_camera_points(
            sampled_points,
            camera_to_display_transform,
        ),
        colors=sampled_colors,
        resolved_scope=resolved_scope,
    )


def pose_bbox(record: dict[str, Any]) -> np.ndarray | None:
    """Return a finite 2D person box from a frame record, when available."""

    pose2d = record.get("pose2d")
    if not isinstance(pose2d, dict):
        return None
    bbox = np.asarray(pose2d.get("bbox_xyxy"), dtype=np.float64)
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        return None
    return bbox


def propagate_segment_bboxes(
    records: Sequence[dict[str, Any]],
    max_age_s: float = 1.1,
) -> list[np.ndarray | None]:
    """Fill only short detection gaps without leaking across segments."""

    if not math.isfinite(max_age_s) or max_age_s < 0:
        raise ValueError("max_age_s must be finite and non-negative.")
    result: list[np.ndarray | None] = [None] * len(records)
    segment_indices: dict[int, list[int]] = {}
    for index, record in enumerate(records):
        segment_indices.setdefault(int(record["segment_id"]), []).append(index)

    for indices in segment_indices.values():
        last_bbox: np.ndarray | None = None
        last_time_s: float | None = None
        for index in indices:
            current = pose_bbox(records[index])
            current_time_s = float(records[index]["relative_time_s"])
            if current is not None:
                last_bbox = current
                last_time_s = current_time_s
            if (
                last_bbox is not None
                and last_time_s is not None
                and current_time_s - last_time_s <= max_age_s
            ):
                result[index] = last_bbox.copy()
        next_bbox: np.ndarray | None = None
        next_time_s: float | None = None
        for index in reversed(indices):
            current = pose_bbox(records[index])
            current_time_s = float(records[index]["relative_time_s"])
            if current is not None:
                next_bbox = current
                next_time_s = current_time_s
            if (
                result[index] is None
                and next_bbox is not None
                and next_time_s is not None
                and next_time_s - current_time_s <= max_age_s
            ):
                result[index] = next_bbox.copy()
    return result


def resolve_frame_sources(
    record: dict[str, Any],
    jsonl_path: str | Path,
    sequence_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """Resolve RGB/depth paths, optionally overriding a moved sequence root."""

    if sequence_dir is not None:
        root = Path(sequence_dir).expanduser().resolve()
        timestamp = str(record["timestamp_raw"])
        return (
            root / f"{timestamp}_r.png",
            root / f"{timestamp}_d.pgm",
        )

    sources = record.get("sources")
    if not isinstance(sources, dict):
        raise ValueError(
            f"Frame {record.get('frame_index')} has no sources object."
        )
    base = Path(jsonl_path).expanduser().resolve().parent
    resolved: list[Path] = []
    for key in ("rgb", "depth"):
        value = sources.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(
                f"Frame {record.get('frame_index')} has no source {key!r}."
            )
        path = Path(value).expanduser()
        resolved.append(path.resolve() if path.is_absolute() else base / path)
    return resolved[0], resolved[1]


def playback_delay_s(
    current: dict[str, Any],
    following: dict[str, Any],
    *,
    timing: Literal["source", "fixed"] = "source",
    playback_speed: float = 1.0,
    fixed_fps: float = 2.0,
) -> float:
    """Return the delay before the next frame, respecting segment resets."""

    if playback_speed <= 0 or not math.isfinite(playback_speed):
        raise ValueError("playback_speed must be finite and positive.")
    if fixed_fps <= 0 or not math.isfinite(fixed_fps):
        raise ValueError("fixed_fps must be finite and positive.")
    if timing not in ("source", "fixed"):
        raise ValueError(f"Unsupported playback timing: {timing!r}")

    fallback = 1.0 / fixed_fps
    if timing == "fixed":
        return fallback / playback_speed
    if int(following["segment_id"]) != int(current["segment_id"]):
        return fallback / playback_speed
    delta = float(following["relative_time_s"]) - float(
        current["relative_time_s"]
    )
    if not math.isfinite(delta) or delta <= 0:
        delta = fallback
    return delta / playback_speed
