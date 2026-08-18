#!/usr/bin/env python3
"""Run RTMPose and metric 3D recovery over one offline RGB-D sequence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.data import discover_rgbd_sequence, split_at_time_gaps
from rgbd_avatar.depth import (
    PointCloudRecoveryConfig,
    depth_to_organized_point_cloud,
    load_depth_m,
    recover_pose3d,
    recover_pose3d_from_point_cloud,
)
from rgbd_avatar.io import atomic_write_json, load_yaml_mapping, resolve_path
from rgbd_avatar.pipeline.metrics import (
    duration_statistics,
    segment_metadata,
    value_statistics,
)
from rgbd_avatar.pose import (
    HALPE26_CONSTRAINT_LINKS,
    HALPE26_NAMES,
    Pose2D,
    Pose3D,
    RTMPoseBackend,
    RTMPoseBackendConfig,
)
from rgbd_avatar.pose.visualization import draw_pose
from rgbd_avatar.tracking import (
    BoneConstraintResult,
    BoneLengthAccumulator,
    BoneLengthCalibrator,
    BoneLengthConstraint,
    FramePresenceConfig,
    FramePresenceDecision,
    PersonFramePresenceGate,
    Pose3DTemporalFilter,
    TemporalPose3D,
)
from rgbd_avatar.visualization import draw_pose_depths


LOGGER = logging.getLogger("process_rgbd_sequence")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        type=Path,
        default=PROJECT_ROOT.parent / "data/4",
        help="Directory containing timestamped *_r.png and *_d.pgm files.",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PROJECT_ROOT / "configs/camera.yaml",
    )
    parser.add_argument(
        "--pose-config",
        type=Path,
        default=PROJECT_ROOT / "configs/pose.yaml",
    )
    parser.add_argument(
        "--tracking-config",
        type=Path,
        default=PROJECT_ROOT / "configs/tracking.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to outputs/sequences/<sequence-directory-name>.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--recovery-method",
        choices=("window_median", "pointcloud_cluster"),
        default=None,
        help="Override camera.depth_recovery.method.",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "whole_image"),
        default=None,
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional smoke-test limit applied after manifest validation.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help="Fixed preview FPS; default is the median source cadence.",
    )
    parser.add_argument("--codec", default="mp4v")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """Compatibility wrapper around the shared configuration loader."""

    return load_yaml_mapping(path)


def resolve_project_path(value: str) -> Path:
    return resolve_path(value, relative_to=PROJECT_ROOT)


def dump_json(path: Path, payload: dict[str, Any]) -> None:
    """Compatibility wrapper that writes JSON atomically."""

    atomic_write_json(path, payload)


def create_manifest(
    sequence_dir: Path,
    frames: list,
    segments: list[list],
    reset_gap_s: float,
    camera_config: dict[str, Any],
    frame_presence_config: FramePresenceConfig,
) -> dict[str, Any]:
    segment_info, valid_intervals_s = segment_metadata(segments)
    frame_to_segment = {
        frame.frame_index: segment_id
        for segment_id, segment in enumerate(segments)
        for frame in segment
    }
    median_interval = (
        float(np.median(valid_intervals_s))
        if valid_intervals_s
        else None
    )
    return {
        "schema_version": 1,
        "sequence_id": sequence_dir.name,
        "sequence_directory": str(sequence_dir),
        "pairing_policy": "exact_timestamp_prefix",
        "timestamp_format": "YYYYMMDD_HHMMSSmmm (local/unspecified timezone)",
        "frame_count": len(frames),
        "reset_gap_s": reset_gap_s,
        "frame_presence": frame_presence_config.to_dict(),
        "segment_count": len(segments),
        "camera": {
            "rgb_width": int(camera_config["rgb_width"]),
            "rgb_height": int(camera_config["rgb_height"]),
            "depth_width": int(camera_config["depth_width"]),
            "depth_height": int(camera_config["depth_height"]),
            "depth_scale": float(camera_config["depth_scale"]),
            "min_depth_m": float(camera_config["min_depth_m"]),
            "max_depth_m": float(camera_config["max_depth_m"]),
            "align_depth_to_rgb": bool(
                camera_config["align_depth_to_rgb"]
            ),
            "images_undistorted": bool(
                camera_config["images_undistorted"]
            ),
            "intrinsics": {
                key: float(camera_config["intrinsics"][key])
                for key in ("fx", "fy", "cx", "cy")
            },
            "distortion": camera_config["distortion"],
            "coordinate_system": camera_config["coordinate_system"],
        },
        "source_timing": {
            "median_interval_s_excluding_gaps": median_interval,
            "nominal_fps": (
                1.0 / median_interval
                if median_interval is not None and median_interval > 0
                else None
            ),
        },
        "segments": segment_info,
        "frames": [
            {
                "frame_index": frame.frame_index,
                "segment_id": frame_to_segment[frame.frame_index],
                "timestamp_raw": frame.timestamp_raw,
                "captured_at": frame.captured_at.isoformat(
                    timespec="milliseconds"
                ),
                "relative_time_s": frame.relative_time_s,
                "rgb": str(frame.rgb_path),
                "depth": str(frame.depth_path),
                "amplitude": (
                    str(frame.amplitude_path)
                    if frame.amplitude_path is not None
                    else None
                ),
                "point_cloud": (
                    str(frame.point_cloud_path)
                    if frame.point_cloud_path is not None
                    else None
                ),
            }
            for frame in frames
        ],
    }


def draw_frame_diagnostics(
    image_bgr: np.ndarray,
    pose2d: Pose2D | None,
    pose3d: Pose3D | None,
    temporal_pose: TemporalPose3D,
    constraint_result: BoneConstraintResult | None,
    intrinsics: CameraIntrinsics,
    *,
    status: str,
    sequence_id: str,
    segment_id: int,
    frame_index: int,
    timestamp_raw: str,
    dt_s: float | None,
    inference_ms: float,
    score_threshold: float,
) -> np.ndarray:
    if pose2d is not None and pose3d is not None:
        canvas = draw_pose_depths(
            image_bgr,
            pose2d,
            pose3d,
            score_threshold=score_threshold,
        )
    elif pose2d is not None:
        canvas = draw_pose(
            image_bgr,
            pose2d,
            score_threshold=score_threshold,
        )
    else:
        canvas = image_bgr.copy()

    if constraint_result is not None:
        for index in np.flatnonzero(constraint_result.corrected):
            x_m, y_m, z_m = constraint_result.pose.joints_m[index]
            if not np.isfinite([x_m, y_m, z_m]).all() or z_m <= 0:
                continue
            constrained_pixel = (
                int(round(intrinsics.fx * x_m / z_m + intrinsics.cx)),
                int(round(intrinsics.fy * y_m / z_m + intrinsics.cy)),
            )
            if pose2d is not None:
                source_pixel = tuple(
                    np.rint(pose2d.keypoints[index]).astype(int)
                )
                cv2.line(
                    canvas,
                    source_pixel,
                    constrained_pixel,
                    (255, 0, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.circle(
                canvas,
                constrained_pixel,
                3,
                (255, 0, 255),
                -1,
                cv2.LINE_AA,
            )

    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (canvas.shape[1], 94), (0, 0, 0), -1)
    canvas = cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0.0)
    status_color = (80, 255, 80) if status == "ok" else (0, 190, 255)
    cv2.putText(
        canvas,
        (
            f"seq={sequence_id} seg={segment_id} frame={frame_index} "
            f"status={status}"
        ),
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        status_color,
        2,
        cv2.LINE_AA,
    )
    dt_text = "start" if dt_s is None else f"{dt_s:.3f}s"
    raw_count = int(np.count_nonzero(pose3d.valid)) if pose3d else 0
    cv2.putText(
        canvas,
        (
            f"{timestamp_raw}  dt={dt_text}  infer={inference_ms:.1f}ms  "
            f"3D raw={raw_count}/26 usable={np.count_nonzero(temporal_pose.usable)}/26"
        ),
        (10, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    calibrated_bones = (
        constraint_result.diagnostics["calibrated_bone_count"]
        if constraint_result is not None
        else 0
    )
    corrected_joints = (
        constraint_result.diagnostics["corrected_joint_count"]
        if constraint_result is not None
        else 0
    )
    cv2.putText(
        canvas,
        (
            f"observed={np.count_nonzero(temporal_pose.observed)}  "
            f"predicted={np.count_nonzero(temporal_pose.predicted)}"
        ),
        (10, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"bone prior={calibrated_bones}/14  "
            f"corrected={corrected_joints}  magenta=constraint"
        ),
        (10, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 160, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def open_video_writer(
    output_path: Path,
    codec: str,
    fps: float,
    frame_size: tuple[int, int],
) -> cv2.VideoWriter:
    if len(codec) != 4:
        raise ValueError("--codec must contain exactly four characters.")
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(
            f"OpenCV failed to open video writer for {output_path}"
        )
    return writer


def build_backend(
    pose_config: dict[str, Any],
    args: argparse.Namespace,
) -> RTMPoseBackend:
    return RTMPoseBackend(
        RTMPoseBackendConfig(
            model_config=resolve_project_path(pose_config["model_config"]),
            model_checkpoint=resolve_project_path(
                pose_config["model_checkpoint"]
            ),
            detector=args.detector or pose_config["detector"],
            model_cache_dir=resolve_project_path(
                pose_config["model_cache_dir"]
            ),
            device=args.device or pose_config["device"],
            bbox_threshold=float(pose_config["bbox_threshold"]),
            keypoint_threshold=float(pose_config["keypoint_threshold"]),
            min_valid_keypoints=int(pose_config["min_valid_keypoints"]),
            min_mean_keypoint_score=float(
                pose_config["min_mean_keypoint_score"]
            ),
        )
    )


def build_temporal_filter(
    tracking_config: dict[str, Any],
) -> Pose3DTemporalFilter:
    one_euro = tracking_config["one_euro"]
    return Pose3DTemporalFilter(
        min_cutoff_hz=float(one_euro["min_cutoff_hz"]),
        beta=float(one_euro["beta"]),
        derivative_cutoff_hz=float(
            one_euro["derivative_cutoff_hz"]
        ),
        reset_gap_s=float(tracking_config["reset_gap_s"]),
        max_prediction_s=float(tracking_config["max_prediction_s"]),
        min_observation_confidence=float(
            tracking_config["min_observation_confidence"]
        ),
    )


def build_bone_constraint(
    tracking_config: dict[str, Any],
) -> tuple[BoneLengthCalibrator, BoneLengthConstraint] | None:
    config = tracking_config.get("bone_constraint", {})
    if not bool(config.get("enabled", False)):
        return None
    calibration = config["calibration"]
    solver = config["solver"]
    return (
        BoneLengthCalibrator(
            min_samples_per_bone=int(
                calibration["min_samples_per_bone"]
            ),
            target_samples_per_bone=int(
                calibration["target_samples_per_bone"]
            ),
            max_samples_per_bone=int(
                calibration["max_samples_per_bone"]
            ),
            min_keypoint_confidence=float(
                calibration["min_keypoint_confidence"]
            ),
            min_depth_confidence=float(
                calibration["min_depth_confidence"]
            ),
            max_relative_mad=float(calibration["max_relative_mad"]),
            outlier_relative_tolerance=float(
                calibration["outlier_relative_tolerance"]
            ),
            outlier_absolute_tolerance_m=float(
                calibration["outlier_absolute_tolerance_m"]
            ),
            min_length_m=float(calibration["min_length_m"]),
            max_length_m=float(calibration["max_length_m"]),
        ),
        BoneLengthConstraint(
            anchor_confidence=float(solver["anchor_confidence"]),
            iterations=int(solver["iterations"]),
            max_joint_correction_m=float(
                solver["max_joint_correction_m"]
            ),
            max_predicted_correction_m=float(
                solver["max_predicted_correction_m"]
            ),
            fixed_joint_indices=tuple(solver["fixed_joint_indices"]),
        ),
    )


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.video_fps is not None and args.video_fps <= 0:
        raise ValueError("--video-fps must be positive.")

    sequence_dir = args.sequence.expanduser().resolve()
    camera_config = load_yaml(args.camera_config)["camera"]
    pose_config = load_yaml(args.pose_config)["pose"]
    tracking_config = load_yaml(args.tracking_config)["tracking"]
    depth_recovery_config = camera_config.get("depth_recovery", {})
    if not isinstance(depth_recovery_config, dict):
        raise ValueError("camera.depth_recovery must be a YAML mapping.")
    recovery_method = (
        args.recovery_method
        or depth_recovery_config.get("method", "window_median")
    )
    if recovery_method not in ("window_median", "pointcloud_cluster"):
        raise ValueError(
            "camera.depth_recovery.method must be window_median or "
            "pointcloud_cluster."
        )
    pointcloud_config = PointCloudRecoveryConfig.from_mapping(
        depth_recovery_config.get("pointcloud_cluster")
    )
    reset_gap_s = float(tracking_config["reset_gap_s"])
    frame_presence_config = FramePresenceConfig.from_mapping(
        tracking_config.get("frame_presence")
    )
    frame_presence_gate = PersonFramePresenceGate(
        frame_presence_config
    )

    # Validate the complete manifest before paying the model startup cost.
    all_frames = discover_rgbd_sequence(sequence_dir)
    frames = (
        all_frames[: args.max_frames]
        if args.max_frames is not None
        else all_frames
    )
    segments = split_at_time_gaps(frames, reset_gap_s)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else PROJECT_ROOT / "outputs/sequences" / sequence_dir.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    jsonl_path = output_dir / "poses.jsonl"
    summary_path = output_dir / "summary.json"
    manifest = create_manifest(
        sequence_dir,
        frames,
        segments,
        reset_gap_s,
        camera_config,
        frame_presence_config,
    )
    dump_json(manifest_path, manifest)

    median_interval_s = manifest["source_timing"][
        "median_interval_s_excluding_gaps"
    ]
    preview_fps = (
        float(args.video_fps)
        if args.video_fps is not None
        else (
            1.0 / median_interval_s
            if median_interval_s is not None and median_interval_s > 0
            else 2.0
        )
    )
    intrinsics = CameraIntrinsics(
        **camera_config["intrinsics"],
        width=int(camera_config["depth_width"]),
        height=int(camera_config["depth_height"]),
    )
    expected_size = (
        int(camera_config["rgb_width"]),
        int(camera_config["rgb_height"]),
    )
    temporal_filter = build_temporal_filter(tracking_config)
    bone_min_confidence = float(
        tracking_config["bone_statistics"]["min_joint_confidence"]
    )
    raw_bones = BoneLengthAccumulator(
        min_joint_confidence=bone_min_confidence
    )
    filtered_bones = BoneLengthAccumulator(
        min_joint_confidence=bone_min_confidence
    )
    constraint_components = build_bone_constraint(tracking_config)
    if constraint_components is None:
        bone_calibrator = None
        bone_constraint = None
    else:
        bone_calibrator, bone_constraint = constraint_components
    temporal_core_bones = BoneLengthAccumulator(
        links=HALPE26_CONSTRAINT_LINKS,
        min_joint_confidence=0.0,
    )
    constrained_core_bones = BoneLengthAccumulator(
        links=HALPE26_CONSTRAINT_LINKS,
        min_joint_confidence=0.0,
    )

    LOGGER.info(
        "Validated %d paired frames in %d segment(s), source cadence %.3f FPS",
        len(frames),
        len(segments),
        manifest["source_timing"]["nominal_fps"] or 0.0,
    )
    LOGGER.info("Using depth recovery method: %s", recovery_method)
    model_start = time.perf_counter()
    backend = build_backend(pose_config, args)
    model_initialization_s = time.perf_counter() - model_start
    LOGGER.info(
        "Initialized RTMPose once on %s in %.2f s",
        backend.device,
        model_initialization_s,
    )

    status_counts: Counter[str] = Counter()
    inference_times_ms: list[float] = []
    recovery_times_ms: list[float] = []
    point_cloud_times_ms: list[float] = []
    joint_lifting_times_ms: list[float] = []
    constraint_times_ms: list[float] = []
    total_times_ms: list[float] = []
    joint_2d_valid = np.zeros(len(HALPE26_NAMES), dtype=np.int64)
    joint_3d_valid = np.zeros(len(HALPE26_NAMES), dtype=np.int64)
    joint_temporal_usable = np.zeros(len(HALPE26_NAMES), dtype=np.int64)
    joint_temporal_observed = np.zeros(len(HALPE26_NAMES), dtype=np.int64)
    joint_temporal_predicted = np.zeros(len(HALPE26_NAMES), dtype=np.int64)
    detected_frame_count = 0
    valid_2d_total = 0
    valid_3d_total = 0
    constraint_frame_diagnostics: list[dict[str, Any]] = []
    constraint_corrections_m: list[float] = []
    constraint_observed_projection_delta_px: list[float] = []
    constraint_predicted_projection_delta_px: list[float] = []
    corrected_observed_joint_frames = 0
    corrected_predicted_joint_frames = 0
    processed_frame_count = 0
    errors: list[dict[str, Any]] = []
    video_paths: list[str] = []
    video_writer: cv2.VideoWriter | None = None
    completed = True
    bone_reset_pending = False
    pipeline_start = time.perf_counter()

    try:
        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for segment_id, segment in enumerate(segments):
                temporal_filter.reset()
                frame_presence_gate.reset()
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                if not args.no_video:
                    video_path = (
                        output_dir / f"segment_{segment_id:03d}_overlay.mp4"
                    )
                    video_writer = open_video_writer(
                        video_path,
                        codec=args.codec,
                        fps=preview_fps,
                        frame_size=expected_size,
                    )
                    video_paths.append(str(video_path))

                previous_timestamp_s: float | None = None
                for frame in segment:
                    frame_start = time.perf_counter()
                    status = "ok"
                    warnings: list[str] = []
                    pose2d: Pose2D | None = None
                    pose3d: Pose3D | None = None
                    recovery_diagnostics: dict[str, Any] | None = None
                    detected_person_count = 0
                    person_selection: str | None = None
                    presence_decision: FramePresenceDecision
                    timing_ms = {
                        "rgb_io": 0.0,
                        "depth_io": 0.0,
                        "inference": 0.0,
                        "point_cloud": 0.0,
                        "joint_lifting": 0.0,
                        "recovery": 0.0,
                        "bone_constraint": 0.0,
                        "visualization": 0.0,
                        "total": 0.0,
                    }
                    dt_s = (
                        None
                        if previous_timestamp_s is None
                        else frame.relative_time_s - previous_timestamp_s
                    )
                    previous_timestamp_s = frame.relative_time_s

                    rgb_start = time.perf_counter()
                    image_bgr = cv2.imread(
                        str(frame.rgb_path), cv2.IMREAD_COLOR
                    )
                    timing_ms["rgb_io"] = (
                        time.perf_counter() - rgb_start
                    ) * 1000.0
                    if image_bgr is None:
                        status = "rgb_read_error"
                        image_bgr = np.zeros(
                            (expected_size[1], expected_size[0], 3),
                            dtype=np.uint8,
                        )
                        warnings.append(
                            f"OpenCV failed to read {frame.rgb_path}"
                        )

                    depth_m: np.ndarray | None = None
                    if status != "rgb_read_error":
                        depth_start = time.perf_counter()
                        try:
                            depth_m = load_depth_m(
                                frame.depth_path,
                                float(camera_config["depth_scale"]),
                            )
                        except Exception as error:
                            status = "depth_read_error"
                            warnings.append(str(error))
                            if args.fail_fast:
                                raise
                        timing_ms["depth_io"] = (
                            time.perf_counter() - depth_start
                        ) * 1000.0

                    if image_bgr.shape[:2] != (
                        expected_size[1],
                        expected_size[0],
                    ):
                        status = "shape_mismatch"
                        warnings.append(
                            "RGB shape does not match camera configuration: "
                            f"{image_bgr.shape[:2]}"
                        )
                    if (
                        depth_m is not None
                        and depth_m.shape != image_bgr.shape[:2]
                    ):
                        status = "shape_mismatch"
                        warnings.append(
                            "Aligned RGB/depth shapes differ: "
                            f"RGB={image_bgr.shape[:2]}, "
                            f"depth={depth_m.shape}"
                        )

                    if status != "rgb_read_error":
                        inference_start = time.perf_counter()
                        try:
                            poses = backend.infer(image_bgr)
                            detected_person_count = len(poses)
                            pose2d = poses[0] if poses else None
                        except Exception as error:
                            status = "inference_error"
                            warnings.append(str(error))
                            if args.fail_fast:
                                raise
                        timing_ms["inference"] = (
                            time.perf_counter() - inference_start
                        ) * 1000.0
                        inference_times_ms.append(timing_ms["inference"])

                    presence_decision = frame_presence_gate.evaluate(
                        pose2d,
                        image_width=expected_size[0],
                        image_height=expected_size[1],
                        keypoint_threshold=float(
                            pose_config["keypoint_threshold"]
                        ),
                    )
                    if pose2d is not None:
                        detected_frame_count += 1
                        if presence_decision.accepted:
                            person_selection = "highest_bbox_score"
                            if bone_reset_pending:
                                if bone_calibrator is not None:
                                    bone_calibrator.reset()
                                bone_reset_pending = False
                        else:
                            person_selection = (
                                "rejected_partial_out_of_frame"
                            )
                            if status == "ok":
                                status = (
                                    "person_partially_out_of_frame"
                                )
                            pose2d = None

                    if pose2d is None:
                        if status == "ok":
                            status = "no_person"
                    else:
                        valid_2d = (
                            pose2d.scores
                            >= float(pose_config["keypoint_threshold"])
                        )
                        joint_2d_valid += valid_2d.astype(np.int64)
                        valid_2d_total += int(np.count_nonzero(valid_2d))

                        geometry_ready = (
                            depth_m is not None
                            and depth_m.shape == image_bgr.shape[:2]
                            and image_bgr.shape[:2]
                            == (expected_size[1], expected_size[0])
                        )
                        if geometry_ready:
                            recovery_start = time.perf_counter()
                            try:
                                lifting_start: float
                                if recovery_method == "pointcloud_cluster":
                                    point_cloud_start = time.perf_counter()
                                    organized_points_m = (
                                        depth_to_organized_point_cloud(
                                            depth_m=depth_m,
                                            intrinsics=intrinsics,
                                            min_depth_m=float(
                                                camera_config["min_depth_m"]
                                            ),
                                            max_depth_m=float(
                                                camera_config["max_depth_m"]
                                            ),
                                        )
                                    )
                                    timing_ms["point_cloud"] = (
                                        time.perf_counter()
                                        - point_cloud_start
                                    ) * 1000.0
                                    point_cloud_times_ms.append(
                                        timing_ms["point_cloud"]
                                    )
                                    lifting_start = time.perf_counter()
                                    recovery_result = (
                                        recover_pose3d_from_point_cloud(
                                            pose2d=pose2d,
                                            organized_points_m=(
                                                organized_points_m
                                            ),
                                            intrinsics=intrinsics,
                                            keypoint_threshold=float(
                                                pose_config[
                                                    "keypoint_threshold"
                                                ]
                                            ),
                                            config=pointcloud_config,
                                        )
                                    )
                                    pose3d = recovery_result.pose3d
                                    recovery_diagnostics = (
                                        recovery_result.diagnostics
                                    )
                                else:
                                    lifting_start = time.perf_counter()
                                    pose3d = recover_pose3d(
                                        pose2d=pose2d,
                                        depth_m=depth_m,
                                        intrinsics=intrinsics,
                                        keypoint_threshold=float(
                                            pose_config[
                                                "keypoint_threshold"
                                            ]
                                        ),
                                        radius=int(
                                            camera_config[
                                                "depth_window_radius"
                                            ]
                                        ),
                                        min_depth_m=float(
                                            camera_config["min_depth_m"]
                                        ),
                                        max_depth_m=float(
                                            camera_config["max_depth_m"]
                                        ),
                                    )
                                    recovery_diagnostics = {
                                        "method": "window_median",
                                        "parameters": {
                                            "radius_px": int(
                                                camera_config[
                                                    "depth_window_radius"
                                                ]
                                            ),
                                            "min_depth_m": float(
                                                camera_config["min_depth_m"]
                                            ),
                                            "max_depth_m": float(
                                                camera_config["max_depth_m"]
                                            ),
                                        },
                                        "valid_joint_count": int(
                                            np.count_nonzero(pose3d.valid)
                                        ),
                                    }
                                timing_ms["joint_lifting"] = (
                                    time.perf_counter() - lifting_start
                                ) * 1000.0
                                joint_lifting_times_ms.append(
                                    timing_ms["joint_lifting"]
                                )
                            except Exception as error:
                                status = "depth_recovery_error"
                                warnings.append(str(error))
                                if args.fail_fast:
                                    raise
                            timing_ms["recovery"] = (
                                time.perf_counter() - recovery_start
                            ) * 1000.0
                            recovery_times_ms.append(
                                timing_ms["recovery"]
                            )
                        if (
                            pose3d is not None
                            and not np.any(pose3d.valid)
                            and status == "ok"
                        ):
                            status = "no_valid_3d_joints"

                    if presence_decision.track_reset_required:
                        temporal_pose = temporal_filter.terminate_track(
                            frame.relative_time_s
                        )
                        bone_reset_pending = True
                    else:
                        temporal_pose = temporal_filter.update(
                            frame.relative_time_s, pose3d
                        )
                    constraint_result: BoneConstraintResult | None = None
                    if (
                        bone_calibrator is not None
                        and bone_constraint is not None
                    ):
                        constraint_start = time.perf_counter()
                        if pose3d is not None and pose2d is not None:
                            bone_calibrator.update(
                                pose3d, pose2d.scores
                            )
                        constraint_result = bone_constraint.apply(
                            temporal_pose, bone_calibrator.prior()
                        )
                        timing_ms["bone_constraint"] = (
                            time.perf_counter() - constraint_start
                        ) * 1000.0
                        constraint_times_ms.append(
                            timing_ms["bone_constraint"]
                        )
                    if pose3d is not None:
                        joint_3d_valid += pose3d.valid.astype(np.int64)
                        valid_3d_total += int(
                            np.count_nonzero(pose3d.valid)
                        )
                        raw_bones.update(
                            pose3d.joints_m,
                            pose3d.valid,
                            pose3d.confidence,
                        )
                    joint_temporal_usable += (
                        temporal_pose.usable.astype(np.int64)
                    )
                    joint_temporal_observed += (
                        temporal_pose.observed.astype(np.int64)
                    )
                    joint_temporal_predicted += (
                        temporal_pose.predicted.astype(np.int64)
                    )
                    filtered_bones.update(
                        temporal_pose.joints_m,
                        temporal_pose.observed,
                        temporal_pose.confidence,
                    )
                    temporal_core_bones.update(
                        temporal_pose.joints_m,
                        temporal_pose.usable,
                        temporal_pose.confidence,
                    )
                    if constraint_result is not None:
                        constrained_core_bones.update(
                            constraint_result.pose.joints_m,
                            constraint_result.pose.usable,
                            constraint_result.pose.confidence,
                        )
                        constraint_frame_diagnostics.append(
                            constraint_result.diagnostics
                        )
                        constraint_corrections_m.extend(
                            constraint_result.correction_m[
                                constraint_result.corrected
                            ].astype(float)
                        )
                        corrected_observed_joint_frames += int(
                            np.count_nonzero(
                                constraint_result.corrected
                                & temporal_pose.observed
                            )
                        )
                        corrected_predicted_joint_frames += int(
                            np.count_nonzero(
                                constraint_result.corrected
                                & temporal_pose.predicted
                            )
                        )
                        for index in np.flatnonzero(
                            constraint_result.corrected
                        ):
                            before_xyz = temporal_pose.joints_m[index]
                            after_xyz = constraint_result.pose.joints_m[index]
                            if (
                                not np.isfinite(before_xyz).all()
                                or not np.isfinite(after_xyz).all()
                                or before_xyz[2] <= 0
                                or after_xyz[2] <= 0
                            ):
                                continue
                            before_uv = np.array(
                                [
                                    intrinsics.fx
                                    * before_xyz[0]
                                    / before_xyz[2]
                                    + intrinsics.cx,
                                    intrinsics.fy
                                    * before_xyz[1]
                                    / before_xyz[2]
                                    + intrinsics.cy,
                                ]
                            )
                            after_uv = np.array(
                                [
                                    intrinsics.fx
                                    * after_xyz[0]
                                    / after_xyz[2]
                                    + intrinsics.cx,
                                    intrinsics.fy
                                    * after_xyz[1]
                                    / after_xyz[2]
                                    + intrinsics.cy,
                                ]
                            )
                            pixel_delta = float(
                                np.linalg.norm(after_uv - before_uv)
                            )
                            target = (
                                constraint_observed_projection_delta_px
                                if temporal_pose.observed[index]
                                else constraint_predicted_projection_delta_px
                            )
                            target.append(pixel_delta)

                    visualization_start = time.perf_counter()
                    diagnostic = draw_frame_diagnostics(
                        image_bgr,
                        pose2d,
                        pose3d,
                        temporal_pose,
                        constraint_result,
                        intrinsics,
                        status=status,
                        sequence_id=frame.sequence_id,
                        segment_id=segment_id,
                        frame_index=frame.frame_index,
                        timestamp_raw=frame.timestamp_raw,
                        dt_s=dt_s,
                        inference_ms=timing_ms["inference"],
                        score_threshold=float(
                            pose_config["keypoint_threshold"]
                        ),
                    )
                    if diagnostic.shape[1::-1] != expected_size:
                        diagnostic = cv2.resize(
                            diagnostic,
                            expected_size,
                            interpolation=cv2.INTER_AREA,
                        )
                    if video_writer is not None:
                        video_writer.write(diagnostic)
                    timing_ms["visualization"] = (
                        time.perf_counter() - visualization_start
                    ) * 1000.0
                    timing_ms["total"] = (
                        time.perf_counter() - frame_start
                    ) * 1000.0

                    record = {
                        "schema_version": 1,
                        "sequence_id": frame.sequence_id,
                        "segment_id": segment_id,
                        "segment_start": dt_s is None,
                        "frame_index": frame.frame_index,
                        "timestamp_raw": frame.timestamp_raw,
                        "captured_at": frame.captured_at.isoformat(
                            timespec="milliseconds"
                        ),
                        "relative_time_s": frame.relative_time_s,
                        "dt_s": dt_s,
                        "status": status,
                        "sources": {
                            "rgb": str(frame.rgb_path),
                            "depth": str(frame.depth_path),
                        },
                        "detected_person_count": detected_person_count,
                        "person_selection": person_selection,
                        "frame_presence": presence_decision.to_dict(),
                        "pose2d": (
                            pose2d.to_dict(
                                score_threshold=float(
                                    pose_config["keypoint_threshold"]
                                )
                            )
                            if pose2d is not None
                            else None
                        ),
                        "pose3d_raw": (
                            pose3d.to_dict()
                            if pose3d is not None
                            else None
                        ),
                        "depth_recovery": recovery_diagnostics,
                        "pose3d_temporal": temporal_pose.to_dict(),
                        "pose3d_constrained": (
                            constraint_result.to_dict()
                            if constraint_result is not None
                            else None
                        ),
                        "timing_ms": timing_ms,
                        "warnings": warnings,
                    }
                    json.dump(
                        record,
                        jsonl_file,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    jsonl_file.write("\n")
                    jsonl_file.flush()

                    status_counts[status] += 1
                    total_times_ms.append(timing_ms["total"])
                    processed_frame_count += 1
                    if warnings:
                        errors.append(
                            {
                                "frame_index": frame.frame_index,
                                "status": status,
                                "warnings": warnings,
                            }
                        )
                    LOGGER.info(
                        "[%d/%d] %s status=%s people=%d raw3d=%d "
                        "temporal=%d constrained=%d/%d inference=%.1f ms",
                        processed_frame_count,
                        len(frames),
                        frame.timestamp_raw,
                        status,
                        detected_person_count,
                        (
                            int(np.count_nonzero(pose3d.valid))
                            if pose3d is not None
                            else 0
                        ),
                        int(np.count_nonzero(temporal_pose.usable)),
                        (
                            constraint_result.diagnostics[
                                "calibrated_bone_count"
                            ]
                            if constraint_result is not None
                            else 0
                        ),
                        (
                            constraint_result.diagnostics[
                                "corrected_joint_count"
                            ]
                            if constraint_result is not None
                            else 0
                        ),
                        timing_ms["inference"],
                    )
    except KeyboardInterrupt:
        completed = False
        LOGGER.warning("Interrupted; closing partial outputs cleanly")
    finally:
        if video_writer is not None:
            video_writer.release()

    processing_elapsed_s = time.perf_counter() - pipeline_start
    per_joint = []
    for index, name in enumerate(HALPE26_NAMES):
        per_joint.append(
            {
                "id": index,
                "name": name,
                "valid_2d_frames": int(joint_2d_valid[index]),
                "valid_3d_frames": int(joint_3d_valid[index]),
                "temporal_usable_frames": int(
                    joint_temporal_usable[index]
                ),
                "temporal_observed_frames": int(
                    joint_temporal_observed[index]
                ),
                "temporal_predicted_frames": int(
                    joint_temporal_predicted[index]
                ),
                "valid_2d_rate_over_detected_frames": (
                    float(joint_2d_valid[index] / detected_frame_count)
                    if detected_frame_count
                    else None
                ),
                "valid_3d_rate_over_detected_frames": (
                    float(joint_3d_valid[index] / detected_frame_count)
                    if detected_frame_count
                    else None
                ),
                "temporal_usable_rate_over_processed_frames": (
                    float(
                        joint_temporal_usable[index]
                        / processed_frame_count
                    )
                    if processed_frame_count
                    else None
                ),
            }
        )

    frames_with_ready_prior = sum(
        item["calibrated_bone_count"] > 0
        for item in constraint_frame_diagnostics
    )
    frames_with_corrections = sum(
        item["corrected_joint_count"] > 0
        for item in constraint_frame_diagnostics
    )
    before_median_errors = [
        item["residual_before"]["median_relative_error"]
        for item in constraint_frame_diagnostics
        if item["residual_before"]["median_relative_error"] is not None
    ]
    after_median_errors = [
        item["residual_after"]["median_relative_error"]
        for item in constraint_frame_diagnostics
        if item["residual_after"]["median_relative_error"] is not None
    ]
    before_p95_errors = [
        item["residual_before"]["p95_relative_error"]
        for item in constraint_frame_diagnostics
        if item["residual_before"]["p95_relative_error"] is not None
    ]
    after_p95_errors = [
        item["residual_after"]["p95_relative_error"]
        for item in constraint_frame_diagnostics
        if item["residual_after"]["p95_relative_error"] is not None
    ]
    projectable_before_medians = [
        item["projectable_residual_before"]["median_relative_error"]
        for item in constraint_frame_diagnostics
        if item["projectable_residual_before"]["median_relative_error"]
        is not None
    ]
    projectable_after_medians = [
        item["projectable_residual_after"]["median_relative_error"]
        for item in constraint_frame_diagnostics
        if item["projectable_residual_after"]["median_relative_error"]
        is not None
    ]
    projectable_before_p95s = [
        item["projectable_residual_before"]["p95_relative_error"]
        for item in constraint_frame_diagnostics
        if item["projectable_residual_before"]["p95_relative_error"]
        is not None
    ]
    projectable_after_p95s = [
        item["projectable_residual_after"]["p95_relative_error"]
        for item in constraint_frame_diagnostics
        if item["projectable_residual_after"]["p95_relative_error"]
        is not None
    ]
    direction_medians = [
        item["direction_change"]["median_angle_deg"]
        for item in constraint_frame_diagnostics
        if item["direction_change"]["median_angle_deg"] is not None
    ]
    direction_p95s = [
        item["direction_change"]["p95_angle_deg"]
        for item in constraint_frame_diagnostics
        if item["direction_change"]["p95_angle_deg"] is not None
    ]
    constraint_summary = {
        "enabled": bone_constraint is not None,
        "core_link_count": len(HALPE26_CONSTRAINT_LINKS),
        "excluded_from_first_baseline": [
            "face_links",
            "foot_links",
        ],
        "frames_with_ready_prior": frames_with_ready_prior,
        "frames_with_corrections": frames_with_corrections,
        "corrected_observed_joint_frames": (
            corrected_observed_joint_frames
        ),
        "corrected_predicted_joint_frames": (
            corrected_predicted_joint_frames
        ),
        "joint_correction_m": value_statistics(
            constraint_corrections_m
        ),
        "observed_projection_delta_px": value_statistics(
            constraint_observed_projection_delta_px
        ),
        "predicted_projection_delta_px": value_statistics(
            constraint_predicted_projection_delta_px
        ),
        "max_anchor_displacement_m": max(
            (
                item["max_anchor_displacement_m"]
                for item in constraint_frame_diagnostics
            ),
            default=0.0,
        ),
        "max_root_displacement_m": max(
            (
                item["max_root_displacement_m"]
                for item in constraint_frame_diagnostics
            ),
            default=0.0,
        ),
        "flipped_bone_count": sum(
            item["direction_change"]["flipped_bone_count"]
            for item in constraint_frame_diagnostics
        ),
        "per_frame_median_relative_error_before": value_statistics(
            before_median_errors
        ),
        "per_frame_median_relative_error_after": value_statistics(
            after_median_errors
        ),
        "per_frame_p95_relative_error_before": value_statistics(
            before_p95_errors
        ),
        "per_frame_p95_relative_error_after": value_statistics(
            after_p95_errors
        ),
        "projectable_per_frame_median_relative_error_before": (
            value_statistics(projectable_before_medians)
        ),
        "projectable_per_frame_median_relative_error_after": (
            value_statistics(projectable_after_medians)
        ),
        "projectable_per_frame_p95_relative_error_before": (
            value_statistics(projectable_before_p95s)
        ),
        "projectable_per_frame_p95_relative_error_after": (
            value_statistics(projectable_after_p95s)
        ),
        "unresolved_anchor_only_violation_count": sum(
            item["anchor_only_residual"]["violating_bone_count"]
            for item in constraint_frame_diagnostics
        ),
        "per_frame_median_direction_change_deg": value_statistics(
            direction_medians
        ),
        "per_frame_p95_direction_change_deg": value_statistics(
            direction_p95s
        ),
        "calibration": (
            bone_calibrator.summary()
            if bone_calibrator is not None
            else None
        ),
        "profile_persists_across_time_segments": True,
        "profile_resets_for_new_sequence_or_track_identity": True,
        "single_person_identity_assumption": (
            "The current runner selects the highest bbox score and does "
            "not yet implement multi-person identity tracking."
        ),
    }

    summary = {
        "schema_version": 1,
        "completed": completed and processed_frame_count == len(frames),
        "sequence_id": sequence_dir.name,
        "sequence_directory": str(sequence_dir),
        "frame_counts": {
            "manifest": len(frames),
            "processed": processed_frame_count,
            "detected_person": detected_frame_count,
            "status": dict(sorted(status_counts.items())),
        },
        "detection_rate": (
            detected_frame_count / processed_frame_count
            if processed_frame_count
            else None
        ),
        "source_timing": manifest["source_timing"],
        "reset_gap_s": reset_gap_s,
        "frame_presence": {
            "config": frame_presence_config.to_dict(),
            "partial_out_of_frame_count": int(
                status_counts["person_partially_out_of_frame"]
            ),
        },
        "segment_count": len(segments),
        "segments": manifest["segments"],
        "processing": {
            "device": backend.device,
            "model_initialized_once": True,
            "model_initialization_s": model_initialization_s,
            "sequence_processing_s": processing_elapsed_s,
            "end_to_end_throughput_fps": (
                processed_frame_count / processing_elapsed_s
                if processing_elapsed_s > 0
                else None
            ),
            "inference": duration_statistics(inference_times_ms),
            "depth_recovery": duration_statistics(recovery_times_ms),
            "point_cloud": duration_statistics(point_cloud_times_ms),
            "joint_lifting": duration_statistics(joint_lifting_times_ms),
            "bone_constraint": duration_statistics(
                constraint_times_ms
            ),
            "per_frame_total": duration_statistics(total_times_ms),
        },
        "joint_recovery": {
            "method": recovery_method,
            "parameters": (
                pointcloud_config.to_dict()
                if recovery_method == "pointcloud_cluster"
                else {
                    "radius_px": int(
                        camera_config["depth_window_radius"]
                    ),
                    "min_depth_m": float(camera_config["min_depth_m"]),
                    "max_depth_m": float(camera_config["max_depth_m"]),
                }
            ),
            "valid_2d_total": valid_2d_total,
            "valid_3d_total": valid_3d_total,
            "temporal_observed_total": int(
                np.sum(joint_temporal_observed)
            ),
            "temporal_predicted_total": int(
                np.sum(joint_temporal_predicted)
            ),
            "valid_2d_rate_over_detected_joints": (
                valid_2d_total
                / (detected_frame_count * len(HALPE26_NAMES))
                if detected_frame_count
                else None
            ),
            "depth_recovery_rate_given_valid_2d": (
                valid_3d_total / valid_2d_total
                if valid_2d_total
                else None
            ),
            "per_joint": per_joint,
        },
        "bone_lengths_raw_observations": raw_bones.summary(),
        "bone_lengths_filtered_observations": filtered_bones.summary(),
        "bone_lengths_temporal_core_usable": (
            temporal_core_bones.summary()
        ),
        "bone_lengths_constrained_core_usable": (
            constrained_core_bones.summary()
        ),
        "bone_constraint": constraint_summary,
        "temporal_filter": {
            "one_euro": tracking_config["one_euro"],
            "max_prediction_s": float(
                tracking_config["max_prediction_s"]
            ),
            "min_observation_confidence": float(
                tracking_config["min_observation_confidence"]
            ),
            "directory_and_segment_boundaries_reset_state": True,
        },
        "outputs": {
            "manifest": str(manifest_path),
            "poses_jsonl": str(jsonl_path),
            "overlay_videos": video_paths,
        },
        "errors": errors,
        "notes": [
            "Source FPS is derived from median in-segment timestamps.",
            "Offline processing FPS is not the camera capture FPS.",
            "Temporal output distinguishes observed, predicted, and missing joints.",
            "Bone calibration uses raw observed RGB-D joints only.",
            (
                "The current dynamic sequence produces a temporary bone prior, "
                "not a definitive anatomical calibration or height estimate."
            ),
            (
                "High-confidence observations and the hip root are immutable "
                "constraint anchors."
            ),
            "The first constraint baseline excludes face and foot links.",
            (
                "Point-cloud recovery uses a bbox plus torso-depth proxy mask; "
                "it is not instance segmentation unless an external mask is "
                "supplied."
            ),
        ],
    }
    dump_json(summary_path, summary)
    LOGGER.info("Saved manifest: %s", manifest_path)
    LOGGER.info("Saved frame records: %s", jsonl_path)
    LOGGER.info("Saved summary: %s", summary_path)
    if video_paths:
        LOGGER.info("Saved %d overlay video(s)", len(video_paths))
    LOGGER.info(
        "Finished %d/%d frames at %.2f processing FPS; detection rate %.1f%%",
        processed_frame_count,
        len(frames),
        summary["processing"]["end_to_end_throughput_fps"] or 0.0,
        100.0 * (summary["detection_rate"] or 0.0),
    )
    return 0 if summary["completed"] else 130


if __name__ == "__main__":
    raise SystemExit(main())
