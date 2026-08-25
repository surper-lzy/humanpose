#!/usr/bin/env python3
"""Run local multi-person RGB-D tracking and optional WebSocket delivery."""

from __future__ import annotations

import argparse
from dataclasses import replace
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np

from rgbd_avatar.avatar import build_stick_figure_avatar
from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import GroundPlaneConfig, PointCloudRecoveryConfig
from rgbd_avatar.io import load_yaml_mapping
from rgbd_avatar.pose import RTMOBackend, RTMOBackendConfig
from rgbd_avatar.live import (
    AdaptiveHybridConfig,
    ApplicationExtrinsics,
    DirectoryRGBDSource,
    LocalMultiPersonConfig,
    LocalMultiPersonPoseProcessor,
    LocalMultiPersonPoseResult,
    KinematicFallbackConfig,
    LiveAutoCalibrationConfig,
    LxCameraRGBDSource,
    Pose3DQualityConfig,
    RGBDSource,
    RgbPreviewPublishConfig,
    RgbPreviewWebSocketPublisher,
    StickmanPublishConfig,
    StickmenWebSocketPublisher,
    calibrate_live_camera,
)
from rgbd_avatar.tracking import FramePresenceConfig, PersonFramePresenceGate
from rgbd_avatar.tracking.shadow_identity import ShadowIdentityConfig

from .live_mannequin import (
    _build_backend,
    _build_bone_components,
    _build_temporal_filter,
    _project_path,
)


LOGGER = logging.getLogger("view_live_multi_person")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
ResultCallback = Callable[[LocalMultiPersonPoseResult], None]
RTMPoseBackendBuilder = Callable[[dict[str, Any], argparse.Namespace], Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live-config",
        type=Path,
        default=PROJECT_ROOT / "configs/live.yaml",
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
        "--ground-config",
        type=Path,
        default=PROJECT_ROOT / "configs/ground.yaml",
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=("directory", "sdk"),
        default=None,
    )
    parser.add_argument(
        "--start-at",
        choices=("latest", "new", "oldest"),
        default=None,
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--pose-backend",
        choices=("rtmpose", "rtmo"),
        default=None,
        help=(
            "2D pose backend. The default remains RTMPose; rtmo enables the "
            "experimental one-stage multi-person path."
        ),
    )
    parser.add_argument(
        "--rtmo-model",
        default=None,
        help="Override the experimental RTMO model name or config path.",
    )
    parser.add_argument(
        "--rtmo-checkpoint",
        default=None,
        help=(
            "Optional local path or URL for RTMO weights. When omitted, "
            "MMPose resolves the official checkpoint from its model index."
        ),
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "whole_image"),
        default=None,
    )
    parser.add_argument(
        "--recovery-method",
        choices=(
            "window_median",
            "guided_window",
            "depth_connected",
            "pointcloud_cluster",
            "hybrid",
            "adaptive_hybrid",
        ),
        default=None,
        help=(
            "Local depth mode. depth_connected uses spatially connected "
            "depth surfaces and soft track history without a point cloud; "
            "adaptive_hybrid invokes point-cloud recovery only for suspicious "
            "face/arm groups; the default hybrid always recovers head/arms."
        ),
    )
    parser.add_argument(
        "--max-persons",
        type=int,
        default=4,
        help=(
            "Maximum people lifted to 3D and tracked locally. Defaults to 4."
        ),
    )
    parser.add_argument("--max-missing-s", type=float, default=0.35)
    parser.add_argument(
        "--identity-tracker",
        choices=("geometry", "shadow"),
        default=None,
        help=(
            "Identity association backend. geometry is the established "
            "fallback; shadow adds RGB-D/appearance occlusion handling."
        ),
    )
    parser.add_argument(
        "--publish-stickmen",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the additive avatar.stickmen.updated event.",
    )
    parser.add_argument(
        "--auto-calibrate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Estimate floor roll/pitch and camera height before publishing. "
            "The SDK-aligned runtime intrinsics are recorded with the result."
        ),
    )
    parser.add_argument(
        "--calibration-output",
        type=Path,
        default=None,
        help="Override the startup camera-calibration JSON output path.",
    )
    parser.add_argument(
        "--publish-url",
        default=None,
        help="Override the multi-person WebSocket hub URL and enable publish.",
    )
    parser.add_argument(
        "--publish-rgb-preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Publish a low-rate JPEG RGB + Halpe26 browser preview.",
    )
    parser.add_argument(
        "--rgb-preview-url",
        default=None,
        help="Override the binary RGB preview WebSocket URL and enable it.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run continuously without opening local GUI windows.",
    )
    parser.add_argument(
        "--view-mode",
        choices=("all", "2d", "3d"),
        default="all",
        help=(
            "Local visualization: all opens 3D plus RGB views, 2d opens an "
            "independent RGB skeleton window, and 3d opens only Open3D."
        ),
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Process local frames without opening GUI windows.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


class LatestLocalMultiPoseWorker:
    """Keep only the newest local multi-person result for the GUI."""

    def __init__(
        self,
        source: RGBDSource,
        processor: LocalMultiPersonPoseProcessor,
        *,
        read_timeout_ms: int,
        max_frames: int | None = None,
        result_callback: ResultCallback | None = None,
        source_already_started: bool = False,
    ) -> None:
        self.source = source
        self.processor = processor
        self.read_timeout_ms = int(read_timeout_ms)
        self.max_frames = max_frames
        self.result_callback = result_callback
        self.source_already_started = bool(source_already_started)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: LocalMultiPersonPoseResult | None = None
        self._version = 0
        self._error: Exception | None = None
        self._ready = threading.Event()
        self._finished = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="local-multi-pose-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self._stop.set()
            raise TimeoutError("Local multi-person RGB-D source did not start.")
        with self._lock:
            startup_error = self._error
        if startup_error is not None and self._finished.is_set():
            raise RuntimeError(
                "Local multi-person RGB-D source failed to start."
            ) from startup_error

    def _run(self) -> None:
        processed = 0
        try:
            if not self.source_already_started:
                self.source.start()
            self._ready.set()
            while not self._stop.is_set():
                if self.max_frames is not None and processed >= self.max_frames:
                    break
                try:
                    frame = self.source.read(timeout_ms=self.read_timeout_ms)
                except TimeoutError:
                    continue
                except RuntimeError:
                    if self._stop.is_set():
                        break
                    raise
                try:
                    result = self.processor.process(frame)
                except Exception as error:
                    LOGGER.exception(
                        "Local multi-person frame %d failed; continuing",
                        frame.frame_number,
                    )
                    with self._lock:
                        self._error = error
                    continue
                processed += 1
                if self.result_callback is not None:
                    try:
                        self.result_callback(result)
                    except Exception:
                        LOGGER.exception(
                            "Multi-person result callback failed; continuing"
                        )
                with self._lock:
                    self._latest = result
                    self._version += 1
                    self._error = None
                usable = sum(
                    int(np.count_nonzero(person.pose3d_output.usable))
                    for person in result.persons
                )
                LOGGER.info(
                    "frame=%d detected=%d tracks=%d usable=%d "
                    "identity=%s fallback=%s inference=%.1f ms "
                    "recovery=%.1f ms (fast=%.1f cloud=%.1f robust=%.1f "
                    "refine=%.1f connected=%.1f guided=%.1f "
                    "full=%d guided_people=%d joints=%d) quality=%.1f ms "
                    "reject=%d invalid=%d total=%.1f ms",
                    result.frame_number,
                    result.detected_person_count,
                    len(result.persons),
                    usable,
                    result.identity_method,
                    result.identity_fallback,
                    result.timing_ms["inference"],
                    result.timing_ms["recovery"],
                    result.timing_ms.get("recovery_fast", 0.0),
                    result.timing_ms.get("recovery_cloud_build", 0.0),
                    result.timing_ms.get("recovery_robust", 0.0),
                    result.timing_ms.get("recovery_refine", 0.0),
                    result.timing_ms.get(
                        "recovery_refine_connected", 0.0
                    ),
                    result.timing_ms.get("recovery_refine_guided", 0.0),
                    result.recovery_stats.get(
                        "depth_connected_full_person_count", 0
                    ),
                    result.recovery_stats.get(
                        "depth_connected_guided_person_count", 0
                    ),
                    result.recovery_stats.get("robust_joint_count", 0),
                    result.timing_ms.get("quality", 0.0),
                    result.recovery_stats.get(
                        "quality_rejected_person_count", 0
                    ),
                    result.recovery_stats.get(
                        "quality_invalidated_joint_count", 0
                    ),
                    result.timing_ms["total"],
                )
        except Exception as error:
            LOGGER.exception("Local multi-person worker stopped unexpectedly")
            with self._lock:
                self._error = error
        finally:
            self._ready.set()
            self.source.close()
            self._finished.set()

    def latest_after(
        self,
        version: int,
    ) -> tuple[int, LocalMultiPersonPoseResult | None]:
        with self._lock:
            if version == self._version:
                return version, None
            return self._version, self._latest

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    @property
    def finished(self) -> bool:
        return self._finished.is_set()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(
                timeout=max(5.0, self.read_timeout_ms / 1000.0 + 2.0)
            )
        if self._thread.is_alive():
            LOGGER.warning("Local multi-person worker did not stop in time.")


def _build_source(
    args: argparse.Namespace,
    live_config: dict[str, Any],
    camera_config: dict[str, Any],
) -> RGBDSource:
    source_config = live_config["source"]
    source_type = args.source or source_config.get("type", "directory")
    if args.input_dir is not None:
        source_type = "directory"
    if source_type == "sdk":
        return LxCameraRGBDSource.from_mapping(
            source_config["sdk"],
            depth_scale=float(camera_config["depth_scale"]),
        )
    if source_type != "directory":
        raise ValueError(f"Unsupported local multi-person source: {source_type!r}.")
    input_directory = (
        args.input_dir.expanduser().resolve()
        if args.input_dir is not None
        else Path(source_config["directory"]).expanduser().resolve()
    )
    intrinsics = CameraIntrinsics(
        **camera_config["intrinsics"],
        width=int(camera_config["depth_width"]),
        height=int(camera_config["depth_height"]),
    )
    return DirectoryRGBDSource(
        input_directory,
        intrinsics=intrinsics,
        depth_scale=float(camera_config["depth_scale"]),
        start_at=args.start_at or source_config["start_at"],
        poll_interval_s=float(source_config["poll_interval_ms"]) / 1000.0,
        stable_interval_s=float(source_config["stable_interval_ms"]) / 1000.0,
    )


def _build_processor(
    args: argparse.Namespace,
    live_config: dict[str, Any],
    camera_config: dict[str, Any],
    pose_config: dict[str, Any],
    tracking_config: dict[str, Any],
    *,
    rtmpose_backend_builder: RTMPoseBackendBuilder | None = None,
    depth_connected_refresh_interval: int = 1,
) -> LocalMultiPersonPoseProcessor:
    pose_backend = getattr(args, "pose_backend", None) or str(
        pose_config.get("backend", "rtmpose")
    )
    if pose_backend == "rtmpose":
        backend_builder = rtmpose_backend_builder or _build_backend
        backend = backend_builder(pose_config, args)
    elif pose_backend == "rtmo":
        rtmo_mapping = dict(pose_config.get("experimental_rtmo", {}))
        model = getattr(args, "rtmo_model", None) or str(
            rtmo_mapping.get("model", "rtmo-t_8xb32-600e_body7-416x416")
        )
        checkpoint = getattr(args, "rtmo_checkpoint", None)
        if checkpoint is None:
            checkpoint = rtmo_mapping.get("model_checkpoint")
        backend = RTMOBackend(
            RTMOBackendConfig(
                model=model,
                model_checkpoint=checkpoint,
                model_cache_dir=_project_path(pose_config["model_cache_dir"]),
                device=args.device or pose_config["device"],
                bbox_threshold=float(
                    rtmo_mapping.get(
                        "bbox_threshold", pose_config["bbox_threshold"]
                    )
                ),
                nms_threshold=float(rtmo_mapping.get("nms_threshold", 0.65)),
                keypoint_threshold=float(
                    rtmo_mapping.get(
                        "keypoint_threshold", pose_config["keypoint_threshold"]
                    )
                ),
                min_valid_keypoints=int(
                    rtmo_mapping.get(
                        "min_valid_keypoints",
                        pose_config["min_valid_keypoints"],
                    )
                ),
                min_mean_keypoint_score=float(
                    rtmo_mapping.get(
                        "min_mean_keypoint_score",
                        pose_config["min_mean_keypoint_score"],
                    )
                ),
            )
        )
    else:
        raise ValueError(f"Unsupported multi-person pose backend: {pose_backend!r}.")
    LOGGER.info("Local multi-person pose backend: %s", pose_backend)
    recovery_mapping = camera_config.get("depth_recovery", {})
    multi_mapping = live_config.get("multi_person", {})
    # Keep this experiment independent from the single-person/WebSocket
    # default. Full point-cloud recovery scales almost linearly with person
    # count, while hybrid retains it for the failure-prone head and arms.
    recovery_method = args.recovery_method or str(
        multi_mapping.get("recovery_method", "hybrid")
    )
    LOGGER.info("Local multi-person depth recovery: %s", recovery_method)
    frame_presence_config = FramePresenceConfig.from_mapping(
        tracking_config.get("frame_presence")
    )
    identity_tracker = args.identity_tracker or str(
        multi_mapping.get("identity_tracker", "geometry")
    )
    LOGGER.info("Multi-person identity tracker: %s", identity_tracker)
    return LocalMultiPersonPoseProcessor(
        backend=backend,
        extrinsics=ApplicationExtrinsics.from_mapping(
            live_config["application_extrinsics"]
        ),
        temporal_filter_factory=lambda: _build_temporal_filter(
            tracking_config,
            multi_mapping.get("one_euro"),
            max_prediction_s_override=(
                float(multi_mapping["max_prediction_s"])
                if "max_prediction_s" in multi_mapping
                else None
            ),
        ),
        presence_gate_factory=lambda: PersonFramePresenceGate(
            frame_presence_config
        ),
        bone_components_factory=lambda: _build_bone_components(
            tracking_config,
            project_observed_override=bool(
                multi_mapping.get("bone_stabilization", {}).get(
                    "project_observed", False
                )
            ),
        ),
        keypoint_threshold=float(backend.keypoint_threshold),
        min_depth_m=float(camera_config["min_depth_m"]),
        max_depth_m=float(camera_config["max_depth_m"]),
        depth_window_radius=int(camera_config["depth_window_radius"]),
        recovery_method=recovery_method,
        pointcloud_config=PointCloudRecoveryConfig.from_mapping(
            recovery_mapping.get("pointcloud_cluster")
        ),
        adaptive_hybrid_config=AdaptiveHybridConfig.from_mapping(
            recovery_mapping.get("adaptive_hybrid")
        ),
        pose3d_quality_config=Pose3DQualityConfig.from_mapping(
            tracking_config.get("pose3d_quality")
        ),
        kinematic_fallback_config=KinematicFallbackConfig.from_mapping(
            multi_mapping.get("kinematic_fallback")
        ),
        multi_person_config=LocalMultiPersonConfig(
            max_persons=args.max_persons,
            max_missing_s=args.max_missing_s,
        ),
        identity_tracker=identity_tracker,
        shadow_identity_config=ShadowIdentityConfig(
            **multi_mapping.get("shadow_identity", {})
        ),
        depth_connected_refresh_interval=depth_connected_refresh_interval,
    )


def _build_multi_publish_config(
    args: argparse.Namespace,
    live_config: dict[str, Any],
) -> StickmanPublishConfig:
    """Inherit hub connection settings without inheriting single publish."""

    common = dict(live_config.get("websocket_publish", {}))
    multi = dict(
        live_config.get("multi_person", {}).get("websocket_publish", {})
    )
    common.update(multi)
    common["enabled"] = bool(multi.get("enabled", False))
    common["event"] = str(
        multi.get("event", "avatar.stickmen.updated")
    )
    return StickmanPublishConfig.from_mapping(
        common,
        enabled_override=args.publish_stickmen,
        url_override=args.publish_url,
    )


def _build_rgb_preview_publish_config(
    args: argparse.Namespace,
    live_config: dict[str, Any],
    pose_config: dict[str, Any],
) -> RgbPreviewPublishConfig:
    common = dict(live_config.get("websocket_publish", {}))
    mapping = dict(
        live_config.get("multi_person", {}).get("rgb_preview_publish", {})
    )
    mapping.setdefault("source_id", common.get("source_id", "camera-01"))
    mapping.setdefault(
        "url",
        "ws://127.0.0.1:8000/api/realtime/ws",
    )
    mapping.setdefault(
        "keypoint_threshold",
        float(pose_config["keypoint_threshold"]),
    )
    for field_name in (
        "open_timeout_s",
        "close_timeout_s",
        "ping_interval_s",
        "ping_timeout_s",
        "reconnect_initial_s",
        "reconnect_max_s",
    ):
        if field_name not in mapping and field_name in common:
            mapping[field_name] = common[field_name]
    enabled_override = args.publish_rgb_preview
    if enabled_override is None and args.publish_stickmen is True:
        # Existing Nano service units already opt in with --publish-stickmen.
        # Make the paired diagnostic preview available without requiring a
        # privileged systemd unit edit; --no-publish-rgb-preview still wins.
        enabled_override = True
    return RgbPreviewPublishConfig.from_mapping(
        mapping,
        enabled_override=enabled_override,
        url_override=args.rgb_preview_url,
    )


def _run_startup_calibration(
    args: argparse.Namespace,
    source: RGBDSource,
    processor: LocalMultiPersonPoseProcessor,
    live_config: dict[str, Any],
    ground_config: GroundPlaneConfig,
    *,
    read_timeout_ms: int,
) -> None:
    config = LiveAutoCalibrationConfig.from_mapping(
        live_config.get("auto_calibration")
    )
    if args.auto_calibrate is not None:
        config = replace(config, enabled=bool(args.auto_calibrate))
    if not config.enabled:
        LOGGER.info(
            "Camera auto-calibration disabled; using configured extrinsics."
        )
        return

    output_path = (
        args.calibration_output.expanduser().resolve()
        if args.calibration_output is not None
        else (PROJECT_ROOT / config.output_path).resolve()
    )
    LOGGER.info(
        "Camera auto-calibration: collecting %d floor frames (max %d)...",
        config.sample_frame_count,
        config.max_attempt_frame_count,
    )
    try:
        result = calibrate_live_camera(
            source,
            processor.backend,
            heading_reference=processor.extrinsics,
            config=config,
            ground_config=ground_config,
            read_timeout_ms=read_timeout_ms,
            output_path=output_path,
            source_already_started=True,
        )
    except Exception:
        if not config.fallback_to_config:
            source.close()
            raise
        LOGGER.exception(
            "Camera auto-calibration failed; retaining configured "
            "application extrinsics."
        )
        return

    processor.extrinsics = result.extrinsics
    LOGGER.info(
        "Camera auto-calibration accepted: fx=%.3f fy=%.3f "
        "cx=%.3f cy=%.3f height=%.3f m tilt=%.2f deg "
        "inliers=%d/%d p95=%.4f m output=%s",
        result.intrinsics.fx,
        result.intrinsics.fy,
        result.intrinsics.cx,
        result.intrinsics.cy,
        result.ground_plane.camera_height_m,
        result.ground_plane.tilt_from_camera_up_deg,
        result.ground_plane.inlier_count,
        result.ground_plane.candidate_count,
        result.ground_plane.residual_p95_m,
        output_path,
    )


def _validate_local_frames(
    source: RGBDSource,
    processor: LocalMultiPersonPoseProcessor,
    *,
    read_timeout_ms: int,
    max_frames: int,
    result_callback: ResultCallback | None = None,
    source_already_started: bool = False,
) -> None:
    if not source_already_started:
        source.start()
    try:
        for index in range(max_frames):
            frame = source.read(timeout_ms=read_timeout_ms)
            result = processor.process(frame)
            if result_callback is not None:
                result_callback(result)
            primitive_count = sum(
                build_stick_figure_avatar(
                    person.joints_application_m,
                    person.pose3d_output.usable,
                    ground_height_m=0.0,
                ).primitive_count
                for person in result.persons
            )
            LOGGER.info(
                "validate [%d/%d] frame=%d detected=%d track_ids=%s "
                "bbox_scores=%s primitives=%d identity=%s fallback=%s "
                "inference=%.1f ms "
                "recovery=%.1f ms (fast=%.1f cloud=%.1f robust=%.1f "
                "refine=%.1f connected=%.1f guided=%.1f "
                "full=%d guided_people=%d joints=%d) quality=%.1f ms "
                "reject=%d invalid=%d kfill=%d complete=%d missing=%d "
                "total=%.1f ms",
                index + 1,
                max_frames,
                frame.frame_number,
                result.detected_person_count,
                [person.track_id for person in result.persons],
                [
                    round(person.pose2d.bbox_score, 3)
                    for person in result.persons
                    if person.pose2d is not None
                ],
                primitive_count,
                result.identity_method,
                result.identity_fallback,
                result.timing_ms["inference"],
                result.timing_ms["recovery"],
                result.timing_ms.get("recovery_fast", 0.0),
                result.timing_ms.get("recovery_cloud_build", 0.0),
                result.timing_ms.get("recovery_robust", 0.0),
                result.timing_ms.get("recovery_refine", 0.0),
                result.timing_ms.get("recovery_refine_connected", 0.0),
                result.timing_ms.get("recovery_refine_guided", 0.0),
                result.recovery_stats.get(
                    "depth_connected_full_person_count", 0
                ),
                result.recovery_stats.get(
                    "depth_connected_guided_person_count", 0
                ),
                result.recovery_stats.get("robust_joint_count", 0),
                result.timing_ms.get("quality", 0.0),
                result.recovery_stats.get(
                    "quality_rejected_person_count", 0
                ),
                result.recovery_stats.get(
                    "quality_invalidated_joint_count", 0
                ),
                result.recovery_stats.get(
                    "kinematic_fallback_joint_count", 0
                ),
                result.recovery_stats.get(
                    "skeleton_completion_joint_count", 0
                ),
                result.recovery_stats.get("missing_output_joint_count", 0),
                result.timing_ms["total"],
            )
    finally:
        source.close()


def main(
    *,
    rtmpose_backend_builder: RTMPoseBackendBuilder | None = None,
    rtmpose_backend_description: str | None = None,
    depth_connected_refresh_interval: int = 1,
) -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.max_persons <= 0:
        raise ValueError("--max-persons must be positive.")
    if args.max_missing_s <= 0:
        raise ValueError("--max-missing-s must be positive.")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    live_config = load_yaml_mapping(args.live_config)["live"]
    camera_config = load_yaml_mapping(args.camera_config)["camera"]
    pose_config = load_yaml_mapping(args.pose_config)["pose"]
    tracking_config = load_yaml_mapping(args.tracking_config)["tracking"]
    ground_config = GroundPlaneConfig.from_mapping(
        load_yaml_mapping(args.ground_config).get("ground")
    )
    source = _build_source(args, live_config, camera_config)
    LOGGER.info("Local multi-person input: %s", source.source_id)
    selected_backend = args.pose_backend or str(
        pose_config.get("backend", "rtmpose")
    )
    if selected_backend == "rtmo":
        if args.detector is not None:
            LOGGER.info("--detector is ignored by the one-stage RTMO backend.")
        LOGGER.warning(
            "Initializing experimental RTMO once; the default RTMPose path "
            "and WebSocket schema remain unchanged."
        )
    else:
        if args.detector == "whole_image":
            LOGGER.warning(
                "whole_image bypasses person detection and cannot provide "
                "true multi-person boxes; use --detector auto."
            )
        LOGGER.info(
            "Initializing %s once...",
            rtmpose_backend_description or "RTMPose",
        )
    processor = _build_processor(
        args,
        live_config,
        camera_config,
        pose_config,
        tracking_config,
        rtmpose_backend_builder=rtmpose_backend_builder,
        depth_connected_refresh_interval=depth_connected_refresh_interval,
    )
    read_timeout_ms = int(live_config["source"]["read_timeout_ms"])
    # Use one continuous SDK stream for startup calibration and steady-state
    # processing. The LANXIN source cannot be started twice back-to-back.
    source.start()
    _run_startup_calibration(
        args,
        source,
        processor,
        live_config,
        ground_config,
        read_timeout_ms=read_timeout_ms,
    )
    publish_config = _build_multi_publish_config(args, live_config)
    publisher = StickmenWebSocketPublisher(publish_config)
    preview_config = _build_rgb_preview_publish_config(
        args,
        live_config,
        pose_config,
    )
    preview_publisher = RgbPreviewWebSocketPublisher(preview_config)
    if publish_config.enabled:
        LOGGER.info(
            "Multi-person WebSocket enabled: event=%s topic=%s",
            publish_config.event,
            publish_config.topic,
        )
        publisher.start()
    else:
        LOGGER.info("Multi-person WebSocket disabled; local processing only.")
    if preview_config.enabled:
        LOGGER.info(
            "RGB skeleton preview enabled: source=%s fps=%.1f quality=%d scale=%.2f",
            preview_config.source_id,
            preview_config.fps,
            preview_config.jpeg_quality,
            preview_config.scale,
        )
        preview_publisher.start()
    else:
        LOGGER.info("RGB skeleton browser preview disabled.")

    callbacks: list[ResultCallback] = []
    if publish_config.enabled:
        callbacks.append(publisher.submit)
    if preview_config.enabled:
        callbacks.append(preview_publisher.submit)

    def publish_result(result: LocalMultiPersonPoseResult) -> None:
        for callback in callbacks:
            callback(result)

    result_callback = publish_result if callbacks else None

    if args.validate_only:
        try:
            _validate_local_frames(
                source,
                processor,
                read_timeout_ms=read_timeout_ms,
                max_frames=args.max_frames or 1,
                result_callback=result_callback,
                source_already_started=True,
            )
        finally:
            publisher.stop()
            preview_publisher.stop()
        LOGGER.info("Local multi-person source statistics: %s", source.stats)
        if publish_config.enabled:
            LOGGER.info("Multi-person publisher statistics: %s", publisher.stats)
        if preview_config.enabled:
            LOGGER.info("RGB preview publisher statistics: %s", preview_publisher.stats)
        return 0

    worker = LatestLocalMultiPoseWorker(
        source,
        processor,
        read_timeout_ms=read_timeout_ms,
        max_frames=args.max_frames,
        result_callback=result_callback,
        source_already_started=True,
    )
    if args.headless:
        try:
            worker.start()
            while not worker.finished:
                time.sleep(0.05)
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user.")
        finally:
            worker.stop()
            publisher.stop()
            preview_publisher.stop()
        if worker.error is not None:
            raise RuntimeError(
                "Local multi-person worker failed."
            ) from worker.error
        LOGGER.info("Local multi-person source statistics: %s", source.stats)
        if publish_config.enabled:
            LOGGER.info("Multi-person publisher statistics: %s", publisher.stats)
        if preview_config.enabled:
            LOGGER.info("RGB preview publisher statistics: %s", preview_publisher.stats)
        return 0

    from rgbd_avatar.visualization.live_multi_person import (
        LocalMultiPerson2DRenderer,
        LocalMultiPersonRenderer,
    )

    viewer_config = live_config["viewer"]
    extrinsics = processor.extrinsics
    if args.view_mode == "2d":
        renderer = LocalMultiPerson2DRenderer(
            rgb_view_scale=float(viewer_config.get("rgb_view_scale", 0.75)),
            keypoint_threshold=float(pose_config["keypoint_threshold"]),
        )
    else:
        renderer = LocalMultiPersonRenderer(
            sphere_resolution=int(viewer_config["sphere_resolution"]),
            grid_extent_m=float(viewer_config["grid_extent_m"]),
            grid_spacing_m=float(viewer_config["grid_spacing_m"]),
            window_width=int(viewer_config["window_width"]),
            window_height=int(viewer_config["window_height"]),
            show_rgb_views=args.view_mode == "all",
            rgb_view_scale=float(viewer_config.get("rgb_view_scale", 0.75)),
            keypoint_threshold=float(pose_config["keypoint_threshold"]),
            camera_forward_application=(
                extrinsics.rotation_application_from_camera[:, 2]
            ),
        )
    version = 0
    try:
        worker.start()
        renderer.open()
        while renderer.poll():
            version, result = worker.latest_after(version)
            if result is not None:
                renderer.update(result)
            if worker.finished and result is None:
                break
            time.sleep(0.005)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        worker.stop()
        renderer.close()
        publisher.stop()
        preview_publisher.stop()
    if worker.error is not None:
        raise RuntimeError("Local multi-person worker failed.") from worker.error
    LOGGER.info("Local multi-person source statistics: %s", source.stats)
    if publish_config.enabled:
        LOGGER.info("Multi-person publisher statistics: %s", publisher.stats)
    if preview_config.enabled:
        LOGGER.info("RGB preview publisher statistics: %s", preview_publisher.stats)
    return 0
