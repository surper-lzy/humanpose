#!/usr/bin/env python3
"""Show live RGB, RGB skeleton, and a standalone metric 3D stickman."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping

import numpy as np

from rgbd_avatar.avatar import (
    build_procedural_avatar,
    build_stick_figure_avatar,
)
from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import PointCloudRecoveryConfig
from rgbd_avatar.io import load_yaml_mapping, resolve_path
from rgbd_avatar.live import (
    ApplicationExtrinsics,
    DirectoryRGBDSource,
    LivePoseProcessor,
    LivePoseResult,
    LxCameraRGBDSource,
    RGBDSource,
    StickmanPublishConfig,
    StickmanWebSocketPublisher,
)
from rgbd_avatar.pose import RTMPoseBackend, RTMPoseBackendConfig
from rgbd_avatar.tracking import (
    BoneLengthCalibrator,
    BoneLengthConstraint,
    FramePresenceConfig,
    PersonFramePresenceGate,
    Pose3DTemporalFilter,
)


LOGGER = logging.getLogger("view_live_mannequin")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


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
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument(
        "--source",
        choices=("directory", "sdk"),
        default=None,
        help="Override live.source.type from the live config.",
    )
    parser.add_argument(
        "--start-at",
        choices=("latest", "new", "oldest"),
        default=None,
        help="latest shows the newest existing pair, then follows new files.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--detector",
        choices=("auto", "whole_image"),
        default=None,
    )
    parser.add_argument(
        "--recovery-method",
        choices=("window_median", "pointcloud_cluster"),
        default=None,
    )
    parser.add_argument(
        "--avatar-model",
        choices=("procedural", "stickman"),
        default=None,
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Process frames and build avatar primitives without opening Open3D.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Process and publish poses without opening local GUI windows.",
    )
    parser.add_argument(
        "--publish-stickman",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable FastAPI realtime-hub publishing.",
    )
    parser.add_argument(
        "--publish-url",
        default=None,
        help=(
            "Override the FastAPI WebSocket URL and enable publishing, for "
            "example ws://HOST:8000/api/realtime/ws."
        ),
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--mixamo-cache",
        type=Path,
        default=None,
        help=(
            "Path to a mixamo_sequence.npz cache (produced by "
            "fit_mixamo_sequence.py). When provided, a browser-based "
            "textured Mixamo avatar window opens alongside the Open3D "
            "mannequin window."
        ),
    )
    parser.add_argument(
        "--mixamo-viewer-port",
        type=int,
        default=8095,
        help="Port for the browser-based Mixamo live viewer (default 8095).",
    )
    parser.add_argument(
        "--mixamo-res",
        type=int,
        default=1024,
        help="Render resolution for the live Mixamo viewer (default 1024).",
    )
    return parser.parse_args()


def _project_path(value: str | Path) -> Path:
    return resolve_path(value, relative_to=PROJECT_ROOT)


def _build_backend(
    pose_config: dict[str, Any],
    args: argparse.Namespace,
) -> RTMPoseBackend:
    return RTMPoseBackend(
        RTMPoseBackendConfig(
            model_config=_project_path(pose_config["model_config"]),
            model_checkpoint=_project_path(pose_config["model_checkpoint"]),
            detector=args.detector or pose_config["detector"],
            model_cache_dir=_project_path(pose_config["model_cache_dir"]),
            device=args.device or pose_config["device"],
            bbox_threshold=float(pose_config["bbox_threshold"]),
            keypoint_threshold=float(pose_config["keypoint_threshold"]),
            min_valid_keypoints=int(pose_config["min_valid_keypoints"]),
            min_mean_keypoint_score=float(
                pose_config["min_mean_keypoint_score"]
            ),
        )
    )


def _build_temporal_filter(
    tracking_config: dict[str, Any],
    one_euro_override: Mapping[str, Any] | None = None,
    *,
    max_prediction_s_override: float | None = None,
) -> Pose3DTemporalFilter:
    one_euro = dict(tracking_config["one_euro"])
    if one_euro_override is not None:
        one_euro.update(one_euro_override)
    return Pose3DTemporalFilter(
        min_cutoff_hz=float(one_euro["min_cutoff_hz"]),
        beta=float(one_euro["beta"]),
        derivative_cutoff_hz=float(one_euro["derivative_cutoff_hz"]),
        reset_gap_s=float(tracking_config["reset_gap_s"]),
        max_prediction_s=float(
            tracking_config["max_prediction_s"]
            if max_prediction_s_override is None
            else max_prediction_s_override
        ),
        min_observation_confidence=float(
            tracking_config["min_observation_confidence"]
        ),
        shared_cutoff=bool(one_euro.get("shared_cutoff", False)),
        shared_speed_percentile=float(
            one_euro.get("shared_speed_percentile", 75.0)
        ),
    )


def _build_bone_components(
    tracking_config: dict[str, Any],
    *,
    project_observed_override: bool | None = None,
) -> tuple[BoneLengthCalibrator | None, BoneLengthConstraint | None]:
    config = tracking_config.get("bone_constraint", {})
    if not bool(config.get("enabled", False)):
        return None, None
    calibration = config["calibration"]
    solver = config["solver"]
    return (
        BoneLengthCalibrator(
            min_samples_per_bone=int(calibration["min_samples_per_bone"]),
            target_samples_per_bone=int(calibration["target_samples_per_bone"]),
            max_samples_per_bone=int(calibration["max_samples_per_bone"]),
            min_keypoint_confidence=float(
                calibration["min_keypoint_confidence"]
            ),
            min_depth_confidence=float(calibration["min_depth_confidence"]),
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
            max_joint_correction_m=float(solver["max_joint_correction_m"]),
            max_predicted_correction_m=float(
                solver["max_predicted_correction_m"]
            ),
            fixed_joint_indices=tuple(solver["fixed_joint_indices"]),
            project_observed=(
                bool(solver.get("project_observed", False))
                if project_observed_override is None
                else bool(project_observed_override)
            ),
        ),
    )


class LatestPoseWorker:
    """Process frames in the background while the GUI remains responsive."""

    def __init__(
        self,
        source: RGBDSource,
        processor: LivePoseProcessor,
        *,
        read_timeout_ms: int,
        max_frames: int | None = None,
        result_sink: Callable[[LivePoseResult], None] | None = None,
    ) -> None:
        self.source = source
        self.processor = processor
        self.read_timeout_ms = int(read_timeout_ms)
        self.max_frames = max_frames
        self.result_sink = result_sink
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: LivePoseResult | None = None
        self._version = 0
        self._error: Exception | None = None
        self._ready = threading.Event()
        self._finished = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="live-pose-worker",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()
        if not self._ready.wait(timeout=10.0):
            self._stop.set()
            raise TimeoutError("Live RGB-D source did not start within 10 s.")
        with self._lock:
            startup_error = self._error
        if startup_error is not None and self._finished.is_set():
            raise RuntimeError("Live RGB-D source failed to start.") from startup_error

    def _run(self) -> None:
        processed = 0
        try:
            # Some native camera SDKs bind stream state to the calling thread.
            # Keep start/read/close on this one owner thread; cross-thread close
            # can otherwise crash during native-library teardown.
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
                except Exception as error:  # keep a live service recoverable
                    LOGGER.exception(
                        "Live frame %d failed; waiting for the next frame",
                        frame.frame_number,
                    )
                    with self._lock:
                        self._error = error
                    continue
                processed += 1
                if self.result_sink is not None:
                    try:
                        self.result_sink(result)
                    except Exception:
                        LOGGER.exception(
                            "Live result sink rejected frame %d; continuing",
                            result.frame_number,
                        )
                with self._lock:
                    self._latest = result
                    self._version += 1
                    self._error = None
                LOGGER.info(
                    "frame=%d status=%s usable=%d inference=%.1f ms "
                    "recovery=%.1f ms total=%.1f ms",
                    result.frame_number,
                    result.status,
                    int(np.count_nonzero(result.pose3d_output.usable)),
                    result.timing_ms["inference"],
                    result.timing_ms["recovery"],
                    result.timing_ms["total"],
                )
        except Exception as error:
            LOGGER.exception("Live pose worker stopped unexpectedly")
            with self._lock:
                self._error = error
        finally:
            self._ready.set()
            self.source.close()
            self._finished.set()

    def latest_after(
        self,
        version: int,
    ) -> tuple[int, LivePoseResult | None]:
        with self._lock:
            if self._version == version:
                return version, None
            return self._version, self._latest

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(
                timeout=max(5.0, self.read_timeout_ms / 1000.0 + 2.0)
            )
        if self._thread.is_alive():
            LOGGER.warning(
                "Live pose worker did not stop before the shutdown timeout."
            )

    def wait(self, timeout_s: float | None = None) -> bool:
        return self._finished.wait(timeout=timeout_s)

    @property
    def finished(self) -> bool:
        return self._finished.is_set()


def _validate_live_frames(
    source: RGBDSource,
    processor: LivePoseProcessor,
    *,
    avatar_model: str,
    read_timeout_ms: int,
    max_frames: int,
) -> None:
    builder = (
        build_stick_figure_avatar
        if avatar_model == "stickman"
        else build_procedural_avatar
    )
    source.start()
    try:
        for index in range(max_frames):
            frame = source.read(timeout_ms=read_timeout_ms)
            result = processor.process(frame)
            avatar = builder(
                result.joints_application_m,
                result.pose3d_output.usable,
                ground_height_m=0.0,
            )
            usable_z = result.joints_application_m[
                result.pose3d_output.usable, 2
            ]
            foot_indices = np.asarray((15, 16, 20, 21, 22, 23, 24, 25))
            usable_feet = foot_indices[
                result.pose3d_output.usable[foot_indices]
            ]
            foot_height = (
                float(np.median(result.joints_application_m[usable_feet, 2]))
                if len(usable_feet)
                else float("nan")
            )
            LOGGER.info(
                "validate [%d/%d] frame=%d status=%s usable=%d parts=%d "
                "application_z=[%.3f,%.3f] foot_z=%.3f total=%.1f ms",
                index + 1,
                max_frames,
                frame.frame_number,
                result.status,
                int(np.count_nonzero(result.pose3d_output.usable)),
                avatar.primitive_count,
                float(np.min(usable_z)) if len(usable_z) else float("nan"),
                float(np.max(usable_z)) if len(usable_z) else float("nan"),
                foot_height,
                result.timing_ms["total"],
            )
    finally:
        source.close()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.validate_only and args.headless:
        raise ValueError("--validate-only and --headless cannot be combined.")
    if (
        args.mixamo_cache is not None
        and os.environ.get("EGL_PLATFORM") == "surfaceless"
    ):
        # The parent owns the GLFW mannequin window. The spawned Mixamo child
        # selects surfaceless EGL for itself before importing Open3D.
        os.environ.pop("EGL_PLATFORM")
        LOGGER.info(
            "Removed EGL_PLATFORM=surfaceless from the GLFW parent process; "
            "the Mixamo renderer will set it inside its child process."
        )

    live_config = load_yaml_mapping(args.live_config)["live"]
    camera_config = load_yaml_mapping(args.camera_config)["camera"]
    pose_config = load_yaml_mapping(args.pose_config)["pose"]
    tracking_config = load_yaml_mapping(args.tracking_config)["tracking"]
    source_config = live_config["source"]
    viewer_config = live_config["viewer"]
    publish_config = StickmanPublishConfig.from_mapping(
        live_config.get("websocket_publish"),
        enabled_override=args.publish_stickman,
        url_override=args.publish_url,
    )
    if args.headless and not publish_config.enabled:
        raise ValueError(
            "--headless requires WebSocket publishing; use "
            "--publish-url or enable live.websocket_publish."
        )
    avatar_model = args.avatar_model or viewer_config["avatar_model"]
    source_type = args.source or source_config.get("type", "directory")
    if args.input_dir is not None:
        source_type = "directory"
    if source_type == "sdk":
        source = LxCameraRGBDSource.from_mapping(
            source_config["sdk"],
            depth_scale=float(camera_config["depth_scale"]),
        )
        LOGGER.info("Live input: %s", source.source_id)
    elif source_type == "directory":
        input_directory = (
            args.input_dir.expanduser().resolve()
            if args.input_dir is not None
            else Path(source_config["directory"]).expanduser().resolve()
        )
        start_at = args.start_at or source_config["start_at"]
        intrinsics = CameraIntrinsics(
            **camera_config["intrinsics"],
            width=int(camera_config["depth_width"]),
            height=int(camera_config["depth_height"]),
        )
        source = DirectoryRGBDSource(
            input_directory,
            intrinsics=intrinsics,
            depth_scale=float(camera_config["depth_scale"]),
            start_at=start_at,
            poll_interval_s=float(source_config["poll_interval_ms"])
            / 1000.0,
            stable_interval_s=float(source_config["stable_interval_ms"])
            / 1000.0,
        )
        LOGGER.info("Live input directory: %s", input_directory)
    else:
        raise ValueError(
            f"Unsupported live source type {source_type!r}; "
            "expected directory or sdk."
        )
    extrinsics = ApplicationExtrinsics.from_mapping(
        live_config["application_extrinsics"]
    )
    recovery_mapping = camera_config.get("depth_recovery", {})
    recovery_method = (
        args.recovery_method
        or live_config.get("recovery_method")
        or recovery_mapping.get("method", "window_median")
    )
    pointcloud_config = PointCloudRecoveryConfig.from_mapping(
        recovery_mapping.get("pointcloud_cluster")
    )
    LOGGER.info("Depth recovery method: %s", recovery_method)

    LOGGER.info(
        "Application extrinsics ZYX: roll=%.2f pitch=%.2f yaw=%.2f "
        "translation_m=%s",
        extrinsics.roll_deg,
        extrinsics.pitch_deg,
        extrinsics.yaw_deg,
        extrinsics.translation_m.tolist(),
    )
    LOGGER.info("Initializing RTMPose once...")
    backend_started = time.perf_counter()
    backend = _build_backend(pose_config, args)
    LOGGER.info(
        "RTMPose initialized on %s in %.2f s",
        backend.device,
        time.perf_counter() - backend_started,
    )
    bone_calibrator, bone_constraint = _build_bone_components(tracking_config)
    processor = LivePoseProcessor(
        backend=backend,
        extrinsics=extrinsics,
        temporal_filter=_build_temporal_filter(tracking_config),
        presence_gate=PersonFramePresenceGate(
            FramePresenceConfig.from_mapping(
                tracking_config.get("frame_presence")
            )
        ),
        keypoint_threshold=float(pose_config["keypoint_threshold"]),
        min_depth_m=float(camera_config["min_depth_m"]),
        max_depth_m=float(camera_config["max_depth_m"]),
        depth_window_radius=int(camera_config["depth_window_radius"]),
        recovery_method=recovery_method,
        pointcloud_config=pointcloud_config,
        bone_calibrator=bone_calibrator,
        bone_constraint=bone_constraint,
    )
    read_timeout_ms = int(source_config["read_timeout_ms"])

    if args.validate_only:
        _validate_live_frames(
            source,
            processor,
            avatar_model=avatar_model,
            read_timeout_ms=read_timeout_ms,
            max_frames=args.max_frames or 1,
        )
        LOGGER.info("Live source statistics: %s", source.stats)
        return 0

    publisher = (
        StickmanWebSocketPublisher(publish_config)
        if publish_config.enabled
        else None
    )
    if publisher is not None:
        LOGGER.info(
            "Stickman publish target: %s event=%s topic=%s",
            publish_config.connection_url,
            publish_config.event,
            publish_config.topic,
        )

    worker = LatestPoseWorker(
        source,
        processor,
        read_timeout_ms=read_timeout_ms,
        max_frames=args.max_frames,
        result_sink=publisher.submit if publisher is not None else None,
    )

    if args.headless:
        try:
            if publisher is not None:
                publisher.start()
            worker.start()
            LOGGER.info("Headless stickman publishing started; Ctrl+C to stop.")
            while not worker.wait(timeout_s=0.5):
                pass
        except KeyboardInterrupt:
            LOGGER.info("Interrupted by user.")
        finally:
            worker.stop()
            if publisher is not None:
                publisher.stop()
        LOGGER.info("Live source statistics: %s", source.stats)
        if publisher is not None:
            LOGGER.info("Stickman publisher statistics: %s", publisher.stats)
        return 0

    from rgbd_avatar.visualization.live_mannequin import LiveMannequinRenderer
    rotation = extrinsics.rotation_application_from_camera
    renderer = LiveMannequinRenderer(
        avatar_model=avatar_model,
        sphere_resolution=int(viewer_config["sphere_resolution"]),
        grid_extent_m=float(viewer_config["grid_extent_m"]),
        grid_spacing_m=float(viewer_config["grid_spacing_m"]),
        window_width=int(viewer_config["window_width"]),
        window_height=int(viewer_config["window_height"]),
        show_rgb_views=bool(viewer_config.get("show_rgb_views", True)),
        rgb_view_scale=float(viewer_config.get("rgb_view_scale", 0.75)),
        keypoint_threshold=float(pose_config["keypoint_threshold"]),
        camera_forward_application=rotation[:, 2],
    )

    mixamo_viewer = None  # type: ignore[assignment]
    if args.mixamo_cache is not None:
        from rgbd_avatar.visualization.live_mixamo_process import (
            LiveMixamoViewerProcess,
        )
        mixamo_viewer = LiveMixamoViewerProcess(
            cache_path=args.mixamo_cache,
            port=args.mixamo_viewer_port,
            res=args.mixamo_res,
        )
        mixamo_viewer.start()
        LOGGER.info(
            "Live Mixamo viewer subprocess: %s",
            mixamo_viewer.url,
        )

    version = 0
    try:
        if publisher is not None:
            publisher.start()
        worker.start()
        renderer.open()
        while renderer.poll():
            version, result = worker.latest_after(version)
            if result is not None:
                renderer.update(result)
                if mixamo_viewer is not None:
                    mixamo_viewer.submit(result)
            time.sleep(0.005)
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")
    finally:
        worker.stop()
        if publisher is not None:
            publisher.stop()
        renderer.close()
        if mixamo_viewer is not None:
            mixamo_viewer.close()
    LOGGER.info("Live source statistics: %s", source.stats)
    if publisher is not None:
        LOGGER.info("Stickman publisher statistics: %s", publisher.stats)
    return 0
