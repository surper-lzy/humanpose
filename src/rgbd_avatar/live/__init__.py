"""Portable live RGB-D and 3DGS integration contracts."""

from .directory_source import (
    DirectoryRGBDSource,
    DirectorySourceStats,
    capture_timestamp_ns,
)
from .auto_calibration import (
    LiveAutoCalibrationConfig,
    LiveCalibrationResult,
    application_extrinsics_from_ground_plane,
    calibrate_live_camera,
)
from .extrinsics import ApplicationExtrinsics, rotation_zyx_from_degrees
from .mapping import LiveSceneMapper, OPTICAL_UPRIGHT_L_FROM_C
from .models import LivePosePacket, RGBDFrame, RGBDSource
from .multi_person_processor import (
    AdaptiveHybridConfig,
    KinematicFallbackConfig,
    LocalMultiPersonConfig,
    LocalMultiPersonPoseProcessor,
    LocalMultiPersonPoseResult,
    LocalPersonPoseResult,
    Pose3DQualityConfig,
)
from .processor import LivePoseProcessor, LivePoseResult, PoseBackend
from .lx_camera_source import (
    LxCameraRGBDSource,
    LxCameraSDKError,
    LxCameraSourceStats,
)
from .stickman_websocket import (
    StickmanPublishConfig,
    StickmanPublisherStats,
    StickmenWebSocketPublisher,
    StickmanWebSocketPublisher,
    build_stickmen_payload,
    build_stickman_payload,
)

__all__ = [
    "AdaptiveHybridConfig",
    "ApplicationExtrinsics",
    "LiveAutoCalibrationConfig",
    "LiveCalibrationResult",
    "DirectoryRGBDSource",
    "DirectorySourceStats",
    "LivePosePacket",
    "LivePoseProcessor",
    "LivePoseResult",
    "LiveSceneMapper",
    "KinematicFallbackConfig",
    "LocalMultiPersonConfig",
    "LocalMultiPersonPoseProcessor",
    "LocalMultiPersonPoseResult",
    "LocalPersonPoseResult",
    "Pose3DQualityConfig",
    "LxCameraRGBDSource",
    "LxCameraSDKError",
    "LxCameraSourceStats",
    "OPTICAL_UPRIGHT_L_FROM_C",
    "RGBDFrame",
    "RGBDSource",
    "PoseBackend",
    "StickmanPublishConfig",
    "StickmanPublisherStats",
    "StickmenWebSocketPublisher",
    "StickmanWebSocketPublisher",
    "build_stickmen_payload",
    "build_stickman_payload",
    "application_extrinsics_from_ground_plane",
    "calibrate_live_camera",
    "capture_timestamp_ns",
    "rotation_zyx_from_degrees",
]
