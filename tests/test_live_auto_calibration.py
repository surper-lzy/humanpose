import json

import numpy as np
import pytest

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import GroundPlaneConfig
from rgbd_avatar.live import (
    ApplicationExtrinsics,
    LiveAutoCalibrationConfig,
    RGBDFrame,
    application_extrinsics_from_ground_plane,
    calibrate_live_camera,
)
from rgbd_avatar.depth.ground_plane import GroundPlaneEstimate


class _EmptyPoseBackend:
    def infer(self, image: np.ndarray) -> list:
        return []


class _FrameSource:
    def __init__(self, frames: list[RGBDFrame]) -> None:
        self.frames = iter(frames)
        self.started = False
        self.start_count = 0
        self.source_id = "sdk:test-camera"

    def start(self) -> None:
        self.started = True
        self.start_count += 1

    def read(self, timeout_ms: int = 1000) -> RGBDFrame:
        assert self.started
        return next(self.frames)

    def close(self) -> None:
        self.started = False


def _ground_estimate() -> GroundPlaneEstimate:
    normal = np.array([0.01, -0.94, -0.341], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    return GroundPlaneEstimate(
        normal_camera=normal,
        offset_m=1.25,
        inlier_count=1000,
        candidate_count=1100,
        inlier_ratio=1000 / 1100,
        residual_median_m=0.004,
        residual_p95_m=0.012,
        residual_rms_m=0.006,
        tilt_from_camera_up_deg=20.0,
    )


def test_ground_extrinsics_preserve_heading_and_map_floor_to_zero() -> None:
    fallback = ApplicationExtrinsics(
        roll_deg=90.38,
        pitch_deg=-179.83,
        yaw_deg=89.95,
        translation_m=np.array([0.2, -0.1, 0.8]),
    )
    estimate = _ground_estimate()

    calibrated = application_extrinsics_from_ground_plane(
        estimate,
        fallback,
    )

    rotation = calibrated.rotation_application_from_camera
    np.testing.assert_allclose(rotation[2], estimate.normal_camera, atol=1e-7)
    assert calibrated.translation_m[0] == pytest.approx(0.2)
    assert calibrated.translation_m[1] == pytest.approx(-0.1)
    assert calibrated.translation_m[2] == pytest.approx(1.25)
    fallback_x = fallback.rotation_application_from_camera[0]
    expected_x = fallback_x - np.dot(
        fallback_x,
        estimate.normal_camera,
    ) * estimate.normal_camera
    expected_x /= np.linalg.norm(expected_x)
    np.testing.assert_allclose(rotation[0], expected_x, atol=1e-7)

    points = np.array(
        [[0.0, estimate.offset_m / -estimate.normal_camera[1], 0.0]]
    )
    transformed = calibrated.transform_points(points)
    assert transformed[0, 2] == pytest.approx(0.0, abs=1e-7)


def test_live_calibration_records_runtime_intrinsics_and_floor(tmp_path) -> None:
    intrinsics = CameraIntrinsics(90.0, 90.0, 39.5, 24.0, 80, 60)
    normal = np.array([0.0, -0.94, -0.342], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    offset = 1.2
    rows, columns = np.indices((intrinsics.height, intrinsics.width))
    ray_x = (columns - intrinsics.cx) / intrinsics.fx
    ray_y = (rows - intrinsics.cy) / intrinsics.fy
    denominator = normal[0] * ray_x + normal[1] * ray_y + normal[2]
    depth = np.full(denominator.shape, np.nan, dtype=np.float32)
    valid = denominator < -1e-6
    depth[valid] = (-offset / denominator[valid]).astype(np.float32)
    depth[(depth < 0.3) | (depth > 6.0)] = np.nan
    frames = [
        RGBDFrame(
            rgb_bgr=np.zeros((60, 80, 3), dtype=np.uint8),
            depth_m=depth,
            intrinsics=intrinsics,
            timestamp_ns=index * 66_666_667,
            frame_number=index,
            source_id="sdk:test-camera",
        )
        for index in range(3)
    ]
    source = _FrameSource(frames)
    source.start()
    output = tmp_path / "calibration.json"

    result = calibrate_live_camera(
        source,
        _EmptyPoseBackend(),
        heading_reference=ApplicationExtrinsics(
            roll_deg=90.0,
            pitch_deg=180.0,
            yaw_deg=90.0,
            translation_m=np.zeros(3),
        ),
        config=LiveAutoCalibrationConfig(
            enabled=True,
            sample_frame_count=3,
            max_attempt_frame_count=3,
            min_inlier_ratio=0.8,
            max_residual_p95_m=0.01,
        ),
        ground_config=GroundPlaneConfig(
            lower_image_fraction=0.45,
            pixel_stride=2,
            max_points_per_frame=2000,
            max_total_points=6000,
            ransac_iterations=100,
            inlier_distance_m=0.005,
            min_inlier_count=300,
        ),
        read_timeout_ms=100,
        output_path=output,
        source_already_started=True,
    )

    assert source.start_count == 1
    assert result.sampled_frame_count == 3
    assert result.intrinsics == intrinsics
    assert result.ground_plane.offset_m == pytest.approx(offset, abs=1e-4)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["intrinsics_source"] == "camera_sdk_aligned_rgb_stream"
    assert payload["intrinsics"]["width"] == 80
    assert payload["ground_plane"]["source_frame_count"] == 3
