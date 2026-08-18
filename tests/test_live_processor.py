import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.live import (
    ApplicationExtrinsics,
    LivePoseProcessor,
    RGBDFrame,
)
from rgbd_avatar.pose import Pose2D
from rgbd_avatar.tracking import (
    FramePresenceConfig,
    PersonFramePresenceGate,
    Pose3DTemporalFilter,
)
from rgbd_avatar.visualization.live_mannequin import build_live_rgb_views


class _FixedPoseBackend:
    def __init__(self, pose: Pose2D) -> None:
        self.pose = pose

    def infer(self, image: np.ndarray) -> list[Pose2D]:
        assert image.shape == (60, 80, 3)
        return [self.pose]


def test_live_processor_lifts_filters_and_applies_extrinsics() -> None:
    intrinsics = CameraIntrinsics(100.0, 100.0, 40.0, 30.0, 80, 60)
    pose2d = Pose2D(
        keypoints=np.tile(np.array([[40.0, 30.0]], dtype=np.float32), (26, 1)),
        scores=np.full(26, 0.9, dtype=np.float32),
        bbox_xyxy=np.array([20.0, 10.0, 60.0, 50.0], dtype=np.float32),
        bbox_score=0.95,
    )
    processor = LivePoseProcessor(
        backend=_FixedPoseBackend(pose2d),
        extrinsics=ApplicationExtrinsics(
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            translation_m=np.array([1.0, 2.0, 3.0]),
        ),
        temporal_filter=Pose3DTemporalFilter(
            reset_gap_s=2.0,
            max_prediction_s=1.0,
        ),
        presence_gate=PersonFramePresenceGate(
            FramePresenceConfig(enabled=False)
        ),
        keypoint_threshold=0.3,
        min_depth_m=0.3,
        max_depth_m=6.0,
        depth_window_radius=1,
        recovery_method="window_median",
    )
    frame = RGBDFrame(
        rgb_bgr=np.zeros((60, 80, 3), dtype=np.uint8),
        depth_m=np.ones((60, 80), dtype=np.float32),
        intrinsics=intrinsics,
        timestamp_ns=1_000_000_000,
        frame_number=0,
        source_id="test",
    )

    result = processor.process(frame)

    assert result.status == "ok"
    assert np.count_nonzero(result.pose3d_output.usable) == 26
    np.testing.assert_allclose(
        result.joints_application_m,
        np.tile([1.0, 2.0, 4.0], (26, 1)),
    )
    raw, overlay = build_live_rgb_views(
        result,
        keypoint_threshold=0.3,
        scale=0.5,
    )
    assert raw.shape == overlay.shape == (30, 40, 3)
    assert np.count_nonzero(raw) == 0
    assert np.count_nonzero(overlay) > 0
