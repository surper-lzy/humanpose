import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.pose import (
    HandPose2D,
    Pose2D,
    build_hand_bbox,
    hand_observation_quality,
    recover_hand_pose3d,
)


def _body_pose() -> Pose2D:
    keypoints = np.zeros((26, 2), dtype=np.float32)
    scores = np.ones(26, dtype=np.float32)
    keypoints[7] = [80.0, 60.0]
    keypoints[9] = [100.0, 80.0]
    keypoints[8] = [40.0, 60.0]
    keypoints[10] = [20.0, 80.0]
    return Pose2D(
        keypoints=keypoints,
        scores=scores,
        bbox_xyxy=np.array([10.0, 10.0, 110.0, 110.0]),
        bbox_score=0.9,
    )


def test_hand_bbox_extends_past_body_wrist_and_stays_in_image() -> None:
    bbox = build_hand_bbox(
        _body_pose(),
        "left",
        image_width=200,
        image_height=160,
        keypoint_threshold=0.3,
        minimum_crop_px=48.0,
    )

    assert bbox is not None
    center = 0.5 * (bbox[:2] + bbox[2:])
    assert center[0] > 100.0
    assert center[1] > 80.0
    assert np.all(bbox >= 0.0)
    assert bbox[2] <= 199.0
    assert bbox[3] <= 159.0


def test_hand_depth_recovery_rejects_background_using_wrist_anchor() -> None:
    pose = HandPose2D(
        side="left",
        keypoints=np.tile([8.0, 8.0], (21, 1)),
        scores=np.ones(21),
        bbox_xyxy=np.array([0.0, 0.0, 16.0, 16.0]),
    )
    depth = np.full((16, 16), 4.0, dtype=np.float32)
    intrinsics = CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=8.0,
        cy=8.0,
        width=16,
        height=16,
    )

    recovered = recover_hand_pose3d(
        pose,
        depth,
        intrinsics,
        anchor_depth_m=2.0,
        max_anchor_delta_m=0.1,
        fallback_depth_confidence=0.3,
        anchor_point_m=np.array([0.1, -0.2, 2.0]),
    )

    assert recovered.valid.all()
    np.testing.assert_allclose(recovered.depth_m, 2.0)
    np.testing.assert_allclose(recovered.joints_m[0], [0.1, -0.2, 2.0])
    np.testing.assert_allclose(recovered.joints_m[:, 2], 2.0)
    np.testing.assert_allclose(recovered.depth_confidence, 0.3)


def test_hand_depth_fallback_is_not_shifted_twice() -> None:
    keypoints = np.tile([12.0, 12.0], (21, 1))
    keypoints[0] = [4.0, 4.0]
    pose = HandPose2D(
        side="right",
        keypoints=keypoints,
        scores=np.ones(21),
        bbox_xyxy=np.array([0.0, 0.0, 16.0, 16.0]),
    )
    depth = np.full((16, 16), 4.0, dtype=np.float32)
    depth[2:7, 2:7] = 2.0
    intrinsics = CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=8.0,
        cy=8.0,
        width=16,
        height=16,
    )

    recovered = recover_hand_pose3d(
        pose,
        depth,
        intrinsics,
        anchor_depth_m=2.1,
        max_anchor_delta_m=0.15,
        anchor_point_m=np.array([0.1, -0.2, 2.1]),
    )

    assert recovered.valid.all()
    np.testing.assert_allclose(recovered.joints_m[:, 2], 2.1)


def test_hand_depth_recovery_rejects_low_support_sample() -> None:
    keypoints = np.tile([12.0, 12.0], (21, 1))
    keypoints[0] = [4.0, 4.0]
    pose = HandPose2D(
        side="left",
        keypoints=keypoints,
        scores=np.ones(21),
        bbox_xyxy=np.array([0.0, 0.0, 16.0, 16.0]),
    )
    depth = np.full((16, 16), np.nan, dtype=np.float32)
    depth[2:7, 2:7] = 2.0
    depth[12, 12] = 2.05
    intrinsics = CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=8.0,
        cy=8.0,
        width=16,
        height=16,
    )

    recovered = recover_hand_pose3d(
        pose,
        depth,
        intrinsics,
        anchor_depth_m=2.0,
        anchor_point_m=np.array([0.0, 0.0, 2.0]),
        minimum_sample_confidence=0.20,
    )

    assert recovered.valid.all()
    np.testing.assert_allclose(recovered.joints_m[:, 2], 2.0)
    assert np.isclose(recovered.depth_confidence[1], 0.35)


def test_hand_depth_recovery_rejects_impossible_phalanx_jump() -> None:
    keypoints = np.tile([8.0, 8.0], (21, 1))
    keypoints[1] = [20.0, 8.0]
    keypoints[2] = [32.0, 8.0]
    pose = HandPose2D(
        side="right",
        keypoints=keypoints,
        scores=np.ones(21),
        bbox_xyxy=np.array([0.0, 0.0, 40.0, 16.0]),
    )
    depth = np.full((48, 48), np.nan, dtype=np.float32)
    depth[6:11, 6:11] = 2.0
    depth[6:11, 18:23] = 2.0
    depth[6:11, 30:35] = 2.12
    intrinsics = CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=24.0,
        cy=24.0,
        width=48,
        height=48,
    )

    recovered = recover_hand_pose3d(
        pose,
        depth,
        intrinsics,
        anchor_depth_m=2.0,
        max_anchor_delta_m=0.15,
        anchor_point_m=np.array([0.0, 0.0, 2.0]),
    )

    assert recovered.valid.all()
    np.testing.assert_allclose(recovered.joints_m[1:3, 2], 2.0)
    assert np.isclose(recovered.depth_confidence[2], 0.35)


def _plausible_hand() -> np.ndarray:
    joints = np.zeros((21, 3), dtype=np.float64)
    joints[:, 2] = 2.0
    bases = {
        1: (-0.035, 0.025),
        5: (-0.030, 0.060),
        9: (0.000, 0.075),
        13: (0.030, 0.065),
        17: (0.050, 0.045),
    }
    for base, (x, y) in bases.items():
        joints[base, :2] = [x, y]
        for offset in range(1, 4):
            joints[base + offset, :2] = [x, y + 0.025 * offset]
    return joints


def test_hand_quality_rejects_collapsed_palm_frame() -> None:
    joints = _plausible_hand()
    valid = np.ones(21, dtype=bool)
    ok, reason, _ = hand_observation_quality(joints, valid)
    assert ok and reason == "ok"

    joints[5] = joints[17] + [0.005, 0.0, 0.0]
    ok, reason, _ = hand_observation_quality(joints, valid)
    assert not ok
    assert reason == "implausible_palm_width"


def test_hand_quality_rejects_multiple_collapsed_finger_links() -> None:
    joints = _plausible_hand()
    valid = np.ones(21, dtype=bool)
    joints[7] = joints[6]
    joints[8] = joints[7]

    ok, reason, _ = hand_observation_quality(joints, valid)

    assert not ok
    assert reason == "collapsed_finger_links"
