import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import recover_pose3d_from_depth_connected
from rgbd_avatar.pose import Pose2D


def _intrinsics(width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=100.0,
        fy=100.0,
        cx=(width - 1) * 0.5,
        cy=(height - 1) * 0.5,
        width=width,
        height=height,
    )


def _pose(
    keypoints: np.ndarray,
    *,
    bbox: np.ndarray | None = None,
) -> Pose2D:
    return Pose2D(
        keypoints=np.asarray(keypoints, dtype=np.float32),
        scores=np.full(26, 0.95, dtype=np.float32),
        bbox_xyxy=(
            np.asarray(bbox, dtype=np.float32)
            if bbox is not None
            else np.array([2.0, 2.0, 18.0, 18.0], dtype=np.float32)
        ),
        bbox_score=0.95,
    )


def test_depth_connected_selects_center_surface_over_large_background() -> None:
    depth = np.full((21, 21), 3.0, dtype=np.float32)
    depth[9:12, 9:12] = 2.0
    keypoints = np.tile(np.array([[10.0, 10.0]], dtype=np.float32), (26, 1))

    pose3d = recover_pose3d_from_depth_connected(
        _pose(keypoints),
        depth,
        _intrinsics(21, 21),
        person_depth_hint_m=2.0,
    )

    assert np.all(pose3d.valid)
    np.testing.assert_allclose(pose3d.depth_m, 2.0)


def test_depth_connected_labels_joint_window_union_once(monkeypatch) -> None:
    import rgbd_avatar.depth.depth_connected as module

    calls = 0
    original = module._connected_components

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "_connected_components", counted)
    depth = np.full((61, 61), 2.0, dtype=np.float32)
    keypoints = np.column_stack(
        (
            np.linspace(15.0, 45.0, 26),
            np.linspace(10.0, 50.0, 26),
        )
    ).astype(np.float32)

    pose3d = recover_pose3d_from_depth_connected(
        _pose(
            keypoints,
            bbox=np.array([5.0, 5.0, 56.0, 56.0], dtype=np.float32),
        ),
        depth,
        _intrinsics(61, 61),
        person_depth_hint_m=2.0,
    )

    assert calls == 1
    assert np.count_nonzero(pose3d.valid) >= 18


def test_depth_connected_history_is_soft_and_can_recover() -> None:
    depth = np.full((21, 21), 3.0, dtype=np.float32)
    depth[9:12, 9:12] = 2.0
    keypoints = np.tile(np.array([[10.0, 10.0]], dtype=np.float32), (26, 1))

    pose3d = recover_pose3d_from_depth_connected(
        _pose(keypoints),
        depth,
        _intrinsics(21, 21),
        person_depth_hint_m=2.0,
        expected_depths_m=np.full(26, 3.0, dtype=np.float32),
    )

    assert np.all(pose3d.valid)
    np.testing.assert_allclose(pose3d.depth_m, 2.0)


def test_depth_connected_rejects_impossible_forearm_length() -> None:
    depth = np.full((101, 101), 2.0, dtype=np.float32)
    keypoints = np.tile(np.array([[50.0, 50.0]], dtype=np.float32), (26, 1))
    keypoints[5] = [40.0, 50.0]
    keypoints[7] = [50.0, 50.0]
    keypoints[9] = [90.0, 50.0]

    pose3d = recover_pose3d_from_depth_connected(
        _pose(
            keypoints,
            bbox=np.array([1.0, 1.0, 100.0, 100.0], dtype=np.float32),
        ),
        depth,
        _intrinsics(101, 101),
        person_depth_hint_m=2.0,
    )

    assert pose3d.valid[5]
    assert pose3d.valid[7]
    assert not pose3d.valid[9]
