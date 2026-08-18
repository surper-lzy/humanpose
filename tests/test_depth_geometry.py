import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import (
    deproject_pixel,
    depth_to_organized_point_cloud,
    sample_joint_depth,
)


def make_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        fx=100.0,
        fy=200.0,
        cx=2.0,
        cy=1.0,
        width=5,
        height=3,
    )


def test_deproject_principal_point() -> None:
    point = deproject_pixel(2.0, 1.0, 3.0, make_intrinsics())
    np.testing.assert_allclose(point, [0.0, 0.0, 3.0])


def test_deproject_offset_pixel() -> None:
    point = deproject_pixel(3.0, 3.0, 2.0, make_intrinsics())
    np.testing.assert_allclose(point, [0.02, 0.02, 2.0])


def test_depth_sampler_ignores_invalid_values_and_outlier() -> None:
    depth = np.array(
        [
            [0.0, 2.0, 2.0],
            [2.0, np.nan, 8.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )
    sample = sample_joint_depth(
        depth,
        u=1,
        v=1,
        radius=1,
        min_depth_m=0.2,
        max_depth_m=6.0,
    )
    assert sample is not None
    assert sample.depth_m == 2.0
    assert sample.valid_count == 6


def test_depth_sampler_chooses_nearest_supported_edge_cluster() -> None:
    depth = np.array(
        [
            [1.4, 1.4, 3.4],
            [1.4, 1.4, 3.4],
            [3.4, 3.4, 3.4],
        ],
        dtype=np.float32,
    )
    sample = sample_joint_depth(depth, u=1, v=1, radius=1)

    assert sample is not None
    assert np.isclose(sample.depth_m, 1.4)
    assert sample.valid_count == 4
    assert sample.used_nearest_cluster


def test_depth_sampler_uses_expected_depth_to_select_far_cluster() -> None:
    depth = np.array(
        [
            [1.4, 1.4, 3.4],
            [1.4, 1.4, 3.4],
            [3.4, 3.4, 3.4],
        ],
        dtype=np.float32,
    )
    sample = sample_joint_depth(
        depth,
        u=1,
        v=1,
        radius=1,
        expected_depth_m=3.3,
    )

    assert sample is not None
    assert np.isclose(sample.depth_m, 3.4)
    assert sample.valid_count == 5
    assert sample.used_nearest_cluster


def test_depth_sampler_rejects_surface_far_from_expected_depth() -> None:
    depth = np.ones((3, 3), dtype=np.float32)

    sample = sample_joint_depth(
        depth,
        u=1,
        v=1,
        radius=1,
        expected_depth_m=2.0,
    )

    assert sample is None


def test_organized_point_cloud_preserves_pixel_grid() -> None:
    intrinsics = make_intrinsics()
    depth = np.ones((3, 5), dtype=np.float32) * 2.0
    depth[0, 0] = 0.0
    points = depth_to_organized_point_cloud(depth, intrinsics)

    assert points.shape == (3, 5, 3)
    assert np.isnan(points[0, 0]).all()
    np.testing.assert_allclose(points[1, 2], [0.0, 0.0, 2.0])
