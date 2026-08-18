from pathlib import Path

import numpy as np
import pytest

from rgbd_avatar.scene import (
    ColmapCamera,
    GaussianAlignmentView,
    load_sparse_cameras,
    quaternion_wxyz_to_rotation,
)


def _view() -> GaussianAlignmentView:
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    depth = np.full((4, 5), 2.0, dtype=np.float32)
    alpha = np.full((4, 5), 0.8, dtype=np.float32)
    camera_to_world = np.eye(4)
    camera_to_world[:3, 3] = [10.0, 20.0, 30.0]
    return GaussianAlignmentView(
        rgb_uint8=rgb,
        expected_depth_g=depth,
        alpha=alpha,
        intrinsic_matrix=np.array(
            [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        camera_to_world_g=camera_to_world,
        camera_name="frame.png",
        metadata={"render_mode": "RGB+ED"},
    )


def test_gaussian_view_unprojects_expected_projection_depth() -> None:
    view = _view()

    point_g, depth_g, alpha = view.unproject_pixel(
        (1, 1),
        patch_radius=0,
    )

    np.testing.assert_allclose(point_g, [11.0, 21.0, 32.0])
    assert depth_g == pytest.approx(2.0)
    assert alpha == pytest.approx(0.8)


def test_gaussian_view_rejects_background_click() -> None:
    view = _view()
    view.alpha[1, 1] = 0.0

    with pytest.raises(ValueError, match="no visible Gaussian depth"):
        view.unproject_pixel((1, 1), patch_radius=0, minimum_alpha=0.1)


def test_gaussian_view_intersects_pixel_ray_with_plane() -> None:
    view = _view()

    point_g = view.intersect_pixel_with_plane(
        (1, 1),
        plane_normal_g=[0.0, 0.0, 2.0],
        plane_offset_g=-68.0,
    )

    np.testing.assert_allclose(point_g, [12.0, 22.0, 34.0])


def test_gaussian_view_rejects_plane_behind_camera() -> None:
    with pytest.raises(ValueError, match="behind camera"):
        _view().intersect_pixel_with_plane(
            (1, 1),
            plane_normal_g=[0.0, 0.0, 1.0],
            plane_offset_g=-29.0,
        )


def test_gaussian_view_cache_round_trip(tmp_path: Path) -> None:
    view = _view()
    path = tmp_path / "alignment_view.npz"

    view.save(path)
    loaded = GaussianAlignmentView.load(path)

    np.testing.assert_array_equal(loaded.rgb_uint8, view.rgb_uint8)
    np.testing.assert_allclose(loaded.expected_depth_g, view.expected_depth_g)
    np.testing.assert_allclose(loaded.intrinsic_matrix, view.intrinsic_matrix)
    assert loaded.camera_name == "frame.png"
    assert loaded.metadata["render_mode"] == "RGB+ED"


def test_colmap_pinhole_intrinsics_scale_to_render_resolution() -> None:
    camera = ColmapCamera(
        camera_id=1,
        model="PINHOLE",
        width=1000,
        height=500,
        parameters=np.array([800.0, 810.0, 500.0, 250.0]),
    )

    intrinsic = camera.intrinsic_matrix(width=500, height=250)

    np.testing.assert_allclose(
        intrinsic,
        [[400.0, 0.0, 250.0], [0.0, 405.0, 125.0], [0.0, 0.0, 1.0]],
    )


def test_colmap_quaternion_identity_is_proper() -> None:
    rotation = quaternion_wxyz_to_rotation(np.array([1.0, 0.0, 0.0, 0.0]))

    np.testing.assert_allclose(rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_experimental_3dgs_colmap_assets_load() -> None:
    sparse = (
        Path(__file__).resolve().parents[2]
        / "data/3DGS/sparse/0"
    )

    cameras, images = load_sparse_cameras(sparse)

    assert len(cameras) == 1
    assert len(images) == 261
    assert images[0].name == "000001.jpg"
    camera = cameras[images[0].camera_id]
    assert camera.model == "PINHOLE"
    assert (camera.width, camera.height) == (1916, 1078)
