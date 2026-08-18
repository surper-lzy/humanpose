import numpy as np

from rgbd_avatar.pipeline.gaussian_avatar_composite import (
    composite_avatar,
    rasterize_mesh_camera,
)
from rgbd_avatar.scene import GaussianAlignmentView


def _view(scene_depth: float = 5.0) -> GaussianAlignmentView:
    return GaussianAlignmentView(
        rgb_uint8=np.full((8, 8, 3), 255, dtype=np.uint8),
        expected_depth_g=np.full((8, 8), scene_depth, dtype=np.float32),
        alpha=np.ones((8, 8), dtype=np.float32),
        intrinsic_matrix=np.array(
            [[4.0, 0.0, 4.0], [0.0, 4.0, 4.0], [0.0, 0.0, 1.0]]
        ),
        camera_to_world_g=np.eye(4),
        camera_name="synthetic.png",
    )


def test_rasterize_mesh_camera_writes_triangle_depth() -> None:
    vertices = np.array(
        [[-1.0, -1.0, 2.0], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]]
    )

    rgb, depth = rasterize_mesh_camera(
        vertices,
        np.array([[0, 1, 2]]),
        _view().intrinsic_matrix,
        height=8,
        width=8,
    )

    assert np.isfinite(depth).any()
    assert np.allclose(depth[np.isfinite(depth)], 2.0)
    assert np.any(rgb > 0.0)


def test_composite_avatar_respects_gaussian_depth() -> None:
    view = _view(scene_depth=3.0)
    mesh_rgb = np.zeros((8, 8, 3), dtype=np.float32)
    mesh_rgb[..., 0] = 1.0
    mesh_depth = np.full((8, 8), np.inf, dtype=np.float32)
    mesh_depth[3, 3] = 2.0
    mesh_depth[4, 4] = 4.0

    output, visible = composite_avatar(view, mesh_rgb, mesh_depth, opacity=1.0)

    assert visible[3, 3]
    assert not visible[4, 4]
    np.testing.assert_array_equal(output[3, 3], [255, 0, 0])
    np.testing.assert_array_equal(output[4, 4], [255, 255, 255])
