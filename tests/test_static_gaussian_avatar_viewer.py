import numpy as np

from rgbd_avatar.pipeline.static_gaussian_avatar_viewer import (
    sample_mesh_surface,
)


def test_sample_mesh_surface_returns_points_on_triangle() -> None:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )
    faces = np.array([[0, 1, 2]])

    points = sample_mesh_surface(vertices, faces, sample_count=100, seed=4)

    assert points.shape == (100, 3)
    np.testing.assert_allclose(points[:, 2], 0.0)
    assert np.all(points[:, :2] >= 0.0)
    assert np.all(np.sum(points[:, :2], axis=1) <= 1.0 + 1e-12)
