import numpy as np

from rgbd_avatar.scene import build_manual_scene_alignment
from rgbd_avatar.pipeline.static_avatar_export import transform_static_avatar

from test_scene_alignment import _cache, _placement


def test_transform_static_avatar_exports_scene_coordinates() -> None:
    cache = _cache()
    alignment = build_manual_scene_alignment(_placement())

    frame, vertices_g, joints_g, faces, metadata = transform_static_avatar(
        cache,
        alignment,
    )

    assert frame == 1
    np.testing.assert_allclose(
        vertices_g,
        alignment.transform_points_w_to_g(cache.vertices_display_m[1]),
    )
    np.testing.assert_allclose(
        joints_g,
        alignment.transform_points_w_to_g(cache.joints_display_m[1]),
    )
    np.testing.assert_array_equal(faces, cache.faces)
    assert metadata["coordinate_system"] == "3dgs_world"
    assert metadata["scale_g_per_m"] == alignment.scale_g_per_m
    assert metadata["vertex_count"] == 4
