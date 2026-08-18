from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.scene import (
    ManualScenePlacement,
    SceneAlignment,
    build_manual_scene_alignment,
    first_avatar_ground_anchor,
    fit_ground_plane,
)


def _placement() -> ManualScenePlacement:
    return ManualScenePlacement(
        known_point_a_g=np.array([0.0, 2.0, 0.0]),
        known_point_b_g=np.array([4.0, 2.0, 0.0]),
        known_distance_m=2.0,
        ground_points_g=np.array(
            [
                [0.0, 2.0, 0.0],
                [4.0, 2.0, 0.0],
                [0.0, 2.0, 5.0],
                [3.0, 2.0, 4.0],
            ]
        ),
        up_reference_g=np.array([0.0, 5.0, 0.0]),
        spawn_point_g=np.array([1.0, 2.2, 3.0]),
        forward_point_g=np.array([1.0, 2.0, 6.0]),
        avatar_anchor_w_m=np.array([0.5, 1.0, 0.0]),
        description="synthetic unrelated scene",
    )


def _cache() -> SMPLSequenceCache:
    vertices = np.full((2, 4, 3), np.nan, dtype=np.float32)
    joints = np.full((2, 24, 3), np.nan, dtype=np.float32)
    vertices[1] = 0.0
    joints[1] = 0.0
    joints[1, 0] = [0.6, 1.2, 0.9]
    joints[1, 10] = [0.4, 1.0, 0.05]
    joints[1, 11] = [0.8, 1.2, 0.06]
    return SMPLSequenceCache(
        frame_indices=np.array([4, 5]),
        present=np.array([False, True]),
        vertices_display_m=vertices,
        joints_display_m=joints,
        faces=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        body_pose=np.full((2, 69), np.nan),
        global_orient=np.full((2, 3), np.nan),
        translation_native_m=np.full((2, 3), np.nan),
        target_counts=np.array([0, 18]),
        error_mean_m=np.array([np.nan, 0.01]),
        error_p95_m=np.array([np.nan, 0.02]),
        error_max_m=np.array([np.nan, 0.03]),
        scale=1.0,
        metadata={},
    )


def test_ground_plane_uses_above_ground_reference_for_normal_sign() -> None:
    normal, offset, residuals = fit_ground_plane(
        [[0, 2, 0], [2, 2, 0], [0, 2, 3], [2, 2, 3]],
        up_reference_g=[0, 4, 0],
    )

    np.testing.assert_allclose(normal, [0, 1, 0], atol=1e-8)
    assert offset == pytest.approx(-2.0)
    np.testing.assert_allclose(residuals, 0.0, atol=1e-8)


def test_manual_alignment_maps_anchor_axes_and_metric_motion() -> None:
    alignment = build_manual_scene_alignment(_placement())

    assert alignment.scale_g_per_m == pytest.approx(2.0)
    np.testing.assert_allclose(alignment.ground_normal_g, [0, 1, 0])
    np.testing.assert_allclose(alignment.spawn_point_g, [1, 2, 3])
    np.testing.assert_allclose(
        alignment.transform_points_w_to_g(_placement().avatar_anchor_w_m),
        alignment.spawn_point_g,
    )
    np.testing.assert_allclose(alignment.forward_g, [0, 0, 1])
    np.testing.assert_allclose(alignment.right_g, [-1, 0, 0])

    one_metre_forward_w = _placement().avatar_anchor_w_m + [0, 1, 0]
    moved_g = alignment.transform_points_w_to_g(one_metre_forward_w)
    np.testing.assert_allclose(
        moved_g - alignment.spawn_point_g,
        [0, 0, 2],
    )


def test_alignment_round_trips_points_and_preserves_nan_frames() -> None:
    alignment = build_manual_scene_alignment(_placement())
    points_w = np.array(
        [
            [0.2, 0.3, 0.4],
            [np.nan, np.nan, np.nan],
            [-1.0, 2.0, 0.0],
        ]
    )

    points_g = alignment.transform_points_w_to_g(points_w)
    restored = alignment.transform_points_g_to_w(points_g)

    np.testing.assert_allclose(restored[[0, 2]], points_w[[0, 2]])
    assert np.isnan(points_g[1]).all()
    assert np.isnan(restored[1]).all()


def test_alignment_json_round_trip(tmp_path: Path) -> None:
    alignment = build_manual_scene_alignment(
        _placement(),
        metadata={"scene_ply": "scene/point_cloud.ply"},
    )
    path = tmp_path / "scene_alignment.json"

    alignment.save(path)
    loaded = SceneAlignment.load(path)

    assert loaded.scale_g_per_m == pytest.approx(alignment.scale_g_per_m)
    np.testing.assert_allclose(
        loaded.transform_g_from_w,
        alignment.transform_g_from_w,
    )
    assert loaded.metadata["scene_ply"] == "scene/point_cloud.ply"


def test_camera_pose_rotation_is_not_scaled() -> None:
    alignment = build_manual_scene_alignment(_placement())
    camera_w = np.eye(4)
    camera_w[:3, 3] = [0.2, 0.3, 1.4]

    camera_g = alignment.camera_to_world_w_to_g(camera_w)

    assert np.linalg.det(camera_g[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(
        camera_g[:3, 3],
        alignment.transform_points_w_to_g(camera_w[:3, 3]),
    )


def test_first_avatar_anchor_uses_first_present_feet_on_ground() -> None:
    cache = _cache()

    feet = first_avatar_ground_anchor(cache, mode="feet")
    pelvis = first_avatar_ground_anchor(cache, mode="pelvis")

    np.testing.assert_allclose(feet, [0.6, 1.1, 0.0])
    np.testing.assert_allclose(pelvis, [0.6, 1.2, 0.0])


def test_manual_alignment_rejects_collinear_ground_points() -> None:
    with pytest.raises(ValueError, match="collinear"):
        fit_ground_plane(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0]],
            up_reference_g=[0, 1, 0],
        )


def test_manual_alignment_rejects_ground_outlier() -> None:
    placement = _placement()
    noisy_ground = placement.ground_points_g.copy()
    noisy_ground[-1, 1] = 3.0

    with pytest.raises(ValueError, match=r"one floor plane.*G\d="):
        build_manual_scene_alignment(
            replace(placement, ground_points_g=noisy_ground)
        )


def test_manual_alignment_robustly_excludes_one_of_five_ground_picks() -> None:
    placement = _placement()
    ground_with_outlier = np.vstack(
        (placement.ground_points_g, [8.0, 3.0, 8.0])
    )

    alignment = build_manual_scene_alignment(
        replace(placement, ground_points_g=ground_with_outlier)
    )

    assert alignment.metadata["ground_point_inliers"] == [
        True,
        True,
        True,
        True,
        False,
    ]
    np.testing.assert_allclose(alignment.ground_normal_g, [0.0, 1.0, 0.0])


def test_manual_alignment_validates_vertical_known_height() -> None:
    vertical = replace(
        _placement(),
        known_point_a_g=np.array([0.0, 2.0, 0.0]),
        known_point_b_g=np.array([0.0, 6.0, 0.0]),
        known_length_direction="vertical",
    )
    alignment = build_manual_scene_alignment(vertical)
    assert alignment.metadata["known_length_angle_error_deg"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="Known vertical length"):
        build_manual_scene_alignment(
            replace(_placement(), known_length_direction="vertical")
        )


def test_manual_alignment_rejects_spawn_off_floor() -> None:
    with pytest.raises(ValueError, match="Spawn pick"):
        build_manual_scene_alignment(
            replace(_placement(), spawn_point_g=np.array([1.0, 3.0, 3.0]))
        )
