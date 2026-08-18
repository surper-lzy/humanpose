import numpy as np
import pytest

from rgbd_avatar.avatar import StickFigureConfig, build_stick_figure_avatar


def _standing_pose() -> tuple[np.ndarray, np.ndarray]:
    joints = np.full((26, 3), np.nan, dtype=np.float64)
    values = {
        0: (0.00, 2.00, 1.67),
        5: (-0.21, 2.00, 1.45),
        6: (0.21, 2.00, 1.45),
        7: (-0.40, 2.00, 1.18),
        8: (0.40, 2.00, 1.18),
        9: (-0.48, 2.00, 0.91),
        10: (0.48, 2.00, 0.91),
        11: (-0.15, 2.00, 0.94),
        12: (0.15, 2.00, 0.94),
        13: (-0.14, 2.01, 0.50),
        14: (0.14, 2.01, 0.50),
        15: (-0.14, 2.02, 0.09),
        16: (0.14, 2.02, 0.09),
        17: (0.00, 2.00, 1.82),
        18: (0.00, 2.00, 1.48),
        19: (0.00, 2.00, 0.94),
        20: (-0.14, 1.85, 0.01),
        21: (0.14, 1.85, 0.01),
        22: (-0.10, 1.86, 0.01),
        23: (0.10, 1.86, 0.01),
        24: (-0.14, 2.08, 0.01),
        25: (0.14, 2.08, 0.01),
    }
    for index, value in values.items():
        joints[index] = value
    return joints, np.isfinite(joints).all(axis=1)


def test_stick_figure_builds_rods_joint_balls_and_head() -> None:
    joints, usable = _standing_pose()

    avatar = build_stick_figure_avatar(
        joints,
        usable,
        ground_height_m=0.0,
    )

    capsule_names = {primitive.name for primitive in avatar.capsules}
    ellipsoid_names = {primitive.name for primitive in avatar.ellipsoids}
    assert {
        "left_upper_arm",
        "right_forearm",
        "left_thigh",
        "right_shin",
        "shoulder_bar",
        "hip_bar",
        "torso_axis",
        "left_foot",
        "right_foot",
    } <= capsule_names
    assert {"head", "hip_center", "shoulder_center"} <= ellipsoid_names
    assert avatar.primitive_count == 29
    assert all(primitive.radius_m > 0 for primitive in avatar.capsules)
    assert all(
        np.all(primitive.radii_m > 0)
        for primitive in avatar.ellipsoids
    )
    colors = {
        primitive.color
        for primitive in (*avatar.capsules, *avatar.ellipsoids)
    }
    assert all(max(color) <= 0.015 for color in colors)


def test_stick_figure_missing_pose_and_invalid_config_are_safe() -> None:
    joints = np.full((26, 3), np.nan)
    assert (
        build_stick_figure_avatar(joints, np.zeros(26, dtype=bool)).primitive_count
        == 0
    )
    with pytest.raises(ValueError, match="positive"):
        build_stick_figure_avatar(
            np.zeros((26, 3)),
            np.ones(26, dtype=bool),
            config=StickFigureConfig(rod_radius_ratio=0.0),
        )
