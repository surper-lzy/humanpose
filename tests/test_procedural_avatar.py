import numpy as np
import pytest

from rgbd_avatar.avatar import (
    ProceduralAvatarConfig,
    build_procedural_avatar,
)


def _standing_halpe26() -> tuple[np.ndarray, np.ndarray]:
    joints = np.full((26, 3), np.nan, dtype=np.float64)
    values = {
        0: (0.00, 2.00, 1.67),
        1: (-0.035, 2.00, 1.70),
        2: (0.035, 2.00, 1.70),
        3: (-0.085, 2.00, 1.68),
        4: (0.085, 2.00, 1.68),
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


def test_full_pose_builds_finite_anatomical_primitives() -> None:
    joints, usable = _standing_halpe26()

    avatar = build_procedural_avatar(
        joints,
        usable,
        ground_height_m=0.0,
    )

    capsule_names = {primitive.name for primitive in avatar.capsules}
    ellipsoid_names = {primitive.name for primitive in avatar.ellipsoids}
    assert capsule_names == {
        "left_upper_arm",
        "left_forearm",
        "right_upper_arm",
        "right_forearm",
        "left_thigh",
        "left_shin",
        "right_thigh",
        "right_shin",
        "neck",
    }
    assert {
        "torso",
        "abdomen",
        "pelvis",
        "head",
        "left_hand",
        "right_hand",
        "left_foot",
        "right_foot",
    } <= ellipsoid_names
    assert avatar.primitive_count == 27

    for primitive in avatar.capsules:
        assert primitive.length_m > 0.025
        assert primitive.radius_m > 0.0
        assert primitive.resolved_end_radius_m > 0.0
        assert np.isfinite(primitive.start_m).all()
        assert np.isfinite(primitive.end_m).all()
    for primitive in avatar.ellipsoids:
        assert np.isfinite(primitive.center_m).all()
        assert np.all(primitive.radii_m > 0.0)
        np.testing.assert_allclose(
            primitive.rotation.T @ primitive.rotation,
            np.eye(3),
            atol=1e-7,
        )
        assert np.linalg.det(primitive.rotation) == pytest.approx(1.0)

    colors = {
        primitive.color
        for primitive in (*avatar.capsules, *avatar.ellipsoids)
    }
    assert len(colors) == 1
    assert any(
        primitive.radius_m > primitive.resolved_end_radius_m
        for primitive in avatar.capsules
        if primitive.name != "neck"
    )


def test_capsule_endpoints_follow_the_input_skeleton() -> None:
    joints, usable = _standing_halpe26()

    avatar = build_procedural_avatar(joints, usable)
    left_forearm = next(
        primitive
        for primitive in avatar.capsules
        if primitive.name == "left_forearm"
    )

    np.testing.assert_allclose(left_forearm.start_m, joints[7])
    np.testing.assert_allclose(left_forearm.end_m, joints[9])
    assert left_forearm.length_m == pytest.approx(
        np.linalg.norm(joints[9] - joints[7])
    )


def test_ground_contact_lifts_foot_volume_without_moving_skeleton() -> None:
    joints, usable = _standing_halpe26()
    original = joints.copy()
    joints[20, 2] = -0.025
    joints[22, 2] = -0.020
    joints[24, 2] = -0.030

    avatar = build_procedural_avatar(
        joints,
        usable,
        ground_height_m=0.0,
    )
    left_foot = next(
        primitive
        for primitive in avatar.ellipsoids
        if primitive.name == "left_foot"
    )

    assert left_foot.center_m[2] - left_foot.radii_m[2] >= -1e-12
    np.testing.assert_allclose(joints[:, :2], original[:, :2])


def test_missing_joints_remove_only_affected_parts() -> None:
    joints, usable = _standing_halpe26()
    usable[[8, 10, 14, 16, 21, 23, 25]] = False

    avatar = build_procedural_avatar(joints, usable)
    names = {
        primitive.name
        for primitive in (*avatar.capsules, *avatar.ellipsoids)
    }

    assert "right_upper_arm" not in names
    assert "right_forearm" not in names
    assert "right_thigh" not in names
    assert "right_shin" not in names
    assert "right_foot" not in names
    assert "left_upper_arm" in names
    assert "left_foot" in names
    assert "torso" in names


def test_empty_pose_and_invalid_shapes_are_handled_explicitly() -> None:
    joints = np.full((26, 3), np.nan)
    avatar = build_procedural_avatar(joints, np.zeros(26, dtype=bool))
    assert avatar.primitive_count == 0

    with pytest.raises(ValueError, match="joints_m must have shape"):
        build_procedural_avatar(np.zeros((25, 3)), np.ones(26, dtype=bool))
    with pytest.raises(ValueError, match="configuration must be positive"):
        build_procedural_avatar(
            np.zeros((26, 3)),
            np.ones(26, dtype=bool),
            config=ProceduralAvatarConfig(joint_radius_ratio=-0.1),
        )
