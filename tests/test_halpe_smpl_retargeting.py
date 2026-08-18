import numpy as np

from rgbd_avatar.retargeting import (
    calibrate_halpe_smpl_profile,
    retarget_halpe26_to_smpl,
)


def _halpe_pose() -> np.ndarray:
    joints = np.zeros((26, 3), dtype=np.float64)
    joints[19] = [0.0, 0.0, 1.00]
    joints[18] = [0.0, 0.0, 1.60]
    joints[17] = [0.0, 0.0, 1.80]
    joints[11], joints[12] = [0.20, 0.0, 0.95], [-0.20, 0.0, 0.95]
    joints[5], joints[6] = [0.30, 0.0, 1.50], [-0.30, 0.0, 1.50]
    joints[13], joints[14] = [0.20, 0.0, 0.55], [-0.20, 0.0, 0.55]
    joints[15], joints[16] = [0.20, 0.02, 0.10], [-0.20, 0.02, 0.10]
    joints[7], joints[8] = [0.55, 0.0, 1.30], [-0.55, 0.0, 1.30]
    joints[9], joints[10] = [0.75, 0.0, 1.10], [-0.75, 0.0, 1.10]
    joints[24], joints[25] = [0.20, -0.08, 0.08], [-0.20, -0.08, 0.08]
    joints[20], joints[21] = [0.20, 0.15, 0.08], [-0.20, 0.15, 0.08]
    joints[22], joints[23] = [0.24, 0.14, 0.08], [-0.24, 0.14, 0.08]
    return joints


def _smpl_rest() -> np.ndarray:
    joints = np.zeros((45, 3), dtype=np.float64)
    joints[0] = [0.0, 0.0, 1.00]
    joints[1], joints[2] = [0.08, 0.0, 0.94], [-0.08, 0.0, 0.94]
    joints[12] = [0.0, 0.0, 1.50]
    joints[15] = [0.0, 0.0, 1.68]
    joints[16], joints[17] = [0.22, 0.0, 1.46], [-0.22, 0.0, 1.46]
    joints[4], joints[5] = [0.08, 0.0, 0.56], [-0.08, 0.0, 0.56]
    joints[7], joints[8] = [0.08, 0.0, 0.12], [-0.08, 0.0, 0.12]
    joints[18], joints[19] = [0.48, 0.0, 1.46], [-0.48, 0.0, 1.46]
    joints[20], joints[21] = [0.72, 0.0, 1.46], [-0.72, 0.0, 1.46]
    joints[29], joints[30], joints[31] = (
        [0.08, 0.18, 0.08],
        [0.11, 0.17, 0.08],
        [0.08, -0.05, 0.08],
    )
    joints[32], joints[33], joints[34] = (
        [-0.08, 0.18, 0.08],
        [-0.11, 0.17, 0.08],
        [-0.08, -0.05, 0.08],
    )
    return joints


def _profile(joints: np.ndarray):
    count = 12
    return calibrate_halpe_smpl_profile(
        [joints.copy() for _ in range(count)],
        [np.full(26, 0.9) for _ in range(count)],
        [np.ones(26, dtype=bool) for _ in range(count)],
        [np.zeros(26, dtype=bool) for _ in range(count)],
    )


def test_semantic_retarget_preserves_smpl_widths_and_limb_lengths() -> None:
    observed = _halpe_pose()
    rest = _smpl_rest()
    result = retarget_halpe26_to_smpl(
        observed,
        np.full(26, 0.9),
        np.ones(26, dtype=bool),
        np.zeros(26, dtype=bool),
        rest_joints_display_m=rest,
        profile=_profile(observed),
    )
    points = dict(zip(result.smpl_joint_indices, result.points_display_m))

    np.testing.assert_allclose(points[0], observed[19])
    np.testing.assert_allclose(
        np.linalg.norm(points[1] - points[2]),
        np.linalg.norm(rest[1] - rest[2]),
    )
    for start, end in ((1, 4), (4, 7), (2, 5), (5, 8), (16, 18), (18, 20)):
        np.testing.assert_allclose(
            np.linalg.norm(points[end] - points[start]),
            np.linalg.norm(rest[end] - rest[start]),
        )
    assert len(result.smpl_direction_pairs) == 4


def test_semantic_retarget_rejects_outlier_bone_and_dependent_foot() -> None:
    baseline = _halpe_pose()
    outlier = baseline.copy()
    outlier[16] = outlier[14] + [0.0, 0.0, -0.05]
    result = retarget_halpe26_to_smpl(
        outlier,
        np.full(26, 0.9),
        np.ones(26, dtype=bool),
        np.zeros(26, dtype=bool),
        rest_joints_display_m=_smpl_rest(),
        profile=_profile(baseline),
    )

    assert "right_shin" in result.rejected_segments
    assert 5 in result.smpl_joint_indices
    assert 8 not in result.smpl_joint_indices
    assert not np.any(np.isin(result.smpl_direction_pairs, [32, 33, 34]))


def test_semantic_retarget_keeps_plausible_length_outlier_as_soft_direction() -> None:
    baseline = _halpe_pose()
    shorter = baseline.copy()
    shorter[16] = shorter[14] + [0.0, 0.0, -0.30]
    result = retarget_halpe26_to_smpl(
        shorter,
        np.full(26, 0.9),
        np.ones(26, dtype=bool),
        np.zeros(26, dtype=bool),
        rest_joints_display_m=_smpl_rest(),
        profile=_profile(baseline),
    )

    assert "right_shin" not in result.rejected_segments
    assert "right_shin" in result.soft_segments
    assert 8 in result.smpl_joint_indices
