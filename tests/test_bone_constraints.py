import numpy as np
import pytest

from rgbd_avatar.pose import (
    HALPE26_CONSTRAINT_LINKS,
    HALPE26_NAMES,
    Pose3D,
)
from rgbd_avatar.tracking import (
    BoneLengthCalibrator,
    BoneLengthConstraint,
    BoneLengthPrior,
    TemporalPose3D,
)


def _raw_pose(
    start: int,
    end: int,
    length_m: float,
    *,
    depth_confidence: float = 1.0,
) -> Pose3D:
    count = len(HALPE26_NAMES)
    joints = np.full((count, 3), np.nan, dtype=np.float32)
    confidence = np.zeros(count, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    depth = np.full(count, np.nan, dtype=np.float32)
    depth_scores = np.zeros(count, dtype=np.float32)
    joints[start] = [0.0, 0.0, 1.0]
    joints[end] = [length_m, 0.0, 1.0]
    confidence[[start, end]] = depth_confidence
    valid[[start, end]] = True
    depth[[start, end]] = 1.0
    depth_scores[[start, end]] = depth_confidence
    return Pose3D(
        joints_m=joints,
        confidence=confidence,
        valid=valid,
        depth_m=depth,
        depth_confidence=depth_scores,
    )


def _temporal_pose(
    positions: dict[int, tuple[float, float, float]],
    *,
    observed: set[int] | None = None,
    predicted: set[int] | None = None,
    confidence: dict[int, float] | None = None,
) -> TemporalPose3D:
    count = len(HALPE26_NAMES)
    joints = np.full((count, 3), np.nan, dtype=np.float32)
    usable = np.zeros(count, dtype=bool)
    observed_mask = np.zeros(count, dtype=bool)
    predicted_mask = np.zeros(count, dtype=bool)
    scores = np.zeros(count, dtype=np.float32)
    ages = np.full(count, np.inf, dtype=np.float32)
    observed = observed or set()
    predicted = predicted or set()
    confidence = confidence or {}
    for index, xyz in positions.items():
        joints[index] = xyz
        usable[index] = index in observed or index in predicted
        observed_mask[index] = index in observed
        predicted_mask[index] = index in predicted
        scores[index] = confidence.get(index, 1.0)
        ages[index] = 0.0 if index in observed else 0.5
    return TemporalPose3D(
        joints_m=joints,
        confidence=scores,
        usable=usable,
        observed=observed_mask,
        predicted=predicted_mask,
        age_s=ages,
    )


def _prior(
    links: tuple[tuple[int, int], ...],
    lengths: tuple[float, ...],
    tolerances: tuple[float, ...] | None = None,
    ready: tuple[bool, ...] | None = None,
) -> BoneLengthPrior:
    count = len(links)
    return BoneLengthPrior(
        links=links,
        target_lengths_m=np.asarray(lengths),
        sample_count=np.full(count, 20),
        relative_mad=np.full(count, 0.01),
        tolerance_ratio=np.asarray(
            tolerances if tolerances is not None else (0.0,) * count
        ),
        ready=np.asarray(
            ready if ready is not None else (True,) * count
        ),
        frozen=np.ones(count, dtype=bool),
    )


def test_constraint_links_exclude_face_and_feet() -> None:
    constrained_joints = {
        index for link in HALPE26_CONSTRAINT_LINKS for index in link
    }
    assert not constrained_joints.intersection(range(0, 5))
    assert not constrained_joints.intersection(range(20, 26))
    assert len(HALPE26_CONSTRAINT_LINKS) == 14


def test_calibrator_uses_raw_confident_inliers_and_freezes() -> None:
    calibrator = BoneLengthCalibrator(
        links=((0, 1),),
        min_samples_per_bone=3,
        target_samples_per_bone=4,
        max_samples_per_bone=6,
        min_keypoint_confidence=0.6,
        min_depth_confidence=0.7,
        max_relative_mad=0.1,
        outlier_relative_tolerance=0.12,
        outlier_absolute_tolerance_m=0.02,
    )
    high_2d = np.ones(len(HALPE26_NAMES), dtype=np.float32)
    low_2d = high_2d.copy()
    low_2d[1] = 0.5

    calibrator.update(_raw_pose(0, 1, 1.0), low_2d)
    calibrator.update(
        _raw_pose(0, 1, 1.0, depth_confidence=0.5), high_2d
    )
    for length in (1.0, 1.02, 1.18, 0.98, 1.01):
        calibrator.update(_raw_pose(0, 1, length), high_2d)

    prior = calibrator.prior()
    summary = calibrator.summary()
    assert prior.ready[0]
    assert prior.frozen[0]
    assert prior.sample_count[0] == 4
    assert prior.target_lengths_m[0] == pytest.approx(
        np.median([1.0, 1.02, 0.98, 1.01])
    )
    assert summary["bones"][0]["outlier_count"] == 1
    assert summary["bones"][0]["observation_count"] == 5


def test_calibrator_warmup_and_high_mad_fail_closed() -> None:
    calibrator = BoneLengthCalibrator(
        links=((0, 1),),
        min_samples_per_bone=4,
        target_samples_per_bone=5,
        max_samples_per_bone=6,
        max_relative_mad=0.01,
        outlier_relative_tolerance=1.0,
        outlier_absolute_tolerance_m=0.01,
    )
    scores = np.ones(len(HALPE26_NAMES), dtype=np.float32)
    for length in (0.8, 1.0, 1.2):
        calibrator.update(_raw_pose(0, 1, length), scores)
    assert not calibrator.prior().ready[0]
    calibrator.update(_raw_pose(0, 1, 1.4), scores)
    assert not calibrator.prior().ready[0]


def test_high_anchor_is_unchanged_and_predicted_child_is_corrected() -> None:
    pose = _temporal_pose(
        {0: (0.0, 0.0, 1.0), 1: (1.5, 0.0, 1.0)},
        observed={0},
        predicted={1},
        confidence={0: 0.9, 1: 0.2},
    )
    constraint = BoneLengthConstraint(
        anchor_confidence=0.55,
        iterations=3,
        max_joint_correction_m=1.0,
        fixed_joint_indices=(),
    )
    result = constraint.apply(pose, _prior(((0, 1),), (1.0,)))

    np.testing.assert_array_equal(
        result.pose.joints_m[0], pose.joints_m[0]
    )
    assert np.linalg.norm(
        result.pose.joints_m[1] - result.pose.joints_m[0]
    ) == np.float32(1.0)
    assert not result.corrected[0]
    assert result.corrected[1]
    assert result.diagnostics["max_anchor_displacement_m"] == 0.0


def test_high_high_violation_is_reported_but_never_moved() -> None:
    pose = _temporal_pose(
        {0: (0.0, 0.0, 1.0), 1: (1.5, 0.0, 1.0)},
        observed={0, 1},
        confidence={0: 0.9, 1: 0.8},
    )
    result = BoneLengthConstraint(
        fixed_joint_indices=()
    ).apply(pose, _prior(((0, 1),), (1.0,)))

    np.testing.assert_array_equal(result.pose.joints_m, pose.joints_m)
    assert not np.any(result.corrected)
    assert (
        result.diagnostics["residual_after"]["violating_bone_count"]
        == 1
    )


def test_observed_projection_preserves_root_and_limits_stretched_bone() -> None:
    pose = _temporal_pose(
        {19: (0.0, 0.0, 1.0), 11: (1.5, 0.0, 1.0)},
        observed={19, 11},
        confidence={19: 0.9, 11: 0.9},
    )
    result = BoneLengthConstraint(
        iterations=3,
        max_joint_correction_m=1.0,
        fixed_joint_indices=(19,),
        project_observed=True,
    ).apply(pose, _prior(((19, 11),), (1.0,)))

    np.testing.assert_array_equal(result.pose.joints_m[19], pose.joints_m[19])
    assert np.linalg.norm(
        result.pose.joints_m[11] - result.pose.joints_m[19]
    ) == pytest.approx(1.0)
    assert result.corrected[11]
    assert result.diagnostics["project_observed"] is True


def test_two_predicted_endpoints_keep_midpoint_and_root_can_be_fixed() -> None:
    free_pose = _temporal_pose(
        {0: (0.0, 0.0, 1.0), 1: (2.0, 0.0, 1.0)},
        predicted={0, 1},
        confidence={0: 0.2, 1: 0.2},
    )
    free_result = BoneLengthConstraint(
        iterations=3,
        fixed_joint_indices=(),
    ).apply(free_pose, _prior(((0, 1),), (1.0,)))
    np.testing.assert_allclose(
        np.mean(free_result.pose.joints_m[[0, 1]], axis=0),
        [1.0, 0.0, 1.0],
    )

    root_pose = _temporal_pose(
        {19: (0.0, 0.0, 1.0), 11: (2.0, 0.0, 1.0)},
        predicted={19, 11},
    )
    root_result = BoneLengthConstraint(
        iterations=3,
        fixed_joint_indices=(19,),
    ).apply(root_pose, _prior(((19, 11),), (1.0,)))
    np.testing.assert_array_equal(
        root_result.pose.joints_m[19], root_pose.joints_m[19]
    )
    assert root_result.diagnostics["max_root_displacement_m"] == 0.0


def test_unready_missing_and_correction_cap_are_safe() -> None:
    pose = _temporal_pose(
        {0: (0.0, 0.0, 1.0), 1: (3.0, 0.0, 1.0)},
        observed={0},
        predicted={1},
    )
    constraint = BoneLengthConstraint(
        iterations=3,
        max_joint_correction_m=0.15,
        fixed_joint_indices=(),
    )
    unready = constraint.apply(
        pose, _prior(((0, 1),), (1.0,), ready=(False,))
    )
    np.testing.assert_array_equal(unready.pose.joints_m, pose.joints_m)

    capped = constraint.apply(pose, _prior(((0, 1),), (1.0,)))
    assert capped.correction_m[1] == np.float32(0.15)
    assert capped.diagnostics["correction_limited_joint_count"] == 1
    assert np.isnan(capped.pose.joints_m[2]).all()
    assert not capped.pose.usable[2]
