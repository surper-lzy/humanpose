import math

import numpy as np
import pytest

from rgbd_avatar.pose import HALPE26_NAMES, Pose3D
from rgbd_avatar.tracking import OneEuroFilter3D, Pose3DTemporalFilter


def _pose_with_joint(
    xyz: tuple[float, float, float],
    *,
    index: int = 0,
    confidence: float = 1.0,
) -> Pose3D:
    count = len(HALPE26_NAMES)
    joints = np.full((count, 3), np.nan, dtype=np.float32)
    scores = np.zeros(count, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    depth = np.full(count, np.nan, dtype=np.float32)
    depth_confidence = np.zeros(count, dtype=np.float32)
    joints[index] = xyz
    scores[index] = confidence
    valid[index] = True
    depth[index] = xyz[2]
    depth_confidence[index] = 1.0
    return Pose3D(
        joints_m=joints,
        confidence=scores,
        valid=valid,
        depth_m=depth,
        depth_confidence=depth_confidence,
    )


def _pose_with_joints(
    positions: dict[int, tuple[float, float, float]],
) -> Pose3D:
    count = len(HALPE26_NAMES)
    joints = np.full((count, 3), np.nan, dtype=np.float32)
    confidence = np.zeros(count, dtype=np.float32)
    valid = np.zeros(count, dtype=bool)
    depth = np.full(count, np.nan, dtype=np.float32)
    depth_confidence = np.zeros(count, dtype=np.float32)
    for index, xyz in positions.items():
        joints[index] = xyz
        confidence[index] = 1.0
        valid[index] = True
        depth[index] = xyz[2]
        depth_confidence[index] = 1.0
    return Pose3D(
        joints_m=joints,
        confidence=confidence,
        valid=valid,
        depth_m=depth,
        depth_confidence=depth_confidence,
    )


def test_one_euro_first_sample_passes_through_and_beta_zero_is_closed_form() -> None:
    filter_3d = OneEuroFilter3D(
        min_cutoff_hz=1.0,
        beta=0.0,
        derivative_cutoff_hz=1.0,
    )
    first = filter_3d.update(0.0, np.array([0.0, 0.0, 0.0]))
    second = filter_3d.update(1.0, np.array([1.0, 0.0, 0.0]))
    expected_alpha = 2.0 * math.pi / (1.0 + 2.0 * math.pi)

    np.testing.assert_allclose(first, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(second, [expected_alpha, 0.0, 0.0])


def test_one_euro_uses_rotation_equivariant_shared_xyz_cutoff() -> None:
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    original = OneEuroFilter3D()
    rotated = OneEuroFilter3D()
    samples = [
        np.array([0.1, 0.2, 1.0]),
        np.array([0.4, 0.1, 1.2]),
        np.array([0.8, -0.2, 1.4]),
    ]

    for timestamp, sample in enumerate(samples):
        output = original.update(float(timestamp), sample)
        rotated_output = rotated.update(
            float(timestamp), rotation @ sample
        )
        np.testing.assert_allclose(
            rotated_output, rotation @ output, atol=1e-12
        )


def test_shared_pose_cutoff_keeps_joint_response_phase_equal() -> None:
    temporal = Pose3DTemporalFilter(
        min_cutoff_hz=1.5,
        beta=1.0,
        derivative_cutoff_hz=1.0,
        shared_cutoff=True,
        shared_speed_percentile=75.0,
    )
    temporal.update(
        0.0,
        _pose_with_joints({
            5: (0.0, 0.0, 1.0),
            7: (1.0, 0.0, 1.0),
        }),
    )
    result = temporal.update(
        1.0 / 16.0,
        _pose_with_joints({
            5: (0.05, 0.0, 1.0),
            7: (1.0, 1.0, 1.0),
        }),
    )

    shoulder_response = result.joints_m[5, 0] / 0.05
    elbow_response = result.joints_m[7, 1]
    assert shoulder_response == pytest.approx(elbow_response, abs=1e-6)


def test_shared_pose_cutoff_percentile_is_validated() -> None:
    with pytest.raises(ValueError, match="shared_speed_percentile"):
        Pose3DTemporalFilter(shared_speed_percentile=0.0)


def test_temporal_filter_predicts_then_expires_and_reacquires() -> None:
    temporal = Pose3DTemporalFilter(
        min_cutoff_hz=0.5,
        beta=0.0,
        reset_gap_s=2.0,
        max_prediction_s=1.1,
    )
    first = temporal.update(0.0, _pose_with_joint((0.0, 0.0, 1.0)))
    second = temporal.update(0.5, _pose_with_joint((0.5, 0.0, 1.0)))
    predicted = temporal.update(1.0, None)
    expired = temporal.update(1.7, None)
    reacquired = temporal.update(2.0, _pose_with_joint((2.0, 0.0, 1.0)))

    np.testing.assert_allclose(first.joints_m[0], [0.0, 0.0, 1.0])
    assert second.observed[0] and not second.predicted[0]
    assert 0.0 < second.joints_m[0, 0] < 0.5
    assert predicted.predicted[0] and not predicted.observed[0]
    assert predicted.age_s[0] == pytest.approx(0.5)
    assert 0.0 < predicted.confidence[0] < second.confidence[0]
    assert not expired.usable[0]
    assert np.isnan(expired.joints_m[0]).all()
    np.testing.assert_allclose(reacquired.joints_m[0], [2.0, 0.0, 1.0])


def test_temporal_filter_resets_only_at_large_capture_gap() -> None:
    temporal = Pose3DTemporalFilter(
        min_cutoff_hz=0.5,
        beta=0.0,
        reset_gap_s=2.0,
        max_prediction_s=0.0,
    )
    temporal.update(0.0, _pose_with_joint((0.0, 0.0, 1.0)))
    sparse = temporal.update(
        1.364, _pose_with_joint((1.0, 0.0, 1.0))
    )
    after_gap = temporal.update(
        3.983, _pose_with_joint((3.0, 0.0, 1.0))
    )

    assert not sparse.reset_occurred
    assert sparse.joints_m[0, 0] < 1.0
    assert after_gap.reset_occurred
    np.testing.assert_allclose(after_gap.joints_m[0], [3.0, 0.0, 1.0])
    assert temporal.discontinuity_reset_count == 1


def test_temporal_filter_track_termination_discards_all_predictions() -> None:
    temporal = Pose3DTemporalFilter(max_prediction_s=1.1)
    temporal.update(0.0, _pose_with_joint((0.0, 0.0, 1.0)))
    terminated = temporal.terminate_track(0.5)
    after_termination = temporal.update(1.0, None)

    assert terminated.reset_occurred
    assert not np.any(terminated.usable)
    assert np.isnan(terminated.joints_m).all()
    assert not np.any(after_termination.usable)
    assert not np.any(after_termination.predicted)


@pytest.mark.parametrize("timestamp", [0.0, -1.0])
def test_temporal_filter_rejects_repeated_or_reversed_time(
    timestamp: float,
) -> None:
    temporal = Pose3DTemporalFilter()
    temporal.update(0.0, None)

    with pytest.raises(ValueError, match="strictly increasing"):
        temporal.update(timestamp, None)
