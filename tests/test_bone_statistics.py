import numpy as np

from rgbd_avatar.pose import HALPE26_NAMES
from rgbd_avatar.tracking import BoneLengthAccumulator


def test_bone_length_accumulator_reports_robust_statistics() -> None:
    accumulator = BoneLengthAccumulator(links=((0, 1),))
    count = len(HALPE26_NAMES)

    for length in (1.0, 2.0):
        joints = np.full((count, 3), np.nan)
        valid = np.zeros(count, dtype=bool)
        confidence = np.zeros(count)
        joints[0] = [0.0, 0.0, 0.0]
        joints[1] = [length, 0.0, 0.0]
        valid[[0, 1]] = True
        confidence[[0, 1]] = 1.0
        accumulator.update(joints, valid, confidence)

    summary = accumulator.summary()
    bone = summary["bones"][0]
    assert summary["total_sample_count"] == 2
    assert summary["bones_with_samples"] == 1
    assert summary["median_relative_mad"] == 1.0 / 3.0
    assert bone["sample_count"] == 2
    assert bone["median_m"] == 1.5
    assert bone["mad_m"] == 0.5
    assert bone["relative_mad"] == 1.0 / 3.0
