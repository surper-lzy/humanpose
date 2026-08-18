import numpy as np
import pytest

from rgbd_avatar.pose.halpe26 import (
    HALPE26_INDEX,
    HALPE26_LINKS,
    HALPE26_NAMES,
)
from rgbd_avatar.pose.models import Pose2D


def test_halpe26_definition_is_consistent() -> None:
    assert len(HALPE26_NAMES) == 26
    assert len(set(HALPE26_NAMES)) == 26
    assert HALPE26_INDEX["hip"] == 19
    assert HALPE26_INDEX["left_heel"] == 24
    assert HALPE26_INDEX["right_heel"] == 25
    assert all(0 <= index < 26 for link in HALPE26_LINKS for index in link)


def test_pose2d_serialization() -> None:
    pose = Pose2D(
        keypoints=np.zeros((26, 2), dtype=np.float32),
        scores=np.ones(26, dtype=np.float32),
        bbox_xyxy=np.array([1, 2, 3, 4], dtype=np.float32),
        bbox_score=0.9,
    )

    serialized = pose.to_dict(score_threshold=0.3)
    assert serialized["keypoint_format"] == "halpe26"
    assert len(serialized["keypoints"]) == 26
    assert all(point["valid"] for point in serialized["keypoints"])


def test_pose2d_rejects_wrong_keypoint_count() -> None:
    with pytest.raises(ValueError, match="Halpe26"):
        Pose2D(
            keypoints=np.zeros((17, 2), dtype=np.float32),
            scores=np.ones(17, dtype=np.float32),
            bbox_xyxy=np.zeros(4, dtype=np.float32),
            bbox_score=1.0,
        )
