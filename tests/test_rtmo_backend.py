from pathlib import Path

import numpy as np
import pytest

from rgbd_avatar.pose import (
    RTMOBackend,
    RTMOBackendConfig,
    coco17_to_halpe26,
)


def _coco17(*, score: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    keypoints = np.stack(
        (
            np.linspace(10.0, 90.0, 17, dtype=np.float32),
            np.linspace(20.0, 180.0, 17, dtype=np.float32),
        ),
        axis=1,
    )
    scores = np.full(17, score, dtype=np.float32)
    return keypoints, scores


def test_coco17_mapping_preserves_body_and_builds_torso_anchors() -> None:
    keypoints, scores = _coco17()

    mapped_points, mapped_scores = coco17_to_halpe26(keypoints, scores)

    np.testing.assert_allclose(mapped_points[:17], keypoints)
    np.testing.assert_allclose(mapped_scores[:17], scores)
    np.testing.assert_allclose(mapped_points[17], keypoints[0])
    np.testing.assert_allclose(
        mapped_points[18], 0.5 * (keypoints[5] + keypoints[6])
    )
    np.testing.assert_allclose(
        mapped_points[19], 0.5 * (keypoints[11] + keypoints[12])
    )
    assert np.all(mapped_scores[20:] == 0.0)


def test_coco17_mapping_uses_conservative_anchor_scores() -> None:
    keypoints, scores = _coco17()
    scores[5] = 0.4
    scores[6] = 0.8
    scores[11] = 0.7
    scores[12] = 0.2

    _, mapped_scores = coco17_to_halpe26(keypoints, scores)

    assert mapped_scores[18] == pytest.approx(0.4)
    assert mapped_scores[19] == pytest.approx(0.2)


def test_rtmo_config_rejects_non_coco_valid_count() -> None:
    with pytest.raises(ValueError, match=r"\[1, 17\]"):
        RTMOBackendConfig(min_valid_keypoints=18)


class _FakeInferencer:
    def __init__(self, predictions: list[dict]) -> None:
        self.predictions = predictions
        self.kwargs = None

    def __call__(self, _image, **kwargs):
        self.kwargs = kwargs
        return iter(({"predictions": [self.predictions]},))


def _fake_backend(predictions: list[dict]) -> RTMOBackend:
    backend = RTMOBackend.__new__(RTMOBackend)
    backend.device = "cpu"
    backend.model_name = "test-rtmo"
    backend.bbox_threshold = 0.3
    backend.nms_threshold = 0.65
    backend.keypoint_threshold = 0.3
    backend.min_valid_keypoints = 10
    backend.min_mean_keypoint_score = 0.3
    backend._inferencer = _FakeInferencer(predictions)
    return backend


def test_rtmo_infer_maps_sorts_and_derives_missing_bbox() -> None:
    keypoints, scores = _coco17()
    lower = {
        "keypoints": keypoints.tolist(),
        "keypoint_scores": scores.tolist(),
        "bbox": [[4.0, 8.0, 96.0, 192.0]],
        "bbox_score": 0.7,
    }
    higher_without_bbox = {
        "keypoints": (keypoints + np.array([100.0, 0.0])).tolist(),
        "keypoint_scores": scores.tolist(),
        "bbox_score": 0.95,
    }
    backend = _fake_backend([lower, higher_without_bbox])

    poses = backend.infer(np.zeros((240, 320, 3), dtype=np.uint8))

    assert [pose.bbox_score for pose in poses] == pytest.approx([0.95, 0.7])
    assert all(pose.keypoints.shape == (26, 2) for pose in poses)
    assert np.all(poses[0].scores[20:] == 0.0)
    assert poses[0].bbox_xyxy[0] < np.min(keypoints[:, 0] + 100.0)
    assert backend._inferencer.kwargs["bbox_thr"] == pytest.approx(0.3)
    assert backend._inferencer.kwargs["nms_thr"] == pytest.approx(0.65)


def test_rtmo_infer_filters_low_quality_prediction() -> None:
    keypoints, scores = _coco17(score=0.1)
    backend = _fake_backend(
        [
            {
                "keypoints": keypoints.tolist(),
                "keypoint_scores": scores.tolist(),
                "bbox": [4.0, 8.0, 96.0, 192.0],
                "bbox_score": 0.9,
            }
        ]
    )

    assert backend.infer(np.zeros((240, 320, 3), dtype=np.uint8)) == []


def test_rtmo_backend_rejects_missing_explicit_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="RTMO checkpoint not found"):
        RTMOBackend(
            RTMOBackendConfig(
                model_checkpoint=tmp_path / "missing.pth",
            )
        )
