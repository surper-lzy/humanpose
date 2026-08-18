from pathlib import Path

import cv2
import numpy as np
import pytest

from rgbd_avatar.pose.tensorrt_backend import (
    TensorRTHalpe26BackendConfig,
    bbox_center_scale,
    decode_simcc,
    pose_warp_matrix,
    preprocess_detector,
    select_person_detections,
)


def test_preprocess_detector_shape_dtype_and_channel_order() -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = 103
    image[..., 1] = 116
    image[..., 2] = 124

    tensor = preprocess_detector(image)

    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous
    expected = (
        np.asarray([103.0, 116.0, 124.0], dtype=np.float32)
        - np.asarray([103.53, 116.28, 123.675], dtype=np.float32)
    ) / np.asarray([57.375, 57.12, 58.395], dtype=np.float32)
    np.testing.assert_allclose(tensor[0, :, 0, 0], expected)


def test_select_person_detections_maps_640_space_to_camera() -> None:
    dets = np.asarray(
        [
            [
                [280.0, 160.0, 380.0, 600.0, 0.9],
                [100.0, 100.0, 200.0, 200.0, 0.8],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=np.float32,
    )
    labels = np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32)

    boxes, scores = select_person_detections(
        dets,
        labels,
        image_wh=(816, 612),
        output_reference_wh=(640, 640),
        score_threshold=0.3,
        max_persons=2,
    )

    assert boxes.shape == (1, 4)
    np.testing.assert_allclose(
        boxes[0],
        [357.0, 153.0, 484.5, 573.75],
        atol=1e-5,
    )
    np.testing.assert_allclose(scores, [0.9])


def test_pose_bbox_and_affine_match_expected_geometry() -> None:
    center, scale = bbox_center_scale(
        np.asarray([280.0, 160.0, 380.0, 600.0], dtype=np.float32)
    )
    np.testing.assert_allclose(center, [330.0, 380.0])
    np.testing.assert_allclose(scale, [412.5, 550.0])

    matrix = pose_warp_matrix(center, scale)
    mapped = cv2.transform(
        np.asarray(
            [[[center[0], center[1]], [center[0] - scale[0] / 2, center[1]]]],
            dtype=np.float32,
        ),
        matrix,
    )[0]
    np.testing.assert_allclose(mapped[0], [96.0, 128.0], atol=1e-4)
    np.testing.assert_allclose(mapped[1], [0.0, 128.0], atol=1e-4)


def test_decode_simcc_maps_crop_coordinates_back_to_image() -> None:
    simcc_x = np.zeros((1, 26, 384), dtype=np.float32)
    simcc_y = np.zeros((1, 26, 512), dtype=np.float32)
    simcc_x[:, :, 192] = 0.8
    simcc_y[:, :, 256] = 0.7
    centers = np.asarray([[330.0, 380.0]], dtype=np.float32)
    scales = np.asarray([[412.5, 550.0]], dtype=np.float32)

    keypoints, scores = decode_simcc(
        simcc_x, simcc_y, centers, scales
    )

    assert keypoints.shape == (1, 26, 2)
    np.testing.assert_allclose(keypoints[0], [[330.0, 380.0]] * 26)
    np.testing.assert_allclose(scores, 0.7)


def test_tensorrt_config_rejects_more_than_exported_batch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_persons"):
        TensorRTHalpe26BackendConfig(
            detector_engine=tmp_path / "det.engine",
            pose_engine=tmp_path / "pose.engine",
            max_persons=5,
        )


def test_tensorrt_config_rejects_invalid_detector_interval(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="detector_interval"):
        TensorRTHalpe26BackendConfig(
            detector_engine=tmp_path / "det.engine",
            pose_engine=tmp_path / "pose.engine",
            detector_interval=0,
        )
