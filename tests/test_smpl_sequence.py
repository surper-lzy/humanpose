from pathlib import Path

import numpy as np
import pytest

from rgbd_avatar.avatar import (
    DISPLAY_FROM_SMPL,
    HALPE_TO_SMPL,
    SMPLFitConfig,
    SMPLHandObservation,
    SMPLSequenceCache,
    SMPLSequenceFitter,
    build_smpl_target,
    display_to_smpl,
    estimate_smpl_scale,
    smpl_to_display,
)


class _ParameterlessSMPL:
    num_betas = 3

    @staticmethod
    def parameters():
        return ()


def test_smpl_fitter_keeps_one_fixed_shape_for_the_sequence() -> None:
    fitter = SMPLSequenceFitter(
        _ParameterlessSMPL(),
        scale=1.0,
        device="cpu",
        config=SMPLFitConfig(),
        betas=np.array([0.5, -0.25, 1.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        fitter.betas.cpu().numpy()[0],
        [0.5, -0.25, 1.0],
    )


def test_smpl_fitter_rejects_wrong_shape_vector() -> None:
    with pytest.raises(ValueError, match="Expected 3 SMPL betas"):
        SMPLSequenceFitter(
            _ParameterlessSMPL(),
            scale=1.0,
            device="cpu",
            config=SMPLFitConfig(),
            betas=np.zeros(2),
        )


def test_smpl_display_transform_is_proper_and_round_trips() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-0.5, 0.8, -1.2]])

    display = smpl_to_display(points)

    assert np.linalg.det(DISPLAY_FROM_SMPL) == pytest.approx(1.0)
    np.testing.assert_allclose(
        DISPLAY_FROM_SMPL @ DISPLAY_FROM_SMPL.T,
        np.eye(3),
    )
    np.testing.assert_allclose(display_to_smpl(display), points)


def test_build_smpl_target_maps_body_and_downweights_prediction() -> None:
    joints = np.arange(26 * 3, dtype=np.float64).reshape(26, 3) * 0.01
    confidence = np.full(26, 0.8)
    usable = np.ones(26, dtype=bool)
    predicted = np.zeros(26, dtype=bool)
    predicted[7] = True
    config = SMPLFitConfig()

    target = build_smpl_target(
        joints,
        confidence,
        usable,
        predicted,
        config=config,
    )

    assert target.count == len(HALPE_TO_SMPL)
    assert target.direction_count == 4
    elbow_position = np.flatnonzero(target.smpl_joint_indices == 18)[0]
    assert target.weights[elbow_position] == pytest.approx(
        0.8 * config.predicted_weight_scale
    )
    assert not np.any(target.smpl_joint_indices == 0)
    assert not np.any(np.isin(target.smpl_joint_indices, [10, 11]))
    np.testing.assert_allclose(
        np.linalg.norm(target.directions_native, axis=1),
        1.0,
    )


def test_smpl_fit_config_requires_positive_spine_regularization() -> None:
    with pytest.raises(ValueError, match="positive"):
        SMPLFitConfig(spine_pose_weight=0.0)


def test_build_smpl_target_uses_hand_landmarks_as_directions_only() -> None:
    joints = np.zeros((26, 3), dtype=np.float64)
    confidence = np.full(26, 0.8)
    usable = np.ones(26, dtype=bool)
    hand_joints = np.zeros((21, 3), dtype=np.float64)
    hand_joints[:, 0] = np.arange(21) * 0.01
    hand = SMPLHandObservation(
        side="left",
        joints_display_m=hand_joints,
        confidence=np.full(21, 0.5),
        valid=np.ones(21, dtype=bool),
    )

    target = build_smpl_target(
        joints,
        confidence,
        usable,
        np.zeros(26, dtype=bool),
        config=SMPLFitConfig(),
        hand_observations=[hand],
    )

    assert target.count == len(HALPE_TO_SMPL)
    assert target.direction_count == 3
    assert not np.any(np.isin(target.smpl_joint_indices, [22, 35, 37, 39]))
    np.testing.assert_array_equal(
        target.smpl_direction_pairs,
        [[20, 37], [39, 36], [20, 35]],
    )
    np.testing.assert_allclose(
        np.linalg.norm(target.directions_native, axis=1),
        1.0,
    )


def test_hand_direction_targets_ignore_detected_finger_length() -> None:
    joints = np.zeros((26, 3), dtype=np.float64)
    hand_joints = np.zeros((21, 3), dtype=np.float64)
    hand_joints[2] = [0.03, 0.02, 0.01]
    hand_joints[5] = [0.04, 0.03, 0.00]
    hand_joints[9] = [0.00, 0.08, 0.01]
    hand_joints[17] = [-0.04, 0.03, 0.00]
    confidence = np.full(21, 0.9)
    valid = np.zeros(21, dtype=bool)
    valid[[0, 2, 5, 9, 17]] = True

    def build(scale: float):
        hand = SMPLHandObservation(
            side="left",
            joints_display_m=hand_joints * scale,
            confidence=confidence,
            valid=valid,
        )
        return build_smpl_target(
            joints,
            np.full(26, 0.8),
            np.ones(26, dtype=bool),
            np.zeros(26, dtype=bool),
            config=SMPLFitConfig(),
            hand_observations=[hand],
        )

    original = build(1.0)
    longer = build(1.8)

    np.testing.assert_allclose(
        original.directions_native,
        longer.directions_native,
    )

def test_sequence_scale_recovers_known_metric_multiplier() -> None:
    rest = np.zeros((24, 3), dtype=np.float64)
    rest[0] = [0.0, 0.9, 0.0]
    rest[1] = [-0.15, 0.9, 0.0]
    rest[2] = [0.15, 0.9, 0.0]
    rest[4] = [-0.15, 0.5, 0.0]
    rest[5] = [0.15, 0.5, 0.0]
    rest[7] = [-0.15, 0.1, 0.0]
    rest[8] = [0.15, 0.1, 0.0]
    rest[12] = [0.0, 1.45, 0.0]
    rest[15] = [0.0, 1.70, 0.0]
    rest[16] = [-0.22, 1.42, 0.0]
    rest[17] = [0.22, 1.42, 0.0]
    rest[18] = [-0.48, 1.42, 0.0]
    rest[19] = [0.48, 1.42, 0.0]
    rest[20] = [-0.73, 1.42, 0.0]
    rest[21] = [0.73, 1.42, 0.0]
    known_scale = 1.08
    joints = np.full((26, 3), np.nan)
    usable = np.zeros(26, dtype=bool)
    for halpe_index, smpl_index in HALPE_TO_SMPL:
        if smpl_index >= len(rest):
            continue
        joints[halpe_index] = smpl_to_display(
            rest[[smpl_index]] * known_scale
        )[0]
        usable[halpe_index] = True

    estimated = estimate_smpl_scale([joints], [usable], rest)

    assert estimated == pytest.approx(known_scale)


def test_smpl_sequence_cache_round_trip(tmp_path: Path) -> None:
    vertices = np.full((2, 4, 3), np.nan, dtype=np.float32)
    joints = np.full((2, 24, 3), np.nan, dtype=np.float32)
    vertices[0] = 0.0
    joints[0] = 0.0
    cache = SMPLSequenceCache(
        frame_indices=np.array([3, 4]),
        present=np.array([True, False]),
        vertices_display_m=vertices,
        joints_display_m=joints,
        faces=np.array(
            [[0, 1, 2], [0, 2, 3]],
            dtype=np.int32,
        ),
        body_pose=np.full((2, 69), np.nan),
        global_orient=np.full((2, 3), np.nan),
        translation_native_m=np.full((2, 3), np.nan),
        target_counts=np.array([17, 0]),
        error_mean_m=np.array([0.01, np.nan]),
        error_p95_m=np.array([0.02, np.nan]),
        error_max_m=np.array([0.03, np.nan]),
        scale=1.02,
        metadata={"pose_layer": "constrained"},
    )
    path = tmp_path / "smpl_sequence.npz"

    cache.save(path)
    loaded = SMPLSequenceCache.load(path)

    np.testing.assert_array_equal(loaded.frame_indices, [3, 4])
    np.testing.assert_array_equal(loaded.present, [True, False])
    np.testing.assert_allclose(loaded.vertices_display_m[0], 0.0)
    assert np.isnan(loaded.vertices_display_m[1]).all()
    assert loaded.scale == pytest.approx(1.02)
    assert loaded.metadata["pose_layer"] == "constrained"
