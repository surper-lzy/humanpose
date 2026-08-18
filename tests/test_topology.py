import numpy as np

from rgbd_avatar.depth.topology import (
    TopologyCandidate,
    bilateral_length_outliers,
    select_face_core_candidates,
    select_foot_group_candidates,
)


def candidate(
    token: int,
    xyz: tuple[float, float, float],
    score: float,
    person_quality: float = 0.9,
) -> TopologyCandidate:
    return TopologyCandidate(
        token=token,
        xyz_m=np.asarray(xyz),
        score=score,
        person_quality=person_quality,
    )


def test_face_group_reselects_eye_to_nose_eye_consensus() -> None:
    selection = select_face_core_candidates(
        {
            0: [candidate(0, (0.00, 0.00, 2.650), 0.95)],
            1: [
                candidate(0, (0.04, -0.02, 2.770), 0.94),
                candidate(1, (0.04, -0.02, 2.645), 0.89),
            ],
            2: [candidate(0, (-0.04, -0.02, 2.649), 0.95)],
        },
        neck_depth_m=2.76,
    )

    assert selection is not None
    assert selection.selected_tokens == {0: 0, 1: 1, 2: 0}


def test_foot_group_can_reselect_ankle_from_coherent_leaves() -> None:
    selection = select_foot_group_candidates(
        {
            15: [
                candidate(0, (0.00, 0.40, 3.90), 0.80),
                candidate(1, (0.00, 0.40, 3.67), 0.65),
            ],
            20: [candidate(0, (0.00, 0.55, 3.58), 0.80)],
            22: [candidate(0, (0.05, 0.55, 3.60), 0.78)],
            24: [candidate(0, (0.00, 0.42, 3.62), 0.77)],
        },
        ankle_id=15,
        big_toe_id=20,
        small_toe_id=22,
        heel_id=24,
        knee_xyz_m=np.asarray((0.00, 0.00, 3.40)),
        min_person_quality=0.10,
        ankle_toe_max_length_m=0.30,
    )

    assert selection is not None
    assert selection.selected_tokens == {15: 1, 20: 0, 22: 0, 24: 0}


def test_foot_group_leaves_ground_surface_missing() -> None:
    selection = select_foot_group_candidates(
        {
            16: [candidate(0, (0.00, 0.40, 1.40), 0.90)],
            21: [candidate(0, (0.00, 0.60, 1.82), 0.85)],
            23: [candidate(0, (0.05, 0.60, 1.84), 0.84)],
            25: [candidate(0, (0.00, 0.42, 1.48), 0.80)],
        },
        ankle_id=16,
        big_toe_id=21,
        small_toe_id=23,
        heel_id=25,
        knee_xyz_m=np.asarray((0.00, 0.00, 1.20)),
        min_person_quality=0.10,
        ankle_toe_max_length_m=0.30,
    )

    assert selection is not None
    assert selection.selected_tokens[16] == 0
    assert selection.selected_tokens[21] is None
    assert selection.selected_tokens[23] is None
    assert selection.selected_tokens[25] == 0


def test_bilateral_gate_rejects_only_long_asymmetric_side() -> None:
    left, right, lengths = bilateral_length_outliers(
        center_xyz_m=np.asarray((0.0, 0.0, 2.0)),
        left_xyz_m=np.asarray((-0.2, 0.0, 2.0)),
        right_xyz_m=np.asarray((0.1, 0.0, 1.66)),
        max_length_m=0.32,
        asymmetry_ratio=1.60,
    )

    assert not left
    assert right
    assert np.isclose(lengths["left_length_m"], 0.2)
    assert lengths["right_length_m"] > 0.35
