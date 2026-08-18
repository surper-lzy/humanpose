import base64
from pathlib import Path

import numpy as np
import pytest

from rgbd_avatar.avatar import (
    MixamoSequenceCache,
    load_mixamo_fbx,
    skin_mixamo_vertices,
)
from rgbd_avatar.retargeting import (
    MixamoAnalyticalIK,
    calibrate_halpe_smpl_profile,
    estimate_mixamo_scale,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIXAMO_FBX = PROJECT_ROOT / "assets/models/mixamo/Ch09_nonPBR.fbx"


@pytest.fixture(scope="module")
def mixamo_asset():
    if not MIXAMO_FBX.is_file():
        pytest.skip("Optional Mixamo FBX test asset is unavailable.")
    return load_mixamo_fbx(MIXAMO_FBX)


def _halpe_pose() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    joints = np.full((26, 3), np.nan, dtype=np.float64)
    joints[19] = [0.0, 2.0, 0.90]
    joints[18] = [0.0, 2.0, 1.43]
    joints[17] = [0.0, 2.0, 1.70]
    joints[5], joints[6] = [0.22, 2.0, 1.42], [-0.22, 2.0, 1.42]
    joints[7], joints[8] = [0.48, 2.0, 1.25], [-0.48, 2.0, 1.25]
    joints[9], joints[10] = [0.70, 2.0, 1.10], [-0.70, 2.0, 1.10]
    joints[11], joints[12] = [0.10, 2.0, 0.89], [-0.10, 2.0, 0.89]
    joints[13], joints[14] = [0.11, 2.0, 0.49], [-0.11, 2.0, 0.49]
    joints[15], joints[16] = [0.12, 2.0, 0.06], [-0.12, 2.0, 0.06]
    joints[24], joints[25] = [0.12, 1.94, 0.03], [-0.12, 1.94, 0.03]
    joints[20], joints[21] = [0.12, 2.17, 0.03], [-0.12, 2.17, 0.03]
    joints[22], joints[23] = [0.16, 2.16, 0.03], [-0.16, 2.16, 0.03]
    valid = np.isfinite(joints).all(axis=1)
    confidence = valid.astype(np.float64) * 0.9
    predicted = np.zeros(26, dtype=bool)
    return joints, confidence, valid, predicted


def test_mixamo_fbx_extracts_topological_skin_and_texture(mixamo_asset) -> None:
    asset = mixamo_asset

    assert asset.vertices_m.shape == (15716, 3)
    assert asset.faces.shape == (31292, 3)
    assert len(asset.bone_names) == 65
    assert asset.bone_names[0] == "Hips"
    assert {"LeftArm", "RightFoot", "LeftHandMiddle1"} <= set(asset.bone_names)
    np.testing.assert_allclose(asset.skin_weights.sum(axis=1), 1.0)
    assert asset.diffuse_png.startswith(b"\x89PNG\r\n\x1a\n")


def test_mixamo_bind_pose_skinning_is_identity(mixamo_asset) -> None:
    deformed = skin_mixamo_vertices(mixamo_asset, mixamo_asset.bind_global_m)

    np.testing.assert_allclose(deformed, mixamo_asset.vertices_m, atol=1e-12)


def test_mixamo_analytical_ik_aligns_arm_without_changing_bone_length(
    mixamo_asset,
) -> None:
    joints, confidence, valid, predicted = _halpe_pose()
    sequence = [joints] * 10
    profile = calibrate_halpe_smpl_profile(
        sequence,
        [confidence] * 10,
        [valid] * 10,
        [predicted] * 10,
    )
    scale = estimate_mixamo_scale(
        mixamo_asset,
        sequence,
        [confidence] * 10,
        [valid] * 10,
        [predicted] * 10,
        profile,
    )
    solver = MixamoAnalyticalIK(mixamo_asset, profile, scale=scale)

    frame = solver.solve(
        joints,
        confidence,
        valid,
        predicted,
        delta_time_s=0.5,
    )

    assert frame is not None
    index = mixamo_asset.bone_index
    positions = frame.bone_global_m[:, :3, 3]
    actual = positions[index["LeftForeArm"]] - positions[index["LeftArm"]]
    target = joints[7] - joints[5]
    cosine = np.dot(actual, target) / (
        np.linalg.norm(actual) * np.linalg.norm(target)
    )
    assert cosine > 0.999
    bind_length = np.linalg.norm(
        mixamo_asset.bind_global_m[index["LeftForeArm"], :3, 3]
        - mixamo_asset.bind_global_m[index["LeftArm"], :3, 3]
    )
    assert np.linalg.norm(actual) == pytest.approx(scale * bind_length)


def test_mixamo_analytical_ik_keeps_level_halpe_soles_level(
    mixamo_asset,
) -> None:
    joints, confidence, valid, predicted = _halpe_pose()
    sequence = [joints] * 10
    profile = calibrate_halpe_smpl_profile(
        sequence,
        [confidence] * 10,
        [valid] * 10,
        [predicted] * 10,
    )
    scale = estimate_mixamo_scale(
        mixamo_asset,
        sequence,
        [confidence] * 10,
        [valid] * 10,
        [predicted] * 10,
        profile,
    )
    solver = MixamoAnalyticalIK(mixamo_asset, profile, scale=scale)
    frame = solver.solve(
        joints,
        confidence,
        valid,
        predicted,
        delta_time_s=0.5,
    )

    assert frame is not None
    index = mixamo_asset.bone_index
    for side, heel_index, toe_indices in (
        ("Left", 24, (20, 22)),
        ("Right", 25, (21, 23)),
    ):
        foot = frame.bone_global_m[index[f"{side}Foot"]]
        toe = frame.bone_global_m[index[f"{side}ToeBase"]]
        solved_forward = toe[:3, 3] - foot[:3, 3]
        solved_forward[2] = 0.0
        target_forward = (
            0.5 * (joints[toe_indices[0]] + joints[toe_indices[1]])
            - joints[heel_index]
        )
        target_forward[2] = 0.0
        cosine = np.dot(solved_forward, target_forward) / (
            np.linalg.norm(solved_forward) * np.linalg.norm(target_forward)
        )

        assert cosine > 0.999
        # The authored mesh has a flat sole even though its ankle-to-toe bone
        # slopes downward. Its bind-pose sole normal must map to live body-up,
        # rather than inheriting the downward ankle-to-toe pitch.
        foot_rotation = foot[:3, :3] / scale
        bind_rotation = mixamo_asset.bind_global_m[index[f"{side}Foot"], :3, :3]
        solved_sole_up = (
            foot_rotation
            @ bind_rotation.T
            @ solver.bind_body_basis[:, 2]
        )
        np.testing.assert_allclose(solved_sole_up, [0.0, 0.0, 1.0], atol=1e-6)


def test_mixamo_cache_round_trip(tmp_path: Path) -> None:
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    cache = MixamoSequenceCache(
        frame_indices=np.array([4]),
        present=np.array([True]),
        vertices_display_m=np.zeros((1, 3, 3), dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        triangle_uvs=np.zeros((1, 3, 2), dtype=np.float32),
        diffuse_png=np.frombuffer(png, dtype=np.uint8),
        bone_names=("Hips",),
        bone_global_m=np.eye(4, dtype=np.float32).reshape(1, 1, 4, 4),
        scale=1.1,
        metadata={"pose_layer": "constrained"},
    )
    path = tmp_path / "mixamo.npz"

    cache.save(path)
    loaded = MixamoSequenceCache.load(path)

    np.testing.assert_array_equal(loaded.faces, cache.faces)
    np.testing.assert_allclose(loaded.vertices_display_m, cache.vertices_display_m)
    assert loaded.bone_names == ("Hips",)
    assert loaded.metadata == cache.metadata
