from __future__ import annotations

import json
import struct

import numpy as np

from rgbd_avatar.avatar.mixamo_asset import MixamoAsset
from rgbd_avatar.avatar.mixamo_gltf import build_mixamo_glb


def _asset() -> MixamoAsset:
    bind = np.repeat(np.eye(4)[None], 2, axis=0)
    bind[1, 1, 3] = 1.0
    return MixamoAsset(
        vertices_m=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float64,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        triangle_uvs=np.asarray(
            [[[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]],
            dtype=np.float64,
        ),
        bone_names=("Hips", "Spine"),
        parent_indices=np.asarray([-1, 0], dtype=np.int32),
        bind_global_m=bind,
        inverse_bind_m=np.linalg.inv(bind),
        skin_joint_indices=np.asarray([[0, 1, 0, 0]] * 3, dtype=np.int32),
        skin_weights=np.asarray([[0.75, 0.25, 0.0, 0.0]] * 3),
        diffuse_png=b"\x89PNG\r\n\x1a\nminimal-test-payload",
        source_path="character.fbx",
        source_sha256="0" * 64,
    )


def _document(payload: bytes) -> tuple[dict, bytes]:
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total_length == len(payload)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    document = json.loads(payload[20 : 20 + json_length])
    binary_offset = 20 + json_length
    binary_length, binary_type = struct.unpack_from("<II", payload, binary_offset)
    assert binary_type == 0x004E4942
    binary = payload[binary_offset + 8 : binary_offset + 8 + binary_length]
    return document, binary


def test_build_mixamo_glb_contains_skin_and_texture() -> None:
    document, binary = _document(build_mixamo_glb(_asset()))

    assert document["scenes"][0]["nodes"] == [0, 2]
    assert document["nodes"][0]["children"] == [1]
    assert document["nodes"][2] == {
        "name": "Ch09_nonPBR",
        "mesh": 0,
        "skin": 0,
    }
    assert document["skins"][0]["joints"] == [0, 1]
    assert document["skins"][0]["skeleton"] == 0
    assert document["images"][0]["mimeType"] == "image/png"
    assert document["buffers"][0]["byteLength"] <= len(binary)
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    assert set(attributes) == {
        "POSITION",
        "NORMAL",
        "TEXCOORD_0",
        "JOINTS_0",
        "WEIGHTS_0",
    }


def test_build_mixamo_glb_flips_fbx_v_coordinates() -> None:
    document, binary = _document(build_mixamo_glb(_asset(), flip_v=True))
    uv_accessor = document["accessors"][2]
    uv_view = document["bufferViews"][uv_accessor["bufferView"]]
    offset = uv_view["byteOffset"]
    length = uv_view["byteLength"]
    values = np.frombuffer(binary[offset : offset + length], dtype="<f4").reshape(-1, 2)

    np.testing.assert_allclose(values, [[0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
