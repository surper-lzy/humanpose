from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "frontend" / "mixamo-avatar-delivery"


def _glb_document(path: Path) -> dict:
    payload = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    assert magic == 0x46546C67
    assert version == 2
    assert total_length == len(payload)
    json_length, json_type = struct.unpack_from("<II", payload, 12)
    assert json_type == 0x4E4F534A
    return json.loads(payload[20 : 20 + json_length])


def test_delivery_manifest_matches_checked_in_glb() -> None:
    model = DELIVERY / "public" / "avatars" / "character-a.glb"
    manifest = json.loads(
        (DELIVERY / "public" / "avatars" / "character-a.manifest.json").read_text()
    )
    payload = model.read_bytes()

    assert manifest["model_bytes"] == len(payload)
    assert manifest["model_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["bone_count"] == 65
    assert manifest["triangle_count"] == 31_292


def test_delivery_glb_contains_one_complete_mixamo_skin() -> None:
    document = _glb_document(
        DELIVERY / "public" / "avatars" / "character-a.glb"
    )

    assert len(document["skins"]) == 1
    assert len(document["skins"][0]["joints"]) == 65
    assert document["skins"][0]["skeleton"] == 0
    assert document["nodes"][0]["name"] == "Hips"
    assert document["nodes"][-1]["skin"] == 0
    primitive = document["meshes"][0]["primitives"][0]
    assert set(primitive["attributes"]) == {
        "POSITION",
        "NORMAL",
        "TEXCOORD_0",
        "JOINTS_0",
        "WEIGHTS_0",
    }
    assert document["images"][0]["mimeType"] == "image/png"


def test_delivery_fixtures_follow_halpe26_contract() -> None:
    fixture = json.loads(
        (DELIVERY / "fixtures" / "halpe26-pose-samples.json").read_text()
    )

    assert len(fixture["frames"]) >= 4
    for frame in fixture["frames"]:
        assert frame["schema_version"] == 1
        assert frame["keypoint_format"] == "halpe26"
        assert len(frame["joints"]) == 26
        for joint in frame["joints"]:
            if joint is not None:
                assert len(joint) == 3
                assert np.isfinite(joint).all()


def test_delivery_contains_required_frontend_modules() -> None:
    expected = {
        "avatarController.ts",
        "avatarRegistry.ts",
        "coordinates.ts",
        "halpe26.ts",
        "index.ts",
        "mixamoRetargeter.ts",
        "multiAvatarController.ts",
        "stickmanRenderer.ts",
        "stickmenWebSocket.ts",
        "stickmanWebSocket.ts",
        "types.ts",
    }
    assert {path.name for path in (DELIVERY / "src").glob("*.ts")} == expected
