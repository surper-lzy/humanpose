import json

import numpy as np
import pytest

from rgbd_avatar.avatar.shape_preset import (
    SMPLShapePreset,
    load_shape_preset,
    save_shape_preset,
)


DIGEST = "a" * 64


def test_shape_preset_round_trip(tmp_path) -> None:
    path = tmp_path / "shape.json"
    expected = SMPLShapePreset(
        model_path="SMPL_NEUTRAL_CLEAN.pkl",
        model_sha256=DIGEST,
        betas=np.array([0.5, -0.25, 0.0], dtype=np.float32),
        scale=1.08,
    )
    save_shape_preset(path, expected, mesh_path="shape.ply")
    loaded = load_shape_preset(path)
    assert np.allclose(loaded.betas, expected.betas)
    assert loaded.scale == pytest.approx(1.08)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["preview_mesh"] == "shape.ply"
    assert payload["mesh_coordinate_system"]["grounded"] is True


def test_shape_preset_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="finite"):
        SMPLShapePreset("model.pkl", DIGEST, np.array([np.nan]))
    with pytest.raises(ValueError, match="positive"):
        SMPLShapePreset("model.pkl", DIGEST, np.zeros(10), scale=0.0)
    with pytest.raises(ValueError, match="SHA-256"):
        SMPLShapePreset("model.pkl", "bad", np.zeros(10))
