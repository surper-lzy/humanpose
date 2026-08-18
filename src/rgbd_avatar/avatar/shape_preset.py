"""Persistent, model-bound SMPL body-shape parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from rgbd_avatar.io import atomic_write_json, load_json_mapping


SHAPE_PRESET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SMPLShapePreset:
    """A reusable zero-pose body shape for one specific SMPL model."""

    model_path: str
    model_sha256: str
    betas: np.ndarray
    scale: float = 1.0

    def __post_init__(self) -> None:
        betas = np.asarray(self.betas, dtype=np.float32)
        if betas.ndim != 1 or betas.size == 0:
            raise ValueError("SMPL shape betas must be a non-empty vector.")
        if not np.isfinite(betas).all():
            raise ValueError("SMPL shape betas must be finite.")
        scale = float(self.scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("SMPL shape scale must be finite and positive.")
        digest = str(self.model_sha256).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("SMPL model SHA-256 must contain 64 hex digits.")
        object.__setattr__(self, "betas", betas.copy())
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "model_sha256", digest)

    def to_dict(self, *, mesh_path: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": SHAPE_PRESET_SCHEMA_VERSION,
            "type": "smpl_neutral_shape_preset",
            "model": {
                "path": self.model_path,
                "sha256": self.model_sha256,
            },
            "betas": self.betas.tolist(),
            "scale": self.scale,
            "pose": "zero_pose",
            "mesh_coordinate_system": {
                "handedness": "right",
                "x": "right",
                "y": "forward",
                "z": "up",
                "unit": "meter",
                "grounded": True,
            },
        }
        if mesh_path is not None:
            payload["preview_mesh"] = mesh_path
        return payload

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "SMPLShapePreset":
        if int(values.get("schema_version", -1)) != SHAPE_PRESET_SCHEMA_VERSION:
            raise ValueError("Unsupported SMPL shape preset schema version.")
        if values.get("type") != "smpl_neutral_shape_preset":
            raise ValueError("File is not an SMPL Neutral shape preset.")
        model = values.get("model")
        if not isinstance(model, Mapping):
            raise ValueError("SMPL shape preset is missing model metadata.")
        return cls(
            model_path=str(model["path"]),
            model_sha256=str(model["sha256"]),
            betas=np.asarray(values["betas"], dtype=np.float32),
            scale=float(values.get("scale", 1.0)),
        )


def save_shape_preset(
    path: str | Path,
    preset: SMPLShapePreset,
    *,
    mesh_path: str | None = None,
) -> None:
    atomic_write_json(path, preset.to_dict(mesh_path=mesh_path))


def load_shape_preset(path: str | Path) -> SMPLShapePreset:
    return SMPLShapePreset.from_mapping(load_json_mapping(path))


__all__ = [
    "SHAPE_PRESET_SCHEMA_VERSION",
    "SMPLShapePreset",
    "load_shape_preset",
    "save_shape_preset",
]
