"""Linear-blend skinning and portable Mixamo sequence caches."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from .mixamo_asset import MixamoAsset


def skin_mixamo_vertices(
    asset: MixamoAsset,
    bone_global_m: np.ndarray,
) -> np.ndarray:
    """Apply four-influence linear blend skinning in display space."""

    current = np.asarray(bone_global_m, dtype=np.float64)
    bone_count = len(asset.bone_names)
    if current.shape != (bone_count, 4, 4) or not np.isfinite(current).all():
        raise ValueError("Mixamo current bone matrices must have shape Bx4x4.")
    skin_matrices = current @ asset.inverse_bind_m
    homogeneous = np.column_stack(
        (asset.vertices_m, np.ones(len(asset.vertices_m), dtype=np.float64))
    )
    selected = skin_matrices[asset.skin_joint_indices]
    transformed = np.einsum("vkij,vj->vki", selected, homogeneous)
    vertices = np.sum(
        transformed[:, :, :3] * asset.skin_weights[:, :, None],
        axis=1,
    )
    if not np.isfinite(vertices).all():
        raise ValueError("Mixamo skinning produced non-finite vertices.")
    return vertices


@dataclass(frozen=True)
class MixamoSequenceCache:
    frame_indices: np.ndarray
    present: np.ndarray
    vertices_display_m: np.ndarray
    faces: np.ndarray
    triangle_uvs: np.ndarray
    diffuse_png: np.ndarray
    bone_names: tuple[str, ...]
    bone_global_m: np.ndarray
    scale: float
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        frame_indices = np.asarray(self.frame_indices, dtype=np.int64)
        present = np.asarray(self.present, dtype=bool)
        vertices = np.asarray(self.vertices_display_m, dtype=np.float32)
        faces = np.asarray(self.faces, dtype=np.int32)
        uvs = np.asarray(self.triangle_uvs, dtype=np.float32)
        texture = np.asarray(self.diffuse_png, dtype=np.uint8)
        matrices = np.asarray(self.bone_global_m, dtype=np.float32)
        frame_count = len(frame_indices)
        if present.shape != (frame_count,):
            raise ValueError("Mixamo cache present mask has invalid shape.")
        if vertices.ndim != 3 or vertices.shape[0] != frame_count or vertices.shape[2] != 3:
            raise ValueError("Mixamo cache vertices must have shape FxVx3.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Mixamo cache faces must have shape Tx3.")
        if uvs.shape != (len(faces), 3, 2):
            raise ValueError("Mixamo cache UVs must have shape Tx3x2.")
        if matrices.shape != (frame_count, len(self.bone_names), 4, 4):
            raise ValueError("Mixamo cache bone matrices have invalid shape.")
        if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[1]):
            raise ValueError("Mixamo cache faces reference invalid vertices.")
        if texture.ndim != 1 or bytes(texture[:8]) != b"\x89PNG\r\n\x1a\n":
            raise ValueError("Mixamo cache texture is not PNG data.")
        if np.any(present) and not np.isfinite(vertices[present]).all():
            raise ValueError("Present Mixamo cache frames must be finite.")
        if np.any(~present) and not np.isnan(vertices[~present]).all():
            raise ValueError("Absent Mixamo cache frames must contain NaN vertices.")
        if not np.isfinite(matrices[present]).all():
            raise ValueError("Present Mixamo bone matrices must be finite.")
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("Mixamo cache scale must be positive.")
        object.__setattr__(self, "frame_indices", frame_indices)
        object.__setattr__(self, "present", present)
        object.__setattr__(self, "vertices_display_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "triangle_uvs", uvs)
        object.__setattr__(self, "diffuse_png", texture)
        object.__setattr__(self, "bone_global_m", matrices)

    def save(self, path: str | Path, *, overwrite: bool = False) -> None:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite Mixamo cache: {target}")
        with tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            with temporary_path.open("wb") as file:
                np.savez_compressed(
                    file,
                    frame_indices=self.frame_indices,
                    present=self.present,
                    vertices_display_m=self.vertices_display_m,
                    faces=self.faces,
                    triangle_uvs=self.triangle_uvs,
                    diffuse_png=self.diffuse_png,
                    bone_names=np.asarray(self.bone_names),
                    bone_global_m=self.bone_global_m,
                    scale=np.asarray(self.scale),
                    metadata_json=np.asarray(
                        json.dumps(self.metadata, ensure_ascii=False, sort_keys=True)
                    ),
                )
            temporary_path.replace(target)
        finally:
            temporary_path.unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> "MixamoSequenceCache":
        source = Path(path).expanduser().resolve()
        with np.load(source, allow_pickle=False) as payload:
            return cls(
                frame_indices=payload["frame_indices"].copy(),
                present=payload["present"].copy(),
                vertices_display_m=payload["vertices_display_m"].copy(),
                faces=payload["faces"].copy(),
                triangle_uvs=payload["triangle_uvs"].copy(),
                diffuse_png=payload["diffuse_png"].copy(),
                bone_names=tuple(str(x) for x in payload["bone_names"].tolist()),
                bone_global_m=payload["bone_global_m"].copy(),
                scale=float(payload["scale"]),
                metadata=json.loads(str(payload["metadata_json"])),
            )


__all__ = ["MixamoSequenceCache", "skin_mixamo_vertices"]
