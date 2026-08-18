"""Export :class:`MixamoAsset` characters as self-contained glTF 2.0 GLB.

The runtime FBX reader deliberately exposes exactly the data needed by a
browser skin: mesh attributes, one skeleton, inverse bind matrices, and an
embedded diffuse texture.  Keeping the exporter here avoids a Blender or
Assimp dependency in deployment and makes the generated frontend asset
reproducible from the checked-in FBX source.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct
import tempfile
from typing import Any

import numpy as np

from .mixamo_asset import MixamoAsset, load_mixamo_fbx


_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963
_FLOAT = 5126
_UNSIGNED_SHORT = 5123
_UNSIGNED_INT = 5125


@dataclass(frozen=True)
class MixamoGlbSummary:
    output_path: Path
    source_vertex_count: int
    exported_vertex_count: int
    triangle_count: int
    bone_count: int
    byte_count: int


class _GlbBufferBuilder:
    def __init__(self) -> None:
        self.data = bytearray()
        self.buffer_views: list[dict[str, Any]] = []
        self.accessors: list[dict[str, Any]] = []

    def _align(self, alignment: int = 4) -> None:
        padding = (-len(self.data)) % alignment
        if padding:
            self.data.extend(b"\x00" * padding)

    def add_view(
        self,
        payload: bytes,
        *,
        target: int | None = None,
        name: str | None = None,
    ) -> int:
        self._align()
        offset = len(self.data)
        self.data.extend(payload)
        view: dict[str, Any] = {
            "buffer": 0,
            "byteOffset": offset,
            "byteLength": len(payload),
        }
        if target is not None:
            view["target"] = target
        if name is not None:
            view["name"] = name
        index = len(self.buffer_views)
        self.buffer_views.append(view)
        return index

    def add_accessor(
        self,
        array: np.ndarray,
        *,
        component_type: int,
        accessor_type: str,
        target: int | None = None,
        name: str | None = None,
        bounds: bool = False,
    ) -> int:
        contiguous = np.ascontiguousarray(array)
        view = self.add_view(contiguous.tobytes(), target=target, name=name)
        accessor: dict[str, Any] = {
            "bufferView": view,
            "componentType": component_type,
            "count": int(len(contiguous)),
            "type": accessor_type,
        }
        if name is not None:
            accessor["name"] = name
        if bounds:
            reshaped = contiguous.reshape(len(contiguous), -1)
            accessor["min"] = [float(value) for value in reshaped.min(axis=0)]
            accessor["max"] = [float(value) for value in reshaped.max(axis=0)]
        index = len(self.accessors)
        self.accessors.append(accessor)
        return index


def _smooth_vertex_normals(asset: MixamoAsset) -> np.ndarray:
    positions = np.asarray(asset.vertices_m, dtype=np.float64)
    faces = np.asarray(asset.faces, dtype=np.int64)
    edges_a = positions[faces[:, 1]] - positions[faces[:, 0]]
    edges_b = positions[faces[:, 2]] - positions[faces[:, 0]]
    face_normals = np.cross(edges_a, edges_b)
    normals = np.zeros_like(positions)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("Mixamo mesh contains vertices with degenerate normals.")
    normals /= lengths[:, None]
    return normals


def _expand_uv_seams(
    asset: MixamoAsset,
    *,
    flip_v: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create glTF per-vertex UVs while retaining indexed geometry.

    FBX stores UVs per polygon corner.  glTF stores all primitive attributes
    per vertex, so positions shared by different UV seams must be split.  The
    key uses float32 bit patterns to merge repeated corners without applying
    a tolerance that could accidentally join distinct texture seams.
    """

    source_normals = _smooth_vertex_normals(asset)
    vertex_by_key: dict[tuple[int, int, int], int] = {}
    source_indices: list[int] = []
    exported_uvs: list[tuple[float, float]] = []
    indices = np.empty((len(asset.faces), 3), dtype=np.uint32)

    for face_index, face in enumerate(asset.faces):
        for corner in range(3):
            source_index = int(face[corner])
            uv = np.asarray(asset.triangle_uvs[face_index, corner], dtype=np.float32)
            u = np.float32(uv[0])
            v = np.float32(1.0 - uv[1] if flip_v else uv[1])
            u_bits = int(np.asarray(u).view(np.uint32))
            v_bits = int(np.asarray(v).view(np.uint32))
            key = (source_index, u_bits, v_bits)
            exported_index = vertex_by_key.get(key)
            if exported_index is None:
                exported_index = len(source_indices)
                vertex_by_key[key] = exported_index
                source_indices.append(source_index)
                exported_uvs.append((float(u), float(v)))
            indices[face_index, corner] = exported_index

    selected = np.asarray(source_indices, dtype=np.int64)
    positions = np.asarray(asset.vertices_m[selected], dtype="<f4")
    normals = np.asarray(source_normals[selected], dtype="<f4")
    uvs = np.asarray(exported_uvs, dtype="<f4")
    joints = np.asarray(asset.skin_joint_indices[selected], dtype="<u2")
    weights = np.asarray(asset.skin_weights[selected], dtype="<f4")
    flat_indices = np.asarray(indices.reshape(-1), dtype="<u4")
    return positions, normals, uvs, joints, weights, flat_indices


def _local_bind_matrices(asset: MixamoAsset) -> np.ndarray:
    bind = np.asarray(asset.bind_global_m, dtype=np.float64)
    local = np.empty_like(bind)
    local[0] = bind[0]
    for index in range(1, len(bind)):
        parent = int(asset.parent_indices[index])
        local[index] = np.linalg.inv(bind[parent]) @ bind[index]
    return local


def build_mixamo_glb(asset: MixamoAsset, *, flip_v: bool = True) -> bytes:
    """Build one self-contained glTF 2.0 binary payload."""

    positions, normals, uvs, joints, weights, indices = _expand_uv_seams(
        asset,
        flip_v=flip_v,
    )
    builder = _GlbBufferBuilder()
    position_accessor = builder.add_accessor(
        positions,
        component_type=_FLOAT,
        accessor_type="VEC3",
        target=_ARRAY_BUFFER,
        name="POSITION",
        bounds=True,
    )
    normal_accessor = builder.add_accessor(
        normals,
        component_type=_FLOAT,
        accessor_type="VEC3",
        target=_ARRAY_BUFFER,
        name="NORMAL",
    )
    uv_accessor = builder.add_accessor(
        uvs,
        component_type=_FLOAT,
        accessor_type="VEC2",
        target=_ARRAY_BUFFER,
        name="TEXCOORD_0",
    )
    joint_accessor = builder.add_accessor(
        joints,
        component_type=_UNSIGNED_SHORT,
        accessor_type="VEC4",
        target=_ARRAY_BUFFER,
        name="JOINTS_0",
    )
    weight_accessor = builder.add_accessor(
        weights,
        component_type=_FLOAT,
        accessor_type="VEC4",
        target=_ARRAY_BUFFER,
        name="WEIGHTS_0",
    )
    index_accessor = builder.add_accessor(
        indices,
        component_type=_UNSIGNED_INT,
        accessor_type="SCALAR",
        target=_ELEMENT_ARRAY_BUFFER,
        name="indices",
        bounds=True,
    )

    # glTF MAT4 accessor values are stored column-major.  NumPy matrices are
    # conventional row-major arrays, hence transpose before serialisation.
    inverse_bind_column_major = np.asarray(
        np.transpose(asset.inverse_bind_m, (0, 2, 1)),
        dtype="<f4",
    )
    inverse_bind_accessor = builder.add_accessor(
        inverse_bind_column_major,
        component_type=_FLOAT,
        accessor_type="MAT4",
        name="inverseBindMatrices",
    )
    image_view = builder.add_view(asset.diffuse_png, name="diffuse.png")

    local_bind = _local_bind_matrices(asset)
    nodes: list[dict[str, Any]] = []
    children_by_parent: list[list[int]] = [
        [] for _ in range(len(asset.bone_names))
    ]
    for index in range(1, len(asset.bone_names)):
        children_by_parent[int(asset.parent_indices[index])].append(index)
    for index, name in enumerate(asset.bone_names):
        node: dict[str, Any] = {
            "name": name,
            "matrix": [
                float(value) for value in local_bind[index].T.reshape(-1)
            ],
        }
        if children_by_parent[index]:
            node["children"] = children_by_parent[index]
        nodes.append(node)

    mesh_node_index = len(nodes)
    nodes.append({"name": "Ch09_nonPBR", "mesh": 0, "skin": 0})

    document: dict[str, Any] = {
        "asset": {
            "version": "2.0",
            "generator": "rgbd-avatar MixamoAsset GLB exporter",
            "extras": {
                "source": Path(asset.source_path).name,
                "source_sha256": asset.source_sha256,
            },
        },
        "scene": 0,
        "scenes": [
            {
                "name": "MixamoAvatar",
                "nodes": [0, mesh_node_index],
            }
        ],
        "nodes": nodes,
        "meshes": [
            {
                "name": "Ch09_nonPBR",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": position_accessor,
                            "NORMAL": normal_accessor,
                            "TEXCOORD_0": uv_accessor,
                            "JOINTS_0": joint_accessor,
                            "WEIGHTS_0": weight_accessor,
                        },
                        "indices": index_accessor,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
        "skins": [
            {
                "name": "MixamoSkeleton",
                "inverseBindMatrices": inverse_bind_accessor,
                "skeleton": 0,
                "joints": list(range(len(asset.bone_names))),
            }
        ],
        "materials": [
            {
                "name": "Ch09_nonPBR",
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                    "baseColorTexture": {"index": 0, "texCoord": 0},
                    "metallicFactor": 0.0,
                    "roughnessFactor": 1.0,
                },
                "alphaMode": "OPAQUE",
                "doubleSided": False,
            }
        ],
        "textures": [{"name": "diffuse", "sampler": 0, "source": 0}],
        "samplers": [
            {
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497,
            }
        ],
        "images": [
            {
                "name": "diffuse.png",
                "mimeType": "image/png",
                "bufferView": image_view,
            }
        ],
        "accessors": builder.accessors,
        "bufferViews": builder.buffer_views,
        "buffers": [{"byteLength": 0}],
    }

    builder._align()
    document["buffers"][0]["byteLength"] = len(builder.data)
    json_payload = json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    json_payload += b" " * ((-len(json_payload)) % 4)
    binary_payload = bytes(builder.data)
    binary_payload += b"\x00" * ((-len(binary_payload)) % 4)

    total_length = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    return b"".join(
        (
            struct.pack("<III", 0x46546C67, 2, total_length),
            struct.pack("<II", len(json_payload), 0x4E4F534A),
            json_payload,
            struct.pack("<II", len(binary_payload), 0x004E4942),
            binary_payload,
        )
    )


def export_mixamo_fbx_glb(
    source_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    flip_v: bool = True,
) -> MixamoGlbSummary:
    """Load one supported Mixamo FBX and atomically export a browser GLB."""

    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite Mixamo GLB: {output}")
    asset = load_mixamo_fbx(source)
    payload = build_mixamo_glb(asset, flip_v=flip_v)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent,
        prefix=f".{output.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(payload)
    try:
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)

    exported_vertex_count = int(
        json.loads(
            payload[20 : 20 + struct.unpack_from("<I", payload, 12)[0]]
        )["accessors"][0]["count"]
    )
    return MixamoGlbSummary(
        output_path=output,
        source_vertex_count=len(asset.vertices_m),
        exported_vertex_count=exported_vertex_count,
        triangle_count=len(asset.faces),
        bone_count=len(asset.bone_names),
        byte_count=len(payload),
    )


__all__ = [
    "MixamoGlbSummary",
    "build_mixamo_glb",
    "export_mixamo_fbx_glb",
]
