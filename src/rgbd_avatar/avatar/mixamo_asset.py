"""Minimal binary-FBX importer for one skinned Mixamo character asset.

The importer intentionally supports the subset emitted by Mixamo: one mesh,
one linear skin deformer, limb-node models, polygon-vertex UVs, and embedded
PNG textures.  It removes Blender/Assimp from the runtime dependency chain
while validating every relationship needed by linear blend skinning.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Any
import zlib

import numpy as np


_FBX_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"


@dataclass(frozen=True)
class _FBXNode:
    name: str
    properties: tuple[Any, ...]
    children: tuple["_FBXNode", ...]

    def all(self, name: str) -> tuple["_FBXNode", ...]:
        return tuple(child for child in self.children if child.name == name)

    def first(self, name: str) -> "_FBXNode":
        matches = self.all(name)
        if not matches:
            raise ValueError(f"FBX node {self.name!r} has no child {name!r}.")
        return matches[0]


class _BinaryFBXReader:
    def __init__(self, payload: bytes) -> None:
        if payload[:23] != _FBX_MAGIC:
            raise ValueError("Only binary FBX files are supported.")
        self.payload = payload
        self.version = struct.unpack_from("<I", payload, 23)[0]
        self.wide = self.version >= 7500

    def parse(self) -> tuple[_FBXNode, ...]:
        offset = 27
        roots: list[_FBXNode] = []
        while offset < len(self.payload):
            node, following = self._node(offset)
            if node is None:
                break
            if following <= offset:
                raise ValueError("FBX node offsets are not strictly increasing.")
            roots.append(node)
            offset = following
        return tuple(roots)

    def _property(self, offset: int) -> tuple[Any, int]:
        payload = self.payload
        kind = chr(payload[offset])
        offset += 1
        scalar_formats = {
            "Y": "h", "C": "?", "I": "i",
            "F": "f", "D": "d", "L": "q",
        }
        if kind in scalar_formats:
            fmt = "<" + scalar_formats[kind]
            return (
                struct.unpack_from(fmt, payload, offset)[0],
                offset + struct.calcsize(fmt),
            )
        if kind in ("S", "R"):
            length = struct.unpack_from("<I", payload, offset)[0]
            offset += 4
            raw = payload[offset:offset + length]
            value = raw.decode("utf-8", "replace") if kind == "S" else raw
            return value, offset + length
        array_dtypes = {
            "f": "<f4", "d": "<f8", "l": "<i8",
            "i": "<i4", "b": "?", "c": "i1",
        }
        if kind in array_dtypes:
            count, encoding, stored_length = struct.unpack_from(
                "<III", payload, offset
            )
            offset += 12
            raw = payload[offset:offset + stored_length]
            if encoding == 1:
                raw = zlib.decompress(raw)
            elif encoding != 0:
                raise ValueError(f"Unsupported FBX array encoding {encoding}.")
            values = np.frombuffer(raw, dtype=array_dtypes[kind], count=count).copy()
            if len(values) != count:
                raise ValueError("Truncated FBX array property.")
            return values, offset + stored_length
        raise ValueError(f"Unsupported FBX property type {kind!r}.")

    def _node(self, offset: int) -> tuple[_FBXNode | None, int]:
        start = offset
        if self.wide:
            end, count, _property_bytes = struct.unpack_from(
                "<QQQ", self.payload, offset
            )
            offset += 24
            null_bytes = 25
        else:
            end, count, _property_bytes = struct.unpack_from(
                "<III", self.payload, offset
            )
            offset += 12
            null_bytes = 13
        name_length = self.payload[offset]
        offset += 1
        if end == 0:
            return None, start + null_bytes
        if end > len(self.payload) or end <= offset:
            raise ValueError("Invalid FBX node end offset.")
        name = self.payload[offset:offset + name_length].decode(
            "utf-8", "replace"
        )
        offset += name_length
        properties: list[Any] = []
        for _ in range(count):
            value, offset = self._property(offset)
            properties.append(value)
        children: list[_FBXNode] = []
        child_end = end - null_bytes
        while offset < child_end:
            child, offset = self._node(offset)
            if child is None:
                break
            children.append(child)
        return _FBXNode(name, tuple(properties), tuple(children)), end


def _clean_object_name(value: str) -> str:
    name = value.split("\x00", 1)[0]
    return name.rsplit(":", 1)[-1]


def _matrix_m(value: np.ndarray, unit_to_m: float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (16,) or not np.isfinite(array).all():
        raise ValueError("FBX bind matrices must contain 16 finite values.")
    # FBX stores matrices in row-vector order.  Internally use conventional
    # column vectors and scale translation from centimetres to metres.
    matrix = array.reshape(4, 4).T.copy()
    matrix[:3, 3] *= unit_to_m
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-7):
        raise ValueError("FBX bind matrix is not affine.")
    return matrix


@dataclass(frozen=True)
class MixamoAsset:
    vertices_m: np.ndarray
    faces: np.ndarray
    triangle_uvs: np.ndarray
    bone_names: tuple[str, ...]
    parent_indices: np.ndarray
    bind_global_m: np.ndarray
    inverse_bind_m: np.ndarray
    skin_joint_indices: np.ndarray
    skin_weights: np.ndarray
    diffuse_png: bytes
    source_path: str
    source_sha256: str

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices_m, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int32)
        uvs = np.asarray(self.triangle_uvs, dtype=np.float64)
        parents = np.asarray(self.parent_indices, dtype=np.int32)
        bind = np.asarray(self.bind_global_m, dtype=np.float64)
        inverse = np.asarray(self.inverse_bind_m, dtype=np.float64)
        joints = np.asarray(self.skin_joint_indices, dtype=np.int32)
        weights = np.asarray(self.skin_weights, dtype=np.float64)
        bone_count = len(self.bone_names)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("Mixamo vertices must have shape Vx3.")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("Mixamo faces must have shape Fx3.")
        if uvs.shape != (len(faces), 3, 2):
            raise ValueError("Mixamo triangle UVs must have shape Fx3x2.")
        if parents.shape != (bone_count,):
            raise ValueError("Mixamo parent indices have invalid shape.")
        if bind.shape != (bone_count, 4, 4) or inverse.shape != bind.shape:
            raise ValueError("Mixamo bind matrices have invalid shape.")
        if joints.shape != (len(vertices), 4) or weights.shape != joints.shape:
            raise ValueError("Mixamo skin influences must have shape Vx4.")
        if len(set(self.bone_names)) != bone_count:
            raise ValueError("Mixamo bone names must be unique.")
        if bone_count == 0 or parents[0] != -1:
            raise ValueError("Mixamo skeleton must start with one root bone.")
        if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, bone_count)):
            raise ValueError("Mixamo bones must be stored in topological order.")
        if faces.size and (faces.min() < 0 or faces.max() >= len(vertices)):
            raise ValueError("Mixamo faces reference invalid vertices.")
        if np.any(joints < 0) or np.any(joints >= bone_count):
            raise ValueError("Mixamo skin references invalid bones.")
        if not all(np.isfinite(x).all() for x in (vertices, uvs, bind, inverse, weights)):
            raise ValueError("Mixamo asset arrays must be finite.")
        if np.any(weights < 0.0) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError("Mixamo skin weights must be normalized.")
        if not self.diffuse_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise ValueError("Mixamo diffuse texture must be embedded PNG data.")
        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "faces", faces)
        object.__setattr__(self, "triangle_uvs", uvs)
        object.__setattr__(self, "parent_indices", parents)
        object.__setattr__(self, "bind_global_m", bind)
        object.__setattr__(self, "inverse_bind_m", inverse)
        object.__setattr__(self, "skin_joint_indices", joints)
        object.__setattr__(self, "skin_weights", weights)

    @property
    def bone_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.bone_names)}


def load_mixamo_fbx(path: str | Path) -> MixamoAsset:
    """Load and validate one standard Mixamo binary FBX character."""

    source = Path(path).expanduser().resolve()
    payload = source.read_bytes()
    roots = _BinaryFBXReader(payload).parse()
    root_by_name = {node.name: node for node in roots}
    if "Objects" not in root_by_name or "Connections" not in root_by_name:
        raise ValueError("FBX is missing Objects or Connections.")
    objects = root_by_name["Objects"]
    connections = [tuple(node.properties) for node in root_by_name["Connections"].all("C")]
    by_id = {
        int(node.properties[0]): node
        for node in objects.children
        if node.properties and isinstance(node.properties[0], (int, np.integer))
    }

    geometries = [
        node for node in objects.all("Geometry")
        if len(node.properties) >= 3 and node.properties[2] == "Mesh"
    ]
    skins = [
        node for node in objects.all("Deformer")
        if len(node.properties) >= 3 and node.properties[2] == "Skin"
    ]
    if len(geometries) != 1 or len(skins) != 1:
        raise ValueError("Mixamo importer requires exactly one mesh and one skin.")
    geometry, skin = geometries[0], skins[0]
    raw_vertices = np.asarray(geometry.first("Vertices").properties[0], dtype=np.float64)
    if len(raw_vertices) % 3:
        raise ValueError("FBX vertex array length is not divisible by three.")
    # Mixamo's UnitScaleFactor=1 means centimetres.
    unit_to_m = 0.01
    vertices_m = raw_vertices.reshape(-1, 3) * unit_to_m

    polygon_indices = np.asarray(
        geometry.first("PolygonVertexIndex").properties[0], dtype=np.int64
    )
    uv_layer = geometry.first("LayerElementUV")
    mapping = str(uv_layer.first("MappingInformationType").properties[0])
    reference = str(uv_layer.first("ReferenceInformationType").properties[0])
    if mapping != "ByPolygonVertex" or reference != "IndexToDirect":
        raise ValueError(
            "Mixamo importer requires ByPolygonVertex/IndexToDirect UVs."
        )
    uv_direct = np.asarray(uv_layer.first("UV").properties[0], dtype=np.float64).reshape(-1, 2)
    uv_indices = np.asarray(uv_layer.first("UVIndex").properties[0], dtype=np.int64)
    if len(uv_indices) != len(polygon_indices):
        raise ValueError("FBX polygon and UV-index arrays are misaligned.")

    faces: list[tuple[int, int, int]] = []
    triangle_uvs: list[np.ndarray] = []
    polygon: list[int] = []
    polygon_uv: list[np.ndarray] = []
    for polygon_vertex, uv_index in zip(polygon_indices, uv_indices, strict=True):
        control_index = int(-polygon_vertex - 1 if polygon_vertex < 0 else polygon_vertex)
        if not 0 <= control_index < len(vertices_m):
            raise ValueError("FBX polygon references an invalid control point.")
        if not 0 <= uv_index < len(uv_direct):
            raise ValueError("FBX polygon references an invalid UV coordinate.")
        polygon.append(control_index)
        polygon_uv.append(uv_direct[uv_index])
        if polygon_vertex < 0:
            if len(polygon) < 3:
                raise ValueError("FBX polygon has fewer than three vertices.")
            for corner in range(1, len(polygon) - 1):
                faces.append((polygon[0], polygon[corner], polygon[corner + 1]))
                triangle_uvs.append(
                    np.asarray(
                        (polygon_uv[0], polygon_uv[corner], polygon_uv[corner + 1]),
                        dtype=np.float64,
                    )
                )
            polygon.clear()
            polygon_uv.clear()
    if polygon:
        raise ValueError("FBX polygon array ends without a terminator.")

    cluster_nodes = [
        by_id[int(connection[1])]
        for connection in connections
        if len(connection) >= 3
        and connection[0] == "OO"
        and int(connection[2]) == int(skin.properties[0])
        and int(connection[1]) in by_id
        and by_id[int(connection[1])].name == "Deformer"
        and by_id[int(connection[1])].properties[2] == "Cluster"
    ]
    if not cluster_nodes:
        raise ValueError("Mixamo skin contains no bone clusters.")
    cluster_by_model_id: dict[int, _FBXNode] = {}
    for cluster in cluster_nodes:
        model_ids = [
            int(connection[1])
            for connection in connections
            if len(connection) >= 3
            and connection[0] == "OO"
            and int(connection[2]) == int(cluster.properties[0])
            and int(connection[1]) in by_id
            and by_id[int(connection[1])].name == "Model"
        ]
        if len(model_ids) != 1:
            raise ValueError("Each Mixamo cluster must connect to one bone model.")
        cluster_by_model_id[model_ids[0]] = cluster

    model_parent: dict[int, int] = {}
    model_children: dict[int, list[int]] = {}
    for connection in connections:
        if len(connection) < 3 or connection[0] != "OO":
            continue
        child, parent = int(connection[1]), int(connection[2])
        if child in cluster_by_model_id and parent in cluster_by_model_id:
            model_parent[child] = parent
            model_children.setdefault(parent, []).append(child)
    roots_ids = [model_id for model_id in cluster_by_model_id if model_id not in model_parent]
    if len(roots_ids) != 1:
        raise ValueError("Mixamo skin must contain exactly one bone root.")
    ordered_ids: list[int] = []

    def visit(model_id: int) -> None:
        ordered_ids.append(model_id)
        for child in model_children.get(model_id, []):
            visit(child)

    visit(roots_ids[0])
    if len(ordered_ids) != len(cluster_by_model_id):
        raise ValueError("Mixamo bone hierarchy is disconnected or cyclic.")
    ordered_index = {model_id: index for index, model_id in enumerate(ordered_ids)}
    bone_names = tuple(
        _clean_object_name(str(by_id[model_id].properties[1]))
        for model_id in ordered_ids
    )
    parent_indices = np.asarray(
        [
            -1 if model_id not in model_parent
            else ordered_index[model_parent[model_id]]
            for model_id in ordered_ids
        ],
        dtype=np.int32,
    )
    bind_global = np.stack(
        [
            _matrix_m(
                cluster_by_model_id[model_id].first("TransformLink").properties[0],
                unit_to_m,
            )
            for model_id in ordered_ids
        ]
    )
    inverse_bind = np.linalg.inv(bind_global)
    # Mixamo stores the same inverse in Cluster.Transform.  Validate rather
    # than depend on exporter-specific naming semantics.
    for model_id, expected in zip(ordered_ids, inverse_bind, strict=True):
        stored = _matrix_m(
            cluster_by_model_id[model_id].first("Transform").properties[0],
            unit_to_m,
        )
        if not np.allclose(stored, expected, atol=2e-4):
            raise ValueError("FBX cluster bind matrices are inconsistent.")

    dense_weights = np.zeros((len(vertices_m), len(ordered_ids)), dtype=np.float64)
    for model_id in ordered_ids:
        cluster = cluster_by_model_id[model_id]
        index_nodes = cluster.all("Indexes")
        weight_nodes = cluster.all("Weights")
        if not index_nodes and not weight_nodes:
            continue
        if len(index_nodes) != 1 or len(weight_nodes) != 1:
            raise ValueError("Mixamo cluster indices/weights are incomplete.")
        indices = np.asarray(index_nodes[0].properties[0], dtype=np.int64)
        weights = np.asarray(weight_nodes[0].properties[0], dtype=np.float64)
        if len(indices) != len(weights) or np.any(indices < 0) or np.any(indices >= len(vertices_m)):
            raise ValueError("Mixamo cluster weights reference invalid vertices.")
        dense_weights[indices, ordered_index[model_id]] += weights
    if np.any(dense_weights.sum(axis=1) <= 1e-8):
        raise ValueError("Mixamo mesh contains unweighted vertices.")
    top = np.argpartition(dense_weights, -4, axis=1)[:, -4:]
    top_weights = np.take_along_axis(dense_weights, top, axis=1)
    order = np.argsort(-top_weights, axis=1)
    top = np.take_along_axis(top, order, axis=1).astype(np.int32)
    top_weights = np.take_along_axis(top_weights, order, axis=1)
    top_weights /= top_weights.sum(axis=1, keepdims=True)

    diffuse_png = b""
    texture_ids = {
        int(connection[1])
        for connection in connections
        if len(connection) >= 4
        and connection[0] == "OP"
        and str(connection[3]) == "DiffuseColor"
    }
    for connection in connections:
        if len(connection) < 3 or connection[0] != "OO":
            continue
        video_id, texture_id = int(connection[1]), int(connection[2])
        if texture_id not in texture_ids or video_id not in by_id:
            continue
        video = by_id[video_id]
        content = video.first("Content").properties[0]
        if isinstance(content, bytes) and content.startswith(b"\x89PNG"):
            diffuse_png = content
            break
    if not diffuse_png:
        raise ValueError("Mixamo FBX has no embedded diffuse PNG texture.")

    return MixamoAsset(
        vertices_m=vertices_m,
        faces=np.asarray(faces, dtype=np.int32),
        triangle_uvs=np.asarray(triangle_uvs, dtype=np.float64),
        bone_names=bone_names,
        parent_indices=parent_indices,
        bind_global_m=bind_global,
        inverse_bind_m=inverse_bind,
        skin_joint_indices=top,
        skin_weights=top_weights,
        diffuse_png=diffuse_png,
        source_path=str(source),
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


__all__ = ["MixamoAsset", "load_mixamo_fbx"]
