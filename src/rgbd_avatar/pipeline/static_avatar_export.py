#!/usr/bin/env python3
"""Export one SMPL pose as a triangle mesh in the 3DGS world coordinates."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.io import atomic_write_json
from rgbd_avatar.pipeline.scene_alignment import (
    DEFAULT_SCENE_ROOT,
    DEFAULT_SMPL_CACHE,
)
from rgbd_avatar.scene import SceneAlignment


LOGGER = logging.getLogger("export_static_avatar_to_3dgs")


def transform_static_avatar(
    cache: SMPLSequenceCache,
    alignment: SceneAlignment,
    *,
    frame_index: int | None = None,
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Transform one fitted SMPL frame from metric world W into scene G."""

    if not isinstance(cache, SMPLSequenceCache):
        raise TypeError("cache must be an SMPLSequenceCache.")
    if not isinstance(alignment, SceneAlignment):
        raise TypeError("alignment must be a SceneAlignment.")
    active = np.flatnonzero(cache.present)
    if len(active) == 0:
        raise ValueError("SMPL cache contains no fitted frame.")
    selected = int(active[0]) if frame_index is None else int(frame_index)
    if not 0 <= selected < len(cache.present) or not cache.present[selected]:
        raise ValueError("frame_index must identify a present SMPL frame.")

    vertices_w_m = np.asarray(cache.vertices_display_m[selected], dtype=np.float64)
    joints_w_m = np.asarray(cache.joints_display_m[selected], dtype=np.float64)
    vertices_g = alignment.transform_points_w_to_g(vertices_w_m)
    joints_g = alignment.transform_points_w_to_g(joints_w_m)
    faces = np.asarray(cache.faces, dtype=np.int32)
    signed_ground_g = (
        vertices_g @ alignment.ground_normal_g + alignment.ground_offset_g
    )
    signed_ground_m = signed_ground_g / alignment.scale_g_per_m
    mapped_anchor_g = alignment.transform_points_w_to_g(
        alignment.avatar_anchor_w_m
    )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "coordinate_system": "3dgs_world",
        "source_coordinate_system": "display_x_right_y_forward_z_up_m",
        "cache_frame_index": selected,
        "source_frame_index": int(cache.frame_indices[selected]),
        "vertex_count": int(len(vertices_g)),
        "face_count": int(len(faces)),
        "joint_count": int(len(joints_g)),
        "scale_g_per_m": alignment.scale_g_per_m,
        "rotation_g_from_w": alignment.rotation_g_from_w.tolist(),
        "translation_g_from_w": alignment.translation_g_from_w.tolist(),
        "transform_g_from_w": alignment.transform_g_from_w.tolist(),
        "avatar_anchor_w_m": alignment.avatar_anchor_w_m.tolist(),
        "mapped_anchor_g": mapped_anchor_g.tolist(),
        "spawn_point_g": alignment.spawn_point_g.tolist(),
        "bounds_min_g": np.min(vertices_g, axis=0).tolist(),
        "bounds_max_g": np.max(vertices_g, axis=0).tolist(),
        "extent_g": np.ptp(vertices_g, axis=0).tolist(),
        "height_along_scene_up_g": float(np.ptp(signed_ground_g)),
        "height_along_scene_up_m": float(np.ptp(signed_ground_m)),
        "ground_clearance_min_m": float(np.min(signed_ground_m)),
        "ground_clearance_max_m": float(np.max(signed_ground_m)),
    }
    return selected, vertices_g, joints_g, faces, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--smpl-cache", type=Path, default=DEFAULT_SMPL_CACHE)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="SMPL cache array index; default: first present frame.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output triangle-mesh PLY in G coordinates.",
    )
    parser.add_argument(
        "--color-rgb",
        type=int,
        nargs=3,
        default=(199, 140, 89),
        metavar=("R", "G", "B"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        scene_root = args.scene_root.expanduser().resolve()
        alignment_path = (
            args.alignment.expanduser().resolve()
            if args.alignment is not None
            else scene_root / "scene_alignment.json"
        )
        smpl_cache_path = args.smpl_cache.expanduser().resolve()
        alignment = SceneAlignment.load(alignment_path)
        cache = SMPLSequenceCache.load(smpl_cache_path)
        selected, vertices_g, joints_g, faces, metadata = (
            transform_static_avatar(
                cache,
                alignment,
                frame_index=args.frame_index,
            )
        )
        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else scene_root
            / "avatar_static"
            / f"avatar_frame_{selected:06d}_3dgs.ply"
        )
        metadata_output = output.with_suffix(".json")
        joints_output = output.with_name(output.stem + "_joints.npy")
        targets = (output, metadata_output, joints_output)
        existing = [path for path in targets if path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {existing[0]}; pass --overwrite."
            )
        color = np.asarray(args.color_rgb, dtype=np.int64)
        if color.shape != (3,) or np.any(color < 0) or np.any(color > 255):
            raise ValueError("--color-rgb values must lie in [0,255].")

        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices_g),
            triangles=o3d.utility.Vector3iVector(faces),
        )
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            np.broadcast_to(color / 255.0, vertices_g.shape).copy()
        )
        mesh.compute_vertex_normals()
        output.parent.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(
            str(output),
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            print_progress=False,
        ):
            raise RuntimeError(f"Could not write triangle mesh: {output}")
        np.save(joints_output, joints_g, allow_pickle=False)
        metadata.update(
            {
                "mesh_ply": str(output),
                "joints_npy": str(joints_output),
                "scene_alignment": str(alignment_path),
                "smpl_cache": str(smpl_cache_path),
            }
        )
        atomic_write_json(metadata_output, metadata)
        LOGGER.info(
            "Saved static 3DGS-space avatar %s: frame=%d vertices=%d "
            "faces=%d height=%.4fm ground_min=%.4fm",
            output,
            selected,
            len(vertices_g),
            len(faces),
            metadata["height_along_scene_up_m"],
            metadata["ground_clearance_min_m"],
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
