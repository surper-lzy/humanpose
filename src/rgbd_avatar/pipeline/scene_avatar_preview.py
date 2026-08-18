#!/usr/bin/env python3
"""Preview an aligned SMPL sequence inside a 3DGS selection point cloud."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.pipeline.scene_alignment import (
    DEFAULT_SCENE_ROOT,
    DEFAULT_SMPL_CACHE,
)
from rgbd_avatar.scene import SceneAlignment


LOGGER = logging.getLogger("view_avatar_in_3dgs")
HELP_TEXT = """
Open3D controls
  Space       play / pause
  Left or A   previous fitted frame
  Right or D  next fitted frame
  H           print this help
  Q           quit

This is an alignment preview: the static scene is displayed as COLMAP/Gaussian
centres. Final 3DGS appearance and depth compositing belong to the renderer.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=DEFAULT_SCENE_ROOT,
    )
    parser.add_argument(
        "--scene-cloud",
        type=Path,
        default=None,
        help="Default: sparse/0/points3D.ply under --scene-root.",
    )
    parser.add_argument(
        "--alignment",
        type=Path,
        default=None,
        help="Default: scene_alignment.json under --scene-root.",
    )
    parser.add_argument(
        "--smpl-cache",
        type=Path,
        default=DEFAULT_SMPL_CACHE,
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional downsampling voxel size in 3DGS units.",
    )
    parser.add_argument(
        "--ground-grid-radius-m",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--ground-grid-step-m",
        type=float,
        default=0.5,
    )
    return parser.parse_args()


def _ground_grid(
    alignment: SceneAlignment,
    *,
    radius_m: float,
    step_m: float,
    o3d: Any,
) -> Any:
    if radius_m <= 0 or step_m <= 0:
        raise ValueError("Ground grid radius and step must be positive.")
    offsets_m = np.arange(-radius_m, radius_m + step_m * 0.5, step_m)
    center = alignment.spawn_point_g
    right = alignment.right_g
    forward = alignment.forward_g
    scale = alignment.scale_g_per_m
    endpoints: list[np.ndarray] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []
    extent_g = radius_m * scale
    for offset_m in offsets_m:
        offset_g = offset_m * scale
        endpoints.extend(
            [
                center + offset_g * right - extent_g * forward,
                center + offset_g * right + extent_g * forward,
                center + offset_g * forward - extent_g * right,
                center + offset_g * forward + extent_g * right,
            ]
        )
        base = len(endpoints) - 4
        lines.extend([[base, base + 1], [base + 2, base + 3]])
        colors.extend([[0.2, 0.5, 0.2], [0.2, 0.5, 0.2]])
    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(np.asarray(endpoints))
    grid.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    grid.colors = o3d.utility.Vector3dVector(np.asarray(colors))
    return grid


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        if not np.isfinite(args.fps) or args.fps <= 0:
            raise ValueError("--fps must be finite and positive.")
        if not np.isfinite(args.voxel_size) or args.voxel_size < 0:
            raise ValueError("--voxel-size must be finite and non-negative.")
        scene_root = args.scene_root.expanduser().resolve()
        scene_cloud_path = (
            args.scene_cloud.expanduser().resolve()
            if args.scene_cloud is not None
            else scene_root / "sparse/0/points3D.ply"
        )
        alignment_path = (
            args.alignment.expanduser().resolve()
            if args.alignment is not None
            else scene_root / "scene_alignment.json"
        )
        smpl_cache_path = args.smpl_cache.expanduser().resolve()
        if not scene_cloud_path.is_file():
            raise FileNotFoundError(f"Scene cloud not found: {scene_cloud_path}")
        alignment = SceneAlignment.load(alignment_path)
        cache = SMPLSequenceCache.load(smpl_cache_path)
        active = np.flatnonzero(cache.present)
        if len(active) == 0:
            raise ValueError("SMPL cache contains no fitted frames.")

        import open3d as o3d

        scene_cloud = o3d.io.read_point_cloud(str(scene_cloud_path))
        if scene_cloud.is_empty():
            raise ValueError(f"No points loaded from {scene_cloud_path}")
        if args.voxel_size > 0:
            scene_cloud = scene_cloud.voxel_down_sample(args.voxel_size)
        first_index = int(active[0])
        vertices_g = alignment.transform_points_w_to_g(
            cache.vertices_display_m[first_index]
        )
        mesh = o3d.geometry.TriangleMesh(
            vertices=o3d.utility.Vector3dVector(vertices_g),
            triangles=o3d.utility.Vector3iVector(cache.faces),
        )
        mesh.paint_uniform_color([0.72, 0.62, 0.52])
        mesh.compute_vertex_normals()
        grid = _ground_grid(
            alignment,
            radius_m=args.ground_grid_radius_m,
            step_m=args.ground_grid_step_m,
            o3d=o3d,
        )
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5 * alignment.scale_g_per_m,
        )
        frame_transform = np.eye(4)
        frame_transform[:3, :3] = alignment.rotation_g_from_w
        frame_transform[:3, 3] = alignment.spawn_point_g
        frame.transform(frame_transform)

        viewer = o3d.visualization.VisualizerWithKeyCallback()
        if not viewer.create_window(
            window_name="Metric SMPL in unrelated 3DGS scene",
            width=1400,
            height=900,
        ):
            raise RuntimeError("Could not create the Open3D preview window.")
        viewer.add_geometry(scene_cloud)
        viewer.add_geometry(grid)
        viewer.add_geometry(frame)
        viewer.add_geometry(mesh)
        viewer.get_render_option().point_size = 2.0
        viewer.get_render_option().mesh_show_back_face = True
        state = {
            "cursor": 0,
            "playing": False,
            "last_advance": time.monotonic(),
        }

        def show_cursor(vis: Any) -> None:
            frame_index = int(active[state["cursor"]])
            mesh.vertices = o3d.utility.Vector3dVector(
                alignment.transform_points_w_to_g(
                    cache.vertices_display_m[frame_index]
                )
            )
            mesh.compute_vertex_normals()
            vis.update_geometry(mesh)
            LOGGER.info(
                "frame_index=%d active=%d/%d",
                int(cache.frame_indices[frame_index]),
                state["cursor"] + 1,
                len(active),
            )

        def move(delta: int) -> Any:
            def callback(vis: Any) -> bool:
                state["playing"] = False
                state["cursor"] = (state["cursor"] + delta) % len(active)
                show_cursor(vis)
                return False

            return callback

        def toggle_play(_: Any) -> bool:
            state["playing"] = not state["playing"]
            state["last_advance"] = time.monotonic()
            return False

        def print_help(_: Any) -> bool:
            print(HELP_TEXT)
            return False

        def close(vis: Any) -> bool:
            vis.close()
            return False

        def animate(vis: Any) -> bool:
            if not state["playing"]:
                return False
            now = time.monotonic()
            if now - state["last_advance"] < 1.0 / args.fps:
                return False
            state["last_advance"] = now
            state["cursor"] = (state["cursor"] + 1) % len(active)
            show_cursor(vis)
            return False

        viewer.register_key_callback(32, toggle_play)
        viewer.register_key_callback(263, move(-1))
        viewer.register_key_callback(262, move(1))
        viewer.register_key_callback(ord("A"), move(-1))
        viewer.register_key_callback(ord("D"), move(1))
        viewer.register_key_callback(ord("H"), print_help)
        viewer.register_key_callback(ord("Q"), close)
        viewer.register_animation_callback(animate)
        print(HELP_TEXT)
        viewer.run()
        viewer.destroy_window()
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
