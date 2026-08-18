#!/usr/bin/env python3
"""Interactively pick measurements and create a case-one 3DGS placement."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.pipeline.scene_alignment import (
    DEFAULT_SCENE_ROOT,
    DEFAULT_SMPL_CACHE,
    write_alignment,
)
from rgbd_avatar.scene import ManualScenePlacement, first_avatar_ground_anchor


LOGGER = logging.getLogger("pick_scene_alignment")


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
        help=(
            "Point cloud used for picking. Default: sparse/0/points3D.ply "
            "under --scene-root; the final Gaussian PLY remains the bound scene."
        ),
    )
    parser.add_argument(
        "--scene-ply",
        type=Path,
        default=None,
        help="Final Gaussian PLY. Default: point_cloud.ply under --scene-root.",
    )
    parser.add_argument(
        "--known-distance-m",
        type=float,
        required=True,
        help="Real metric distance between the two points picked in stage 1.",
    )
    parser.add_argument(
        "--description",
        default="Interactive placement in an unrelated 3DGS scene",
    )
    parser.add_argument(
        "--smpl-cache",
        type=Path,
        default=DEFAULT_SMPL_CACHE,
    )
    parser.add_argument(
        "--anchor-mode",
        choices=("feet", "pelvis", "origin"),
        default="feet",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.0,
        help="Optional positive downsampling voxel size in 3DGS units.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: scene_alignment.json under --scene-root.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _pick_indices(
    cloud: Any,
    *,
    title: str,
    instruction: str,
    minimum: int,
    maximum: int | None,
) -> list[int]:
    import open3d as o3d

    print("\n" + instruction)
    print("Open3D: Shift+左键选点，Shift+右键撤销，Q 关闭当前窗口。")
    viewer = o3d.visualization.VisualizerWithEditing()
    if not viewer.create_window(window_name=title, width=1280, height=800):
        raise RuntimeError("Could not create the Open3D picking window.")
    viewer.add_geometry(cloud)
    viewer.get_render_option().point_size = 4.0
    viewer.run()
    viewer.destroy_window()
    picked = [int(index) for index in viewer.get_picked_points()]
    if len(picked) < minimum or (maximum is not None and len(picked) > maximum):
        expected = (
            str(minimum)
            if maximum == minimum
            else f"{minimum}..{maximum if maximum is not None else 'N'}"
        )
        raise ValueError(
            f"{title}: expected {expected} picked points, got {len(picked)}."
        )
    return picked


def _picked_points(cloud: Any, indices: list[int]) -> np.ndarray:
    points = np.asarray(cloud.points, dtype=np.float64)
    return points[np.asarray(indices, dtype=np.int64)]


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        if not np.isfinite(args.known_distance_m) or args.known_distance_m <= 0:
            raise ValueError("--known-distance-m must be finite and positive.")
        if not np.isfinite(args.voxel_size) or args.voxel_size < 0:
            raise ValueError("--voxel-size must be finite and non-negative.")
        scene_root = args.scene_root.expanduser().resolve()
        selection_cloud_path = (
            args.scene_cloud.expanduser().resolve()
            if args.scene_cloud is not None
            else scene_root / "sparse/0/points3D.ply"
        )
        scene_ply = (
            args.scene_ply.expanduser().resolve()
            if args.scene_ply is not None
            else scene_root / "point_cloud.ply"
        )
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else scene_root / "scene_alignment.json"
        )
        smpl_cache_path = args.smpl_cache.expanduser().resolve()
        if not selection_cloud_path.is_file():
            raise FileNotFoundError(
                f"Selection point cloud not found: {selection_cloud_path}"
            )

        import open3d as o3d

        cloud = o3d.io.read_point_cloud(str(selection_cloud_path))
        if cloud.is_empty():
            raise ValueError(f"No points loaded from {selection_cloud_path}")
        if args.voxel_size > 0:
            cloud = cloud.voxel_down_sample(args.voxel_size)
            if cloud.is_empty():
                raise ValueError("Voxel downsampling removed every scene point.")
        LOGGER.info(
            "Loaded %d selection points from %s",
            len(cloud.points),
            selection_cloud_path,
        )

        length_indices = _pick_indices(
            cloud,
            title="1/4 Known metric length",
            instruction=(
                "阶段 1/4：选择已知真实距离的两个端点，选择顺序不影响尺度。"
            ),
            minimum=2,
            maximum=2,
        )
        ground_indices = _pick_indices(
            cloud,
            title="2/4 Ground plane",
            instruction=(
                "阶段 2/4：在同一地面上选择至少 3 个分散且不共线的点。"
            ),
            minimum=3,
            maximum=None,
        )
        up_indices = _pick_indices(
            cloud,
            title="3/4 Above-ground reference",
            instruction=(
                "阶段 3/4：选择一个明确位于地面上方的点，用来确定向上方向。"
            ),
            minimum=1,
            maximum=1,
        )
        placement_indices = _pick_indices(
            cloud,
            title="4/4 Spawn and forward",
            instruction=(
                "阶段 4/4：先选择人物出生点，再选择人物面朝方向上的一点。"
            ),
            minimum=2,
            maximum=2,
        )

        length_points = _picked_points(cloud, length_indices)
        ground_points = _picked_points(cloud, ground_indices)
        up_reference = _picked_points(cloud, up_indices)[0]
        placement_points = _picked_points(cloud, placement_indices)
        cache = SMPLSequenceCache.load(smpl_cache_path)
        anchor = first_avatar_ground_anchor(cache, mode=args.anchor_mode)
        placement = ManualScenePlacement(
            known_point_a_g=length_points[0],
            known_point_b_g=length_points[1],
            known_distance_m=args.known_distance_m,
            ground_points_g=ground_points,
            up_reference_g=up_reference,
            spawn_point_g=placement_points[0],
            forward_point_g=placement_points[1],
            avatar_anchor_w_m=anchor,
            description=args.description,
        )
        write_alignment(
            placement,
            output_path=output_path,
            scene_root=scene_root,
            scene_ply=scene_ply,
            smpl_cache_path=smpl_cache_path,
            overwrite=args.overwrite,
            extra_metadata={
                "selection_cloud": str(selection_cloud_path),
                "selection_cloud_point_count": len(cloud.points),
                "avatar_anchor_mode": args.anchor_mode,
            },
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
