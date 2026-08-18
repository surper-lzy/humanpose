#!/usr/bin/env python3
"""Composite an aligned SMPL mesh into a cached true-Gaussian view."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.pipeline.scene_alignment import (
    DEFAULT_SCENE_ROOT,
    DEFAULT_SMPL_CACHE,
)
from rgbd_avatar.scene import GaussianAlignmentView, SceneAlignment


LOGGER = logging.getLogger("render_avatar_in_3dgs")


def rasterize_mesh_camera(
    vertices_camera: np.ndarray,
    faces: np.ndarray,
    intrinsic_matrix: np.ndarray,
    *,
    height: int,
    width: int,
    base_color_rgb: tuple[float, float, float] = (0.78, 0.55, 0.35),
) -> tuple[np.ndarray, np.ndarray]:
    """Return flat-shaded RGB and camera-Z buffers for one triangle mesh."""

    vertices = np.asarray(vertices_camera, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    intrinsic = np.asarray(intrinsic_matrix, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError("vertices_camera must have shape Vx3.")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape Fx3.")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise ValueError("faces contain an out-of-range vertex index.")
    if intrinsic.shape != (3, 3):
        raise ValueError("intrinsic_matrix must have shape 3x3.")
    if height <= 0 or width <= 0:
        raise ValueError("Raster dimensions must be positive.")

    projected_h = vertices @ intrinsic.T
    projected = np.full((len(vertices), 2), np.nan, dtype=np.float64)
    positive = vertices[:, 2] > 1e-6
    projected[positive] = (
        projected_h[positive, :2] / projected_h[positive, 2:3]
    )
    depth = np.full((height, width), np.inf, dtype=np.float32)
    color = np.zeros((height, width, 3), dtype=np.float32)
    base_color = np.asarray(base_color_rgb, dtype=np.float32)
    if base_color.shape != (3,) or not np.isfinite(base_color).all():
        raise ValueError("base_color_rgb must be a finite RGB triple.")
    light_camera = np.array([-0.25, -0.45, 1.0], dtype=np.float64)
    light_camera /= np.linalg.norm(light_camera)

    for triangle in triangles:
        camera_triangle = vertices[triangle]
        if np.any(camera_triangle[:, 2] <= 1e-6):
            continue
        screen = projected[triangle]
        x_min = max(0, int(np.floor(np.min(screen[:, 0]))))
        x_max = min(width - 1, int(np.ceil(np.max(screen[:, 0]))))
        y_min = max(0, int(np.floor(np.min(screen[:, 1]))))
        y_max = min(height - 1, int(np.ceil(np.max(screen[:, 1]))))
        if x_min > x_max or y_min > y_max:
            continue
        p0, p1, p2 = screen
        area = (
            (p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        if abs(float(area)) <= 1e-10:
            continue
        xs = np.arange(x_min, x_max + 1, dtype=np.float64) + 0.5
        ys = np.arange(y_min, y_max + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        weight0 = (
            (p1[0] - grid_x) * (p2[1] - grid_y)
            - (p1[1] - grid_y) * (p2[0] - grid_x)
        ) / area
        weight1 = (
            (p2[0] - grid_x) * (p0[1] - grid_y)
            - (p2[1] - grid_y) * (p0[0] - grid_x)
        ) / area
        weight2 = 1.0 - weight0 - weight1
        inside = (
            (weight0 >= -1e-7)
            & (weight1 >= -1e-7)
            & (weight2 >= -1e-7)
        )
        if not np.any(inside):
            continue
        reciprocal_depth = (
            weight0 / camera_triangle[0, 2]
            + weight1 / camera_triangle[1, 2]
            + weight2 / camera_triangle[2, 2]
        )
        triangle_depth = np.full_like(reciprocal_depth, np.inf)
        valid_depth = reciprocal_depth > 1e-12
        triangle_depth[valid_depth] = 1.0 / reciprocal_depth[valid_depth]
        depth_slice = depth[y_min : y_max + 1, x_min : x_max + 1]
        update = inside & (triangle_depth < depth_slice)
        if not np.any(update):
            continue
        normal = np.cross(
            camera_triangle[1] - camera_triangle[0],
            camera_triangle[2] - camera_triangle[0],
        )
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm > 1e-12:
            normal /= normal_norm
        brightness = 0.42 + 0.58 * abs(float(np.dot(normal, light_camera)))
        face_color = np.clip(base_color * brightness, 0.0, 1.0)
        depth_slice[update] = triangle_depth[update].astype(np.float32)
        color_slice = color[y_min : y_max + 1, x_min : x_max + 1]
        color_slice[update] = face_color
    return color, depth


def composite_avatar(
    view: GaussianAlignmentView,
    mesh_rgb: np.ndarray,
    mesh_depth_g: np.ndarray,
    *,
    opacity: float = 0.92,
    scene_depth_tolerance_g: float = 0.03,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth-composite mesh buffers over a Gaussian RGB+depth render."""

    if mesh_rgb.shape != view.rgb_uint8.shape:
        raise ValueError("Mesh RGB shape differs from Gaussian view RGB.")
    if mesh_depth_g.shape != view.expected_depth_g.shape:
        raise ValueError("Mesh depth shape differs from Gaussian view depth.")
    if not np.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        raise ValueError("opacity must lie in [0,1].")
    if not np.isfinite(scene_depth_tolerance_g) or scene_depth_tolerance_g < 0:
        raise ValueError("scene_depth_tolerance_g must be non-negative.")
    mesh_present = np.isfinite(mesh_depth_g)
    scene_present = (view.alpha >= 0.05) & (view.expected_depth_g > 0.0)
    visible = mesh_present & (
        ~scene_present
        | (
            mesh_depth_g
            <= view.expected_depth_g + scene_depth_tolerance_g
        )
    )
    alpha = visible.astype(np.float32) * float(opacity)
    background = view.rgb_uint8.astype(np.float32) / 255.0
    output = background * (1.0 - alpha[..., None]) + mesh_rgb * alpha[..., None]
    return np.clip(np.rint(output * 255.0), 0, 255).astype(np.uint8), visible


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", type=Path, required=True)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--smpl-cache", type=Path, default=DEFAULT_SMPL_CACHE)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="SMPL cache array index; default: first present frame.",
    )
    parser.add_argument("--opacity", type=float, default=0.92)
    parser.add_argument("--depth-tolerance-g", type=float, default=0.03)
    parser.add_argument("--output", type=Path, default=None)
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
        view_path = args.view.expanduser().resolve()
        view = GaussianAlignmentView.load(view_path)
        alignment = SceneAlignment.load(alignment_path)
        cache = SMPLSequenceCache.load(args.smpl_cache.expanduser().resolve())
        active = np.flatnonzero(cache.present)
        if len(active) == 0:
            raise ValueError("SMPL cache contains no fitted frame.")
        frame_index = int(active[0]) if args.frame_index is None else args.frame_index
        if not 0 <= frame_index < len(cache.present) or not cache.present[frame_index]:
            raise ValueError("--frame-index must identify a present SMPL frame.")
        output = (
            args.output.expanduser().resolve()
            if args.output is not None
            else scene_root
            / "avatar_composites"
            / f"{Path(view.camera_name).stem}_frame_{frame_index:06d}.png"
        )
        if output.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output}; pass --overwrite to replace it."
            )

        vertices_g = alignment.transform_points_w_to_g(
            cache.vertices_display_m[frame_index]
        )
        camera_rotation = view.camera_to_world_g[:3, :3]
        camera_translation = view.camera_to_world_g[:3, 3]
        vertices_camera = (vertices_g - camera_translation) @ camera_rotation
        mesh_rgb, mesh_depth = rasterize_mesh_camera(
            vertices_camera,
            cache.faces,
            view.intrinsic_matrix,
            height=view.height,
            width=view.width,
        )
        composite, visible = composite_avatar(
            view,
            mesh_rgb,
            mesh_depth,
            opacity=args.opacity,
            scene_depth_tolerance_g=args.depth_tolerance_g,
        )
        if not np.any(np.isfinite(mesh_depth)):
            raise ValueError("Aligned avatar does not project into this camera view.")
        output.parent.mkdir(parents=True, exist_ok=True)
        import cv2

        if not cv2.imwrite(
            str(output),
            cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
        ):
            raise RuntimeError(f"Could not write composite image: {output}")
        LOGGER.info(
            "Saved %s: frame_index=%d rasterized_pixels=%d visible_pixels=%d",
            output,
            frame_index,
            int(np.count_nonzero(np.isfinite(mesh_depth))),
            int(np.count_nonzero(visible)),
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
