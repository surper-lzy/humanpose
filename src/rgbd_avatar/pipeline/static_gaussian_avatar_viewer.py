#!/usr/bin/env python3
"""Interactively view a static 3DGS scene and aligned avatar in one world."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from rgbd_avatar.pipeline.gaussian_alignment_view import (
    load_gaussian_ply_tensors,
)
from rgbd_avatar.pipeline.scene_alignment import DEFAULT_SCENE_ROOT
from rgbd_avatar.scene import GaussianAlignmentView, SceneAlignment


LOGGER = logging.getLogger("view_static_avatar_gaussian_scene")
_SH_C0 = 0.28209479177387814


def sample_mesh_surface(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sample_count: int,
    seed: int = 0,
) -> np.ndarray:
    """Area-sample a triangle mesh for temporary Gaussian visualization."""

    selected, weights = mesh_surface_sample_coordinates(
        vertices,
        faces,
        sample_count=sample_count,
        seed=seed,
    )
    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    return np.sum(points[triangles[selected]] * weights[..., None], axis=1)


def mesh_surface_sample_coordinates(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    sample_count: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reusable face indices and barycentric surface coordinates."""

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("vertices must have shape Vx3.")
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape Fx3.")
    if np.any(triangles < 0) or np.any(triangles >= len(points)):
        raise ValueError("faces contain an out-of-range vertex index.")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive.")
    triangle_vertices = points[triangles]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        ),
        axis=1,
    )
    total_area = float(np.sum(areas))
    if not math.isfinite(total_area) or total_area <= 0.0:
        raise ValueError("Avatar mesh has no positive-area triangle.")
    generator = np.random.default_rng(seed)
    selected = generator.choice(
        len(triangles),
        size=sample_count,
        replace=True,
        p=areas / total_area,
    )
    random_uv = generator.random((sample_count, 2))
    sqrt_u = np.sqrt(random_uv[:, 0])
    weights = np.column_stack(
        (
            1.0 - sqrt_u,
            sqrt_u * (1.0 - random_uv[:, 1]),
            sqrt_u * random_uv[:, 1],
        )
    )
    return selected.astype(np.int64), weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene-ply", type=Path, default=None)
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--avatar-ply", type=Path, default=None)
    parser.add_argument("--initial-view", type=Path, default=None)
    parser.add_argument("--avatar-samples", type=int, default=50_000)
    parser.add_argument("--avatar-radius-m", type=float, default=0.012)
    parser.add_argument(
        "--avatar-color-rgb",
        type=int,
        nargs=3,
        default=(215, 125, 55),
        metavar=("R", "G", "B"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewer-resolution", type=int, default=960)
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    scene_root = args.scene_root.expanduser().resolve()
    scene_ply = (
        args.scene_ply.expanduser().resolve()
        if args.scene_ply is not None
        else scene_root / "point_cloud.ply"
    )
    alignment_path = (
        args.alignment.expanduser().resolve()
        if args.alignment is not None
        else scene_root / "scene_alignment.json"
    )
    avatar_ply = (
        args.avatar_ply.expanduser().resolve()
        if args.avatar_ply is not None
        else scene_root / "avatar_static/avatar_frame_000000_3dgs.ply"
    )
    initial_view = (
        args.initial_view.expanduser().resolve()
        if args.initial_view is not None
        else None
    )
    return scene_ply, alignment_path, avatar_ply, initial_view


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        if not 1 <= args.port <= 65535:
            raise ValueError("--port must lie in [1,65535].")
        if args.avatar_samples <= 0:
            raise ValueError("--avatar-samples must be positive.")
        if not np.isfinite(args.avatar_radius_m) or args.avatar_radius_m <= 0:
            raise ValueError("--avatar-radius-m must be finite and positive.")
        scene_ply, alignment_path, avatar_ply, initial_view_path = (
            _resolve_paths(args)
        )
        for path in (scene_ply, alignment_path, avatar_ply):
            if not path.is_file():
                raise FileNotFoundError(path)
        alignment = SceneAlignment.load(alignment_path)
        if initial_view_path is None:
            selection_view = alignment.metadata.get("selection_view")
            if isinstance(selection_view, str):
                candidate = Path(selection_view).expanduser().resolve()
                if candidate.is_file():
                    initial_view_path = candidate
        initial_view = (
            GaussianAlignmentView.load(initial_view_path)
            if initial_view_path is not None
            else None
        )

        import open3d as o3d
        import torch
        import torch.nn.functional as functional
        import viser
        from gsplat.rendering import rasterization
        from nerfview import CameraState, RenderTabState, Viewer

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Interactive Gaussian viewing requires CUDA.")
        LOGGER.info("Loading Gaussian scene %s on %s", scene_ply, device)
        scene_splats, sh_degree = load_gaussian_ply_tensors(
            scene_ply,
            device=device,
            torch=torch,
        )
        avatar_mesh = o3d.io.read_triangle_mesh(str(avatar_ply))
        avatar_vertices = np.asarray(avatar_mesh.vertices)
        avatar_faces = np.asarray(avatar_mesh.triangles)
        if len(avatar_vertices) == 0 or len(avatar_faces) == 0:
            raise ValueError(f"Avatar PLY is not a triangle mesh: {avatar_ply}")
        avatar_means_np = sample_mesh_surface(
            avatar_vertices,
            avatar_faces,
            sample_count=args.avatar_samples,
        ).astype(np.float32)
        coefficient_count = int(scene_splats["colors"].shape[1])
        avatar_color = np.asarray(args.avatar_color_rgb, dtype=np.float32)
        if np.any(avatar_color < 0) or np.any(avatar_color > 255):
            raise ValueError("--avatar-color-rgb values must lie in [0,255].")
        avatar_sh = np.zeros(
            (args.avatar_samples, coefficient_count, 3),
            dtype=np.float32,
        )
        avatar_sh[:, 0, :] = (avatar_color / 255.0 - 0.5) / _SH_C0
        avatar_means = torch.from_numpy(avatar_means_np).to(device)
        avatar_quats = torch.zeros(
            (args.avatar_samples, 4), dtype=torch.float32, device=device
        )
        avatar_quats[:, 0] = 1.0
        avatar_scales = torch.full(
            (args.avatar_samples, 3),
            args.avatar_radius_m * alignment.scale_g_per_m,
            dtype=torch.float32,
            device=device,
        )
        avatar_opacities = torch.full(
            (args.avatar_samples,), 0.97, dtype=torch.float32, device=device
        )
        means = torch.cat((scene_splats["means"], avatar_means), dim=0)
        quats = torch.cat(
            (
                functional.normalize(scene_splats["quats"], dim=-1),
                avatar_quats,
            ),
            dim=0,
        )
        scales = torch.cat(
            (torch.exp(scene_splats["scales"]), avatar_scales), dim=0
        )
        opacities = torch.cat(
            (torch.sigmoid(scene_splats["opacities"]), avatar_opacities),
            dim=0,
        )
        colors = torch.cat(
            (
                scene_splats["colors"],
                torch.from_numpy(avatar_sh).to(device),
            ),
            dim=0,
        )
        LOGGER.info(
            "Viewer scene: %d scene Gaussians + %d avatar Gaussians",
            len(scene_splats["means"]),
            args.avatar_samples,
        )

        def render_fn(
            camera_state: CameraState,
            render_state: RenderTabState,
        ) -> np.ndarray:
            if render_state.preview_render:
                width = render_state.render_width
                height = render_state.render_height
            else:
                width = render_state.viewer_width
                height = render_state.viewer_height
            c2w = torch.as_tensor(
                camera_state.c2w,
                dtype=torch.float32,
                device=device,
            )
            intrinsic = torch.as_tensor(
                camera_state.get_K((width, height)),
                dtype=torch.float32,
                device=device,
            )
            with torch.inference_mode():
                rendered, alpha, _ = rasterization(
                    means=means,
                    quats=quats,
                    scales=scales,
                    opacities=opacities,
                    colors=colors,
                    viewmats=torch.linalg.inv(c2w).unsqueeze(0),
                    Ks=intrinsic.unsqueeze(0),
                    width=width,
                    height=height,
                    sh_degree=sh_degree,
                    packed=True,
                    render_mode="RGB",
                )
            rgb = rendered[0, ..., :3]
            rgb = rgb + (1.0 - alpha[0])
            return rgb.clamp(0.0, 1.0).cpu().numpy()

        server = viser.ViserServer(host=args.host, port=args.port, verbose=False)
        server.scene.set_up_direction(alignment.ground_normal_g)

        @server.on_client_connect
        def initialize_camera(client: Any) -> None:
            if initial_view is not None:
                camera_to_world = initial_view.camera_to_world_g
                client.camera.position = camera_to_world[:3, 3]
                client.camera.look_at = (
                    camera_to_world[:3, 3]
                    + 5.0 * camera_to_world[:3, 2]
                )
                client.camera.up_direction = -camera_to_world[:3, 1]
            else:
                client.camera.position = (
                    alignment.spawn_point_g
                    + 3.0 * alignment.scale_g_per_m * alignment.forward_g
                    + 1.5 * alignment.scale_g_per_m * alignment.ground_normal_g
                )
                client.camera.look_at = (
                    alignment.spawn_point_g
                    + 0.9 * alignment.scale_g_per_m * alignment.ground_normal_g
                )
                client.camera.up_direction = alignment.ground_normal_g

        viewer = Viewer(
            server=server,
            render_fn=render_fn,
            output_dir=scene_ply.parent / "viewer_output",
            mode="rendering",
        )
        viewer.render_tab_state.viewer_res = args.viewer_resolution
        LOGGER.info(
            "Interactive 3DGS + static avatar viewer: http://%s:%d",
            args.host,
            args.port,
        )
        print("Press Ctrl-C to stop the viewer.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            LOGGER.info("Viewer stopped.")
        finally:
            server.stop()
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
