#!/usr/bin/env python3
"""Play a metric SMPL sequence inside a static interactive 3DGS scene."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.pipeline.gaussian_alignment_view import (
    load_gaussian_ply_tensors,
)
from rgbd_avatar.pipeline.scene_alignment import (
    DEFAULT_SCENE_ROOT,
    DEFAULT_SMPL_CACHE,
)
from rgbd_avatar.pipeline.static_gaussian_avatar_viewer import (
    _SH_C0,
    mesh_surface_sample_coordinates,
)
from rgbd_avatar.scene import GaussianAlignmentView, SceneAlignment


LOGGER = logging.getLogger("view_dynamic_avatar_gaussian_scene")

# Standard SMPL-24 kinematic tree, expressed as parent/child bone segments.
SMPL24_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 4),
    (2, 5),
    (3, 6),
    (4, 7),
    (5, 8),
    (6, 9),
    (7, 10),
    (8, 11),
    (9, 12),
    (9, 13),
    (9, 14),
    (12, 15),
    (13, 16),
    (14, 17),
    (16, 18),
    (17, 19),
    (18, 20),
    (19, 21),
    (20, 22),
    (21, 23),
)


def build_dynamic_avatar_samples(
    cache: SMPLSequenceCache,
    alignment: SceneAlignment,
    *,
    sample_count: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return active cache indices and temporally corresponding G samples."""

    active = np.flatnonzero(cache.present)
    if len(active) == 0:
        raise ValueError("SMPL cache contains no fitted frame.")
    first_vertices_g = alignment.transform_points_w_to_g(
        cache.vertices_display_m[int(active[0])]
    )
    selected_faces, weights = mesh_surface_sample_coordinates(
        first_vertices_g,
        cache.faces,
        sample_count=sample_count,
        seed=seed,
    )
    selected_triangles = np.asarray(cache.faces, dtype=np.int64)[selected_faces]
    samples = np.empty((len(active), sample_count, 3), dtype=np.float32)
    for cursor, cache_index in enumerate(active):
        vertices_g = alignment.transform_points_w_to_g(
            cache.vertices_display_m[int(cache_index)]
        )
        samples[cursor] = np.sum(
            vertices_g[selected_triangles] * weights[..., None],
            axis=1,
        ).astype(np.float32)
    return active.astype(np.int64), samples


def build_dynamic_stick_samples(
    cache: SMPLSequenceCache,
    alignment: SceneAlignment,
    *,
    samples_per_bone: int = 12,
    bone_radius_m: float = 0.022,
    joint_radius_m: float = 0.035,
    head_radius_m: float = 0.105,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build temporally corresponding Gaussian centers for an SMPL stick rig."""

    active = np.flatnonzero(cache.present)
    if len(active) == 0:
        raise ValueError("SMPL cache contains no fitted frame.")
    if samples_per_bone <= 0:
        raise ValueError("samples_per_bone must be positive.")
    radii = np.asarray(
        (bone_radius_m, joint_radius_m, head_radius_m), dtype=np.float64
    )
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("Stick-figure radii must be finite and positive.")
    interpolation = np.linspace(
        0.0,
        1.0,
        samples_per_bone + 2,
        dtype=np.float64,
    )[1:-1]
    point_count = len(SMPL24_BONES) * samples_per_bone + 24
    samples = np.empty((len(active), point_count, 3), dtype=np.float32)
    for cursor, cache_index in enumerate(active):
        joints_g = alignment.transform_points_w_to_g(
            cache.joints_display_m[int(cache_index)]
        )
        bone_points = [
            (1.0 - interpolation[:, None]) * joints_g[parent]
            + interpolation[:, None] * joints_g[child]
            for parent, child in SMPL24_BONES
        ]
        samples[cursor] = np.vstack((*bone_points, joints_g)).astype(np.float32)
    sample_radii_m = np.concatenate(
        (
            np.full(
                len(SMPL24_BONES) * samples_per_bone,
                bone_radius_m,
                dtype=np.float32,
            ),
            np.full(24, joint_radius_m, dtype=np.float32),
        )
    )
    # SMPL joint 15 is the head; enlarge it into the matchstick head.
    sample_radii_m[-24 + 15] = head_radius_m
    return active.astype(np.int64), samples, sample_radii_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-root", type=Path, default=DEFAULT_SCENE_ROOT)
    parser.add_argument("--scene-ply", type=Path, default=None)
    parser.add_argument("--alignment", type=Path, default=None)
    parser.add_argument("--smpl-cache", type=Path, default=DEFAULT_SMPL_CACHE)
    parser.add_argument("--initial-view", type=Path, default=None)
    parser.add_argument(
        "--avatar-style",
        choices=("surface", "stick"),
        default="surface",
        help="Render the SMPL skin surface or a joint/bone matchstick rig.",
    )
    parser.add_argument("--avatar-samples", type=int, default=50_000)
    parser.add_argument("--avatar-radius-m", type=float, default=0.012)
    parser.add_argument("--stick-samples-per-bone", type=int, default=12)
    parser.add_argument("--stick-bone-radius-m", type=float, default=0.022)
    parser.add_argument("--stick-joint-radius-m", type=float, default=0.035)
    parser.add_argument("--stick-head-radius-m", type=float, default=0.105)
    parser.add_argument(
        "--avatar-color-rgb",
        type=int,
        nargs=3,
        default=None,
        metavar=("R", "G", "B"),
    )
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--viewer-resolution", type=int, default=960)
    return parser.parse_args()


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
        if not np.isfinite(args.fps) or args.fps <= 0:
            raise ValueError("--fps must be finite and positive.")
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
        smpl_cache_path = args.smpl_cache.expanduser().resolve()
        for path in (scene_ply, alignment_path, smpl_cache_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        alignment = SceneAlignment.load(alignment_path)
        cache = SMPLSequenceCache.load(smpl_cache_path)
        if args.avatar_style == "surface":
            active, avatar_frames = build_dynamic_avatar_samples(
                cache,
                alignment,
                sample_count=args.avatar_samples,
            )
            avatar_radii_m = np.full(
                args.avatar_samples,
                args.avatar_radius_m,
                dtype=np.float32,
            )
            default_color = (215, 125, 55)
        else:
            active, avatar_frames, avatar_radii_m = (
                build_dynamic_stick_samples(
                    cache,
                    alignment,
                    samples_per_bone=args.stick_samples_per_bone,
                    bone_radius_m=args.stick_bone_radius_m,
                    joint_radius_m=args.stick_joint_radius_m,
                    head_radius_m=args.stick_head_radius_m,
                )
            )
            default_color = (25, 25, 25)
        avatar_point_count = int(avatar_frames.shape[1])
        initial_view_path = (
            args.initial_view.expanduser().resolve()
            if args.initial_view is not None
            else None
        )
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

        import torch
        import torch.nn.functional as functional
        import viser
        from gsplat.rendering import rasterization
        from nerfview import CameraState, RenderTabState, Viewer

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Interactive Gaussian viewing requires CUDA.")
        LOGGER.info("Loading static Gaussian scene %s on %s", scene_ply, device)
        scene_splats, sh_degree = load_gaussian_ply_tensors(
            scene_ply,
            device=device,
            torch=torch,
        )
        coefficient_count = int(scene_splats["colors"].shape[1])
        avatar_color = np.asarray(
            default_color
            if args.avatar_color_rgb is None
            else args.avatar_color_rgb,
            dtype=np.float32,
        )
        if np.any(avatar_color < 0) or np.any(avatar_color > 255):
            raise ValueError("--avatar-color-rgb values must lie in [0,255].")
        avatar_sh = np.zeros(
            (avatar_point_count, coefficient_count, 3), dtype=np.float32
        )
        avatar_sh[:, 0, :] = (avatar_color / 255.0 - 0.5) / _SH_C0
        avatar_means = torch.from_numpy(avatar_frames[0]).to(device)
        avatar_quats = torch.zeros(
            (avatar_point_count, 4), dtype=torch.float32, device=device
        )
        avatar_quats[:, 0] = 1.0
        avatar_scales = torch.from_numpy(
            np.repeat(
                (
                    avatar_radii_m[:, None]
                    * alignment.scale_g_per_m
                ),
                3,
                axis=1,
            )
        ).to(device=device, dtype=torch.float32)
        avatar_opacities = torch.full(
            (avatar_point_count,), 0.97, dtype=torch.float32, device=device
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
        avatar_start = len(scene_splats["means"])
        gpu_lock = threading.Lock()
        LOGGER.info(
            "Dynamic viewer: %d static scene Gaussians + %d avatar Gaussians, "
            "%d fitted frames, style=%s",
            avatar_start,
            avatar_point_count,
            len(active),
            args.avatar_style,
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
                camera_state.c2w, dtype=torch.float32, device=device
            )
            intrinsic = torch.as_tensor(
                camera_state.get_K((width, height)),
                dtype=torch.float32,
                device=device,
            )
            with gpu_lock, torch.inference_mode():
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
            rgb = rendered[0, ..., :3] + (1.0 - alpha[0])
            return rgb.clamp(0.0, 1.0).cpu().numpy()

        server = viser.ViserServer(host=args.host, port=args.port, verbose=False)
        server.scene.set_up_direction(alignment.ground_normal_g)

        @server.on_client_connect
        def initialize_camera(client: Any) -> None:
            if initial_view is not None:
                c2w = initial_view.camera_to_world_g
                client.camera.position = c2w[:3, 3]
                client.camera.look_at = c2w[:3, 3] + 5.0 * c2w[:3, 2]
                client.camera.up_direction = -c2w[:3, 1]

        viewer = Viewer(
            server=server,
            render_fn=render_fn,
            output_dir=scene_root / "viewer_output",
            mode="rendering",
        )
        viewer.render_tab_state.viewer_res = args.viewer_resolution
        state = {"cursor": 0, "last_advance": time.monotonic()}

        with server.gui.add_folder("Avatar Animation"):
            playing = server.gui.add_checkbox("Play", initial_value=False)
            frame_slider = server.gui.add_slider(
                "Frame",
                min=0,
                max=len(active) - 1,
                step=1,
                initial_value=0,
            )
            source_frame = server.gui.add_number(
                "Source frame",
                initial_value=int(cache.frame_indices[int(active[0])]),
                disabled=True,
            )
            fps_control = server.gui.add_number(
                "FPS",
                initial_value=args.fps,
                min=0.5,
                max=60.0,
                step=0.5,
            )
            previous_button = server.gui.add_button("Previous")
            next_button = server.gui.add_button("Next")

        def show_cursor(cursor: int, event: Any = None) -> None:
            cursor = int(cursor) % len(active)
            state["cursor"] = cursor
            frame_slider.value = cursor
            cache_index = int(active[cursor])
            source_frame.value = int(cache.frame_indices[cache_index])
            frame_tensor = torch.from_numpy(avatar_frames[cursor]).to(device)
            with gpu_lock, torch.inference_mode():
                means[avatar_start:].copy_(frame_tensor)
            viewer.rerender(event)

        @frame_slider.on_update
        def change_frame(event: Any) -> None:
            requested = int(frame_slider.value)
            if requested != state["cursor"]:
                show_cursor(requested, event)

        @previous_button.on_click
        def previous_frame(event: Any) -> None:
            playing.value = False
            show_cursor(state["cursor"] - 1, event)

        @next_button.on_click
        def next_frame(event: Any) -> None:
            playing.value = False
            show_cursor(state["cursor"] + 1, event)

        LOGGER.info(
            "Dynamic 3DGS avatar viewer: http://%s:%d",
            args.host,
            args.port,
        )
        print("Use Avatar Animation controls in the browser; Ctrl-C stops it.")
        try:
            while True:
                now = time.monotonic()
                fps = float(fps_control.value)
                if playing.value and now - state["last_advance"] >= 1.0 / fps:
                    state["last_advance"] = now
                    show_cursor(state["cursor"] + 1)
                time.sleep(0.01)
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
