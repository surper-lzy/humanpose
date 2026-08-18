#!/usr/bin/env python3
"""Browser-based live viewer for the textured Mixamo avatar.

Runs alongside the existing Open3D mannequin window: a ``viser`` server
serves a browser page, ``nerfview`` streams rendered frames, and every
frame the latest Halpe26 pose is retargeted onto the Mixamo skeleton,
skinned via LBS, and rendered with Open3D's Filament ``OffscreenRenderer``.

Usage (added to the existing live mannequin pipeline)::

    PYTHONPATH=src python scripts/view_live_mannequin.py \
        --mixamo-cache outputs/sequences/.../mixamo_sequence.npz

Then open the printed http://127.0.0.1:<port> URL in a browser next to the
Open3D mannequin window.
"""

from __future__ import annotations

import gc
import logging
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rgbd_avatar.avatar import (
    MixamoSequenceCache,
    load_mixamo_fbx,
    skin_mixamo_vertices,
)
from rgbd_avatar.avatar.mixamo_asset import MixamoAsset
from rgbd_avatar.retargeting.halpe_mixamo import (
    MixamoAnalyticalIK,
    MixamoIKConfig,
    MixamoIKFrame,
)
from rgbd_avatar.retargeting.halpe_smpl import (
    HalpeSMPLRetargetProfile,
    RobustLengthPrior,
)

try:
    import open3d as o3d
except ImportError as error:
    raise RuntimeError(
        "Open3D is required for Filament offscreen rendering. "
        "Run from the gsplat environment."
    ) from error

try:
    import nerfview
    import viser
except ImportError as error:
    raise RuntimeError(
        "viser + nerfview are required. Run from the gsplat environment."
    ) from error


LOGGER = logging.getLogger("live_mixamo_viewer")


# ---------------------------------------------------------------------------
# Profile / asset reconstruction from an existing Mixamo cache
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveMixamoSetup:
    """Pre-built objects needed by the live Mixamo viewer."""

    asset: MixamoAsset
    solver: MixamoAnalyticalIK
    diffuse_texture_rgb: np.ndarray
    faces: np.ndarray
    triangle_uvs: np.ndarray


def load_live_mixamo_setup(cache_path: str | Path) -> LiveMixamoSetup:
    """Reconstruct the IK solver and rendering data from a Mixamo cache.

    The ``mixamo_sequence.npz`` created by ``fit_mixamo_sequence.py`` holds
    the retarget profile, effective scale, IK config, and FBX path. This
    function loads the FBX once, rebuilds the solver, and returns everything
    needed for per-frame live skinning + rendering.
    """
    cache = MixamoSequenceCache.load(cache_path)
    meta = cache.metadata
    length_priors: dict[str, RobustLengthPrior] = {}
    for name, prior_dict in meta["retarget_profile"]["length_priors"].items():
        length_priors[name] = RobustLengthPrior(**prior_dict)
    profile = HalpeSMPLRetargetProfile(length_priors=length_priors)
    model_path = Path(meta["model"]).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Mixamo FBX model not found: {model_path}. The cache was "
            "created on a different machine or the asset was moved."
        )
    # ``fit_mixamo_sequence.py --scale`` can override the estimated value.
    # The cache field records the scale that was actually used by the offline
    # solver, whereas metadata["estimated_scale"] only records the estimate.
    scale = float(cache.scale)
    ik_cfg = meta.get("ik_config", {})
    config = MixamoIKConfig(
        maximum_rotation_speed_deg_s=float(
            ik_cfg.get("maximum_rotation_speed_deg_s", 180.0)
        ),
        rotation_response=float(ik_cfg.get("rotation_response", 0.78)),
        minimum_segment_confidence=float(
            ik_cfg.get("minimum_segment_confidence", 0.35)
        ),
        minimum_hand_confidence=float(
            ik_cfg.get("minimum_hand_confidence", 0.12)
        ),
    )
    asset = load_mixamo_fbx(model_path)
    solver = MixamoAnalyticalIK(asset, profile, scale=scale, config=config)
    diffuse_rgb = cv2.cvtColor(
        cv2.imdecode(cache.diffuse_png, cv2.IMREAD_COLOR),
        cv2.COLOR_BGR2RGB,
    )
    if diffuse_rgb is None:
        raise ValueError("Could not decode Mixamo diffuse texture.")
    return LiveMixamoSetup(
        asset=asset,
        solver=solver,
        diffuse_texture_rgb=diffuse_rgb,
        faces=cache.faces,
        triangle_uvs=cache.triangle_uvs.reshape(-1, 2).astype(np.float64),
    )


# ---------------------------------------------------------------------------
# Per-frame Filament offscreen renderer (single engine, reused)
# ---------------------------------------------------------------------------


# (width, height) → OffscreenRenderer cache to avoid destroying the old
# engine before creating the new one (this build segfaults on dual engines).
_ENGINE_LOCK = threading.Lock()


class LiveMixamoOffscreen:
    """Holds the static geometry (faces, UVs, texture) and one Filament
    engine. Vertices are uploaded only when the solved-pose version changes."""

    def __init__(
        self,
        setup: LiveMixamoSetup,
        width: int,
        height: int,
    ) -> None:
        self._faces = setup.faces
        self._uvs = setup.triangle_uvs
        self._diffuse = setup.diffuse_texture_rgb
        self._view_size: tuple[int, int] | None = None
        self._renderer: Any = None
        self._material: Any = None
        self._mesh_added = False
        self._uploaded_geometry_version = -1
        self._resize(max(int(width), 1), max(int(height), 1))

    def _resize(self, width: int, height: int) -> None:
        if self._view_size == (width, height) and self._renderer is not None:
            return
        self._view_size = (width, height)
        with _ENGINE_LOCK:
            self._renderer = None
            gc.collect()
            self._renderer = o3d.visualization.rendering.OffscreenRenderer(
                width, height
            )
        scene = self._renderer.scene
        scene.set_background(np.array([0.05, 0.05, 0.08, 1.0]))
        scene.set_lighting(
            scene.LightingProfile.MED_SHADOWS,
            np.array([0.35, -0.5, -0.8], dtype=np.float32),
        )
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        mat.albedo_img = o3d.geometry.Image(self._diffuse)
        self._material = mat
        self._mesh_added = False
        self._uploaded_geometry_version = -1

    def render(
        self,
        vertices_display_m: np.ndarray,
        c2w: np.ndarray,
        fov_rad: float,
        geometry_version: int,
    ) -> np.ndarray:
        """Upload changed vertices and render into a uint8 image."""
        scene = self._renderer.scene
        if int(geometry_version) != self._uploaded_geometry_version:
            first_upload = self._uploaded_geometry_version < 0
            mesh = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(
                    np.asarray(vertices_display_m, dtype=np.float64)
                ),
                o3d.utility.Vector3iVector(self._faces),
            )
            mesh.triangle_uvs = o3d.utility.Vector2dVector(self._uvs)
            mesh.compute_vertex_normals()
            if self._mesh_added:
                scene.remove_geometry("mixamo")
            scene.add_geometry("mixamo", mesh, self._material)
            self._mesh_added = True
            self._uploaded_geometry_version = int(geometry_version)
            if first_upload:
                LOGGER.info(
                    "Live Mixamo mesh uploaded: version=%d vertices=%d "
                    "triangles=%d",
                    self._uploaded_geometry_version,
                    len(vertices_display_m),
                    len(self._faces),
                )

        c2w = np.asarray(c2w, dtype=np.float64)
        eye = c2w[:3, 3]
        forward = c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])
        up = c2w[:3, :3] @ np.array([0.0, -1.0, 0.0])
        forward /= np.linalg.norm(forward) or 1.0
        up /= np.linalg.norm(up) or 1.0
        center = eye + forward * max(
            float(np.linalg.norm(eye)), 0.1
        )
        self._renderer.setup_camera(
            math.degrees(float(fov_rad)),
            center,
            eye,
            up,
            0.05,
            100.0,
        )
        return np.asarray(self._renderer.render_to_image())


# ---------------------------------------------------------------------------
# viser + nerfview viewer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SolvedLivePose:
    """Latest-only pose result consumed by per-client render callbacks."""

    input_version: int
    geometry_version: int
    frame_number: int
    vertices_display_m: np.ndarray | None
    status: str
    solve_ms: float


class _LiveMixamoPoseSolver:
    """Advance stateful IK exactly once per processed sensor frame."""

    def __init__(self, setup: LiveMixamoSetup) -> None:
        self._setup = setup
        self._last_vertices: np.ndarray | None = None
        self._last_timestamp_ns: int | None = None
        self._geometry_version = 0

    def solve(
        self,
        result: Any,
        *,
        input_version: int,
        reset: bool = False,
    ) -> _SolvedLivePose:
        timestamp_ns = int(result.timestamp_ns)
        timestamp_regressed = bool(
            self._last_timestamp_ns is not None
            and timestamp_ns < self._last_timestamp_ns
        )
        if reset or timestamp_regressed:
            self._setup.solver.reset()
            self._last_vertices = None
            self._last_timestamp_ns = None

        if self._last_timestamp_ns is None or timestamp_ns <= self._last_timestamp_ns:
            delta_time_s = 0.033
        else:
            delta_time_s = max(
                (timestamp_ns - self._last_timestamp_ns) * 1e-9,
                1e-3,
            )
        self._last_timestamp_ns = timestamp_ns

        tic = time.perf_counter()
        temporal = result.pose3d_output
        ik_frame: MixamoIKFrame | None = self._setup.solver.solve(
            np.asarray(result.joints_application_m, dtype=np.float64),
            np.asarray(temporal.confidence, dtype=np.float64),
            np.asarray(temporal.usable, dtype=bool),
            np.asarray(temporal.predicted, dtype=bool),
            delta_time_s=delta_time_s,
        )

        if ik_frame is None:
            vertices = self._last_vertices
            status = (
                "Waiting for a valid root"
                if vertices is None
                else "Holding (no valid root)"
            )
        else:
            vertices = skin_mixamo_vertices(
                self._setup.asset,
                ik_frame.bone_global_m,
            )
            self._last_vertices = vertices
            self._geometry_version += 1
            status = "Live"

        return _SolvedLivePose(
            input_version=int(input_version),
            geometry_version=self._geometry_version,
            frame_number=int(result.frame_number),
            vertices_display_m=vertices,
            status=status,
            solve_ms=(time.perf_counter() - tic) * 1000.0,
        )


class ViserLiveMixamoViewer(nerfview.Viewer):
    """nerfview Viewer that streams the live Mixamo avatar to a browser.

    Call :meth:`update` from the main / Open3D thread whenever a fresh
    ``LivePoseResult`` is available. A latest-only worker runs IK + skinning
    once per sensor frame and wakes nerfview; render threads only draw the
    newest solved vertices and push JPEGs to their browser clients.
    """

    def __init__(
        self,
        server: viser.ViserServer,
        setup: LiveMixamoSetup,
        *,
        res: int = 1024,
    ) -> None:
        self._setup = setup
        self._res = int(res)
        self._condition = threading.Condition()
        self._latest_pose: Any = None  # LivePoseResult (imported lazily)
        self._latest_pose_key: tuple[str, int, int] | None = None
        self._pending_version = 0
        self._reset_pending = False
        self._closed = False
        self._solved_pose: _SolvedLivePose | None = None
        self._pose_solver = _LiveMixamoPoseSolver(setup)
        self._offscreen: LiveMixamoOffscreen | None = None
        self._camera_focused_on_pose = False
        self._viewer_center: np.ndarray | None = None
        # Coarse scene extent for the initial camera.
        verts = setup.asset.vertices_m
        self._viewer_center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        self._view_distance = max(
            float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))) * 1.5,
            1.0,
        )
        super().__init__(
            server=server,
            render_fn=self._render_frame,
            mode="rendering",
        )
        server.gui.set_panel_label("Live Mixamo Avatar")
        self._solve_thread = threading.Thread(
            target=self._solve_loop,
            name="live-mixamo-solver",
            daemon=True,
        )
        self._solve_thread.start()

    # ------------------------------------------------------------------
    # GUI – keep it minimal: the Open3D window is the primary control.
    # ------------------------------------------------------------------

    def _init_rendering_tab(self) -> None:
        self.render_tab_state = nerfview.RenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Live Mixamo")

    def _populate_rendering_tab(self) -> None:
        with self._rendering_folder:
            status_text = self.server.gui.add_text(
                "Status",
                initial_value="Waiting for first pose…",
            )
            frame_number = self.server.gui.add_number(
                "Frame",
                min=0,
                max=999999,
                step=1,
                initial_value=0,
                disabled=True,
            )
            solve_time = self.server.gui.add_number(
                "IK ms",
                min=0.0,
                max=999.0,
                step=0.1,
                initial_value=0.0,
                disabled=True,
            )
            reset_button = self.server.gui.add_button(
                "Reset View",
                hint="Return to the initial camera view.",
            )

            @reset_button.on_click
            def _(_) -> None:
                self._reset_client_cameras()

        self._handles = {
            "status_text": status_text,
            "frame_number": frame_number,
            "solve_time": solve_time,
            "reset_button": reset_button,
        }
        self._rendering_tab_handles.update(self._handles)

    # ------------------------------------------------------------------
    # Client camera
    # ------------------------------------------------------------------

    def _connect_client(self, client: viser.ClientHandle) -> None:
        super()._connect_client(client)
        self._set_client_camera(client)

    def _set_client_camera(self, client: viser.ClientHandle) -> None:
        with self._condition:
            center = self._viewer_center.copy()
            distance = self._view_distance
        with self.server.atomic():
            client.camera.look_at = center
            client.camera.position = center + np.array(
                [0.0, -distance, distance * 0.5]
            )
            client.camera.fov = 0.6

    def _reset_client_cameras(self) -> None:
        for client in self.server.get_clients().values():
            self._set_client_camera(client)

    # ------------------------------------------------------------------
    # Pose input (called from main / Open3D thread)
    # ------------------------------------------------------------------

    def update(self, result: Any) -> None:  # result: LivePoseResult
        """Enqueue the latest live pose for rendering."""
        key = (
            str(result.source_id),
            int(result.frame_number),
            int(result.timestamp_ns),
        )
        presence = result.presence
        reset_requested = bool(
            presence.track_reset_required or presence.reacquired_after_exit
        )
        with self._condition:
            if self._closed or key == self._latest_pose_key:
                return
            if (
                self._latest_pose_key is not None
                and key[0] != self._latest_pose_key[0]
            ):
                reset_requested = True
            self._latest_pose = result
            self._latest_pose_key = key
            self._pending_version += 1
            # Do not lose a reset event when the latest-only queue replaces the
            # frame that originally carried it.
            self._reset_pending = self._reset_pending or reset_requested
            self._condition.notify()

    def _solve_loop(self) -> None:
        processed_version = 0
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._closed
                    or self._pending_version != processed_version
                )
                if self._closed:
                    return
                result = self._latest_pose
                version = self._pending_version
                reset = self._reset_pending
                self._reset_pending = False
                if reset:
                    self._camera_focused_on_pose = False

            assert result is not None
            try:
                solved = self._pose_solver.solve(
                    result,
                    input_version=version,
                    reset=reset,
                )
            except Exception:
                LOGGER.exception(
                    "Live Mixamo solve failed for frame %s; continuing",
                    getattr(result, "frame_number", "unknown"),
                )
                processed_version = version
                continue

            focus_camera = False
            focus_center = None
            focus_distance = None
            with self._condition:
                processed_version = version
                # A newer frame arrived while this one was being solved. Keep
                # the solver's temporal advance, but publish only the newest
                # available result to the browser.
                if version != self._pending_version:
                    continue
                self._solved_pose = solved
                vertices = solved.vertices_display_m
                if vertices is not None and not self._camera_focused_on_pose:
                    minimum = vertices.min(axis=0)
                    maximum = vertices.max(axis=0)
                    focus_center = (minimum + maximum) / 2.0
                    focus_distance = max(
                        float(np.linalg.norm(maximum - minimum)) * 1.5,
                        1.0,
                    )
                    self._viewer_center = focus_center
                    self._view_distance = focus_distance
                    self._camera_focused_on_pose = True
                    focus_camera = True
            if focus_camera:
                LOGGER.info(
                    "Live Mixamo camera focused on frame=%d center=%s "
                    "distance=%.3f m",
                    solved.frame_number,
                    np.round(focus_center, 3).tolist(),
                    focus_distance,
                )
                self._reset_client_cameras()
            try:
                self.rerender(None)
            except Exception:
                with self._condition:
                    closing = self._closed
                if not closing:
                    LOGGER.exception(
                        "Could not request a live Mixamo browser rerender"
                    )

    # ------------------------------------------------------------------
    # Render callback (nerfview render thread)
    # ------------------------------------------------------------------

    def _render_frame(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        with self._condition:
            solved = self._solved_pose
        if solved is None:
            return self._empty_image(render_tab_state)

        # Update GUI (rate-limited by nerfview's render cadence).
        with self.server.atomic():
            self._handles["status_text"].value = solved.status
            self._handles["frame_number"].value = solved.frame_number
            self._handles["solve_time"].value = float(
                round(solved.solve_ms, 1)
            )

        if solved.vertices_display_m is None:
            return self._empty_image(render_tab_state)

        # Resolve render resolution.
        res = int(self._res)
        aspect = float(camera_state.aspect)
        if aspect >= 1.0:
            width, height = res, max(int(round(res / aspect)), 1)
        else:
            width, height = max(int(round(res * aspect)), 1), res

        if self._offscreen is None:
            LOGGER.info(
                "Live Mixamo Filament offscreen renderer ready (EGL "
                "headless); first render at %dx%d.",
                width,
                height,
            )
            self._offscreen = LiveMixamoOffscreen(
                self._setup, width, height
            )
            self._reset_client_cameras()

        return self._offscreen.render(
            solved.vertices_display_m,
            camera_state.c2w,
            camera_state.fov,
            solved.geometry_version,
        )

    def close(self) -> None:
        """Stop the latest-only IK worker before shutting down the server."""
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._solve_thread.is_alive():
            self._solve_thread.join(timeout=2.0)

    @staticmethod
    def _empty_image(rt: nerfview.RenderTabState) -> np.ndarray:
        h = max(int(getattr(rt, "viewer_height", 240)), 1)
        w = max(int(getattr(rt, "viewer_width", 320)), 1)
        return np.full((h, w, 3), 20, dtype=np.uint8)
