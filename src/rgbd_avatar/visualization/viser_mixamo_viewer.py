#!/usr/bin/env python3
"""Browser-based viewer for the textured Mixamo motion sequence.

Architecture mirrors the gsplat ``simple_viewer2.py`` example: a ``viser``
server serves a browser page (orbit camera + GUI panels), ``nerfview``
streams rendered frames back to the page as JPEGs, and Python renders the
textured Mixamo mesh with Open3D's Filament ``OffscreenRenderer`` (EGL
headless, so it also works on headless machines).

Why not the legacy Open3D Visualizer?  Open3D 0.18/0.19 segfault inside
``poll_events()`` as soon as a ``TriangleMesh`` with ``textures`` is added
to the legacy GL viewer (reproduced on CPU and CUDA builds, NVIDIA and
software GL, any texture size/format).  The Filament renderer used here is
unaffected and renders textures correctly.

Usage::

    PYTHONPATH=src python scripts/view_mixamo_sequence_viser.py \
        --results-dir outputs/sequences/4_pointcloud_exit_gate

then open the printed http://127.0.0.1:<port> URL in a browser.

Browser controls:
  Mouse drag  orbit the camera
  Wheel       zoom
  Button      play / pause
  Sliders     time, speed
  Checkbox    loop
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rgbd_avatar.avatar import MixamoSequenceCache

try:
    import open3d as o3d
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "Open3D is required. Install it in the environment that also "
        "provides viser/nerfview (e.g. the gsplat env)."
    ) from error

try:
    import nerfview
    import viser
except ImportError as error:  # pragma: no cover
    raise RuntimeError(
        "The browser viewer needs viser + nerfview. Run it from the gsplat "
        "environment (which already provides both)."
    ) from error


LOGGER = logging.getLogger("view_mixamo_sequence_viser")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class MixamoFilamentRenderer:
    """Renders one textured Mixamo frame with the Filament offscreen renderer.

    The renderer holds a persistent scene; the mesh is re-uploaded with
    ``remove_geometry`` + ``add_geometry`` only when the frame changes (the
    per-frame vertex update API in this Open3D build only accepts tensor
    point clouds, not triangle meshes).  The EGL context is created lazily
    on the thread that first calls :meth:`render` and must stay on one
    thread; nerfview serializes all ``render_fn`` calls with a global lock,
    so a single local browser client is safe.
    """

    def __init__(self, cache: MixamoSequenceCache, width: int, height: int) -> None:
        self._cache = cache
        rgb = cv2.cvtColor(
            cv2.imdecode(cache.diffuse_png, cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB,
        )
        if rgb is None:
            raise ValueError("Could not decode Mixamo diffuse texture.")
        material = o3d.visualization.rendering.MaterialRecord()
        material.shader = "defaultLit"
        material.albedo_img = o3d.geometry.Image(rgb)
        self._material = material
        self._uvs = cache.triangle_uvs.reshape(-1, 2).astype(np.float64)
        self._faces = cache.faces
        self._view_size: tuple[int, int] | None = None
        self._renderer: Any = None
        verts = cache.vertices_display_m[cache.present].reshape(-1, 3)
        self._center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        self._view_distance = float(
            np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))
        )
        self._added_frame = -1
        self._first_finite_frame = int(np.argmax(cache.present))
        self._resize(max(int(width), 1), max(int(height), 1))

    def _resize(self, width: int, height: int) -> None:
        """(Re)create the offscreen engine at a new render resolution.

        ``Open3DS.set_view_size`` only affects the viewport, not the fixed
        offscreen framebuffer, so a resolution change requires rebuilding the
        ``OffscreenRenderer``.  This Open3D build crashes when a second
        Filament engine is created while the first is still alive, so the old
        engine is dropped and garbage-collected *before* the new one is
        created.  Rebuilding drops the scene, so any geometry that was
        uploaded is re-added afterwards.
        """
        self._view_size = (width, height)
        self._renderer = None
        gc.collect()
        self._renderer = o3d.visualization.rendering.OffscreenRenderer(
            width,
            height,
        )
        scene = self._renderer.scene
        scene.set_background(np.array([0.05, 0.05, 0.08, 1.0]))
        scene.set_lighting(
            scene.LightingProfile.MED_SHADOWS,
            np.array([0.35, -0.5, -0.8], dtype=np.float32),
        )
        if self._added_frame >= 0:
            scene.add_geometry(
                "mixamo",
                self._build_mesh(self._cache.vertices_display_m[self._added_frame]),
                self._material,
            )

    def _build_mesh(self, vertices: np.ndarray) -> Any:
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(self._faces),
        )
        mesh.triangle_uvs = o3d.utility.Vector2dVector(self._uvs)
        mesh.compute_vertex_normals()
        return mesh

    def render(
        self,
        frame: int,
        c2w: np.ndarray,
        fov_rad: float,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Render frame ``frame`` into a uint8 [H, W, 3] image."""
        width, height = max(int(width), 1), max(int(height), 1)
        if (width, height) != self._view_size:
            self._resize(width, height)
        scene = self._renderer.scene
        vertices = self._cache.vertices_display_m[frame]
        if not np.isfinite(vertices).all():
            # Absent frame: hold the last valid pose instead of NaN geometry.
            if self._added_frame >= 0:
                return self._render_view(c2w, fov_rad)
            vertices = self._cache.vertices_display_m[self._first_finite_frame]
        if frame != self._added_frame:
            scene.remove_geometry("mixamo")
            scene.add_geometry("mixamo", self._build_mesh(vertices), self._material)
            self._added_frame = frame
        return self._render_view(c2w, fov_rad)

    def _render_view(self, c2w: np.ndarray, fov_rad: float) -> np.ndarray:
        c2w = np.asarray(c2w, dtype=np.float64)
        eye = c2w[:3, 3]
        forward = c2w[:3, :3] @ np.array([0.0, 0.0, 1.0])
        up = c2w[:3, :3] @ np.array([0.0, -1.0, 0.0])
        forward /= np.linalg.norm(forward) or 1.0
        up /= np.linalg.norm(up) or 1.0
        distance = max(float(np.linalg.norm(self._center - eye)), 0.1)
        center = eye + forward * distance
        self._renderer.setup_camera(
            math.degrees(float(fov_rad)),
            center,
            eye,
            up,
            0.05,
            100.0,
        )
        return np.asarray(self._renderer.render_to_image())


class MixamoViewer(nerfview.Viewer):
    """nerfview Viewer that plays the textured Mixamo sequence."""

    def __init__(
        self,
        server: viser.ViserServer,
        cache: MixamoSequenceCache,
        fps: float = 30.0,
        res: int = 1024,
    ) -> None:
        self._cache = cache
        self._n_frames = len(cache.present)
        self._fps = float(fps)
        self._res = int(res)
        self._frame = 0
        self._playing = False
        self._loop = True
        self._speed = 1.0
        self._offscreen: MixamoFilamentRenderer | None = None
        self._stop = threading.Event()
        self._state_lock = threading.Lock()
        self._handles: dict[str, Any] = {}
        verts = cache.vertices_display_m[cache.present].reshape(-1, 3)
        self._viewer_center = (verts.min(axis=0) + verts.max(axis=0)) / 2.0
        self._view_distance = max(
            float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0))),
            1.0,
        )
        super().__init__(
            server=server,
            render_fn=self._render_frame,
            mode="rendering",
        )
        self._playback_thread = threading.Thread(
            target=self._playback_loop,
            name="mixamo-playback",
            daemon=True,
        )
        self._playback_thread.start()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------

    def _init_rendering_tab(self) -> None:
        self.render_tab_state = nerfview.RenderTabState()
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Mixamo Playback")

    def _populate_rendering_tab(self) -> None:
        assert self.render_tab_state is not None
        assert self._rendering_folder is not None
        with self._rendering_folder:
            play_button = self.server.gui.add_button(
                "Play",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Start / stop playback.",
            )

            @play_button.on_click
            def _(_) -> None:
                self._set_playing(not self._playing)

            time_slider = self.server.gui.add_slider(
                "Time",
                min=0,
                max=self._n_frames - 1,
                step=1,
                initial_value=0,
            )

            @time_slider.on_update
            def _(_) -> None:
                with self._state_lock:
                    self._frame = int(time_slider.value)
                self.rerender(None)

            speed_slider = self.server.gui.add_slider(
                "Speed",
                min=0.1,
                max=4.0,
                step=0.1,
                initial_value=1.0,
            )

            @speed_slider.on_update
            def _(_) -> None:
                self._speed = float(speed_slider.value)

            loop_checkbox = self.server.gui.add_checkbox(
                "Loop",
                initial_value=True,
                hint="Restart playback at the last frame.",
            )

            @loop_checkbox.on_update
            def _(_) -> None:
                self._loop = bool(loop_checkbox.value)

            frame_number = self.server.gui.add_number(
                "Frame",
                min=0,
                max=self._n_frames - 1,
                step=1,
                initial_value=0,
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
            "play_button": play_button,
            "time_slider": time_slider,
            "speed_slider": speed_slider,
            "loop_checkbox": loop_checkbox,
            "frame_number": frame_number,
            "reset_button": reset_button,
        }
        self._rendering_tab_handles.update(self._handles)

    # ------------------------------------------------------------------
    # Client connection / camera
    # ------------------------------------------------------------------

    def _connect_client(self, client: viser.ClientHandle) -> None:
        super()._connect_client(client)
        self._set_client_camera(client)

    def _set_client_camera(self, client: viser.ClientHandle) -> None:
        with self.server.atomic():
            client.camera.look_at = self._viewer_center
            client.camera.position = self._viewer_center + np.array(
                [0.0, -self._view_distance, self._view_distance * 0.5]
            )
            client.camera.fov = 0.6

    def _reset_client_cameras(self) -> None:
        for client in self.server.get_clients().values():
            self._set_client_camera(client)

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _set_playing(self, playing: bool) -> None:
        with self._state_lock:
            self._playing = bool(playing)
        if playing:
            with self.server.atomic():
                self._handles["play_button"].icon = viser.Icon.PLAYER_PAUSE
        else:
            with self.server.atomic():
                self._handles["play_button"].icon = viser.Icon.PLAYER_PLAY

    def _playback_loop(self) -> None:
        while not self._stop.is_set():
            if not self._playing:
                self._stop.wait(0.02)
                continue
            interval = 1.0 / max(self._fps * self._speed, 1e-3)
            if self._stop.wait(interval):
                break
            stopped_at_end = False
            with self._state_lock:
                nxt = self._frame + 1
                if nxt >= self._n_frames:
                    if not self._loop:
                        nxt = self._n_frames - 1
                        self._playing = False
                        stopped_at_end = True
                    else:
                        nxt = 0
                self._frame = nxt
            if stopped_at_end:
                with self.server.atomic():
                    self._handles["play_button"].icon = viser.Icon.PLAYER_PLAY
            self._sync_time_gui()
            self.rerender(None)

    def _sync_time_gui(self) -> None:
        with self.server.atomic():
            self._handles["time_slider"].value = self._frame
            self._handles["frame_number"].value = self._frame

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_frame(
        self,
        camera_state: nerfview.CameraState,
        render_tab_state: nerfview.RenderTabState,
    ) -> np.ndarray:
        with self._state_lock:
            frame = self._frame
        # Render at a fixed resolution (--res).  nerfview switches between
        # low (camera moving) and high (static) render resolutions, which
        # would rebuild the offscreen engine on every camera move; the engine
        # is rebuilt only when the browser-window aspect ratio changes.
        res = int(self._res)
        aspect = float(camera_state.aspect)
        if aspect >= 1.0:
            width, height = res, max(int(round(res / aspect)), 1)
        else:
            width, height = max(int(round(res * aspect)), 1), res
        if self._offscreen is None:
            self._offscreen = MixamoFilamentRenderer(self._cache, width, height)
            LOGGER.info(
                "Filament offscreen renderer ready (EGL headless); first "
                "render at %dx%d.",
                width,
                height,
            )
            self._reset_client_cameras()
        return self._offscreen.render(
            frame,
            camera_state.c2w,
            camera_state.fov,
            width,
            height,
        )

    def close(self) -> None:
        self._stop.set()
        if self._playback_thread.is_alive():
            self._playback_thread.join(timeout=2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/sequences/4_pointcloud_exit_gate",
        help="Directory containing mixamo_sequence.npz.",
    )
    parser.add_argument(
        "--mixamo-cache",
        type=Path,
        default=None,
        help=(
            "Explicit npz cache path. Default: "
            "<results-dir>/mixamo_sequence.npz."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8090,
        help="Local viewer port (default 8090).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Playback rate in frames per second (default 30).",
    )
    parser.add_argument(
        "--res",
        type=int,
        default=1024,
        help="Initial viewer render resolution (default 1024).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cache_path = (
        args.mixamo_cache.expanduser().resolve()
        if args.mixamo_cache is not None
        else args.results_dir.expanduser().resolve() / "mixamo_sequence.npz"
    )
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Mixamo cache not found: {cache_path}. Run "
            "scripts/fit_mixamo_sequence.py first."
        )
    cache = MixamoSequenceCache.load(cache_path)
    server = viser.ViserServer(host="127.0.0.1", port=args.port)
    viewer = MixamoViewer(server=server, cache=cache, fps=args.fps, res=args.res)
    url = f"http://127.0.0.1:{args.port}"
    print(f"\nMixamo sequence viewer: {url}  (Ctrl+C to exit)\n")
    print("Mouse drag orbits, wheel zooms; Play/Pause button or Space toggles.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        LOGGER.info("Stopping viewer.")
        viewer.close()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
