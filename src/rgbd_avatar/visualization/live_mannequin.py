"""Standalone Open3D renderer for live application-space Halpe26 poses."""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

from rgbd_avatar.avatar import (
    build_procedural_avatar,
    build_stick_figure_avatar,
)
from rgbd_avatar.live.processor import LivePoseResult
from rgbd_avatar.pose import HALPE26_LINKS
from rgbd_avatar.pose.visualization import draw_pose

from .open3d_avatar import replace_procedural_avatar_mesh


LOGGER = logging.getLogger("live_mannequin_viewer")
RGB_WINDOW_NAME = "Live RGB"
RGB_SKELETON_WINDOW_NAME = "Live RGB + Halpe26 Skeleton"


def build_live_rgb_views(
    result: LivePoseResult,
    *,
    keypoint_threshold: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the raw RGB view and its 2D Halpe26 diagnostic overlay."""

    if keypoint_threshold < 0:
        raise ValueError("keypoint_threshold must be non-negative.")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("RGB view scale must be finite and positive.")
    raw = result.rgb_bgr.copy()
    overlay = (
        draw_pose(
            result.rgb_bgr,
            result.pose2d,
            score_threshold=keypoint_threshold,
        )
        if result.pose2d is not None
        else result.rgb_bgr.copy()
    )
    cv2.putText(
        overlay,
        (
            f"frame={result.frame_number} status={result.status} "
            f"3D={int(np.count_nonzero(result.pose3d_output.usable))}"
        ),
        (10, overlay.shape[0] - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        (
            f"infer={result.timing_ms['inference']:.1f}ms "
            f"recover={result.timing_ms['recovery']:.1f}ms "
            f"total={result.timing_ms['total']:.1f}ms"
        ),
        (10, overlay.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not np.isclose(scale, 1.0):
        size = (
            max(1, int(round(raw.shape[1] * scale))),
            max(1, int(round(raw.shape[0] * scale))),
        )
        raw = cv2.resize(raw, size, interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(overlay, size, interpolation=cv2.INTER_AREA)
    return raw, overlay


def _ground_grid_arrays(
    extent_m: float,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if extent_m <= 0 or spacing_m <= 0:
        raise ValueError("Ground-grid extent and spacing must be positive.")
    coordinates = np.arange(-extent_m, extent_m + 0.5 * spacing_m, spacing_m)
    points: list[list[float]] = []
    lines: list[list[int]] = []
    colors: list[list[float]] = []
    for coordinate in coordinates:
        start = len(points)
        points.extend(
            (
                [-extent_m, float(coordinate), 0.0],
                [extent_m, float(coordinate), 0.0],
                [float(coordinate), -extent_m, 0.0],
                [float(coordinate), extent_m, 0.0],
            )
        )
        lines.extend(([start, start + 1], [start + 2, start + 3]))
        color = [0.60, 0.60, 0.60] if abs(coordinate) > 1e-9 else [0.30, 0.30, 0.30]
        colors.extend((color, color))
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(lines, dtype=np.int32),
        np.asarray(colors, dtype=np.float64),
    )


class LiveMannequinRenderer:
    """Pollable legacy Open3D window that preserves the user's camera view."""

    _STYLES = ("mannequin", "skeleton", "both")

    def __init__(
        self,
        *,
        avatar_model: str = "procedural",
        sphere_resolution: int = 6,
        grid_extent_m: float = 4.0,
        grid_spacing_m: float = 0.25,
        window_width: int = 1280,
        window_height: int = 800,
        show_rgb_views: bool = True,
        rgb_view_scale: float = 0.75,
        keypoint_threshold: float = 0.3,
        camera_forward_application: np.ndarray | None = None,
    ) -> None:
        if avatar_model not in ("procedural", "stickman"):
            raise ValueError("avatar_model must be procedural or stickman.")
        if sphere_resolution < 3:
            raise ValueError("sphere_resolution must be at least 3.")
        if not np.isfinite(rgb_view_scale) or rgb_view_scale <= 0:
            raise ValueError("rgb_view_scale must be finite and positive.")
        if keypoint_threshold < 0:
            raise ValueError("keypoint_threshold must be non-negative.")
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError(
                "Open3D is required for the live mannequin window."
            ) from error
        self.o3d = o3d
        self.avatar_model = avatar_model
        self.sphere_resolution = int(sphere_resolution)
        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self.show_rgb_views = bool(show_rgb_views)
        self.rgb_view_scale = float(rgb_view_scale)
        self.keypoint_threshold = float(keypoint_threshold)
        self.running = False
        self.style_index = 0
        self.focused = False
        forward = np.asarray(
            camera_forward_application
            if camera_forward_application is not None
            else [0.0, 1.0, 0.0],
            dtype=np.float64,
        )
        if forward.shape != (3,) or not np.isfinite(forward).all():
            raise ValueError("camera_forward_application must be finite XYZ.")
        horizontal = forward.copy()
        horizontal[2] = 0.0
        if np.linalg.norm(horizontal) <= 1e-8:
            horizontal = np.array([0.0, 1.0, 0.0])
        self.camera_forward_application = horizontal / np.linalg.norm(horizontal)

        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.avatar_geometry = o3d.geometry.TriangleMesh()
        self.skeleton_geometry = o3d.geometry.LineSet()
        self.avatar_added = False
        self.skeleton_added = False
        grid_points, grid_lines, grid_colors = _ground_grid_arrays(
            grid_extent_m,
            grid_spacing_m,
        )
        self.grid_geometry = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(grid_points),
            lines=o3d.utility.Vector2iVector(grid_lines),
        )
        self.grid_geometry.colors = o3d.utility.Vector3dVector(grid_colors)
        self.axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.30
        )

    @property
    def render_style(self) -> str:
        return self._STYLES[self.style_index]

    def open(self) -> None:
        created = self.visualizer.create_window(
            window_name=(
                "Live RGB-D 3D Stickman"
                if self.avatar_model == "stickman"
                else "Live RGB-D Mannequin"
            ),
            width=self.window_width,
            height=self.window_height,
        )
        if not created:
            raise RuntimeError("Open3D failed to create the live mannequin window.")
        self.visualizer.add_geometry(self.grid_geometry)
        self.visualizer.add_geometry(self.axis_geometry, reset_bounding_box=False)
        self.visualizer.register_key_callback(ord("Q"), self._quit)
        self.visualizer.register_key_callback(ord("M"), self._cycle_style)
        render_option = self.visualizer.get_render_option()
        render_option.background_color = np.array([0.92, 0.92, 0.92])
        render_option.mesh_show_back_face = True
        render_option.line_width = 3.0
        if self.show_rgb_views:
            cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow(RGB_SKELETON_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(RGB_WINDOW_NAME, 20, 40)
            cv2.moveWindow(RGB_SKELETON_WINDOW_NAME, 660, 40)
        self.running = True
        LOGGER.info(
            "Three live views enabled: RGB, RGB skeleton, and 3D %s. "
            "M cycles 3D styles; Q quits.",
            self.avatar_model,
        )

    def _quit(self, _visualizer: Any) -> bool:
        self.running = False
        return False

    def _cycle_style(self, _visualizer: Any) -> bool:
        self.style_index = (self.style_index + 1) % len(self._STYLES)
        self._sync_visibility()
        LOGGER.info("Live render style: %s", self.render_style)
        return False

    def _sync_visibility(self) -> None:
        show_avatar = self.render_style in ("mannequin", "both")
        show_skeleton = self.render_style in ("skeleton", "both")
        has_avatar = self.avatar_geometry.has_vertices()
        has_skeleton = self.skeleton_geometry.has_lines()
        if show_avatar and has_avatar:
            if self.avatar_added:
                self.visualizer.update_geometry(self.avatar_geometry)
            else:
                self.avatar_added = bool(
                    self.visualizer.add_geometry(
                        self.avatar_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif self.avatar_added:
            self.visualizer.remove_geometry(
                self.avatar_geometry,
                reset_bounding_box=False,
            )
            self.avatar_added = False
        if show_skeleton and has_skeleton:
            if self.skeleton_added:
                self.visualizer.update_geometry(self.skeleton_geometry)
            else:
                self.skeleton_added = bool(
                    self.visualizer.add_geometry(
                        self.skeleton_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif self.skeleton_added:
            self.visualizer.remove_geometry(
                self.skeleton_geometry,
                reset_bounding_box=False,
            )
            self.skeleton_added = False

    def _replace_skeleton(self, result: LivePoseResult) -> None:
        usable = result.pose3d_output.usable
        original_indices = np.flatnonzero(
            usable & np.isfinite(result.joints_application_m).all(axis=1)
        )
        remap = {int(original): index for index, original in enumerate(original_indices)}
        lines = [
            (remap[start], remap[end])
            for start, end in HALPE26_LINKS
            if start in remap and end in remap
        ]
        self.skeleton_geometry.points = self.o3d.utility.Vector3dVector(
            result.joints_application_m[original_indices]
        )
        self.skeleton_geometry.lines = self.o3d.utility.Vector2iVector(
            np.asarray(lines, dtype=np.int32).reshape(-1, 2)
        )
        line_colors = np.tile(np.array([[0.08, 0.08, 0.08]]), (len(lines), 1))
        self.skeleton_geometry.colors = self.o3d.utility.Vector3dVector(line_colors)

    def update(self, result: LivePoseResult) -> None:
        avatar_builder = (
            build_stick_figure_avatar
            if self.avatar_model == "stickman"
            else build_procedural_avatar
        )
        avatar = avatar_builder(
            result.joints_application_m,
            result.pose3d_output.usable,
            ground_height_m=0.0,
        )
        replace_procedural_avatar_mesh(
            self.o3d,
            self.avatar_geometry,
            avatar,
            sphere_resolution=self.sphere_resolution,
        )
        self._replace_skeleton(result)
        self._sync_visibility()
        if self.show_rgb_views:
            raw, overlay = build_live_rgb_views(
                result,
                keypoint_threshold=self.keypoint_threshold,
                scale=self.rgb_view_scale,
            )
            cv2.imshow(RGB_WINDOW_NAME, raw)
            cv2.imshow(RGB_SKELETON_WINDOW_NAME, overlay)
        if not self.focused and np.any(result.pose3d_output.usable):
            center = np.nanmedian(
                result.joints_application_m[result.pose3d_output.usable],
                axis=0,
            )
            view = self.visualizer.get_view_control()
            view.set_lookat(center)
            view.set_front(-self.camera_forward_application)
            view.set_up([0.0, 0.0, 1.0])
            view.set_zoom(0.72)
            self.focused = True

    def poll(self) -> bool:
        if not self.running:
            return False
        self.running = bool(self.visualizer.poll_events())
        if self.running:
            self.visualizer.update_renderer()
            if self.show_rgb_views:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    self.running = False
        return self.running

    def close(self) -> None:
        self.running = False
        if self.show_rgb_views:
            for window_name in (RGB_WINDOW_NAME, RGB_SKELETON_WINDOW_NAME):
                try:
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    pass
        self.visualizer.destroy_window()
