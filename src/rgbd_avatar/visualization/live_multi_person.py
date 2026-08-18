"""Local OpenCV/Open3D views for multi-person RGB-D pose experiments."""

from __future__ import annotations

from collections import Counter
import logging
from typing import Any

import cv2
import numpy as np

from rgbd_avatar.avatar import (
    ProceduralAvatarFrame,
    StickFigureConfig,
    build_stick_figure_avatar,
)
from rgbd_avatar.live.multi_person_processor import (
    LocalMultiPersonPoseResult,
    LocalPersonPoseResult,
)
from rgbd_avatar.pose import HALPE26_LINKS

from .live_mannequin import _ground_grid_arrays
from .open3d_avatar import replace_procedural_avatar_mesh


LOGGER = logging.getLogger("local_multi_person_viewer")
RGB_WINDOW_NAME = "Local Multi-Person RGB"
RGB_SKELETON_WINDOW_NAME = "Local Multi-Person RGB + Halpe26"
DETECTION_2D_WINDOW_NAME = "Local Multi-Person 2D Detection"

_TRACK_COLORS_RGB: tuple[tuple[float, float, float], ...] = (
    (0.95, 0.20, 0.18),
    (0.15, 0.55, 0.95),
    (0.20, 0.75, 0.35),
    (0.90, 0.55, 0.10),
    (0.65, 0.30, 0.90),
    (0.05, 0.75, 0.75),
)


def track_color_rgb(track_id: int) -> tuple[float, float, float]:
    if track_id <= 0:
        raise ValueError("track_id must be positive.")
    return _TRACK_COLORS_RGB[(track_id - 1) % len(_TRACK_COLORS_RGB)]


def _track_color_bgr_255(track_id: int) -> tuple[int, int, int]:
    red, green, blue = track_color_rgb(track_id)
    return int(round(blue * 255)), int(round(green * 255)), int(round(red * 255))


def _draw_person_pose(
    canvas: np.ndarray,
    person: LocalPersonPoseResult,
    *,
    score_threshold: float,
) -> None:
    pose = person.pose2d
    if pose is None:
        return
    color = _track_color_bgr_255(person.track_id)
    for start, end in HALPE26_LINKS:
        if (
            pose.scores[start] < score_threshold
            or pose.scores[end] < score_threshold
        ):
            continue
        first = tuple(np.rint(pose.keypoints[start]).astype(int))
        second = tuple(np.rint(pose.keypoints[end]).astype(int))
        cv2.line(canvas, first, second, color, 2, cv2.LINE_AA)
    for index, point in enumerate(pose.keypoints):
        if pose.scores[index] < score_threshold:
            continue
        center = tuple(np.rint(point).astype(int))
        cv2.circle(canvas, center, 4, color, -1, cv2.LINE_AA)

    x1, y1, x2, y2 = np.rint(pose.bbox_xyxy).astype(int)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    usable_count = int(np.count_nonzero(person.pose3d_output.usable))
    fallback_count = int(np.count_nonzero(person.kinematic_fallback))
    missing_count = len(person.joint_sources) - usable_count
    label = (
        f"ID {person.track_id} {person.status} "
        f"3D={usable_count} K={fallback_count} M={missing_count}"
    )
    cv2.putText(
        canvas,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    if missing_count:
        aliases = {
            "low_2d_confidence": "2d",
            "no_depth_candidate": "depth",
            "temporal_depth_jump": "jump",
            "bone_length_violation": "bone",
            "spine_projection_violation": "spine",
            "face_geometry_violation": "face",
            "person_quality_rejected": "reject",
            "low_3d_confidence": "conf",
            "prediction_expired": "expire",
            "skeleton_completion": "complete",
        }
        reason_counts = Counter(
            aliases.get(person.joint_sources[index], person.joint_sources[index])
            for index in np.flatnonzero(~person.pose3d_output.usable)
        )
        diagnostics = "miss " + " ".join(
            f"{reason}:{count}" for reason, count in reason_counts.most_common(3)
        )
        cv2.putText(
            canvas,
            diagnostics,
            (x1, min(canvas.shape[0] - 8, y1 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.44,
            color,
            1,
            cv2.LINE_AA,
        )


def build_local_multi_rgb_views(
    result: LocalMultiPersonPoseResult,
    *,
    keypoint_threshold: float,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return raw RGB and a colored local overlay keyed by stable track IDs."""

    if not 0 <= keypoint_threshold <= 1:
        raise ValueError("keypoint_threshold must be in [0, 1].")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("RGB view scale must be finite and positive.")
    raw = result.rgb_bgr.copy()
    overlay = result.rgb_bgr.copy()
    for person in result.persons:
        _draw_person_pose(
            overlay,
            person,
            score_threshold=keypoint_threshold,
        )
    cv2.putText(
        overlay,
        (
            f"frame={result.frame_number} detected={result.detected_person_count} "
            f"tracks={len(result.persons)} status={result.status} "
            "qreject="
            f"{result.recovery_stats.get('quality_rejected_person_count', 0)} "
            "qinvalid="
            f"{result.recovery_stats.get('quality_invalidated_joint_count', 0)} "
            "kfill="
            f"{result.recovery_stats.get('kinematic_fallback_joint_count', 0)} "
            "complete="
            f"{result.recovery_stats.get('skeleton_completion_joint_count', 0)} "
            "missing="
            f"{result.recovery_stats.get('missing_output_joint_count', 0)}"
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
            f"match={result.timing_ms['matching']:.1f}ms "
            f"quality={result.timing_ms.get('quality', 0.0):.1f}ms "
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


class LocalMultiPerson2DRenderer:
    """Standalone RGB detector/pose view without an Open3D dependency."""

    def __init__(
        self,
        *,
        rgb_view_scale: float = 0.75,
        keypoint_threshold: float = 0.3,
    ) -> None:
        if not np.isfinite(rgb_view_scale) or rgb_view_scale <= 0:
            raise ValueError("rgb_view_scale must be finite and positive.")
        if not 0 <= keypoint_threshold <= 1:
            raise ValueError("keypoint_threshold must be in [0, 1].")
        self.rgb_view_scale = float(rgb_view_scale)
        self.keypoint_threshold = float(keypoint_threshold)
        self.running = False

    def open(self) -> None:
        try:
            cv2.namedWindow(DETECTION_2D_WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
            cv2.moveWindow(DETECTION_2D_WINDOW_NAME, 40, 40)
        except cv2.error as error:
            raise RuntimeError(
                "OpenCV could not create the 2D detection window. Check the "
                "desktop DISPLAY/GUI environment and do not use --headless."
            ) from error
        self.running = True
        LOGGER.info("Standalone 2D detection view enabled. Q quits.")

    def update(self, result: LocalMultiPersonPoseResult) -> None:
        _raw, overlay = build_local_multi_rgb_views(
            result,
            keypoint_threshold=self.keypoint_threshold,
            scale=self.rgb_view_scale,
        )
        cv2.imshow(DETECTION_2D_WINDOW_NAME, overlay)

    def poll(self) -> bool:
        if not self.running:
            return False
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            self.running = False
        return self.running

    def close(self) -> None:
        self.running = False
        try:
            cv2.destroyWindow(DETECTION_2D_WINDOW_NAME)
        except cv2.error:
            pass


def build_local_multi_avatar(
    result: LocalMultiPersonPoseResult,
) -> ProceduralAvatarFrame:
    """Merge colored stick figures for all locally usable tracks."""

    capsules = []
    ellipsoids = []
    for person in result.persons:
        if not person.observed_in_frame:
            continue
        color = track_color_rgb(person.track_id)
        config = StickFigureConfig(
            rod_color=color,
            torso_color=color,
            joint_color=color,
            head_color=tuple(0.72 * channel for channel in color),
        )
        avatar = build_stick_figure_avatar(
            person.joints_application_m,
            person.pose3d_output.usable,
            ground_height_m=0.0,
            config=config,
        )
        capsules.extend(avatar.capsules)
        ellipsoids.extend(avatar.ellipsoids)
    return ProceduralAvatarFrame(
        capsules=tuple(capsules),
        ellipsoids=tuple(ellipsoids),
    )


def build_local_multi_skeleton_arrays(
    result: LocalMultiPersonPoseResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return merged Open3D points, links, and per-track link colors."""

    all_points: list[np.ndarray] = []
    all_lines: list[tuple[int, int]] = []
    all_colors: list[tuple[float, float, float]] = []
    for person in result.persons:
        if not person.observed_in_frame:
            continue
        usable = person.pose3d_output.usable & np.isfinite(
            person.joints_application_m
        ).all(axis=1)
        original_indices = np.flatnonzero(usable)
        remap = {
            int(original): len(all_points) + index
            for index, original in enumerate(original_indices)
        }
        all_points.extend(person.joints_application_m[original_indices])
        color = track_color_rgb(person.track_id)
        for start, end in HALPE26_LINKS:
            if start in remap and end in remap:
                all_lines.append((remap[start], remap[end]))
                all_colors.append(color)
    return (
        np.asarray(all_points, dtype=np.float64).reshape(-1, 3),
        np.asarray(all_lines, dtype=np.int32).reshape(-1, 2),
        np.asarray(all_colors, dtype=np.float64).reshape(-1, 3),
    )


class LocalMultiPersonRenderer:
    """Pollable local-only renderer for several colored stick figures."""

    _STYLES = ("stickmen", "skeletons", "both")

    def __init__(
        self,
        *,
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
        if sphere_resolution < 3:
            raise ValueError("sphere_resolution must be at least 3.")
        if not np.isfinite(rgb_view_scale) or rgb_view_scale <= 0:
            raise ValueError("rgb_view_scale must be finite and positive.")
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError(
                "Open3D is required for the local multi-person view."
            ) from error
        self.o3d = o3d
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
            window_name="Local Multi-Person RGB-D Stickmen",
            width=self.window_width,
            height=self.window_height,
        )
        if not created:
            raise RuntimeError("Open3D failed to create the local multi-person window.")
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
            # Keep the diagnostic skeleton view on the primary screen.  The
            # previous right-hand position could place it outside narrower
            # desktops, making the run look like it only opened two views.
            cv2.moveWindow(RGB_SKELETON_WINDOW_NAME, 20, 40)
            cv2.moveWindow(RGB_WINDOW_NAME, 660, 40)
        self.running = True
        LOGGER.info(
            "Local multi-person views enabled: 3D, raw RGB, and 2D Halpe26. "
            "M cycles styles; Q quits."
        )

    def _quit(self, _visualizer: Any) -> bool:
        self.running = False
        return False

    def _cycle_style(self, _visualizer: Any) -> bool:
        self.style_index = (self.style_index + 1) % len(self._STYLES)
        self._sync_visibility()
        LOGGER.info("Local multi-person render style: %s", self.render_style)
        return False

    def _sync_visibility(self) -> None:
        show_avatar = self.render_style in ("stickmen", "both")
        show_skeleton = self.render_style in ("skeletons", "both")
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

    def update(self, result: LocalMultiPersonPoseResult) -> None:
        avatar = build_local_multi_avatar(result)
        replace_procedural_avatar_mesh(
            self.o3d,
            self.avatar_geometry,
            avatar,
            sphere_resolution=self.sphere_resolution,
        )
        points, lines, colors = build_local_multi_skeleton_arrays(result)
        self.skeleton_geometry.points = self.o3d.utility.Vector3dVector(points)
        self.skeleton_geometry.lines = self.o3d.utility.Vector2iVector(lines)
        self.skeleton_geometry.colors = self.o3d.utility.Vector3dVector(colors)
        self._sync_visibility()
        if self.show_rgb_views:
            raw, overlay = build_local_multi_rgb_views(
                result,
                keypoint_threshold=self.keypoint_threshold,
                scale=self.rgb_view_scale,
            )
            cv2.imshow(RGB_WINDOW_NAME, raw)
            cv2.imshow(RGB_SKELETON_WINDOW_NAME, overlay)
        all_usable_points = [
            person.joints_application_m[person.pose3d_output.usable]
            for person in result.persons
            if np.any(person.pose3d_output.usable)
        ]
        if not self.focused and all_usable_points:
            center = np.nanmedian(np.vstack(all_usable_points), axis=0)
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
