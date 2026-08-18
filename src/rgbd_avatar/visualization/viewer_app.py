#!/usr/bin/env python3
"""Replay a metric 3D Halpe26 skeleton, optionally with its RGB-D cloud."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from rgbd_avatar.avatar import (
    MixamoSequenceCache,
    ProceduralAvatarFrame,
    SMPLSequenceCache,
    build_procedural_avatar,
    build_stick_figure_avatar,
    sha256_file,
)
from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.data import load_hand_records
from rgbd_avatar.depth import depth_to_organized_point_cloud, load_depth_m
from rgbd_avatar.io import load_camera_config
from rgbd_avatar.visualization import (
    CloudDisplayArrays,
    GroundGridDisplayArrays,
    PoseDisplayData,
    SkeletonDisplayArrays,
    build_cloud_display_arrays,
    build_ground_grid_display_arrays,
    build_skeleton_display_arrays,
    empty_cloud_display_arrays,
    load_pose_records,
    parse_pose_layer,
    playback_delay_s,
    propagate_segment_bboxes,
    resolve_frame_sources,
    transform_camera_points,
)
from rgbd_avatar.visualization.contracts import (
    load_ground_alignment,
    verify_manifest_camera,
)
from rgbd_avatar.visualization.hand3d import build_hand_skeleton_display_arrays
from rgbd_avatar.visualization.open3d_avatar import (
    replace_procedural_avatar_mesh,
    rotation_from_local_z,
)


LOGGER = logging.getLogger("view_pose3d_sequence")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
HELP_TEXT = """
Open3D controls
  Space       play / pause
  Left or A   previous frame (and pause)
  Right or D  next frame (and pause)
  R / T / C   raw / temporal / constrained skeleton
  M           cycle skeleton / avatar mesh / both
  G           original RGB point-cloud colors
  X / Y / Z   X / Y / Z coordinate pseudo-colors
  O           Open3D default point-cloud color
  N           point-normal color (only when normals exist)
  0..9        disabled; reset to RGB to avoid Open3D key conflicts
  B           cycle bbox/full/no point cloud
  K           toggle metric ground grid
  L           toggle looping
  F           restore the front camera view
  H           print this help
  Q           quit

Mouse: left-drag rotates, wheel zooms, Shift+left-drag pans.
Colors: raw red, observed green, predicted orange, corrected magenta.
Hands: left cyan, right yellow, rejected/degenerate detection red.
Avatar: procedural, classic stick figure, SMPL Neutral, or textured Mixamo.
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/sequences/4_pointcloud_exit_gate",
        help="Directory containing poses.jsonl.",
    )
    parser.add_argument(
        "--poses-jsonl",
        type=Path,
        default=None,
        help="Explicit JSONL path; overrides --results-dir.",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PROJECT_ROOT / "configs/camera.yaml",
    )
    parser.add_argument(
        "--ground-plane",
        type=Path,
        default=None,
        help=(
            "Ground calibration JSON. Default: ground_plane.json beside "
            "poses.jsonl when present."
        ),
    )
    parser.add_argument(
        "--no-ground-alignment",
        action="store_true",
        help="Ignore saved ground calibration and use optical camera axes.",
    )
    parser.add_argument(
        "--sequence-dir",
        type=Path,
        default=None,
        help="Override RGB/depth source paths when a sequence was moved.",
    )
    parser.add_argument(
        "--allow-camera-config-mismatch",
        action="store_true",
        help="Continue even when manifest calibration differs from YAML.",
    )
    parser.add_argument(
        "--pose-layer",
        choices=("raw", "temporal", "constrained"),
        default=None,
        help=(
            "Initial layer. Default: raw, or constrained with "
            "--skeleton-only."
        ),
    )
    parser.add_argument(
        "--cloud-scope",
        choices=("bbox", "full", "none"),
        default=None,
        help=(
            "Point cloud mode. Default: bbox, or none with "
            "--skeleton-only."
        ),
    )
    parser.add_argument(
        "--skeleton-only",
        action="store_true",
        help=(
            "Shorthand for a constrained skeleton in an empty metric 3D "
            "space with a ground grid."
        ),
    )
    parser.add_argument(
        "--mannequin",
        action="store_true",
        help=(
            "Shorthand for a constrained procedural mannequin in an empty "
            "metric 3D space with a ground grid."
        ),
    )
    parser.add_argument(
        "--smpl",
        action="store_true",
        help=(
            "Shorthand for --mannequin --avatar-model smpl using the "
            "cached fitted sequence."
        ),
    )
    parser.add_argument(
        "--mixamo",
        action="store_true",
        help=(
            "Shorthand for --mannequin --avatar-model mixamo using the "
            "cached direct-IK sequence."
        ),
    )
    parser.add_argument(
        "--stickman",
        action="store_true",
        help=(
            "Shorthand for --mannequin --avatar-model stickman in an "
            "empty metric 3D space."
        ),
    )
    parser.add_argument(
        "--avatar-model",
        choices=("procedural", "stickman", "smpl", "mixamo"),
        default="procedural",
        help="Geometry backend used by mannequin render styles.",
    )
    parser.add_argument(
        "--smpl-cache",
        type=Path,
        default=None,
        help="Default: smpl_sequence.npz beside poses.jsonl.",
    )
    parser.add_argument(
        "--mixamo-cache",
        type=Path,
        default=None,
        help="Default: mixamo_sequence.npz beside poses.jsonl.",
    )
    parser.add_argument(
        "--smpl-smooth-iterations",
        type=int,
        default=2,
        help=(
            "Display-only non-shrinking Taubin smoothing passes for SMPL; "
            "0 disables smoothing (default: 2)."
        ),
    )
    parser.add_argument(
        "--hands-jsonl",
        type=Path,
        default=None,
        help="Default: hands.jsonl beside poses.jsonl when present.",
    )
    parser.add_argument(
        "--show-hands",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Display Hand21 points; defaults on when hands.jsonl exists.",
    )
    parser.add_argument(
        "--render-style",
        choices=("skeleton", "mannequin", "both"),
        default=None,
        help=(
            "Initial pose rendering. Default: skeleton, or mannequin with "
            "--mannequin."
        ),
    )
    parser.add_argument("--point-stride", type=int, default=3)
    parser.add_argument("--bbox-margin-px", type=int, default=30)
    parser.add_argument(
        "--bbox-hold-s",
        type=float,
        default=1.1,
        help="Maximum time to reuse a nearby detection box.",
    )
    parser.add_argument(
        "--timing",
        choices=("source", "fixed"),
        default="source",
        help="Use capture timestamps or a fixed playback rate.",
    )
    parser.add_argument("--playback-speed", type=float, default=1.0)
    parser.add_argument("--fixed-fps", type=float, default=2.0)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional prefix limit, useful for a smoke test.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Build all arrays without opening a GUI window.",
    )
    parser.add_argument("--joint-radius-m", type=float, default=0.025)
    parser.add_argument("--hand-joint-radius-m", type=float, default=0.009)
    parser.add_argument("--sphere-resolution", type=int, default=6)
    parser.add_argument(
        "--ground-grid",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Show a metric reference grid; defaults on in skeleton-only mode.",
    )
    parser.add_argument("--grid-spacing-m", type=float, default=0.25)
    parser.add_argument("--grid-margin-m", type=float, default=0.50)
    parser.add_argument("--window-width", type=int, default=1280)
    parser.add_argument("--window-height", type=int, default=800)
    return parser.parse_args()


def load_hand_pose_records(
    path: Path,
    body_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load a Hand21 JSONL cache aligned to a body-pose prefix."""

    return load_hand_records(path, body_records)


@dataclass(frozen=True)
class FrameScene:
    record: dict[str, Any]
    pose: PoseDisplayData
    skeleton: SkeletonDisplayArrays
    hand_skeleton: SkeletonDisplayArrays
    rejected_hands: tuple[str, ...]
    avatar: ProceduralAvatarFrame
    mesh_vertices_display_m: np.ndarray
    cloud: CloudDisplayArrays
    rgb_path: Path | None
    depth_path: Path | None


def _front_view_parameters(
    scene: FrameScene,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a stable chest-level, slightly elevated front camera pose."""

    if len(scene.mesh_vertices_display_m):
        points = scene.mesh_vertices_display_m
    elif len(scene.skeleton.points):
        points = scene.skeleton.points
    elif len(scene.cloud.points):
        points = scene.cloud.points
    else:
        points = np.empty((0, 3), dtype=np.float64)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points):
        lower = np.quantile(points, 0.02, axis=0)
        upper = np.quantile(points, 0.98, axis=0)
        center = np.median(points, axis=0)
        # The coordinate-wise median is biased toward the numerous leg/foot
        # joints. Aim at the upper torso instead to avoid a persistent
        # low-angle view.
        center[2] = lower[2] + 0.62 * (upper[2] - lower[2])
    else:
        center = np.array([0.0, 2.0, 0.9], dtype=np.float64)

    elevation_rad = np.deg2rad(6.0)
    front = np.array(
        [0.0, -np.cos(elevation_rad), np.sin(elevation_rad)],
        dtype=np.float64,
    )
    up = np.array(
        [0.0, np.sin(elevation_rad), np.cos(elevation_rad)],
        dtype=np.float64,
    )
    return center, front, up


class FrameSceneLoader:
    """Load RGB-D geometry lazily and retain a small seek cache."""

    def __init__(
        self,
        *,
        records: list[dict[str, Any]],
        jsonl_path: Path,
        camera: dict[str, Any],
        intrinsics: CameraIntrinsics,
        sequence_dir: Path | None,
        point_stride: int,
        bbox_margin_px: int,
        bbox_hold_s: float,
        camera_to_display_transform: np.ndarray | None = None,
        smpl_cache: SMPLSequenceCache | None = None,
        mixamo_cache: MixamoSequenceCache | None = None,
        hand_records: list[dict[str, Any]] | None = None,
        avatar_model: str = "procedural",
        cache_size: int = 4,
    ) -> None:
        self.records = records
        self.jsonl_path = jsonl_path
        self.camera = camera
        self.intrinsics = intrinsics
        self.sequence_dir = sequence_dir
        self.point_stride = point_stride
        self.bbox_margin_px = bbox_margin_px
        self.camera_to_display_transform = camera_to_display_transform
        if smpl_cache is not None and mixamo_cache is not None:
            raise ValueError("Only one skinned avatar cache may be active.")
        self.smpl_cache = smpl_cache
        self.mixamo_cache = mixamo_cache
        self.mesh_cache = smpl_cache if smpl_cache is not None else mixamo_cache
        self.hand_records = hand_records
        self.avatar_model = avatar_model
        self.cache_size = cache_size
        if self.mesh_cache is not None:
            record_indices = np.asarray(
                [record["frame_index"] for record in records],
                dtype=np.int64,
            )
            if len(self.mesh_cache.frame_indices) < len(record_indices) or not np.array_equal(
                self.mesh_cache.frame_indices[: len(record_indices)],
                record_indices,
            ):
                raise ValueError(
                    "SMPL cache frame indices do not match poses.jsonl prefix."
                )
        if hand_records is not None and len(hand_records) != len(records):
            raise ValueError("Hand record count does not match body records.")
        self.bboxes = propagate_segment_bboxes(
            records,
            max_age_s=bbox_hold_s,
        )
        self._cloud_cache: OrderedDict[
            tuple[int, str],
            tuple[CloudDisplayArrays, Path | None, Path | None],
        ] = OrderedDict()

    def _load_cloud(
        self,
        index: int,
        scope: str,
    ) -> tuple[CloudDisplayArrays, Path | None, Path | None]:
        cache_key = (index, scope)
        cached = self._cloud_cache.get(cache_key)
        if cached is not None:
            self._cloud_cache.move_to_end(cache_key)
            return cached

        if scope == "none":
            return empty_cloud_display_arrays(), None, None

        record = self.records[index]
        rgb_path, depth_path = resolve_frame_sources(
            record,
            self.jsonl_path,
            self.sequence_dir,
        )
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"OpenCV failed to read RGB image: {rgb_path}")
        rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        depth_m = load_depth_m(
            depth_path,
            float(self.camera["depth_scale"]),
        )
        expected_shape = (self.intrinsics.height, self.intrinsics.width)
        if rgb.shape[:2] != expected_shape or depth_m.shape != expected_shape:
            raise ValueError(
                "Aligned RGB/depth shape mismatch at frame "
                f"{record['frame_index']}: RGB={rgb.shape[:2]}, "
                f"depth={depth_m.shape}, expected={expected_shape}."
            )

        organized_points = depth_to_organized_point_cloud(
            depth_m,
            self.intrinsics,
            min_depth_m=float(self.camera["min_depth_m"]),
            max_depth_m=float(self.camera["max_depth_m"]),
        )
        cloud = build_cloud_display_arrays(
            organized_points,
            rgb,
            stride=self.point_stride,
            scope=scope,
            bbox_xyxy=self.bboxes[index],
            bbox_margin_px=self.bbox_margin_px,
            camera_to_display_transform=(
                self.camera_to_display_transform
            ),
        )
        result = (cloud, rgb_path, depth_path)
        self._cloud_cache[cache_key] = result
        self._cloud_cache.move_to_end(cache_key)
        while len(self._cloud_cache) > self.cache_size:
            self._cloud_cache.popitem(last=False)
        return result

    def build(self, index: int, layer: str, scope: str) -> FrameScene:
        record = self.records[index]
        pose = parse_pose_layer(record, layer)
        skeleton = build_skeleton_display_arrays(
            pose,
            camera_to_display_transform=(
                self.camera_to_display_transform
            ),
        )
        hand_skeleton, rejected_hands = build_hand_skeleton_display_arrays(
            self.hand_records[index] if self.hand_records is not None else None,
            camera_to_display_transform=self.camera_to_display_transform,
        )
        avatar_joints = np.full_like(pose.joints_camera_m, np.nan)
        avatar_usable = pose.usable & np.isfinite(
            pose.joints_camera_m
        ).all(axis=1)
        avatar_joints[avatar_usable] = transform_camera_points(
            pose.joints_camera_m[avatar_usable],
            self.camera_to_display_transform,
        )
        avatar_builder = (
            build_stick_figure_avatar
            if self.avatar_model == "stickman"
            else build_procedural_avatar
        )
        avatar = avatar_builder(
            avatar_joints,
            avatar_usable,
            ground_height_m=(
                0.0
                if self.camera_to_display_transform is not None
                else None
            ),
        )
        mesh_vertices = np.empty((0, 3), dtype=np.float64)
        if self.mesh_cache is not None and self.mesh_cache.present[index]:
            mesh_vertices = self.mesh_cache.vertices_display_m[
                index
            ].astype(np.float64, copy=False)
        cloud, rgb_path, depth_path = self._load_cloud(index, scope)
        return FrameScene(
            record=record,
            pose=pose,
            skeleton=skeleton,
            hand_skeleton=hand_skeleton,
            rejected_hands=rejected_hands,
            avatar=avatar,
            mesh_vertices_display_m=mesh_vertices,
            cloud=cloud,
            rgb_path=rgb_path,
            depth_path=depth_path,
        )


def validate_scenes(
    loader: FrameSceneLoader,
    *,
    layer: str,
    scope: str,
    avatar_model: str = "procedural",
) -> None:
    point_counts: list[int] = []
    joint_counts: list[int] = []
    line_counts: list[int] = []
    hand_joint_counts: list[int] = []
    hand_line_counts: list[int] = []
    avatar_counts: list[int] = []
    fallback_count = 0
    started = time.perf_counter()
    for index in range(len(loader.records)):
        scene = loader.build(index, layer, scope)
        point_counts.append(len(scene.cloud.points))
        joint_counts.append(len(scene.skeleton.points))
        line_counts.append(len(scene.skeleton.lines))
        hand_joint_counts.append(len(scene.hand_skeleton.points))
        hand_line_counts.append(len(scene.hand_skeleton.lines))
        avatar_counts.append(
            int(len(scene.mesh_vertices_display_m) > 0)
            if avatar_model in ("smpl", "mixamo")
            else scene.avatar.primitive_count
        )
        fallback_count += int(scene.pose.resolved_layer != layer)
        LOGGER.info(
            "[%d/%d] frame=%s status=%s layer=%s points=%d joints=%d "
            "links=%d hand_joints=%d hand_links=%d rejected_hands=%s "
            "avatar=%s avatar_parts=%d",
            index + 1,
            len(loader.records),
            scene.record["frame_index"],
            scene.record.get("status"),
            scene.pose.resolved_layer,
            point_counts[-1],
            joint_counts[-1],
            line_counts[-1],
            hand_joint_counts[-1],
            hand_line_counts[-1],
            ",".join(scene.rejected_hands) or "none",
            avatar_model,
            avatar_counts[-1],
        )
    elapsed_s = time.perf_counter() - started
    LOGGER.info(
        "Validation complete: frames=%d point_count=[%d,%d] "
        "joint_count=[%d,%d] link_count=[%d,%d] "
        "hand_joint_count=[%d,%d] hand_link_count=[%d,%d] "
        "avatar=%s avatar_parts=[%d,%d] fallbacks=%d time=%.2fs",
        len(loader.records),
        min(point_counts),
        max(point_counts),
        min(joint_counts),
        max(joint_counts),
        min(line_counts),
        max(line_counts),
        min(hand_joint_counts),
        max(hand_joint_counts),
        min(hand_line_counts),
        max(hand_line_counts),
        avatar_model,
        min(avatar_counts),
        max(avatar_counts),
        fallback_count,
        elapsed_s,
    )


class Pose3DSequenceViewer:
    """Legacy Open3D viewer with source-timestamp playback."""

    def __init__(
        self,
        *,
        loader: FrameSceneLoader,
        layer: str,
        cloud_scope: str,
        render_style: str,
        avatar_model: str,
        timing: str,
        playback_speed: float,
        fixed_fps: float,
        loop: bool,
        start_paused: bool,
        joint_radius_m: float,
        hand_joint_radius_m: float,
        show_hands: bool,
        sphere_resolution: int,
        smpl_smooth_iterations: int,
        show_ground_grid: bool,
        grid_spacing_m: float,
        grid_margin_m: float,
        window_width: int,
        window_height: int,
    ) -> None:
        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError(
                "Open3D is required for interactive playback."
            ) from error

        self.o3d = o3d
        self.loader = loader
        self.records = loader.records
        self.layer = layer
        self.cloud_scope = cloud_scope
        self.render_style = render_style
        self.avatar_model = avatar_model
        self.timing = timing
        self.playback_speed = playback_speed
        self.fixed_fps = fixed_fps
        self.loop = loop
        self.paused = start_paused
        self.joint_radius_m = joint_radius_m
        self.hand_joint_radius_m = hand_joint_radius_m
        self.show_hands = show_hands
        self.sphere_resolution = sphere_resolution
        self.smpl_smooth_iterations = int(smpl_smooth_iterations)
        self.show_ground_grid = show_ground_grid
        self.window_width = window_width
        self.window_height = window_height

        self.index = 0
        self.running = False
        self.next_due = float("inf")
        self.scene: FrameScene | None = None
        self.has_focused_on_skeleton = False
        self.visualizer = o3d.visualization.VisualizerWithKeyCallback()
        self.cloud_geometry = o3d.geometry.PointCloud()
        self.line_geometry = o3d.geometry.LineSet()
        self.joint_geometry = o3d.geometry.TriangleMesh()
        self.hand_line_geometry = o3d.geometry.LineSet()
        self.hand_joint_geometry = o3d.geometry.TriangleMesh()
        self.avatar_geometry = o3d.geometry.TriangleMesh()
        self.avatar_texture = None
        if loader.mixamo_cache is not None:
            encoded = loader.mixamo_cache.diffuse_png
            image_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise ValueError("Could not decode Mixamo diffuse texture.")
            self.avatar_texture = o3d.geometry.Image(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            )
        self.grid_arrays: GroundGridDisplayArrays = (
            build_ground_grid_display_arrays(
                self.records,
                layer=layer,
                spacing_m=grid_spacing_m,
                margin_m=grid_margin_m,
                camera_to_display_transform=(
                    loader.camera_to_display_transform
                ),
                floor_height_m=(
                    0.0
                    if loader.camera_to_display_transform is not None
                    else None
                ),
            )
        )
        self.grid_geometry = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(self.grid_arrays.points),
            lines=o3d.utility.Vector2iVector(self.grid_arrays.lines),
        )
        self.grid_geometry.colors = o3d.utility.Vector3dVector(
            self.grid_arrays.colors
        )
        self.cloud_geometry_added = False
        self.line_geometry_added = False
        self.joint_geometry_added = False
        self.hand_line_geometry_added = False
        self.hand_joint_geometry_added = False
        self.avatar_geometry_added = False
        self.grid_geometry_added = False
        self.axis_geometry_added = False
        self.axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.25
        )
        self.axis_geometry.translate(
            [
                self.grid_arrays.center_xy_m[0],
                self.grid_arrays.center_xy_m[1],
                self.grid_arrays.floor_height_m,
            ]
        )

    def _replace_joint_mesh(
        self,
        skeleton: SkeletonDisplayArrays,
        geometry: Any,
        radius_m: float,
    ) -> None:
        geometry.clear()
        for point, color in zip(
            skeleton.points,
            skeleton.point_colors,
        ):
            sphere = self.o3d.geometry.TriangleMesh.create_sphere(
                radius=radius_m,
                resolution=self.sphere_resolution,
            )
            sphere.translate(point)
            sphere.paint_uniform_color(color)
            geometry += sphere
        if geometry.has_vertices():
            geometry.compute_vertex_normals()

    @staticmethod
    def _rotation_from_local_z(direction: np.ndarray) -> np.ndarray:
        """Build a proper rotation whose local Z axis follows direction."""

        return rotation_from_local_z(direction)

    def _replace_procedural_avatar_mesh(
        self,
        avatar: ProceduralAvatarFrame,
    ) -> None:
        replace_procedural_avatar_mesh(
            self.o3d,
            self.avatar_geometry,
            avatar,
            sphere_resolution=self.sphere_resolution,
        )

    def _replace_avatar_mesh(self, scene: FrameScene) -> None:
        if getattr(self, "avatar_model", "procedural") not in (
            "smpl", "mixamo"
        ):
            self._replace_procedural_avatar_mesh(scene.avatar)
            return

        self.avatar_geometry.clear()
        vertices = scene.mesh_vertices_display_m
        cache = self.loader.mesh_cache
        if len(vertices) == 0 or cache is None:
            return
        self.avatar_geometry.vertices = self.o3d.utility.Vector3dVector(
            vertices
        )
        self.avatar_geometry.triangles = self.o3d.utility.Vector3iVector(
            cache.faces
        )
        if self.avatar_model == "smpl" and self.smpl_smooth_iterations > 0:
            smoothed = self.avatar_geometry.filter_smooth_taubin(
                number_of_iterations=self.smpl_smooth_iterations,
            )
            self.avatar_geometry.vertices = smoothed.vertices
            self.avatar_geometry.triangles = smoothed.triangles
        if self.avatar_model == "mixamo":
            assert isinstance(cache, MixamoSequenceCache)
            self.avatar_geometry.triangle_uvs = (
                self.o3d.utility.Vector2dVector(
                    cache.triangle_uvs.reshape(-1, 2)
                )
            )
            if self.avatar_texture is not None:
                self.avatar_geometry.textures = [self.avatar_texture]
        else:
            self.avatar_geometry.paint_uniform_color((0.72, 0.74, 0.76))
        self.avatar_geometry.compute_vertex_normals()

    def _assign_scene(self, scene: FrameScene) -> None:
        self.cloud_geometry.points = self.o3d.utility.Vector3dVector(
            scene.cloud.points
        )
        self.cloud_geometry.colors = self.o3d.utility.Vector3dVector(
            scene.cloud.colors
        )
        self.line_geometry.points = self.o3d.utility.Vector3dVector(
            scene.skeleton.points
        )
        self.line_geometry.lines = self.o3d.utility.Vector2iVector(
            scene.skeleton.lines
        )
        self.line_geometry.colors = self.o3d.utility.Vector3dVector(
            scene.skeleton.line_colors
        )
        self._replace_joint_mesh(
            scene.skeleton,
            self.joint_geometry,
            self.joint_radius_m,
        )
        self.hand_line_geometry.points = self.o3d.utility.Vector3dVector(
            scene.hand_skeleton.points
        )
        self.hand_line_geometry.lines = self.o3d.utility.Vector2iVector(
            scene.hand_skeleton.lines
        )
        self.hand_line_geometry.colors = self.o3d.utility.Vector3dVector(
            scene.hand_skeleton.line_colors
        )
        self._replace_joint_mesh(
            scene.hand_skeleton,
            self.hand_joint_geometry,
            self.hand_joint_radius_m,
        )
        self._replace_avatar_mesh(scene)

    def _sync_cloud_geometry(self) -> None:
        has_cloud = (
            self.scene is not None
            and len(self.scene.cloud.points) > 0
        )
        if has_cloud:
            if self.cloud_geometry_added:
                self.visualizer.update_geometry(self.cloud_geometry)
            else:
                self.cloud_geometry_added = bool(
                    self.visualizer.add_geometry(
                        self.cloud_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif self.cloud_geometry_added:
            self.visualizer.remove_geometry(
                self.cloud_geometry,
                reset_bounding_box=False,
            )
            self.cloud_geometry_added = False

    def _sync_grid_geometry(self) -> None:
        if self.show_ground_grid and len(self.grid_arrays.lines) > 0:
            if self.grid_geometry_added:
                self.visualizer.update_geometry(self.grid_geometry)
            else:
                self.grid_geometry_added = bool(
                    self.visualizer.add_geometry(
                        self.grid_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif self.grid_geometry_added:
            self.visualizer.remove_geometry(
                self.grid_geometry,
                reset_bounding_box=False,
            )
            self.grid_geometry_added = False

    def _sync_dynamic_geometries(self) -> None:
        show_skeleton = self.render_style in ("skeleton", "both")
        show_avatar = self.render_style in ("mannequin", "both")
        if (
            show_skeleton
            and self.scene is not None
            and len(self.scene.skeleton.lines) > 0
        ):
            if self.line_geometry_added:
                self.visualizer.update_geometry(self.line_geometry)
            else:
                self.line_geometry_added = bool(
                    self.visualizer.add_geometry(
                        self.line_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif self.line_geometry_added:
            self.visualizer.remove_geometry(
                self.line_geometry,
                reset_bounding_box=False,
            )
            self.line_geometry_added = False

        if (
            show_skeleton
            and self.scene is not None
            and len(self.scene.skeleton.points) > 0
        ):
            if self.joint_geometry_added:
                self.visualizer.update_geometry(self.joint_geometry)
            else:
                self.joint_geometry_added = bool(
                    self.visualizer.add_geometry(
                        self.joint_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif self.joint_geometry_added:
            self.visualizer.remove_geometry(
                self.joint_geometry,
                reset_bounding_box=False,
            )
            self.joint_geometry_added = False

        show_hand_layer = show_skeleton and getattr(
            self,
            "show_hands",
            False,
        )
        if (
            show_hand_layer
            and self.scene is not None
            and len(self.scene.hand_skeleton.lines) > 0
        ):
            if self.hand_line_geometry_added:
                self.visualizer.update_geometry(self.hand_line_geometry)
            else:
                self.hand_line_geometry_added = bool(
                    self.visualizer.add_geometry(
                        self.hand_line_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif getattr(self, "hand_line_geometry_added", False):
            self.visualizer.remove_geometry(
                self.hand_line_geometry,
                reset_bounding_box=False,
            )
            self.hand_line_geometry_added = False

        if (
            show_hand_layer
            and self.scene is not None
            and len(self.scene.hand_skeleton.points) > 0
        ):
            if self.hand_joint_geometry_added:
                self.visualizer.update_geometry(self.hand_joint_geometry)
            else:
                self.hand_joint_geometry_added = bool(
                    self.visualizer.add_geometry(
                        self.hand_joint_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif getattr(self, "hand_joint_geometry_added", False):
            self.visualizer.remove_geometry(
                self.hand_joint_geometry,
                reset_bounding_box=False,
            )
            self.hand_joint_geometry_added = False

        if self.scene is None:
            avatar_parts = 0
        elif getattr(self, "avatar_model", "procedural") in ("smpl", "mixamo"):
            avatar_parts = int(len(self.scene.mesh_vertices_display_m) > 0)
        else:
            avatar_parts = self.scene.avatar.primitive_count
        if show_avatar and avatar_parts > 0:
            if self.avatar_geometry_added:
                self.visualizer.update_geometry(self.avatar_geometry)
            else:
                self.avatar_geometry_added = bool(
                    self.visualizer.add_geometry(
                        self.avatar_geometry,
                        reset_bounding_box=False,
                    )
                )
        elif self.avatar_geometry_added:
            self.visualizer.remove_geometry(
                self.avatar_geometry,
                reset_bounding_box=False,
            )
            self.avatar_geometry_added = False

    def _log_scene(self, scene: FrameScene) -> None:
        LOGGER.info(
            "frame=%s/%d timestamp=%s status=%s layer=%s "
            "observed=%d predicted=%d corrected=%d style=%s "
            "hand_joints=%d hand_links=%d rejected_hands=%s "
            "avatar=%s avatar_parts=%d cloud=%s/%d",
            scene.record["frame_index"],
            len(self.records) - 1,
            scene.record["timestamp_raw"],
            scene.record.get("status"),
            scene.pose.resolved_layer,
            int(np.count_nonzero(scene.pose.observed)),
            int(np.count_nonzero(scene.pose.predicted)),
            int(np.count_nonzero(scene.pose.corrected)),
            self.render_style,
            len(scene.hand_skeleton.points),
            len(scene.hand_skeleton.lines),
            ",".join(scene.rejected_hands) or "none",
            getattr(self, "avatar_model", "procedural"),
            (
                int(len(scene.mesh_vertices_display_m) > 0)
                if getattr(self, "avatar_model", "procedural") in ("smpl", "mixamo")
                else scene.avatar.primitive_count
            ),
            scene.cloud.resolved_scope,
            len(scene.cloud.points),
        )

    def _render_current(self) -> None:
        scene = self.loader.build(
            self.index,
            self.layer,
            self.cloud_scope,
        )
        self._assign_scene(scene)
        self.scene = scene
        self._sync_cloud_geometry()
        self._sync_grid_geometry()
        self._sync_dynamic_geometries()
        if (
            len(scene.skeleton.points) > 0
            and not self.has_focused_on_skeleton
        ):
            self._focus_current()
            self.has_focused_on_skeleton = True
        self.visualizer.update_renderer()
        self._log_scene(scene)

    def _delay_to_following(self) -> float:
        if self.index + 1 < len(self.records):
            return playback_delay_s(
                self.records[self.index],
                self.records[self.index + 1],
                timing=self.timing,
                playback_speed=self.playback_speed,
                fixed_fps=self.fixed_fps,
            )
        return 1.0 / (self.fixed_fps * self.playback_speed)

    def _schedule(self, *, reanchor: bool = False) -> None:
        delay_s = self._delay_to_following()
        if reanchor or not np.isfinite(self.next_due):
            self.next_due = time.monotonic() + delay_s
        else:
            # Advance from the prior deadline so RGB-D loading and rendering
            # time do not accumulate into source playback timing.
            self.next_due += delay_s

    def _advance(self) -> None:
        if self.index + 1 < len(self.records):
            self.index += 1
        elif self.loop:
            self.index = 0
        else:
            self.paused = True
            self.next_due = float("inf")
            LOGGER.info("Reached the final frame; playback paused.")
            return
        self._render_current()
        self._schedule()

    def _seek(self, delta: int) -> None:
        self.paused = True
        self.next_due = float("inf")
        self.index = int(
            np.clip(self.index + delta, 0, len(self.records) - 1)
        )
        self._render_current()

    def _switch_layer(self, layer: str) -> None:
        self.layer = layer
        if getattr(self, "avatar_model", "procedural") in ("smpl", "mixamo"):
            fitted_layer = self.loader.mesh_cache.metadata.get("pose_layer")
            if layer != fitted_layer:
                LOGGER.info(
                    "Skeleton overlay switched to %s; SMPL mesh remains the "
                    "cached %s fit.",
                    layer,
                    fitted_layer,
                )
        self._render_current()
        if not self.paused:
            self._schedule(reanchor=True)

    def _focus_current(self) -> None:
        if self.scene is None:
            return
        if (
            len(self.scene.mesh_vertices_display_m)
            or len(self.scene.skeleton.points)
            or len(self.scene.cloud.points)
        ):
            center, front, up = _front_view_parameters(self.scene)
        elif self.show_ground_grid:
            center = np.array(
                [
                    self.grid_arrays.center_xy_m[0],
                    self.grid_arrays.center_xy_m[1],
                    self.grid_arrays.floor_height_m + 0.8,
                ]
            )
            front = np.array([0.0, -1.0, 0.0])
            up = np.array([0.0, 0.0, 1.0])
        else:
            center = np.array([0.0, 2.0, 0.0])
            front = np.array([0.0, -1.0, 0.0])
            up = np.array([0.0, 0.0, 1.0])

        # update_geometry() deliberately preserves the user's view and does
        # not refresh the Visualizer bounding box.  Explicit focus actions
        # must recompute it.  Exclude the camera-origin axis while fitting so
        # a distant person crop is not made artificially small.
        restore_axis = self.axis_geometry_added
        if restore_axis:
            self.visualizer.remove_geometry(
                self.axis_geometry,
                reset_bounding_box=False,
            )
            self.axis_geometry_added = False
        self.visualizer.reset_view_point(reset_bounding_box=True)
        if restore_axis:
            self.axis_geometry_added = bool(
                self.visualizer.add_geometry(
                    self.axis_geometry,
                    reset_bounding_box=False,
                )
            )
        control = self.visualizer.get_view_control()
        control.set_lookat(center)
        control.set_front(front)
        control.set_up(up)
        control.set_zoom(0.72)
        self.visualizer.update_renderer()

    def _toggle_pause(self, _visualizer: Any) -> bool:
        self.paused = not self.paused
        LOGGER.info("Playback %s.", "paused" if self.paused else "running")
        if self.paused:
            self.next_due = float("inf")
        else:
            self._schedule(reanchor=True)
        return False

    def _previous(self, _visualizer: Any) -> bool:
        self._seek(-1)
        return False

    def _next(self, _visualizer: Any) -> bool:
        self._seek(1)
        return False

    def _raw(self, _visualizer: Any) -> bool:
        self._switch_layer("raw")
        return False

    def _temporal(self, _visualizer: Any) -> bool:
        self._switch_layer("temporal")
        return False

    def _constrained(self, _visualizer: Any) -> bool:
        self._switch_layer("constrained")
        return False

    def _set_point_color(self, option: Any, label: str) -> bool:
        render = self.visualizer.get_render_option()
        render.point_color_option = option
        self.visualizer.update_geometry(self.cloud_geometry)
        self.visualizer.update_renderer()
        LOGGER.info("Point-cloud color mode: %s.", label)
        return False

    def _point_color_default(self, _visualizer: Any) -> bool:
        return self._set_point_color(
            self.o3d.visualization.PointColorOption.Default,
            "Open3D default",
        )

    def _point_color_rgb(self, _visualizer: Any) -> bool:
        return self._set_point_color(
            self.o3d.visualization.PointColorOption.Color,
            "source RGB",
        )

    def _point_color_x(self, _visualizer: Any) -> bool:
        return self._set_point_color(
            self.o3d.visualization.PointColorOption.XCoordinate,
            "X-coordinate pseudo-color",
        )

    def _point_color_y(self, _visualizer: Any) -> bool:
        return self._set_point_color(
            self.o3d.visualization.PointColorOption.YCoordinate,
            "Y-coordinate pseudo-color",
        )

    def _point_color_z(self, _visualizer: Any) -> bool:
        return self._set_point_color(
            self.o3d.visualization.PointColorOption.ZCoordinate,
            "Z-coordinate pseudo-color",
        )

    def _point_color_normal(self, _visualizer: Any) -> bool:
        return self._set_point_color(
            self.o3d.visualization.PointColorOption.Normal,
            "point-normal color",
        )

    def _disable_numeric_color_shortcut(
        self,
        _visualizer: Any,
    ) -> bool:
        return self._set_point_color(
            self.o3d.visualization.PointColorOption.Color,
            "source RGB (numeric shortcuts disabled; use G/X/Y/Z/O/N)",
        )

    def _toggle_cloud_scope(self, _visualizer: Any) -> bool:
        scopes = ("bbox", "full", "none")
        self.cloud_scope = scopes[
            (scopes.index(self.cloud_scope) + 1) % len(scopes)
        ]
        LOGGER.info("Point-cloud scope: %s.", self.cloud_scope)
        self._render_current()
        self._focus_current()
        return False

    def _toggle_render_style(self, _visualizer: Any) -> bool:
        styles = ("skeleton", "mannequin", "both")
        self.render_style = styles[
            (styles.index(self.render_style) + 1) % len(styles)
        ]
        self._sync_dynamic_geometries()
        self._focus_current()
        self.visualizer.update_renderer()
        LOGGER.info("Pose render style: %s.", self.render_style)
        return False

    def _toggle_ground_grid(self, _visualizer: Any) -> bool:
        self.show_ground_grid = not self.show_ground_grid
        self._sync_grid_geometry()
        self._focus_current()
        LOGGER.info(
            "Metric ground grid %s.",
            "enabled" if self.show_ground_grid else "disabled",
        )
        return False

    def _toggle_loop(self, _visualizer: Any) -> bool:
        self.loop = not self.loop
        LOGGER.info("Looping %s.", "enabled" if self.loop else "disabled")
        return False

    def _focus(self, _visualizer: Any) -> bool:
        self._focus_current()
        if self.scene is not None and len(self.scene.skeleton.points) > 0:
            self.has_focused_on_skeleton = True
        return False

    def _help(self, _visualizer: Any) -> bool:
        print(HELP_TEXT, flush=True)
        return False

    def _quit(self, _visualizer: Any) -> bool:
        self.running = False
        return False

    def _register_callbacks(self) -> None:
        callbacks = {
            32: self._toggle_pause,
            263: self._previous,
            262: self._next,
            ord("A"): self._previous,
            ord("D"): self._next,
            ord("R"): self._raw,
            ord("T"): self._temporal,
            ord("C"): self._constrained,
            ord("M"): self._toggle_render_style,
            ord("G"): self._point_color_rgb,
            ord("X"): self._point_color_x,
            ord("Y"): self._point_color_y,
            ord("Z"): self._point_color_z,
            ord("O"): self._point_color_default,
            ord("N"): self._point_color_normal,
            ord("B"): self._toggle_cloud_scope,
            ord("K"): self._toggle_ground_grid,
            ord("L"): self._toggle_loop,
            ord("F"): self._focus,
            ord("H"): self._help,
            ord("Q"): self._quit,
        }
        for key in range(ord("0"), ord("9") + 1):
            callbacks[key] = self._disable_numeric_color_shortcut
        # GLFW keypad digits are the contiguous key codes 320..329.
        for key in range(320, 330):
            callbacks[key] = self._disable_numeric_color_shortcut
        for key, callback in callbacks.items():
            self.visualizer.register_key_callback(key, callback)

    def run(self) -> None:
        title = (
            (
                "SMPL Neutral Motion Avatar"
                if self.avatar_model == "smpl"
                else (
                    "Mixamo Motion Avatar"
                    if self.avatar_model == "mixamo"
                    else (
                        "Stick Figure Motion Avatar"
                        if self.avatar_model == "stickman"
                        else "Metric 3D Motion Avatar"
                    )
                )
            )
            if self.cloud_scope == "none"
            else "RGB-D 3D Skeleton"
        )
        created = self.visualizer.create_window(
            window_name=(
                f"{title} | "
                f"{self.loader.jsonl_path.parent.name} | "
                "display axes: X right, Y forward, Z up"
            ),
            width=self.window_width,
            height=self.window_height,
        )
        if not created:
            raise RuntimeError(
                "Open3D could not create a window. Run this command from a "
                "desktop terminal with a valid DISPLAY; use --validate-only "
                "on a headless machine."
            )

        try:
            initial = self.loader.build(
                self.index,
                self.layer,
                self.cloud_scope,
            )
            self._assign_scene(initial)
            self.scene = initial
            self._sync_cloud_geometry()
            self._sync_grid_geometry()
            self._sync_dynamic_geometries()
            if self.avatar_model != "stickman":
                self.axis_geometry_added = bool(
                    self.visualizer.add_geometry(
                        self.axis_geometry,
                        reset_bounding_box=False,
                    )
                )
            render = self.visualizer.get_render_option()
            render.background_color = np.array(
                [0.62, 0.62, 0.62]
                if self.avatar_model == "stickman"
                else [0.025, 0.025, 0.035]
            )
            render.point_size = 2.0
            render.line_width = 4.0
            render.mesh_show_back_face = True
            render.point_color_option = (
                self.o3d.visualization.PointColorOption.Color
            )
            self._register_callbacks()
            self._focus_current()
            self.has_focused_on_skeleton = bool(
                len(initial.skeleton.points)
            )
            self._log_scene(initial)
            print(HELP_TEXT, flush=True)

            self.running = True
            if not self.paused:
                self._schedule(reanchor=True)
            while self.running:
                if not self.visualizer.poll_events():
                    break
                if not self.paused and time.monotonic() >= self.next_due:
                    self._advance()
                self.visualizer.update_renderer()
                time.sleep(0.005)
        finally:
            self.visualizer.destroy_window()


def _resolve_display_mode(args: argparse.Namespace) -> None:
    smpl_shorthand = bool(getattr(args, "smpl", False))
    mixamo_shorthand = bool(getattr(args, "mixamo", False))
    stickman_shorthand = bool(getattr(args, "stickman", False))
    mannequin_shorthand = bool(getattr(args, "mannequin", False))
    if not hasattr(args, "avatar_model"):
        args.avatar_model = "procedural"
    if sum((smpl_shorthand, mixamo_shorthand, stickman_shorthand)) > 1:
        raise ValueError("--smpl, --mixamo and --stickman are mutually exclusive.")
    if smpl_shorthand:
        args.avatar_model = "smpl"
        args.mannequin = True
        mannequin_shorthand = True
    elif mixamo_shorthand:
        args.avatar_model = "mixamo"
        args.mannequin = True
        mannequin_shorthand = True
    elif stickman_shorthand:
        args.avatar_model = "stickman"
        args.mannequin = True
        mannequin_shorthand = True
    render_style = getattr(args, "render_style", None)
    if mannequin_shorthand:
        if render_style not in (None, "mannequin"):
            raise ValueError(
                "--mannequin cannot be combined with a different "
                "--render-style."
            )
        args.render_style = "mannequin"
    elif render_style is None:
        args.render_style = "skeleton"

    empty_metric_space = args.skeleton_only or mannequin_shorthand
    if empty_metric_space:
        if args.cloud_scope not in (None, "none"):
            raise ValueError(
                "--skeleton-only/--mannequin cannot be combined with a "
                "visible --cloud-scope."
            )
        args.cloud_scope = "none"
        if args.pose_layer is None:
            args.pose_layer = "constrained"
    else:
        if args.cloud_scope is None:
            args.cloud_scope = "bbox"
        if args.pose_layer is None:
            args.pose_layer = "raw"
    if args.ground_grid is None:
        args.ground_grid = (
            args.cloud_scope == "none" and not stickman_shorthand
        )


def load_smpl_sequence_cache(
    jsonl_path: Path,
    *,
    explicit_path: Path | None,
    records: list[dict[str, Any]],
    pose_layer: str,
    camera_to_display_transform: np.ndarray | None,
) -> tuple[SMPLSequenceCache, Path]:
    """Load a cache and reject stale or coordinate-incompatible data."""

    path = (
        explicit_path.expanduser().resolve()
        if explicit_path is not None
        else jsonl_path.parent / "smpl_sequence.npz"
    )
    cache = SMPLSequenceCache.load(path)
    record_indices = np.asarray(
        [record["frame_index"] for record in records],
        dtype=np.int64,
    )
    if len(cache.frame_indices) < len(record_indices) or not np.array_equal(
        cache.frame_indices[: len(record_indices)],
        record_indices,
    ):
        raise ValueError(
            "SMPL cache frame indices do not match the selected poses.jsonl "
            "prefix. Re-run scripts/fit_smpl_sequence.py."
        )
    fitted_layer = cache.metadata.get("pose_layer")
    if fitted_layer != pose_layer:
        raise ValueError(
            f"SMPL cache was fitted from {fitted_layer!r}, but the viewer "
            f"starts on {pose_layer!r}. Select --pose-layer {fitted_layer} "
            "or regenerate the cache."
        )
    recorded_poses_hash = cache.metadata.get("poses_sha256")
    if (
        recorded_poses_hash is not None
        and recorded_poses_hash != sha256_file(jsonl_path)
    ):
        raise ValueError(
            "SMPL cache was generated from different poses.jsonl content. "
            "Re-run scripts/fit_smpl_sequence.py."
        )
    recorded_hands_path = cache.metadata.get("hands_jsonl")
    recorded_hands_hash = cache.metadata.get("hands_sha256")
    if recorded_hands_path is not None and recorded_hands_hash is not None:
        adjacent_hands = jsonl_path.parent / Path(recorded_hands_path).name
        hands_path = (
            adjacent_hands
            if adjacent_hands.is_file()
            else Path(recorded_hands_path).expanduser().resolve()
        )
        if not hands_path.is_file() or sha256_file(hands_path) != recorded_hands_hash:
            raise ValueError(
                "SMPL cache was generated from a missing or changed Hand21 "
                "cache. Re-run scripts/fit_smpl_sequence.py."
            )
    recorded_transform = cache.metadata.get(
        "camera_to_display_transform"
    )
    if recorded_transform is None:
        transform_matches = camera_to_display_transform is None
    else:
        recorded_array = np.asarray(recorded_transform, dtype=np.float64)
        transform_matches = (
            camera_to_display_transform is not None
            and recorded_array.shape == (4, 4)
            and np.allclose(
                recorded_array,
                camera_to_display_transform,
                rtol=1e-9,
                atol=1e-9,
            )
        )
    if not transform_matches:
        raise ValueError(
            "SMPL cache and viewer use different ground-alignment modes. "
            "Use the same --ground-plane/--no-ground-alignment choice used "
            "during fitting."
        )
    return cache, path


def load_mixamo_sequence_cache(
    jsonl_path: Path,
    *,
    explicit_path: Path | None,
    records: list[dict[str, Any]],
    pose_layer: str,
    camera_to_display_transform: np.ndarray | None,
) -> tuple[MixamoSequenceCache, Path]:
    """Load a direct-IK Mixamo cache and reject stale inputs."""

    path = (
        explicit_path.expanduser().resolve()
        if explicit_path is not None
        else jsonl_path.parent / "mixamo_sequence.npz"
    )
    cache = MixamoSequenceCache.load(path)
    record_indices = np.asarray(
        [record["frame_index"] for record in records], dtype=np.int64
    )
    if len(cache.frame_indices) < len(record_indices) or not np.array_equal(
        cache.frame_indices[: len(record_indices)], record_indices
    ):
        raise ValueError(
            "Mixamo cache frame indices do not match poses.jsonl prefix. "
            "Re-run scripts/fit_mixamo_sequence.py."
        )
    if cache.metadata.get("pose_layer") != pose_layer:
        raise ValueError(
            "Mixamo cache pose layer differs from the viewer. Select "
            f"--pose-layer {cache.metadata.get('pose_layer')} or rebuild it."
        )
    if cache.metadata.get("poses_sha256") != sha256_file(jsonl_path):
        raise ValueError(
            "Mixamo cache was generated from different poses.jsonl content."
        )
    recorded_hands = cache.metadata.get("hands_jsonl")
    recorded_hands_hash = cache.metadata.get("hands_sha256")
    if recorded_hands and recorded_hands_hash:
        adjacent = jsonl_path.parent / Path(recorded_hands).name
        hands_path = adjacent if adjacent.is_file() else Path(recorded_hands)
        if not hands_path.is_file() or sha256_file(hands_path) != recorded_hands_hash:
            raise ValueError("Mixamo cache Hand21 input is missing or changed.")
    recorded_transform = cache.metadata.get("camera_to_display_transform")
    if recorded_transform is None:
        matches = camera_to_display_transform is None
    else:
        recorded = np.asarray(recorded_transform, dtype=np.float64)
        matches = (
            camera_to_display_transform is not None
            and recorded.shape == (4, 4)
            and np.allclose(recorded, camera_to_display_transform, atol=1e-9)
        )
    if not matches:
        raise ValueError(
            "Mixamo cache and viewer use different ground-alignment modes."
        )
    return cache, path


def _validate_args(args: argparse.Namespace) -> None:
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    if args.point_stride <= 0:
        raise ValueError("--point-stride must be positive.")
    if args.bbox_margin_px < 0:
        raise ValueError("--bbox-margin-px must be non-negative.")
    if not np.isfinite(args.bbox_hold_s) or args.bbox_hold_s < 0:
        raise ValueError("--bbox-hold-s must be finite and non-negative.")
    if not np.isfinite(args.playback_speed) or args.playback_speed <= 0:
        raise ValueError("--playback-speed must be finite and positive.")
    if not np.isfinite(args.fixed_fps) or args.fixed_fps <= 0:
        raise ValueError("--fixed-fps must be finite and positive.")
    if not np.isfinite(args.joint_radius_m) or args.joint_radius_m <= 0:
        raise ValueError("--joint-radius-m must be finite and positive.")
    if (
        not np.isfinite(args.hand_joint_radius_m)
        or args.hand_joint_radius_m <= 0
    ):
        raise ValueError("--hand-joint-radius-m must be finite and positive.")
    if args.sphere_resolution < 3:
        raise ValueError("--sphere-resolution must be at least 3.")
    if args.smpl_smooth_iterations < 0:
        raise ValueError("--smpl-smooth-iterations must be non-negative.")
    if not np.isfinite(args.grid_spacing_m) or args.grid_spacing_m <= 0:
        raise ValueError("--grid-spacing-m must be finite and positive.")
    if not np.isfinite(args.grid_margin_m) or args.grid_margin_m < 0:
        raise ValueError("--grid-margin-m must be finite and non-negative.")
    if args.window_width <= 0 or args.window_height <= 0:
        raise ValueError("Window dimensions must be positive.")


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        _resolve_display_mode(args)
        _validate_args(args)
        jsonl_path = (
            args.poses_jsonl.expanduser().resolve()
            if args.poses_jsonl is not None
            else args.results_dir.expanduser().resolve() / "poses.jsonl"
        )
        records = load_pose_records(jsonl_path)
        if args.max_frames is not None:
            records = records[: args.max_frames]
        hand_candidate = (
            args.hands_jsonl.expanduser().resolve()
            if args.hands_jsonl is not None
            else jsonl_path.parent / "hands.jsonl"
        )
        if args.show_hands is None:
            args.show_hands = hand_candidate.is_file()
        hand_records = None
        hand_path = None
        if args.show_hands:
            if not hand_candidate.is_file():
                raise FileNotFoundError(
                    f"Hand21 cache not found: {hand_candidate}. Run "
                    "scripts/extract_hand_pose_sequence.py first."
                )
            hand_path = hand_candidate
            hand_records = load_hand_pose_records(hand_path, records)
        camera, intrinsics = load_camera_config(args.camera_config)
        verify_manifest_camera(
            jsonl_path,
            camera,
            allow_mismatch=args.allow_camera_config_mismatch,
        )
        ground_estimate, camera_to_display_transform = (
            load_ground_alignment(
                jsonl_path,
                ground_plane_path=args.ground_plane,
                disabled=args.no_ground_alignment,
            )
        )
        smpl_cache = None
        smpl_cache_path = None
        mixamo_cache = None
        mixamo_cache_path = None
        if args.avatar_model == "smpl":
            smpl_cache, smpl_cache_path = load_smpl_sequence_cache(
                jsonl_path,
                explicit_path=args.smpl_cache,
                records=records,
                pose_layer=args.pose_layer,
                camera_to_display_transform=camera_to_display_transform,
            )
        elif args.avatar_model == "mixamo":
            mixamo_cache, mixamo_cache_path = load_mixamo_sequence_cache(
                jsonl_path,
                explicit_path=args.mixamo_cache,
                records=records,
                pose_layer=args.pose_layer,
                camera_to_display_transform=camera_to_display_transform,
            )
        loader = FrameSceneLoader(
            records=records,
            jsonl_path=jsonl_path,
            camera=camera,
            intrinsics=intrinsics,
            sequence_dir=args.sequence_dir,
            point_stride=args.point_stride,
            bbox_margin_px=args.bbox_margin_px,
            bbox_hold_s=args.bbox_hold_s,
            camera_to_display_transform=camera_to_display_transform,
            smpl_cache=smpl_cache,
            mixamo_cache=mixamo_cache,
            hand_records=hand_records,
            avatar_model=args.avatar_model,
        )
        LOGGER.info(
            "Loaded %d records from %s; layer=%s style=%s avatar=%s "
            "cloud=%s hands=%s timing=%s space=%s%s",
            len(records),
            jsonl_path,
            args.pose_layer,
            args.render_style,
            args.avatar_model,
            args.cloud_scope,
            hand_path if hand_path is not None else "off",
            args.timing,
            (
                "ground_aligned"
                if ground_estimate is not None
                else "optical_camera"
            ),
            (
                f" smpl_cache={smpl_cache_path}"
                if smpl_cache_path is not None
                else (
                    f" mixamo_cache={mixamo_cache_path}"
                    if mixamo_cache_path is not None
                    else ""
                )
            ),
        )
        if args.validate_only:
            validate_scenes(
                loader,
                layer=args.pose_layer,
                scope=args.cloud_scope,
                avatar_model=args.avatar_model,
            )
            return 0

        viewer = Pose3DSequenceViewer(
            loader=loader,
            layer=args.pose_layer,
            cloud_scope=args.cloud_scope,
            render_style=args.render_style,
            avatar_model=args.avatar_model,
            timing=args.timing,
            playback_speed=args.playback_speed,
            fixed_fps=args.fixed_fps,
            loop=args.loop,
            start_paused=args.start_paused,
            joint_radius_m=args.joint_radius_m,
            hand_joint_radius_m=args.hand_joint_radius_m,
            show_hands=args.show_hands,
            sphere_resolution=args.sphere_resolution,
            smpl_smooth_iterations=args.smpl_smooth_iterations,
            show_ground_grid=args.ground_grid,
            grid_spacing_m=args.grid_spacing_m,
            grid_margin_m=args.grid_margin_m,
            window_width=args.window_width,
            window_height=args.window_height,
        )
        viewer.run()
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
