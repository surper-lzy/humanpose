import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rgbd_avatar.visualization.viewer_app import (
    FrameSceneLoader,
    Pose3DSequenceViewer,
    _front_view_parameters,
    _resolve_display_mode,
    load_smpl_sequence_cache,
)
from rgbd_avatar.avatar import SMPLSequenceCache
from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import GroundPlaneEstimate
from rgbd_avatar.pose import HALPE26_LINKS, HALPE26_NAMES
from rgbd_avatar.visualization.contracts import (
    load_ground_alignment,
    verify_manifest_camera,
)
from rgbd_avatar.visualization.hand3d import build_hand_skeleton_display_arrays
from rgbd_avatar.visualization.sequence3d import (
    CORRECTED_COLOR,
    DISPLAY_FROM_CAMERA,
    PREDICTED_COLOR,
    build_cloud_display_arrays,
    build_ground_grid_display_arrays,
    build_skeleton_display_arrays,
    camera_to_display,
    load_pose_records,
    parse_pose_layer,
    playback_delay_s,
    propagate_segment_bboxes,
    resolve_frame_sources,
)


def _joint_payload(layer: str) -> list[dict]:
    joints = []
    for index, name in enumerate(HALPE26_NAMES):
        joint = {
            "id": index,
            "name": name,
            "xyz_m": [0.01 * index, 0.02 * index, 2.0],
            "confidence": 0.9,
        }
        if layer == "raw":
            joint["valid"] = True
            joint["depth_m"] = 2.0
            joint["depth_confidence"] = 0.8
        else:
            joint["usable"] = True
            joint["observed"] = True
            joint["predicted"] = False
            joint["age_s"] = 0.0
            if layer == "constrained":
                joint["corrected"] = False
                joint["correction_m"] = 0.0
        joints.append(joint)
    return joints


def _record(
    *,
    frame_index: int = 0,
    segment_id: int = 0,
    relative_time_s: float = 0.0,
) -> dict:
    timestamp = f"20260730_150513{270 + frame_index:03d}"
    return {
        "schema_version": 1,
        "frame_index": frame_index,
        "timestamp_raw": timestamp,
        "relative_time_s": relative_time_s,
        "segment_id": segment_id,
        "segment_start": frame_index == 0,
        "sources": {
            "rgb": f"{timestamp}_r.png",
            "depth": f"{timestamp}_d.pgm",
        },
        "pose2d": {
            "bbox_xyxy": [10.0, 20.0, 100.0, 200.0],
        },
        "pose3d_raw": {"joints": _joint_payload("raw")},
        "pose3d_temporal": {"joints": _joint_payload("temporal")},
        "pose3d_constrained": {
            "joints": _joint_payload("constrained")
        },
    }


def test_load_pose_records_validates_jsonl_schema(tmp_path: Path) -> None:
    path = tmp_path / "poses.jsonl"
    records = [_record(), _record(frame_index=1, relative_time_s=0.5)]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_pose_records(path)

    assert [record["frame_index"] for record in loaded] == [0, 1]

    path.write_text('{"schema_version": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="schema_version"):
        load_pose_records(path)


def test_load_pose_records_rejects_non_monotonic_time(
    tmp_path: Path,
) -> None:
    path = tmp_path / "poses.jsonl"
    records = [_record(), _record(frame_index=1, relative_time_s=0.0)]
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="relative_time_s"):
        load_pose_records(path)


def test_manifest_camera_verification_detects_changed_intrinsics(
    tmp_path: Path,
) -> None:
    camera = {
        "rgb_width": 816,
        "rgb_height": 612,
        "depth_width": 816,
        "depth_height": 612,
        "depth_scale": 0.001,
        "min_depth_m": 0.3,
        "max_depth_m": 6.0,
        "align_depth_to_rgb": True,
        "images_undistorted": True,
        "intrinsics": {
            "fx": 390.0,
            "fy": 391.0,
            "cx": 408.0,
            "cy": 322.0,
        },
    }
    manifest_camera = json.loads(json.dumps(camera))
    (tmp_path / "poses.jsonl").touch()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"camera": manifest_camera}),
        encoding="utf-8",
    )

    verify_manifest_camera(
        tmp_path / "poses.jsonl",
        camera,
        allow_mismatch=False,
    )
    manifest_camera["intrinsics"]["fx"] = 400.0
    (tmp_path / "manifest.json").write_text(
        json.dumps({"camera": manifest_camera}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="intrinsics.fx"):
        verify_manifest_camera(
            tmp_path / "poses.jsonl",
            camera,
            allow_mismatch=False,
        )


def test_ground_alignment_loader_uses_saved_plane(tmp_path: Path) -> None:
    estimate = GroundPlaneEstimate(
        normal_camera=np.array([0.0, -1.0, 0.0]),
        offset_m=1.8,
        inlier_count=1000,
        candidate_count=1200,
        inlier_ratio=1000 / 1200,
        residual_median_m=0.005,
        residual_p95_m=0.015,
        residual_rms_m=0.008,
        tilt_from_camera_up_deg=0.0,
    )
    path = tmp_path / "ground_plane.json"
    path.write_text(json.dumps(estimate.to_dict()), encoding="utf-8")

    loaded, transform = load_ground_alignment(
        tmp_path / "poses.jsonl",
        ground_plane_path=path,
        disabled=False,
    )

    assert loaded is not None
    assert transform is not None
    np.testing.assert_allclose(transform, estimate.camera_to_ground_transform())
    assert transform[2, 3] == pytest.approx(1.8)


def test_parse_layers_preserves_provenance_and_falls_back() -> None:
    record = _record()
    record["pose3d_raw"]["joints"][4]["valid"] = False
    record["pose3d_raw"]["joints"][4]["xyz_m"] = None
    record["pose3d_temporal"]["joints"][9].update(
        observed=False,
        predicted=True,
    )
    record["pose3d_constrained"]["joints"][10].update(
        corrected=True,
        correction_m=0.04,
    )

    raw = parse_pose_layer(record, "raw")
    temporal = parse_pose_layer(record, "temporal")
    constrained = parse_pose_layer(record, "constrained")

    assert not raw.usable[4]
    assert np.isnan(raw.joints_camera_m[4]).all()
    assert temporal.predicted[9] and not temporal.observed[9]
    assert constrained.corrected[10]

    record["pose3d_constrained"] = None
    fallback = parse_pose_layer(record, "constrained")
    assert fallback.requested_layer == "constrained"
    assert fallback.resolved_layer == "temporal"


def test_missing_raw_pose_keeps_frame_with_empty_skeleton() -> None:
    record = _record()
    record["pose3d_raw"] = None

    pose = parse_pose_layer(record, "raw")
    skeleton = build_skeleton_display_arrays(pose)

    assert not np.any(pose.usable)
    assert skeleton.points.shape == (0, 3)
    assert skeleton.lines.shape == (0, 2)


def test_skeleton_compacts_joint_indices_and_filters_links() -> None:
    record = _record()
    pose = parse_pose_layer(record, "raw")
    all_valid = build_skeleton_display_arrays(pose)

    assert all_valid.points.shape == (26, 3)
    assert all_valid.lines.shape == (len(HALPE26_LINKS), 2)

    record["pose3d_raw"]["joints"][19]["valid"] = False
    pose_without_hip = parse_pose_layer(record, "raw")
    compact = build_skeleton_display_arrays(pose_without_hip)
    expected_lines = sum(19 not in link for link in HALPE26_LINKS)

    assert compact.points.shape == (25, 3)
    assert compact.lines.shape == (expected_lines, 2)
    assert np.max(compact.lines) < len(compact.points)
    assert np.isfinite(compact.points).all()


def test_camera_to_display_is_a_proper_rotation() -> None:
    points = np.array([[1.0, 2.0, 3.0], [-2.0, 0.5, 4.0]])
    original = points.copy()

    transformed = camera_to_display(points)

    np.testing.assert_allclose(transformed[0], [1.0, 3.0, -2.0])
    np.testing.assert_array_equal(points, original)
    np.testing.assert_allclose(
        np.linalg.norm(transformed, axis=1),
        np.linalg.norm(points, axis=1),
    )
    assert np.linalg.det(DISPLAY_FROM_CAMERA) == pytest.approx(1.0)


def test_cloud_arrays_filter_nan_decimate_and_normalize_rgb() -> None:
    rows, columns = np.indices((4, 4), dtype=np.float64)
    organized = np.stack(
        (columns * 0.1, rows * 0.1, np.full((4, 4), 2.0)),
        axis=-1,
    )
    organized[0, 0] = np.nan
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    rgb[..., 0] = 255

    cloud = build_cloud_display_arrays(
        organized,
        rgb,
        stride=2,
        scope="full",
    )

    assert cloud.points.shape == (3, 3)
    assert cloud.colors.shape == (3, 3)
    np.testing.assert_allclose(
        cloud.colors,
        np.tile([1.0, 0.0, 0.0], (3, 1)),
    )
    assert np.isfinite(cloud.points).all()

    with pytest.raises(ValueError, match="shapes must match"):
        build_cloud_display_arrays(organized, rgb[:3], scope="full")


def test_cloud_arrays_apply_bbox_crop_and_invalid_bbox_falls_back() -> None:
    rows, columns = np.indices((5, 6), dtype=np.float64)
    organized = np.stack(
        (columns, rows, np.full((5, 6), 2.0)),
        axis=-1,
    )
    rgb = np.full((5, 6, 3), 127, dtype=np.uint8)

    cropped = build_cloud_display_arrays(
        organized,
        rgb,
        stride=1,
        scope="bbox",
        bbox_xyxy=np.array([1.0, 1.0, 3.0, 3.0]),
        bbox_margin_px=0,
    )
    fallback = build_cloud_display_arrays(
        organized,
        rgb,
        stride=1,
        scope="bbox",
        bbox_xyxy=np.array([4.0, 3.0, 2.0, 1.0]),
        bbox_margin_px=0,
    )

    assert cropped.resolved_scope == "bbox"
    assert cropped.points.shape == (9, 3)
    np.testing.assert_allclose(cropped.points[0], [1.0, 2.0, -1.0])
    assert fallback.resolved_scope == "full"
    assert fallback.points.shape == (30, 3)


def test_skeleton_colors_prioritize_corrected_over_predicted() -> None:
    record = _record()
    record["pose3d_constrained"]["joints"][9].update(
        observed=False,
        predicted=True,
    )
    record["pose3d_constrained"]["joints"][10].update(
        observed=False,
        predicted=True,
        corrected=True,
    )
    skeleton = build_skeleton_display_arrays(
        parse_pose_layer(record, "constrained")
    )

    np.testing.assert_allclose(skeleton.point_colors[9], PREDICTED_COLOR)
    np.testing.assert_allclose(skeleton.point_colors[10], CORRECTED_COLOR)
    corrected_link = HALPE26_LINKS.index((8, 10))
    np.testing.assert_allclose(
        skeleton.line_colors[corrected_link],
        CORRECTED_COLOR,
    )


def test_ground_grid_uses_foot_height_and_covers_metric_skeleton() -> None:
    grid = build_ground_grid_display_arrays(
        [_record()],
        layer="constrained",
        spacing_m=0.25,
        margin_m=0.5,
    )

    assert grid.points.shape[1] == 3
    assert grid.lines.shape[1] == 2
    assert len(grid.colors) == len(grid.lines)
    assert grid.floor_height_m == pytest.approx(-0.5)
    assert np.ptp(grid.points[:, 0]) >= 2.0
    assert np.ptp(grid.points[:, 1]) >= 2.0


def test_skeleton_only_loader_does_not_read_rgbd_sources(
    tmp_path: Path,
) -> None:
    loader = FrameSceneLoader(
        records=[_record()],
        jsonl_path=tmp_path / "poses.jsonl",
        camera={},
        intrinsics=CameraIntrinsics(
            fx=390.0,
            fy=390.0,
            cx=408.0,
            cy=322.0,
            width=816,
            height=612,
        ),
        sequence_dir=None,
        point_stride=3,
        bbox_margin_px=30,
        bbox_hold_s=1.1,
    )

    scene = loader.build(0, "constrained", "none")

    assert scene.cloud.resolved_scope == "none"
    assert scene.cloud.points.shape == (0, 3)
    assert scene.rgb_path is None
    assert scene.depth_path is None
    assert len(scene.skeleton.points) == 26
    assert scene.avatar.primitive_count > 0


def test_hand_skeleton_arrays_show_valid_and_flag_collapsed_hand() -> None:
    xyz = np.zeros((21, 3), dtype=np.float64)
    xyz[:, 2] = 2.0
    bases = {
        1: (-0.035, 0.025),
        5: (-0.030, 0.060),
        9: (0.000, 0.075),
        13: (0.030, 0.065),
        17: (0.050, 0.045),
    }
    for base, (x, y) in bases.items():
        xyz[base, :2] = [x, y]
        for offset in range(1, 4):
            xyz[base + offset, :2] = [x, y + 0.025 * offset]
    joints = []
    for index in range(21):
        joints.append(
            {
                "xyz_m": xyz[index].tolist(),
                "confidence": 0.8,
                "valid": True,
            }
        )
    record = {"hands": {"left": {"pose3d": {"joints": joints}}}}

    skeleton, rejected = build_hand_skeleton_display_arrays(
        record,
        camera_to_display_transform=None,
    )

    assert len(skeleton.points) == 21
    assert len(skeleton.lines) == 20
    assert rejected == ()

    for joint in joints:
        joint["xyz_m"] = [0.0, 0.0, 2.0]
    _, rejected = build_hand_skeleton_display_arrays(
        record,
        camera_to_display_transform=None,
    )
    assert rejected == ("left:collapsed_hand",)


def test_front_view_targets_upper_torso_with_elevated_camera() -> None:
    points = np.array(
        [[0.0, 2.0, 0.0]] * 12
        + [[0.0, 2.0, 0.8]] * 8
        + [[0.0, 2.0, 1.8]] * 4,
        dtype=np.float64,
    )
    scene = SimpleNamespace(
        mesh_vertices_display_m=np.empty((0, 3)),
        skeleton=SimpleNamespace(points=points),
        cloud=SimpleNamespace(points=np.empty((0, 3))),
    )

    center, front, up = _front_view_parameters(scene)

    assert center[2] > np.median(points[:, 2])
    assert front[2] > 0.0
    np.testing.assert_allclose(np.linalg.norm(front), 1.0)
    np.testing.assert_allclose(np.linalg.norm(up), 1.0)
    np.testing.assert_allclose(np.dot(front, up), 0.0, atol=1e-12)


def test_skeleton_only_mode_defaults_to_constrained_grid_space() -> None:
    args = SimpleNamespace(
        skeleton_only=True,
        cloud_scope=None,
        pose_layer=None,
        ground_grid=None,
    )

    _resolve_display_mode(args)

    assert args.cloud_scope == "none"
    assert args.pose_layer == "constrained"
    assert args.ground_grid is True
    assert args.render_style == "skeleton"


def test_mannequin_mode_defaults_to_constrained_empty_metric_space() -> None:
    args = SimpleNamespace(
        skeleton_only=False,
        mannequin=True,
        render_style=None,
        cloud_scope=None,
        pose_layer=None,
        ground_grid=None,
    )

    _resolve_display_mode(args)

    assert args.cloud_scope == "none"
    assert args.pose_layer == "constrained"
    assert args.render_style == "mannequin"


def test_mixamo_shorthand_selects_cached_mannequin_mode() -> None:
    args = SimpleNamespace(
        skeleton_only=False,
        mannequin=False,
        smpl=False,
        mixamo=True,
        stickman=False,
        avatar_model="procedural",
        render_style=None,
        cloud_scope=None,
        pose_layer=None,
        ground_grid=None,
    )

    _resolve_display_mode(args)

    assert args.avatar_model == "mixamo"
    assert args.render_style == "mannequin"
    assert args.cloud_scope == "none"
    assert args.pose_layer == "constrained"
    assert args.ground_grid is True


def test_smpl_mode_defaults_to_constrained_smpl_mannequin() -> None:
    args = SimpleNamespace(
        skeleton_only=False,
        mannequin=False,
        smpl=True,
        avatar_model="procedural",
        render_style=None,
        cloud_scope=None,
        pose_layer=None,
        ground_grid=None,
    )

    _resolve_display_mode(args)

    assert args.avatar_model == "smpl"
    assert args.cloud_scope == "none"
    assert args.pose_layer == "constrained"
    assert args.render_style == "mannequin"
    assert args.ground_grid is True


def test_stickman_mode_defaults_to_constrained_empty_metric_space() -> None:
    args = SimpleNamespace(
        skeleton_only=False,
        mannequin=False,
        smpl=False,
        stickman=True,
        avatar_model="procedural",
        render_style=None,
        cloud_scope=None,
        pose_layer=None,
        ground_grid=None,
    )

    _resolve_display_mode(args)

    assert args.avatar_model == "stickman"
    assert args.cloud_scope == "none"
    assert args.pose_layer == "constrained"
    assert args.render_style == "mannequin"
    assert args.ground_grid is False


def test_smpl_cache_loader_accepts_selected_prefix_and_checks_space(
    tmp_path: Path,
) -> None:
    frame_count = 2
    vertices = np.zeros((frame_count, 4, 3), dtype=np.float32)
    joints = np.zeros((frame_count, 24, 3), dtype=np.float32)
    cache = SMPLSequenceCache(
        frame_indices=np.array([0, 1]),
        present=np.ones(frame_count, dtype=bool),
        vertices_display_m=vertices,
        joints_display_m=joints,
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        body_pose=np.zeros((frame_count, 69), dtype=np.float32),
        global_orient=np.zeros((frame_count, 3), dtype=np.float32),
        translation_native_m=np.zeros((frame_count, 3), dtype=np.float32),
        target_counts=np.full(frame_count, 15, dtype=np.int16),
        error_mean_m=np.zeros(frame_count, dtype=np.float32),
        error_p95_m=np.zeros(frame_count, dtype=np.float32),
        error_max_m=np.zeros(frame_count, dtype=np.float32),
        scale=1.0,
        metadata={
            "pose_layer": "constrained",
            "ground_plane": "plane.json",
            "camera_to_display_transform": np.eye(4).tolist(),
        },
    )
    path = tmp_path / "smpl_sequence.npz"
    cache.save(path)

    loaded, loaded_path = load_smpl_sequence_cache(
        tmp_path / "poses.jsonl",
        explicit_path=path,
        records=[_record(frame_index=0)],
        pose_layer="constrained",
        camera_to_display_transform=np.eye(4),
    )

    assert loaded_path == path
    np.testing.assert_array_equal(loaded.frame_indices, [0, 1])
    with pytest.raises(ValueError, match="ground-alignment"):
        load_smpl_sequence_cache(
            tmp_path / "poses.jsonl",
            explicit_path=path,
            records=[_record(frame_index=0)],
            pose_layer="constrained",
            camera_to_display_transform=None,
        )


def test_bbox_propagation_stays_within_each_segment() -> None:
    records = [
        _record(frame_index=0, segment_id=0),
        _record(frame_index=1, segment_id=0),
        _record(frame_index=2, segment_id=1),
        _record(frame_index=3, segment_id=1),
    ]
    records[0]["pose2d"] = None
    records[2]["pose2d"] = None
    records[3]["pose2d"] = None

    bboxes = propagate_segment_bboxes(records)

    np.testing.assert_allclose(bboxes[0], [10, 20, 100, 200])
    np.testing.assert_allclose(bboxes[1], [10, 20, 100, 200])
    assert bboxes[2] is None
    assert bboxes[3] is None


def test_bbox_propagation_expires_after_short_gap() -> None:
    records = [
        _record(frame_index=0, relative_time_s=0.0),
        _record(frame_index=1, relative_time_s=0.5),
        _record(frame_index=2, relative_time_s=2.0),
    ]
    records[1]["pose2d"] = None
    records[2]["pose2d"] = None

    bboxes = propagate_segment_bboxes(records, max_age_s=1.1)

    np.testing.assert_allclose(bboxes[1], [10, 20, 100, 200])
    assert bboxes[2] is None


def test_source_override_and_playback_timing(tmp_path: Path) -> None:
    first = _record(frame_index=0, relative_time_s=0.0)
    second = _record(frame_index=1, relative_time_s=0.5)
    rgb, depth = resolve_frame_sources(
        first,
        tmp_path / "poses.jsonl",
        sequence_dir=tmp_path / "moved",
    )
    assert rgb.name.endswith("_r.png")
    assert depth.name.endswith("_d.pgm")
    assert rgb.parent == (tmp_path / "moved").resolve()

    relative_rgb, relative_depth = resolve_frame_sources(
        first,
        tmp_path / "poses.jsonl",
    )
    assert relative_rgb == tmp_path / first["sources"]["rgb"]
    assert relative_depth == tmp_path / first["sources"]["depth"]

    assert playback_delay_s(first, second) == pytest.approx(0.5)
    assert playback_delay_s(
        first,
        second,
        playback_speed=2.0,
    ) == pytest.approx(0.25)
    assert playback_delay_s(
        first,
        second,
        timing="fixed",
        fixed_fps=20.0,
    ) == pytest.approx(0.05)

    second["segment_id"] = 1
    assert playback_delay_s(first, second) == pytest.approx(0.5)


def test_dynamic_geometries_add_update_and_remove_without_window() -> None:
    class FakeVisualizer:
        def __init__(self) -> None:
            self.added = []
            self.updated = []
            self.removed = []

        def add_geometry(self, geometry, reset_bounding_box):
            self.added.append((geometry, reset_bounding_box))
            return True

        def update_geometry(self, geometry):
            self.updated.append(geometry)
            return True

        def remove_geometry(self, geometry, reset_bounding_box):
            self.removed.append((geometry, reset_bounding_box))
            return True

    viewer = Pose3DSequenceViewer.__new__(Pose3DSequenceViewer)
    viewer.visualizer = FakeVisualizer()
    viewer.line_geometry = object()
    viewer.joint_geometry = object()
    viewer.avatar_geometry = object()
    viewer.line_geometry_added = False
    viewer.joint_geometry_added = False
    viewer.avatar_geometry_added = False
    viewer.render_style = "skeleton"
    viewer.scene = SimpleNamespace(
        skeleton=SimpleNamespace(
            lines=np.array([[0, 1]], dtype=np.int32),
            points=np.zeros((2, 3)),
        ),
        avatar=SimpleNamespace(primitive_count=1),
    )

    viewer._sync_dynamic_geometries()
    assert len(viewer.visualizer.added) == 2
    assert viewer.line_geometry_added
    assert viewer.joint_geometry_added

    viewer._sync_dynamic_geometries()
    assert viewer.visualizer.updated == [
        viewer.line_geometry,
        viewer.joint_geometry,
    ]

    viewer.scene = SimpleNamespace(
        skeleton=SimpleNamespace(
            lines=np.empty((0, 2), dtype=np.int32),
            points=np.empty((0, 3)),
        ),
        avatar=SimpleNamespace(primitive_count=0),
    )
    viewer._sync_dynamic_geometries()
    assert len(viewer.visualizer.removed) == 2
    assert not viewer.line_geometry_added
    assert not viewer.joint_geometry_added


def test_mannequin_geometry_adds_and_removes_with_render_style() -> None:
    class FakeVisualizer:
        def __init__(self) -> None:
            self.added = []
            self.updated = []
            self.removed = []

        def add_geometry(self, geometry, reset_bounding_box):
            self.added.append((geometry, reset_bounding_box))
            return True

        def update_geometry(self, geometry):
            self.updated.append(geometry)
            return True

        def remove_geometry(self, geometry, reset_bounding_box):
            self.removed.append((geometry, reset_bounding_box))
            return True

    viewer = Pose3DSequenceViewer.__new__(Pose3DSequenceViewer)
    viewer.visualizer = FakeVisualizer()
    viewer.line_geometry = object()
    viewer.joint_geometry = object()
    viewer.avatar_geometry = object()
    viewer.line_geometry_added = False
    viewer.joint_geometry_added = False
    viewer.avatar_geometry_added = False
    viewer.render_style = "mannequin"
    viewer.scene = SimpleNamespace(
        skeleton=SimpleNamespace(
            lines=np.array([[0, 1]], dtype=np.int32),
            points=np.zeros((2, 3)),
        ),
        avatar=SimpleNamespace(primitive_count=3),
    )

    viewer._sync_dynamic_geometries()
    assert viewer.avatar_geometry_added
    assert viewer.visualizer.added == [(viewer.avatar_geometry, False)]

    viewer._sync_dynamic_geometries()
    assert viewer.visualizer.updated == [viewer.avatar_geometry]

    viewer.render_style = "skeleton"
    viewer._sync_dynamic_geometries()
    assert not viewer.avatar_geometry_added
    assert viewer.visualizer.removed[-1] == (
        viewer.avatar_geometry,
        False,
    )


def test_cloud_geometry_can_be_removed_for_skeleton_only_mode() -> None:
    class FakeVisualizer:
        def __init__(self) -> None:
            self.added = []
            self.updated = []
            self.removed = []

        def add_geometry(self, geometry, reset_bounding_box):
            self.added.append((geometry, reset_bounding_box))
            return True

        def update_geometry(self, geometry):
            self.updated.append(geometry)
            return True

        def remove_geometry(self, geometry, reset_bounding_box):
            self.removed.append((geometry, reset_bounding_box))
            return True

    viewer = Pose3DSequenceViewer.__new__(Pose3DSequenceViewer)
    viewer.visualizer = FakeVisualizer()
    viewer.cloud_geometry = object()
    viewer.cloud_geometry_added = False
    viewer.scene = SimpleNamespace(
        cloud=SimpleNamespace(points=np.ones((2, 3)))
    )

    viewer._sync_cloud_geometry()
    assert viewer.cloud_geometry_added
    assert len(viewer.visualizer.added) == 1

    viewer.scene = SimpleNamespace(
        cloud=SimpleNamespace(points=np.empty((0, 3)))
    )
    viewer._sync_cloud_geometry()
    assert not viewer.cloud_geometry_added
    assert len(viewer.visualizer.removed) == 1


def test_letter_color_keys_and_numeric_conflict_guard_are_registered() -> None:
    class FakeVisualizer:
        def __init__(self) -> None:
            self.callbacks = {}

        def register_key_callback(self, key, callback):
            self.callbacks[key] = callback

    viewer = Pose3DSequenceViewer.__new__(Pose3DSequenceViewer)
    viewer.visualizer = FakeVisualizer()
    viewer._register_callbacks()

    assert (
        viewer.visualizer.callbacks[ord("G")].__func__
        is Pose3DSequenceViewer._point_color_rgb
    )
    assert (
        viewer.visualizer.callbacks[ord("Z")].__func__
        is Pose3DSequenceViewer._point_color_z
    )
    assert (
        viewer.visualizer.callbacks[ord("R")].__func__
        is Pose3DSequenceViewer._raw
    )
    assert (
        viewer.visualizer.callbacks[ord("M")].__func__
        is Pose3DSequenceViewer._toggle_render_style
    )
    for key in (
        *range(ord("0"), ord("9") + 1),
        *range(320, 330),
    ):
        assert (
            viewer.visualizer.callbacks[key].__func__
            is Pose3DSequenceViewer._disable_numeric_color_shortcut
        )


def test_numeric_color_guard_forces_rgb_and_refreshes_cloud() -> None:
    class FakeRenderOption:
        point_color_option = None

    class FakeVisualizer:
        def __init__(self) -> None:
            self.callbacks = {}
            self.render = FakeRenderOption()
            self.updated = []
            self.render_count = 0

        def register_key_callback(self, key, callback):
            self.callbacks[key] = callback

        def get_render_option(self):
            return self.render

        def update_geometry(self, geometry):
            self.updated.append(geometry)

        def update_renderer(self):
            self.render_count += 1

    viewer = Pose3DSequenceViewer.__new__(Pose3DSequenceViewer)
    viewer.visualizer = FakeVisualizer()
    viewer.cloud_geometry = object()
    viewer.o3d = SimpleNamespace(
        visualization=SimpleNamespace(
            PointColorOption=SimpleNamespace(Color="source-rgb")
        )
    )
    viewer._register_callbacks()

    result = viewer.visualizer.callbacks[ord("4")](viewer.visualizer)

    assert result is False
    assert viewer.visualizer.render.point_color_option == "source-rgb"
    assert viewer.visualizer.updated == [viewer.cloud_geometry]
    assert viewer.visualizer.render_count == 1
