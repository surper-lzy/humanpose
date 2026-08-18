#!/usr/bin/env python3
"""Build a directly retargeted Mixamo mesh sequence from metric Halpe26."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from rgbd_avatar.avatar import (
    MixamoSequenceCache,
    load_mixamo_fbx,
    sha256_file,
    skin_mixamo_vertices,
)
from rgbd_avatar.data import load_hand_records
from rgbd_avatar.pose import hand_observation_quality
from rgbd_avatar.retargeting import (
    MixamoAnalyticalIK,
    MixamoHandObservation,
    MixamoIKConfig,
    calibrate_halpe_smpl_profile,
    estimate_mixamo_scale,
)
from rgbd_avatar.visualization import (
    load_pose_records,
    parse_pose_layer,
    transform_camera_points,
)
from rgbd_avatar.visualization.contracts import load_ground_alignment


LOGGER = logging.getLogger("fit_mixamo_sequence")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/sequences/4_pointcloud_exit_gate",
    )
    parser.add_argument("--poses-jsonl", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "assets/models/mixamo/Ch09_nonPBR.fbx",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--hands-jsonl", type=Path, default=None)
    parser.add_argument("--no-hand-targets", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--pose-layer", choices=("temporal", "constrained"), default="constrained"
    )
    parser.add_argument("--ground-plane", type=Path, default=None)
    parser.add_argument("--no-ground-alignment", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument(
        "--maximum-rotation-speed-deg-s", type=float, default=180.0
    )
    parser.add_argument("--rotation-response", type=float, default=0.78)
    parser.add_argument(
        "--ground-contact",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Translate each posed mesh vertically so its lowest vertex "
            "touches the calibrated z=0 ground (default: enabled)."
        ),
    )
    return parser.parse_args()


def _display_pose_arrays(
    records: list[dict[str, Any]],
    *,
    layer: str,
    transform: np.ndarray | None,
) -> tuple[list[Any], list[np.ndarray]]:
    poses: list[Any] = []
    joints_sequence: list[np.ndarray] = []
    for record in records:
        pose = parse_pose_layer(record, layer)
        joints = np.full((26, 3), np.nan, dtype=np.float64)
        keep = pose.usable & np.isfinite(pose.joints_camera_m).all(axis=1)
        if np.any(keep):
            joints[keep] = transform_camera_points(
                pose.joints_camera_m[keep], transform
            )
        poses.append(pose)
        joints_sequence.append(joints)
    return poses, joints_sequence


def _parse_hands(
    record: dict[str, Any],
    transform: np.ndarray | None,
) -> tuple[dict[str, MixamoHandObservation], tuple[str, ...]]:
    result: dict[str, MixamoHandObservation] = {}
    rejected: list[str] = []
    hands = record.get("hands")
    if not isinstance(hands, dict):
        return result, ()
    for side in ("left", "right"):
        hand = hands.get(side)
        pose3d = hand.get("pose3d") if isinstance(hand, dict) else None
        payload = pose3d.get("joints") if isinstance(pose3d, dict) else None
        if not isinstance(payload, list) or len(payload) != 21:
            continue
        joints = np.full((21, 3), np.nan, dtype=np.float64)
        confidence = np.zeros(21, dtype=np.float64)
        valid = np.zeros(21, dtype=bool)
        for index, joint in enumerate(payload):
            xyz = joint.get("xyz_m")
            if bool(joint.get("valid")) and isinstance(xyz, list) and len(xyz) == 3:
                joints[index] = np.asarray(xyz, dtype=np.float64)
                valid[index] = np.isfinite(joints[index]).all()
                confidence[index] = float(joint.get("confidence", 0.0))
        quality_ok, reason, _ = hand_observation_quality(joints, valid)
        if not quality_ok:
            rejected.append(f"{side}:{reason}")
            continue
        joints[valid] = transform_camera_points(joints[valid], transform)
        result[side] = MixamoHandObservation(joints, confidence, valid)
    return result, tuple(rejected)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        poses_path = (
            args.poses_jsonl.expanduser().resolve()
            if args.poses_jsonl is not None
            else args.results_dir.expanduser().resolve() / "poses.jsonl"
        )
        model_path = args.model.expanduser().resolve()
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else poses_path.parent / "mixamo_sequence.npz"
        )
        if not poses_path.is_file():
            raise FileNotFoundError(f"Pose sequence not found: {poses_path}")
        if not model_path.is_file():
            raise FileNotFoundError(f"Mixamo FBX not found: {model_path}")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite Mixamo cache: {output_path}. "
                "Pass --overwrite to replace it atomically."
            )
        records = load_pose_records(poses_path)
        if args.max_frames is not None:
            if args.max_frames <= 0:
                raise ValueError("--max-frames must be positive.")
            records = records[: args.max_frames]
        hands_path: Path | None = None
        hand_records: list[dict[str, Any]] | None = None
        if not args.no_hand_targets:
            candidate = (
                args.hands_jsonl.expanduser().resolve()
                if args.hands_jsonl is not None
                else poses_path.parent / "hands.jsonl"
            )
            if candidate.is_file():
                hands_path = candidate
                hand_records = load_hand_records(candidate, records)
            elif args.hands_jsonl is not None:
                raise FileNotFoundError(f"Hand cache not found: {candidate}")
        _ground, transform = load_ground_alignment(
            poses_path,
            ground_plane_path=args.ground_plane,
            disabled=args.no_ground_alignment,
        )
        ground_path = None if args.no_ground_alignment else (
            args.ground_plane.expanduser().resolve()
            if args.ground_plane is not None
            else poses_path.parent / "ground_plane.json"
        )
        if ground_path is not None and not ground_path.is_file():
            ground_path = None
        poses, display_joints = _display_pose_arrays(
            records, layer=args.pose_layer, transform=transform
        )
        profile = calibrate_halpe_smpl_profile(
            display_joints,
            [pose.confidence for pose in poses],
            [pose.usable for pose in poses],
            [pose.predicted for pose in poses],
        )
        started = time.perf_counter()
        asset = load_mixamo_fbx(model_path)
        estimated_scale = estimate_mixamo_scale(
            asset,
            display_joints,
            [pose.confidence for pose in poses],
            [pose.usable for pose in poses],
            [pose.predicted for pose in poses],
            profile,
        )
        scale = float(args.scale) if args.scale is not None else estimated_scale
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Mixamo scale must be finite and positive.")
        config = MixamoIKConfig(
            maximum_rotation_speed_deg_s=args.maximum_rotation_speed_deg_s,
            rotation_response=args.rotation_response,
        )
        solver = MixamoAnalyticalIK(asset, profile, scale=scale, config=config)
        frame_count = len(records)
        vertex_count = len(asset.vertices_m)
        bone_count = len(asset.bone_names)
        vertices = np.full((frame_count, vertex_count, 3), np.nan, dtype=np.float32)
        bone_matrices = np.full(
            (frame_count, bone_count, 4, 4), np.nan, dtype=np.float32
        )
        present = np.zeros(frame_count, dtype=bool)
        rejected_counts: dict[str, int] = {}
        held_counts: dict[str, int] = {}
        hand_rejections_by_frame: list[dict[str, Any]] = []
        ground_contact_offsets_m = np.full(frame_count, np.nan, dtype=np.float32)
        if args.ground_contact and transform is None:
            LOGGER.warning(
                "Ground contact requested without ground alignment; "
                "contact correction is disabled."
            )
        apply_ground_contact = bool(args.ground_contact and transform is not None)
        previous_time: float | None = None
        for index, (record, pose, joints) in enumerate(
            zip(records, poses, display_joints, strict=True)
        ):
            relative_time = float(record.get("relative_time_s", index * 0.5))
            delta_time = (
                max(1e-3, relative_time - previous_time)
                if previous_time is not None else 0.5
            )
            previous_time = relative_time
            if bool(record.get("segment_start")) and index > 0:
                solver.reset()
            hands, hand_rejections = (
                _parse_hands(hand_records[index], transform)
                if hand_records is not None else ({}, ())
            )
            if hand_rejections:
                hand_rejections_by_frame.append(
                    {"frame_index": int(record["frame_index"]), "hands": list(hand_rejections)}
                )
            frame = solver.solve(
                joints,
                pose.confidence,
                pose.usable,
                pose.predicted,
                delta_time_s=delta_time,
                hands=hands,
            )
            if frame is None:
                LOGGER.info(
                    "[%d/%d] frame=%s skipped status=%s",
                    index + 1, frame_count, record["frame_index"], record.get("status"),
                )
                continue
            present[index] = True
            frame_matrices = frame.bone_global_m.copy()
            frame_vertices = skin_mixamo_vertices(asset, frame_matrices)
            if apply_ground_contact:
                vertical_offset = -float(np.min(frame_vertices[:, 2]))
                frame_matrices[:, 2, 3] += vertical_offset
                frame_vertices[:, 2] += vertical_offset
                ground_contact_offsets_m[index] = vertical_offset
            bone_matrices[index] = frame_matrices
            vertices[index] = frame_vertices
            for value in frame.rejected_segments:
                rejected_counts[value] = rejected_counts.get(value, 0) + 1
            for value in frame.held_bones:
                held_counts[value] = held_counts.get(value, 0) + 1
            LOGGER.info(
                "[%d/%d] frame=%s vertices=%d rejected=%s held=%s",
                index + 1,
                frame_count,
                record["frame_index"],
                vertex_count,
                ",".join(frame.rejected_segments) or "none",
                ",".join(frame.held_bones) or "none",
            )

        cache = MixamoSequenceCache(
            frame_indices=np.asarray(
                [record["frame_index"] for record in records], dtype=np.int64
            ),
            present=present,
            vertices_display_m=vertices,
            faces=asset.faces,
            triangle_uvs=asset.triangle_uvs,
            diffuse_png=np.frombuffer(asset.diffuse_png, dtype=np.uint8).copy(),
            bone_names=asset.bone_names,
            bone_global_m=bone_matrices,
            scale=scale,
            metadata={
                "schema_version": 1,
                "backend": "direct_halpe26_analytical_ik",
                "poses_jsonl": str(poses_path),
                "poses_sha256": sha256_file(poses_path),
                "hands_jsonl": str(hands_path) if hands_path else None,
                "hands_sha256": sha256_file(hands_path) if hands_path else None,
                "model": str(model_path),
                "model_sha256": asset.source_sha256,
                "pose_layer": args.pose_layer,
                "estimated_scale": estimated_scale,
                "scale_source": "command_line" if args.scale is not None else "sequence_bone_estimate",
                "camera_to_display_transform": transform.tolist() if transform is not None else None,
                "ground_plane": str(ground_path) if ground_path else None,
                "ground_plane_sha256": sha256_file(ground_path) if ground_path else None,
                "retarget_profile": profile.to_mapping(),
                "ik_config": asdict(config),
                "ground_contact": apply_ground_contact,
                "ground_contact_offsets_m": [
                    float(value) if np.isfinite(value) else None
                    for value in ground_contact_offsets_m
                ],
                "rejected_segment_counts": rejected_counts,
                "held_bone_counts": held_counts,
                "hand_rejections_by_frame": hand_rejections_by_frame,
                "coordinate_system": "display_x_right_y_forward_z_up_m",
            },
        )
        cache.save(output_path, overwrite=args.overwrite)
        elapsed = time.perf_counter() - started
        LOGGER.info(
            "Saved %s: frames=%d fitted=%d vertices=%d bones=%d "
            "scale=%.5f time=%.2fs",
            output_path,
            frame_count,
            int(np.count_nonzero(present)),
            vertex_count,
            bone_count,
            scale,
            elapsed,
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
