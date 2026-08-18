#!/usr/bin/env python3
"""Fit a cached SMPL mesh sequence to constrained Halpe26 3D joints."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

from rgbd_avatar.avatar import (
    SMPLFitConfig,
    SMPLHandObservation,
    SMPLSequenceCache,
    SMPLSequenceFitter,
    build_smpl_target,
    estimate_smpl_scale,
    load_shape_preset,
    sha256_file,
    smpl_to_display,
)
from rgbd_avatar.depth import GroundPlaneEstimate
from rgbd_avatar.data import load_hand_records
from rgbd_avatar.io import load_json_mapping
from rgbd_avatar.pose import hand_observation_quality
from rgbd_avatar.retargeting import (
    calibrate_halpe_smpl_profile,
    retarget_halpe26_to_smpl,
)
from rgbd_avatar.visualization import (
    load_pose_records,
    parse_pose_layer,
    transform_camera_points,
)


LOGGER = logging.getLogger("fit_smpl_sequence")
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
        default=(
            PROJECT_ROOT
            / "assets/models/smpl/SMPL_NEUTRAL_CLEAN.pkl"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--hands-jsonl",
        type=Path,
        default=None,
        help="Default: hands.jsonl beside poses.jsonl when present.",
    )
    parser.add_argument(
        "--no-hand-targets",
        action="store_true",
        help="Ignore an available Hand21 cache.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace an existing output cache.",
    )
    parser.add_argument(
        "--pose-layer",
        choices=("temporal", "constrained"),
        default="constrained",
    )
    parser.add_argument(
        "--retarget-mode",
        choices=("semantic", "legacy"),
        default="semantic",
        help=(
            "semantic preserves SMPL proportions and transfers Halpe bone "
            "directions; legacy directly matches equally named joints."
        ),
    )
    parser.add_argument("--ground-plane", type=Path, default=None)
    parser.add_argument("--no-ground-alignment", action="store_true")
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument(
        "--spine-pose-weight",
        type=float,
        default=5e-2,
        help=(
            "Regularize the three unobserved SMPL spine rotations; larger "
            "values produce a straighter torso."
        ),
    )
    parser.add_argument(
        "--end-effector-direction-weight",
        type=float,
        default=0.02,
        help=(
            "Weight for scale-invariant hand/foot orientation constraints. "
            "Hand and foot landmark positions never stretch the SMPL mesh."
        ),
    )
    parser.add_argument(
        "--allow-end-effector-articulation",
        action="store_true",
        help=(
            "Allow legacy SMPL hand/foot terminal-joint rotations. By "
            "default these joints are rigid and only their parents rotate."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional prefix limit for a fitting smoke test.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override robust sequence-level metric scale estimation.",
    )
    parser.add_argument(
        "--shape-preset",
        type=Path,
        default=None,
        help="Saved SMPL shape JSON whose betas are fixed for every frame.",
    )
    parser.add_argument(
        "--use-preset-scale",
        action="store_true",
        help=(
            "Use the shape preset's global scale instead of estimating it "
            "from the observed sequence. --scale still takes precedence."
        ),
    )
    return parser.parse_args()


def _load_display_transform(
    poses_path: Path,
    *,
    ground_plane_path: Path | None,
    disabled: bool,
) -> tuple[np.ndarray | None, Path | None]:
    if disabled:
        return None, None
    path = (
        ground_plane_path.expanduser().resolve()
        if ground_plane_path is not None
        else poses_path.parent / "ground_plane.json"
    )
    if not path.is_file():
        if ground_plane_path is not None:
            raise FileNotFoundError(f"Ground plane not found: {path}")
        LOGGER.warning(
            "No ground_plane.json beside poses; using optical display axes."
        )
        return None, None
    payload = load_json_mapping(path)
    estimate = GroundPlaneEstimate.from_mapping(payload)
    return estimate.camera_to_ground_transform(), path


def _resolve_device(requested: str, torch: Any) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable.")
    return requested


def _display_pose_arrays(
    records: list[dict[str, Any]],
    *,
    layer: str,
    camera_to_display_transform: np.ndarray | None,
) -> tuple[list[Any], list[np.ndarray]]:
    poses = []
    display_joints = []
    for record in records:
        pose = parse_pose_layer(record, layer)
        joints = np.full((26, 3), np.nan, dtype=np.float64)
        keep = pose.usable & np.isfinite(pose.joints_camera_m).all(axis=1)
        if np.any(keep):
            joints[keep] = transform_camera_points(
                pose.joints_camera_m[keep],
                camera_to_display_transform,
            )
        poses.append(pose)
        display_joints.append(joints)
    return poses, display_joints


def _load_hand_records(
    path: Path,
    pose_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return load_hand_records(path, pose_records)


def _parse_hand_observations(
    record: dict[str, Any],
    camera_to_display_transform: np.ndarray | None,
) -> tuple[list[SMPLHandObservation], tuple[str, ...]]:
    observations: list[SMPLHandObservation] = []
    rejections: list[str] = []
    hands = record.get("hands")
    if not isinstance(hands, dict):
        return observations, ()
    for side in ("left", "right"):
        hand = hands.get(side)
        pose3d = hand.get("pose3d") if isinstance(hand, dict) else None
        joints_payload = (
            pose3d.get("joints") if isinstance(pose3d, dict) else None
        )
        if not isinstance(joints_payload, list) or len(joints_payload) != 21:
            continue
        joints = np.full((21, 3), np.nan, dtype=np.float64)
        confidence = np.zeros(21, dtype=np.float64)
        valid = np.zeros(21, dtype=bool)
        for index, joint in enumerate(joints_payload):
            xyz = joint.get("xyz_m")
            is_valid = bool(joint.get("valid")) and isinstance(xyz, list)
            if is_valid and len(xyz) == 3:
                joints[index] = np.asarray(xyz, dtype=np.float64)
                confidence[index] = float(joint.get("confidence", 0.0))
                valid[index] = np.isfinite(joints[index]).all()
        quality_ok, reason, _ = hand_observation_quality(joints, valid)
        if not quality_ok:
            rejections.append(f"{side}:{reason}")
            continue
        if np.any(valid):
            joints[valid] = transform_camera_points(
                joints[valid],
                camera_to_display_transform,
            )
            observations.append(
                SMPLHandObservation(
                    side=side,
                    joints_display_m=joints,
                    confidence=confidence,
                    valid=valid,
                )
            )
    return observations, tuple(rejections)


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
            else poses_path.parent / "smpl_sequence.npz"
        )
        if not model_path.is_file():
            raise FileNotFoundError(
                "Chumpy-free SMPL model not found: "
                f"{model_path}. Run scripts/convert_smpl_model.py first."
            )
        if args.use_preset_scale and args.shape_preset is None:
            raise ValueError("--use-preset-scale requires --shape-preset.")
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing SMPL cache: {output_path}. "
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
                hand_records = _load_hand_records(candidate, records)
                LOGGER.info("Loaded Hand21 targets from %s.", candidate)
            elif args.hands_jsonl is not None:
                raise FileNotFoundError(f"Hand cache not found: {candidate}")
            else:
                LOGGER.info("No hands.jsonl found; fitting body targets only.")
        display_transform, ground_path = _load_display_transform(
            poses_path,
            ground_plane_path=args.ground_plane,
            disabled=args.no_ground_alignment,
        )
        poses, display_joints = _display_pose_arrays(
            records,
            layer=args.pose_layer,
            camera_to_display_transform=display_transform,
        )

        import smplx
        import torch

        device = _resolve_device(args.device, torch)
        model = smplx.SMPL(
            str(model_path),
            batch_size=1,
            create_betas=False,
            create_body_pose=False,
            create_global_orient=False,
            create_transl=False,
        ).to(device).eval()
        model_digest = sha256_file(model_path)
        shape_preset_path: Path | None = None
        shape_preset_digest: str | None = None
        shape_preset = None
        betas = np.zeros(int(model.num_betas), dtype=np.float32)
        if args.shape_preset is not None:
            shape_preset_path = args.shape_preset.expanduser().resolve()
            shape_preset = load_shape_preset(shape_preset_path)
            if shape_preset.model_sha256 != model_digest:
                raise ValueError(
                    "Shape preset was created for a different SMPL model."
                )
            if shape_preset.betas.shape != betas.shape:
                raise ValueError(
                    f"Shape preset has {len(shape_preset.betas)} betas, "
                    f"model expects {len(betas)}."
                )
            betas = shape_preset.betas.copy()
            shape_preset_digest = sha256_file(shape_preset_path)
            LOGGER.info(
                "Loaded fixed SMPL shape from %s: betas=%s scale=%.5f",
                shape_preset_path,
                np.round(betas, 3).tolist(),
                shape_preset.scale,
            )
        betas_tensor = torch.as_tensor(
            betas, dtype=torch.float32, device=device
        ).reshape(1, -1)
        with torch.no_grad():
            rest = model(
                betas=betas_tensor,
                body_pose=torch.zeros((1, 69), device=device),
                global_orient=torch.zeros((1, 3), device=device),
                return_verts=False,
            )
        estimated_sequence_scale: float | None = None
        try:
            estimated_sequence_scale = estimate_smpl_scale(
                display_joints,
                [pose.usable for pose in poses],
                rest.joints[0, :24].cpu().numpy(),
            )
        except ValueError as error:
            if args.scale is None and not args.use_preset_scale:
                raise
            LOGGER.warning("Could not estimate sequence scale: %s", error)

        if args.scale is not None:
            scale = float(args.scale)
            scale_source = "command_line"
        elif args.use_preset_scale:
            assert shape_preset is not None
            scale = float(shape_preset.scale)
            scale_source = "shape_preset"
        else:
            assert estimated_sequence_scale is not None
            scale = estimated_sequence_scale
            scale_source = "sequence_bone_estimate"
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("SMPL scale must be finite and positive.")
        if (
            estimated_sequence_scale is not None
            and abs(scale / estimated_sequence_scale - 1.0) > 0.03
        ):
            LOGGER.warning(
                "Selected SMPL scale %.5f differs from the robust sequence "
                "estimate %.5f by %.1f%%; this can force compensating limb "
                "rotations.",
                scale,
                estimated_sequence_scale,
                100.0 * (scale / estimated_sequence_scale - 1.0),
            )
        config = SMPLFitConfig(
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            spine_pose_weight=args.spine_pose_weight,
            end_effector_direction_weight=(
                args.end_effector_direction_weight
            ),
            rigid_end_effectors=not args.allow_end_effector_articulation,
        )
        fitter = SMPLSequenceFitter(
            model,
            scale=scale,
            device=device,
            config=config,
            betas=betas,
        )
        retarget_profile = None
        rest_joints_display_m = None
        if args.retarget_mode == "semantic":
            retarget_profile = calibrate_halpe_smpl_profile(
                display_joints,
                [pose.confidence for pose in poses],
                [pose.usable for pose in poses],
                [pose.predicted for pose in poses],
            )
            rest_joints_display_m = smpl_to_display(
                rest.joints[0].cpu().numpy()
            ) * scale
            LOGGER.info(
                "Using semantic Halpe26->SMPL retargeting with fixed avatar "
                "proportions and robust per-segment gates."
            )

        frame_count = len(records)
        vertex_count = int(model.v_template.shape[0])
        vertices = np.full(
            (frame_count, vertex_count, 3),
            np.nan,
            dtype=np.float32,
        )
        joints = np.full((frame_count, 24, 3), np.nan, dtype=np.float32)
        body_pose = np.full((frame_count, 69), np.nan, dtype=np.float32)
        global_orient = np.full((frame_count, 3), np.nan, dtype=np.float32)
        translation = np.full((frame_count, 3), np.nan, dtype=np.float32)
        present = np.zeros(frame_count, dtype=bool)
        target_counts = np.zeros(frame_count, dtype=np.int16)
        error_mean = np.full(frame_count, np.nan, dtype=np.float32)
        error_p95 = np.full(frame_count, np.nan, dtype=np.float32)
        error_max = np.full(frame_count, np.nan, dtype=np.float32)

        previous_fit = None
        hand_rejection_counts: dict[str, int] = {}
        hand_rejections_by_frame: list[dict[str, Any]] = []
        rejected_segment_counts: dict[str, int] = {}
        soft_segment_counts: dict[str, int] = {}
        retarget_rejections_by_frame: list[dict[str, Any]] = []
        retarget_soft_segments_by_frame: list[dict[str, Any]] = []
        started = time.perf_counter()
        for index, (record, pose, frame_joints) in enumerate(
            zip(records, poses, display_joints)
        ):
            hand_observations, hand_rejections = (
                _parse_hand_observations(
                    hand_records[index],
                    display_transform,
                )
                if hand_records is not None
                else ([], ())
            )
            if hand_rejections:
                hand_rejections_by_frame.append(
                    {
                        "frame_index": int(record["frame_index"]),
                        "hands": list(hand_rejections),
                    }
                )
                for rejection in hand_rejections:
                    hand_rejection_counts[rejection] = (
                        hand_rejection_counts.get(rejection, 0) + 1
                    )
            retargeted_body = None
            if retarget_profile is not None:
                assert rest_joints_display_m is not None
                retargeted_body = retarget_halpe26_to_smpl(
                    frame_joints,
                    pose.confidence,
                    pose.usable,
                    pose.predicted,
                    rest_joints_display_m=rest_joints_display_m,
                    profile=retarget_profile,
                    minimum_weight=config.minimum_joint_weight,
                    predicted_weight_scale=config.predicted_weight_scale,
                )
                if np.any(pose.usable) and retargeted_body.rejected_segments:
                    retarget_rejections_by_frame.append(
                        {
                            "frame_index": int(record["frame_index"]),
                            "segments": list(
                                retargeted_body.rejected_segments
                            ),
                        }
                    )
                    for segment in retargeted_body.rejected_segments:
                        rejected_segment_counts[segment] = (
                            rejected_segment_counts.get(segment, 0) + 1
                        )
                if np.any(pose.usable) and retargeted_body.soft_segments:
                    retarget_soft_segments_by_frame.append(
                        {
                            "frame_index": int(record["frame_index"]),
                            "segments": list(retargeted_body.soft_segments),
                        }
                    )
                    for segment in retargeted_body.soft_segments:
                        soft_segment_counts[segment] = (
                            soft_segment_counts.get(segment, 0) + 1
                        )
            target = build_smpl_target(
                frame_joints,
                pose.confidence,
                pose.usable,
                pose.predicted,
                config=config,
                hand_observations=hand_observations,
                retargeted_body=retargeted_body,
            )
            target_counts[index] = target.count
            if target.count < config.minimum_target_count:
                previous_fit = None
                LOGGER.info(
                    "[%d/%d] frame=%s skipped targets=%d directions=%d "
                    "status=%s",
                    index + 1,
                    frame_count,
                    record["frame_index"],
                    target.count,
                    target.direction_count,
                    record.get("status"),
                )
                continue
            fit = fitter.fit(target, previous_fit)
            previous_fit = fit
            present[index] = True
            vertices[index] = fit.vertices_display_m
            joints[index] = fit.joints_display_m
            body_pose[index] = fit.body_pose
            global_orient[index] = fit.global_orient
            translation[index] = fit.translation_native_m
            error_mean[index] = fit.error_mean_m
            error_p95[index] = fit.error_p95_m
            error_max[index] = fit.error_max_m
            LOGGER.info(
                "[%d/%d] frame=%s targets=%d directions=%d "
                "error_mean=%.3f m p95=%.3f m max=%.3f m iterations=%d",
                index + 1,
                frame_count,
                record["frame_index"],
                fit.target_count,
                target.direction_count,
                fit.error_mean_m,
                fit.error_p95_m,
                fit.error_max_m,
                fit.iterations,
            )

        cache = SMPLSequenceCache(
            frame_indices=np.asarray(
                [record["frame_index"] for record in records],
                dtype=np.int64,
            ),
            present=present,
            vertices_display_m=vertices,
            joints_display_m=joints,
            faces=np.asarray(model.faces, dtype=np.int32),
            body_pose=body_pose,
            global_orient=global_orient,
            translation_native_m=translation,
            target_counts=target_counts,
            error_mean_m=error_mean,
            error_p95_m=error_p95,
            error_max_m=error_max,
            scale=scale,
            metadata={
                "poses_jsonl": str(poses_path),
                "poses_sha256": sha256_file(poses_path),
                "hands_jsonl": str(hands_path) if hands_path else None,
                "hands_sha256": (
                    sha256_file(hands_path) if hands_path else None
                ),
                "model": str(model_path),
                "model_sha256": model_digest,
                "shape_preset": (
                    str(shape_preset_path) if shape_preset_path else None
                ),
                "shape_preset_sha256": shape_preset_digest,
                "betas": betas.tolist(),
                "scale_source": scale_source,
                "estimated_sequence_scale": estimated_sequence_scale,
                "retarget_mode": args.retarget_mode,
                "retarget_profile": (
                    retarget_profile.to_mapping()
                    if retarget_profile is not None
                    else None
                ),
                "retarget_rejected_segment_counts": rejected_segment_counts,
                "retarget_soft_segment_counts": soft_segment_counts,
                "retarget_rejections_by_frame": retarget_rejections_by_frame,
                "retarget_soft_segments_by_frame": (
                    retarget_soft_segments_by_frame
                ),
                "hand_rejection_counts": hand_rejection_counts,
                "hand_rejections_by_frame": hand_rejections_by_frame,
                "pose_layer": args.pose_layer,
                "ground_plane": str(ground_path) if ground_path else None,
                "ground_plane_sha256": (
                    sha256_file(ground_path) if ground_path else None
                ),
                "camera_to_display_transform": (
                    display_transform.tolist()
                    if display_transform is not None
                    else None
                ),
                "coordinate_system": "display_x_right_y_forward_z_up_m",
                "fit_config": asdict(config),
            },
        )
        cache.save(output_path)
        elapsed_s = time.perf_counter() - started
        if rejected_segment_counts:
            LOGGER.info(
                "Semantic retarget rejected anomalous observations: %s",
                dict(sorted(rejected_segment_counts.items())),
            )
        if soft_segment_counts:
            LOGGER.info(
                "Semantic retarget kept statistical length outliers as "
                "low-weight directions: %s",
                dict(sorted(soft_segment_counts.items())),
            )
        if hand_rejection_counts:
            LOGGER.info(
                "Rejected degenerate Hand21 observations: %s",
                dict(sorted(hand_rejection_counts.items())),
            )
        active_errors = error_mean[present]
        LOGGER.info(
            "Saved %s: frames=%d fitted=%d scale=%.5f "
            "mean_error=%.3f m p95_frame_error=%.3f m time=%.2fs",
            output_path,
            frame_count,
            int(np.count_nonzero(present)),
            scale,
            float(np.mean(active_errors)),
            float(np.percentile(active_errors, 95)),
            elapsed_s,
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
