#!/usr/bin/env python3
"""Detect and lift Hand21 landmarks for an existing Halpe26 RGB-D run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from rgbd_avatar.avatar import sha256_file
from rgbd_avatar.depth import load_depth_m
from rgbd_avatar.io import (
    atomic_write_json,
    atomic_write_jsonl,
    load_camera_config,
)
from rgbd_avatar.pose import (
    HAND_TIP_INDICES,
    Pose2D,
    RTMPoseHandBackend,
    RTMPoseHandBackendConfig,
    recover_hand_pose3d,
)
from rgbd_avatar.visualization import load_pose_records, resolve_frame_sources


LOGGER = logging.getLogger("extract_hand_pose_sequence")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/sequences/4_pointcloud_exit_gate",
    )
    parser.add_argument("--poses-jsonl", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--model-config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "assets/models/rtmpose_hand/"
            "rtmpose-m_8xb256-210e_hand5-256x256.py"
        ),
    )
    parser.add_argument(
        "--model-checkpoint",
        type=Path,
        default=(
            PROJECT_ROOT
            / "assets/models/rtmpose_hand/"
            "rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-"
            "74fb594_20230320.pth"
        ),
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PROJECT_ROOT / "configs/camera.yaml",
    )
    parser.add_argument("--sequence-dir", type=Path, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--body-keypoint-threshold", type=float, default=0.3)
    parser.add_argument("--hand-keypoint-threshold", type=float, default=0.10)
    parser.add_argument("--minimum-valid-keypoints", type=int, default=8)
    parser.add_argument("--minimum-mean-score", type=float, default=0.12)
    parser.add_argument("--depth-radius", type=int, default=2)
    parser.add_argument("--minimum-depth-confidence", type=float, default=0.20)
    parser.add_argument("--max-wrist-depth-delta-m", type=float, default=0.12)
    parser.add_argument(
        "--disable-topology-depth-gate",
        action="store_true",
        help="Keep raw per-joint depths without Hand21 bone-depth gating.",
    )
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_body_pose(record: dict[str, Any]) -> Pose2D | None:
    payload = record.get("pose2d")
    if not isinstance(payload, dict):
        return None
    joints = payload.get("keypoints")
    if not isinstance(joints, list) or len(joints) != 26:
        return None
    return Pose2D(
        keypoints=np.asarray(
            [[joint["x"], joint["y"]] for joint in joints],
            dtype=np.float32,
        ),
        scores=np.asarray(
            [joint["confidence"] for joint in joints],
            dtype=np.float32,
        ),
        bbox_xyxy=np.asarray(payload["bbox_xyxy"], dtype=np.float32),
        bbox_score=float(payload["bbox_score"]),
    )


def _body_wrist_anchor(
    record: dict[str, Any],
    side: str,
) -> np.ndarray | None:
    """Use the displayed constrained wrist as the Hand21 root anchor."""

    payload = record.get("pose3d_constrained")
    joints = payload.get("joints") if isinstance(payload, dict) else None
    if not isinstance(joints, list) or len(joints) != 26:
        return None
    wrist_index = 9 if side == "left" else 10
    xyz = joints[wrist_index].get("xyz_m")
    if not isinstance(xyz, list) or len(xyz) != 3:
        return None
    anchor = np.asarray(xyz, dtype=np.float32)
    if not np.isfinite(anchor).all() or anchor[2] <= 0:
        return None
    return anchor


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        if not 0.0 <= args.minimum_depth_confidence <= 1.0:
            raise ValueError("--minimum-depth-confidence must be in [0, 1].")
        if args.max_wrist_depth_delta_m <= 0.0:
            raise ValueError("--max-wrist-depth-delta-m must be positive.")
        poses_path = (
            args.poses_jsonl.expanduser().resolve()
            if args.poses_jsonl is not None
            else args.results_dir.expanduser().resolve() / "poses.jsonl"
        )
        output_path = (
            args.output.expanduser().resolve()
            if args.output is not None
            else poses_path.parent / "hands.jsonl"
        )
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite {output_path}; pass --overwrite."
            )
        records = load_pose_records(poses_path)
        if args.max_frames is not None:
            if args.max_frames <= 0:
                raise ValueError("--max-frames must be positive.")
            records = records[: args.max_frames]
        camera, intrinsics = load_camera_config(args.camera_config)
        backend = RTMPoseHandBackend(
            RTMPoseHandBackendConfig(
                model_config=args.model_config,
                model_checkpoint=args.model_checkpoint,
                device=args.device,
                body_keypoint_threshold=args.body_keypoint_threshold,
                hand_keypoint_threshold=args.hand_keypoint_threshold,
                minimum_valid_keypoints=args.minimum_valid_keypoints,
                minimum_mean_score=args.minimum_mean_score,
            )
        )

        output_records: list[dict[str, Any]] = []
        detected_counts = {"left": 0, "right": 0}
        lifted_tip_counts = {"left": 0, "right": 0}
        started = time.perf_counter()
        for index, record in enumerate(records):
            body_pose = _parse_body_pose(record)
            hand_payload: dict[str, Any] = {}
            if body_pose is not None:
                rgb_path, depth_path = resolve_frame_sources(
                    record,
                    poses_path,
                    args.sequence_dir,
                )
                image = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Failed to read RGB image: {rgb_path}")
                depth = load_depth_m(depth_path, float(camera["depth_scale"]))
                if image.shape[:2] != depth.shape:
                    raise ValueError(
                        f"RGB/depth shape mismatch at frame {record['frame_index']}."
                    )
                detected = backend.infer(image, body_pose)
                for side, pose2d in detected.items():
                    wrist_anchor = _body_wrist_anchor(record, side)
                    if wrist_anchor is None:
                        continue
                    pose3d = recover_hand_pose3d(
                        pose2d,
                        depth,
                        intrinsics,
                        keypoint_threshold=args.hand_keypoint_threshold,
                        radius=args.depth_radius,
                        min_depth_m=float(camera["min_depth_m"]),
                        max_depth_m=float(camera["max_depth_m"]),
                        anchor_depth_m=float(wrist_anchor[2]),
                        max_anchor_delta_m=args.max_wrist_depth_delta_m,
                        minimum_sample_confidence=(
                            args.minimum_depth_confidence
                        ),
                        topology_depth_gate=(
                            not args.disable_topology_depth_gate
                        ),
                        anchor_point_m=wrist_anchor,
                    )
                    valid_tips = int(
                        np.count_nonzero(pose3d.valid[list(HAND_TIP_INDICES)])
                    )
                    detected_counts[side] += 1
                    lifted_tip_counts[side] += valid_tips
                    hand_payload[side] = {
                        "pose2d": pose2d.to_dict(
                            score_threshold=args.hand_keypoint_threshold
                        ),
                        "pose3d": pose3d.to_dict(),
                        "valid_tip_count": valid_tips,
                    }
            output_records.append(
                {
                    "schema_version": 1,
                    "frame_index": int(record["frame_index"]),
                    "timestamp_raw": record["timestamp_raw"],
                    "hands": hand_payload,
                }
            )
            LOGGER.info(
                "[%d/%d] frame=%s hands=%s",
                index + 1,
                len(records),
                record["frame_index"],
                ",".join(sorted(hand_payload)) or "none",
            )

        atomic_write_jsonl(output_path, output_records)
        manifest = {
            "schema_version": 1,
            "poses_jsonl": str(poses_path),
            "poses_sha256": sha256_file(poses_path),
            "hands_jsonl": str(output_path),
            "hands_sha256": sha256_file(output_path),
            "model_config": str(args.model_config.expanduser().resolve()),
            "model_checkpoint": str(
                args.model_checkpoint.expanduser().resolve()
            ),
            "model_sha256": sha256_file(args.model_checkpoint),
            "hand_keypoint_threshold": args.hand_keypoint_threshold,
            "depth_radius": args.depth_radius,
            "minimum_depth_confidence": args.minimum_depth_confidence,
            "max_wrist_depth_delta_m": args.max_wrist_depth_delta_m,
            "topology_depth_gate": not args.disable_topology_depth_gate,
            "frames": len(records),
            "detected_frames": detected_counts,
            "lifted_tip_counts": lifted_tip_counts,
        }
        manifest_path = output_path.with_name("hands_manifest.json")
        atomic_write_json(manifest_path, manifest)
        LOGGER.info(
            "Saved %s and %s in %.2fs; detected=%s lifted_tips=%s",
            output_path,
            manifest_path,
            time.perf_counter() - started,
            detected_counts,
            lifted_tip_counts,
        )
        return 0
    except Exception as error:
        LOGGER.error("%s", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
