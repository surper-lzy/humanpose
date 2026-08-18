#!/usr/bin/env python3
"""Estimate an indoor ground plane from saved RGB-D sequence frames."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import (
    GroundPlaneConfig,
    depth_to_organized_point_cloud,
    fit_ground_plane_ransac,
    load_depth_m,
    sample_ground_candidates,
)
from rgbd_avatar.io import (
    atomic_write_json,
    load_json_mapping,
    load_yaml_mapping,
    resolve_path,
)
from rgbd_avatar.visualization import load_pose_records, parse_pose_layer


LOGGER = logging.getLogger("estimate_ground_plane")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FOOT_IDS = np.asarray((20, 21, 22, 23, 24, 25), dtype=np.int64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/sequences/4_pointcloud_exit_gate",
    )
    parser.add_argument(
        "--ground-config",
        type=Path,
        default=PROJECT_ROOT / "configs/ground.yaml",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <results-dir>/ground_plane.json.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=16,
        help="Uniformly sampled depth frames used for plane fitting.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return load_json_mapping(path)


def load_ground_config(path: Path) -> GroundPlaneConfig:
    payload = load_yaml_mapping(path)
    return GroundPlaneConfig.from_mapping(payload.get("ground"))


def resolve_source(path_value: str, manifest_path: Path) -> Path:
    return resolve_path(path_value, relative_to=manifest_path.parent)


def selected_frame_indices(
    frame_count: int,
    max_frames: int,
) -> np.ndarray:
    if frame_count <= 0:
        raise ValueError("Manifest contains no frames.")
    if max_frames <= 0:
        raise ValueError("--max-frames must be positive.")
    count = min(frame_count, max_frames)
    return np.unique(
        np.linspace(0, frame_count - 1, count, dtype=np.int64)
    )


def skeleton_support_statistics(
    records: list[dict[str, Any]],
    camera_to_ground: np.ndarray,
) -> dict[str, Any]:
    heights: list[float] = []
    for record in records:
        pose = parse_pose_layer(record, "constrained")
        valid_ids = FOOT_IDS[pose.usable[FOOT_IDS]]
        if not len(valid_ids):
            continue
        points = pose.joints_camera_m[valid_ids]
        finite = np.isfinite(points).all(axis=1)
        if not np.any(finite):
            continue
        homogeneous = np.column_stack(
            (points[finite], np.ones(np.count_nonzero(finite)))
        )
        world = homogeneous @ camera_to_ground.T
        heights.append(float(np.min(world[:, 2])))
    if not heights:
        return {
            "frame_count": 0,
            "minimum_m": None,
            "median_m": None,
            "p95_absolute_m": None,
            "maximum_m": None,
        }
    values = np.asarray(heights, dtype=np.float64)
    return {
        "frame_count": len(values),
        "minimum_m": float(np.min(values)),
        "median_m": float(np.median(values)),
        "p95_absolute_m": float(np.percentile(np.abs(values), 95)),
        "maximum_m": float(np.max(values)),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    results_dir = args.results_dir.expanduser().resolve()
    manifest_path = results_dir / "manifest.json"
    poses_path = results_dir / "poses.jsonl"
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else results_dir / "ground_plane.json"
    )
    manifest = load_json(manifest_path)
    records = load_pose_records(poses_path)
    config = load_ground_config(args.ground_config)
    camera = manifest.get("camera")
    frames = manifest.get("frames")
    if not isinstance(camera, dict) or not isinstance(frames, list):
        raise ValueError(
            "Manifest must contain camera calibration and frame sources."
        )
    intrinsics = CameraIntrinsics(
        **camera["intrinsics"],
        width=int(camera["depth_width"]),
        height=int(camera["depth_height"]),
    )
    records_by_frame = {
        int(record["frame_index"]): record for record in records
    }

    candidates: list[np.ndarray] = []
    source_frames: list[dict[str, Any]] = []
    for manifest_index in selected_frame_indices(
        len(frames),
        args.max_frames,
    ):
        frame = frames[int(manifest_index)]
        frame_index = int(frame["frame_index"])
        depth_path = resolve_source(frame["depth"], manifest_path)
        depth_m = load_depth_m(
            depth_path,
            float(camera["depth_scale"]),
        )
        organized = depth_to_organized_point_cloud(
            depth_m,
            intrinsics,
            min_depth_m=config.min_depth_m,
            max_depth_m=config.max_depth_m,
        )
        record = records_by_frame.get(frame_index, {})
        pose2d = record.get("pose2d")
        bbox = (
            np.asarray(pose2d.get("bbox_xyxy"), dtype=np.float64)
            if isinstance(pose2d, dict)
            else None
        )
        frame_candidates = sample_ground_candidates(
            organized,
            config,
            person_bbox_xyxy=bbox,
        )
        if not len(frame_candidates):
            continue
        candidates.append(frame_candidates)
        source_frames.append(
            {
                "frame_index": frame_index,
                "timestamp_raw": frame["timestamp_raw"],
                "candidate_count": len(frame_candidates),
                "depth": str(depth_path),
            }
        )
        LOGGER.info(
            "frame=%d ground_candidates=%d",
            frame_index,
            len(frame_candidates),
        )

    if not candidates:
        raise RuntimeError("No ground candidates were sampled.")
    estimate = fit_ground_plane_ransac(
        np.concatenate(candidates, axis=0),
        config,
        source_frame_count=len(source_frames),
    )
    payload = estimate.to_dict()
    payload.update(
        {
            "method": "lower_image_person_excluded_constrained_ransac",
            "config": config.to_dict(),
            "results_directory": str(results_dir),
            "source_frames": source_frames,
            "skeleton_support_validation": skeleton_support_statistics(
                records,
                estimate.camera_to_ground_transform(),
            ),
        }
    )
    atomic_write_json(output_path, payload)
    LOGGER.info(
        "Ground: normal=%s height=%.4f m tilt=%.2f deg "
        "inliers=%d/%d residual_p95=%.4f m",
        np.array2string(estimate.normal_camera, precision=5),
        estimate.camera_height_m,
        estimate.tilt_from_camera_up_deg,
        estimate.inlier_count,
        estimate.candidate_count,
        estimate.residual_p95_m,
    )
    LOGGER.info("Saved ground calibration: %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
