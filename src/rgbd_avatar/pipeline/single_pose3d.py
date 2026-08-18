#!/usr/bin/env python3
"""Recover one metric 3D Halpe26 pose from aligned RGB and depth."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d
import yaml

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import (
    PointCloudRecoveryConfig,
    depth_to_organized_point_cloud,
    load_depth_m,
    recover_pose3d,
    recover_pose3d_from_point_cloud,
)
from rgbd_avatar.pose import RTMPoseBackend, RTMPoseBackendConfig
from rgbd_avatar.visualization import draw_pose_depths, save_pose3d_scene

LOGGER = logging.getLogger("test_pose_3d_single")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rgb",
        type=Path,
        default=PROJECT_ROOT.parent
        / "data/1/20260730_145911656_r.png",
    )
    parser.add_argument(
        "--depth",
        type=Path,
        default=PROJECT_ROOT.parent
        / "data/1/20260730_145911656_d.pgm",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PROJECT_ROOT / "configs/camera.yaml",
    )
    parser.add_argument(
        "--pose-config",
        type=Path,
        default=PROJECT_ROOT / "configs/pose.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/pose3d",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--recovery-method",
        choices=("window_median", "pointcloud_cluster"),
        default=None,
        help="Override camera.depth_recovery.method.",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "whole_image"),
        default=None,
    )
    return parser.parse_args()


def resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def write_metric_point_cloud(
    output_path: Path,
    points_m: np.ndarray,
    rgb: np.ndarray,
) -> int:
    valid = np.isfinite(points_m).all(axis=2)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points_m[valid])
    cloud.colors = o3d.utility.Vector3dVector(
        rgb[valid].astype(np.float64) / 255.0
    )
    if not o3d.io.write_point_cloud(
        str(output_path), cloud, write_ascii=False, compressed=False
    ):
        raise RuntimeError(f"Failed to write point cloud: {output_path}")
    return int(np.count_nonzero(valid))


def compare_with_exported_pcd(
    pcd_path: Path,
    points_m: np.ndarray,
    exported_unit_scale: float,
) -> dict | None:
    if not pcd_path.is_file():
        return None

    valid = np.isfinite(points_m).all(axis=2)
    generated = points_m[valid].astype(np.float64)
    exported = np.asarray(
        o3d.io.read_point_cloud(str(pcd_path)).points
    ) * exported_unit_scale
    comparison = {
        "source_pcd": str(pcd_path),
        "generated_point_count": int(len(generated)),
        "exported_point_count": int(len(exported)),
        "pointwise_comparable": len(generated) == len(exported),
    }
    if len(generated) == len(exported):
        errors = np.linalg.norm(generated - exported, axis=1)
        comparison["rmse_m"] = float(
            np.sqrt(np.mean(np.square(errors)))
        )
        comparison["max_error_m"] = float(np.max(errors))
    return comparison


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    rgb_path = args.rgb.expanduser().resolve()
    depth_path = args.depth.expanduser().resolve()
    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"OpenCV failed to read RGB image: {rgb_path}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    camera = load_yaml(args.camera_config)["camera"]
    pose_config = load_yaml(args.pose_config)["pose"]
    depth_recovery_config = camera.get("depth_recovery", {})
    if not isinstance(depth_recovery_config, dict):
        raise ValueError("camera.depth_recovery must be a YAML mapping.")
    recovery_method = (
        args.recovery_method
        or depth_recovery_config.get("method", "window_median")
    )
    if recovery_method not in ("window_median", "pointcloud_cluster"):
        raise ValueError(
            "camera.depth_recovery.method must be window_median or "
            "pointcloud_cluster."
        )
    pointcloud_config = PointCloudRecoveryConfig.from_mapping(
        depth_recovery_config.get("pointcloud_cluster")
    )
    intrinsics = CameraIntrinsics(
        **camera["intrinsics"],
        width=int(camera["depth_width"]),
        height=int(camera["depth_height"]),
    )
    depth_m = load_depth_m(depth_path, float(camera["depth_scale"]))
    if depth_m.shape != rgb.shape[:2]:
        raise ValueError(
            "Aligned RGB/depth shapes differ: "
            f"RGB={rgb.shape[:2]}, depth={depth_m.shape}."
        )

    backend = RTMPoseBackend(
        RTMPoseBackendConfig(
            model_config=resolve_project_path(pose_config["model_config"]),
            model_checkpoint=resolve_project_path(
                pose_config["model_checkpoint"]
            ),
            detector=args.detector or pose_config["detector"],
            model_cache_dir=resolve_project_path(
                pose_config["model_cache_dir"]
            ),
            device=args.device or pose_config["device"],
            bbox_threshold=float(pose_config["bbox_threshold"]),
            keypoint_threshold=float(pose_config["keypoint_threshold"]),
            min_valid_keypoints=int(pose_config["min_valid_keypoints"]),
            min_mean_keypoint_score=float(
                pose_config["min_mean_keypoint_score"]
            ),
        )
    )
    poses = backend.infer(rgb_path)
    if not poses:
        LOGGER.info("No person passed the 2D pose quality gates")
        return 0
    pose2d = poses[0]

    points_m = depth_to_organized_point_cloud(
        depth_m=depth_m,
        intrinsics=intrinsics,
        min_depth_m=float(camera["min_depth_m"]),
        max_depth_m=float(camera["max_depth_m"]),
    )
    if recovery_method == "pointcloud_cluster":
        recovery_result = recover_pose3d_from_point_cloud(
            pose2d=pose2d,
            organized_points_m=points_m,
            intrinsics=intrinsics,
            keypoint_threshold=float(pose_config["keypoint_threshold"]),
            config=pointcloud_config,
        )
        pose3d = recovery_result.pose3d
        recovery_diagnostics = recovery_result.diagnostics
    else:
        pose3d = recover_pose3d(
            pose2d=pose2d,
            depth_m=depth_m,
            intrinsics=intrinsics,
            keypoint_threshold=float(pose_config["keypoint_threshold"]),
            radius=int(camera["depth_window_radius"]),
            min_depth_m=float(camera["min_depth_m"]),
            max_depth_m=float(camera["max_depth_m"]),
        )
        recovery_diagnostics = {
            "method": "window_median",
            "parameters": {
                "radius_px": int(camera["depth_window_radius"]),
                "min_depth_m": float(camera["min_depth_m"]),
                "max_depth_m": float(camera["max_depth_m"]),
            },
            "valid_joint_count": int(np.count_nonzero(pose3d.valid)),
        }

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = rgb_path.stem.removesuffix("_r")
    json_path = output_dir / f"{stem}_pose3d.json"
    overlay_path = output_dir / f"{stem}_depth_overlay.png"
    scene_path = output_dir / f"{stem}_scene3d.png"
    cloud_path = output_dir / f"{stem}_metric_cloud.ply"

    overlay = draw_pose_depths(
        rgb_bgr,
        pose2d,
        pose3d,
        score_threshold=float(pose_config["keypoint_threshold"]),
    )
    if not cv2.imwrite(str(overlay_path), overlay):
        raise RuntimeError(f"Failed to write overlay: {overlay_path}")
    save_pose3d_scene(scene_path, points_m, rgb, pose2d, pose3d)
    point_count = write_metric_point_cloud(cloud_path, points_m, rgb)
    exported_pcd_path = rgb_path.with_name(f"{stem}_t.pcd")
    pcd_comparison = compare_with_exported_pcd(
        exported_pcd_path,
        points_m,
        exported_unit_scale=float(camera["depth_scale"]),
    )

    result = {
        "source_rgb": str(rgb_path),
        "source_depth": str(depth_path),
        "depth_scale": float(camera["depth_scale"]),
        "intrinsics": camera["intrinsics"],
        "pose2d": pose2d.to_dict(
            score_threshold=float(pose_config["keypoint_threshold"])
        ),
        "pose3d": pose3d.to_dict(),
        "depth_recovery": recovery_diagnostics,
        "metric_point_count": point_count,
        "exported_pcd_comparison": pcd_comparison,
    }
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
        file.write("\n")

    LOGGER.info(
        "Recovered %d/26 metric joints and %d metric point-cloud points",
        np.count_nonzero(pose3d.valid),
        point_count,
    )
    LOGGER.info("Saved 3D pose JSON: %s", json_path)
    LOGGER.info("Saved depth overlay: %s", overlay_path)
    LOGGER.info("Saved 3D diagnostic: %s", scene_path)
    LOGGER.info("Saved metric point cloud: %s", cloud_path)
    if pcd_comparison is not None:
        if pcd_comparison["pointwise_comparable"]:
            LOGGER.info(
                "Generated-vs-exported PCD: RMSE=%.9f m, max=%.9f m",
                pcd_comparison["rmse_m"],
                pcd_comparison["max_error_m"],
            )
        else:
            LOGGER.warning(
                "Generated/exported point counts differ: %d vs %d",
                pcd_comparison["generated_point_count"],
                pcd_comparison["exported_point_count"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
