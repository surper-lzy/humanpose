#!/usr/bin/env python3
"""Visualize high-confidence 3D bone-length violations on RGB and depth."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rgbd_avatar.depth import load_depth_m
from rgbd_avatar.io import (
    atomic_write_json,
    load_json_mapping,
    load_jsonl_objects,
    load_yaml_mapping,
)
from rgbd_avatar.pose import HALPE26_LINKS


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        type=Path,
        default=PROJECT_ROOT / "outputs/sequences/4/summary.json",
    )
    parser.add_argument(
        "--poses",
        type=Path,
        default=PROJECT_ROOT / "outputs/sequences/4/poses.jsonl",
    )
    parser.add_argument(
        "--tracking-config",
        type=Path,
        default=PROJECT_ROOT / "configs/tracking.yaml",
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=PROJECT_ROOT / "configs/camera.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs/sequences/4/bone_violation_diagnostics"
        ),
    )
    parser.add_argument("--top-frames", type=int, default=3)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return load_json_mapping(path)


def load_yaml(path: Path) -> dict[str, Any]:
    return load_yaml_mapping(path)


def find_anchor_violations(
    rows: list[dict[str, Any]],
    bones: list[dict[str, Any]],
    anchor_confidence: float,
) -> list[dict[str, Any]]:
    """Retrospectively compare every frame with the final sequence prior."""

    violations: list[dict[str, Any]] = []
    for row in rows:
        joints = row["pose3d_temporal"]["joints"]
        for bone in bones:
            if not bone["ready"]:
                continue
            start = int(bone["start_id"])
            end = int(bone["end_id"])
            start_joint = joints[start]
            end_joint = joints[end]
            if (
                not start_joint["observed"]
                or not end_joint["observed"]
                or start_joint["confidence"] < anchor_confidence
                or end_joint["confidence"] < anchor_confidence
                or start_joint["xyz_m"] is None
                or end_joint["xyz_m"] is None
            ):
                continue
            start_xyz = np.asarray(start_joint["xyz_m"], dtype=np.float64)
            end_xyz = np.asarray(end_joint["xyz_m"], dtype=np.float64)
            actual_length_m = float(np.linalg.norm(end_xyz - start_xyz))
            target_length_m = float(bone["target_length_m"])
            relative_error = (
                abs(actual_length_m - target_length_m) / target_length_m
            )
            tolerance_ratio = float(bone["tolerance_ratio"])
            relative_violation = max(
                0.0, relative_error - tolerance_ratio
            )
            if relative_violation <= 0:
                continue
            violations.append(
                {
                    "frame_index": int(row["frame_index"]),
                    "timestamp_raw": row["timestamp_raw"],
                    "rgb_path": row["sources"]["rgb"],
                    "depth_path": row["sources"]["depth"],
                    "start_id": start,
                    "start_name": bone["start_name"],
                    "end_id": end,
                    "end_name": bone["end_name"],
                    "actual_length_m": actual_length_m,
                    "target_length_m": target_length_m,
                    "relative_error": relative_error,
                    "tolerance_ratio": tolerance_ratio,
                    "relative_violation": relative_violation,
                    "start_confidence": float(
                        start_joint["confidence"]
                    ),
                    "end_confidence": float(end_joint["confidence"]),
                    "start_depth_m": float(start_xyz[2]),
                    "end_depth_m": float(end_xyz[2]),
                }
            )
    violations.sort(key=lambda item: item["relative_violation"], reverse=True)
    return violations


def depth_colormap(
    depth_m: np.ndarray,
    min_depth_m: float,
    max_depth_m: float,
) -> np.ndarray:
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= min_depth_m)
        & (depth_m <= max_depth_m)
    )
    normalized = np.zeros(depth_m.shape, dtype=np.uint8)
    normalized[valid] = np.clip(
        255.0
        * (depth_m[valid] - min_depth_m)
        / (max_depth_m - min_depth_m),
        0,
        255,
    ).astype(np.uint8)
    colored = cv2.applyColorMap(255 - normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def draw_context_skeleton(
    image: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.3,
) -> None:
    for start, end in HALPE26_LINKS:
        if scores[start] < threshold or scores[end] < threshold:
            continue
        start_pixel = tuple(np.rint(keypoints[start]).astype(int))
        end_pixel = tuple(np.rint(keypoints[end]).astype(int))
        cv2.line(
            image,
            start_pixel,
            end_pixel,
            (0, 210, 255),
            2,
            cv2.LINE_AA,
        )
    for index, point in enumerate(keypoints):
        if scores[index] < threshold:
            continue
        cv2.circle(
            image,
            tuple(np.rint(point).astype(int)),
            3,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )


def render_frame(
    row: dict[str, Any],
    frame_violations: list[dict[str, Any]],
    depth_scale: float,
    min_depth_m: float,
    max_depth_m: float,
    anchor_confidence: float,
) -> np.ndarray:
    rgb = cv2.imread(row["sources"]["rgb"], cv2.IMREAD_COLOR)
    if rgb is None:
        raise RuntimeError(f"Failed to read RGB: {row['sources']['rgb']}")
    depth_m = load_depth_m(row["sources"]["depth"], depth_scale)
    depth_view = depth_colormap(depth_m, min_depth_m, max_depth_m)
    pose2d = row["pose2d"]
    keypoints = np.asarray(
        [[item["x"], item["y"]] for item in pose2d["keypoints"]],
        dtype=np.float32,
    )
    scores = np.asarray(
        [item["confidence"] for item in pose2d["keypoints"]],
        dtype=np.float32,
    )
    draw_context_skeleton(rgb, keypoints, scores)
    draw_context_skeleton(depth_view, keypoints, scores)

    for violation in frame_violations:
        start = violation["start_id"]
        end = violation["end_id"]
        start_pixel = tuple(np.rint(keypoints[start]).astype(int))
        end_pixel = tuple(np.rint(keypoints[end]).astype(int))
        for canvas in (rgb, depth_view):
            cv2.line(
                canvas,
                start_pixel,
                end_pixel,
                (0, 0, 255),
                6,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                start_pixel,
                8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.circle(
                canvas,
                end_pixel,
                8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            depth_view,
            (
                f"{violation['start_name']} "
                f"z={violation['start_depth_m']:.2f}m"
            ),
            (start_pixel[0] + 8, start_pixel[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            depth_view,
            (
                f"{violation['end_name']} "
                f"z={violation['end_depth_m']:.2f}m"
            ),
            (end_pixel[0] + 8, end_pixel[1] + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    width = rgb.shape[1] + depth_view.shape[1]
    header_height = 106 + 27 * len(frame_violations)
    header = np.full((header_height, width, 3), 25, dtype=np.uint8)
    cv2.putText(
        header,
        (
            f"Frame {row['frame_index']}  timestamp={row['timestamp_raw']}  "
            f"anchor threshold={anchor_confidence:.2f}"
        ),
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.67,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        (
            "RED = both endpoints are trusted anchors, but their 3D "
            "distance violates the calibrated length band"
        ),
        (14, 57),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (80, 170, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "RGB + 2D skeleton",
        (14, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        header,
        "Depth view (near=red/yellow, far=blue)",
        (rgb.shape[1] + 14, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    for line_index, violation in enumerate(frame_violations):
        y = 113 + 27 * line_index
        relative_percent = 100.0 * violation["relative_error"]
        text = (
            f"{violation['start_name']} -> {violation['end_name']}: "
            f"actual={violation['actual_length_m']:.3f}m, "
            f"prior={violation['target_length_m']:.3f}m, "
            f"error={relative_percent:.1f}%, "
            f"conf=({violation['start_confidence']:.2f},"
            f"{violation['end_confidence']:.2f})"
        )
        cv2.putText(
            header,
            text,
            (14, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (80, 170, 255),
            1,
            cv2.LINE_AA,
        )
    return np.vstack((header, np.hstack((rgb, depth_view))))


def main() -> int:
    args = parse_args()
    if args.top_frames <= 0:
        raise ValueError("--top-frames must be positive.")
    summary = load_json(args.summary)
    rows = load_jsonl_objects(args.poses)
    tracking = load_yaml(args.tracking_config)["tracking"]
    camera = load_yaml(args.camera_config)["camera"]
    anchor_confidence = float(
        tracking["bone_constraint"]["solver"]["anchor_confidence"]
    )
    violations = find_anchor_violations(
        rows,
        summary["bone_constraint"]["calibration"]["bones"],
        anchor_confidence,
    )
    if not violations:
        raise RuntimeError("No high-confidence bone violations were found.")

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for violation in violations:
        grouped[violation["frame_index"]].append(violation)
    ranked_frames = sorted(
        grouped,
        key=lambda frame_index: grouped[frame_index][0][
            "relative_violation"
        ],
        reverse=True,
    )[: args.top_frames]
    rows_by_index = {row["frame_index"]: row for row in rows}

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[np.ndarray] = []
    output_images: list[str] = []
    for rank, frame_index in enumerate(ranked_frames, start=1):
        row = rows_by_index[frame_index]
        frame_violations = grouped[frame_index]
        canvas = render_frame(
            row,
            frame_violations,
            depth_scale=float(camera["depth_scale"]),
            min_depth_m=float(camera["min_depth_m"]),
            max_depth_m=float(camera["max_depth_m"]),
            anchor_confidence=anchor_confidence,
        )
        output_path = (
            output_dir
            / f"rank_{rank:02d}_frame_{frame_index:03d}_"
            f"{row['timestamp_raw']}.png"
        )
        if not cv2.imwrite(str(output_path), canvas):
            raise RuntimeError(f"Failed to write {output_path}")
        output_images.append(str(output_path))
        rendered.append(
            cv2.resize(canvas, (1400, 650), interpolation=cv2.INTER_AREA)
        )

    overview_path = output_dir / "worst_anchor_violations_overview.png"
    overview = np.vstack(rendered)
    if not cv2.imwrite(str(overview_path), overview):
        raise RuntimeError(f"Failed to write {overview_path}")
    report = {
        "schema_version": 1,
        "method": (
            "Retrospective comparison against the final sequence bone prior"
        ),
        "anchor_confidence": anchor_confidence,
        "violation_count_using_final_prior": len(violations),
        "ranked_frame_indices": ranked_frames,
        "overview": str(overview_path),
        "images": output_images,
        "violations": violations,
        "note": (
            "This retrospective count differs from the online summary count "
            "because bone priors became ready gradually during the sequence."
        ),
    }
    report_path = output_dir / "violations.json"
    atomic_write_json(report_path, report)
    print(f"Saved overview: {overview_path}")
    print(f"Saved ranked report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
