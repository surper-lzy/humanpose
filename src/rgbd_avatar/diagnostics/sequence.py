#!/usr/bin/env python3
"""Export per-frame joint positions and diagnose RGB-D skeleton anomalies."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from rgbd_avatar.io import atomic_write_json, load_jsonl_objects
from rgbd_avatar.pose import HALPE26_LINKS, HALPE26_NAMES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAYERS = ("raw", "temporal", "constrained")
TORSO_REFERENCE_IDS = (5, 6, 11, 12, 18, 19)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poses-jsonl",
        type=Path,
        default=PROJECT_ROOT
        / "outputs/sequences/4_pointcloud_exit_gate/poses.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <poses directory>/diagnostics.",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        choices=LAYERS,
        default=list(LAYERS),
    )
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--mad-scale", type=float, default=6.0)
    parser.add_argument(
        "--bone-relative-tolerance",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--bone-absolute-tolerance-m",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--joint-depth-offset-tolerance-m",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--joint-depth-jump-tolerance-m",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--selection-margin-warning",
        type=float,
        default=0.06,
        help="Warn when multiple local surfaces have a small score margin.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Maximum rows per anomaly table in report.md.",
    )
    return parser.parse_args()


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load diagnostic input through the shared JSONL contract."""

    records = load_jsonl_objects(path)
    if not records:
        raise ValueError(f"No records found in {path.expanduser().resolve()}.")
    return records


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if np.isfinite(result) else None


def _xyz(payload: dict[str, Any] | None) -> np.ndarray | None:
    if not payload:
        return None
    value = payload.get("xyz_m")
    if value is None:
        return None
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        return None
    return point


def _indexed(
    payload: dict[str, Any] | None,
    key: str,
) -> dict[int, dict[str, Any]]:
    if not payload:
        return {}
    values = payload.get(key)
    if not isinstance(values, list):
        return {}
    return {
        int(value["id"]): value
        for value in values
        if isinstance(value, dict) and "id" in value
    }


def _layer_payload(
    record: dict[str, Any],
    layer: str,
) -> dict[str, Any] | None:
    key = {
        "raw": "pose3d_raw",
        "temporal": "pose3d_temporal",
        "constrained": "pose3d_constrained",
    }[layer]
    payload = record.get(key)
    return payload if isinstance(payload, dict) else None


def _layer_joint_is_usable(
    joint: dict[str, Any] | None,
    layer: str,
) -> bool:
    if not joint:
        return False
    key = "valid" if layer == "raw" else "usable"
    return bool(joint.get(key)) and _xyz(joint) is not None


def _torso_reference_depth(
    raw_joints: dict[int, dict[str, Any]],
) -> float | None:
    depths = [
        point[2]
        for joint_id in TORSO_REFERENCE_IDS
        if (point := _xyz(raw_joints.get(joint_id))) is not None
        and bool(raw_joints[joint_id].get("valid"))
    ]
    return float(np.median(depths)) if depths else None


def build_joint_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        pose2d = _indexed(record.get("pose2d"), "keypoints")
        raw = _indexed(record.get("pose3d_raw"), "joints")
        temporal = _indexed(record.get("pose3d_temporal"), "joints")
        constrained = _indexed(
            record.get("pose3d_constrained"),
            "joints",
        )
        recovery = _indexed(record.get("depth_recovery"), "joints")
        person_filter = (
            (record.get("depth_recovery") or {}).get("person_filter") or {}
        )
        frame_presence = record.get("frame_presence")
        if not isinstance(frame_presence, dict):
            frame_presence = {}
        person_depth = _as_float(
            person_filter.get("person_depth_prior_m")
        )
        torso_depth = _torso_reference_depth(raw)

        for joint_id, joint_name in enumerate(HALPE26_NAMES):
            p2 = pose2d.get(joint_id, {})
            raw_joint = raw.get(joint_id, {})
            temporal_joint = temporal.get(joint_id, {})
            constrained_joint = constrained.get(joint_id, {})
            diagnostic = recovery.get(joint_id, {})
            raw_xyz = _xyz(raw_joint)
            temporal_xyz = _xyz(temporal_joint)
            constrained_xyz = _xyz(constrained_joint)
            raw_valid = bool(raw_joint.get("valid")) and raw_xyz is not None
            temporal_usable = (
                bool(temporal_joint.get("usable"))
                and temporal_xyz is not None
            )
            constrained_usable = (
                bool(constrained_joint.get("usable"))
                and constrained_xyz is not None
            )
            topology = diagnostic.get("topology_gate")
            if not isinstance(topology, dict):
                topology = {}
            face_group = diagnostic.get("face_group_gate")
            if not isinstance(face_group, dict):
                face_group = {}
            foot_group = diagnostic.get("foot_group_gate")
            if not isinstance(foot_group, dict):
                foot_group = {}
            self_occlusion = diagnostic.get("self_occlusion_gate")
            if not isinstance(self_occlusion, dict):
                self_occlusion = {}

            row: dict[str, Any] = {
                "frame_index": record.get("frame_index"),
                "timestamp_raw": record.get("timestamp_raw"),
                "relative_time_s": record.get("relative_time_s"),
                "dt_s": record.get("dt_s"),
                "segment_id": record.get("segment_id"),
                "frame_status": record.get("status"),
                "frame_presence_accepted": frame_presence.get("accepted"),
                "frame_presence_reason": frame_presence.get("reason"),
                "frame_border_contacts": "|".join(
                    frame_presence.get("border_contacts") or ()
                ),
                "frame_presence_quality_failures": "|".join(
                    frame_presence.get("quality_failures") or ()
                ),
                "frame_presence_track_reset": bool(
                    frame_presence.get("track_reset_required")
                ),
                "joint_id": joint_id,
                "joint_name": joint_name,
                "rgb_x_px": _as_float(p2.get("x")),
                "rgb_y_px": _as_float(p2.get("y")),
                "pose2d_confidence": _as_float(p2.get("confidence")),
                "pose2d_valid": bool(p2.get("valid")),
                "raw_valid": raw_valid,
                "raw_x_m": raw_xyz[0] if raw_valid else None,
                "raw_y_m": raw_xyz[1] if raw_valid else None,
                "raw_z_m": raw_xyz[2] if raw_valid else None,
                "raw_depth_m": _as_float(raw_joint.get("depth_m")),
                "raw_confidence": _as_float(
                    raw_joint.get("confidence")
                ),
                "raw_depth_confidence": _as_float(
                    raw_joint.get("depth_confidence")
                ),
                "torso_reference_depth_m": torso_depth,
                "raw_depth_offset_torso_m": (
                    float(raw_xyz[2] - torso_depth)
                    if raw_valid and torso_depth is not None
                    else None
                ),
                "person_depth_prior_m": person_depth,
                "raw_depth_offset_person_m": (
                    float(raw_xyz[2] - person_depth)
                    if raw_valid and person_depth is not None
                    else None
                ),
                "temporal_usable": temporal_usable,
                "temporal_x_m": (
                    temporal_xyz[0] if temporal_usable else None
                ),
                "temporal_y_m": (
                    temporal_xyz[1] if temporal_usable else None
                ),
                "temporal_z_m": (
                    temporal_xyz[2] if temporal_usable else None
                ),
                "temporal_confidence": _as_float(
                    temporal_joint.get("confidence")
                ),
                "temporal_observed": bool(
                    temporal_joint.get("observed")
                ),
                "temporal_predicted": bool(
                    temporal_joint.get("predicted")
                ),
                "temporal_age_s": _as_float(
                    temporal_joint.get("age_s")
                ),
                "constrained_usable": constrained_usable,
                "constrained_x_m": (
                    constrained_xyz[0] if constrained_usable else None
                ),
                "constrained_y_m": (
                    constrained_xyz[1] if constrained_usable else None
                ),
                "constrained_z_m": (
                    constrained_xyz[2] if constrained_usable else None
                ),
                "constrained_corrected": bool(
                    constrained_joint.get("corrected")
                ),
                "constrained_correction_m": _as_float(
                    constrained_joint.get("correction_m")
                ),
                "recovery_status": diagnostic.get("status"),
                "radius_px": diagnostic.get("radius_px"),
                "expanded_radius": bool(
                    diagnostic.get("expanded_radius")
                ),
                "bbox_mismatch": bool(diagnostic.get("bbox_mismatch")),
                "candidate_point_count": diagnostic.get(
                    "candidate_point_count"
                ),
                "cluster_count": diagnostic.get("cluster_count"),
                "feasible_cluster_count": diagnostic.get(
                    "feasible_cluster_count"
                ),
                "selected_point_count": diagnostic.get(
                    "selected_point_count"
                ),
                "selected_depth_m": _as_float(
                    diagnostic.get("selected_depth_m")
                ),
                "depth_mad_m": _as_float(
                    diagnostic.get("depth_mad_m")
                ),
                "center_distance_px": _as_float(
                    diagnostic.get("center_distance_px")
                ),
                "selection_score": _as_float(
                    diagnostic.get("selection_score")
                ),
                "selection_margin": _as_float(
                    diagnostic.get("selection_margin")
                ),
                "topology_gate_applied": bool(topology.get("applied")),
                "topology_rejected_cluster_count": topology.get(
                    "rejected_cluster_count"
                ),
                "topology_selected_eye_ear_length_m": _as_float(
                    topology.get("selected_eye_ear_length_m")
                ),
                "face_group_gate_applied": bool(
                    face_group.get("applied")
                ),
                "face_group_candidate_rank": face_group.get(
                    "selected_candidate_rank"
                ),
                "face_group_objective": _as_float(
                    face_group.get("objective")
                ),
                "face_group_feasible_combinations": face_group.get(
                    "feasible_combination_count"
                ),
                "foot_group_gate_applied": bool(
                    foot_group.get("applied")
                ),
                "foot_group_candidate_rank": foot_group.get(
                    "selected_candidate_rank"
                ),
                "foot_group_objective": _as_float(
                    foot_group.get("objective")
                ),
                "foot_group_feasible_combinations": foot_group.get(
                    "feasible_combination_count"
                ),
                "foot_group_rejection_reason": foot_group.get("reason"),
                "self_occlusion_gate_applied": bool(
                    self_occlusion.get("applied")
                ),
                "self_occlusion_group": self_occlusion.get("group"),
                "self_occlusion_center_id": self_occlusion.get(
                    "center_id"
                ),
                "self_occlusion_left_length_m": _as_float(
                    self_occlusion.get("left_length_m")
                ),
                "self_occlusion_right_length_m": _as_float(
                    self_occlusion.get("right_length_m")
                ),
            }
            rows.append(row)
    return rows


def build_connection_rows(
    records: list[dict[str, Any]],
    layers: Iterable[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for layer in layers:
            joints = _indexed(_layer_payload(record, layer), "joints")
            for link_id, (start_id, end_id) in enumerate(HALPE26_LINKS):
                start = joints.get(start_id)
                end = joints.get(end_id)
                start_xyz = _xyz(start)
                end_xyz = _xyz(end)
                usable = (
                    _layer_joint_is_usable(start, layer)
                    and _layer_joint_is_usable(end, layer)
                    and start_xyz is not None
                    and end_xyz is not None
                )
                delta = (
                    end_xyz - start_xyz
                    if usable
                    else np.full(3, np.nan)
                )
                rows.append(
                    {
                        "frame_index": record.get("frame_index"),
                        "timestamp_raw": record.get("timestamp_raw"),
                        "relative_time_s": record.get("relative_time_s"),
                        "segment_id": record.get("segment_id"),
                        "frame_status": record.get("status"),
                        "layer": layer,
                        "link_id": link_id,
                        "start_id": start_id,
                        "start_name": HALPE26_NAMES[start_id],
                        "end_id": end_id,
                        "end_name": HALPE26_NAMES[end_id],
                        "usable": usable,
                        "start_z_m": (
                            float(start_xyz[2]) if usable else None
                        ),
                        "end_z_m": (
                            float(end_xyz[2]) if usable else None
                        ),
                        "delta_x_m": float(delta[0]) if usable else None,
                        "delta_y_m": float(delta[1]) if usable else None,
                        "delta_z_m": float(delta[2]) if usable else None,
                        "depth_gap_m": (
                            float(abs(delta[2])) if usable else None
                        ),
                        "length_m": (
                            float(np.linalg.norm(delta))
                            if usable
                            else None
                        ),
                    }
                )
    return rows


def _robust_statistics(
    values: list[float],
) -> tuple[float | None, float | None, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return None, None, None
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    robust_sigma = 1.4826 * mad
    return median, mad, robust_sigma


def annotate_connection_anomalies(
    rows: list[dict[str, Any]],
    *,
    min_samples: int,
    mad_scale: float,
    relative_tolerance: float,
    absolute_tolerance_m: float,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["usable"]:
            groups[(str(row["layer"]), int(row["link_id"]))].append(row)

    stats: dict[tuple[str, int], dict[str, Any]] = {}
    for key, group in groups.items():
        lengths = [float(row["length_m"]) for row in group]
        median, mad, robust_sigma = _robust_statistics(lengths)
        if median is None or mad is None or robust_sigma is None:
            continue
        threshold = max(
            absolute_tolerance_m,
            relative_tolerance * median,
            mad_scale * robust_sigma,
        )
        stats[key] = {
            "sample_count": len(lengths),
            "median_length_m": median,
            "mad_length_m": mad,
            "robust_sigma_m": robust_sigma,
            "outlier_threshold_m": threshold,
        }

    for row in rows:
        stat = stats.get((str(row["layer"]), int(row["link_id"])))
        row.update(
            {
                "sample_count": stat["sample_count"] if stat else 0,
                "median_length_m": (
                    stat["median_length_m"] if stat else None
                ),
                "mad_length_m": (
                    stat["mad_length_m"] if stat else None
                ),
                "length_deviation_m": None,
                "length_relative_deviation": None,
                "outlier_threshold_m": (
                    stat["outlier_threshold_m"] if stat else None
                ),
                "length_outlier": False,
            }
        )
        if (
            not row["usable"]
            or stat is None
            or stat["sample_count"] < min_samples
        ):
            continue
        deviation = abs(
            float(row["length_m"]) - stat["median_length_m"]
        )
        row["length_deviation_m"] = deviation
        row["length_relative_deviation"] = (
            deviation / stat["median_length_m"]
            if stat["median_length_m"] > 0
            else None
        )
        row["length_outlier"] = (
            deviation > stat["outlier_threshold_m"]
        )
    return rows


def annotate_joint_depth_anomalies(
    rows: list[dict[str, Any]],
    *,
    min_samples: int,
    mad_scale: float,
    offset_tolerance_m: float,
    jump_tolerance_m: float,
) -> list[dict[str, Any]]:
    by_joint: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["raw_valid"] and row["raw_depth_offset_torso_m"] is not None:
            by_joint[int(row["joint_id"])].append(row)

    for joint_rows in by_joint.values():
        offsets = [
            float(row["raw_depth_offset_torso_m"]) for row in joint_rows
        ]
        median, mad, robust_sigma = _robust_statistics(offsets)
        threshold = (
            max(offset_tolerance_m, mad_scale * robust_sigma)
            if robust_sigma is not None
            else None
        )
        for row in joint_rows:
            row["joint_offset_median_m"] = median
            row["joint_offset_mad_m"] = mad
            row["joint_offset_threshold_m"] = threshold
            if (
                len(joint_rows) >= min_samples
                and median is not None
                and threshold is not None
            ):
                deviation = abs(
                    float(row["raw_depth_offset_torso_m"]) - median
                )
                row["joint_offset_deviation_m"] = deviation
                row["joint_offset_outlier"] = deviation > threshold

        jumps: list[tuple[dict[str, Any], float]] = []
        previous: dict[str, Any] | None = None
        for row in sorted(
            joint_rows,
            key=lambda item: (
                int(item["segment_id"]),
                int(item["frame_index"]),
            ),
        ):
            if (
                previous is not None
                and row["segment_id"] == previous["segment_id"]
            ):
                jump = abs(
                    float(row["raw_depth_offset_torso_m"])
                    - float(previous["raw_depth_offset_torso_m"])
                )
                jumps.append((row, jump))
                row["previous_valid_frame_index"] = previous["frame_index"]
                row["joint_offset_jump_m"] = jump
            previous = row
        _, jump_mad, jump_sigma = _robust_statistics(
            [value for _, value in jumps]
        )
        jump_threshold = (
            max(jump_tolerance_m, mad_scale * jump_sigma)
            if jump_sigma is not None
            else None
        )
        for row, jump in jumps:
            row["joint_jump_mad_m"] = jump_mad
            row["joint_jump_threshold_m"] = jump_threshold
            row["joint_jump_outlier"] = (
                len(jumps) >= min_samples - 1
                and jump_threshold is not None
                and jump > jump_threshold
            )

    defaults = {
        "joint_offset_median_m": None,
        "joint_offset_mad_m": None,
        "joint_offset_threshold_m": None,
        "joint_offset_deviation_m": None,
        "joint_offset_outlier": False,
        "previous_valid_frame_index": None,
        "joint_offset_jump_m": None,
        "joint_jump_mad_m": None,
        "joint_jump_threshold_m": None,
        "joint_jump_outlier": False,
    }
    for row in rows:
        for key, value in defaults.items():
            row.setdefault(key, value)
    return rows


def build_frame_rows(
    records: list[dict[str, Any]],
    joint_rows: list[dict[str, Any]],
    connection_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    joints_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    links_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in joint_rows:
        joints_by_frame[int(row["frame_index"])].append(row)
    for row in connection_rows:
        if row["layer"] == "raw":
            links_by_frame[int(row["frame_index"])].append(row)

    frame_rows: list[dict[str, Any]] = []
    for record in records:
        frame_index = int(record["frame_index"])
        joints = joints_by_frame[frame_index]
        links = links_by_frame[frame_index]
        link_outliers = [
            row for row in links if row["length_outlier"]
        ]
        offset_outliers = [
            row for row in joints if row["joint_offset_outlier"]
        ]
        jump_outliers = [
            row for row in joints if row["joint_jump_outlier"]
        ]
        relative_deviations = [
            float(row["length_relative_deviation"])
            for row in link_outliers
            if row["length_relative_deviation"] is not None
        ]
        frame_rows.append(
            {
                "frame_index": frame_index,
                "timestamp_raw": record.get("timestamp_raw"),
                "relative_time_s": record.get("relative_time_s"),
                "segment_id": record.get("segment_id"),
                "frame_status": record.get("status"),
                "raw_valid_joint_count": sum(
                    bool(row["raw_valid"]) for row in joints
                ),
                "raw_usable_link_count": sum(
                    bool(row["usable"]) for row in links
                ),
                "bone_length_outlier_count": len(link_outliers),
                "joint_depth_offset_outlier_count": len(offset_outliers),
                "joint_depth_jump_outlier_count": len(jump_outliers),
                "max_bone_relative_deviation": (
                    max(relative_deviations)
                    if relative_deviations
                    else None
                ),
                "anomaly_score": (
                    3 * len(link_outliers)
                    + 2 * len(offset_outliers)
                    + len(jump_outliers)
                ),
            }
        )
    return frame_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(
    headers: list[str],
    rows: list[list[Any]],
) -> list[str]:
    result = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        result.append(
            "| "
            + " | ".join(
                "" if value is None else str(value) for value in row
            )
            + " |"
        )
    return result


def build_report(
    *,
    input_path: Path,
    frame_rows: list[dict[str, Any]],
    joint_rows: list[dict[str, Any]],
    connection_rows: list[dict[str, Any]],
    top_k: int,
    parameters: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    raw_links = [
        row for row in connection_rows if row["layer"] == "raw"
    ]
    bone_outliers = sorted(
        (row for row in raw_links if row["length_outlier"]),
        key=lambda row: float(row["length_relative_deviation"] or 0),
        reverse=True,
    )
    offset_outliers = sorted(
        (row for row in joint_rows if row["joint_offset_outlier"]),
        key=lambda row: float(row["joint_offset_deviation_m"] or 0),
        reverse=True,
    )
    jump_outliers = sorted(
        (row for row in joint_rows if row["joint_jump_outlier"]),
        key=lambda row: float(row["joint_offset_jump_m"] or 0),
        reverse=True,
    )
    selection_margin_warning = float(
        parameters["selection_margin_warning"]
    )
    uncertain_selections = sorted(
        (
            row
            for row in joint_rows
            if row["recovery_status"] == "selected"
            and int(row["cluster_count"] or 0) > 1
            and row["selection_margin"] is not None
            and float(row["selection_margin"])
            < selection_margin_warning
        ),
        key=lambda row: float(row["selection_margin"]),
    )
    recovery_counts = Counter(
        str(row["recovery_status"])
        for row in joint_rows
        if row["recovery_status"] is not None
    )
    suspect_counts: Counter[str] = Counter()
    for row in bone_outliers:
        suspect_counts[str(row["start_name"])] += 1
        suspect_counts[str(row["end_name"])] += 1
    for row in offset_outliers:
        suspect_counts[str(row["joint_name"])] += 2
    for row in jump_outliers:
        suspect_counts[str(row["joint_name"])] += 1

    ranked_frames = sorted(
        frame_rows,
        key=lambda row: (
            int(row["anomaly_score"]),
            float(row["max_bone_relative_deviation"] or 0),
        ),
        reverse=True,
    )
    report: list[str] = [
        "# 三维骨架逐帧深度与连接异常排查",
        "",
        f"- 输入：`{input_path}`",
        f"- 帧数：`{len(frame_rows)}`",
        f"- 逐关节记录：`{len(joint_rows)}` 行",
        f"- 三层骨架连接记录：`{len(connection_rows)}` 行",
        f"- raw 骨长异常：`{len(bone_outliers)}` 条",
        f"- raw 相对躯干深度异常：`{len(offset_outliers)}` 个关节帧",
        f"- raw 相对躯干深度跳变：`{len(jump_outliers)}` 个关节帧",
        f"- 多表面低选择裕量：`{len(uncertain_selections)}` 个关节帧",
        "",
        "异常标记使用整段序列的中位数和 MAD，是无真值情况下的排查线索，"
        "不能直接等价为解剖关节误差。",
        "",
        "## 高风险帧",
        "",
    ]
    report.extend(
        _markdown_table(
            [
                "frame",
                "timestamp",
                "valid joints",
                "bone outliers",
                "depth offsets",
                "depth jumps",
                "score",
            ],
            [
                [
                    row["frame_index"],
                    row["timestamp_raw"],
                    row["raw_valid_joint_count"],
                    row["bone_length_outlier_count"],
                    row["joint_depth_offset_outlier_count"],
                    row["joint_depth_jump_outlier_count"],
                    row["anomaly_score"],
                ]
                for row in ranked_frames[:top_k]
                if int(row["anomaly_score"]) > 0
            ],
        )
    )
    report.extend(["", "## 异常连接", ""])
    report.extend(
        _markdown_table(
            [
                "frame",
                "connection",
                "length m",
                "median m",
                "relative deviation",
                "|Δz| m",
            ],
            [
                [
                    row["frame_index"],
                    f"{row['start_name']}→{row['end_name']}",
                    f"{float(row['length_m']):.3f}",
                    f"{float(row['median_length_m']):.3f}",
                    f"{100 * float(row['length_relative_deviation']):.1f}%",
                    f"{float(row['depth_gap_m']):.3f}",
                ]
                for row in bone_outliers[:top_k]
            ],
        )
    )
    report.extend(["", "## 关节相对躯干深度异常", ""])
    report.extend(
        _markdown_table(
            [
                "frame",
                "joint",
                "z m",
                "z-torso m",
                "deviation m",
                "recovery",
            ],
            [
                [
                    row["frame_index"],
                    row["joint_name"],
                    f"{float(row['raw_z_m']):.3f}",
                    f"{float(row['raw_depth_offset_torso_m']):.3f}",
                    f"{float(row['joint_offset_deviation_m']):.3f}",
                    row["recovery_status"],
                ]
                for row in offset_outliers[:top_k]
            ],
        )
    )
    report.extend(["", "## 相邻有效帧相对深度跳变", ""])
    report.extend(
        _markdown_table(
            ["frame", "previous", "joint", "jump m", "recovery"],
            [
                [
                    row["frame_index"],
                    row["previous_valid_frame_index"],
                    row["joint_name"],
                    f"{float(row['joint_offset_jump_m']):.3f}",
                    row["recovery_status"],
                ]
                for row in jump_outliers[:top_k]
            ],
        )
    )
    report.extend(["", "## 疑似关节累计", ""])
    report.extend(
        _markdown_table(
            ["joint", "diagnostic hits"],
            [[name, count] for name, count in suspect_counts.most_common()],
        )
    )
    report.extend(["", "## 深度恢复状态", ""])
    report.extend(
        _markdown_table(
            ["status", "joint frames"],
            [[name, count] for name, count in recovery_counts.most_common()],
        )
    )
    report.extend(["", "## 多表面低选择裕量", ""])
    report.extend(
        _markdown_table(
            [
                "frame",
                "joint",
                "z m",
                "clusters",
                "selection margin",
            ],
            [
                [
                    row["frame_index"],
                    row["joint_name"],
                    f"{float(row['raw_z_m']):.3f}",
                    row["cluster_count"],
                    f"{float(row['selection_margin']):.4f}",
                ]
                for row in uncertain_selections[:top_k]
            ],
        )
    )
    report.extend(
        [
            "",
            "## 输出说明",
            "",
            "- `joint_positions.csv`：每帧 26 个关节的 2D、raw、temporal、"
            "constrained 和点云恢复诊断。",
            "- `bone_connections.csv`：每帧 25 条连接在三层骨架中的长度、"
            "深度差和稳健异常标记。",
            "- `frame_summary.csv`：每帧异常数量与排序分数。",
            "- `anomalies.json`：便于程序读取的完整异常列表。",
            "",
            "优先查看同时满足“连接异常”和“相对躯干深度异常/跳变”的"
            "关节帧；单独出现的骨长异常也可能来自二维姿态误差、遮挡或"
            "解剖表面与关节中心的偏差。",
            "",
        ]
    )

    anomaly_payload = {
        "input": str(input_path),
        "parameters": parameters,
        "summary": {
            "frame_count": len(frame_rows),
            "joint_row_count": len(joint_rows),
            "connection_row_count": len(connection_rows),
            "raw_bone_length_outlier_count": len(bone_outliers),
            "raw_joint_depth_offset_outlier_count": len(offset_outliers),
            "raw_joint_depth_jump_outlier_count": len(jump_outliers),
            "uncertain_multi_surface_selection_count": len(
                uncertain_selections
            ),
            "recovery_status_counts": dict(recovery_counts),
        },
        "ranked_frames": ranked_frames,
        "raw_bone_length_outliers": bone_outliers,
        "raw_joint_depth_offset_outliers": offset_outliers,
        "raw_joint_depth_jump_outliers": jump_outliers,
        "uncertain_multi_surface_selections": uncertain_selections,
        "suspect_joint_counts": dict(suspect_counts),
    }
    return "\n".join(report), anomaly_payload


def validate_args(args: argparse.Namespace) -> None:
    if args.min_samples < 3:
        raise ValueError("--min-samples must be at least 3.")
    if args.mad_scale <= 0:
        raise ValueError("--mad-scale must be positive.")
    for name in (
        "bone_relative_tolerance",
        "bone_absolute_tolerance_m",
        "joint_depth_offset_tolerance_m",
        "joint_depth_jump_tolerance_m",
        "selection_margin_warning",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    input_path = args.poses_jsonl.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_path.parent / "diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    joint_rows = annotate_joint_depth_anomalies(
        build_joint_rows(records),
        min_samples=args.min_samples,
        mad_scale=args.mad_scale,
        offset_tolerance_m=args.joint_depth_offset_tolerance_m,
        jump_tolerance_m=args.joint_depth_jump_tolerance_m,
    )
    connection_rows = annotate_connection_anomalies(
        build_connection_rows(records, args.layers),
        min_samples=args.min_samples,
        mad_scale=args.mad_scale,
        relative_tolerance=args.bone_relative_tolerance,
        absolute_tolerance_m=args.bone_absolute_tolerance_m,
    )
    frame_rows = build_frame_rows(records, joint_rows, connection_rows)
    parameters = {
        "layers": args.layers,
        "min_samples": args.min_samples,
        "mad_scale": args.mad_scale,
        "bone_relative_tolerance": args.bone_relative_tolerance,
        "bone_absolute_tolerance_m": args.bone_absolute_tolerance_m,
        "joint_depth_offset_tolerance_m": (
            args.joint_depth_offset_tolerance_m
        ),
        "joint_depth_jump_tolerance_m": (
            args.joint_depth_jump_tolerance_m
        ),
        "selection_margin_warning": args.selection_margin_warning,
    }
    report, anomalies = build_report(
        input_path=input_path,
        frame_rows=frame_rows,
        joint_rows=joint_rows,
        connection_rows=connection_rows,
        top_k=args.top_k,
        parameters=parameters,
    )

    _write_csv(output_dir / "joint_positions.csv", joint_rows)
    _write_csv(output_dir / "bone_connections.csv", connection_rows)
    _write_csv(output_dir / "frame_summary.csv", frame_rows)
    atomic_write_json(output_dir / "anomalies.json", anomalies)
    (output_dir / "report.md").write_text(report, encoding="utf-8")

    print(f"Saved per-joint positions: {output_dir / 'joint_positions.csv'}")
    print(f"Saved bone connections: {output_dir / 'bone_connections.csv'}")
    print(f"Saved frame summary: {output_dir / 'frame_summary.csv'}")
    print(f"Saved anomaly data: {output_dir / 'anomalies.json'}")
    print(f"Saved report: {output_dir / 'report.md'}")
    print(json.dumps(anomalies["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
