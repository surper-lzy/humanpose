"""Validated pose and hand JSONL record contracts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from rgbd_avatar.io import load_jsonl_objects


def load_pose_records(path: str | Path) -> list[dict[str, Any]]:
    """Load and validate the sequence processor's pose JSONL output."""

    records = load_jsonl_objects(path)
    resolved = Path(path).expanduser().resolve()
    required_fields = (
        "frame_index",
        "timestamp_raw",
        "relative_time_s",
        "segment_id",
        "segment_start",
        "sources",
        "pose3d_temporal",
    )
    for record_index, record in enumerate(records):
        if record.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported schema_version at record {record_index} in "
                f"{resolved}: {record.get('schema_version')!r}"
            )
        missing = [field for field in required_fields if field not in record]
        if missing:
            raise ValueError(
                f"Missing fields {missing!r} at record {record_index} in "
                f"{resolved}."
            )

    previous_frame_index: int | None = None
    previous_time_s: float | None = None
    previous_segment_id: int | None = None
    for record_index, record in enumerate(records):
        frame_index = int(record["frame_index"])
        relative_time_s = float(record["relative_time_s"])
        segment_id = int(record["segment_id"])
        segment_start = bool(record["segment_start"])
        if not math.isfinite(relative_time_s):
            raise ValueError(
                f"Frame {frame_index} has a non-finite relative_time_s."
            )
        if (
            previous_frame_index is not None
            and frame_index <= previous_frame_index
        ):
            raise ValueError("frame_index must be strictly increasing.")
        if previous_time_s is not None and relative_time_s <= previous_time_s:
            raise ValueError("relative_time_s must be strictly increasing.")
        if previous_segment_id is not None and segment_id < previous_segment_id:
            raise ValueError("segment_id must be non-decreasing.")
        expected_segment_start = (
            record_index == 0 or segment_id != previous_segment_id
        )
        if segment_start != expected_segment_start:
            raise ValueError(
                f"Frame {frame_index} has an inconsistent segment_start flag."
            )
        previous_frame_index = frame_index
        previous_time_s = relative_time_s
        previous_segment_id = segment_id
    return records


def load_hand_records(
    path: str | Path,
    pose_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load a Hand21 cache and align it to a selected pose prefix."""

    records = load_jsonl_objects(path)
    if any(record.get("schema_version") != 1 for record in records):
        raise ValueError(f"Unsupported hand schema in {Path(path)}.")
    if len(records) < len(pose_records):
        raise ValueError("Hand cache is shorter than the selected pose sequence.")
    records = records[: len(pose_records)]
    for hand_record, pose_record in zip(records, pose_records, strict=True):
        if hand_record.get("frame_index") != pose_record.get("frame_index"):
            raise ValueError("Hand and body frame indices do not match.")
        if hand_record.get("timestamp_raw") != pose_record.get("timestamp_raw"):
            raise ValueError("Hand and body timestamps do not match.")
    return records
