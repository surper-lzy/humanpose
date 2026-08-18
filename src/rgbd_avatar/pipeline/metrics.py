"""Pure sequence statistics shared by processing and reporting tools."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def duration_statistics(
    values_ms: Sequence[float],
) -> dict[str, float | int | None]:
    """Summarize stage latency measurements in milliseconds."""

    if not values_ms:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "max_ms": None,
            "throughput_fps_from_mean": None,
        }
    values = np.asarray(values_ms, dtype=np.float64)
    mean_ms = float(np.mean(values))
    return {
        "count": int(values.size),
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
        "throughput_fps_from_mean": 1000.0 / mean_ms if mean_ms > 0 else None,
    }


def value_statistics(
    values: Sequence[float],
) -> dict[str, float | int | None]:
    """Summarize an arbitrary finite scalar measurement."""

    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def segment_metadata(
    segments: Sequence[Sequence[Any]],
) -> tuple[list[dict[str, Any]], list[float]]:
    """Describe timestamp-contiguous frame segments."""

    metadata: list[dict[str, Any]] = []
    valid_intervals_s: list[float] = []
    for segment_id, segment in enumerate(segments):
        intervals = [
            current.relative_time_s - previous.relative_time_s
            for previous, current in zip(segment, segment[1:])
        ]
        valid_intervals_s.extend(intervals)
        median_interval = float(np.median(intervals)) if intervals else None
        metadata.append(
            {
                "segment_id": segment_id,
                "frame_count": len(segment),
                "first_frame_index": segment[0].frame_index,
                "last_frame_index": segment[-1].frame_index,
                "start_timestamp_raw": segment[0].timestamp_raw,
                "end_timestamp_raw": segment[-1].timestamp_raw,
                "start_relative_time_s": segment[0].relative_time_s,
                "end_relative_time_s": segment[-1].relative_time_s,
                "duration_s": segment[-1].relative_time_s
                - segment[0].relative_time_s,
                "median_interval_s": median_interval,
                "nominal_fps": (
                    1.0 / median_interval
                    if median_interval is not None and median_interval > 0
                    else None
                ),
            }
        )
    return metadata, valid_intervals_s


__all__ = ["duration_statistics", "segment_metadata", "value_statistics"]
