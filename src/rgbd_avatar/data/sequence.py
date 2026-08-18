"""Discover strictly paired frames in an offline RGB-D recording."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re


_FRAME_PATTERN = re.compile(
    r"^(?P<timestamp>\d{8}_\d{9})_"
    r"(?P<stream>[rdat])\.(?P<extension>png|pgm|pcd)$"
)
_EXPECTED_EXTENSION = {
    "r": "png",
    "d": "pgm",
    "a": "pgm",
    "t": "pcd",
}


@dataclass(frozen=True)
class RGBDFramePaths:
    """Paths and capture time for one strictly paired RGB-D frame."""

    sequence_id: str
    frame_index: int
    timestamp_raw: str
    captured_at: datetime
    relative_time_s: float
    rgb_path: Path
    depth_path: Path
    amplitude_path: Path | None
    point_cloud_path: Path | None


def _parse_capture_time(timestamp_raw: str) -> datetime:
    try:
        # ``%f`` accepts three digits and interprets them as milliseconds.
        return datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S%f")
    except ValueError as error:
        raise ValueError(
            "Invalid RGB-D filename timestamp "
            f"{timestamp_raw!r}; expected YYYYMMDD_HHMMSSmmm."
        ) from error


def discover_rgbd_sequence(
    sequence_dir: str | Path,
) -> list[RGBDFramePaths]:
    """Build a validated manifest without nearest-timestamp matching.

    RGB and depth are required. Amplitude and exported point-cloud files are
    recorded when present but are not required by the pose pipeline.
    """

    directory = Path(sequence_dir).expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(f"RGB-D sequence directory not found: {directory}")

    streams_by_timestamp: dict[str, dict[str, Path]] = {}
    for path in directory.iterdir():
        if not path.is_file():
            continue
        match = _FRAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        stream = match.group("stream")
        extension = match.group("extension")
        if extension != _EXPECTED_EXTENSION[stream]:
            continue
        timestamp_raw = match.group("timestamp")
        streams = streams_by_timestamp.setdefault(timestamp_raw, {})
        if stream in streams:
            raise ValueError(
                f"Duplicate stream {stream!r} for timestamp {timestamp_raw}: "
                f"{streams[stream]} and {path}"
            )
        streams[stream] = path.resolve()

    if not streams_by_timestamp:
        raise FileNotFoundError(
            f"No RGB-D files matching the expected naming scheme in {directory}"
        )

    missing_required: list[str] = []
    capture_records: list[tuple[str, datetime, dict[str, Path]]] = []
    for timestamp_raw, streams in streams_by_timestamp.items():
        missing = [stream for stream in ("r", "d") if stream not in streams]
        if missing:
            missing_required.append(
                f"{timestamp_raw}: missing {','.join(missing)}"
            )
            continue
        capture_records.append(
            (timestamp_raw, _parse_capture_time(timestamp_raw), streams)
        )

    if missing_required:
        details = "; ".join(sorted(missing_required))
        raise FileNotFoundError(
            f"Unpaired required RGB-D streams in {directory}: {details}"
        )

    capture_records.sort(key=lambda item: item[1])
    first_capture = capture_records[0][1]
    frames: list[RGBDFramePaths] = []
    previous_capture: datetime | None = None
    for frame_index, (timestamp_raw, captured_at, streams) in enumerate(
        capture_records
    ):
        if previous_capture is not None and captured_at <= previous_capture:
            raise ValueError(
                "RGB-D timestamps must be strictly increasing after sorting: "
                f"{timestamp_raw}"
            )
        frames.append(
            RGBDFramePaths(
                sequence_id=directory.name,
                frame_index=frame_index,
                timestamp_raw=timestamp_raw,
                captured_at=captured_at,
                relative_time_s=(captured_at - first_capture).total_seconds(),
                rgb_path=streams["r"],
                depth_path=streams["d"],
                amplitude_path=streams.get("a"),
                point_cloud_path=streams.get("t"),
            )
        )
        previous_capture = captured_at
    return frames


def split_at_time_gaps(
    frames: list[RGBDFramePaths],
    max_gap_s: float,
) -> list[list[RGBDFramePaths]]:
    """Split a manifest so temporal state never crosses capture outages."""

    if not frames:
        return []
    if max_gap_s <= 0:
        raise ValueError("max_gap_s must be positive.")

    segments: list[list[RGBDFramePaths]] = [[frames[0]]]
    for frame in frames[1:]:
        gap_s = frame.relative_time_s - segments[-1][-1].relative_time_s
        if gap_s > max_gap_s:
            segments.append([frame])
        else:
            segments[-1].append(frame)
    return segments
