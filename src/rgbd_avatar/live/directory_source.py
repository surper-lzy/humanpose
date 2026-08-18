"""Latest-frame RGB-D source for a directory populated by camera software."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import time
from typing import Literal

import cv2

from rgbd_avatar.camera import CameraIntrinsics
from rgbd_avatar.depth import load_depth_m

from .models import RGBDFrame


_REQUIRED_FRAME_PATTERN = re.compile(
    r"^(?P<timestamp>\d{8}_\d{9})_(?P<stream>[rd])\."
    r"(?P<extension>png|pgm)$"
)
_EXPECTED_EXTENSION = {"r": "png", "d": "pgm"}
StartAt = Literal["latest", "new", "oldest"]


def capture_timestamp_ns(timestamp_raw: str) -> int:
    """Convert the filename timestamp to a timezone-neutral integer clock."""

    try:
        captured_at = datetime.strptime(timestamp_raw, "%Y%m%d_%H%M%S%f")
    except ValueError as error:
        raise ValueError(
            f"Invalid capture timestamp {timestamp_raw!r}; expected "
            "YYYYMMDD_HHMMSSmmm."
        ) from error
    epoch = datetime(1970, 1, 1)
    delta = captured_at - epoch
    return (
        delta.days * 86_400_000_000_000
        + delta.seconds * 1_000_000_000
        + delta.microseconds * 1_000
    )


@dataclass(frozen=True)
class DirectorySourceStats:
    delivered_frame_count: int
    dropped_complete_frame_count: int
    last_timestamp_raw: str | None


class DirectoryRGBDSource:
    """Watch exact ``*_r.png``/``*_d.pgm`` pairs and return the newest frame.

    A new pair must keep the same size and modification timestamp for
    ``stable_interval_s`` before it is decoded. This avoids consuming files
    while the external camera process is still writing them.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        intrinsics: CameraIntrinsics,
        depth_scale: float,
        start_at: StartAt = "latest",
        poll_interval_s: float = 0.05,
        stable_interval_s: float = 0.10,
    ) -> None:
        source_directory = Path(directory).expanduser().resolve()
        if depth_scale <= 0:
            raise ValueError("depth_scale must be positive.")
        if start_at not in ("latest", "new", "oldest"):
            raise ValueError("start_at must be latest, new, or oldest.")
        if poll_interval_s <= 0 or stable_interval_s < 0:
            raise ValueError(
                "poll_interval_s must be positive and stable_interval_s "
                "must be non-negative."
            )
        self.directory = source_directory
        self.intrinsics = intrinsics
        self.depth_scale = float(depth_scale)
        self.start_at = start_at
        self.poll_interval_s = float(poll_interval_s)
        self.stable_interval_s = float(stable_interval_s)
        self._started = False
        self._closed = False
        self._cursor_timestamp_raw: str | None = None
        self._pending_initial_timestamp: str | None = None
        self._signatures: dict[str, tuple[tuple[int, int], tuple[int, int]]] = {}
        self._stable_since: dict[str, float] = {}
        self._delivered_frame_count = 0
        self._dropped_complete_frame_count = 0

    @property
    def source_id(self) -> str:
        return f"directory:{self.directory}"

    @property
    def stats(self) -> DirectorySourceStats:
        return DirectorySourceStats(
            delivered_frame_count=self._delivered_frame_count,
            dropped_complete_frame_count=self._dropped_complete_frame_count,
            last_timestamp_raw=self._cursor_timestamp_raw,
        )

    def _scan_pairs(self) -> dict[str, tuple[Path, Path]]:
        streams_by_timestamp: dict[str, dict[str, Path]] = {}
        for path in self.directory.iterdir():
            if not path.is_file():
                continue
            match = _REQUIRED_FRAME_PATTERN.fullmatch(path.name)
            if match is None:
                continue
            stream = match.group("stream")
            if match.group("extension") != _EXPECTED_EXTENSION[stream]:
                continue
            timestamp = match.group("timestamp")
            streams_by_timestamp.setdefault(timestamp, {})[stream] = path
        return {
            timestamp: (streams["r"], streams["d"])
            for timestamp, streams in streams_by_timestamp.items()
            if "r" in streams and "d" in streams
        }

    @staticmethod
    def _pair_signature(
        pair: tuple[Path, Path],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        rgb_stat = pair[0].stat()
        depth_stat = pair[1].stat()
        return (
            (int(rgb_stat.st_size), int(rgb_stat.st_mtime_ns)),
            (int(depth_stat.st_size), int(depth_stat.st_mtime_ns)),
        )

    def start(self) -> None:
        if self._started and not self._closed:
            return
        if not self.directory.is_dir():
            raise NotADirectoryError(
                f"Live RGB-D directory not found: {self.directory}"
            )
        pairs = self._scan_pairs()
        now = time.monotonic()
        for timestamp, pair in pairs.items():
            self._signatures[timestamp] = self._pair_signature(pair)
            self._stable_since[timestamp] = now
        timestamps = sorted(pairs)
        self._cursor_timestamp_raw = None
        self._pending_initial_timestamp = None
        if timestamps and self.start_at == "new":
            self._cursor_timestamp_raw = timestamps[-1]
        elif timestamps and self.start_at == "latest":
            self._cursor_timestamp_raw = timestamps[-1]
            self._pending_initial_timestamp = timestamps[-1]
        self._started = True
        self._closed = False

    def _eligible_stable_pairs(
        self,
        now: float,
    ) -> list[tuple[str, tuple[Path, Path]]]:
        pairs = self._scan_pairs()
        stable: list[tuple[str, tuple[Path, Path]]] = []
        for timestamp, pair in pairs.items():
            try:
                signature = self._pair_signature(pair)
            except FileNotFoundError:
                continue
            if self._signatures.get(timestamp) != signature:
                self._signatures[timestamp] = signature
                self._stable_since[timestamp] = now
                continue
            if now - self._stable_since.get(timestamp, now) < (
                self.stable_interval_s
            ):
                continue
            is_pending_initial = timestamp == self._pending_initial_timestamp
            is_newer = (
                self._cursor_timestamp_raw is None
                or timestamp > self._cursor_timestamp_raw
            )
            if is_pending_initial or is_newer:
                stable.append((timestamp, pair))
        stable.sort(key=lambda item: item[0])
        return stable

    def _load_pair(
        self,
        timestamp: str,
        pair: tuple[Path, Path],
    ) -> RGBDFrame:
        rgb_bgr = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            raise RuntimeError(f"OpenCV failed to read RGB image: {pair[0]}")
        depth_m = load_depth_m(pair[1], self.depth_scale)
        return RGBDFrame(
            rgb_bgr=rgb_bgr,
            depth_m=depth_m,
            intrinsics=self.intrinsics,
            timestamp_ns=capture_timestamp_ns(timestamp),
            frame_number=self._delivered_frame_count,
            source_id=self.source_id,
        )

    def read(self, timeout_ms: int = 1000) -> RGBDFrame:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive.")
        if not self._started or self._closed:
            raise RuntimeError("DirectoryRGBDSource must be started before read().")
        deadline = time.monotonic() + timeout_ms / 1000.0
        last_error: Exception | None = None
        while not self._closed:
            now = time.monotonic()
            stable = self._eligible_stable_pairs(now)
            if stable:
                selected_index = 0 if self.start_at == "oldest" else -1
                timestamp, pair = stable[selected_index]
                try:
                    frame = self._load_pair(timestamp, pair)
                except (OSError, RuntimeError, TypeError, ValueError) as error:
                    last_error = error
                else:
                    skipped = sum(
                        1
                        for candidate_timestamp, _ in stable
                        if candidate_timestamp < timestamp
                    )
                    self._dropped_complete_frame_count += skipped
                    self._cursor_timestamp_raw = timestamp
                    self._pending_initial_timestamp = None
                    self._delivered_frame_count += 1
                    return frame
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                detail = f" Last decode error: {last_error}" if last_error else ""
                raise TimeoutError(
                    f"No new complete RGB-D pair in {self.directory}." + detail
                )
            time.sleep(min(self.poll_interval_s, remaining))
        raise RuntimeError("DirectoryRGBDSource is closed.")

    def close(self) -> None:
        self._closed = True
