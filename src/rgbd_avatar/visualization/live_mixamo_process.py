"""Process-isolated controller for the live browser Mixamo viewer.

The local mannequin uses Open3D's GLFW visualizer while the textured browser
viewer uses Filament/EGL. Some Open3D builds cannot keep both native graphics
contexts in one process and abort after ``eglMakeCurrent`` fails. This module
keeps the parent free of Filament imports and starts the browser renderer with
the ``spawn`` multiprocessing context.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty, Full
import traceback
from typing import Any

import numpy as np


LOGGER = logging.getLogger("live_mixamo_process")


@dataclass(frozen=True)
class LiveMixamoPosePacket:
    """Small picklable subset of ``LivePoseResult`` needed by Mixamo IK."""

    source_id: str
    frame_number: int
    timestamp_ns: int
    joints_application_m: np.ndarray
    confidence: np.ndarray
    usable: np.ndarray
    predicted: np.ndarray
    track_reset_required: bool = False
    reacquired_after_exit: bool = False

    def __post_init__(self) -> None:
        joints = np.asarray(self.joints_application_m, dtype=np.float32)
        confidence = np.asarray(self.confidence, dtype=np.float32)
        usable = np.asarray(self.usable, dtype=bool)
        predicted = np.asarray(self.predicted, dtype=bool)
        count = len(joints)
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError("Live Mixamo joints must have shape Jx3.")
        if confidence.shape != (count,):
            raise ValueError("Live Mixamo confidence must have shape J.")
        if usable.shape != (count,) or predicted.shape != (count,):
            raise ValueError("Live Mixamo masks must have shape J.")
        if np.any(usable) and not np.isfinite(joints[usable]).all():
            raise ValueError("Usable live Mixamo joints must be finite.")
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "frame_number", int(self.frame_number))
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))
        object.__setattr__(self, "joints_application_m", joints.copy())
        object.__setattr__(self, "confidence", confidence.copy())
        object.__setattr__(self, "usable", usable.copy())
        object.__setattr__(self, "predicted", predicted.copy())

    @classmethod
    def from_result(cls, result: Any) -> "LiveMixamoPosePacket":
        temporal = result.pose3d_output
        presence = result.presence
        return cls(
            source_id=result.source_id,
            frame_number=result.frame_number,
            timestamp_ns=result.timestamp_ns,
            joints_application_m=result.joints_application_m,
            confidence=temporal.confidence,
            usable=temporal.usable,
            predicted=temporal.predicted,
            track_reset_required=presence.track_reset_required,
            reacquired_after_exit=presence.reacquired_after_exit,
        )

    # ``ViserLiveMixamoViewer`` intentionally consumes a small structural
    # interface instead of importing the parent pipeline's result classes.
    @property
    def pose3d_output(self) -> "LiveMixamoPosePacket":
        return self

    @property
    def presence(self) -> "LiveMixamoPosePacket":
        return self


def _run_live_mixamo_viewer(
    cache_path: str,
    host: str,
    port: int,
    resolution: int,
    pose_queue: Any,
    stop_event: Any,
    status_queue: Any,
) -> None:
    """Child-process entry point; import Open3D only after selecting EGL."""
    os.environ["EGL_PLATFORM"] = "surfaceless"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    viewer = None
    server = None
    try:
        import viser

        from rgbd_avatar.visualization.viser_live_mixamo import (
            ViserLiveMixamoViewer,
            load_live_mixamo_setup,
        )

        setup = load_live_mixamo_setup(cache_path)
        server = viser.ViserServer(host=host, port=port)
        viewer = ViserLiveMixamoViewer(
            server=server,
            setup=setup,
            res=resolution,
        )
        status_queue.put(("ready", ""))

        first_packet = True
        while not stop_event.is_set():
            try:
                packet = pose_queue.get(timeout=0.1)
            except Empty:
                continue
            if packet is None:
                break
            # Collapse any packets already waiting in the inter-process queue.
            while True:
                try:
                    newer = pose_queue.get_nowait()
                except Empty:
                    break
                if newer is None:
                    stop_event.set()
                    break
                packet = newer
            if not stop_event.is_set():
                if first_packet:
                    LOGGER.info(
                        "Live Mixamo child received first pose: source=%s "
                        "frame=%d usable=%d",
                        packet.source_id,
                        packet.frame_number,
                        int(np.count_nonzero(packet.usable)),
                    )
                    first_packet = False
                viewer.update(packet)
    except BaseException:
        try:
            status_queue.put(("error", traceback.format_exc()))
        except Exception:
            pass
        raise
    finally:
        if viewer is not None:
            viewer.close()
        if server is not None:
            server.stop()


class LiveMixamoViewerProcess:
    """Own a spawned browser-viewer process and a bounded latest-pose queue."""

    def __init__(
        self,
        cache_path: str | Path,
        *,
        host: str = "127.0.0.1",
        port: int = 8095,
        res: int = 1024,
    ) -> None:
        source = Path(cache_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Mixamo cache not found: {source}")
        if not 1 <= int(port) <= 65535:
            raise ValueError("Mixamo viewer port must be in [1, 65535].")
        if int(res) <= 0:
            raise ValueError("Mixamo viewer resolution must be positive.")

        self.cache_path = source
        self.host = str(host)
        self.port = int(port)
        self.res = int(res)
        self._context = mp.get_context("spawn")
        self._pose_queue = self._context.Queue(maxsize=2)
        self._status_queue = self._context.Queue(maxsize=2)
        self._stop_event = self._context.Event()
        self._process = self._context.Process(
            target=_run_live_mixamo_viewer,
            args=(
                str(self.cache_path),
                self.host,
                self.port,
                self.res,
                self._pose_queue,
                self._stop_event,
                self._status_queue,
            ),
            name="live-mixamo-viewer",
            daemon=True,
        )
        self._started = False
        self._closed = False

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def is_alive(self) -> bool:
        return bool(self._started and self._process.is_alive())

    def start(self, *, timeout_s: float = 30.0) -> None:
        if self._closed:
            raise RuntimeError("Live Mixamo viewer process is already closed.")
        if self._started:
            return
        self._process.start()
        self._started = True
        try:
            status, detail = self._status_queue.get(timeout=timeout_s)
        except Empty as error:
            exit_code = self._process.exitcode
            self.close()
            raise RuntimeError(
                "Live Mixamo viewer did not start before the timeout "
                f"(exit_code={exit_code})."
            ) from error
        if status != "ready":
            self.close()
            raise RuntimeError(
                "Live Mixamo viewer subprocess failed to start:\n" + detail
            )

    def submit(self, result: Any) -> bool:
        """Submit a compact pose without ever blocking the Open3D GUI loop."""
        if not self._started or self._closed:
            return False
        if not self._process.is_alive():
            raise RuntimeError(
                "Live Mixamo viewer subprocess stopped unexpectedly "
                f"(exit_code={self._process.exitcode})."
            )
        packet = LiveMixamoPosePacket.from_result(result)
        try:
            self._pose_queue.put_nowait(packet)
            return True
        except Full:
            try:
                self._pose_queue.get_nowait()
            except Empty:
                return False
            try:
                self._pose_queue.put_nowait(packet)
                return True
            except Full:
                return False

    def close(self, *, timeout_s: float = 5.0) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        if self._started and self._process.is_alive():
            self._process.join(timeout=timeout_s)
        if self._started and self._process.is_alive():
            LOGGER.warning(
                "Terminating unresponsive live Mixamo viewer subprocess."
            )
            self._process.terminate()
            self._process.join(timeout=2.0)
        for queue in (self._pose_queue, self._status_queue):
            queue.close()
            queue.join_thread()


__all__ = ["LiveMixamoPosePacket", "LiveMixamoViewerProcess"]
