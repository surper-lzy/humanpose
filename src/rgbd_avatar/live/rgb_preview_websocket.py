"""Latest-only JPEG publisher for the browser RGB skeleton preview."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2

from .multi_person_processor import LocalMultiPersonPoseResult


LOGGER = logging.getLogger("rgb_preview_websocket")


@dataclass(frozen=True)
class RgbPreviewPublishConfig:
    """Transport and JPEG settings for the diagnostic camera overlay."""

    enabled: bool
    url: str = "ws://127.0.0.1:8000/api/realtime/ws"
    source_id: str = "camera-01"
    fps: float = 5.0
    jpeg_quality: int = 75
    scale: float = 0.75
    keypoint_threshold: float = 0.3
    open_timeout_s: float = 5.0
    close_timeout_s: float = 2.0
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    reconnect_initial_s: float = 0.5
    reconnect_max_s: float = 10.0

    def __post_init__(self) -> None:
        if not self.url.strip() or not self.source_id.strip():
            raise ValueError("url and source_id must be non-empty.")
        if self.fps <= 0:
            raise ValueError("fps must be positive.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100].")
        if self.scale <= 0:
            raise ValueError("scale must be positive.")
        if not 0 <= self.keypoint_threshold <= 1:
            raise ValueError("keypoint_threshold must be in [0, 1].")
        for field_name in (
            "open_timeout_s",
            "close_timeout_s",
            "ping_interval_s",
            "ping_timeout_s",
            "reconnect_initial_s",
            "reconnect_max_s",
        ):
            if float(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if self.reconnect_max_s < self.reconnect_initial_s:
            raise ValueError(
                "reconnect_max_s must be at least reconnect_initial_s."
            )
        if self.enabled:
            parsed = urlsplit(self.url)
            if parsed.scheme not in ("ws", "wss") or not parsed.netloc:
                raise ValueError(
                    "Enabled preview URL must be a complete ws:// or wss:// URL."
                )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        enabled_override: bool | None = None,
        url_override: str | None = None,
    ) -> "RgbPreviewPublishConfig":
        values = dict(mapping or {})
        enabled = bool(values.pop("enabled", False))
        if url_override is not None:
            values["url"] = url_override
            enabled = True
        if enabled_override is not None:
            enabled = enabled_override
        return cls(enabled=enabled, **values)

    @property
    def connection_url(self) -> str:
        parsed = urlsplit(self.url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.update(
            {
                "preview_role": "producer",
                "preview_source_id": self.source_id,
            }
        )
        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                urlencode(query),
                parsed.fragment,
            )
        )


@dataclass(frozen=True)
class RgbPreviewPublisherStats:
    submitted_frame_count: int
    rate_limited_frame_count: int
    overwritten_unsent_frame_count: int
    sent_frame_count: int
    successful_connection_count: int
    connected: bool
    last_sent_frame: int | None
    last_jpeg_bytes: int | None
    last_error: str | None


def encode_rgb_preview_jpeg(
    result: LocalMultiPersonPoseResult,
    config: RgbPreviewPublishConfig,
) -> bytes:
    """Draw the existing stable-ID Halpe26 overlay and encode one JPEG."""

    # Import lazily so a headless run that disables preview doesn't pull in
    # visualization modules or add any drawing work to the pose path.
    from rgbd_avatar.visualization.live_multi_person import (
        build_local_multi_rgb_views,
    )

    _raw, overlay = build_local_multi_rgb_views(
        result,
        keypoint_threshold=config.keypoint_threshold,
        scale=config.scale,
    )
    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        overlay,
        [cv2.IMWRITE_JPEG_QUALITY, config.jpeg_quality],
    )
    if not encoded_ok:
        raise RuntimeError("OpenCV failed to encode the RGB preview JPEG.")
    return encoded.tobytes()


class RgbPreviewWebSocketPublisher:
    """Encode off the pose thread and send only the newest accepted frame."""

    def __init__(
        self,
        config: RgbPreviewPublishConfig,
        *,
        connect_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._connect_factory = connect_factory
        self._lock = threading.Lock()
        self._latest_result: LocalMultiPersonPoseResult | None = None
        self._pending = False
        self._last_accepted_at = float("-inf")
        self._submitted_frame_count = 0
        self._rate_limited_frame_count = 0
        self._overwritten_unsent_frame_count = 0
        self._sent_frame_count = 0
        self._successful_connection_count = 0
        self._connected = False
        self._last_sent_frame: int | None = None
        self._last_jpeg_bytes: int | None = None
        self._last_error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._latest_event: asyncio.Event | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def stats(self) -> RgbPreviewPublisherStats:
        with self._lock:
            return RgbPreviewPublisherStats(
                submitted_frame_count=self._submitted_frame_count,
                rate_limited_frame_count=self._rate_limited_frame_count,
                overwritten_unsent_frame_count=self._overwritten_unsent_frame_count,
                sent_frame_count=self._sent_frame_count,
                successful_connection_count=self._successful_connection_count,
                connected=self._connected,
                last_sent_frame=self._last_sent_frame,
                last_jpeg_bytes=self._last_jpeg_bytes,
                last_error=self._last_error,
            )

    def submit(self, result: LocalMultiPersonPoseResult) -> None:
        if not self.config.enabled:
            return
        now = time.monotonic()
        with self._lock:
            self._submitted_frame_count += 1
            if now - self._last_accepted_at < 1.0 / self.config.fps:
                self._rate_limited_frame_count += 1
                return
            self._last_accepted_at = now
            if self._pending:
                self._overwritten_unsent_frame_count += 1
            # LocalMultiPersonPoseResult is immutable by contract. Keeping the
            # newest result here makes submit() constant-time and prevents
            # OpenCV drawing/JPEG encoding from extending the pose frame gap.
            self._latest_result = result
            self._pending = True
            loop = self._loop
            latest_event = self._latest_event
        if loop is not None and latest_event is not None:
            loop.call_soon_threadsafe(latest_event.set)

    def start(self) -> None:
        if not self.config.enabled:
            return
        if self._thread is not None and self._thread.is_alive():
            return
        if self._connect_factory is None:
            try:
                from websockets.asyncio.client import connect
            except ImportError as error:
                raise RuntimeError(
                    "RGB preview publishing requires the 'websockets' package."
                ) from error
            self._connect_factory = connect
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="rgb-preview-websocket-publisher",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("RGB preview publisher thread did not initialize.")

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._latest_event = asyncio.Event()
        self._stop_event = asyncio.Event()
        self._ready.set()
        try:
            loop.run_until_complete(self._run())
        except Exception as error:
            with self._lock:
                self._last_error = f"{type(error).__name__}: {error}"
            LOGGER.exception("RGB preview publisher stopped unexpectedly")
        finally:
            with self._lock:
                self._connected = False
                self._loop = None
                self._latest_event = None
                self._stop_event = None
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _run(self) -> None:
        assert self._connect_factory is not None
        assert self._stop_event is not None
        reconnect_delay = self.config.reconnect_initial_s
        while not self._stop_event.is_set():
            try:
                context = self._connect_factory(
                    self.config.connection_url,
                    open_timeout=self.config.open_timeout_s,
                    close_timeout=self.config.close_timeout_s,
                    ping_interval=self.config.ping_interval_s,
                    ping_timeout=self.config.ping_timeout_s,
                    compression=None,
                    max_size=2 * 1024 * 1024,
                    proxy=None,
                )
                async with context as websocket:
                    with self._lock:
                        self._connected = True
                        self._last_error = None
                        self._successful_connection_count += 1
                        if self._latest_result is not None:
                            self._pending = True
                    LOGGER.info(
                        "RGB preview WebSocket connected: %s source=%s",
                        self.config.connection_url,
                        self.config.source_id,
                    )
                    assert self._latest_event is not None
                    self._latest_event.set()
                    sender = asyncio.create_task(self._send_loop(websocket))
                    stopper = asyncio.create_task(self._stop_event.wait())
                    done, pending = await asyncio.wait(
                        (sender, stopper),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        if task is not stopper:
                            task.result()
                    if not self._stop_event.is_set():
                        raise ConnectionError("Preview WebSocket connection closed.")
                reconnect_delay = self.config.reconnect_initial_s
            except asyncio.CancelledError:
                raise
            except Exception as error:
                with self._lock:
                    self._connected = False
                    self._last_error = f"{type(error).__name__}: {error}"
                LOGGER.warning(
                    "RGB preview WebSocket unavailable (%s); retrying in %.1f s",
                    error,
                    reconnect_delay,
                )
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=reconnect_delay,
                    )
                except asyncio.TimeoutError:
                    pass
                reconnect_delay = min(
                    self.config.reconnect_max_s,
                    reconnect_delay * 2.0,
                )
            finally:
                with self._lock:
                    self._connected = False

    async def _send_loop(self, websocket: Any) -> None:
        assert self._latest_event is not None
        assert self._stop_event is not None
        while not self._stop_event.is_set():
            await self._latest_event.wait()
            self._latest_event.clear()
            with self._lock:
                if not self._pending or self._latest_result is None:
                    continue
                result = self._latest_result
                self._pending = False
            try:
                # This coroutine runs on the dedicated publisher thread. Keep
                # all OpenCV overlay/JPEG work away from the pose worker.
                jpeg = encode_rgb_preview_jpeg(result, self.config)
            except Exception as error:
                with self._lock:
                    self._last_error = f"{type(error).__name__}: {error}"
                LOGGER.exception("RGB skeleton preview encoding failed")
                continue
            await websocket.send(jpeg)
            with self._lock:
                self._sent_frame_count += 1
                self._last_sent_frame = int(result.frame_number)
                self._last_jpeg_bytes = len(jpeg)
                self._last_error = None

    def stop(self) -> None:
        thread = self._thread
        if thread is None:
            return
        with self._lock:
            loop = self._loop
            stop_event = self._stop_event
            latest_event = self._latest_event
        if loop is not None and stop_event is not None:
            loop.call_soon_threadsafe(stop_event.set)
            if latest_event is not None:
                loop.call_soon_threadsafe(latest_event.set)
        if thread.is_alive():
            thread.join(
                timeout=(
                    self.config.open_timeout_s
                    + self.config.close_timeout_s
                    + 1.0
                )
            )
