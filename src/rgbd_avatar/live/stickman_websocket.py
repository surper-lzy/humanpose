"""Latest-only stickman publisher for a topic-based WebSocket hub."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
import threading
import time
from typing import Any, Callable, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import uuid

import numpy as np

from rgbd_avatar.pose import HALPE26_NAMES

from .multi_person_processor import LocalMultiPersonPoseResult
from .processor import LivePoseResult


LOGGER = logging.getLogger("stickman_websocket")


@dataclass(frozen=True)
class StickmanPublishConfig:
    """Connection and topic contract for the FastAPI realtime hub."""

    enabled: bool
    url: str
    client_type: str = "python"
    client_id: str = "rgbd-avatar-camera"
    source_id: str = "camera-01"
    event: str = "avatar.stickman.updated"
    topic: str = "avatar:stickman:camera-01"
    open_timeout_s: float = 5.0
    close_timeout_s: float = 2.0
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    reconnect_initial_s: float = 0.5
    reconnect_max_s: float = 10.0

    def __post_init__(self) -> None:
        for field_name in (
            "url",
            "client_type",
            "client_id",
            "source_id",
            "event",
            "topic",
        ):
            if not str(getattr(self, field_name)).strip():
                raise ValueError(f"{field_name} must be non-empty.")
        for field_name in (
            "open_timeout_s",
            "close_timeout_s",
            "ping_interval_s",
            "ping_timeout_s",
            "reconnect_initial_s",
            "reconnect_max_s",
        ):
            if float(getattr(self, field_name)) <= 0.0:
                raise ValueError(f"{field_name} must be positive.")
        if self.reconnect_max_s < self.reconnect_initial_s:
            raise ValueError(
                "reconnect_max_s must be at least reconnect_initial_s."
            )
        if self.enabled:
            if "FASTAPI_HOST" in self.url:
                raise ValueError(
                    "Replace FASTAPI_HOST with the other computer's LAN IP "
                    "before enabling WebSocket publishing."
                )
            parsed = urlsplit(self.url)
            if parsed.scheme not in ("ws", "wss") or not parsed.netloc:
                raise ValueError(
                    "Enabled WebSocket URL must use ws:// or wss:// and "
                    "contain a host."
                )

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None,
        *,
        enabled_override: bool | None = None,
        url_override: str | None = None,
    ) -> "StickmanPublishConfig":
        values = dict(mapping or {})
        values.setdefault(
            "url", "ws://FASTAPI_HOST:8000/api/realtime/ws"
        )
        enabled = bool(values.pop("enabled", False))
        if url_override is not None:
            values["url"] = url_override
            enabled = True
        if enabled_override is not None:
            enabled = enabled_override
        return cls(enabled=enabled, **values)

    @property
    def connection_url(self) -> str:
        """Return the hub URL with the documented connection identity query."""

        parsed = urlsplit(self.url)
        query = dict(parse_qsl(parsed.query, keep_blank_values=True))
        query.update(
            {
                "client_type": self.client_type,
                "client_id": self.client_id,
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
class StickmanPublisherStats:
    submitted_frame_count: int
    overwritten_unsent_frame_count: int
    sent_frame_count: int
    publish_ack_count: int
    successful_connection_count: int
    connected: bool
    last_sent_frame: int | None
    last_ack_message_id: str | None
    last_error: str | None


def build_stickman_payload(result: LivePoseResult) -> dict[str, Any]:
    """Convert one final application-space pose into JSON-safe data."""

    joints = np.asarray(result.joints_application_m, dtype=np.float32)
    usable = np.asarray(result.pose3d_output.usable, dtype=bool)
    count = len(HALPE26_NAMES)
    if joints.shape != (count, 3) or usable.shape != (count,):
        raise ValueError("Live result must contain application Halpe26 joints.")
    usable = usable & np.isfinite(joints).all(axis=1)
    return {
        "schema_version": 1,
        "keypoint_format": "halpe26",
        "coordinate_system": {
            "name": "application",
            "handedness": "right",
            "unit": "meter",
            "up_axis": "+z",
            "ground_z_m": 0.0,
        },
        "source_id": result.source_id,
        "frame_number": int(result.frame_number),
        # Milliseconds remain exactly representable by JavaScript Number.
        "timestamp_ms": int(result.timestamp_ns // 1_000_000),
        "status": str(result.status),
        "joints": [
            [float(value) for value in joints[index]]
            if usable[index]
            else None
            for index in range(count)
        ],
    }


def build_stickmen_payload(
    result: LocalMultiPersonPoseResult,
    *,
    stream_id: str,
) -> dict[str, Any]:
    """Convert tracked people into the additive multi-person wire schema."""

    count = len(HALPE26_NAMES)
    persons: list[dict[str, Any]] = []
    for person in result.persons:
        joints = np.asarray(person.joints_application_m, dtype=np.float32)
        usable = np.asarray(person.pose3d_output.usable, dtype=bool)
        joint_sources = tuple(
            getattr(person, "joint_sources", ("unknown",) * count)
        )
        if joints.shape != (count, 3) or usable.shape != (count,):
            raise ValueError(
                "Every multi-person result must contain Halpe26 joints."
            )
        if len(joint_sources) != count:
            raise ValueError("Every multi-person result must contain 26 joint sources.")
        usable = usable & np.isfinite(joints).all(axis=1)
        if not person.observed_in_frame:
            usable.fill(False)
        persons.append(
            {
                "track_id": int(person.track_id),
                "status": str(person.status),
                "observed_in_frame": bool(person.observed_in_frame),
                "joint_sources": list(joint_sources),
                "kinematic_fallback_joint_count": int(
                    np.count_nonzero(
                        getattr(
                            person,
                            "kinematic_fallback",
                            np.zeros(count, dtype=bool),
                        )
                    )
                ),
                "skeleton_completion_joint_count": int(
                    sum(source == "skeleton_completion" for source in joint_sources)
                ),
                "joints": [
                    [float(value) for value in joints[index]]
                    if usable[index]
                    else None
                    for index in range(count)
                ],
            }
        )
    return {
        "schema_version": 1,
        "keypoint_format": "halpe26",
        "coordinate_system": {
            "name": "application",
            "handedness": "right",
            "unit": "meter",
            "up_axis": "+z",
            "ground_z_m": 0.0,
        },
        "source_id": result.source_id,
        "stream_id": stream_id,
        "frame_number": int(result.frame_number),
        "timestamp_ms": int(result.timestamp_ns // 1_000_000),
        "status": str(result.status),
        "identity_method": str(result.identity_method),
        "identity_fallback": bool(result.identity_fallback),
        "detected_person_count": int(result.detected_person_count),
        "published_person_count": len(persons),
        "persons": persons,
    }


class StickmanWebSocketPublisher:
    """Publish only the newest pose without blocking capture or inference."""

    def __init__(
        self,
        config: StickmanPublishConfig,
        *,
        connect_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._connect_factory = connect_factory
        self._session_id = uuid.uuid4().hex[:12]
        self._lock = threading.Lock()
        self._latest_json: str | None = None
        self._latest_frame: int | None = None
        self._pending = False
        self._submitted_frame_count = 0
        self._overwritten_unsent_frame_count = 0
        self._sent_frame_count = 0
        self._publish_ack_count = 0
        self._successful_connection_count = 0
        self._connected = False
        self._last_sent_frame: int | None = None
        self._last_ack_message_id: str | None = None
        self._last_error: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._latest_event: asyncio.Event | None = None
        self._stop_event: asyncio.Event | None = None
        self._ready = threading.Event()
        self._ack_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def stats(self) -> StickmanPublisherStats:
        with self._lock:
            return StickmanPublisherStats(
                submitted_frame_count=self._submitted_frame_count,
                overwritten_unsent_frame_count=(
                    self._overwritten_unsent_frame_count
                ),
                sent_frame_count=self._sent_frame_count,
                publish_ack_count=self._publish_ack_count,
                successful_connection_count=self._successful_connection_count,
                connected=self._connected,
                last_sent_frame=self._last_sent_frame,
                last_ack_message_id=self._last_ack_message_id,
                last_error=self._last_error,
            )

    def _publish_envelope(self, result: LivePoseResult) -> dict[str, Any]:
        frame = int(result.frame_number)
        payload = build_stickman_payload(result)
        payload["source_id"] = self.config.source_id
        return {
            "type": "publish",
            "message_id": f"pose-{self._session_id}-{frame}",
            "event": self.config.event,
            "topics": [self.config.topic],
            "payload": payload,
        }

    def submit(self, result: LivePoseResult) -> None:
        """Replace the pending pose and return immediately."""

        if not self.config.enabled:
            return
        message = json.dumps(
            self._publish_envelope(result),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock:
            if self._pending:
                self._overwritten_unsent_frame_count += 1
            self._latest_json = message
            self._latest_frame = int(result.frame_number)
            self._pending = True
            self._submitted_frame_count += 1
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
                    "WebSocket publishing requires the 'websockets' package."
                ) from error
            self._connect_factory = connect
        self._ready.clear()
        self._thread = threading.Thread(
            target=self._thread_main,
            name="stickman-websocket-publisher",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=5.0):
            raise RuntimeError("WebSocket publisher thread did not initialize.")

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
            LOGGER.exception("Stickman WebSocket publisher stopped unexpectedly")
        finally:
            with self._lock:
                self._connected = False
                self._loop = None
                self._latest_event = None
                self._stop_event = None
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _wait_or_stop(self, delay_s: float) -> bool:
        assert self._stop_event is not None
        try:
            await asyncio.wait_for(self._stop_event.wait(), timeout=delay_s)
        except asyncio.TimeoutError:
            return False
        return True

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
                    max_size=1_000_000,
                    proxy=None,
                )
                async with context as websocket:
                    with self._lock:
                        self._connected = True
                        self._last_error = None
                        self._successful_connection_count += 1
                        # Re-send the newest pose after every reconnect.
                        if self._latest_json is not None:
                            self._pending = True
                    LOGGER.info(
                        "Stickman WebSocket connected: %s topic=%s",
                        self.config.connection_url,
                        self.config.topic,
                    )
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "hello",
                                "message_id": f"hello-{self._session_id}",
                                "client_type": self.config.client_type,
                                "client_id": self.config.client_id,
                                "topics": [],
                            },
                            separators=(",", ":"),
                        )
                    )
                    assert self._latest_event is not None
                    self._latest_event.set()
                    sender = asyncio.create_task(self._send_loop(websocket))
                    receiver = asyncio.create_task(
                        self._receive_loop(websocket)
                    )
                    stopper = asyncio.create_task(self._stop_event.wait())
                    done, pending = await asyncio.wait(
                        (sender, receiver, stopper),
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        if task is not stopper:
                            task.result()
                    if not self._stop_event.is_set():
                        raise ConnectionError(
                            "FastAPI WebSocket connection closed."
                        )
                reconnect_delay = self.config.reconnect_initial_s
            except asyncio.CancelledError:
                raise
            except Exception as error:
                with self._lock:
                    self._connected = False
                    self._last_error = f"{type(error).__name__}: {error}"
                LOGGER.warning(
                    "Stickman WebSocket unavailable (%s); retrying in %.1f s",
                    error,
                    reconnect_delay,
                )
                if await self._wait_or_stop(reconnect_delay):
                    break
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
                if not self._pending or self._latest_json is None:
                    continue
                message = self._latest_json
                frame = self._latest_frame
                self._pending = False
            await websocket.send(message)
            with self._lock:
                self._sent_frame_count += 1
                self._last_sent_frame = frame

    async def _receive_loop(self, websocket: Any) -> None:
        async for raw_message in websocket:
            try:
                message = json.loads(raw_message)
            except (TypeError, UnicodeDecodeError, json.JSONDecodeError):
                continue
            if not isinstance(message, dict):
                continue
            if message.get("type") == "system.ack":
                with self._lock:
                    if message.get("event") == "publish":
                        self._publish_ack_count += 1
                        self._ack_event.set()
                    reply_to = message.get("reply_to")
                    if isinstance(reply_to, str):
                        self._last_ack_message_id = reply_to

    def stop(self, *, drain_timeout_s: float = 0.5) -> None:
        """Close the publisher after briefly draining in-flight ACKs."""

        if drain_timeout_s < 0.0:
            raise ValueError("drain_timeout_s cannot be negative.")
        thread = self._thread
        if thread is None:
            return
        deadline = time.monotonic() + drain_timeout_s
        while thread.is_alive():
            with self._lock:
                waiting_for_ack = (
                    self._connected
                    and self._publish_ack_count < self._sent_frame_count
                )
            remaining = deadline - time.monotonic()
            if not waiting_for_ack or remaining <= 0.0:
                break
            self._ack_event.wait(timeout=remaining)
            self._ack_event.clear()
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


class StickmenWebSocketPublisher(StickmanWebSocketPublisher):
    """Publish the newest tracked multi-person frame without blocking."""

    def _publish_envelope(
        self,
        result: LocalMultiPersonPoseResult,
    ) -> dict[str, Any]:
        frame = int(result.frame_number)
        payload = build_stickmen_payload(
            result,
            stream_id=self._session_id,
        )
        payload["source_id"] = self.config.source_id
        return {
            "type": "publish",
            "message_id": f"poses-{self._session_id}-{frame}",
            "event": self.config.event,
            "topics": [self.config.topic],
            "payload": payload,
        }
