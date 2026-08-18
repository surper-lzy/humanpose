import asyncio
import json
from types import SimpleNamespace
import time

import numpy as np
import pytest

from rgbd_avatar.live import (
    StickmanPublishConfig,
    StickmenWebSocketPublisher,
    StickmanWebSocketPublisher,
    build_stickmen_payload,
    build_stickman_payload,
)
from rgbd_avatar.pipeline.live_multi_person import _build_multi_publish_config


def _result(frame: int, *, missing: int | None = None):
    joints = np.arange(26 * 3, dtype=np.float32).reshape(26, 3) * 0.01
    usable = np.ones(26, dtype=bool)
    if missing is not None:
        usable[missing] = False
        joints[missing] = np.nan
    return SimpleNamespace(
        frame_number=frame,
        timestamp_ns=1_785_927_947_534_505_000 + frame * 1_000_000,
        source_id="lx-sdk:id:FF6690772788",
        status="ok",
        joints_application_m=joints,
        pose3d_output=SimpleNamespace(usable=usable),
    )


def _config(**overrides) -> StickmanPublishConfig:
    values = {
        "enabled": True,
        "url": "ws://192.168.30.10:8000/api/realtime/ws?existing=1",
        "client_id": "rgbd-avatar-FF6690772788",
        "source_id": "FF6690772788",
        "topic": "avatar:stickman:FF6690772788",
        "open_timeout_s": 0.2,
        "close_timeout_s": 0.2,
        "reconnect_initial_s": 0.01,
        "reconnect_max_s": 0.02,
    }
    values.update(overrides)
    return StickmanPublishConfig(**values)


def _multi_result(frame: int):
    first = _result(frame, missing=3)
    second = _result(frame, missing=7)
    return SimpleNamespace(
        frame_number=frame,
        timestamp_ns=first.timestamp_ns,
        source_id=first.source_id,
        status="ok",
        identity_method="shadow",
        identity_fallback=False,
        detected_person_count=2,
        persons=(
            SimpleNamespace(
                track_id=4,
                status="ok",
                observed_in_frame=True,
                joints_application_m=first.joints_application_m,
                pose3d_output=first.pose3d_output,
            ),
            SimpleNamespace(
                track_id=9,
                status="temporarily_missing",
                observed_in_frame=False,
                joints_application_m=second.joints_application_m,
                pose3d_output=second.pose3d_output,
            ),
        ),
    )


def test_stickman_payload_contains_only_json_safe_final_joints() -> None:
    payload = build_stickman_payload(_result(7, missing=3))

    assert payload["schema_version"] == 1
    assert payload["keypoint_format"] == "halpe26"
    assert payload["coordinate_system"]["up_axis"] == "+z"
    assert payload["frame_number"] == 7
    assert payload["timestamp_ms"] == 1_785_927_947_541
    assert len(payload["joints"]) == 26
    assert payload["joints"][3] is None
    assert payload["joints"][4] == pytest.approx([0.12, 0.13, 0.14])
    json.dumps(payload, allow_nan=False)


def test_stickmen_payload_preserves_independent_tracks() -> None:
    payload = build_stickmen_payload(_multi_result(8), stream_id="run-1")

    assert payload["stream_id"] == "run-1"
    assert payload["identity_method"] == "shadow"
    assert payload["published_person_count"] == 2
    assert [person["track_id"] for person in payload["persons"]] == [4, 9]
    assert payload["persons"][0]["joints"][3] is None
    assert len(payload["persons"][0]["joint_sources"]) == 26
    assert payload["persons"][0]["skeleton_completion_joint_count"] == 0
    assert payload["persons"][1]["observed_in_frame"] is False
    assert all(joint is None for joint in payload["persons"][1]["joints"])
    json.dumps(payload, allow_nan=False)


def test_multi_publisher_uses_additive_event_without_changing_single_event() -> None:
    multi = StickmenWebSocketPublisher(
        _config(event="avatar.stickmen.updated")
    )._publish_envelope(_multi_result(8))
    single = StickmanWebSocketPublisher(_config())._publish_envelope(_result(8))

    assert multi["event"] == "avatar.stickmen.updated"
    assert multi["message_id"].startswith("poses-")
    assert len(multi["payload"]["persons"]) == 2
    assert single["event"] == "avatar.stickman.updated"
    assert "persons" not in single["payload"]


def test_publish_config_cli_url_enables_and_adds_connection_identity() -> None:
    config = StickmanPublishConfig.from_mapping(
        {
            "enabled": False,
            "url": "ws://FASTAPI_HOST:8000/api/realtime/ws",
            "client_id": "camera-a",
            "source_id": "camera-a",
            "topic": "avatar:stickman:camera-a",
        },
        url_override="ws://10.0.0.8:8000/api/realtime/ws?x=1",
    )

    assert config.enabled
    assert config.connection_url.startswith(
        "ws://10.0.0.8:8000/api/realtime/ws?"
    )
    assert "x=1" in config.connection_url
    assert "client_type=python" in config.connection_url
    assert "client_id=camera-a" in config.connection_url


def test_multi_publish_does_not_inherit_single_enabled_or_event() -> None:
    args = SimpleNamespace(publish_stickmen=None, publish_url=None)
    config = _build_multi_publish_config(
        args,
        {
            "websocket_publish": {
                "enabled": True,
                "url": "ws://10.0.0.8:8000/api/realtime/ws",
                "event": "avatar.stickman.updated",
                "topic": "avatar:stickman:camera-a",
            },
            "multi_person": {"websocket_publish": {"enabled": False}},
        },
    )

    assert not config.enabled
    assert config.event == "avatar.stickmen.updated"
    assert config.topic == "avatar:stickman:camera-a"


class _FakeWebSocket:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.incoming: asyncio.Queue[str] = asyncio.Queue()

    async def send(self, raw_message: str) -> None:
        message = json.loads(raw_message)
        self.sent.append(message)
        if message["type"] == "publish":
            await self.incoming.put(
                json.dumps(
                    {
                        "type": "system.ack",
                        "event": "publish",
                        "reply_to": message["message_id"],
                        "payload": {"published_to": 1},
                    }
                )
            )

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        return await self.incoming.get()


class _FakeConnectionContext:
    def __init__(self, websocket: _FakeWebSocket) -> None:
        self.websocket = websocket

    async def __aenter__(self) -> _FakeWebSocket:
        return self.websocket

    async def __aexit__(self, *_args) -> None:
        return None


def test_publisher_keeps_latest_frame_and_uses_documented_hub_envelope() -> None:
    websocket = _FakeWebSocket()
    connect_calls = []

    def connect_factory(uri: str, **kwargs):
        connect_calls.append((uri, kwargs))
        return _FakeConnectionContext(websocket)

    publisher = StickmanWebSocketPublisher(
        _config(), connect_factory=connect_factory
    )
    # Queue before starting makes latest-only replacement deterministic.
    publisher.submit(_result(1))
    publisher.submit(_result(2))
    publisher.start()
    deadline = time.monotonic() + 1.0
    while publisher.stats.publish_ack_count < 1 and time.monotonic() < deadline:
        time.sleep(0.005)
    publisher.stop()

    assert connect_calls
    assert "client_type=python" in connect_calls[0][0]
    assert websocket.sent[0]["type"] == "hello"
    publishes = [message for message in websocket.sent if message["type"] == "publish"]
    assert len(publishes) == 1
    message = publishes[0]
    assert message["event"] == "avatar.stickman.updated"
    assert message["topics"] == ["avatar:stickman:FF6690772788"]
    assert message["payload"]["frame_number"] == 2
    assert message["payload"]["source_id"] == "FF6690772788"
    assert publisher.stats.submitted_frame_count == 2
    assert publisher.stats.overwritten_unsent_frame_count == 1
    assert publisher.stats.sent_frame_count == 1
    assert publisher.stats.publish_ack_count == 1


def test_publisher_stop_rejects_negative_drain_timeout() -> None:
    publisher = StickmanWebSocketPublisher(_config())

    with pytest.raises(ValueError, match="cannot be negative"):
        publisher.stop(drain_timeout_s=-0.1)


def test_publisher_connects_to_real_websocket_transport() -> None:
    from websockets.asyncio.server import serve

    async def scenario() -> None:
        received: list[dict] = []
        publish_received = asyncio.Event()

        async def handler(websocket) -> None:
            async for raw_message in websocket:
                message = json.loads(raw_message)
                received.append(message)
                if message["type"] == "publish":
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "system.ack",
                                "event": "publish",
                                "reply_to": message["message_id"],
                                "payload": {"published_to": 1},
                            }
                        )
                    )
                    publish_received.set()

        async with serve(handler, "127.0.0.1", 0) as server:
            await server.start_serving()
            if not server.sockets:
                pytest.skip("Loopback sockets are unavailable in this sandbox.")
            port = server.sockets[0].getsockname()[1]
            publisher = StickmanWebSocketPublisher(
                _config(url=f"ws://127.0.0.1:{port}/api/realtime/ws")
            )
            publisher.start()
            publisher.submit(_result(9))
            await asyncio.wait_for(publish_received.wait(), timeout=2.0)
            deadline = time.monotonic() + 1.0
            while (
                publisher.stats.publish_ack_count < 1
                and time.monotonic() < deadline
            ):
                await asyncio.sleep(0.005)
            await asyncio.to_thread(publisher.stop)

        assert [message["type"] for message in received[:2]] == [
            "hello",
            "publish",
        ]
        assert received[1]["payload"]["frame_number"] == 9
        assert publisher.stats.publish_ack_count == 1

    asyncio.run(scenario())
