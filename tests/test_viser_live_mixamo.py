"""Tests for latest-only live Mixamo solving and browser refreshes."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import cv2
import numpy as np

import rgbd_avatar.visualization.viser_live_mixamo as live_mixamo


class _FakeIKSolver:
    def __init__(self, outputs: list[object | None]) -> None:
        self.outputs = list(outputs)
        self.delta_times: list[float] = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def solve(self, *_args, delta_time_s: float, **_kwargs):
        self.delta_times.append(float(delta_time_s))
        return self.outputs.pop(0)


def _result(
    frame_number: int,
    timestamp_ns: int,
    *,
    reset: bool = False,
    reacquired: bool = False,
):
    temporal = SimpleNamespace(
        confidence=np.ones(26, dtype=np.float64),
        usable=np.ones(26, dtype=bool),
        predicted=np.zeros(26, dtype=bool),
    )
    presence = SimpleNamespace(
        track_reset_required=reset,
        reacquired_after_exit=reacquired,
    )
    return SimpleNamespace(
        source_id="camera",
        frame_number=frame_number,
        timestamp_ns=timestamp_ns,
        joints_application_m=np.zeros((26, 3), dtype=np.float64),
        pose3d_output=temporal,
        presence=presence,
    )


def test_pose_solver_uses_sensor_time_and_holds_only_within_track(monkeypatch) -> None:
    ik_frame = SimpleNamespace(bone_global_m=np.eye(4)[None])
    solver = _FakeIKSolver([ik_frame, ik_frame, None, None])
    setup = SimpleNamespace(solver=solver, asset=object())
    skin_calls = 0

    def _skin(_asset, _matrices):
        nonlocal skin_calls
        skin_calls += 1
        return np.full((4, 3), float(skin_calls), dtype=np.float64)

    monkeypatch.setattr(live_mixamo, "skin_mixamo_vertices", _skin)
    pose_solver = live_mixamo._LiveMixamoPoseSolver(setup)

    first = pose_solver.solve(
        _result(1, 1_000_000_000), input_version=1
    )
    second = pose_solver.solve(
        _result(2, 1_050_000_000), input_version=2
    )
    held = pose_solver.solve(
        _result(3, 1_100_000_000), input_version=3
    )
    reset = pose_solver.solve(
        _result(4, 2_000_000_000), input_version=4, reset=True
    )

    assert np.allclose(solver.delta_times, [0.033, 0.05, 0.05, 0.033])
    assert first.geometry_version == 1
    assert second.geometry_version == 2
    assert held.geometry_version == 2
    assert held.vertices_display_m is second.vertices_display_m
    assert held.status == "Holding (no valid root)"
    assert solver.reset_count == 1
    assert reset.vertices_display_m is None
    assert reset.status == "Waiting for a valid root"


def test_viewer_update_solves_latest_pose_and_requests_rerender() -> None:
    viewer = object.__new__(live_mixamo.ViserLiveMixamoViewer)
    viewer._condition = threading.Condition()
    viewer._latest_pose = None
    viewer._latest_pose_key = None
    viewer._pending_version = 0
    viewer._reset_pending = False
    viewer._closed = False
    viewer._solved_pose = None
    viewer._camera_focused_on_pose = False
    viewer._viewer_center = np.array([10.0, 10.0, 10.0])
    viewer._view_distance = 10.0

    calls: list[tuple[int, int, bool]] = []

    class _FakePoseSolver:
        def solve(self, result, *, input_version: int, reset: bool):
            calls.append((result.frame_number, input_version, reset))
            return live_mixamo._SolvedLivePose(
                input_version=input_version,
                geometry_version=input_version,
                frame_number=result.frame_number,
                vertices_display_m=np.array(
                    [[-1.0, -2.0, 0.0], [1.0, 2.0, 2.0]],
                    dtype=np.float64,
                ),
                status="Live",
                solve_ms=1.0,
            )

    viewer._pose_solver = _FakePoseSolver()
    rerendered = threading.Event()
    focused = threading.Event()
    viewer.rerender = lambda _: rerendered.set()
    viewer._reset_client_cameras = lambda: focused.set()
    viewer._solve_thread = threading.Thread(
        target=viewer._solve_loop,
        name="test-live-mixamo-solver",
        daemon=True,
    )
    viewer._solve_thread.start()

    result = _result(7, 7_000_000_000, reset=True)
    viewer.update(result)
    assert rerendered.wait(timeout=1.0)
    assert focused.wait(timeout=1.0)
    with viewer._condition:
        assert viewer._solved_pose is not None
        assert viewer._solved_pose.frame_number == 7
        assert np.allclose(viewer._viewer_center, [0.0, 0.0, 1.0])
        assert np.isclose(viewer._view_distance, np.sqrt(24.0) * 1.5)
        pending_version = viewer._pending_version

    # Re-submitting the same sensor frame must not advance stateful IK.
    viewer.update(result)
    with viewer._condition:
        assert viewer._pending_version == pending_version
    assert calls == [(7, 1, True)]
    viewer.close()
    assert not viewer._solve_thread.is_alive()


def test_viewer_drops_stale_solved_pose_before_rerender() -> None:
    viewer = object.__new__(live_mixamo.ViserLiveMixamoViewer)
    viewer._condition = threading.Condition()
    viewer._latest_pose = None
    viewer._latest_pose_key = None
    viewer._pending_version = 0
    viewer._reset_pending = False
    viewer._closed = False
    viewer._solved_pose = None
    viewer._camera_focused_on_pose = True

    first_started = threading.Event()
    release_first = threading.Event()
    rerendered = threading.Event()
    solved_frames: list[int] = []

    class _SlowFirstPoseSolver:
        def solve(self, result, *, input_version: int, reset: bool):
            del reset
            solved_frames.append(result.frame_number)
            if result.frame_number == 1:
                first_started.set()
                assert release_first.wait(timeout=1.0)
            return live_mixamo._SolvedLivePose(
                input_version=input_version,
                geometry_version=input_version,
                frame_number=result.frame_number,
                vertices_display_m=np.zeros((4, 3), dtype=np.float64),
                status="Live",
                solve_ms=1.0,
            )

    viewer._pose_solver = _SlowFirstPoseSolver()
    viewer.rerender = lambda _: rerendered.set()
    viewer._solve_thread = threading.Thread(
        target=viewer._solve_loop,
        name="test-live-mixamo-latest-only",
        daemon=True,
    )
    viewer._solve_thread.start()

    viewer.update(_result(1, 1_000_000_000))
    assert first_started.wait(timeout=1.0)
    viewer.update(_result(2, 1_033_000_000))
    release_first.set()
    assert rerendered.wait(timeout=1.0)
    with viewer._condition:
        assert viewer._solved_pose is not None
        assert viewer._solved_pose.frame_number == 2
    assert solved_frames == [1, 2]
    viewer.close()


def test_live_setup_uses_effective_cache_scale(monkeypatch, tmp_path) -> None:
    model_path = tmp_path / "avatar.fbx"
    model_path.write_bytes(b"placeholder")
    ok, encoded = cv2.imencode(
        ".png",
        np.zeros((2, 2, 3), dtype=np.uint8),
    )
    assert ok
    cache = SimpleNamespace(
        scale=1.75,
        metadata={
            "model": str(model_path),
            "estimated_scale": 9.0,
            "retarget_profile": {"length_priors": {}},
            "ik_config": {},
        },
        diffuse_png=np.frombuffer(encoded, dtype=np.uint8),
        faces=np.array([[0, 1, 2]], dtype=np.int32),
        triangle_uvs=np.zeros((1, 3, 2), dtype=np.float32),
    )
    asset = object()
    captured: dict[str, float] = {}
    solver = object()

    monkeypatch.setattr(
        live_mixamo,
        "MixamoSequenceCache",
        SimpleNamespace(load=lambda _path: cache),
    )
    monkeypatch.setattr(
        live_mixamo,
        "HalpeSMPLRetargetProfile",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(live_mixamo, "load_mixamo_fbx", lambda _path: asset)

    def _make_solver(_asset, _profile, *, scale, config):
        del config
        captured["scale"] = float(scale)
        return solver

    monkeypatch.setattr(live_mixamo, "MixamoAnalyticalIK", _make_solver)
    setup = live_mixamo.load_live_mixamo_setup(tmp_path / "cache.npz")

    assert captured["scale"] == 1.75
    assert setup.solver is solver
