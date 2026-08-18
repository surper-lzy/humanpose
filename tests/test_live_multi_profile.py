from pathlib import Path

from rgbd_avatar.io import load_yaml_mapping
from rgbd_avatar.pipeline.live_mannequin import (
    _build_bone_components,
    _build_temporal_filter,
)
from rgbd_avatar.live import KinematicFallbackConfig, Pose3DQualityConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_live_multi_profile_uses_connected_depth_and_coherent_stabilization() -> None:
    live = load_yaml_mapping(PROJECT_ROOT / "configs/live.yaml")["live"]
    tracking = load_yaml_mapping(PROJECT_ROOT / "configs/tracking.yaml")[
        "tracking"
    ]
    multi = live["multi_person"]
    fallback = KinematicFallbackConfig.from_mapping(
        multi["kinematic_fallback"]
    )
    quality = Pose3DQualityConfig.from_mapping(tracking["pose3d_quality"])

    temporal = _build_temporal_filter(
        tracking,
        multi["one_euro"],
        max_prediction_s_override=multi["max_prediction_s"],
    )
    _, constraint = _build_bone_components(
        tracking,
        project_observed_override=multi["bone_stabilization"][
            "project_observed"
        ],
    )

    assert multi["recovery_method"] == "depth_connected"
    assert temporal.shared_cutoff
    assert temporal.max_prediction_s == 0.12
    assert fallback.enabled
    assert fallback.max_age_s == 0.25
    assert fallback.complete_skeleton
    assert fallback.reconstruct_from_current_2d
    assert fallback.min_core_2d_joint_count == 2
    assert fallback.min_core_3d_joint_count == 1
    assert fallback.min_history_joint_count == 8
    assert quality.max_spine_projection_ratio == 1.35
    assert quality.spine_projection_slack_m == 0.10
    assert temporal.shared_speed_percentile == 75.0
    assert temporal._filters[0].min_cutoff_hz == 1.5
    assert temporal._filters[0].beta == 1.0
    assert constraint is not None
    assert constraint.project_observed
    assert constraint.fixed_joint_indices == (19,)
