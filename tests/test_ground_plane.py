import numpy as np
import pytest

from rgbd_avatar.depth import (
    GroundPlaneConfig,
    fit_ground_plane_ransac,
    sample_ground_candidates,
)


def synthetic_floor(
    *,
    count: int = 2000,
    noise_m: float = 0.004,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(7)
    tilt = np.deg2rad(20.0)
    normal = np.array([0.02, -np.cos(tilt), -np.sin(tilt)])
    normal /= np.linalg.norm(normal)
    offset = 1.8
    x = rng.uniform(-2.0, 2.0, count)
    z = rng.uniform(1.0, 4.5, count)
    y = -(normal[0] * x + normal[2] * z + offset) / normal[1]
    points = np.column_stack((x, y, z))
    points += rng.normal(scale=noise_m, size=points.shape) * normal
    return points, normal, offset


def test_ground_ransac_recovers_tilted_floor_with_outliers() -> None:
    floor, expected_normal, expected_offset = synthetic_floor()
    rng = np.random.default_rng(8)
    outliers = rng.uniform(
        low=(-2.0, -1.0, 0.5),
        high=(2.0, 1.5, 5.0),
        size=(700, 3),
    )
    estimate = fit_ground_plane_ransac(
        np.concatenate((floor, outliers), axis=0),
        GroundPlaneConfig(
            ransac_iterations=600,
            min_inlier_count=1000,
            inlier_distance_m=0.02,
        ),
        source_frame_count=5,
    )

    np.testing.assert_allclose(
        estimate.normal_camera,
        expected_normal,
        atol=0.005,
    )
    assert estimate.offset_m == pytest.approx(expected_offset, abs=0.01)
    assert estimate.residual_p95_m < 0.012
    assert estimate.source_frame_count == 5


def test_ground_transform_maps_plane_to_world_z_zero() -> None:
    floor, _, _ = synthetic_floor(count=200)
    estimate = fit_ground_plane_ransac(
        floor,
        GroundPlaneConfig(
            ransac_iterations=100,
            min_inlier_count=100,
            inlier_distance_m=0.02,
        ),
    )
    transform = estimate.camera_to_ground_transform()
    homogeneous = np.column_stack((floor, np.ones(len(floor))))
    world = homogeneous @ transform.T

    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1.0)
    assert np.median(np.abs(world[:, 2])) < 0.01
    assert transform[2, 3] == pytest.approx(estimate.camera_height_m)


def test_ground_candidate_sampling_excludes_person_bbox() -> None:
    height, width = 20, 30
    rows, columns = np.indices((height, width), dtype=np.float64)
    organized = np.stack(
        (columns * 0.01, rows * 0.01, np.full((height, width), 2.0)),
        axis=-1,
    )
    config = GroundPlaneConfig(
        lower_image_fraction=0.5,
        pixel_stride=1,
        bbox_exclusion_margin_px=0,
        max_points_per_frame=1000,
        min_inlier_count=10,
    )
    all_candidates = sample_ground_candidates(organized, config)
    excluded = sample_ground_candidates(
        organized,
        config,
        person_bbox_xyxy=np.array([10.0, 10.0, 19.0, 19.0]),
    )

    assert len(all_candidates) == 300
    assert len(excluded) == 200
    assert not np.any(
        (excluded[:, 0] >= 0.10) & (excluded[:, 0] <= 0.19)
    )
