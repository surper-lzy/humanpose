import numpy as np

from rgbd_avatar.pipeline.dynamic_gaussian_avatar_viewer import (
    build_dynamic_avatar_samples,
    build_dynamic_stick_samples,
)
from rgbd_avatar.scene import build_manual_scene_alignment

from test_scene_alignment import _cache, _placement


def test_dynamic_samples_apply_one_alignment_to_every_frame() -> None:
    cache = _cache()
    cache.vertices_display_m[1] = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    alignment = build_manual_scene_alignment(_placement())

    active, samples = build_dynamic_avatar_samples(
        cache,
        alignment,
        sample_count=20,
        seed=5,
    )

    np.testing.assert_array_equal(active, [1])
    assert samples.shape == (1, 20, 3)
    assert np.isfinite(samples).all()


def test_dynamic_stick_samples_follow_smpl24_bones() -> None:
    cache = _cache()
    alignment = build_manual_scene_alignment(_placement())

    active, samples, radii_m = build_dynamic_stick_samples(
        cache,
        alignment,
        samples_per_bone=2,
    )

    np.testing.assert_array_equal(active, [1])
    assert samples.shape == (1, 23 * 2 + 24, 3)
    assert radii_m.shape == (23 * 2 + 24,)
    assert radii_m[-24 + 15] == np.float32(0.105)
    assert np.isfinite(samples).all()
