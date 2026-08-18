import numpy as np
import pytest

from rgbd_avatar.avatar.smpl_conversion import _clean_value, _count_chumpy
from rgbd_avatar.avatar.smpl_validation import _validate_mesh


class _FakeChumpy:
    def __init__(self, value) -> None:
        self.r = np.asarray(value)


def test_recursive_clean_replaces_only_chumpy_values() -> None:
    payload = {
        "plain": np.array([1.0]),
        "nested": [_FakeChumpy([2.0, 3.0]), ("keep",)],
    }

    assert _count_chumpy(payload, _FakeChumpy) == 1
    cleaned, count = _clean_value(payload, _FakeChumpy)

    assert count == 1
    assert _count_chumpy(cleaned, _FakeChumpy) == 0
    np.testing.assert_array_equal(cleaned["plain"], [1.0])
    np.testing.assert_array_equal(cleaned["nested"][0], [2.0, 3.0])
    assert cleaned["nested"][1] == ("keep",)


def test_mesh_validation_accepts_closed_surface_and_rejects_boundary() -> None:
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 1, 3],
            [0, 3, 2],
            [1, 2, 3],
        ]
    )

    metrics = _validate_mesh(vertices, faces)
    assert metrics["unique_edges"] == 6
    assert metrics["euler_characteristic"] == 2

    with pytest.raises(ValueError, match="not a closed two-manifold"):
        _validate_mesh(vertices, faces[:-1])
