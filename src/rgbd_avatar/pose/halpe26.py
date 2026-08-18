"""Halpe26 keypoint names and skeleton connectivity."""

HALPE26_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)

# Index pairs are kept explicit so downstream depth and retargeting code does
# not depend on MMPose's visualizer internals.
HALPE26_LINKS: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),
    (11, 19),
    (16, 14),
    (14, 12),
    (12, 19),
    (17, 18),
    (18, 19),
    (18, 5),
    (5, 7),
    (7, 9),
    (18, 6),
    (6, 8),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (15, 20),
    (15, 22),
    (15, 24),
    (16, 21),
    (16, 23),
    (16, 25),
)

# Core anatomical links used by the first bone-length constraint baseline.
# Unlike HALPE26_LINKS, this deliberately excludes the disconnected face
# cycle and all foot leaves, whose RGB-D depths are not yet stabilized by a
# person mask or ground/contact model.
HALPE26_CONSTRAINT_LINKS: tuple[tuple[int, int], ...] = (
    (19, 11),
    (11, 13),
    (13, 15),
    (19, 12),
    (12, 14),
    (14, 16),
    (19, 18),
    (18, 17),
    (18, 5),
    (5, 7),
    (7, 9),
    (18, 6),
    (6, 8),
    (8, 10),
)

# Limb lengths receive a tight 5% band. Hip/shoulder/head offsets receive 8%,
# while neck-to-hip is intentionally soft because its Euclidean distance can
# change during bending even though anatomical spine segment lengths do not.
HALPE26_CONSTRAINT_TOLERANCE_RATIOS: tuple[float, ...] = (
    0.08,
    0.05,
    0.05,
    0.08,
    0.05,
    0.05,
    0.15,
    0.08,
    0.08,
    0.05,
    0.05,
    0.08,
    0.05,
    0.05,
)

HALPE26_INDEX: dict[str, int] = {
    name: index for index, name in enumerate(HALPE26_NAMES)
}
