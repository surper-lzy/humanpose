"""Offline RGB-D recording discovery and frame metadata."""

from .sequence import (
    RGBDFramePaths,
    discover_rgbd_sequence,
    split_at_time_gaps,
)
from .records import load_hand_records, load_pose_records

__all__ = [
    "RGBDFramePaths",
    "discover_rgbd_sequence",
    "split_at_time_gaps",
    "load_hand_records",
    "load_pose_records",
]
