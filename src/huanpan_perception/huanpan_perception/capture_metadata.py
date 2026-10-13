"""ROS-agnostic helpers for synchronized capture metadata."""

from __future__ import annotations

import re
from pathlib import Path


def stamp_to_ns(stamp) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def next_capture_index(image_dir: Path, depth_dir: Path, prefix: str) -> int:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)\.png$")
    indices: list[int] = []
    for directory in (image_dir, depth_dir):
        if not directory.exists():
            continue
        for path in directory.glob(f"{prefix}_*.png"):
            match = pattern.match(path.name)
            if match:
                indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def camera_info_dict(msg) -> dict:
    return {
        "frame_id": msg.header.frame_id,
        "width": int(msg.width),
        "height": int(msg.height),
        "distortion_model": msg.distortion_model,
        "d": [float(value) for value in msg.d],
        "k": [float(value) for value in msg.k],
        "r": [float(value) for value in msg.r],
        "p": [float(value) for value in msg.p],
        "binning_x": int(msg.binning_x),
        "binning_y": int(msg.binning_y),
        "roi": {
            "x_offset": int(msg.roi.x_offset),
            "y_offset": int(msg.roi.y_offset),
            "height": int(msg.roi.height),
            "width": int(msg.roi.width),
            "do_rectify": bool(msg.roi.do_rectify),
        },
    }
