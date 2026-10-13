"""Image conversion helpers that do not depend on ``cv_bridge``.

``cv_bridge`` from ROS 2 Humble is compiled against NumPy 1.x and raises
``AttributeError: _ARRAY_API not found`` when imported with the NumPy 2.x that
the perception virtualenv uses. Converting ``sensor_msgs/Image`` by hand is a
few lines and removes that coupling entirely.
"""

from __future__ import annotations

import numpy as np

# encoding -> (channels, numpy dtype, needs colour flip to RGB)
_ENCODINGS = {
    "rgb8": (3, np.uint8, False),
    "bgr8": (3, np.uint8, True),
    "rgba8": (4, np.uint8, False),
    "bgra8": (4, np.uint8, True),
    "mono8": (1, np.uint8, False),
    "8UC1": (1, np.uint8, False),
    "8UC3": (3, np.uint8, False),
    "mono16": (1, np.uint16, False),
    "16UC1": (1, np.uint16, False),
    "32FC1": (1, np.float32, False),
}


def encoding_channels(encoding: str) -> int:
    if encoding not in _ENCODINGS:
        raise ValueError(f"unsupported image encoding: {encoding!r}")
    return _ENCODINGS[encoding][0]


def image_msg_to_array(msg) -> np.ndarray:
    """Convert a ``sensor_msgs/Image`` to a NumPy array.

    The result keeps the message's row padding removed: it is always
    ``(height, width)`` for single-channel encodings and ``(height, width, 3)``
    for colour. Colour data is returned as RGB.
    """
    if msg.encoding not in _ENCODINGS:
        raise ValueError(
            f"unsupported image encoding: {msg.encoding!r}; "
            f"supported: {sorted(_ENCODINGS)}"
        )
    channels, dtype, flip = _ENCODINGS[msg.encoding]

    itemsize = np.dtype(dtype).itemsize
    height, width, step = int(msg.height), int(msg.width), int(msg.step)
    row_bytes = width * channels * itemsize
    if step < row_bytes:
        raise ValueError(
            f"image step {step} is smaller than one row ({row_bytes} bytes)"
        )

    buffer = np.frombuffer(msg.data, dtype=np.uint8)
    if step == row_bytes:
        rows = buffer[: height * row_bytes].reshape(height, row_bytes)
    else:
        rows = np.ascontiguousarray(
            buffer[: height * step].reshape(height, step)[:, :row_bytes]
        )

    # One row holds ``width * channels`` elements of the payload dtype.
    array = rows.view(dtype).reshape(height, width, channels)

    if channels == 1:
        array = array[:, :, 0]
    elif channels == 4:
        array = array[:, :, :3]
        if flip:
            array = array[:, :, ::-1]
    elif flip:
        array = array[:, :, ::-1]

    return np.ascontiguousarray(array)


def array_to_rgb8_image_msg(array: np.ndarray, frame_id: str = "", stamp=None):
    """Build a ``sensor_msgs/Image`` with ``rgb8`` encoding from a NumPy array.

    Imported lazily so that the pure-NumPy helpers in this module stay usable
    without a ROS environment.
    """
    from sensor_msgs.msg import Image

    data = np.ascontiguousarray(array)
    if data.dtype != np.uint8:
        data = np.clip(data, 0, 255).astype(np.uint8)
    if data.ndim == 2:
        data = np.repeat(data[:, :, None], 3, axis=2)
    if data.shape[2] == 4:
        data = data[:, :, :3]

    msg = Image()
    msg.height, msg.width = int(data.shape[0]), int(data.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(data.shape[1]) * 3
    msg.data = data.tobytes()
    msg.header.frame_id = frame_id
    if stamp is not None:
        msg.header.stamp = stamp
    return msg


def to_bgr(array: np.ndarray) -> np.ndarray:
    """RGB array (or mono) to a BGR array, the layout OpenCV and YOLO expect."""
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)[:, :, ::-1].copy()
    return np.ascontiguousarray(array[:, :, ::-1])


def to_rgb(array: np.ndarray) -> np.ndarray:
    """BGR array (or mono) to an RGB array."""
    if array.ndim == 2:
        return np.repeat(array[:, :, None], 3, axis=2)
    return np.ascontiguousarray(array[:, :, ::-1])
