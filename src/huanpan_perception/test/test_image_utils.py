"""Contract tests for the hand-written image conversion (no cv_bridge)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from huanpan_perception.image_utils import (
    encoding_channels,
    image_msg_to_array,
    to_bgr,
    to_rgb,
)


def make_image(array: np.ndarray, encoding: str, step: int | None = None):
    channels = 1 if array.ndim == 2 else array.shape[2]
    row_bytes = array.shape[1] * channels * array.dtype.itemsize
    return SimpleNamespace(
        height=int(array.shape[0]),
        width=int(array.shape[1]),
        encoding=encoding,
        step=int(step if step is not None else row_bytes),
        is_bigendian=False,
        data=np.ascontiguousarray(array).tobytes(),
    )


def test_rgb8_round_trip():
    array = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
    msg = make_image(array, "rgb8")
    assert np.array_equal(image_msg_to_array(msg), array)


def test_bgr8_is_flipped_to_rgb():
    array = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
    msg = make_image(array, "bgr8")
    assert np.array_equal(image_msg_to_array(msg), array[:, :, ::-1])


def test_mono8_is_two_dimensional():
    array = np.arange(20, dtype=np.uint8).reshape(4, 5)
    result = image_msg_to_array(make_image(array, "mono8"))
    assert result.shape == (4, 5)
    assert np.array_equal(result, array)


def test_16uc1_depth_is_read_as_uint16():
    array = (np.arange(20, dtype=np.uint16) * 100).reshape(4, 5)
    result = image_msg_to_array(make_image(array, "16UC1"))
    assert result.dtype == np.uint16
    assert result.shape == (4, 5)
    assert np.array_equal(result, array)


def test_32fc1_depth_is_read_as_float():
    array = np.linspace(0.2, 1.4, 20, dtype=np.float32).reshape(4, 5)
    result = image_msg_to_array(make_image(array, "32FC1"))
    assert result.dtype == np.float32
    assert np.allclose(result, array)


def test_rgba8_drops_the_alpha_channel():
    array = np.random.randint(0, 255, (4, 5, 4), dtype=np.uint8)
    msg = make_image(array, "rgba8")
    result = image_msg_to_array(msg)
    assert result.shape == (4, 5, 3)
    assert np.array_equal(result, array[:, :, :3])


def test_padded_rows_are_stripped():
    array = np.random.randint(0, 255, (4, 5, 3), dtype=np.uint8)
    padding = 6
    step = array.shape[1] * 3 + padding
    padded = np.zeros((4, step), dtype=np.uint8)
    padded[:, : array.shape[1] * 3] = array.reshape(4, -1)
    msg = make_image(array, "rgb8", step=step)
    msg.data = padded.tobytes()
    assert np.array_equal(image_msg_to_array(msg), array)


def test_padded_16uc1_is_stripped_correctly():
    array = (np.arange(15, dtype=np.uint16) * 10).reshape(3, 5)
    step = 5 * 2 + 4
    padded = np.zeros((3, step), dtype=np.uint8)
    padded[:, :10] = array.view(np.uint8).reshape(3, 10)
    msg = make_image(array, "16UC1", step=step)
    msg.data = padded.tobytes()
    assert np.array_equal(image_msg_to_array(msg), array)


def test_unsupported_encoding_raises():
    array = np.zeros((2, 2, 3), dtype=np.uint8)
    with pytest.raises(ValueError):
        image_msg_to_array(make_image(array, "yuv422"))


def test_step_smaller_than_a_row_raises():
    array = np.zeros((2, 4, 3), dtype=np.uint8)
    msg = make_image(array, "rgb8", step=4)
    with pytest.raises(ValueError):
        image_msg_to_array(msg)


def test_encoding_channels():
    assert encoding_channels("rgb8") == 3
    assert encoding_channels("mono8") == 1
    with pytest.raises(ValueError):
        encoding_channels("nope")


def test_to_bgr_and_to_rgb_are_inverse():
    array = np.random.randint(0, 255, (3, 4, 3), dtype=np.uint8)
    assert np.array_equal(to_rgb(to_bgr(array)), array)


def test_to_bgr_expands_mono():
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    bgr = to_bgr(array)
    assert bgr.shape == (3, 4, 3)
    assert np.array_equal(bgr[:, :, 0], array)
