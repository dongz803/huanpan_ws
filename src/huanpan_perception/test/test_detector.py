"""Detector wrapper tests that do not require Torch."""

from __future__ import annotations

import numpy as np
import pytest

from huanpan_perception.detector import Detection, DetectorError, RackDetector
from huanpan_perception.viz import draw_detections


def test_detection_geometry_helpers():
    detection = Detection(
        class_id=0, class_name="server_unit", confidence=0.9, xyxy=(10.0, 20.0, 110.0, 70.0)
    )
    assert detection.center == (60.0, 45.0)
    assert detection.width == 100.0
    assert detection.height == 50.0
    assert detection.area == 5000.0


def test_largest_prefers_confidence():
    detector = RackDetector("model.pt")
    small = Detection(0, "server_unit", 0.4, (0, 0, 10, 10))
    big = Detection(0, "server_unit", 0.9, (0, 0, 200, 200))
    assert detector.largest([small, big]) is big
    assert detector.largest([]) is None


def test_missing_weights_report_an_actionable_error(tmp_path):
    detector = RackDetector(tmp_path / "does_not_exist.pt")
    with pytest.raises(DetectorError) as excinfo:
        detector.load()
    message = str(excinfo.value)
    assert "not found" in message
    assert "train_detector" in message


def test_wrong_input_shape_is_rejected_before_loading(tmp_path):
    detector = RackDetector(tmp_path / "does_not_exist.pt")
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10, 4), dtype=np.uint8))


def test_default_class_names():
    detector = RackDetector("model.pt")
    assert detector.class_names == ("server_unit",)
    assert not detector.loaded


def test_detection_preview_can_be_downscaled_without_changing_source():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    original = frame.copy()
    detection = Detection(0, "slot_opening", 0.9, (20, 20, 180, 80))

    preview = draw_detections(
        frame, [detection], crosshair=True, max_width=100
    )

    assert preview.shape == (50, 100, 3)
    assert np.array_equal(frame, original)
    assert np.any(preview)
