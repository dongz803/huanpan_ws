"""Tests for the orientation-only front-view stage."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from huanpan_perception.fine_localization import DepthPlane
from huanpan_perception.front_view import (
    FrontViewGate,
    estimate_front_view,
    estimate_roll_from_horizontal_edges,
)


def _plane(normal) -> DepthPlane:
    return DepthPlane(
        normal=np.asarray(normal, dtype=np.float64),
        distance_m=0.4,
        median_depth_m=0.4,
        valid_ratio=1.0,
        inlier_ratio=1.0,
        rmse_m=0.0,
    )


def test_front_view_converts_panel_normal_to_orientation_only_correction():
    angle = np.deg2rad(10.0)
    measurement = estimate_front_view(
        _plane([np.sin(angle), 0.0, np.cos(angle)]), -4.0
    )

    assert measurement.tilt_deg == pytest.approx(10.0)
    assert np.rad2deg(measurement.correction_rotvec[1]) == pytest.approx(10.0)
    assert np.rad2deg(measurement.correction_rotvec[2]) == pytest.approx(-4.0)


def test_roll_estimator_follows_long_rotated_rectangle_edges():
    image = np.full((300, 500, 3), 215, dtype=np.uint8)
    rectangle = ((250.0, 150.0), (300.0, 80.0), -7.0)
    points = cv2.boxPoints(rectangle).astype(np.int32)
    cv2.polylines(image, [points], True, (10, 10, 10), 5, cv2.LINE_AA)

    angle, quality = estimate_roll_from_horizontal_edges(
        image, (80.0, 70.0, 420.0, 230.0)
    )

    assert angle == pytest.approx(-7.0, abs=1.5)
    assert quality > 0.25


def test_front_view_gate_requires_consecutive_aligned_frames():
    gate = FrontViewGate(window_size=3, confirm_frames=3)
    measurement = estimate_front_view(_plane([0.0, 0.0, 1.0]), 1.0)

    assert gate.update(measurement).state == "ORIENTING"
    assert gate.update(measurement).state == "ORIENTING"
    state = gate.update(measurement)

    assert state.state == "FRONT_ALIGNED"
    assert state.aligned


def test_front_view_gate_leaves_alignment_after_large_tilt():
    gate = FrontViewGate(window_size=1, confirm_frames=1)
    aligned = estimate_front_view(_plane([0.0, 0.0, 1.0]), 0.0)
    assert gate.update(aligned).aligned

    angle = np.deg2rad(12.0)
    tilted = estimate_front_view(
        _plane([np.sin(angle), 0.0, np.cos(angle)]), 0.0
    )
    state = gate.update(tilted)

    assert not state.aligned
    assert state.state == "ORIENTING"
