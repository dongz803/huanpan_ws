"""Coarse camera-orientation guidance before millimetric slot localization.

The D435 depth plane supplies pitch/yaw alignment while long horizontal image
edges supply roll.  Translation is intentionally absent: this stage only makes
the camera image plane parallel to the rack face and levels the slot in view.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from .fine_localization import DepthPlane


@dataclass(frozen=True)
class FrontViewMeasurement:
    plane_normal: np.ndarray
    tilt_deg: float
    roll_deg: float
    correction_rotvec: np.ndarray
    line_quality: float


@dataclass(frozen=True)
class FrontViewState:
    measurement: FrontViewMeasurement | None
    state: str
    aligned: bool
    confirm_count: int
    reason: str = ""


def _normalise_line_angle(angle_rad: float) -> float:
    """Map an unoriented line angle to [-pi/2, pi/2)."""
    return float((angle_rad + np.pi / 2.0) % np.pi - np.pi / 2.0)


def estimate_roll_from_horizontal_edges(
    image_rgb: np.ndarray,
    bbox: Iterable[float],
    *,
    max_abs_angle_deg: float = 28.0,
    min_length_fraction: float = 0.28,
) -> tuple[float, float]:
    """Return dominant slot-edge angle in degrees and a [0, 1] quality.

    Positive angles follow image coordinates (the line falls down toward the
    right because image y points downward).  Long nearly-horizontal segments
    dominate the axial circular mean; a second pass removes discordant lines.
    """
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image_rgb.shape}")
    height, width = image_rgb.shape[:2]
    x0, y0, x1, y1 = (float(value) for value in bbox)
    xa = max(0, min(width - 2, int(np.floor(min(x0, x1)))))
    xb = max(xa + 2, min(width, int(np.ceil(max(x0, x1)))))
    ya = max(0, min(height - 2, int(np.floor(min(y0, y1)))))
    yb = max(ya + 2, min(height, int(np.ceil(max(y0, y1)))))
    roi = image_rgb[ya:yb, xa:xb]
    roi_width = roi.shape[1]
    if roi_width < 24 or roi.shape[0] < 12:
        raise ValueError("coarse box is too small for front-view roll estimation")

    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.9)
    median = float(np.median(gray))
    lower = int(max(20.0, 0.66 * median))
    upper = int(min(255.0, max(lower + 20.0, 1.33 * median)))
    edges = cv2.Canny(gray, lower, upper, apertureSize=3, L2gradient=True)
    lines = cv2.HoughLinesP(
        edges,
        1.0,
        np.pi / 360.0,
        threshold=max(15, int(round(roi_width * 0.08))),
        minLineLength=max(18, int(round(roi_width * min_length_fraction))),
        maxLineGap=max(6, int(round(roi_width * 0.06))),
    )
    if lines is None:
        raise ValueError("no long horizontal edge for front-view roll")

    angles: list[float] = []
    weights: list[float] = []
    max_angle = np.deg2rad(float(max_abs_angle_deg))
    for raw in lines.reshape(-1, 4):
        dx = float(raw[2] - raw[0])
        dy = float(raw[3] - raw[1])
        length = float(np.hypot(dx, dy))
        if length <= 1.0:
            continue
        angle = _normalise_line_angle(np.arctan2(dy, dx))
        if abs(angle) <= max_angle:
            angles.append(angle)
            weights.append(length)
    if not angles:
        raise ValueError("long edges exist but none is close to horizontal")

    angle_values = np.asarray(angles, dtype=np.float64)
    weight_values = np.asarray(weights, dtype=np.float64)

    def axial_mean(values: np.ndarray, values_weight: np.ndarray) -> float:
        sine = float(np.sum(values_weight * np.sin(2.0 * values)))
        cosine = float(np.sum(values_weight * np.cos(2.0 * values)))
        return 0.5 * float(np.arctan2(sine, cosine))

    mean = axial_mean(angle_values, weight_values)
    residual = np.abs(
        np.asarray([_normalise_line_angle(value - mean) for value in angle_values])
    )
    keep = residual <= np.deg2rad(7.0)
    if np.any(keep):
        angle_values = angle_values[keep]
        weight_values = weight_values[keep]
        mean = axial_mean(angle_values, weight_values)
        residual = np.abs(
            np.asarray(
                [_normalise_line_angle(value - mean) for value in angle_values]
            )
        )

    coverage = min(float(np.sum(weight_values)) / max(2.0 * roi_width, 1.0), 1.0)
    dispersion_deg = float(np.rad2deg(np.average(residual, weights=weight_values)))
    quality = float(np.clip(coverage * np.exp(-dispersion_deg / 5.0), 0.0, 1.0))
    return float(np.rad2deg(mean)), quality


def estimate_front_view(
    plane: DepthPlane,
    roll_deg: float,
    *,
    line_quality: float = 1.0,
) -> FrontViewMeasurement:
    """Build the orientation-only camera correction in the optical frame.

    ``correction_rotvec`` is a small camera-frame physical rotation command:
    its x/y components rotate the optical axis onto the observed panel normal,
    and its z component levels the observed long edge.  Robot-base conversion
    and rate limiting deliberately remain outside this perception module.
    """
    normal = np.asarray(plane.normal, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        raise ValueError("plane normal has zero length")
    normal /= norm
    if normal[2] < 0.0:
        normal *= -1.0
    optical_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    cosine = float(np.clip(np.dot(optical_axis, normal), -1.0, 1.0))
    tilt_rad = float(np.arccos(cosine))
    axis = np.cross(optical_axis, normal)
    axis_norm = float(np.linalg.norm(axis))
    tilt_rotvec = (
        np.zeros(3, dtype=np.float64)
        if axis_norm < 1e-9
        else axis / axis_norm * tilt_rad
    )
    correction = tilt_rotvec
    correction[2] = np.deg2rad(float(roll_deg))
    return FrontViewMeasurement(
        plane_normal=normal,
        tilt_deg=float(np.rad2deg(tilt_rad)),
        roll_deg=float(roll_deg),
        correction_rotvec=correction,
        line_quality=float(line_quality),
    )


class FrontViewGate:
    """Smooth orientation measurements and require consecutive alignment."""

    def __init__(
        self,
        *,
        window_size: int = 5,
        confirm_frames: int = 5,
        tilt_tolerance_deg: float = 5.0,
        roll_tolerance_deg: float = 3.0,
        exit_tilt_tolerance_deg: float = 8.0,
        exit_roll_tolerance_deg: float = 5.0,
    ) -> None:
        if window_size < 1 or confirm_frames < 1:
            raise ValueError("front-view window and confirmation must be >= 1")
        self.window_size = int(window_size)
        self.confirm_frames = int(confirm_frames)
        self.tilt_tolerance_deg = float(tilt_tolerance_deg)
        self.roll_tolerance_deg = float(roll_tolerance_deg)
        self.exit_tilt_tolerance_deg = float(exit_tilt_tolerance_deg)
        self.exit_roll_tolerance_deg = float(exit_roll_tolerance_deg)
        self._measurements: deque[FrontViewMeasurement] = deque(
            maxlen=self.window_size
        )
        self._confirm_count = 0
        self._aligned = False

    def _filtered(self) -> FrontViewMeasurement:
        measurements = list(self._measurements)
        normals = np.asarray([value.plane_normal for value in measurements])
        normal = np.median(normals, axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1e-12)
        roll = float(np.median([value.roll_deg for value in measurements]))
        quality = float(np.median([value.line_quality for value in measurements]))
        # Recompute x/y correction from the filtered plane instead of taking a
        # component-wise median of rotation vectors.
        representative = DepthPlane(
            normal=normal,
            distance_m=0.0,
            median_depth_m=0.0,
            valid_ratio=1.0,
            inlier_ratio=1.0,
            rmse_m=0.0,
        )
        return estimate_front_view(representative, roll, line_quality=quality)

    def update(self, measurement: FrontViewMeasurement) -> FrontViewState:
        self._measurements.append(measurement)
        filtered = self._filtered()
        if self._aligned:
            still_aligned = (
                filtered.tilt_deg <= self.exit_tilt_tolerance_deg
                and abs(filtered.roll_deg) <= self.exit_roll_tolerance_deg
            )
            if not still_aligned:
                self._aligned = False
                self._confirm_count = 0
        else:
            within_entry = (
                filtered.tilt_deg <= self.tilt_tolerance_deg
                and abs(filtered.roll_deg) <= self.roll_tolerance_deg
            )
            self._confirm_count = self._confirm_count + 1 if within_entry else 0
            if self._confirm_count >= self.confirm_frames:
                self._aligned = True
        return FrontViewState(
            measurement=filtered,
            state="FRONT_ALIGNED" if self._aligned else "ORIENTING",
            aligned=self._aligned,
            confirm_count=self._confirm_count,
        )

    def miss(self, reason: str) -> FrontViewState:
        self._measurements.clear()
        self._confirm_count = 0
        self._aligned = False
        return FrontViewState(None, "ORIENTATION_UNKNOWN", False, 0, reason)
