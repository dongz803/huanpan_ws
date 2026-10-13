"""Pure OpenCV geometry for slot refinement and planar pose estimation.

The YOLO detector intentionally remains the coarse L0 stage.  This module
turns its axis-aligned ROI into four sub-pixel edges/corners and solves the
pose of the known 100 x 25 mm planar slot.  It has no ROS imports so the same
implementation is usable from tests, offline evaluation, and the live node.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class CameraModel:
    width: int
    height: int
    k: np.ndarray
    d: np.ndarray
    frame_id: str = "camera_color_optical_frame"

    @classmethod
    def from_mapping(cls, value: Mapping) -> "CameraModel":
        return cls(
            width=int(value["width"]),
            height=int(value["height"]),
            k=np.asarray(value["k"], dtype=np.float64).reshape(3, 3),
            d=np.asarray(value.get("d", []), dtype=np.float64).reshape(-1, 1),
            frame_id=str(value.get("frame_id", "camera_color_optical_frame")),
        )


@dataclass(frozen=True)
class RefinedRectangle:
    """Corners are ordered top-left, top-right, bottom-right, bottom-left."""

    corners: np.ndarray
    quality: float
    method: str
    diagnostics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PoseEstimate:
    translation: np.ndarray
    rotation: np.ndarray
    rvec: np.ndarray
    quaternion_xyzw: np.ndarray
    reprojection_error_px: float
    alternate_solutions: int


@dataclass(frozen=True)
class DepthPlane:
    normal: np.ndarray
    distance_m: float
    median_depth_m: float
    valid_ratio: float
    inlier_ratio: float
    rmse_m: float


@dataclass(frozen=True)
class _SideLineCandidate:
    line: np.ndarray
    strength: float
    peak: float


def _clamp_bbox(
    bbox: Iterable[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    xa = max(1, min(width - 2, int(np.floor(min(x0, x1)))))
    xb = max(2, min(width - 1, int(np.ceil(max(x0, x1)))))
    ya = max(1, min(height - 2, int(np.floor(min(y0, y1)))))
    yb = max(2, min(height - 1, int(np.ceil(max(y0, y1)))))
    if xb - xa < 12 or yb - ya < 8:
        raise ValueError(f"coarse box is too small after clipping: {(xa, ya, xb, yb)}")
    return xa, ya, xb, yb


def _fit_line(points_xy: np.ndarray, orientation: str) -> np.ndarray:
    if len(points_xy) < 12:
        raise ValueError(f"not enough {orientation} edge points")
    points = np.asarray(points_xy, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x, y = cv2.fitLine(
        points, cv2.DIST_HUBER, 0, 0.01, 0.01
    ).reshape(-1)
    if orientation == "horizontal" and abs(vx) < 0.75:
        raise ValueError("candidate horizontal edge has the wrong direction")
    if orientation == "vertical" and abs(vy) < 0.75:
        raise ValueError("candidate vertical edge has the wrong direction")
    # ax + by + c = 0, normalised to make point-to-line distance direct.
    line = np.array([vy, -vx, -(vy * x - vx * y)], dtype=np.float64)
    line /= max(np.linalg.norm(line[:2]), 1e-12)

    raw = np.asarray(points_xy, dtype=np.float64)
    residual = np.abs(raw @ line[:2] + line[2])
    keep = residual <= max(1.5, float(np.percentile(residual, 60)))
    if int(keep.sum()) >= 12 and int(keep.sum()) < len(raw):
        return _fit_line(raw[keep], orientation)
    return line


def _side_line(
    gradient: np.ndarray,
    orthogonal_gradient: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[np.ndarray, float, float]:
    candidate = _side_line_candidates(
        gradient, orthogonal_gradient, bbox, side, max_candidates=1
    )[0]
    return candidate.line, candidate.strength, candidate.peak


def _side_line_candidates(
    gradient: np.ndarray,
    orthogonal_gradient: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    *,
    max_candidates: int = 3,
) -> list[_SideLineCandidate]:
    """Return separated edge hypotheses near one coarse-box side.

    A tilted or reflective slot often has several strong responses near a YOLO
    side.  Keeping only the strongest row/column made the old implementation
    jump between the physical rim, grille texture and highlights.  We retain a
    few non-overlapping hypotheses so the four sides can later be selected as a
    geometrically consistent quadrilateral.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    x0, y0, x1, y1 = bbox
    box_w, box_h = x1 - x0, y1 - y0
    if side in {"top", "bottom"}:
        expected = y0 if side == "top" else y1
        radius = max(4, int(round(box_h * 0.28)))
        lo = max(1, expected - radius)
        hi = min(gradient.shape[0] - 1, expected + radius + 1)
        xa = max(1, x0 - int(0.04 * box_w))
        xb = min(gradient.shape[1] - 1, x1 + int(0.04 * box_w))
        strip = gradient[lo:hi, xa:xb]
        profile = np.percentile(strip, 80, axis=1)
        positions = np.arange(lo, hi)
        sigma = max(radius * 0.75, 1.0)
        score = profile * np.exp(-0.5 * ((positions - expected) / sigma) ** 2)
        candidates: list[_SideLineCandidate] = []
        selected_peaks: list[int] = []
        for index in np.argsort(score)[::-1]:
            peak = int(positions[index])
            if any(abs(peak - previous) < 4 for previous in selected_peaks):
                continue
            band = slice(max(1, peak - 2), min(gradient.shape[0] - 1, peak + 3))
            local = gradient[band, xa:xb]
            other = orthogonal_gradient[band, xa:xb]
            threshold = max(float(np.percentile(local, 65)), 8.0)
            mask = (local >= threshold) & (local >= other * 1.05)
            yy, xx = np.nonzero(mask)
            points = np.column_stack((xx + xa, yy + band.start))
            try:
                line = _fit_line(points, "horizontal")
            except ValueError:
                continue
            candidates.append(
                _SideLineCandidate(line, float(profile[index]), float(peak))
            )
            selected_peaks.append(peak)
            if len(candidates) >= max_candidates:
                break
        if not candidates:
            raise ValueError(f"not enough horizontal edge points for {side}")
        return candidates

    expected = x0 if side == "left" else x1
    radius = max(4, int(round(box_w * 0.16)))
    lo = max(1, expected - radius)
    hi = min(gradient.shape[1] - 1, expected + radius + 1)
    ya = max(1, y0 - int(0.08 * box_h))
    yb = min(gradient.shape[0] - 1, y1 + int(0.08 * box_h))
    strip = gradient[ya:yb, lo:hi]
    profile = np.percentile(strip, 80, axis=0)
    positions = np.arange(lo, hi)
    sigma = max(radius * 0.75, 1.0)
    score = profile * np.exp(-0.5 * ((positions - expected) / sigma) ** 2)
    candidates = []
    selected_peaks = []
    for index in np.argsort(score)[::-1]:
        peak = int(positions[index])
        if any(abs(peak - previous) < 4 for previous in selected_peaks):
            continue
        band = slice(max(1, peak - 2), min(gradient.shape[1] - 1, peak + 3))
        local = gradient[ya:yb, band]
        other = orthogonal_gradient[ya:yb, band]
        threshold = max(float(np.percentile(local, 65)), 8.0)
        mask = (local >= threshold) & (local >= other * 1.05)
        yy, xx = np.nonzero(mask)
        points = np.column_stack((xx + band.start, yy + ya))
        try:
            line = _fit_line(points, "vertical")
        except ValueError:
            continue
        candidates.append(
            _SideLineCandidate(line, float(profile[index]), float(peak))
        )
        selected_peaks.append(peak)
        if len(candidates) >= max_candidates:
            break
    if not candidates:
        raise ValueError(f"not enough vertical edge points for {side}")
    return candidates


def _intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    point = np.cross(first, second)
    if abs(point[2]) < 1e-9:
        raise ValueError("refined slot edges are parallel")
    return point[:2] / point[2]


def polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def coarse_rectangle(bbox: Iterable[float]) -> RefinedRectangle:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    corners = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64
    )
    return RefinedRectangle(corners, 0.0, "coarse_bbox", {})


def refine_rectangle(image_rgb: np.ndarray, bbox: Iterable[float]) -> RefinedRectangle:
    """Fit four robust sub-pixel lines close to a YOLO coarse box.

    The detector already identifies the correct front face.  Restricting each
    line search to a narrow band around one side prevents the grille texture
    inside the black drive tray from dominating the fit.
    """
    rectangles = refine_rectangle_candidates(image_rgb, bbox, max_candidates=1)
    selected = rectangles[0]
    return RefinedRectangle(
        corners=selected.corners,
        quality=selected.quality,
        method="gradient_lines",
        diagnostics=selected.diagnostics,
    )


def refine_rectangle_candidates(
    image_rgb: np.ndarray,
    bbox: Iterable[float],
    *,
    max_candidates: int = 3,
) -> list[RefinedRectangle]:
    """Build perspective quadrilateral hypotheses from multiple side lines."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image_rgb.shape}")
    height, width = image_rgb.shape[:2]
    clipped = _clamp_bbox(bbox, width, height)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0.8)
    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    top_candidates = _side_line_candidates(
        grad_y, grad_x, clipped, "top", max_candidates=max_candidates
    )
    right_candidates = _side_line_candidates(
        grad_x, grad_y, clipped, "right", max_candidates=max_candidates
    )
    bottom_candidates = _side_line_candidates(
        grad_y, grad_x, clipped, "bottom", max_candidates=max_candidates
    )
    left_candidates = _side_line_candidates(
        grad_x, grad_y, clipped, "left", max_candidates=max_candidates
    )

    x0, y0, x1, y1 = clipped
    margin_x = max(8.0, (x1 - x0) * 0.35)
    margin_y = max(8.0, (y1 - y0) * 0.45)
    coarse_area = float((x1 - x0) * (y1 - y0))
    rectangles: list[RefinedRectangle] = []
    for top, right, bottom, left in itertools.product(
        top_candidates, right_candidates, bottom_candidates, left_candidates
    ):
        try:
            corners = np.vstack(
                [
                    _intersection(top.line, left.line),
                    _intersection(top.line, right.line),
                    _intersection(bottom.line, right.line),
                    _intersection(bottom.line, left.line),
                ]
            )
        except ValueError:
            continue
        if not np.all(np.isfinite(corners)):
            continue
        if (
            np.any(corners[:, 0] < x0 - margin_x)
            or np.any(corners[:, 0] > x1 + margin_x)
            or np.any(corners[:, 1] < y0 - margin_y)
            or np.any(corners[:, 1] > y1 + margin_y)
        ):
            continue
        area = polygon_area(corners)
        if area < 0.35 * coarse_area or area > 1.8 * coarse_area:
            continue
        edge_strengths = np.array(
            [top.strength, right.strength, bottom.strength, left.strength]
        )
        quality = float(np.clip(np.min(edge_strengths) / 96.0, 0.0, 1.0))
        boundary_distance = (
            (abs(top.peak - y0) + abs(bottom.peak - y1)) / max(y1 - y0, 1)
            + (abs(left.peak - x0) + abs(right.peak - x1)) / max(x1 - x0, 1)
        )
        rectangles.append(
            RefinedRectangle(
                corners=corners.astype(np.float64),
                quality=quality,
                method="gradient_lines_multi",
                diagnostics={
                    "top_y": top.peak,
                    "right_x": right.peak,
                    "bottom_y": bottom.peak,
                    "left_x": left.peak,
                    "area_ratio": area / coarse_area,
                    "min_edge_strength": float(np.min(edge_strengths)),
                    "boundary_distance": float(boundary_distance),
                },
            )
        )
    if not rectangles:
        raise ValueError("no geometrically consistent edge quadrilateral")
    rectangles.sort(
        key=lambda rectangle: (
            rectangle.diagnostics["boundary_distance"]
            - 0.05 * rectangle.quality
        )
    )
    return rectangles


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
        w = 0.25 * scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    return quaternion / max(np.linalg.norm(quaternion), 1e-12)


def object_corners(width_m: float, height_m: float) -> np.ndarray:
    half_w, half_h = float(width_m) / 2.0, float(height_m) / 2.0
    return np.array(
        [
            [-half_w, -half_h, 0.0],
            [half_w, -half_h, 0.0],
            [half_w, half_h, 0.0],
            [-half_w, half_h, 0.0],
        ],
        dtype=np.float64,
    )


def solve_planar_pose(
    corners: np.ndarray,
    camera: CameraModel,
    *,
    width_m: float = 0.100,
    height_m: float = 0.025,
) -> PoseEstimate:
    image_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    model_points = object_corners(width_m, height_m)
    result = cv2.solvePnPGeneric(
        model_points,
        image_points,
        camera.k,
        camera.d,
        flags=cv2.SOLVEPNP_IPPE,
    )
    if not result[0] or not result[1]:
        raise ValueError("SOLVEPNP_IPPE did not return a pose")
    rvecs, tvecs = result[1], result[2]
    candidates: list[tuple[float, np.ndarray, np.ndarray]] = []
    for rvec, tvec in zip(rvecs, tvecs):
        translation = np.asarray(tvec, dtype=np.float64).reshape(3)
        if translation[2] <= 0:
            continue
        projected, _ = cv2.projectPoints(
            model_points, rvec, tvec, camera.k, camera.d
        )
        error = float(
            np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - image_points) ** 2, axis=1)))
        )
        candidates.append((error, np.asarray(rvec).reshape(3), translation))
    if not candidates:
        raise ValueError("all planar PnP solutions are behind the camera")
    error, rvec, translation = min(candidates, key=lambda item: item[0])
    rotation, _ = cv2.Rodrigues(rvec)
    return PoseEstimate(
        translation=translation,
        rotation=rotation,
        rvec=rvec,
        quaternion_xyzw=_rotation_matrix_to_quaternion(rotation),
        reprojection_error_px=error,
        alternate_solutions=len(candidates),
    )


def select_best_planar_pose(
    rectangles: Iterable[RefinedRectangle],
    camera: CameraModel,
    *,
    width_m: float = 0.100,
    height_m: float = 0.025,
) -> tuple[RefinedRectangle, PoseEstimate]:
    """Select a four-edge hypothesis using image and PnP consistency.

    Reprojection error alone can overfit arbitrary grille lines, while choosing
    only the strongest gradient repeats the old reflection-sensitive failure.
    The combined score keeps the hypotheses close to the YOLO boundary and
    rewards supported edges, then uses planar reprojection to enforce that the
    four lines jointly explain the known slot rectangle.
    """
    ranked: list[tuple[float, RefinedRectangle, PoseEstimate]] = []
    for rectangle in rectangles:
        try:
            pose = solve_planar_pose(
                rectangle.corners,
                camera,
                width_m=width_m,
                height_m=height_m,
            )
        except ValueError:
            continue
        boundary_distance = float(
            rectangle.diagnostics.get("boundary_distance", 0.0)
        )
        score = (
            pose.reprojection_error_px
            + 2.5 * boundary_distance
            - 0.12 * min(rectangle.quality, 2.0)
        )
        ranked.append((score, rectangle, pose))
    if not ranked:
        raise ValueError("no planar pose from edge quadrilateral candidates")
    _, rectangle, pose = min(ranked, key=lambda item: item[0])
    return rectangle, pose


def pose_correction(current: PoseEstimate, desired: PoseEstimate) -> tuple[np.ndarray, np.ndarray]:
    """Camera movement from the current pose to the taught camera pose.

    Both inputs describe the same stationary slot in their respective camera
    frames.  The returned translation and Rodrigues rotation vector are
    expressed in the current camera frame.
    """
    current_slot = np.eye(4, dtype=np.float64)
    current_slot[:3, :3] = current.rotation
    current_slot[:3, 3] = current.translation
    desired_slot = np.eye(4, dtype=np.float64)
    desired_slot[:3, :3] = desired.rotation
    desired_slot[:3, 3] = desired.translation
    current_to_desired_camera = current_slot @ np.linalg.inv(desired_slot)
    rotation_vector, _ = cv2.Rodrigues(current_to_desired_camera[:3, :3])
    return current_to_desired_camera[:3, 3], rotation_vector.reshape(3)


def pose_to_mapping(pose: PoseEstimate) -> dict:
    return {
        "translation_m": [float(value) for value in pose.translation],
        "quaternion_xyzw": [float(value) for value in pose.quaternion_xyzw],
        "rotation_matrix": [float(value) for value in pose.rotation.reshape(-1)],
        "reprojection_error_px": float(pose.reprojection_error_px),
    }


def pose_from_mapping(value: Mapping) -> PoseEstimate:
    rotation = np.asarray(value["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    rvec, _ = cv2.Rodrigues(rotation)
    return PoseEstimate(
        translation=np.asarray(value["translation_m"], dtype=np.float64),
        rotation=rotation,
        rvec=rvec.reshape(3),
        quaternion_xyzw=np.asarray(value["quaternion_xyzw"], dtype=np.float64),
        reprojection_error_px=float(value.get("reprojection_error_px", 0.0)),
        alternate_solutions=1,
    )


def load_references(path: str | Path) -> dict[int, PoseEstimate]:
    source = Path(path).expanduser()
    if not source.exists():
        return {}
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    return {
        int(slot): pose_from_mapping(record)
        for slot, record in (raw.get("slots", {}) or {}).items()
    }


def save_reference(path: str | Path, slot: int, pose: PoseEstimate) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    raw = {}
    if destination.exists():
        raw = yaml.safe_load(destination.read_text(encoding="utf-8")) or {}
    slots = raw.setdefault("slots", {})
    slots[str(int(slot))] = pose_to_mapping(pose)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(raw, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(destination)


def _depth_metres(depth: np.ndarray) -> np.ndarray:
    values = np.asarray(depth)
    if np.issubdtype(values.dtype, np.integer):
        return values.astype(np.float32) / 1000.0
    return values.astype(np.float32)


def fit_depth_plane(
    depth: np.ndarray,
    corners: np.ndarray,
    camera: CameraModel,
    *,
    distance_threshold_m: float = 0.006,
    iterations: int = 100,
    outside_band_fraction: float = 0.15,
    min_outside_band_px: int = 6,
    max_outside_band_px: int = 24,
    boundary_clearance_px: int = 2,
) -> DepthPlane | None:
    """Fit the cabinet faceplate immediately outside the detected slot.

    The slot polygon itself is excluded completely.  The old implementation
    sampled an *inner* ring, so an empty bay could contribute the rear wall or
    objects seen through the opening.  Here the polygon mask is dilated and
    only the outside annulus is retained.  A small clearance also avoids mixed
    RGB/depth pixels directly on the opening boundary.
    """
    depth_m = _depth_metres(depth)
    height, width = depth_m.shape[:2]
    polygon_points = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    polygon = np.rint(polygon_points).astype(np.int32)
    slot_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(slot_mask, polygon, 1)

    side_lengths = np.linalg.norm(
        np.roll(polygon_points, -1, axis=0) - polygon_points, axis=1
    )
    short_side = float(np.min(side_lengths))
    clearance = max(int(boundary_clearance_px), 0)
    band_width = int(round(short_side * float(outside_band_fraction)))
    band_width = max(int(min_outside_band_px), band_width, clearance + 2)
    band_width = min(int(max_outside_band_px), band_width)
    if band_width <= clearance:
        raise ValueError("outside depth band must be wider than its clearance")

    outer_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * band_width + 1, 2 * band_width + 1)
    )
    outer_mask = cv2.dilate(slot_mask, outer_kernel)
    if clearance:
        clearance_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * clearance + 1, 2 * clearance + 1)
        )
        excluded_mask = cv2.dilate(slot_mask, clearance_kernel)
    else:
        excluded_mask = slot_mask
    outside_ring = (outer_mask != 0) & (excluded_mask == 0)

    valid = (
        outside_ring
        & np.isfinite(depth_m)
        & (depth_m > 0.08)
        & (depth_m < 2.0)
    )
    ring_pixels = max(int(np.count_nonzero(outside_ring)), 1)
    valid_ratio = float(np.count_nonzero(valid) / ring_pixels)
    yy, xx = np.nonzero(valid)
    if len(xx) < 80:
        return None
    if len(xx) > 6000:
        select = np.linspace(0, len(xx) - 1, 6000).astype(int)
        xx, yy = xx[select], yy[select]
    z = depth_m[yy, xx].astype(np.float64)
    fx, fy = camera.k[0, 0], camera.k[1, 1]
    cx, cy = camera.k[0, 2], camera.k[1, 2]
    points = np.column_stack(((xx - cx) * z / fx, (yy - cy) * z / fy, z))

    rng = np.random.default_rng(0)
    best = np.zeros(len(points), dtype=bool)
    for _ in range(max(int(iterations), 1)):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        distances = np.abs((points - sample[0]) @ normal)
        inliers = distances <= distance_threshold_m
        if int(inliers.sum()) > int(best.sum()):
            best = inliers
    if int(best.sum()) < 60:
        return None
    inlier_points = points[best]
    centroid = np.mean(inlier_points, axis=0)
    _, _, vh = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    residuals = np.abs((inlier_points - centroid) @ normal)
    return DepthPlane(
        normal=normal,
        distance_m=abs(float(np.dot(normal, centroid))),
        # Use only the dominant faceplate-plane inliers.  Other valid depth
        # pixels in the outside band may belong to neighbouring openings.
        median_depth_m=float(np.median(inlier_points[:, 2])),
        valid_ratio=valid_ratio,
        inlier_ratio=float(np.mean(best)),
        rmse_m=float(np.sqrt(np.mean(residuals**2))),
    )
