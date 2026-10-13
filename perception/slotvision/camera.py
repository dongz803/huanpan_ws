"""Pinhole camera model for the simulated perception head.

MuJoCo cameras look along their local ``-z`` axis with ``+y`` up, so a point
``p_w`` in world coordinates projects to::

    p_cam = R_cw^T @ (p_w - t_cw)
    depth = -p_cam[2]
    u = cx + fx * p_cam[0] / depth
    v = cy - fy * p_cam[1] / depth

``R_cw`` / ``t_cw`` are the camera pose expressed in the world frame, which is
exactly what MuJoCo stores in ``model.cam_pos`` / ``model.cam_quat``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics in pixels, image origin at the top-left corner."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_fovy(cls, width: int, height: int, fovy_deg: float) -> "Intrinsics":
        """Build intrinsics from a vertical field of view, square pixels.

        ``cx``/``cy`` follow the OpenGL viewport convention used by MuJoCo's
        offscreen renderer: continuous pixel coordinates span ``[0, width]``
        and pixel ``i`` is centred at ``i + 0.5``, so the principal point is
        exactly ``width / 2``. Use ``floor`` to convert to a pixel index.
        """
        fy = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
        return cls(
            width=width,
            height=height,
            fx=fy,
            fy=fy,
            cx=width / 2.0,
            cy=height / 2.0,
        )

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def hfov_deg(self) -> float:
        return 2.0 * math.degrees(math.atan((self.width / 2.0) / self.fx))

    def to_dict(self) -> dict:
        return asdict(self)


def quat_to_matrix(quat_wxyz) -> np.ndarray:
    """Rotation matrix from a MuJoCo ``(w, x, y, z)`` quaternion."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("zero-norm quaternion")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    """MuJoCo ``(w, x, y, z)`` quaternion from a 3x3 rotation matrix."""
    m = np.asarray(matrix, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (m[2, 1] - m[1, 2]) / s,
                (m[0, 2] - m[2, 0]) / s,
                (m[1, 0] - m[0, 1]) / s,
            ]
        )
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        quat = np.array(
            [
                (m[2, 1] - m[1, 2]) / s,
                0.25 * s,
                (m[0, 1] + m[1, 0]) / s,
                (m[0, 2] + m[2, 0]) / s,
            ]
        )
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        quat = np.array(
            [
                (m[0, 2] - m[2, 0]) / s,
                (m[0, 1] + m[1, 0]) / s,
                0.25 * s,
                (m[1, 2] + m[2, 1]) / s,
            ]
        )
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        quat = np.array(
            [
                (m[1, 0] - m[0, 1]) / s,
                (m[0, 2] + m[2, 0]) / s,
                (m[1, 2] + m[2, 1]) / s,
                0.25 * s,
            ]
        )
    return quat / np.linalg.norm(quat)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera orientation (MuJoCo ``wxyz``) looking from ``eye`` at ``target``.

    The returned rotation maps camera-local axes to world: camera ``+x`` is
    image right, ``+y`` is image up and ``-z`` is the viewing direction, which
    matches MuJoCo's camera convention and the ``xyaxes`` used in the MJCF.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)

    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        raise ValueError("eye and target coincide")
    forward = forward / norm

    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("viewing direction is parallel to up")
    right = right / np.linalg.norm(right)
    true_up = np.cross(-forward, right)

    rotation = np.column_stack([right, true_up, -forward])
    return matrix_to_quat(rotation)


def project(points_world, cam_pos, cam_quat, intrinsics: Intrinsics):
    """Project world points; returns ``(uv, depth)`` with ``uv`` in pixels."""
    points = np.atleast_2d(np.asarray(points_world, dtype=float))
    rotation = quat_to_matrix(cam_quat)
    points_cam = (points - np.asarray(cam_pos, dtype=float)) @ rotation
    depth = -points_cam[:, 2]
    if np.any(depth <= 1e-6):
        # Behind or on the image plane: caller decides how to reject.
        depth = np.where(depth <= 1e-6, np.nan, depth)
    u = intrinsics.cx + intrinsics.fx * points_cam[:, 0] / depth
    v = intrinsics.cy - intrinsics.fy * points_cam[:, 1] / depth
    return np.column_stack([u, v]), depth


def unproject(pixel, depth, cam_pos, cam_quat, intrinsics: Intrinsics) -> np.ndarray:
    """Back-project one pixel with a metric depth to a world point."""
    u, v = float(pixel[0]), float(pixel[1])
    x = (u - intrinsics.cx) / intrinsics.fx * depth
    y = -(v - intrinsics.cy) / intrinsics.fy * depth
    point_cam = np.array([x, y, -depth])
    rotation = quat_to_matrix(cam_quat)
    return np.asarray(cam_pos, dtype=float) + rotation @ point_cam
