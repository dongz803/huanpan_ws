"""Temporal stability and hysteresis for frame-wise planar pose estimates."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from .fine_localization import PoseEstimate


@dataclass(frozen=True)
class TemporalPoseState:
    pose: PoseEstimate | None
    state: str
    measurement_valid: bool
    tracking: bool
    reason: str = ""
    translation_jump_m: float = 0.0
    rotation_jump_deg: float = 0.0


def _quaternion_to_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = max(float(np.linalg.norm([x, y, z, w])), 1e-12)
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotation_delta_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(np.arccos(cosine)))


def _smooth_pose(poses: list[PoseEstimate]) -> PoseEstimate:
    if len(poses) == 1:
        return poses[0]
    translation = np.median(
        np.asarray([pose.translation for pose in poses], dtype=np.float64), axis=0
    )
    quaternions = np.asarray(
        [pose.quaternion_xyzw for pose in poses], dtype=np.float64
    )
    reference = quaternions[0]
    for index in range(len(quaternions)):
        if np.dot(reference, quaternions[index]) < 0:
            quaternions[index] *= -1.0
    accumulator = sum(np.outer(q, q) for q in quaternions)
    _, vectors = np.linalg.eigh(accumulator)
    quaternion = vectors[:, -1]
    if quaternion[3] < 0:
        quaternion *= -1.0
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    rotation = _quaternion_to_rotation(quaternion)
    rvec, _ = cv2.Rodrigues(rotation)
    return PoseEstimate(
        translation=translation,
        rotation=rotation,
        rvec=rvec.reshape(3),
        quaternion_xyzw=quaternion,
        reprojection_error_px=float(
            np.median([pose.reprojection_error_px for pose in poses])
        ),
        alternate_solutions=poses[-1].alternate_solutions,
    )


class PoseStabilizer:
    """Median pose smoothing with acquire/lost hysteresis and jump rejection."""

    def __init__(
        self,
        *,
        window_size: int = 5,
        confirm_frames: int = 3,
        lost_frames: int = 3,
        max_translation_jump_m: float = 0.05,
        max_rotation_jump_deg: float = 15.0,
    ) -> None:
        if window_size < 1 or confirm_frames < 1 or lost_frames < 1:
            raise ValueError("temporal window and frame counts must be >= 1")
        self.window_size = int(window_size)
        self.confirm_frames = int(confirm_frames)
        self.lost_frames = int(lost_frames)
        self.max_translation_jump_m = float(max_translation_jump_m)
        self.max_rotation_jump_deg = float(max_rotation_jump_deg)
        self._poses: deque[PoseEstimate] = deque(maxlen=self.window_size)
        self._valid_streak = 0
        self._miss_streak = 0
        self._tracking = False
        self._last_pose: PoseEstimate | None = None

    def update(self, pose: PoseEstimate) -> TemporalPoseState:
        translation_jump = 0.0
        rotation_jump = 0.0
        # Confirmation means consecutive *consistent* frames, not merely three
        # successful single-frame PnP calls.  Apply the jump check while
        # acquiring as well as while tracking.
        if self._last_pose is not None:
            translation_jump = float(
                np.linalg.norm(pose.translation - self._last_pose.translation)
            )
            rotation_jump = _rotation_delta_deg(
                self._last_pose.rotation, pose.rotation
            )
            translation_bad = (
                self.max_translation_jump_m > 0
                and translation_jump > self.max_translation_jump_m
            )
            rotation_bad = (
                self.max_rotation_jump_deg > 0
                and rotation_jump > self.max_rotation_jump_deg
            )
            if translation_bad or rotation_bad:
                return self.miss(
                    "pose_jump",
                    translation_jump_m=translation_jump,
                    rotation_jump_deg=rotation_jump,
                )

        self._poses.append(pose)
        self._valid_streak += 1
        self._miss_streak = 0
        filtered = _smooth_pose(list(self._poses))
        self._last_pose = filtered
        if not self._tracking and self._valid_streak >= self.confirm_frames:
            self._tracking = True
        return TemporalPoseState(
            pose=filtered,
            state="TRACKING" if self._tracking else "ACQUIRING",
            measurement_valid=True,
            tracking=self._tracking,
            translation_jump_m=translation_jump,
            rotation_jump_deg=rotation_jump,
        )

    def miss(
        self,
        reason: str,
        *,
        translation_jump_m: float = 0.0,
        rotation_jump_deg: float = 0.0,
    ) -> TemporalPoseState:
        self._valid_streak = 0
        self._miss_streak += 1
        if self._tracking and self._miss_streak < self.lost_frames:
            return TemporalPoseState(
                pose=self._last_pose,
                state="HOLDING",
                measurement_valid=False,
                tracking=True,
                reason=reason,
                translation_jump_m=translation_jump_m,
                rotation_jump_deg=rotation_jump_deg,
            )
        self._tracking = False
        self._poses.clear()
        self._last_pose = None
        return TemporalPoseState(
            pose=None,
            state="LOST",
            measurement_valid=False,
            tracking=False,
            reason=reason,
            translation_jump_m=translation_jump_m,
            rotation_jump_deg=rotation_jump_deg,
        )
