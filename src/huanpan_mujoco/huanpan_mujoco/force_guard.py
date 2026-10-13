"""Contact-force filtering and retreat-target selection.

This module is intentionally independent from ROS so the safety decisions can
be unit-tested without launching the simulator.
"""

from __future__ import annotations

import math
from typing import Deque, Iterable, Tuple

import mujoco
import numpy as np


HistorySample = Tuple[float, np.ndarray]


def body_contact_wrench_world(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_id: int,
) -> np.ndarray:
    """Return the net contact wrench acting on ``body_id`` in world axes.

    MuJoCo reports each contact wrench in its contact frame and with the force
    acting on geom2.  The torque is shifted from the contact point to the body
    origin before the individual wrenches are accumulated.
    """

    if body_id < 0 or body_id >= model.nbody:
        raise ValueError("body_id is outside the MuJoCo model")

    force_sum = np.zeros(3, dtype=float)
    torque_sum = np.zeros(3, dtype=float)
    body_origin = np.asarray(data.xpos[body_id], dtype=float)

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        if body_id not in (body1, body2):
            continue

        contact_wrench = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(
            model, data, contact_index, contact_wrench
        )
        contact_rotation = np.asarray(contact.frame).reshape(3, 3)
        force_world = contact_rotation.T @ contact_wrench[:3]
        torque_world = contact_rotation.T @ contact_wrench[3:]
        sign = 1.0 if body_id == body2 else -1.0
        force_world *= sign
        torque_world *= sign

        force_sum += force_world
        torque_sum += torque_world + np.cross(
            np.asarray(contact.pos) - body_origin, force_world
        )

    return np.concatenate((force_sum, torque_sum))


class FilteredForceGuard:
    """Low-pass filtered, time-debounced over-force latch."""

    def __init__(
        self,
        threshold_n: float,
        release_n: float,
        trip_duration_s: float,
        filter_time_constant_s: float,
    ) -> None:
        if threshold_n <= 0.0:
            raise ValueError("threshold_n must be positive")
        if release_n < 0.0 or release_n >= threshold_n:
            raise ValueError(
                "release_n must be non-negative and below threshold_n"
            )
        if trip_duration_s <= 0.0:
            raise ValueError("trip_duration_s must be positive")
        if filter_time_constant_s < 0.0:
            raise ValueError(
                "filter_time_constant_s must be non-negative"
            )

        self.threshold_n = float(threshold_n)
        self.release_n = float(release_n)
        self.trip_duration_s = float(trip_duration_s)
        self.filter_time_constant_s = float(filter_time_constant_s)
        self.filtered_force = np.zeros(3, dtype=float)
        self.over_limit_time_s = 0.0
        self.tripped = False

    @property
    def force_norm_n(self) -> float:
        return float(np.linalg.norm(self.filtered_force))

    def reset_latch(self) -> None:
        """Clear the decision latch while retaining the filter state."""

        self.over_limit_time_s = 0.0
        self.tripped = False

    def update(
        self,
        force_world: Iterable[float],
        dt: float,
        armed: bool,
    ) -> bool:
        """Update the filter and return True only on a new trip event."""

        if dt <= 0.0:
            raise ValueError("dt must be positive")
        force = np.asarray(tuple(force_world), dtype=float)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ValueError("force_world must contain three finite values")

        if self.filter_time_constant_s == 0.0:
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(
                -dt / self.filter_time_constant_s
            )
        self.filtered_force += alpha * (force - self.filtered_force)

        if not armed:
            self.reset_latch()
            return False
        if self.tripped:
            return False

        force_norm = self.force_norm_n
        if force_norm >= self.threshold_n:
            self.over_limit_time_s += dt
        elif force_norm <= self.release_n:
            self.over_limit_time_s = 0.0

        if self.over_limit_time_s >= self.trip_duration_s:
            self.tripped = True
            return True
        return False


def select_retreat_target(
    history: Deque[HistorySample],
    trigger_time: float,
    lookback_s: float,
) -> np.ndarray:
    """Select the newest recorded target at least ``lookback_s`` old."""

    if not history:
        raise ValueError("retreat history is empty")
    if lookback_s < 0.0:
        raise ValueError("lookback_s must be non-negative")

    cutoff = trigger_time - lookback_s
    selected = history[0][1]
    for timestamp, positions in history:
        if timestamp > cutoff:
            break
        selected = positions
    return np.asarray(selected, dtype=float).copy()


def append_history_sample(
    history: Deque[HistorySample],
    timestamp: float,
    positions: np.ndarray,
    keep_duration_s: float,
) -> None:
    """Append one target and discard samples older than the retention window."""

    history.append((timestamp, np.asarray(positions, dtype=float).copy()))
    oldest_allowed = timestamp - keep_duration_s
    while len(history) > 1 and history[1][0] < oldest_allowed:
        history.popleft()
