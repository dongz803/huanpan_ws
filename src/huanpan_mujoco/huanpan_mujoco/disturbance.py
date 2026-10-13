"""Reproducible colored joint-torque disturbances for MuJoCo."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class OrnsteinUhlenbeckTorqueDisturbance:
    """Generate bounded, time-correlated joint torque noise."""

    def __init__(
        self,
        standard_deviation_nm: Sequence[float],
        limit_nm: Sequence[float],
        correlation_time_s: float,
        seed: int,
    ) -> None:
        self._standard_deviation = np.asarray(
            standard_deviation_nm, dtype=float
        )
        self._limit = np.asarray(limit_nm, dtype=float)
        if self._standard_deviation.ndim != 1:
            raise ValueError("standard_deviation_nm must be one-dimensional")
        if self._limit.shape != self._standard_deviation.shape:
            raise ValueError("limit_nm must match standard_deviation_nm")
        if np.any(self._standard_deviation < 0.0):
            raise ValueError("standard deviations must be non-negative")
        if np.any(self._limit < 0.0):
            raise ValueError("torque limits must be non-negative")
        if correlation_time_s <= 0.0:
            raise ValueError("correlation_time_s must be positive")

        self._correlation_time = float(correlation_time_s)
        self._seed = int(seed)
        self._generator = np.random.default_rng(self._seed)
        self._value = np.zeros_like(self._standard_deviation)

    @property
    def value(self) -> np.ndarray:
        return self._value.copy()

    def reset(self) -> None:
        """Restart the colored-noise sequence from its configured seed."""

        self._generator = np.random.default_rng(self._seed)
        self._value.fill(0.0)

    def step(self, time_step_s: float) -> np.ndarray:
        if time_step_s <= 0.0:
            raise ValueError("time_step_s must be positive")
        drift = -self._value * time_step_s / self._correlation_time
        diffusion = self._standard_deviation * np.sqrt(
            2.0 * time_step_s / self._correlation_time
        )
        self._value += drift + diffusion * self._generator.standard_normal(
            self._value.shape
        )
        self._value = np.clip(self._value, -self._limit, self._limit)
        return self.value
