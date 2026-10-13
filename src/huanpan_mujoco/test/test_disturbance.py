import numpy as np
import pytest

from huanpan_mujoco.disturbance import (
    OrnsteinUhlenbeckTorqueDisturbance,
)


def make_disturbance(seed=7):
    return OrnsteinUhlenbeckTorqueDisturbance(
        standard_deviation_nm=[0.8, 0.8, 0.5, 0.12, 0.12, 0.08],
        limit_nm=[2.4, 2.4, 1.5, 0.36, 0.36, 0.24],
        correlation_time_s=0.25,
        seed=seed,
    )


def test_disturbance_is_reproducible_for_fixed_seed() -> None:
    first = make_disturbance()
    second = make_disturbance()
    for _ in range(200):
        assert np.array_equal(first.step(0.002), second.step(0.002))


def test_disturbance_respects_torque_limits() -> None:
    disturbance = make_disturbance()
    limit = np.array([2.4, 2.4, 1.5, 0.36, 0.36, 0.24])
    for _ in range(5000):
        value = disturbance.step(0.002)
        assert np.all(np.abs(value) <= limit)


def test_disturbance_reset_restarts_seeded_sequence() -> None:
    disturbance = make_disturbance()
    expected = [disturbance.step(0.002) for _ in range(20)]
    for _ in range(100):
        disturbance.step(0.002)
    disturbance.reset()
    actual = [disturbance.step(0.002) for _ in range(20)]
    for first, second in zip(expected, actual):
        assert np.array_equal(first, second)


def test_disturbance_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        OrnsteinUhlenbeckTorqueDisturbance(
            standard_deviation_nm=[1.0, 1.0],
            limit_nm=[1.0],
            correlation_time_s=0.25,
            seed=7,
        )
    with pytest.raises(ValueError):
        OrnsteinUhlenbeckTorqueDisturbance(
            standard_deviation_nm=[1.0],
            limit_nm=[1.0],
            correlation_time_s=0.0,
            seed=7,
        )
