from collections import deque

import mujoco
import numpy as np
import pytest

from huanpan_mujoco.force_guard import (
    FilteredForceGuard,
    append_history_sample,
    body_contact_wrench_world,
    select_retreat_target,
)


def make_guard() -> FilteredForceGuard:
    return FilteredForceGuard(
        threshold_n=25.0,
        release_n=15.0,
        trip_duration_s=0.03,
        filter_time_constant_s=0.0,
    )


def test_short_force_spike_does_not_trip() -> None:
    guard = make_guard()
    assert not guard.update((30.0, 0.0, 0.0), 0.01, armed=True)
    assert not guard.update((30.0, 0.0, 0.0), 0.01, armed=True)
    assert not guard.update((0.0, 0.0, 0.0), 0.01, armed=True)
    assert not guard.tripped


def test_sustained_force_trips_and_latches() -> None:
    guard = make_guard()
    assert not guard.update((30.0, 0.0, 0.0), 0.01, armed=True)
    assert not guard.update((30.0, 0.0, 0.0), 0.01, armed=True)
    assert guard.update((30.0, 0.0, 0.0), 0.01, armed=True)
    assert guard.tripped
    assert not guard.update((0.0, 0.0, 0.0), 0.01, armed=True)
    guard.reset_latch()
    assert not guard.tripped


def test_disarmed_guard_filters_but_cannot_trip() -> None:
    guard = make_guard()
    for _ in range(10):
        assert not guard.update((100.0, 0.0, 0.0), 0.01, armed=False)
    assert guard.force_norm_n == pytest.approx(100.0)
    assert not guard.tripped


def test_retreat_target_uses_pre_contact_history() -> None:
    history = deque()
    append_history_sample(history, 1.0, np.array([1.0, 1.0]), 2.0)
    append_history_sample(history, 1.2, np.array([2.0, 2.0]), 2.0)
    append_history_sample(history, 1.4, np.array([3.0, 3.0]), 2.0)
    selected = select_retreat_target(history, 1.5, lookback_s=0.25)
    assert np.array_equal(selected, np.array([2.0, 2.0]))


def test_contact_wrench_points_up_on_supported_box() -> None:
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option timestep="0.001" gravity="0 0 -9.81"/>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="box" pos="0 0 0.05">
              <freejoint/>
              <geom name="box_geom" type="box" size="0.05 0.05 0.05"
                    mass="1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    for _ in range(2000):
        mujoco.mj_step(model, data)
    box_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "box"
    )
    wrench = body_contact_wrench_world(model, data, box_id)
    assert wrench[2] == pytest.approx(9.81, rel=0.02)
    assert np.linalg.norm(wrench[:2]) < 1.0e-6
