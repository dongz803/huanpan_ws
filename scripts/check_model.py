#!/usr/bin/env python3
"""Headless smoke test for the UR5e MuJoCo model."""

from pathlib import Path

import mujoco
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    WORKSPACE_ROOT
    / "models"
    / "universal_robots_ur5e"
    / "scene.xml"
)

EXPECTED_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


def require_named_objects(model: mujoco.MjModel) -> None:
    for joint_name in EXPECTED_JOINTS:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            raise RuntimeError(f"Missing UR5e joint: {joint_name}")

    home_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    if home_id < 0:
        raise RuntimeError("Missing UR5e home keyframe")


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    require_named_objects(model)
    if (model.nq, model.nv, model.nu) != (6, 6, 6):
        raise RuntimeError(
            "Unexpected UR5e dimensions: "
            f"nq={model.nq}, nv={model.nv}, nu={model.nu}"
        )

    home_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    initial_qpos = data.qpos.copy()
    for _ in range(500):
        mujoco.mj_step(model, data)

    if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
        raise RuntimeError("Simulation produced non-finite joint state")

    max_drift = float(np.max(np.abs(data.qpos - initial_qpos)))
    print(f"model: {MODEL_PATH}")
    print(
        f"bodies={model.nbody}, joints={model.njnt}, "
        f"actuators={model.nu}, timestep={model.opt.timestep:.4f}s"
    )
    print(f"simulated_time={data.time:.3f}s, max_home_drift={max_drift:.6f}rad")
    print("UR5e MuJoCo smoke test: PASS")


if __name__ == "__main__":
    main()

