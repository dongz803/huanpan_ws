#!/usr/bin/env python3
"""Launch MuJoCo's interactive viewer at the UR5e home pose."""

from pathlib import Path

import mujoco
import mujoco.viewer


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    WORKSPACE_ROOT
    / "models"
    / "universal_robots_ur5e"
    / "scene.xml"
)


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, "home"
    )
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()

