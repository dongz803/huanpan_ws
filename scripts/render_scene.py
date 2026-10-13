#!/usr/bin/env python3
"""Render a deterministic UR5e preview without opening a GUI window."""

import argparse
import os
from pathlib import Path

# EGL works in a headless shell and can be overridden by the caller.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
from PIL import Image  # noqa: E402


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = (
    WORKSPACE_ROOT
    / "models"
    / "universal_robots_ur5e"
    / "scene.xml"
)
DEFAULT_OUTPUT = WORKSPACE_ROOT / "outputs" / "ur5e_home.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="MJCF model path (relative paths are resolved from huanpan_ws)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="PNG output path",
    )
    parser.add_argument("--azimuth", type=float, default=135.0)
    parser.add_argument("--elevation", type=float, default=-20.0)
    parser.add_argument("--distance", type=float, default=2.2)
    parser.add_argument(
        "--keyframe",
        default="home",
        help="MuJoCo keyframe to render",
    )
    parser.add_argument(
        "--lookat",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.45),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser()
    if not model_path.is_absolute():
        model_path = (WORKSPACE_ROOT / model_path).resolve()
    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    keyframe_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_KEY, args.keyframe
    )
    if keyframe_id < 0:
        raise ValueError(f"Unknown MuJoCo keyframe: {args.keyframe}")
    mujoco.mj_resetDataKeyframe(model, data, keyframe_id)
    mujoco.mj_forward(model, data)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation

    # The stock Menagerie scene uses MuJoCo's default 640x480 offscreen buffer.
    renderer = mujoco.Renderer(model, height=480, width=640)
    try:
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        Image.fromarray(image).save(output_path)
    finally:
        renderer.close()

    print(f"Rendered UR5e preview: {output_path}")


if __name__ == "__main__":
    main()
