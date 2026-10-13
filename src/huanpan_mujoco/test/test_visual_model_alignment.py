import math
from pathlib import Path
from xml.etree import ElementTree

import mujoco
import numpy as np
import yaml


ARM_LINKS = {
    "base_link": 2,
    "shoulder_link": 3,
    "upper_arm_link": 4,
    "forearm_link": 4,
    "wrist_1_link": 3,
    "wrist_2_link": 3,
    "wrist_3_link": 1,
}
BODY_LINK_MAP = {
    "base_link": "base",
    "shoulder_link": "shoulder_link",
    "upper_arm_link": "upper_arm_link",
    "forearm_link": "forearm_link",
    "wrist_1_link": "wrist_1_link",
    "wrist_2_link": "wrist_2_link",
    "wrist_3_link": "wrist_3_link",
    "gripper_base": "gripper_base",
    "carried_drive": "carried_drive",
}
AG95_BODIES = {
    "gripper_base",
    "R1_1_link",
    "R1_2_link",
    "R2_1_link",
    "M_link",
    "MR_link",
    "MR_1_1_link",
    "MR_2_1_link",
    "ML_link",
    "ML_1_1_link",
    "ML_2_1_link",
    "L1_1_link",
    "L1_2_link",
    "L2_1_link",
}


def description_path() -> Path:
    return Path(__file__).resolve().parents[2] / "huanpan_description"


def model_path() -> Path:
    return (
        description_path()
        / "mujoco"
        / "universal_robots_ur5e"
        / "workcell.xml"
    )


def test_ag95_visual_geometry_is_rendered_as_mesh() -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path()))
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    visual_count = 0
    for body_name in AG95_BODIES:
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, body_name
        )
        assert body_id >= 0
        first_geom = model.body_geomadr[body_id]
        for geom_id in range(first_geom, first_geom + model.body_geomnum[body_id]):
            if model.geom_group[geom_id] == 2:
                assert int(model.geom_type[geom_id]) == mesh_type
                visual_count += 1
    assert visual_count == len(AG95_BODIES)


def test_rviz_ur5e_uses_complete_mesh_visuals() -> None:
    urdf = ElementTree.parse(
        description_path() / "urdf" / "huanpan_workcell.urdf"
    )
    for link_name, expected_count in ARM_LINKS.items():
        link = urdf.find(f"link[@name='{link_name}']")
        assert link is not None
        visuals = link.findall("visual")
        meshes = [visual.find("geometry/mesh") for visual in visuals]
        assert len(meshes) == expected_count
        assert all(mesh is not None for mesh in meshes)
        assert all(
            mesh.attrib["filename"].endswith(".obj") for mesh in meshes
        )


def test_rviz_defaults_to_arm_planning_group() -> None:
    config_path = (
        description_path().parent
        / "huanpan_moveit_config"
        / "rviz"
        / "moveit.rviz"
    )
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert all(
        panel["Class"] != "moveit_rviz_plugin/MotionPlanning"
        for panel in config["Panels"]
    )
    motion_planning = next(
        display
        for display in config["Visualization Manager"]["Displays"]
        if display["Class"] == "moveit_rviz_plugin/MotionPlanning"
    )
    assert (
        motion_planning["Planning Request"]["Planning Group"]
        == "ur_manipulator"
    )


def _transform(rotation=None, translation=None):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def _rpy_matrix(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _axis_angle(axis, angle: float):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (
        np.eye(3)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _urdf_forward_kinematics(urdf, joint_values):
    children = {}
    for joint in urdf.findall("joint"):
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        children.setdefault(parent, []).append((joint, child))

    transforms = {"world": np.eye(4)}
    pending = ["world"]
    while pending:
        parent = pending.pop()
        for joint, child in children.get(parent, []):
            origin = joint.find("origin")
            xyz = [0.0, 0.0, 0.0]
            rpy = [0.0, 0.0, 0.0]
            if origin is not None:
                xyz = [
                    float(value)
                    for value in origin.attrib.get("xyz", "0 0 0").split()
                ]
                rpy = [
                    float(value)
                    for value in origin.attrib.get("rpy", "0 0 0").split()
                ]
            local = _transform(_rpy_matrix(*rpy), xyz)
            position = joint_values.get(joint.attrib["name"], 0.0)
            if joint.attrib["type"] in {"revolute", "continuous"}:
                axis = [
                    float(value)
                    for value in joint.find("axis").attrib["xyz"].split()
                ]
                local = local @ _transform(_axis_angle(axis, position))
            elif joint.attrib["type"] == "prismatic":
                axis = np.array(
                    [
                        float(value)
                        for value in joint.find("axis").attrib["xyz"].split()
                    ]
                )
                local = local @ _transform(translation=axis * position)
            transforms[child] = transforms[parent] @ local
            pending.append(child)
    return transforms


def test_urdf_and_mujoco_frames_match_at_home() -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path()))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    joint_values = {}
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_values[name] = data.qpos[model.jnt_qposadr[joint_id]]
    urdf = ElementTree.parse(
        description_path() / "urdf" / "huanpan_workcell.urdf"
    ).getroot()
    urdf_transforms = _urdf_forward_kinematics(urdf, joint_values)

    for urdf_link, mujoco_body in BODY_LINK_MAP.items():
        body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, mujoco_body
        )
        mujoco_transform = _transform(
            data.xmat[body_id].reshape(3, 3), data.xpos[body_id]
        )
        assert np.allclose(
            urdf_transforms[urdf_link], mujoco_transform, atol=1e-8
        ), urdf_link
