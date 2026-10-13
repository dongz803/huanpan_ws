from pathlib import Path
from xml.etree import ElementTree

import mujoco
import yaml


ARM_JOINTS = {
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
}
GRIPPER_JOINTS = {"R1_1_joint"}
AG95_MIMIC_MULTIPLIERS = {
    "R1_2_joint": 1.0,
    "R2_1_joint": 1.0,
    "M_joint": -0.032,
    "MR_joint": 0.37,
    "MR_1_1_joint": 1.0,
    "MR_2_1_joint": 1.38,
    "ML_joint": 0.37,
    "ML_1_1_joint": 1.05,
    "ML_2_1_joint": 1.38,
    "L1_1_joint": 1.0,
    "L1_2_joint": 1.0,
    "L2_1_joint": -1.0,
}


def source_description() -> Path:
    return Path(__file__).resolve().parents[2] / "huanpan_description"


def test_mujoco_and_urdf_joint_contract() -> None:
    description = source_description()
    model = mujoco.MjModel.from_xml_path(
        str(
            description
            / "mujoco"
            / "universal_robots_ur5e"
            / "workcell.xml"
        )
    )
    mujoco_joints = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
        for index in range(model.njnt)
    }
    urdf = ElementTree.parse(
        description / "urdf" / "huanpan_workcell.urdf"
    )
    movable = [
        joint
        for joint in urdf.findall("joint")
        if joint.attrib["type"] in {"revolute", "continuous", "prismatic"}
    ]
    urdf_movable_joints = {joint.attrib["name"] for joint in movable}
    urdf_command_joints = {
        joint.attrib["name"] for joint in movable if joint.find("mimic") is None
    }
    controlled = ARM_JOINTS | GRIPPER_JOINTS
    assert controlled <= mujoco_joints
    assert urdf_movable_joints <= mujoco_joints
    assert controlled == urdf_command_joints
    assert urdf_movable_joints == controlled | set(AG95_MIMIC_MULTIPLIERS)
    assert model.nu == len(controlled)


def test_home_keyframe_is_complete() -> None:
    model_path = (
        source_description()
        / "mujoco"
        / "universal_robots_ur5e"
        / "workcell.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    assert home_id >= 0
    assert model.nq == 19
    assert model.nu == 7


def test_ag95_model_and_mimic_contract() -> None:
    description = source_description()
    model = mujoco.MjModel.from_xml_path(
        str(
            description
            / "mujoco"
            / "universal_robots_ur5e"
            / "workcell.xml"
        )
    )
    urdf = ElementTree.parse(
        description / "urdf" / "huanpan_workcell.urdf"
    )

    for joint_name, multiplier in AG95_MIMIC_MULTIPLIERS.items():
        mimic = urdf.find(f"joint[@name='{joint_name}']/mimic")
        assert mimic is not None
        assert mimic.attrib["joint"] == "R1_1_joint"
        assert float(mimic.attrib["multiplier"]) == multiplier

    ag95_links = {
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
    assert all(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
        for name in ag95_links
    )
    assert model.neq == len(AG95_MIMIC_MULTIPLIERS)

    master = urdf.find("joint[@name='R1_1_joint']/limit")
    assert master is not None
    assert float(master.attrib["lower"]) == -0.94
    assert float(master.attrib["upper"]) == 0.0


def test_sa5212_rack_geometry_contract() -> None:
    description = source_description()
    model = mujoco.MjModel.from_xml_path(
        str(
            description
            / "mujoco"
            / "universal_robots_ur5e"
            / "workcell.xml"
        )
    )

    chassis_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "sa5212_chassis"
    )
    assert chassis_id >= 0
    assert max(
        abs(actual - expected)
        for actual, expected in zip(
            model.body_pos[chassis_id], (0.7075, 0.0, 0.750)
        )
    ) < 1e-9

    bay_names = {f"drive_bay_{index:02d}_recess" for index in range(12)}
    model_geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index)
        for index in range(model.ngeom)
    }
    assert bay_names <= model_geom_names

    top_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "sa5212_top"
    )
    side_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "sa5212_left_side"
    )
    assert top_id >= 0 and side_id >= 0
    assert abs(2.0 * model.geom_size[top_id][0] - 0.1750) < 1e-9
    assert abs(2.0 * model.geom_size[top_id][1] - 0.4300) < 1e-9
    assert abs(2.0 * model.geom_size[side_id][2] - 0.0850) < 1e-9

    slide_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, "drive_slide_joint"
    )
    old_drive_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_drive"
    )
    assert slide_id == -1
    assert old_drive_id == -1

    carried_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "carried_drive"
    )
    gripper_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base"
    )
    drive_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "carried_drive_collision"
    )
    drive_visual_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "carried_drive_visual"
    )
    drive_tip_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "drive_tip"
    )
    assert carried_id >= 0 and gripper_id >= 0
    assert model.body_parentid[carried_id] == gripper_id
    assert drive_geom_id >= 0 and drive_visual_id >= 0 and drive_tip_id >= 0
    assert max(
        abs(actual - expected)
        for actual, expected in zip(
            2.0 * model.geom_size[drive_geom_id], (0.025, 0.100, 0.175)
        )
    ) < 1e-9
    assert max(
        abs(actual - expected)
        for actual, expected in zip(
            2.0 * model.geom_size[drive_visual_id], (0.025, 0.100, 0.175)
        )
    ) < 1e-9
    assert model.geom_group[drive_visual_id] == 2
    assert model.geom_contype[drive_visual_id] == 0
    assert model.geom_conaffinity[drive_visual_id] == 0
    assert model.geom_group[drive_geom_id] == 3

    slot_entry_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "slot_09_entry"
    )
    slot_goal_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "slot_09_goal"
    )
    assert slot_entry_id >= 0 and slot_goal_id >= 0

    urdf = ElementTree.parse(description / "urdf" / "huanpan_workcell.urdf")
    rack_joint = urdf.find("joint[@name='world_to_rack']/origin")
    drive_joint = urdf.find(
        "joint[@name='gripper_tcp_to_carried_drive']/origin"
    )
    gripper_tcp_joint = urdf.find(
        "joint[@name='gripper_tcp_joint']/origin"
    )
    drive_tip_joint = urdf.find(
        "joint[@name='carried_drive_to_drive_tip']/origin"
    )
    slot_entry_joint = urdf.find(
        "joint[@name='rack_to_slot_09_entry']/origin"
    )
    slot_goal_joint = urdf.find(
        "joint[@name='rack_to_slot_09_goal']/origin"
    )
    assert rack_joint is not None
    assert rack_joint.attrib["xyz"] == "0.7075 0 0.750"
    assert gripper_tcp_joint is not None
    assert gripper_tcp_joint.attrib["xyz"] == "0 -0.01985 0.200"
    assert drive_joint is not None and drive_joint.attrib["xyz"] == "0 0 0.0875"
    assert drive_tip_joint is not None
    assert drive_tip_joint.attrib["xyz"] == "0 0 0.0875"
    assert slot_entry_joint is not None
    assert slot_entry_joint.attrib["xyz"] == "-0.1025 0.0525 -0.0275"
    assert slot_goal_joint is not None
    assert slot_goal_joint.attrib["xyz"] == "0.080 0.0525 -0.0275"
    assert urdf.find("link[@name='target_drive']") is None


def test_moveit_controller_joint_contract() -> None:
    moveit_config = (
        source_description().parent
        / "huanpan_moveit_config"
        / "config"
    )
    with (moveit_config / "moveit_controllers.yaml").open(
        encoding="utf-8"
    ) as stream:
        controllers = yaml.safe_load(stream)[
            "moveit_simple_controller_manager"
        ]

    assert set(controllers["ur5e_arm_controller"]["joints"]) == ARM_JOINTS
    assert set(controllers["gripper_controller"]["joints"]) == GRIPPER_JOINTS

    srdf = ElementTree.parse(moveit_config / "huanpan.srdf")
    manipulator_chain = srdf.find("group[@name='ur_manipulator']/chain")
    assert manipulator_chain is not None
    assert manipulator_chain.attrib["tip_link"] == "drive_tip"
    group_states = {
        (state.attrib["group"], state.attrib["name"])
        for state in srdf.findall("group_state")
    }
    assert ("ur_manipulator", "home") in group_states
    assert ("gripper", "open") in group_states
    assert ("gripper", "closed") in group_states

    disabled_pairs = {
        frozenset((entry.attrib["link1"], entry.attrib["link2"]))
        for entry in srdf.findall("disable_collisions")
    }
    required_internal_pairs = {
        frozenset(("base_link", "shoulder_link")),
        frozenset(("wrist_3_link", "gripper_base")),
        frozenset(("L1_2_link", "ML_2_1_link")),
        frozenset(("ML_1_1_link", "gripper_base")),
        frozenset(("gripper_base", "MR_1_1_link")),
        frozenset(("R1_2_link", "MR_2_1_link")),
    }
    assert required_internal_pairs <= disabled_pairs

    with (moveit_config / "insertion_targets.yaml").open(
        encoding="utf-8"
    ) as stream:
        insertion = yaml.safe_load(stream)["insertion"]
    assert insertion["tip_link"] == "drive_tip"
    assert insertion["entry_frame"] == "slot_09_entry"
    assert insertion["goal_frame"] == "slot_09_goal"
    assert insertion["drive_tip_orientation_xyzw"] == [
        0.70710678,
        0.0,
        0.70710678,
        0.0,
    ]


def test_slot_09_insertion_reference_keyframes() -> None:
    model = mujoco.MjModel.from_xml_path(
        str(
            source_description()
            / "mujoco"
            / "universal_robots_ur5e"
            / "workcell.xml"
        )
    )
    data = mujoco.MjData(model)
    drive_tip_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "drive_tip"
    )

    for key_name, target_site_name in (
        ("slot_09_entry", "slot_09_entry"),
        ("slot_09_inserted", "slot_09_goal"),
    ):
        key_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_KEY, key_name
        )
        target_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, target_site_name
        )
        assert key_id >= 0 and target_id >= 0
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        mujoco.mj_forward(model, data)
        assert max(
            abs(actual - expected)
            for actual, expected in zip(
                data.site_xpos[drive_tip_id], data.site_xpos[target_id]
            )
        ) < 5e-5
        drive_rotation = data.site_xmat[drive_tip_id].reshape(3, 3)
        assert max(
            abs(actual - expected)
            for actual, expected in zip(
                drive_rotation[:, 2], (1.0, 0.0, 0.0)
            )
        ) < 5e-5
        assert data.ncon == 0
