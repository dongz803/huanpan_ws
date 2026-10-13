from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("huanpan_motion"))
    description_share = Path(get_package_share_directory("huanpan_description"))
    moveit_share = Path(get_package_share_directory("huanpan_moveit_config"))

    robot_description = {
        "robot_description": (
            description_share / "urdf" / "huanpan_workcell.urdf"
        ).read_text(encoding="utf-8")
    }
    robot_description_semantic = {
        "robot_description_semantic": (
            moveit_share / "config" / "huanpan.srdf"
        ).read_text(encoding="utf-8")
    }
    robot_description_kinematics = {
        "robot_description_kinematics": load_yaml(
            moveit_share / "config" / "kinematics.yaml"
        )
    }
    return LaunchDescription(
        [
            Node(
                package="huanpan_motion",
                executable="motion_node",
                name="huanpan_motion_node",
                output="screen",
                parameters=[
                    str(package_share / "config" / "motion_parameters.yaml"),
                    robot_description,
                    robot_description_semantic,
                    robot_description_kinematics,
                ],
            )
        ]
    )
