from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def generate_launch_description() -> LaunchDescription:
    description_share = Path(
        get_package_share_directory("huanpan_description")
    )
    moveit_share = Path(get_package_share_directory("huanpan_moveit_config"))
    mujoco_share = Path(get_package_share_directory("huanpan_mujoco"))

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
    robot_description_planning = {
        "robot_description_planning": load_yaml(
            moveit_share / "config" / "joint_limits.yaml"
        )
    }
    ompl_pipeline = {
        "planning_pipelines": ["ompl"],
        "default_planning_pipeline": "ompl",
        "ompl": load_yaml(moveit_share / "config" / "ompl_planning.yaml"),
    }
    trajectory_execution = {
        "allow_trajectory_execution": True,
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.5,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.02,
    }
    planning_scene_monitor = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }
    controller_config = load_yaml(
        moveit_share / "config" / "moveit_controllers.yaml"
    )

    show_mujoco = LaunchConfiguration("show_mujoco")
    use_rviz = LaunchConfiguration("use_rviz")
    disturbance_enabled = LaunchConfiguration("disturbance_enabled")
    disturbance_scale = LaunchConfiguration("disturbance_scale")
    force_guard_enabled = LaunchConfiguration("force_guard_enabled")
    force_guard_threshold_n = LaunchConfiguration(
        "force_guard_threshold_n"
    )
    common_moveit_parameters = [
        robot_description,
        robot_description_semantic,
        robot_description_kinematics,
        robot_description_planning,
    ]

    return LaunchDescription(
        [
            DeclareLaunchArgument("show_mujoco", default_value="true"),
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument(
                "disturbance_enabled", default_value="false"
            ),
            DeclareLaunchArgument("disturbance_scale", default_value="1.0"),
            DeclareLaunchArgument(
                "force_guard_enabled", default_value="false"
            ),
            DeclareLaunchArgument(
                "force_guard_threshold_n", default_value="25.0"
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                parameters=[robot_description],
                output="screen",
            ),
            Node(
                package="huanpan_mujoco",
                executable="mujoco_bridge",
                parameters=[
                    str(mujoco_share / "config" / "bridge.yaml"),
                    {
                        "show_viewer": show_mujoco,
                        "disturbance_enabled": disturbance_enabled,
                        "disturbance_scale": disturbance_scale,
                        "force_guard_enabled": force_guard_enabled,
                        "force_guard_threshold_n": (
                            force_guard_threshold_n
                        ),
                    },
                ],
                output="screen",
            ),
            Node(
                package="moveit_ros_move_group",
                executable="move_group",
                output="screen",
                parameters=common_moveit_parameters
                + [
                    ompl_pipeline,
                    trajectory_execution,
                    planning_scene_monitor,
                    controller_config,
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                arguments=["-d", str(moveit_share / "rviz" / "moveit.rviz")],
                output="screen",
                condition=IfCondition(use_rviz),
                parameters=common_moveit_parameters,
            ),
        ]
    )
