from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    description_share = Path(
        get_package_share_directory("huanpan_description")
    )
    robot_description = (
        description_share / "urdf" / "huanpan_workcell.urdf"
    ).read_text(encoding="utf-8")

    show_viewer = LaunchConfiguration("show_viewer")
    disturbance_enabled = LaunchConfiguration("disturbance_enabled")
    disturbance_scale = LaunchConfiguration("disturbance_scale")
    force_guard_enabled = LaunchConfiguration("force_guard_enabled")
    force_guard_threshold_n = LaunchConfiguration(
        "force_guard_threshold_n"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("show_viewer", default_value="true"),
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
                parameters=[{"robot_description": robot_description}],
                output="screen",
            ),
            Node(
                package="huanpan_mujoco",
                executable="mujoco_bridge",
                parameters=[
                    str(
                        Path(get_package_share_directory("huanpan_mujoco"))
                        / "config"
                        / "bridge.yaml"
                    ),
                    {
                        "show_viewer": show_viewer,
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
        ]
    )
