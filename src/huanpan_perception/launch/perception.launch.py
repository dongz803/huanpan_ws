"""Launch the D435 (optional) and the YOLO rack/slot detector.

    ros2 launch huanpan_perception perception.launch.py
    ros2 launch huanpan_perception perception.launch.py start_camera:=true

The detector node runs under ``ros2 run``'s interpreter and re-executes itself
with the workspace virtualenv (see ``huanpan_perception.launcher``), so no
special python command is needed.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_file = PathJoinSubstitution(
        [FindPackageShare("huanpan_perception"), "config", "perception.yaml"]
    )

    declared = [
        DeclareLaunchArgument(
            "huanpan_params_file",
            default_value=config_file,
            description="Parameter file for huanpan_detector. Named differently "
            "from realsense2_camera's own config_file on purpose.",
        ),
        DeclareLaunchArgument(
            "start_camera",
            default_value="false",
            description="Also start realsense2_camera with aligned depth",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="",
            description="Override the detector weights path",
        ),
        DeclareLaunchArgument(
            "color_profile",
            default_value="1280x720x30",
            description="D435 colour stream profile",
        ),
        DeclareLaunchArgument(
            "depth_profile",
            default_value="848x480x30",
            description="D435 source depth profile; aligned output follows colour",
        ),
    ]

    overrides = {"model_path": LaunchConfiguration("model_path")}

    # The camera node is declared directly instead of including realsense2_camera's
    # rs_launch.py: that file treats `config_file` as its own argument and warns
    # about every launch configuration it does not recognise, which collided
    # with our parameter file. See capture.launch.py for the full note.
    camera = Node(
        package="realsense2_camera",
        executable="realsense2_camera_node",
        name="camera",
        namespace="camera",
        output="screen",
        condition=IfCondition(LaunchConfiguration("start_camera")),
        parameters=[
            {
                "align_depth.enable": True,
                "enable_color": True,
                "enable_depth": True,
                "enable_gyro": False,
                "enable_accel": False,
                "enable_infra": False,
                "enable_infra1": False,
                "enable_infra2": False,
                "pointcloud.enable": False,
                "rgb_camera.color_profile": LaunchConfiguration("color_profile"),
                "depth_module.depth_profile": LaunchConfiguration("depth_profile"),
            }
        ],
    )

    detector = Node(
        package="huanpan_perception",
        executable="detector_node",
        name="huanpan_detector",
        output="screen",
        parameters=[LaunchConfiguration("huanpan_params_file"), overrides],
    )

    return LaunchDescription(declared + [camera, detector])
