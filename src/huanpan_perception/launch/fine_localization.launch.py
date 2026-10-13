"""Start the D435 and the YOLO-assisted fine-localization pipeline."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    config_file = PathJoinSubstitution(
        [FindPackageShare("huanpan_perception"), "config", "perception.yaml"]
    )
    declared = [
        DeclareLaunchArgument("huanpan_params_file", default_value=config_file),
        DeclareLaunchArgument("start_camera", default_value="true"),
        DeclareLaunchArgument(
            "model_path",
            default_value=(
                "/home/lzt/huanpan_ws/outputs/perception/runs/"
                "slot_opening_v1/weights/best.pt"
            ),
        ),
        DeclareLaunchArgument("target_class", default_value="slot_opening"),
        DeclareLaunchArgument("target_slot", default_value="0"),
        DeclareLaunchArgument("refinement_mode", default_value="edges"),
        DeclareLaunchArgument(
            "reference_file",
            default_value=(
                "/home/lzt/huanpan_ws/outputs/perception/references/"
                "slot_references.yaml"
            ),
        ),
        DeclareLaunchArgument("color_profile", default_value="1280x720x30"),
        DeclareLaunchArgument("depth_profile", default_value="848x480x30"),
    ]
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
    localizer = Node(
        package="huanpan_perception",
        executable="fine_localization_node",
        name="huanpan_fine_localizer",
        output="screen",
        parameters=[
            LaunchConfiguration("huanpan_params_file"),
            {
                "model_path": LaunchConfiguration("model_path"),
                "target_class": LaunchConfiguration("target_class"),
                "target_slot": LaunchConfiguration("target_slot"),
                "refinement_mode": LaunchConfiguration("refinement_mode"),
                "reference_file": LaunchConfiguration("reference_file"),
            },
        ],
    )
    return LaunchDescription(declared + [camera, localizer])
