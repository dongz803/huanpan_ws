"""Launch a dataset capture session from the D435.

    ./scripts/capture_quick.sh                       # 推荐入口
    ros2 launch huanpan_perception capture.launch.py start_camera:=true

The camera node is created directly rather than by including
``realsense2_camera``'s ``rs_launch.py``. That launch file treats
``config_file`` as its own argument and iterates over *every* launch
configuration in the context, so including it caused our ``perception.yaml`` to
be loaded as the camera's config and produced a wall of misleading
"Parameter ... is not supported" warnings. Declaring the node here keeps the two
parameter spaces separate and lets us set the stream profiles explicitly.
"""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    params_file = PathJoinSubstitution(
        [FindPackageShare("huanpan_perception"), "config", "perception.yaml"]
    )

    declared = [
        DeclareLaunchArgument(
            "huanpan_params_file",
            default_value=params_file,
            description="Parameter file for huanpan_capture. Named differently "
            "from realsense2_camera's own config_file on purpose.",
        ),
        DeclareLaunchArgument(
            "start_camera",
            default_value="true",
            description="Also start the RealSense driver",
        ),
        DeclareLaunchArgument(
            "output_dir",
            default_value="outputs/perception/datasets/rack_grid_v1",
            description="Dataset ROOT. Colour frames go to <root>/images and "
            "depth to <root>/depth; do not append /images yourself.",
        ),
        DeclareLaunchArgument(
            "prefix",
            default_value="capture",
            description="Frame name prefix; frames become <prefix>_00001.png. "
            "Use one prefix per session so the train/val split can keep a "
            "session on one side.",
        ),
        DeclareLaunchArgument(
            "save_depth",
            default_value="true",
            description="Save the aligned depth frame next to each colour frame",
        ),
        DeclareLaunchArgument("max_frames", default_value="100000"),
        DeclareLaunchArgument(
            "interval_s",
            default_value="1.0",
            description="Seconds between frames in interval mode",
        ),
        DeclareLaunchArgument(
            "mode",
            default_value="auto",
            description="interval (headless), interactive (terminal), or auto",
        ),
        DeclareLaunchArgument(
            "preview",
            default_value="true",
            description="Live OpenCV window: SPACE saves, d toggles depth, q quits. "
            "Works under ros2 launch because the keyboard comes from the window.",
        ),
        DeclareLaunchArgument(
            "color_profile",
            default_value="1280x720x30",
            description="D435 colour stream profile for fine localization. "
            "Override with 640x480x30 if the USB link is unstable.",
        ),
        DeclareLaunchArgument(
            "depth_profile",
            default_value="848x480x30",
            description="D435 depth stream profile (depth maxes out at 1280x720)",
        ),
    ]

    overrides = {
        "output_dir": LaunchConfiguration("output_dir"),
        "prefix": LaunchConfiguration("prefix"),
        "save_depth": LaunchConfiguration("save_depth"),
        "max_frames": LaunchConfiguration("max_frames"),
        "interval_s": LaunchConfiguration("interval_s"),
        "mode": LaunchConfiguration("mode"),
        "preview": LaunchConfiguration("preview"),
    }

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
                # 精定位采集只使用 RGB-D。直接启动相机节点时需要显式关闭
                # D435i 的 Motion Module，否则可能触发 IIO 权限/硬件错误。
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

    capture = Node(
        package="huanpan_perception",
        executable="capture_node",
        name="huanpan_capture",
        output="screen",
        parameters=[LaunchConfiguration("huanpan_params_file"), overrides],
    )

    return LaunchDescription(declared + [camera, capture])
