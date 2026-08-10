from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("gsi_search_bridge"))
    mavros_share = Path(get_package_share_directory("mavros"))
    default_bridge_config = str(share / "config" / "visionflow_gz_bridge.yaml")
    default_search_config = str(share / "config" / "visionflow_search_params.yaml")

    start_mavros = LaunchConfiguration("start_mavros")
    start_sensor_bridge = LaunchConfiguration("start_sensor_bridge")
    start_baseline_detector = LaunchConfiguration("start_baseline_detector")
    start_yolo_detector = LaunchConfiguration("start_yolo_detector")
    fcu_url = LaunchConfiguration("fcu_url")
    bridge_config = LaunchConfiguration("sensor_bridge_config")
    search_config = LaunchConfiguration("search_config")
    search_time_budget_s = LaunchConfiguration("search_time_budget_s")

    return LaunchDescription([
        DeclareLaunchArgument("start_mavros", default_value="true"),
        DeclareLaunchArgument("start_sensor_bridge", default_value="true"),
        DeclareLaunchArgument("start_baseline_detector", default_value="false"),
        DeclareLaunchArgument("start_yolo_detector", default_value="true"),
        DeclareLaunchArgument(
            "sensor_bridge_config",
            default_value=default_bridge_config,
        ),
        DeclareLaunchArgument(
            "search_config",
            default_value=default_search_config,
        ),
        DeclareLaunchArgument(
            "search_time_budget_s",
            default_value="180.0",
        ),
        DeclareLaunchArgument(
            "fcu_url",
            default_value="udp://:14540@localhost:14557",
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(str(mavros_share / "launch" / "px4.launch")),
            launch_arguments={"fcu_url": fcu_url}.items(),
            condition=IfCondition(start_mavros),
        ),
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="gsi_visionflow_sensor_bridge",
            parameters=[{"config_file": bridge_config}],
            output="screen",
            condition=IfCondition(start_sensor_bridge),
        ),
        Node(
            package="gsi_search_bridge",
            executable="color_target_detector",
            name="gsi_color_target_detector",
            parameters=[search_config],
            output="screen",
            condition=IfCondition(start_baseline_detector),
        ),
        Node(
            package="gsi_search_bridge",
            executable="yolo_target_detector",
            name="gsi_yolo_target_detector",
            parameters=[search_config],
            output="screen",
            condition=IfCondition(start_yolo_detector),
        ),
        Node(
            package="gsi_search_bridge",
            executable="mavros_offboard_controller",
            name="gsi_mavros_offboard_controller",
            parameters=[search_config],
            output="screen",
        ),
        Node(
            package="gsi_search_bridge",
            executable="search_node",
            name="gsi_search_node",
            parameters=[
                search_config,
                {
                    "search_time_budget_s": ParameterValue(
                        search_time_budget_s,
                        value_type=float,
                    ),
                },
            ],
            output="screen",
        ),
    ])
