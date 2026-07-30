from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    share = Path(get_package_share_directory("gsi_search_bridge"))
    bridge_config = str(share / "config" / "gz_bridge.yaml")
    search_config = str(share / "config" / "search_params.yaml")
    return LaunchDescription([
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="gsi_gz_bridge",
            parameters=[{"config_file": bridge_config}],
            output="screen",
        ),
        Node(
            package="gsi_search_bridge",
            executable="position_controller",
            name="gsi_position_controller",
            parameters=[search_config],
            output="screen",
        ),
        Node(
            package="gsi_search_bridge",
            executable="search_node",
            name="gsi_search_node",
            parameters=[search_config],
            output="screen",
        ),
    ])
