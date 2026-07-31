from glob import glob
from setuptools import find_packages, setup


package_name = "gsi_search_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="GSI Maintainer",
    maintainer_email="maintainer@example.com",
    description="Gazebo sensor and viewpoint bridge for GSI active search.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "search_node = gsi_search_bridge.search_node:main",
            "position_controller = gsi_search_bridge.position_controller:main",
            "mavros_offboard_controller = gsi_search_bridge.mavros_offboard_controller:main",
            "color_target_detector = gsi_search_bridge.color_target_detector:main",
            "generate_search_world = gsi_search_bridge.search_world_generator:main",
            "searchworld_stability = gsi_search_bridge.stability:main",
            "searchworld_v2_results = gsi_search_bridge.v2_results:main",
        ],
    },
)
