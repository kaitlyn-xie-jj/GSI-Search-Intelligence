# robots.py
from ..base.enums import Category, RobotStatus, RobotName, RobotAttributeName

ROBOT_TEMPLATES = {
    RobotName.UAV.value: {
        "category": Category.ROBOT.value,
        "type": RobotName.UAV.value,
        "description": "Aerial robot with photo and inspection abilities",
        "attributes": {
            RobotAttributeName.BATTERY_LEVEL.value: {"value": 100.0, "min": 0, "max": 100},
            RobotAttributeName.ALTITUDE.value: {"value": 0.0, "desc": "Altitude", "min": 0, "max": 50},
        },
        "size": {"length": 5.0, "width": 5.0, "height": 0.5, "unit": "meters"},
        "status": {"options": [s.value for s in RobotStatus], "default": RobotStatus.IDLE.value}
    },
    RobotName.FW_UAV.value: {
        "category": Category.ROBOT.value,
        "type": RobotName.FW_UAV.value,
        "description": "Fixed-wing aerial robot for long-range surveillance and mapping",
        "attributes": {
            RobotAttributeName.BATTERY_LEVEL.value: {"value": 100.0, "min": 0, "max": 100},
            RobotAttributeName.ALTITUDE.value: {"value": 0.0, "desc": "Altitude (m)", "min": 0, "max": 300},
        },
        "size": {"length": 10.0, "width": 15.0, "height": 1.0, "unit": "meters"},
        "status": {"options": [s.value for s in RobotStatus], "default": RobotStatus.IDLE.value}
    },
    RobotName.UGV.value: {
        "category": Category.ROBOT.value,
        "type": RobotName.UGV.value,
        "description": "Ground robot for transport and search",
        "attributes": {
            RobotAttributeName.BATTERY_LEVEL.value: {"value": 100.0, "min": 0, "max": 100},
            RobotAttributeName.MAX_PAYLOAD_KG.value: {"value": 150.0, "desc": "Max payload (kg)", "min": 0, "max": 200},
            RobotAttributeName.CURRENT_LOAD.value: {"value": 0.0, "desc": "Current load (kg)", "min": 0}
        },
        "size": {"length": 7.0, "width": 7.0, "height": 2.0, "unit": "meters"},
        "status": {"options": [s.value for s in RobotStatus], "default": RobotStatus.IDLE.value}
    },
    RobotName.Quadruped.value: {
        "category": Category.ROBOT.value,
        "type": RobotName.Quadruped.value,
        "description": "Four-legged robot for rough terrain and following",
        "attributes": {
            RobotAttributeName.BATTERY_LEVEL.value: {"value": 100.0, "min": 0, "max": 100},
        },
        "size": {"length": 3.0, "width": 2.0, "height": 1.5, "unit": "meters"},
        "status": {"options": [s.value for s in RobotStatus], "default": RobotStatus.IDLE.value}
    },
    RobotName.Humanoid.value: {
        "category": Category.ROBOT.value,
        "type": RobotName.Humanoid.value,
        "description": "Humanoid robot for object manipulation",
        "attributes": {
            RobotAttributeName.BATTERY_LEVEL.value: {"value": 100.0, "min": 0, "max": 100},
        },
        "size": {"length": 1.5, "width": 1.0, "height": 2.0, "unit": "meters"},
        "status": {"options": [s.value for s in RobotStatus], "default": RobotStatus.IDLE.value}
    }
}
