from ..base.enums import Category, BuildingName, ShapeType, RenderStyle, ColorScheme, BuildingStatus

BUILDING_TEMPLATES = {
    BuildingName.HOSPITAL.value: {
        "category": Category.BUILDING.value,
        "type": BuildingName.HOSPITAL.value,
        "description": "Medical facility",
        "size": {"length": 90.0, "width": 70.0, "height": 80.0, "unit": "meters"},
        "status": {
            "options": [BuildingStatus.DISCOVERED.value, BuildingStatus.UNDISCOVERED.value],
            "default": BuildingStatus.UNDISCOVERED.value
        },
        "appearance": {
            "type": ShapeType.CROSS.value,
            "style": RenderStyle.SOLID.value,
            "color": ColorScheme.RED.value,
            "border_width": 2.0,
            "border_color": "#FFFFFF"
        }
    },
    BuildingName.POWER_STATION.value: {
        "category": Category.BUILDING.value,
        "type": BuildingName.POWER_STATION.value,
        "description": "Electric power facility",
        "size": {"length": 100.0, "width": 100.0, "height": 40.0, "unit": "meters"},
        "status": {
            "options": [BuildingStatus.DISCOVERED.value, BuildingStatus.UNDISCOVERED.value],
            "default": BuildingStatus.UNDISCOVERED.value
        },
        "appearance": {
            "type": ShapeType.RECTANGLE.value,
            "style": RenderStyle.SOLID.value,
            "color": ColorScheme.ORANGE.value,
            "border_width": 3.0,
            "border_color": "#000000"
        }
    },
    BuildingName.MALL.value: {
        "category": Category.BUILDING.value,
        "type": BuildingName.MALL.value,
        "description": "Commercial shopping center",
        "size": {"length": 180.0, "width": 180.0, "height": 15.0, "unit": "meters"},
        "status": {
            "options": [BuildingStatus.DISCOVERED.value, BuildingStatus.UNDISCOVERED.value],
            "default": BuildingStatus.UNDISCOVERED.value
        },
        "appearance": {
            "type": ShapeType.L_SHAPE.value,
            "style": RenderStyle.SOLID.value,
            "color": ColorScheme.PURPLE.value,
            "border_width": 2.5,
            "border_color": "#FFFFFF"
        }
    },
    BuildingName.PARKING.value: {
        "category": Category.BUILDING.value,
        "type": BuildingName.PARKING.value,
        "description": "Parking lot",
        "size": {"length": 80.0, "width": 70.0, "height": 2.0, "unit": "meters"},
        "status": {
            "options": [BuildingStatus.DISCOVERED.value, BuildingStatus.UNDISCOVERED.value],
            "default": BuildingStatus.UNDISCOVERED.value
        },
        "appearance": {
            "type": ShapeType.RECTANGLE.value,
            "style": RenderStyle.DASHED.value,
            "color": ColorScheme.GRAY.value,
            "border_width": 2.0,
            "border_color": "#424242"
        }
    },
    BuildingName.LIBRARY.value: {
        "category": Category.BUILDING.value,
        "type": BuildingName.LIBRARY.value,
        "description": "Public library",
        "size": {"length": 80.0, "width": 80.0, "height": 30.0, "unit": "meters"},
        "status": {
            "options": [BuildingStatus.DISCOVERED.value, BuildingStatus.UNDISCOVERED.value],
            "default": BuildingStatus.UNDISCOVERED.value
        },
        "appearance": {
            "type": ShapeType.RECTANGLE.value,
            "style": RenderStyle.SOLID.value,
            "color": ColorScheme.BLUE.value,
            "border_width": 2.0,
            "border_color": "#1976D2"
        }
    },
    BuildingName.HOTEL.value: {
        "category": Category.BUILDING.value,
        "type": BuildingName.HOTEL.value,
        "description": "Hotel facility for storing materials and equipment",
        "size": {"length": 120.0, "width": 80.0, "height": 25.0, "unit": "meters"},
        "status": {
            "options": [BuildingStatus.DISCOVERED.value, BuildingStatus.UNDISCOVERED.value],
            "default": BuildingStatus.UNDISCOVERED.value
        },
        "appearance": {
            "type": ShapeType.RECTANGLE.value,
            "style": RenderStyle.SOLID.value,
            "color": ColorScheme.BROWN.value if hasattr(ColorScheme, "BROWN") else ColorScheme.GRAY.value,
            "border_width": 2.5,
            "border_color": "#6D4C41"
        }
    },
    BuildingName.ROBOT_BASE.value: {
        "category": Category.BUILDING.value,
        "type": BuildingName.ROBOT_BASE.value,
        "description": "Robot operation base",
        "size": {"radius": 60.0, "height": 10.0, "unit": "meters"},
        "status": {
            "options": [BuildingStatus.DISCOVERED.value, BuildingStatus.UNDISCOVERED.value],
            "default": BuildingStatus.DISCOVERED.value
        },
        "appearance": {
            "type": ShapeType.CIRCLE.value,
            "style": RenderStyle.SOLID.value,
            "color": ColorScheme.CYAN.value,
            "border_width": 3.0,
            "border_color": "#00BCD4"
        }
    },
}