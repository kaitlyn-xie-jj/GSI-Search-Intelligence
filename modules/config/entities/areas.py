# areas.py
from ..base.enums import AreaType, Category, PropStatus

AREA_TEMPLATES = {
    AreaType.GARDEN.value: {
        "category": Category.AREA.value,
        "type": AreaType.GARDEN.value,
        "description": "Green public garden",
        "generation_params": {
            "method": "voronoi_or_subdivision",
            "size_hint": "large",
            "shape_complexity": 0.5,
            "placement_rule": {"try_near": "water_body", "avoid": ["industrial_park"]}
        },
        "appearance": {"style": "solid", "color": "#4CAF50"},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    AreaType.SQUARE.value: {
        "category": Category.AREA.value,
        "type": AreaType.SQUARE.value,
        "description": "City square / plaza",
        "generation_params": {
            "method": "subdivision_grid",
            "size_hint": "medium",
            "shape_complexity": 0.2,
            "placement_rule": {"try_near": "street_segment"}
        },
        "appearance": {"style": "solid", "color": "#9E9E9E"},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    AreaType.WATER_BODY.value: {
        "category": Category.AREA.value,
        "type": AreaType.WATER_BODY.value,
        "description": "Lakes or rivers",
        "generation_params": {"method": "perlin_or_ovals", "size_hint": "large", "shape_complexity": 0.8},
        "appearance": {"style": "solid", "color": "#00BCD4"},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    AreaType.GREENBELT.value: {
        "category": Category.AREA.value,
        "type": AreaType.GREENBELT.value,
        "description": "Linear greenbelt",
        "generation_params": {
            "method": "subdivision_stripes",
            "size_hint": "medium",
            "shape_complexity": 0.3,
            "placement_rule": {"try_near": "street_segment"}
        },
        "appearance": {"style": "solid", "color": "#81C784"},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    AreaType.INDUSTRIAL_PARK.value: {
        "category": Category.AREA.value,
        "type": AreaType.INDUSTRIAL_PARK.value,
        "description": "Industrial garden zone",
        "generation_params": {"method": "voronoi_or_subdivision", "size_hint": "large", "shape_complexity": 0.4},
        "appearance": {"style": "solid", "color": "#FF9800"},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    AreaType.CAMPUS.value: {
        "category": Category.AREA.value,
        "type": AreaType.CAMPUS.value,
        "description": "Campus zone",
        "generation_params": {"method": "voronoi_or_subdivision", "size_hint": "large", "shape_complexity": 0.4},
        "appearance": {"style": "solid", "color": "#3F51B5"},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    AreaType.NEIGHBORHOOD.value: {
        "category": Category.AREA.value,
        "type": AreaType.NEIGHBORHOOD.value,
        "description": "Neighborhood area",
        "generation_params": {"method": "subdivision_grid", "size_hint": "large", "shape_complexity": 0.3},
        "appearance": {"style": "solid", "color": "#90CAF9"},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    }
}
