# managers/district_manager.py
from typing import Dict, Optional, List
from ..base.enums import AreaType, Category, TransFacilityType, POIType

# Display names
DISTRICT_DEFINITIONS = {
    "center_district": "Center District",
    "north_district": "North District",
    "south_district": "South District",
    "west_district": "West District",
    "east_district": "East District",
    "north_west_district": "Northwest District",
    "north_east_district": "Northeast District",
    "south_west_district": "Southwest District",
    "south_east_district": "Southeast District"
}

# Adjacency table
DISTRICT_ADJACENCY = {
    "center_district": ["north_district", "south_district", "west_district", "east_district", "north_west_district", "north_east_district", "south_west_district", "south_east_district"],
    "north_district": ["center_district", "north_west_district", "north_east_district"],
    "south_district": ["center_district", "south_west_district", "south_east_district"],
    "west_district": ["center_district", "north_west_district", "south_west_district"],
    "east_district": ["center_district", "north_east_district", "south_east_district"],
    "north_west_district": ["center_district", "north_district", "west_district"],
    "north_east_district": ["center_district", "north_district", "east_district"],
    "south_west_district": ["center_district", "south_district", "west_district"],
    "south_east_district": ["center_district", "south_district", "east_district"]
}

class DistrictManager:
    """District layout manager: computes a 3x3 grid layout from input bounds and provides queries."""

    @staticmethod
    def _calculate_layout(bounds: Dict[str, float]) -> Dict[str, Dict]:
        x_min, x_max = bounds.get('x_min', -1500.0), bounds.get('x_max', 1500.0)
        y_min, y_max = bounds.get('y_min', -1500.0), bounds.get('y_max', 1500.0)

        map_w = x_max - x_min
        map_h = y_max - y_min
        grid_w = map_w / 3.0
        grid_h = map_h / 3.0

        xs = [x_min, x_min + grid_w, x_min + 2 * grid_w, x_max]
        ys = [y_min, y_min + grid_h, y_min + 2 * grid_h, y_max]

        grid_to_id = [
            ["south_west_district", "south_district", "south_east_district"],
            ["west_district",       "center_district", "east_district"],
            ["north_west_district", "north_district",  "north_east_district"]
        ]

        layout = {}
        for j in range(3):
            for i in range(3):
                did = grid_to_id[j][i]
                layout[did] = {
                    "name": DISTRICT_DEFINITIONS[did],
                    "bounds": {
                        "x_min": xs[i], "x_max": xs[i + 1],
                        "y_min": ys[j], "y_max": ys[j + 1]
                    },
                    "index": (i, j)
                }
        return layout

    @staticmethod
    def get_all_districts_info(bounds: Dict[str, float]) -> Dict[str, Dict]:
        return DistrictManager._calculate_layout(bounds)

    @staticmethod
    def get_district_ids() -> List[str]:
        return list(DISTRICT_DEFINITIONS.keys())

    @staticmethod
    def get_district_boundary(district_id: str, bounds: Dict[str, float]) -> Optional[Dict[str, float]]:
        layout = DistrictManager._calculate_layout(bounds)
        return layout.get(district_id, {}).get("bounds")

    @staticmethod
    def get_district_for_position(x: float, y: float, bounds: Dict[str, float]) -> str:
        layout = DistrictManager._calculate_layout(bounds)
        for did, info in layout.items():
            b = info["bounds"]
            is_x_in = b["x_min"] <= x < b["x_max"]
            is_y_in = b["y_min"] <= y < b["y_max"]

            # Let the rightmost/topmost cell include its upper bound
            i, j = info["index"]
            if x == b["x_max"] and i == 2:
                is_x_in = True
            if y == b["y_max"] and j == 2:
                is_y_in = True

            if is_x_in and is_y_in:
                return did
        return "out_of_bounds"

    @staticmethod
    def get_adjacent_districts(district_id: str) -> List[str]:
        return DISTRICT_ADJACENCY.get(district_id, [])
