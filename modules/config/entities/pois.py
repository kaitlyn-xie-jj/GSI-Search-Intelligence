# pois.py
from ..base.enums import POIType, Category, PropStatus

POI_TEMPLATES = {
    POIType.CHARGING_STATION.value: {
        "category": Category.POI.value,
        "type": POIType.CHARGING_STATION.value,
        "attributes": {},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    POIType.BUS_STOP.value: {
        "category": Category.POI.value,
        "type": POIType.BUS_STOP.value,
        "attributes": {},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    POIType.ENTRANCE.value: {
        "category": Category.POI.value,
        "type": POIType.ENTRANCE.value,
        "attributes": {},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    POIType.EXIT.value: {
        "category": Category.POI.value,
        "type": POIType.EXIT.value,
        "attributes": {},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    }
}
