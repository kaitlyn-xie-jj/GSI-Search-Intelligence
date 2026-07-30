# trans_facilitys.py
from ..base.enums import TransFacilityType, Category, PropStatus

TRANS_Facility_TEMPLATES = {
    TransFacilityType.INTERSECTION.value: {
        "category": Category.TRANS_Facility.value,
        "type": TransFacilityType.INTERSECTION.value,
        "attributes": {},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    TransFacilityType.STREET_SEGMENT.value: {
        "category": Category.TRANS_Facility.value,
        "type": TransFacilityType.STREET_SEGMENT.value,
        "attributes": {},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    TransFacilityType.SIDEWALK.value: {
        "category": Category.TRANS_Facility.value,
        "type": TransFacilityType.SIDEWALK.value,
        "attributes": {"width": 3.0},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    },
    TransFacilityType.BRIDGE.value: {
        "category": Category.TRANS_Facility.value,
        "type": TransFacilityType.BRIDGE.value,
        "attributes": {"width": 12.0},
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value}
    }
}
