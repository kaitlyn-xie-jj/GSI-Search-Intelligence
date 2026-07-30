# props.py
from ..base.enums import Category, PropName, PropStatus, PropAttributeName, ColorOption, CarSubtype, BoatSubtype, PersonItem, CargoSubtype, AssemblyComponentType

PROP_TEMPLATES = {
    PropName.VEHICLE.value: {
        "category": Category.PROP.value,
        "type": PropName.VEHICLE.value,
        "attributes": {
            PropAttributeName.LICENSE_PLATE.value: {"required": True},
            PropAttributeName.COLOR.value: {
                "options": [c.value for c in ColorOption],
                "default": ColorOption.WHITE.value
            },
            PropAttributeName.SUBTYPE.value: {
                "options": [s.value for s in CarSubtype],
                "default": CarSubtype.SEDAN.value
            },
        },
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value},
        "size": {"length": 5.0, "width": 2.0, "height": 1.5, "unit": "meters"}
    },
    PropName.BOAT.value: {
        "category": Category.PROP.value,
        "type": PropName.BOAT.value,
        "attributes": {
            PropAttributeName.SUBTYPE.value: {
                "options": [s.value for s in BoatSubtype],
                "default": BoatSubtype.SPEEDBOAT.value
            },
            PropAttributeName.COLOR.value: {
                "options": [c.value for c in ColorOption],
                "default": ColorOption.WHITE.value
            }
        },
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value},
        "size": {"length": 8.0, "width": 2.5, "height": 2.5, "unit": "meters"}
    },
    PropName.CARGO.value: {
        "category": Category.PROP.value,
        "type": PropName.CARGO.value,
        "attributes": {
            PropAttributeName.WEIGHT_KG.value: {"default": 10.0, "min": 0},
            PropAttributeName.COLOR.value: {
                "options": [c.value for c in ColorOption],
                "default": ColorOption.WHITE.value
            },
            PropAttributeName.SUBTYPE.value: {
                "options": [s.value for s in CargoSubtype],
                "default": CargoSubtype.BOX.value
            },
        },
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNLOADED.value},
        "size": {"length": 2.0, "width": 2.0, "height": 1.0, "unit": "meters"}
    },
    PropName.PERSON.value: {
        "category": Category.PROP.value,
        "type": PropName.PERSON.value,
        "attributes": {
            PropAttributeName.CLOTHING_COLOR.value: {
                "options": [c.value for c in ColorOption],
                "default": ColorOption.BLACK.value
            },
            PropAttributeName.ITEM.value: {
                "options": [i.value for i in PersonItem],
                "default": PersonItem.BACKPACK.value
            }
        },
        "status": {"options": [PropStatus.DISCOVERED.value, PropStatus.UNDISCOVERED.value], "default": PropStatus.UNDISCOVERED.value},
        "size": {"length": 0.5, "width": 0.5, "height": 1.75, "unit": "meters"}
    },
    PropName.FIRE.value: {
        "category": Category.PROP.value,
        "type": PropName.FIRE.value,
        "attributes": {}, 
        "status": {
            "options": [s.value for s in PropStatus],
            "default": PropStatus.UNDISCOVERED.value
        },
        "size": {"length": 1.0, "width": 1.0, "height": 1.0, "unit": "meters"}
    },
    PropName.HAZMAT.value: {
        "category": Category.PROP.value,
        "type": PropName.HAZMAT.value,
        "attributes": {
            PropAttributeName.WEIGHT_KG.value: {"default": 5.0, "min": 0},
            PropAttributeName.COLOR.value: {
                "options": [c.value for c in ColorOption],
                "default": ColorOption.WHITE.value
            },
        },
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.UNDISCOVERED.value},
        "size": {"length": 1.0, "width": 1.0, "height": 1.0, "unit": "meters"}
    },
    PropName.EQUIPMENT_FAILURE.value: {
        "category": Category.PROP.value,
        "type": PropName.EQUIPMENT_FAILURE.value,
        "attributes": {},
        "status": {
            "options": [PropStatus.DISCOVERED.value, PropStatus.UNDISCOVERED.value, PropStatus.RESOLVED.value, PropStatus.UNRESOLVED.value],
            "default": PropStatus.UNDISCOVERED.value
        },
        "size": {"length": 1.0, "width": 1.0, "height": 0.3, "unit": "meters"}
    },
    PropName.ASSEMBLY_COMPONENT.value: {
        "category": Category.PROP.value,
        "type": PropName.ASSEMBLY_COMPONENT.value,
        "attributes": {
            PropAttributeName.SUBTYPE.value: {
                "options": [t.value for t in AssemblyComponentType],
                "default": AssemblyComponentType.FOUNDATION_BASE.value
            },
            PropAttributeName.COLOR.value: {
                "options": [c.value for c in ColorOption],
                "default": ColorOption.WHITE.value
            }
        },
        "status": {"options": [s.value for s in PropStatus], "default": PropStatus.DISCOVERED.value},
        "size": {"length": 1.2, "width": 0.8, "height": 0.3, "unit": "meters"}
    }
}
