"""Configuration module (unified entry point).

Provides standardized template aggregation and lightweight read interfaces
for all entities (building/robot/prop/area/trans_facility/poi) used by
SceneGraphBuilder and other components.
"""

from typing import Dict, Any, Optional, List

# ===== Enums =====
from .base.enums import (
    RobotName, RobotStatus, PropStatus, PropName, SkillName, NewSituationType,
    RobotAttributeName, PropAttributeName, BuildingName, 
    Category, EdgeType,
    TransFacilityType, POIType,
)

# ===== Raw templates (pure data from each sub-module) =====
from .entities import (
    BUILDING_TEMPLATES,
    ROBOT_TEMPLATES,
    PROP_TEMPLATES,
    AREA_TEMPLATES,
    TRANS_Facility_TEMPLATES,
    POI_TEMPLATES,
)

# ===== Managers that contain algorithms (non-pure-data) =====
from .managers import DistrictManager, UnifiedTemplateManager

# --------------------------------------------------------------------------- #
#                           Unified Template Aggregator
# --------------------------------------------------------------------------- #

# Unified aggregation: all template dicts grouped by category
UNIFIED_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "building":       BUILDING_TEMPLATES,
    "robot":          ROBOT_TEMPLATES,
    "prop":           PROP_TEMPLATES,
    "area":           AREA_TEMPLATES,
    "trans_facility": TRANS_Facility_TEMPLATES,
    "poi":            POI_TEMPLATES,
}


# Factory: for direct use by SceneGraphBuilder
def make_unified_template_manager() -> UnifiedTemplateManager:
    # Deep-copy UNIFIED_TEMPLATES here if needed to prevent external mutation
    return UnifiedTemplateManager(libs=UNIFIED_TEMPLATES)


# Default instance (use this directly in most scenarios)
unified_template_manager = make_unified_template_manager()

district_manager = DistrictManager


# --------------------------------------------------------------------------- #
#                                   __all__
# --------------------------------------------------------------------------- #

__all__ = [
    # Enums
    "RobotName", "RobotStatus", "PropStatus", "PropName", "SkillName", "NewSituationType",
    "RobotAttributeName", "PropAttributeName", "BuildingName",
    "Category", "EdgeType",
    "TransFacilityType", "POIType",

    # Unified templates
    "UNIFIED_TEMPLATES",
    "UnifiedTemplateManager",
    "make_unified_template_manager",
    "unified_template_manager",

    # Algorithm managers
    "district_manager",
]
