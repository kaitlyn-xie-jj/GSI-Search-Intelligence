# -*- coding: utf-8 -*-
"""
Data Types - standard data type definitions

Defines standard data formats used by the platform abstraction layer.
Uses the semantic platform scene graph format as the standard format.
"""

from typing import TypedDict, List, Optional, Dict, Any, Union


# ==================== Shape Type Definitions ====================

class ShapePoint(TypedDict):
    """Point shape."""
    type: str  # "point"
    center: List[float]  # [x, y]


class ShapeCircle(TypedDict):
    """Circle shape."""
    type: str  # "circle"
    center: List[float]  # [x, y]
    radius: float


class ShapeRectangle(TypedDict):
    """Rectangle shape."""
    type: str  # "rectangle"
    min_corner: List[float]  # [x, y]
    max_corner: List[float]  # [x, y]


class ShapePolygon(TypedDict):
    """Polygon shape."""
    type: str  # "polygon"
    vertices: List[List[float]]  # [[x1, y1], [x2, y2], ...]


class ShapeLinestring(TypedDict):
    """Linestring shape."""
    type: str  # "linestring"
    points: List[List[float]]  # [[x1, y1], [x2, y2], ...]


# Shape data union type.
ShapeData = Union[ShapePoint, ShapeCircle, ShapeRectangle, ShapePolygon, ShapeLinestring]


# ==================== Node Property Type Definitions ====================

class NodeProperties(TypedDict, total=False):
    """Node properties.
    
    Attributes:
        category: Node category, such as district, area, trans_facility, building, robot, or prop.
        type: Node type, such as UAV, UGV, or parking.
        label: Node label as a unique identifier.
        status: Node status.
        
        # Robot specific
        battery_level: Battery level.
        comm: Communication status.
        location: Location information.
        
        # Trans facility specific
        congestion: Congestion status.
        is_fire: Whether there is a fire.
        is_spill: Whether there is a spill.
    """
    # Common properties.
    category: str
    type: str
    label: str
    status: str
    
    # Robot specific
    battery_level: float
    comm: str
    location: Dict[str, Any]
    
    # Trans facility specific
    congestion: str
    is_fire: bool
    is_spill: bool


# ==================== Node Data Type Definitions ====================

class NodeData(TypedDict):
    """Node data.
    
    Standard node format containing id, properties, and shape.
    
    Attributes:
        id: Unique node identifier.
        properties: Node property dictionary.
        shape: Node shape data.
    """
    id: str
    properties: NodeProperties
    shape: ShapeData


# ==================== Edge Data Type Definitions ====================

class EdgeData(TypedDict, total=False):
    """Edge data.
    
    Standard edge format containing source, target, type, and optional properties.
    
    Attributes:
        source: Source node ID.
        target: Target node ID.
        type: Edge type, or relation type.
        properties: Edge property dictionary, optional.
    """
    source: str
    target: str
    type: str
    properties: Dict[str, Any]


# ==================== Scene Graph Data Type Definitions ====================

class SceneGraphData(TypedDict):
    """Scene graph data.
    
    Complete scene graph data structure containing node and edge lists.
    
    Attributes:
        nodes: Node list.
        edges: Edge list.
    """
    nodes: List[NodeData]
    edges: List[EdgeData]


# ==================== Goal Data Type Definitions ====================

class GoalData(TypedDict, total=False):
    """Goal data.
    
    Goal definition data structure.
    
    Attributes:
        id: Unique goal identifier.
        description: Goal description.
        goal_type: Goal type.
        conditions: Goal condition list.
    """
    id: str
    description: str
    goal_type: str
    conditions: List[Dict[str, Any]]


# ==================== Node Category Constants ====================

class NodeCategory:
    """Node category constants."""
    DISTRICT = "district"
    AREA = "area"
    TRANS_FACILITY = "trans_facility"
    BUILDING = "building"
    ROBOT = "robot"
    PROP = "prop"


# ==================== Robot Type Constants ====================

class RobotType:
    """Robot type constants."""
    UAV = "UAV"
    UGV = "UGV"
    QUADRUPED = "Quadruped"
    HUMANOID = "Humanoid"
    FW_UAV = "FW_UAV"


# ==================== Shape Type Constants ====================

class ShapeType:
    """Shape type constants."""
    POINT = "point"
    CIRCLE = "circle"
    RECTANGLE = "rectangle"
    POLYGON = "polygon"
    LINESTRING = "linestring"


# ==================== Transportation Facility Type Constants ====================

class TransFacilityType:
    """Transportation facility type constants."""
    INTERSECTION = "intersection"
    STREET_SEGMENT = "street_segment"
    BRIDGE = "bridge"
