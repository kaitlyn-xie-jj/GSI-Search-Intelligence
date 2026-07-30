# -*- coding: utf-8 -*-
"""
Unreal Data Adapters - data format adapters

Converts Unreal Engine data formats to standard formats and internal formats to Unreal API formats.
The standard format follows the semantic platform scene graph format.

Core responsibilities:
1. 3D -> 2D conversion: convert 3D coordinates from UE5 to 2D by ignoring the z component.
2. 2D -> 3D conversion: convert 2D coordinates sent to UE5 to 3D by adding a z component, defaulting to 0.
3. Unit conversion: UE5 uses centimeters (cm), while internal data uses meters (m).
   - When reading from UE5: cm -> m by dividing by 100.
   - When sending to UE5: m -> cm by multiplying by 100.
4. Shape type conversion: convert UE5-specific shape types, such as prism, to standard 2D types, such as polygon.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import json


# Unit conversion factors.
CM_TO_M = 0.01  # Centimeters to meters.
M_TO_CM = 100.0  # Meters to centimeters.

# Default global area boundary from platform_factory.
from modules.platform.platform_factory import UNREAL_DEFAULT_BOUNDARY
DEFAULT_GLOBAL_BOUNDARY = UNREAL_DEFAULT_BOUNDARY

# Skill name normalization mapping: UE5 name -> internal standard name.
# Resolves naming differences between UE5 and Python.
SKILL_NAME_MAPPING = {
    "takeoff": "take_off",
    "take_off": "take_off",
    "land": "land",
    "search": "search",
    "follow": "follow",
    "patrol": "patrol",
    "place": "place",
}


class UnrealDataAdapter:
    """Unreal data format adapter.
    
    Converts between Unreal format and the standard format.
    The standard format follows the semantic platform scene graph format.
    
    Key conversions:
    - 3D coordinates -> 2D coordinates when reading from UE5.
    - 2D coordinates -> 3D coordinates when sending to UE5.
    - prism -> polygon for building shapes.
    - linestring stays unchanged, but coordinates are converted to 2D.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    # =========================================================================
    # Coordinate Conversion Utilities
    # =========================================================================
    
    @staticmethod
    def coord_3d_to_2d(coord: List[float]) -> List[float]:
        """Convert 3D coordinates to 2D, ignoring z and converting cm to m.
        
        Args:
            coord: 3D coordinates [x, y, z] or 2D coordinates [x, y], in centimeters.
            
        Returns:
            2D coordinates [x, y], in meters.
        """
        if not coord or len(coord) < 2:
            return [0.0, 0.0]
        return [float(coord[0]) * CM_TO_M, float(coord[1]) * CM_TO_M]
    
    @staticmethod
    def coord_2d_to_3d(coord: List[float], z: float = 0.0) -> List[float]:
        """Convert 2D coordinates to 3D by adding z and converting m to cm.
        
        Args:
            coord: 2D coordinates [x, y], in meters.
            z: z component value, in meters. Defaults to 0.
            
        Returns:
            3D coordinates [x, y, z], in centimeters.
        """
        if not coord or len(coord) < 2:
            return [0.0, 0.0, z * M_TO_CM]
        return [float(coord[0]) * M_TO_CM, float(coord[1]) * M_TO_CM, z * M_TO_CM]
    
    @staticmethod
    def coords_3d_to_2d(coords: List[List[float]]) -> List[List[float]]:
        """Convert a list of 3D coordinates to 2D, converting cm to m.
        
        Args:
            coords: 3D coordinate list [[x1, y1, z1], [x2, y2, z2], ...], in centimeters.
            
        Returns:
            2D coordinate list [[x1, y1], [x2, y2], ...], in meters.
        """
        return [UnrealDataAdapter.coord_3d_to_2d(c) for c in coords if c]
    
    @staticmethod
    def coords_2d_to_3d(coords: List[List[float]], z: float = 0.0) -> List[List[float]]:
        """Convert a list of 2D coordinates to 3D, converting m to cm.
        
        Args:
            coords: 2D coordinate list [[x1, y1], [x2, y2], ...], in meters.
            z: z component value, in meters. Defaults to 0.
            
        Returns:
            3D coordinate list [[x1, y1, z], [x2, y2, z], ...], in centimeters.
        """
        return [UnrealDataAdapter.coord_2d_to_3d(c, z) for c in coords if c]
    
    @staticmethod
    def value_cm_to_m(value: float) -> float:
        """Convert a single value from centimeters to meters."""
        return float(value) * CM_TO_M
    
    @staticmethod
    def value_m_to_cm(value: float) -> float:
        """Convert a single value from meters to centimeters."""
        return float(value) * M_TO_CM
    
    # =========================================================================
    # Shape Conversion Methods
    # =========================================================================
    
    @staticmethod
    def convert_shape_to_2d(shape: Dict[str, Any]) -> Dict[str, Any]:
        """Convert UE5 shape data to a standard 2D shape.
        
        Supported shape types:
        - prism -> polygon by using base vertices and ignoring height.
        - point -> point with center converted to 2D.
        - linestring -> linestring with points and vertices converted to 2D.
        - polygon -> polygon with vertices converted to 2D.
        - rectangle -> rectangle with corners converted to 2D.
        - circle -> circle with center converted to 2D.
        
        Args:
            shape: UE5 shape data.
            
        Returns:
            Standard 2D shape data.
        """
        if not shape or not isinstance(shape, dict):
            return {"type": "point", "center": [0.0, 0.0]}
        
        shape_type = shape.get("type", "").lower()
        
        # prism -> polygon.
        # Buildings in UE5 are prism shapes with base vertices and height.
        if shape_type == "prism":
            vertices = shape.get("vertices", [])
            return {
                "type": "polygon",
                "vertices": UnrealDataAdapter.coords_3d_to_2d(vertices)
            }
        
        # point.
        if shape_type == "point":
            center = shape.get("center", [0.0, 0.0, 0.0])
            result = {
                "type": "point",
                "center": UnrealDataAdapter.coord_3d_to_2d(center)
            }
            # Convert vertices too if present, such as intersection area vertices.
            if "vertices" in shape:
                result["vertices"] = UnrealDataAdapter.coords_3d_to_2d(shape["vertices"])
            return result
        
        # linestring for street-like objects.
        if shape_type == "linestring":
            result = {"type": "linestring"}
            if "points" in shape:
                result["points"] = UnrealDataAdapter.coords_3d_to_2d(shape["points"])
            if "vertices" in shape:
                result["vertices"] = UnrealDataAdapter.coords_3d_to_2d(shape["vertices"])
            return result
        
        # polygon.
        if shape_type == "polygon":
            return {
                "type": "polygon",
                "vertices": UnrealDataAdapter.coords_3d_to_2d(shape.get("vertices", []))
            }
        
        # rectangle.
        if shape_type == "rectangle":
            return {
                "type": "rectangle",
                "min_corner": UnrealDataAdapter.coord_3d_to_2d(shape.get("min_corner", [0, 0])),
                "max_corner": UnrealDataAdapter.coord_3d_to_2d(shape.get("max_corner", [0, 0]))
            }
        
        # circle.
        if shape_type == "circle":
            return {
                "type": "circle",
                "center": UnrealDataAdapter.coord_3d_to_2d(shape.get("center", [0, 0])),
                "radius": UnrealDataAdapter.value_cm_to_m(shape.get("radius", 0))
            }
        
        # Unknown type, try generic conversion.
        result = {"type": shape_type}
        if "center" in shape:
            result["center"] = UnrealDataAdapter.coord_3d_to_2d(shape["center"])
        if "vertices" in shape:
            result["vertices"] = UnrealDataAdapter.coords_3d_to_2d(shape["vertices"])
        if "points" in shape:
            result["points"] = UnrealDataAdapter.coords_3d_to_2d(shape["points"])
        
        return result if len(result) > 1 else {"type": "point", "center": [0.0, 0.0]}
    
    # =========================================================================
    # Node Conversion Methods
    # =========================================================================
    
    @staticmethod
    def unreal_nodes_to_standard(unreal_nodes: List[Dict]) -> List[Dict[str, Any]]:
        """Convert Unreal node format to standard format.
        
        Main conversions:
        1. Convert shape data from 3D to 2D.
        2. Convert prism type to polygon type.
        3. Ignore the z component for all coordinates.
        
        Args:
            unreal_nodes: Unreal-format node list.
            
        Returns:
            Standard-format node list with 2D coordinates.
        """
        standard_nodes = []
        
        for node in unreal_nodes:
            # Copy the basic node structure.
            standard_node = {
                "id": node.get("id"),
                "properties": node.get("properties", {}).copy() if node.get("properties") else {}
            }
            
            # Convert shape data to 2D.
            if "shape" in node:
                standard_node["shape"] = UnrealDataAdapter.convert_shape_to_2d(node["shape"])
            else:
                # Try extracting position information from other fields.
                standard_node["shape"] = UnrealDataAdapter._extract_shape_from_node(node)
            
            standard_nodes.append(standard_node)
        
        return standard_nodes
    
    @staticmethod
    def _extract_shape_from_node(node: Dict) -> Dict[str, Any]:
        """Extract shape information from node data as a fallback.
        
        Args:
            node: Node data.
            
        Returns:
            Standard-format shape data.
        """
        # Extract from the position field.
        if "position" in node:
            pos = node["position"]
            if isinstance(pos, dict):
                x = pos.get("x", pos.get("X", 0))
                y = pos.get("y", pos.get("Y", 0))
                return {"type": "point", "center": [float(x), float(y)]}
            elif isinstance(pos, (list, tuple)) and len(pos) >= 2:
                return {"type": "point", "center": [float(pos[0]), float(pos[1])]}
        
        # Extract from the location field.
        if "location" in node:
            loc = node["location"]
            if isinstance(loc, dict) and ("x" in loc or "X" in loc):
                x = loc.get("x", loc.get("X", 0))
                y = loc.get("y", loc.get("Y", 0))
                return {"type": "point", "center": [float(x), float(y)]}
            elif isinstance(loc, (list, tuple)) and len(loc) >= 2:
                return {"type": "point", "center": [float(loc[0]), float(loc[1])]}
        
        # Default to the origin.
        return {"type": "point", "center": [0.0, 0.0]}
    
    @staticmethod
    def ensure_cybertown_node(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Ensure the node list contains the global cybertown area node.
        
        Adds a default global area node if no node has label cybertown.
        
        Args:
            nodes: Node list.
            
        Returns:
            Node list containing the cybertown node.
        """
        # Check whether the cybertown node already exists.
        for node in nodes:
            props = node.get("properties", {})
            if props.get("label", "").lower() == "cybertown":
                return nodes
        
        # Add the default cybertown node if it does not exist.
        cybertown_node = {
            "id": "-1",
            "properties": {
                "category": "area",
                "type": "area",
                "label": "cybertown",
                "status": "active"
            },
            "shape": {
                "type": "polygon",
                "vertices": DEFAULT_GLOBAL_BOUNDARY.copy()
            }
        }
        
        return nodes + [cybertown_node]
    
    # =========================================================================
    # Edge Conversion Methods
    # =========================================================================
    
    @staticmethod
    def unreal_edges_to_standard(unreal_edges: List[Dict]) -> List[Dict[str, Any]]:
        """Convert Unreal edge format to standard format.
        
        Handles different field names that Unreal may use and converts them to standard format.
        
        Args:
            unreal_edges: Unreal-format edge list.
            
        Returns:
            Standard-format edge list.
        """
        standard_edges = []
        
        for edge in unreal_edges:
            # Extract source, supporting multiple field names.
            source = str(edge.get("source", edge.get("from_id", edge.get("from", edge.get("source_id", "")))))
            
            # Extract target, supporting multiple field names.
            target = str(edge.get("target", edge.get("to_id", edge.get("to", edge.get("target_id", "")))))
            
            # Extract type, supporting multiple field names.
            edge_type = edge.get("type", edge.get("relation", edge.get("relation_type", edge.get("edge_type", "unknown"))))
            
            standard_edge: Dict[str, Any] = {
                "source": source,
                "target": target,
                "type": edge_type,
            }
            
            # Copy other attributes to properties.
            excluded_keys = {
                "source", "from_id", "from", "source_id",
                "target", "to_id", "to", "target_id",
                "type", "relation", "relation_type", "edge_type"
            }
            
            props = {}
            for key, value in edge.items():
                if key not in excluded_keys:
                    props[key] = value
            
            if props:
                standard_edge["properties"] = props
            
            standard_edges.append(standard_edge)
        
        return standard_edges
    
    # =========================================================================
    # Skill Conversion Methods (2D -> 3D)
    # =========================================================================
    
    @staticmethod
    def skills_to_unreal_format(
        skills_by_timestep: Dict[int, Dict],
        default_z: float = 0.0
    ) -> Dict[int, Dict[str, Dict[str, Any]]]:
        """Convert internal skill format to Unreal API format.
        
        Keeps the dictionary organized by timestep, adds z to coordinate parameters,
        and converts units from m to cm.
        
        Args:
            skills_by_timestep: Skill plan organized by timestep.
                Format: {timestep: {robot_label: {skill: str, params: dict}}}
            default_z: Default z coordinate value, in meters.
                
        Returns:
            Converted skill dictionary with the same format as input, but coordinates
            are converted to 3D and units are centimeters.
            Format: {timestep: {robot_label: {skill: str, params: dict}}}
        """
        result: Dict[int, Dict[str, Dict[str, Any]]] = {}
        
        for timestep, robots_skills in skills_by_timestep.items():
            result[timestep] = {}
            
            for robot_label, skill_info in robots_skills.items():
                # Copy skill information.
                converted_skill: Dict[str, Any] = {
                    "skill": skill_info.get("skill", skill_info.get("action", ""))
                }
                
                # Handle params.
                params = skill_info.get("params", skill_info.get("parameters", {})).copy()
                
                # Convert the dest parameter to 3D coordinates in cm.
                if "dest" in params:
                    dest = params["dest"]
                    if isinstance(dest, dict):
                        params["dest"] = {
                            "x": float(dest.get("x", 0)) * M_TO_CM,
                            "y": float(dest.get("y", 0)) * M_TO_CM,
                            "z": float(dest.get("z", default_z)) * M_TO_CM
                        }
                    elif isinstance(dest, (list, tuple)) and len(dest) >= 2:
                        params["dest"] = {
                            "x": float(dest[0]) * M_TO_CM,
                            "y": float(dest[1]) * M_TO_CM,
                            "z": (float(dest[2]) if len(dest) > 2 else default_z) * M_TO_CM
                        }
                
                # Convert coordinates in area to 3D in cm.
                if "area" in params and isinstance(params["area"], dict):
                    area = params["area"].copy()
                    if "coords" in area:
                        area["coords"] = UnrealDataAdapter.coords_2d_to_3d(area["coords"], default_z)
                    if "center" in area:
                        area["center"] = UnrealDataAdapter.coord_2d_to_3d(area["center"], default_z)
                    if "radius" in area:
                        area["radius"] = UnrealDataAdapter.value_m_to_cm(area["radius"])
                    params["area"] = area
                
                converted_skill["params"] = params
                result[timestep][robot_label] = converted_skill
        
        return result
    
    # =========================================================================
    # Feedback Conversion Methods
    # =========================================================================
    
    @staticmethod
    def feedback_to_internal_format(unreal_feedback: Dict) -> Dict[str, Any]:
        """Convert Unreal feedback to internal outcome format.
        
        Converts feedback data returned by the Unreal API to the internal format.
        
        Args:
            unreal_feedback: Feedback data returned by the Unreal API.
            
        Returns:
            Internal-format execution result.
        """
        # Extract status.
        status = unreal_feedback.get("status", "unknown")
        success = unreal_feedback.get("success", status == "completed")
        
        # Convert result list.
        outcomes = []
        results = unreal_feedback.get("results", unreal_feedback.get("outcomes", []))
        
        for item in results:
            outcome = UnrealDataAdapter.single_feedback_to_outcome(item)
            if outcome:
                outcomes.append(outcome)
        
        return {
            "status": status,
            "success": success,
            "outcomes": outcomes,
            "execution_id": unreal_feedback.get("execution_id", ""),
            "timestamp": unreal_feedback.get("timestamp", ""),
        }
    
    @staticmethod
    def extract_discovered_nodes(feedback_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract the discovered_nodes array from feedback data.
        
        Supports the new format with discovered_nodes and the old format with indexed prefixes.
        
        Args:
            feedback_data: Feedback data field.
            
        Returns:
            Discovered node list in scene graph node format.
        """
        if not feedback_data or not isinstance(feedback_data, dict):
            return []
        
        # Return the discovered_nodes array directly.
        if "discovered_nodes" in feedback_data:
            discovered_nodes = feedback_data.get("discovered_nodes", [])
            # Parse JSON first if it is a string.
            if isinstance(discovered_nodes, str):
                try:
                    discovered_nodes = json.loads(discovered_nodes)
                except json.JSONDecodeError:
                    discovered_nodes = []
            return discovered_nodes if isinstance(discovered_nodes, list) else []
        
        # Extract from indexed prefixes.
        # Example: object_0_id, object_0_type, object_1_id, object_1_type...
        discovered = []
        idx = 0
        while True:
            prefix = f"object_{idx}_"
            # Check whether an object exists at this index.
            has_object = any(k.startswith(prefix) for k in feedback_data.keys())
            if not has_object:
                break
            
            # Extract this object's attributes.
            obj_id = feedback_data.get(f"{prefix}id")
            obj_type = feedback_data.get(f"{prefix}type")
            obj_label = feedback_data.get(f"{prefix}label")
            obj_category = feedback_data.get(f"{prefix}category", "prop")
            
            # Extract position information.
            center = None
            if f"{prefix}position" in feedback_data:
                pos = feedback_data[f"{prefix}position"]
                if isinstance(pos, dict):
                    center = [pos.get("x", 0), pos.get("y", 0), pos.get("z", 0)]
                elif isinstance(pos, (list, tuple)):
                    center = list(pos)
            elif f"{prefix}x" in feedback_data:
                center = [
                    feedback_data.get(f"{prefix}x", 0),
                    feedback_data.get(f"{prefix}y", 0),
                    feedback_data.get(f"{prefix}z", 0)
                ]
            
            # Build the node dictionary.
            node = {
                "id": str(obj_id) if obj_id is not None else str(idx),
                "properties": {
                    "category": obj_category,
                    "type": obj_type or "unknown",
                    "label": obj_label or f"object_{idx}",
                }
            }
            
            # Add other attributes.
            for key, value in feedback_data.items():
                if key.startswith(prefix) and key not in [
                    f"{prefix}id", f"{prefix}type", f"{prefix}label", 
                    f"{prefix}category", f"{prefix}position",
                    f"{prefix}x", f"{prefix}y", f"{prefix}z"
                ]:
                    prop_name = key[len(prefix):]
                    node["properties"][prop_name] = value
            
            # Add shape information.
            if center:
                node["shape"] = {
                    "type": "point",
                    "center": center
                }
            
            discovered.append(node)
            idx += 1
        
        return discovered
    
    @staticmethod
    def convert_node_to_standard(node: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a UE5 node to standard 2D format.
        
        Performs these conversions:
        - 3D coordinates to 2D by ignoring z.
        - Centimeters to meters (cm -> m).
        
        Args:
            node: UE5-format scene graph node.
            
        Returns:
            Standard 2D node.
        """
        if not node or not isinstance(node, dict):
            return {"id": "", "properties": {}, "shape": {"type": "point", "center": [0.0, 0.0]}}
        
        standard_node = {
            "id": node.get("id", ""),
            "properties": node.get("properties", {}).copy() if node.get("properties") else {}
        }
        
        # Convert shape data to 2D.
        if "shape" in node:
            standard_node["shape"] = UnrealDataAdapter.convert_shape_to_2d(node["shape"])
        else:
            standard_node["shape"] = {"type": "point", "center": [0.0, 0.0]}
        
        return standard_node
    
    @staticmethod
    def normalize_skill_name(skill: str) -> str:
        """Normalize a skill name.
        
        Converts UE5 skill names to internal standard names to resolve naming differences.
        Example: takeoff -> take_off
        
        Args:
            skill: Raw skill name.
            
        Returns:
            Normalized skill name.
        """
        if not skill:
            return ""
        normalized = skill.lower().strip()
        return SKILL_NAME_MAPPING.get(normalized, normalized)
    
    @staticmethod
    def single_feedback_to_outcome(feedback: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Convert a single UE5 feedback item to internal outcome format.
        
        Supports the new format with discovered_nodes and the old format with indexed prefixes.
        The new format places the node array in outcome.data.entities.
        
        Args:
            feedback: Single UE5 skill execution feedback with format:
                {agent_id, skill, success, message, data}
                
        Returns:
            Internal outcome format, or None if input is invalid.
        """
        if not feedback:
            return None
        
        raw_skill = feedback.get("skill", feedback.get("action", "")).lower().strip()
        # Normalize skill name to resolve naming differences between UE5 and Python.
        skill = UnrealDataAdapter.normalize_skill_name(raw_skill)
        success = feedback.get("success", feedback.get("succeeded", False))
        feedback_data = feedback.get("data", feedback.get("meta", feedback.get("metadata", {})))
        
        # Determine outcome type.
        outcome_type = "KNOWLEDGE_ACQUIRED"  # Default type.
        
        # Build the base outcome.
        outcome = {
            "type": outcome_type,
            "data": {},
            "meta": {
                "skill": skill,
                "success": success,
                "robot_label": feedback.get("agent_id", feedback.get("robot", feedback.get("robot_label", ""))),
                "task_id": feedback.get("data", {}).get("task_id", "")
            }
        }
        
        # Handle feedback data.
        if isinstance(feedback_data, dict):
            # Extract discovered nodes.
            discovered_nodes = UnrealDataAdapter.extract_discovered_nodes(feedback_data)
            
            # Convert nodes to standard 2D format.
            standard_entities = [
                UnrealDataAdapter.convert_node_to_standard(node) 
                for node in discovered_nodes
            ]
            
            # Build the data field.
            outcome_data = {
                "message": feedback.get("message", feedback.get("feedback", feedback.get("description", ""))),
            }
            
            # Add entity list if discovered nodes exist.
            if standard_entities:
                outcome_data["entities"] = standard_entities
                # If only one object is found, also set object_id for GoalProgressMonitor.
                if len(standard_entities) == 1:
                    try:
                        outcome_data["object_id"] = int(standard_entities[0].get("id", 0))
                    except (ValueError, TypeError):
                        outcome_data["object_id"] = standard_entities[0].get("id")
            
            # Copy fields required by GoalProgressMonitor.
            gpm_fields = [
                "found", "area_token", "area_searched", "area", "duration_s", 
                "persist_s", "persist_ge_s", "target_spec", "object_id",
                "robot_id", "target_id", "robot_target_distance", "message_text",
                "knowledge_type", "conf_ge"
            ]
            for field in gpm_fields:
                if field in feedback_data:
                    outcome_data[field] = feedback_data[field]
            
            # Keep other fields.
            excluded_keys = set(gpm_fields) | {"discovered_nodes"}
            # Exclude indexed prefix fields.
            excluded_keys |= {k for k in feedback_data.keys() if k.startswith("object_") and "_" in k[7:]}
            
            for key, value in feedback_data.items():
                if key not in excluded_keys and key not in outcome_data:
                    outcome_data[key] = value
            
            outcome["data"] = outcome_data
        else:
            outcome["data"] = {
                "message": feedback.get("message", feedback.get("feedback", feedback.get("description", "")))
            }
        
        # Copy other fields to meta.
        excluded_keys = {
            "agent_id", "robot", "robot_label", "skill", "action",
            "success", "succeeded", "message", "feedback", "description",
            "data", "meta", "metadata"
        }
        for key, value in feedback.items():
            if key not in excluded_keys:
                outcome["meta"][key] = value
        
        return outcome
    
    # =========================================================================
    # Emergency Event Detection Methods
    # =========================================================================

    @staticmethod
    def is_emergency_feedback(feedback: Dict[str, Any]) -> bool:
        """Check whether a single feedback item is an emergency event from a precheck or runtime failure.
        
        UE5 sets is_emergency_event="true" in data when a condition check fails.
        
        Args:
            feedback: Single UE5 skill feedback.
            
        Returns:
            Whether this is an emergency event.
        """
        data = feedback.get("data", {})
        if not isinstance(data, dict):
            return False
        return str(data.get("is_emergency_event", "")).lower() == "true"

    @staticmethod
    def extract_emergency_description(feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the description data required by NewCaseEvent from emergency feedback.
        
        Args:
            feedback: UE5 emergency feedback.
            
        Returns:
            Description dict consistent with semantic platform _publish_new_case_event.
        """
        data = feedback.get("data", {})
        if not isinstance(data, dict):
            data = {}

        severity = data.get("event_severity", "warning")
        event_key = data.get("event_key", "unknown")

        # Use "condition_check" to mark that this comes from a UE5 condition check.
        phase = "condition_check"

        skill = UnrealDataAdapter.normalize_skill_name(
            feedback.get("skill", feedback.get("action", ""))
        )

        return {
            "phase": phase,
            "skill": skill,
            "task_id": data.get("task_id"),
            "reason": event_key,
            "event_kind": "incident",
            "details": {
                "event_category": data.get("event_category", ""),
                "event_type": data.get("event_type", ""),
                "event_severity": severity,
                "event_key": event_key,
                "message": feedback.get("message", ""),
            },
            "robot": {
                "id": data.get("robot_id"),
                "label": data.get("robot_label", data.get("robot_id", "")),
            },
        }

    @staticmethod
    def extract_info_events(feedback: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract info-level events from normal feedback.
        
        Args:
            feedback: Normal UE5 skill feedback.
            
        Returns:
            List of info event descriptions, possibly empty.
        """
        data = feedback.get("data", {})
        if not isinstance(data, dict):
            return []

        raw = data.get("info_events")
        if not raw:
            return []

        # info_events is a JSON array string.
        if isinstance(raw, str):
            try:
                events = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return []
        elif isinstance(raw, list):
            events = raw
        else:
            return []

        skill = UnrealDataAdapter.normalize_skill_name(
            feedback.get("skill", feedback.get("action", ""))
        )

        descriptions = []
        for evt in events:
            if not isinstance(evt, dict):
                continue
            descriptions.append({
                "phase": "postexec",
                "skill": skill,
                "task_id": data.get("task_id"),
                "reason": evt.get("type", "info"),
                "event_kind": "info",
                "details": {
                    "event_category": evt.get("category", ""),
                    "event_type": evt.get("type", ""),
                    "event_severity": evt.get("severity", "info"),
                    "message": evt.get("message", ""),
                },
                "robot": {
                    "id": data.get("robot_id"),
                    "label": data.get("robot_label", data.get("robot_id", "")),
                },
            })

        return descriptions

    # =========================================================================
    # Standard Nodes to Unreal Format (Optional, for Sending Data to Unreal)
    # =========================================================================
    
    @staticmethod
    def standard_nodes_to_unreal(
        standard_nodes: List[Dict[str, Any]],
        default_z: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Convert standard-format nodes to Unreal format.
        
        Args:
            standard_nodes: Standard-format node list.
            default_z: Default z coordinate value.
            
        Returns:
            Unreal-format node list.
        """
        unreal_nodes = []
        
        for node in standard_nodes:
            properties = node.get("properties", {})
            shape = node.get("shape", {})
            
            unreal_node = {
                "id": node.get("id", ""),
                "category": properties.get("category", "unknown"),
                "type": properties.get("type", "unknown"),
                "label": properties.get("label", ""),
                "status": properties.get("status", "unknown"),
            }
            
            # Add position information converted to 3D.
            if shape.get("type") == "point":
                center = shape.get("center", [0, 0])
                unreal_node["position"] = {
                    "x": center[0],
                    "y": center[1],
                    "z": default_z
                }
            
            # Copy other attributes.
            for key, value in properties.items():
                if key not in ("category", "type", "label", "status"):
                    unreal_node[key] = value
            
            unreal_nodes.append(unreal_node)
        
        return unreal_nodes
