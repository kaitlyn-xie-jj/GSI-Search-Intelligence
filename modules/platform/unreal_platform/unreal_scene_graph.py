# -*- coding: utf-8 -*-
"""
Unreal Scene Graph - Unreal Engine scene graph implementation

Communicates with Unreal Engine over HTTP and implements the AbstractSceneGraph interface.
Maintains a local scene graph cache and syncs the latest state through sync_from_unreal.
"""

import asyncio
import json
import logging
import math
import re
import uuid
from typing import Dict, List, Optional, Any, Union, Literal, Set, TYPE_CHECKING

from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.unreal_platform.data_adapters import UnrealDataAdapter

if TYPE_CHECKING:
    from modules.communication.unified_communicator import UnifiedCommunicator


class UnrealSceneGraph(AbstractSceneGraph):
    """Unreal platform scene graph implementation.
    
    Communicates with Unreal Engine over HTTP and implements the AbstractSceneGraph interface.
    Uses the singleton pattern to ensure one global instance.
    
    Attributes:
        _communicator: Unified communicator instance.
        _owns_communicator: Whether this instance owns the communicator for cleanup.
        _adapter: Data format adapter.
        _nodes: Local node cache in standard format.
        _edges: Local edge cache in standard format.
        _goal: Goal data.
    """
    
    # Singleton pattern.
    _instance: Optional["UnrealSceneGraph"] = None
    _lock = asyncio.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Implement the singleton pattern."""
        if cls._instance is None:
            cls._instance = super(UnrealSceneGraph, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        initial_goal: Optional[Dict[str, Any]] = None,
        base_url: str = "http://localhost:8080",
        server_port: int = 8081,
        timeout: float = 30.0,
        polling_interval: float = 0.5,
        shared_communicator: Optional[Any] = None,
        hitl_enabled: bool = False,
        communicator: Optional["UnifiedCommunicator"] = None
    ):
        """Initialize the Unreal scene graph.
        
        Args:
            initial_goal: Initial goal data.
            base_url: Base URL for the Unreal API, the UE5 server. Defaults to port 8080.
            server_port: Python server listen port for UE5 polling. Defaults to 8081 and must differ from the UE5 port.
            timeout: HTTP request timeout.
            polling_interval: Polling interval.
            shared_communicator: Deprecated. Use the communicator parameter.
            hitl_enabled: Whether HITL is enabled.
            communicator: Injected UnifiedCommunicator instance, the recommended approach.
        """
        # Avoid repeated initialization.
        if hasattr(self, "_initialized") and self._initialized:
            return
        
        # Use the injected communicator or create a new instance.
        if communicator is not None:
            self._communicator = communicator
            self._owns_communicator = False
        else:
            # Create an internal UnifiedCommunicator instance.
            from modules.communication.unified_communicator import UnifiedCommunicator
            self._communicator = UnifiedCommunicator(
                unreal_url=base_url,
                server_port=server_port,
                timeout=timeout,
                hitl_enabled=hitl_enabled
            )
            self._owns_communicator = True
        
        self._adapter = UnrealDataAdapter()
        
        # Local cache in standard format.
        self._nodes: List[Dict[str, Any]] = []
        self._edges: List[Dict[str, Any]] = []
        self._goal: Optional[Dict[str, Any]] = initial_goal
        
        self._cache_lock = asyncio.Lock()
        self.logger = logging.getLogger(__name__)
        self._initialized = True
        
        hitl_status = ", hitl_enabled=True" if hitl_enabled else ""
        self.logger.info(f"UnrealSceneGraph initialized with base_url={base_url}, server_port={server_port}{hitl_status}")
    
    async def sync_from_unreal(self) -> None:
        """Sync the latest state from Unreal to the local cache.
        
        Gets the current Unreal Engine world state, converts it to standard 2D format,
        and updates the local cache.
        
        Conversion flow:
        1. Get the 3D world state from UE5.
        2. Convert 3D coordinates to 2D by ignoring the z component.
        3. Convert 3D shapes such as prism to 2D shapes such as polygon.
        4. Ensure the global cybertown area node exists.
        """
        async with self._cache_lock:
            try:
                world_state = await self._communicator.get_world_state()
                
                # Convert node data from 3D to 2D.
                raw_nodes = world_state.get("nodes", world_state.get("entities", []))
                standard_nodes = self._adapter.unreal_nodes_to_standard(raw_nodes)
                
                # Ensure the global cybertown area node exists.
                self._nodes = self._adapter.ensure_cybertown_node(standard_nodes)
                
                # Convert edge data.
                raw_edges = world_state.get("edges", world_state.get("relations", []))
                self._edges = self._adapter.unreal_edges_to_standard(raw_edges)
                
                # Update goal data.
                if "goal" in world_state:
                    self._goal = world_state["goal"]
                
                self.logger.info(
                    f"Synced from Unreal: {len(self._nodes)} nodes, {len(self._edges)} edges"
                )
            except Exception as e:
                self.logger.error(f"Failed to sync from Unreal: {e}")
                raise

    # ==================== Goal Query Interface ====================
    
    def get_goal(self) -> Optional[Dict[str, Any]]:
        """Get goal data."""
        if self._goal:
            return self._goal.copy()
        return None
    
    def get_goal_description(self) -> Optional[str]:
        """Get the goal description."""
        if self._goal:
            if isinstance(self._goal, dict):
                return self._goal.get("description", self._goal.get("goal", {}).get("description"))
        return None
    
    def evaluate_goal(self) -> bool:
        """Evaluate whether the goal is achieved."""
        # Unreal is responsible for goal evaluation, so this returns False.
        # Actual evaluation should be requested from Unreal over HTTP.
        return False

    # ==================== Node Query Interface ====================
    
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes."""
        return [node.copy() for node in self._nodes]
    
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID."""
        for node in self._nodes:
            if str(node.get("id")) == str(node_id):
                return node.copy()
        return None
    
    def get_node_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """Get a node by label."""
        for node in self._nodes:
            if node.get("properties", {}).get("label") == label:
                return node.copy()
        return None
    
    def get_nodes_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get nodes by category."""
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == category
        ]
    
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get nodes by type."""
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("type") == node_type
        ]
    
    def search_nodes(self, **criteria) -> List[Dict[str, Any]]:
        """Search nodes by criteria."""
        result = []
        for node in self._nodes:
            match = True
            for key, value in criteria.items():
                if "." in key:
                    keys = key.split(".")
                    node_value = node
                    for k in keys:
                        if isinstance(node_value, dict) and k in node_value:
                            node_value = node_value[k]
                        else:
                            node_value = None
                            break
                    if node_value != value:
                        match = False
                        break
                else:
                    if node.get(key) != value:
                        match = False
                        break
            if match:
                result.append(node.copy())
        return result
    
    def get_node_info_by_label(self, label: str) -> Optional[Dict[str, str]]:
        """Get a node's category and type by label."""
        for node in self._nodes:
            properties = node.get("properties", {})
            if properties.get("label") == label:
                category = properties.get("category")
                node_type = properties.get("type")
                if category is not None and node_type is not None:
                    return {"category": category, "type": node_type}
        return None

    # ==================== Edge Query Interface ====================
    
    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Get all edges."""
        return [edge.copy() for edge in self._edges]
    
    def get_edges_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Get all edges from the specified source node."""
        return [
            edge.copy()
            for edge in self._edges
            if str(edge.get("source")) == str(source_id)
        ]
    
    def get_edges_by_target(self, target_id: str) -> List[Dict[str, Any]]:
        """Get all edges to the specified target node."""
        return [
            edge.copy()
            for edge in self._edges
            if str(edge.get("target")) == str(target_id)
        ]
    
    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get the specified edge."""
        for edge in self._edges:
            if (str(edge.get("source")) == str(source_id) and 
                str(edge.get("target")) == str(target_id)):
                return edge.copy()
        return None
    
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """Get the neighbor node ID list for a node."""
        neighbors = set()
        
        if direction in ["out", "both"]:
            for edge in self._edges:
                if str(edge.get("source")) == str(node_id):
                    neighbors.add(str(edge.get("target")))
        
        if direction in ["in", "both"]:
            for edge in self._edges:
                if str(edge.get("target")) == str(node_id):
                    neighbors.add(str(edge.get("source")))
        
        return list(neighbors)
    
    def get_neighbors_by_relation(
        self, node_id: Union[int, str], relation: str
    ) -> List[Union[int, str]]:
        """Get neighbor nodes by relation type."""
        neighbor_ids: Set[Union[int, str]] = set()
        
        for edge in self._edges:
            if edge.get("type") == relation:
                source = edge.get("source")
                target = edge.get("target")
                
                if source == str(node_id):
                    neighbor_ids.add(target)
                elif target == str(node_id):
                    neighbor_ids.add(source)
        
        return list(neighbor_ids)
    
    def get_edges_from_node(
        self, node_id: Union[int, str], edge_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get edges from the specified node."""
        edges = []
        node_id_str = str(node_id)
        
        for edge in self._edges:
            if str(edge.get("source")) == node_id_str:
                if edge_type is None or edge.get("type") == edge_type:
                    edges.append(edge.copy())
        
        return edges
    
    def get_edges_by_type(self, edge_type: str) -> List[Dict[str, Any]]:
        """Get all edges of the specified type."""
        return [edge.copy() for edge in self._edges if edge.get("type") == edge_type]
    
    def has_edge_of_type(
        self, source_id: Union[int, str], target_id: Union[int, str], edge_type: str
    ) -> bool:
        """Check whether an edge of the specified type exists between two nodes."""
        source_str = str(source_id)
        target_str = str(target_id)
        
        for edge in self._edges:
            if (str(edge.get("source")) == source_str and
                str(edge.get("target")) == target_str and
                edge.get("type") == edge_type):
                return True
        
        return False

    # ==================== Entity Query Interface ====================
    
    def get_robot(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get a robot node by ID."""
        for node in self._nodes:
            if (str(node.get("id")) == str(robot_id) and
                node.get("properties", {}).get("category") == "robot"):
                return node.copy()
        return None
    
    def get_prop(self, prop_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a prop node by ID."""
        if not prop_id:
            return [
                node.copy()
                for node in self._nodes
                if node.get("properties", {}).get("category") == "prop"
            ]
        else:
            for node in self._nodes:
                if (str(node.get("id")) == str(prop_id) and
                    node.get("properties", {}).get("category") == "prop"):
                    return node.copy()
            return None
    
    def get_building(self, building_id: str) -> Optional[Dict[str, Any]]:
        """Get a building node by ID."""
        for node in self._nodes:
            if (str(node.get("id")) == str(building_id) and
                node.get("properties", {}).get("category") == "building"):
                return node.copy()
        return None
    
    def get_all_robots(self) -> List[Dict[str, Any]]:
        """Get all robot nodes."""
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == "robot"
        ]
    
    def get_all_props(self) -> List[Dict[str, Any]]:
        """Get all prop nodes."""
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == "prop"
        ]
    
    def get_all_buildings(self) -> List[Dict[str, Any]]:
        """Get all building nodes."""
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == "building"
        ]
    
    def get_all_trans_facilitys(self) -> List[Dict[str, Any]]:
        """Get all transportation facility nodes."""
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == "trans_facility"
        ]
    
    def get_goal_node(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get a goal node by ID."""
        for node in self._nodes:
            if (str(node.get("id")) == str(goal_id) and
                node.get("properties", {}).get("category") == "goal"):
                return node.copy()
        return None

    def get_available_robots(
        self, types: Optional[List[str]] = None, concise: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Get available robots."""
        def canonicalize(t: Optional[str]) -> Optional[str]:
            if not t:
                return None
            t_norm = t.strip().lower().replace(" ", "").replace("-", "_")
            aliases = {
                "uav": "UAV",
                "fw_uav": "FW_UAV",
                "fwuav": "FW_UAV",
                "fixedwing": "FW_UAV",
                "fixed_wing": "FW_UAV",
                "ugv": "UGV",
                "quadruped": "Quadruped",
                "quadrupte": "Quadruped",
                "quad": "Quadruped",
                "humanoid": "Humanoid",
            }
            return aliases.get(t_norm)
        
        id_tail_re = re.compile(r"(\d+)$")
        
        def extract_numeric_tail(label: str) -> Optional[int]:
            m = id_tail_re.search(label.strip())
            return int(m.group(1)) if m else None
        
        include_set: Optional[Set[str]] = None
        if types is not None:
            include_set = {c for c in (canonicalize(t) for t in types) if c is not None}
        
        detailed: Dict[str, Dict[str, Any]] = {}
        
        for node in self._nodes:
            props = node.get("properties", {})
            if props.get("category") != "robot":
                continue
            if (props.get("status") == "error" or
                props.get("battery_level", 100) < 20.0 or
                props.get("comm") == "jammed"):
                continue
            
            robot_type_raw = props.get("type")
            robot_label = props.get("label")
            canonical_type = canonicalize(robot_type_raw)
            if not canonical_type or not robot_label:
                continue
            
            if include_set is not None and canonical_type not in include_set:
                continue
            
            if canonical_type not in detailed:
                detailed[canonical_type] = {"labels": [], "num": 0}
            detailed[canonical_type]["labels"].append(robot_label)
            detailed[canonical_type]["num"] += 1
        
        if not concise:
            return detailed
        
        concise_map: Dict[str, List[int]] = {}
        for rtype, info in detailed.items():
            nums: List[int] = []
            for label in info.get("labels", []):
                n = extract_numeric_tail(label)
                if n is not None:
                    nums.append(n)
            if nums:
                concise_map[rtype] = nums
        
        return concise_map
    
    def get_robot_states(self) -> Dict[str, Dict[str, Any]]:
        """Get robot state information."""
        robot_states = {}
        
        for node in self._nodes:
            properties = node.get("properties", {})
            if properties.get("category") != "robot":
                continue
            
            robot_label = properties.get("label")
            robot_type = properties.get("type")
            
            if robot_label and robot_type:
                robot_states[robot_label] = {
                    "type": robot_type,
                    "status": properties.get("status", "unknown"),
                    "location": properties.get("location", "unknown"),
                }
        
        return robot_states

    # ==================== Position Query Interface ====================
    
    def _get_node_position(self, node: Dict) -> List[float]:
        """Get a representative position for a node."""
        if not node:
            return [0.0, 0.0]
        
        shape = node.get("shape") or {}
        stype = shape.get("type")
        
        if stype == "point":
            c = shape.get("center", [0.0, 0.0])
            return [float(c[0]), float(c[1])]
        if stype == "circle":
            c = shape.get("center", [0.0, 0.0])
            return [float(c[0]), float(c[1])]
        if stype == "rectangle":
            minc = shape.get("min_corner", [0.0, 0.0])
            maxc = shape.get("max_corner", [0.0, 0.0])
            return [
                (float(minc[0]) + float(maxc[0])) / 2.0,
                (float(minc[1]) + float(maxc[1])) / 2.0,
            ]
        if stype == "polygon":
            verts = shape.get("vertices", [])
            if verts:
                sx = sy = 0.0
                n = 0
                for v in verts:
                    if v and len(v) >= 2:
                        sx += float(v[0])
                        sy += float(v[1])
                        n += 1
                if n > 0:
                    return [sx / n, sy / n]
        if stype == "linestring":
            pts = shape.get("points", [])
            if not pts:
                return [0.0, 0.0]
            if len(pts) == 1:
                p = pts[0]
                return [float(p[0]), float(p[1])]
            # Calculate the arc-length midpoint.
            seg_lens = []
            total = 0.0
            for i in range(len(pts) - 1):
                x1, y1 = float(pts[i][0]), float(pts[i][1])
                x2, y2 = float(pts[i + 1][0]), float(pts[i + 1][1])
                L = math.hypot(x2 - x1, y2 - y1)
                seg_lens.append(L)
                total += L
            if total <= 1e-9:
                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[-1][0]), float(pts[-1][1])
                return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
            
            half = total / 2.0
            acc = 0.0
            for i, L in enumerate(seg_lens):
                if acc + L >= half:
                    x1, y1 = float(pts[i][0]), float(pts[i][1])
                    x2, y2 = float(pts[i + 1][0]), float(pts[i + 1][1])
                    t = (half - acc) / (L if L > 1e-12 else 1.0)
                    return [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]
                acc += L
            pe = pts[-1]
            return [float(pe[0]), float(pe[1])]
        
        return [0.0, 0.0]
    
    def get_position_by_id(self, node_id: int) -> Optional[List[float]]:
        """Get a position by node ID."""
        node = self.get_node_by_id(str(node_id))
        if node:
            return self._get_node_position(node)
        return None
    
    def get_position_by_label(self, node_label: str) -> Optional[List[float]]:
        """Get a position by node label."""
        nodes_with_label = self.search_nodes(**{"properties.label": node_label})
        if nodes_with_label:
            return self._get_node_position(nodes_with_label[0])
        return None
    
    def get_positions_by_entity_type(
        self, entity_categories: List[str] = ["building", "robot"]
    ) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """Batch-get positions for specified entity categories."""
        results = {}
        for category in entity_categories:
            nodes_in_category = self.get_nodes_by_category(category)
            if not nodes_in_category:
                continue
            
            results[category] = {}
            
            for node in nodes_in_category:
                properties = node.get("properties", {})
                subtype = properties.get("type")
                label = properties.get("label")
                
                if category == "robot":
                    status = properties.get("status")
                    if status == "error":
                        continue
                
                position = self._get_node_position(node)
                
                if not (subtype and label and position):
                    continue
                
                if subtype not in results[category]:
                    results[category][subtype] = {}
                
                results[category][subtype][label] = position
        
        return results
    
    def get_boundary_features(
        self, district: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Extract boundary features.
        
        Unreal platform behavior:
        - All shape types return kind="area".
        - Types such as point and linestring get coordinates from vertices because
          streets and intersections have real extents in UE5.
        """
        results: Dict[str, Dict[str, Any]] = {}
        allowed = {"area", "trans_facility", "building"}
        
        for node in self._nodes:
            props = node.get("properties") or {}
            if not isinstance(props, dict):
                continue
            if props.get("category") not in allowed:
                continue
            if district is not None and props.get("district") != district:
                continue
            
            label = str(
                props.get("label") or node.get("label") or 
                node.get("id") or props.get("id")
            )
            shape = node.get("shape") or {}
            t = isinstance(shape, dict) and shape.get("type")
            if not t:
                continue
            
            feature: Dict[str, Any] = {}
            try:
                # Unreal platform: get coordinates from vertices and use kind="area".
                if t == "polygon":
                    verts = shape.get("vertices") or []
                    coords = [[float(v[0]), float(v[1])] for v in verts if v and len(v) >= 2]
                    if coords:
                        feature = {"kind": "area", "coords": coords}
                elif t == "rectangle":
                    a = shape.get("min_corner") or []
                    b = shape.get("max_corner") or []
                    x1, y1 = float(a[0]), float(a[1])
                    x2, y2 = float(b[0]), float(b[1])
                    feature = {
                        "kind": "area",
                        "coords": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    }
                elif t == "circle":
                    c = shape.get("center") or []
                    r = shape.get("radius")
                    cx, cy, rr = float(c[0]), float(c[1]), float(r)
                    feature = {"kind": "circle", "center": [cx, cy], "radius": rr}
                elif t == "point":
                    verts = shape.get("vertices") or []
                    coords = [[float(v[0]), float(v[1])] for v in verts if v and len(v) >= 2]
                    if coords:
                        feature = {"kind": "area", "coords": coords}
                    else:
                        c = shape.get("center") or []
                        if c and len(c) >= 2:
                            cx, cy = float(c[0]), float(c[1])
                            feature = {"kind": "area", "coords": [[cx, cy]]}
                elif t in ("linestring", "polyline"):
                    verts = shape.get("vertices") or []
                    coords = [[float(v[0]), float(v[1])] for v in verts if v and len(v) >= 2]
                    if coords:
                        feature = {"kind": "area", "coords": coords}
                    else:
                        pts = shape.get("points") or []
                        coords = [[float(p[0]), float(p[1])] for p in pts if p and len(p) >= 2]
                        if coords:
                            feature = {"kind": "area", "coords": coords}
            except Exception:
                feature = {}
            
            if feature:
                results[label] = feature
        
        return results

    # ==================== Mapping Interface ====================
    
    def get_node_map(
        self, map_type: Literal["id_to_label", "label_to_id"] = "id_to_label"
    ) -> Dict:
        """Generate a mapping dictionary between IDs and labels."""
        if map_type not in ["id_to_label", "label_to_id"]:
            raise ValueError("Invalid map_type. Must be 'id_to_label' or 'label_to_id'.")
        
        node_map = {}
        for node in self._nodes:
            node_id = node.get("id")
            label = node.get("properties", {}).get("label")
            
            if node_id is not None and label is not None:
                if map_type == "id_to_label":
                    node_map[node_id] = label
                else:
                    node_map[label] = node_id
        
        return node_map
    
    def get_all_node_categories(self) -> Dict[str, str]:
        """Generate a label-to-category mapping dictionary."""
        category_map: Dict[str, str] = {}
        for node in self._nodes:
            properties = node.get("properties", {})
            label = properties.get("label")
            category = properties.get("category")
            
            if label is not None and category is not None:
                category_map[label] = category
        
        return category_map

    # ==================== Path Query Interface ====================
    
    def find_path(self, src_id: str, dst_id: str) -> List[str]:
        """Find a path between two nodes using BFS."""
        if src_id == dst_id:
            return [src_id]
        
        # Build the adjacency list.
        adjacency: Dict[str, List[str]] = {}
        for edge in self._edges:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            if source not in adjacency:
                adjacency[source] = []
            adjacency[source].append(target)
            # Undirected graph.
            if target not in adjacency:
                adjacency[target] = []
            adjacency[target].append(source)
        
        # BFS
        visited = {src_id}
        queue = [(src_id, [src_id])]
        
        while queue:
            current, path = queue.pop(0)
            for neighbor in adjacency.get(current, []):
                if neighbor == dst_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []
    
    def has_path(self, src_id: str, dst_id: str) -> bool:
        """Check whether a path exists between two nodes."""
        return len(self.find_path(src_id, dst_id)) > 0
    
    def find_side_edges_near_path(
        self, src_id: str, dst_id: str
    ) -> List[Dict[str, Any]]:
        """Find side edges near a path."""
        path = self.find_path(src_id, dst_id)
        if not path:
            return []
        
        path_set = set(path)
        side_edges = []
        
        for edge in self._edges:
            source = str(edge.get("source"))
            target = str(edge.get("target"))
            # If one edge endpoint is on the path and the other is not.
            if (source in path_set and target not in path_set) or \
               (target in path_set and source not in path_set):
                side_edges.append(edge.copy())
        
        return side_edges
    
    def find_candidate_locations(
        self, object_or_carrier_id: str, prefer_parking: bool = True
    ) -> List[Dict[str, Any]]:
        """Find candidate locations."""
        candidates = []
        
        # Use all buildings as candidate locations.
        buildings = self.get_all_buildings()
        
        for building in buildings:
            props = building.get("properties", {})
            building_type = props.get("type", "")
            
            candidate = {
                "id": building.get("id"),
                "label": props.get("label"),
                "type": building_type,
                "position": self._get_node_position(building)
            }
            
            # If parking is preferred, move parking types to the front.
            if prefer_parking and "parking" in building_type.lower():
                candidates.insert(0, candidate)
            else:
                candidates.append(candidate)
        
        return candidates

    # ==================== Data Modification Interface (Sync) ====================
    
    def update_edge_target(
        self,
        source_id: Union[int, str],
        old_target_id: Union[int, str],
        new_target_id: Union[int, str],
        edge_type: Optional[str] = None,
    ) -> bool:
        """Update an edge's target node."""
        source_str = str(source_id)
        old_target_str = str(old_target_id)
        
        for i, edge in enumerate(self._edges):
            if (str(edge.get("source")) == source_str and
                str(edge.get("target")) == old_target_str):
                if edge_type is None or edge.get("type") == edge_type:
                    new_edge = edge.copy()
                    new_edge["target"] = str(new_target_id)
                    self._edges[i] = new_edge
                    return True
        
        return False
    
    def remove_edges_from_node(
        self, node_id: Union[int, str], edge_type: Optional[str] = None
    ) -> int:
        """Remove all edges from the specified node."""
        node_id_str = str(node_id)
        edges_to_remove = []
        
        for edge in self._edges:
            if str(edge.get("source")) == node_id_str:
                if edge_type is None or edge.get("type") == edge_type:
                    edges_to_remove.append(edge)
        
        for edge in edges_to_remove:
            self._edges.remove(edge)
        
        return len(edges_to_remove)

    # ==================== Data Modification Interface (Async) ====================
    
    async def add_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Add a node asynchronously as a local cache operation."""
        async with self._cache_lock:
            if "id" not in node_data:
                node_data["id"] = str(uuid.uuid4())
            
            node_id = str(node_data["id"])
            
            # Check whether the ID already exists.
            if self.get_node_by_id(node_id):
                self.logger.warning(f"Node ID {node_id} already exists")
                return False
            
            self._nodes.append(node_data.copy())
            self.logger.info(f"Added node: {node_id}")
            return True
    
    async def update_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Update a node asynchronously as a local cache operation."""
        async with self._cache_lock:
            node_id = node_data.get("id")
            if not node_id:
                self.logger.warning("Missing ID when updating node")
                return False
            
            node_id = str(node_id)
            
            for i, node in enumerate(self._nodes):
                if str(node.get("id")) == node_id:
                    updated_node = node.copy()
                    self._deep_update(updated_node, node_data)
                    self._nodes[i] = updated_node
                    self.logger.info(f"Updated node: {node_id}")
                    return True
            
            self.logger.warning(f"Node not found for update: {node_id}")
            return False
    
    async def remove_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Remove a node asynchronously as a local cache operation."""
        async with self._cache_lock:
            node_id = node_data.get("id")
            if not node_id:
                self.logger.warning("Missing ID when deleting node")
                return False
            
            node_id = str(node_id)
            
            for i, node in enumerate(self._nodes):
                if str(node.get("id")) == node_id:
                    self._nodes.pop(i)
                    # Remove related edges.
                    self._edges = [
                        edge for edge in self._edges
                        if (str(edge.get("source")) != node_id and
                            str(edge.get("target")) != node_id)
                    ]
                    self.logger.info(f"Removed node: {node_id}")
                    return True
            
            self.logger.warning(f"Node not found for removal: {node_id}")
            return False
    
    async def add_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Add an edge asynchronously as a local cache operation."""
        async with self._cache_lock:
            source_id = edge_data.get("source")
            target_id = edge_data.get("target")
            
            if not source_id or not target_id:
                self.logger.warning("Missing source or target when adding edge")
                return False
            
            # Check whether the edge already exists.
            if self.get_edge(str(source_id), str(target_id)):
                self.logger.warning(f"Edge {source_id} -> {target_id} already exists")
                return False
            
            self._edges.append(edge_data.copy())
            self.logger.info(f"Added edge: {source_id} -> {target_id}")
            return True
    
    async def update_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Update an edge asynchronously as a local cache operation."""
        async with self._cache_lock:
            source_id = edge_data.get("source")
            new_target_id = edge_data.get("target")
            edge_type = edge_data.get("type")
            
            if not all([source_id, new_target_id, edge_type]):
                self.logger.warning("Missing required keys when updating edge")
                return False
            
            for i, edge in enumerate(self._edges):
                if (str(edge.get("source")) == str(source_id) and
                    str(edge.get("type")) == edge_type):
                    self._edges[i]["target"] = new_target_id
                    return True
            
            return False
    
    async def remove_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Remove an edge asynchronously as a local cache operation."""
        async with self._cache_lock:
            source_id = edge_data.get("source")
            target_id = edge_data.get("target")
            
            if not source_id or not target_id:
                self.logger.warning("Missing source or target when deleting edge")
                return False
            
            source_str = str(source_id)
            target_str = str(target_id)
            
            for i, edge in enumerate(self._edges):
                if (str(edge.get("source")) == source_str and
                    str(edge.get("target")) == target_str):
                    self._edges.pop(i)
                    self.logger.info(f"Removed edge: {source_id} -> {target_id}")
                    return True
            
            self.logger.warning(f"Edge not found for removal: {source_id} -> {target_id}")
            return False
    
    def _deep_update(self, target_dict: Dict[str, Any], source: Dict[str, Any]) -> bool:
        """Deep-update a dictionary."""
        updated = False
        
        for key, value in list(target_dict.items()):
            if key in source:
                target_dict[key] = source[key]
                updated = True
            elif isinstance(value, dict):
                if self._deep_update(value, source):
                    updated = True
        
        return updated

    # ==================== Utility Methods ====================
    
    def get_id(self) -> str:
        """Get the scene graph ID."""
        return f"unreal_scene_graph_{id(self)}"
    
    def get_scene_config(self) -> Dict[str, Any]:
        """Get the scene configuration."""
        return {"nodes": self.get_all_nodes(), "edges": self.get_all_edges()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary."""
        return {
            "id": self.get_id(),
            "nodes": self.get_all_nodes(),
            "edges": self.get_all_edges(),
            "statistics": {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "nodes_by_category": self._get_node_statistics(),
            },
        }
    
    def _get_node_statistics(self) -> Dict[str, int]:
        """Get node statistics."""
        stats = {}
        for node in self._nodes:
            category = node.get("properties", {}).get("category", "unknown")
            stats[category] = stats.get(category, 0) + 1
        return stats
    
    def to_json(self) -> str:
        """Convert to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
    
    def clear(self) -> None:
        """Clear all data."""
        self._nodes.clear()
        self._edges.clear()
        self.logger.info("Cleared all nodes and edges")
    
    def load_from_dict(self, data: Dict[str, Any]) -> bool:
        """Load data from a dictionary."""
        try:
            self._nodes = data.get("nodes", [])
            self._edges = data.get("edges", [])
            self.logger.info(
                f"Loaded data from dict: {len(self._nodes)} nodes, {len(self._edges)} edges"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to load data from dict: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Asynchronous cleanup method."""
        try:
            # Close the communicator only when this instance owns it.
            if self._owns_communicator:
                await self._communicator.close()
            self._nodes.clear()
            self._edges.clear()
            self._initialized = False
            UnrealSceneGraph._instance = None
            self.logger.info("UnrealSceneGraph cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise
    
    def __str__(self) -> str:
        return f"UnrealSceneGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
    
    def __repr__(self) -> str:
        stats = self._get_node_statistics()
        return (
            f"UnrealSceneGraph(id={self.get_id()}, "
            f"nodes={len(self._nodes)}, edges={len(self._edges)}, "
            f"categories={stats})"
        )


# ==================== Global Singleton Management ====================

_global_unreal_scene_graph: Optional[UnrealSceneGraph] = None
_unreal_context_lock = asyncio.Lock()


def get_global_unreal_scene_graph() -> Optional[UnrealSceneGraph]:
    """Get the global UnrealSceneGraph instance."""
    global _global_unreal_scene_graph
    return _global_unreal_scene_graph


async def initialize_global_unreal_scene_graph(
    base_url: str = "http://localhost:8080",
    server_port: int = 8081,
    timeout: float = 30.0,
    polling_interval: float = 0.5,
    communicator: Optional[Any] = None
) -> UnrealSceneGraph:
    """Initialize the global UnrealSceneGraph instance.
    
    Args:
        base_url: Base URL for the UE5 API. Defaults to port 8080.
        server_port: Python server listen port. Defaults to 8081 and must differ from the UE5 port.
        timeout: HTTP request timeout.
        polling_interval: Polling interval.
        communicator: Injected UnifiedCommunicator instance, the recommended approach.
    """
    async with _unreal_context_lock:
        global _global_unreal_scene_graph
        if _global_unreal_scene_graph is None:
            _global_unreal_scene_graph = UnrealSceneGraph(
                base_url=base_url,
                server_port=server_port,
                timeout=timeout,
                polling_interval=polling_interval,
                communicator=communicator
            )
            await _global_unreal_scene_graph.sync_from_unreal()
        return _global_unreal_scene_graph


async def cleanup_global_unreal_scene_graph():
    """Clean up the global UnrealSceneGraph instance."""
    async with _unreal_context_lock:
        global _global_unreal_scene_graph
        if _global_unreal_scene_graph:
            await _global_unreal_scene_graph.cleanup()
            _global_unreal_scene_graph = None


def reset_global_unreal_scene_graph():
    """Reset the global UnrealSceneGraph instance."""
    global _global_unreal_scene_graph
    UnrealSceneGraph._instance = None
    _global_unreal_scene_graph = None
