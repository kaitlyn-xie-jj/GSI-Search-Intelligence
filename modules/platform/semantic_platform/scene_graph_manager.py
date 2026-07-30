#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scene graph manager - event-driven node and edge management.

Core features:
1. Provides public query APIs for fast node and edge lookup.
2. Requires all data modification operations to go through the event mechanism.
3. Supports async event handling and concurrency control.
4. Provides complete type hints and detailed comments.

Design principles:
- Query operations: direct method calls for fast access.
- Modification operations: must go through DataModificationEvent events.
- Data consistency: event mechanism keeps data synchronized.
- Concurrency safety: async locks protect key data structures.

Event support:
- DataModificationEvent: data modification events (add, update, remove).
- Supports both node and edge entity types.
- Automatically subscribes to and handles data modification events.
- Query operations use direct method calls and do not need the event mechanism.
"""

import asyncio
import json
import logging
import uuid
import math
import re
from typing import Any, Dict, List, Optional, Union, Literal, Set, Tuple

from modules.config.events import EventType, DataModificationEvent
from modules.platform.abstract_scene_graph import AbstractSceneGraph

# Import event system
from modules.events.event_bus import (
    subscribe_event,
    unsubscribe_event,
    publish_event_sync,
    publish_reply_event,
)
from modules.platform.semantic_platform.utils.scene_graph_utils import (
    find_path as _nav_find_path,
    find_side_edges_near_path as _nav_find_side_edges_near_path,
    find_candidate_locations as _nav_find_candidate_locations,
)


class SemanticSceneGraph(AbstractSceneGraph):
    """Scene graph manager with event-driven node and edge management (singleton mode).

    This class provides a lightweight scene graph manager focused on:
    1. Fast node and edge query APIs.
    2. Event-based data modification operations.
    3. Async-safe data access.
    4. Full event-driven architecture support.
    5. Singleton mode to ensure one global instance.

    All data modification operations must go through DataModificationEvent
    to ensure data consistency and preserve event-driven architecture principles.

    Attributes:
        _instance: Singleton instance.
        _nodes: Node data list.
        _edges: Edge data list.
        _data_lock: Async lock that protects concurrent access to data structures.
        _event_subscriptions: Event subscription list.
        _is_initialized: Initialization state flag.
    """

    # Singleton-related attributes
    _instance: Optional["SemanticSceneGraph"] = None
    _lock = asyncio.Lock()

    def __new__(cls, *args, **kwargs):
        """Singleton implementation.

        Ensures there is only one SemanticSceneGraph instance in the application.

        Returns:
            SemanticSceneGraph: Global singleton instance.
        """
        if cls._instance is None:
            cls._instance = super(SemanticSceneGraph, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        initial_nodes: Optional[List[Dict[str, Any]]] = None,
        initial_edges: Optional[List[Dict[str, Any]]] = None,
        initial_goal: Optional[str] = None,
    ):
        """Initialize the scene graph.

        Initializes data storage, async lock, and event subscriptions.
        Automatically subscribes to DataModificationEvent and QueryEvent.
        Avoids repeated initialization because of singleton mode.

        Args:
            initial_nodes: Initial node list.
            initial_edges: Initial edge list.
            initial_goal: Initial goal description.
            new_case: New case.
        """
        # Avoid repeated initialization
        if hasattr(self, "_is_initialized") and self._is_initialized:
            return

        # Scene graph data: core data structures
        self._nodes: List[Dict[str, Any]] = initial_nodes or []
        self._edges: List[Dict[str, Any]] = initial_edges or []
        self._goal: Optional[Any] = initial_goal if initial_goal else None

        # Error if no initial data is provided
        if not self._nodes or not self._edges:
            raise RuntimeError("Initial nodes and edges must be provided; scene graph does not support default data loading")
        # Async lock protecting data structures
        self._data_lock = asyncio.Lock()

        # Event subscription management
        self._event_subscriptions: List[str] = []
        self._is_initialized = False

        # Logging
        self.logger = logging.getLogger(f"scene_graph.{self.get_id()}")

        # Initialize event subscriptions
        self._setup_event_subscriptions()

        self.logger.info(
            f"Scene graph initialized, nodes: {len(self._nodes)}, edges: {len(self._edges)}"
        )

    def _setup_event_subscriptions(self):
        """Set up event subscriptions.

        Subscribes to DataModificationEvent so all relevant data modification requests are handled.
        Query operations are direct method calls and do not need event driving.
        """
        try:
            # Subscribe to data modification events
            data_mod_subscription_id = subscribe_event(
                event_type=EventType.DATA_MODIFICATION.value,
                handler=self._handle_data_modification_event,
                subscriber_id=f"simplified_context_{id(self)}",
                priority=1,
            )
            self._event_subscriptions.append(data_mod_subscription_id)

            self._is_initialized = True
            self.logger.info(
                f"Event subscriptions setup complete: {len(self._event_subscriptions)} subscriptions"
            )

        except Exception as e:
            self.logger.error(f"Failed to setup event subscriptions: {e}")
            raise

    async def _handle_data_modification_event(self, event: DataModificationEvent):
        """Handle data modification events.

        Args:
            event: Data modification event.
        """
        async with self._data_lock:
            try:
                success = False
                if event.entity_type == "node":
                    success = self.handle_node_event(event.operation, event.data)
                elif event.entity_type == "edge":
                    success = self.handle_edge_event(event.operation, event.data)
                else:
                    self.logger.warning(f"Unknown entity type: {event.entity_type}")

                # # Send reply event
                # if event.request_id:
                #     await publish_reply_event(
                #         request_id=event.request_id,
                #         reply_to="scene_graph",
                #         success=success,
                #         result={"operation": event.operation, "entity_type": event.entity_type,
                #                 "entity_id": event.entity_id}
                #     )

            except Exception as e:
                self.logger.error(f"Error handling data modification event: {e}")
                if event.request_id:
                    await publish_reply_event(
                        request_id=event.request_id,
                        reply_to="scene_graph",
                        success=False,
                        error_message=str(e),
                    )

    # ==================== Goal Query API ====================

    def get_goal_description(self) -> Optional[str]:
        """Get the goal description.

        Returns:
            Goal description string.
        """
        return self._goal["goal"]["description"]

    def get_goal(self) -> Optional[Dict[str, Any]]:
        """Get goal data.

        Returns:
            Goal data dictionary, or None if no goal is set.
        """
        if self._goal:
            return self._goal.copy()
        return None

    def evaluate_goal(self) -> bool:
        """Evaluate whether the goal is achieved.

        Returns:
            bool: Whether the goal is achieved.
        """
        if self._goal:
            world_state = {
                "entities": self._nodes,
            }
            return self._goal.evaluate(world_state)

        return False

    # ==================== Node Query API ====================

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes.

        Returns:
            List of node copies.
        """
        return [node.copy() for node in self._nodes]

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID.

        Args:
            node_id: Node ID.

        Returns:
            Node data copy, or None if it does not exist.
        """
        for node in self._nodes:
            if str(node.get("id")) == str(node_id):
                return node.copy()
        return None

    def get_node_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """Get a node by label.

        Args:
            label: Node label.

        Returns:
            Node data copy, or None if it does not exist.
        """
        for node in self._nodes:
            if node.get("properties", {}).get("label") == label:
                return node.copy()
        return None

    def get_nodes_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get nodes by category.

        Args:
            category: Node category (robot, prop, building, goal, etc.).

        Returns:
            List of nodes matching the category.
        """
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == category
        ]

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get nodes by type.

        Args:
            node_type: Node type.

        Returns:
            List of nodes matching the type.
        """
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("type") == node_type
        ]

    def search_nodes(self, **criteria) -> List[Dict[str, Any]]:
        """Search nodes by criteria.

        Args:
            **criteria: Search criteria; supports nested property queries.

        Returns:
            List of nodes matching the criteria.
        """
        result = []
        for node in self._nodes:
            match = True
            for key, value in criteria.items():
                if "." in key:
                    # Support nested property queries, such as 'properties.category'
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
        """
        Return category and type information for the node with the given label.

        Args:
            label (str): Node label name.

        Returns:
            Optional[Dict[str, str]]: Dictionary containing the node's category and type.
                                    Returns None if no matching node is found.

        Example output:
            {
                'category': 'building',
                'type': 'semantic'
            }
        """
        for node in self._nodes:
            properties = node.get("properties", {})
            if properties.get("label") == label:
                category = properties.get("category")
                node_type = properties.get("type")  # Assumes type is also stored in properties
                if category is not None and node_type is not None:
                    return {"category": category, "type": node_type}
                else:
                    return None  # If fields are missing, this could also return partial fields
        return None  # No matching label found

    # ==================== Edge Query API ====================

    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Get all edges.

        Returns:
            List of edge copies.
        """
        return [edge.copy() for edge in self._edges]

    def get_edges_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Get all edges from a source node.

        Args:
            source_id: Source node ID.

        Returns:
            List of edges outgoing from the specified node.
        """
        return [
            edge.copy()
            for edge in self._edges
            if str(edge.get("source")) == str(source_id)
        ]

    def get_edges_by_target(self, target_id: str) -> List[Dict[str, Any]]:
        """Get all edges pointing to a target node.

        Args:
            target_id: Target node ID.

        Returns:
            List of edges pointing to the specified node.
        """
        return [
            edge.copy()
            for edge in self._edges
            if str(edge.get("target")) == str(target_id)
        ]

    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific edge.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.

        Returns:
            Edge data copy, or None if it does not exist.
        """
        for edge in self._edges:
            if str(edge.get("source")) == str(source_id) and str(
                edge.get("target")
            ) == str(target_id):
                return edge.copy()
        return None

    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """Get neighbor node IDs for a node.

        Args:
            node_id: Node ID.
            direction: Direction ('in', 'out', 'both').

        Returns:
            Neighbor node ID list.
        """
        neighbors = set()

        if direction in ["out", "both"]:
            # Target nodes of outgoing edges
            for edge in self._edges:
                if str(edge.get("source")) == str(node_id):
                    neighbors.add(str(edge.get("target")))

        if direction in ["in", "both"]:
            # Source nodes of incoming edges
            for edge in self._edges:
                if str(edge.get("target")) == str(node_id):
                    neighbors.add(str(edge.get("source")))

        return list(neighbors)

    def get_neighbors_by_relation(
        self, node_id: Union[int, str], relation: str
    ) -> List[Union[int, str]]:
        """
        Find and return all directly related neighbor node IDs for the given node ID and relation type.

        This function searches both directions:
        1. Edges where node_id is the source.
        2. Edges where node_id is the target.

        Args:
            node_id (Union[int, str]): Unique ID of the node to query.
            relation (str): Edge relation type to filter by (for example, 'stationed_at', 'carrying', 'stored_at').

        Returns:
            List[Union[int, str]]: Matching neighbor node IDs, or an empty list if none are found.
        """
        # Use a set to automatically handle possible duplicate IDs
        neighbor_ids: Set[Union[int, str]] = set()

        for edge in self._edges:
            edge_type = edge.get("type")

            # 1. Check whether the edge type matches
            if edge_type == relation:
                source = edge.get("source")
                target = edge.get("target")

                # 2. Check both directions for whether the node ID participates in the edge
                if source == str(node_id):
                    # If node_id is the source, the neighbor is the target
                    neighbor_ids.add(target)
                elif target == str(node_id):
                    # If node_id is the target, the neighbor is the source
                    neighbor_ids.add(source)

        return list(neighbor_ids)

    def get_edges_from_node(
        self, node_id: Union[int, str], edge_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get edges outgoing from a specified node, optionally filtered by type.

        Args:
            node_id: Source node ID.
            edge_type: Edge type; if provided, only edges of this type are returned.

        Returns:
            List of edges outgoing from the specified node (edge copies).
        """
        edges = []
        node_id_str = str(node_id)

        for edge in self._edges:
            if str(edge.get("source")) == node_id_str:
                # Filter by edge type if specified
                if edge_type is None or edge.get("type") == edge_type:
                    edges.append(edge.copy())

        return edges

    def get_edges_by_type(self, edge_type: str) -> List[Dict[str, Any]]:
        """
        Get all edges of the specified type.

        Args:
            edge_type: Edge type.

        Returns:
            List of edges of the specified type.
        """
        return [edge.copy() for edge in self._edges if edge.get("type") == edge_type]

    def update_edge_target(
        self,
        source_id: Union[int, str],
        old_target_id: Union[int, str],
        new_target_id: Union[int, str],
        edge_type: Optional[str] = None,
    ) -> bool:
        """
        Update an edge target node.

        Args:
            source_id: Source node ID.
            old_target_id: Old target node ID.
            new_target_id: New target node ID.
            edge_type: Edge type (optional, for more precise matching).

        Returns:
            Whether the update succeeded.
        """
        source_str = str(source_id)
        old_target_str = str(old_target_id)
        new_target_str = str(new_target_id)

        for i, edge in enumerate(self._edges):
            if (
                str(edge.get("source")) == source_str
                and str(edge.get("target")) == old_target_str
            ):
                if edge_type is None or edge.get("type") == edge_type:
                    # Create a new edge, preserving existing properties and changing only target
                    new_edge = edge.copy()
                    new_edge["target"] = new_target_id  # Preserve original type
                    self._edges[i] = new_edge
                    return True

        return False

    def remove_edges_from_node(
        self, node_id: Union[int, str], edge_type: Optional[str] = None
    ) -> int:
        """
        Remove all edges outgoing from the specified node.

        Args:
            node_id: Source node ID.
            edge_type: Edge type; if provided, only edges of this type are removed.

        Returns:
            Number of removed edges.
        """
        node_id_str = str(node_id)
        edges_to_remove = []

        for edge in self._edges:
            if str(edge.get("source")) == node_id_str:
                if edge_type is None or edge.get("type") == edge_type:
                    edges_to_remove.append(edge)

        # Remove found edges
        for edge in edges_to_remove:
            self._edges.remove(edge)

        return len(edges_to_remove)

    def has_edge_of_type(
        self, source_id: Union[int, str], target_id: Union[int, str], edge_type: str
    ) -> bool:
        """
        Check whether an edge of the specified type exists between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Edge type.

        Returns:
            Whether an edge of this type exists.
        """
        source_str = str(source_id)
        target_str = str(target_id)

        for edge in self._edges:
            if (
                str(edge.get("source")) == source_str
                and str(edge.get("target")) == target_str
                and edge.get("type") == edge_type
            ):
                return True

        return False

    # ==================== Entity Query API ====================

    def get_robot(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get a robot node by ID.

        Args:
            robot_id: Robot ID.

        Returns:
            Robot node data copy, or None if it does not exist.
        """
        for node in self._nodes:
            if (
                str(node.get("id")) == str(robot_id)
                and node.get("properties", {}).get("category") == "robot"
            ):
                return node.copy()
        return None

    def get_prop(self, prop_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a prop node by ID.

        Args:
            prop_id: Prop ID.

        Returns:
            Prop node data copy, or None if it does not exist.
        """
        if not prop_id:
            # Return all prop nodes
            return [
                node.copy()
                for node in self._nodes
                if node.get("properties", {}).get("category") == "prop"
            ]
        else:
            for node in self._nodes:
                if (
                    str(node.get("id")) == str(prop_id)
                    and node.get("properties", {}).get("category") == "prop"
                ):
                    return node.copy()
            return None

    def get_building(self, building_id: str) -> Optional[Dict[str, Any]]:
        """Get a building node by ID.

        Args:
            building_id: Building ID.

        Returns:
            Building node data copy, or None if it does not exist.
        """
        for node in self._nodes:
            if (
                str(node.get("id")) == str(building_id)
                and node.get("properties", {}).get("category") == "building"
            ):
                return node.copy()
        return None

    def get_all_robots(self) -> List[Dict[str, Any]]:
        """Get all robot nodes.

        Returns:
            List of robot node data copies (may be empty).
        """
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == "robot"
        ]

    def get_all_props(self) -> List[Dict[str, Any]]:
        """Get all prop nodes.

        Returns:
            List of prop node data copies (may be empty).
        """
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == "prop"
        ]

    def get_all_buildings(self) -> List[Dict[str, Any]]:
        """Get all building nodes.

        Returns:
            List of building node data copies (may be empty).
        """
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == "building"
        ]

    def get_all_trans_facilitys(self) -> List[Dict[str, Any]]:
        """Get all transportation facility nodes.

        Returns:
            List of transportation facility node data copies (may be empty).
        """
        return [
            node.copy()
            for node in self._nodes
            if node.get("properties", {}).get("category") == "trans_facility"
        ]

    def get_goal_node(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get a goal node by ID.

        Args:
            goal_id: Goal ID.

        Returns:
            Goal node data copy, or None if it does not exist.
        """
        for node in self._nodes:
            if (
                str(node.get("id")) == str(goal_id)
                and node.get("properties", {}).get("category") == "goal"
            ):
                return node.copy()
        return None

    # ==================== Position Access API ====================

    def _get_node_position(self, node: Dict) -> List[float]:
        """
        Get the representative position of a node:
        - shape.type == 'point'      -> center
        - shape.type == 'circle'     -> center
        - shape.type == 'rectangle'  -> bbox midpoint
        - shape.type == 'polygon'    -> average of vertices (simple centroid approximation)
        - shape.type == 'linestring' -> point at half the polyline arc length
        Return [0, 0] for all other cases.
        """
        shape = node.get("shape") or {}
        stype = shape.get("type")

        # 1) Point / circle
        if stype == "point":
            c = shape.get("center", [0.0, 0.0])
            return [float(c[0]), float(c[1])]
        if stype == "circle":
            c = shape.get("center", [0.0, 0.0])
            return [float(c[0]), float(c[1])]

        # 2) Rectangle: center point
        if stype == "rectangle":
            minc = shape.get("min_corner", [0.0, 0.0])
            maxc = shape.get("max_corner", [0.0, 0.0])
            return [
                (float(minc[0]) + float(maxc[0])) / 2.0,
                (float(minc[1]) + float(maxc[1])) / 2.0,
            ]

        # 3) Polygon: average of vertices (simplified area center)
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

        # 4) Polyline: use the midpoint by arc length
        if stype == "linestring":
            pts = shape.get("points", [])
            if not pts:
                return [0.0, 0.0]
            if len(pts) == 1:
                p = pts[0]
                return [float(p[0]), float(p[1])]
            # Calculate total length
            seg_lens = []
            total = 0.0
            for i in range(len(pts) - 1):
                x1, y1 = float(pts[i][0]), float(pts[i][1])
                x2, y2 = float(pts[i + 1][0]), float(pts[i + 1][1])
                L = math.hypot(x2 - x1, y2 - y1)
                seg_lens.append(L)
                total += L
            if total <= 1e-9:
                # Degenerate case: use midpoint of first and last points
                x1, y1 = float(pts[0][0]), float(pts[0][1])
                x2, y2 = float(pts[-1][0]), float(pts[-1][1])
                return [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

            half = total / 2.0
            acc = 0.0
            for i, L in enumerate(seg_lens):
                if acc + L >= half:
                    # Interpolate within segment i
                    x1, y1 = float(pts[i][0]), float(pts[i][1])
                    x2, y2 = float(pts[i + 1][0]), float(pts[i + 1][1])
                    t = (half - acc) / (L if L > 1e-12 else 1.0)
                    return [x1 + (x2 - x1) * t, y1 + (y2 - y1) * t]
                acc += L
            # Fallback: return last point
            pe = pts[-1]
            return [float(pe[0]), float(pe[1])]

        # Other unknown types
        return [0.0, 0.0]

    def get_position_by_id(self, node_id: int) -> Optional[List[float]]:
        """
        Get a single entity position by node ID.

        Args:
            node_id (int): Unique ID of the node to query.

        Returns:
            Optional[List[float]]: Entity position coordinates [x, y], or None if not found.
        """
        node = self.get_node_by_id(node_id)
        return self._get_node_position(node)

    def get_position_by_label(self, node_label: str) -> Optional[List[float]]:
        """
        Get a single entity position by node label. Assumes labels are unique.

        Args:
            node_label (str): Label of the node to query.

        Returns:
            Optional[List[float]]: Entity position coordinates [x, y], or None if not found.
        """
        # Assumes 'label' is stored in the 'properties' field
        nodes_with_label = self.search_nodes(**{"properties.label": node_label})
        if nodes_with_label:
            return self._get_node_position(nodes_with_label[0])
        return None

    def get_positions_by_entity_type(
        self, entity_categories: List[str] = ["building", "robot"]
    ) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """
        Batch-get positions for specified entity categories and return a structured dictionary.

        Args:
            entity_categories (List[str]): Entity categories to query;
                                         defaults to ['building', 'robot'].

        Returns:
            Dict[str, Dict[str, Dict[str, List[float]]]]: Nested dictionary containing position information.
        """
        results = {}
        for category in entity_categories:
            # Get all nodes in this category
            nodes_in_category = self.get_nodes_by_category(category)
            if not nodes_in_category:
                continue

            # Initialize a dictionary for this category
            results[category] = {}

            for node in nodes_in_category:
                # Safely get subtype and label
                properties = node.get("properties", {})
                subtype = properties.get("type")
                label = properties.get("label")

                # If this is the robot category, check whether status is error
                if category == "robot":
                    status = properties.get("status")
                    if status == "error":
                        continue  # Skip robots with error status

                # Extract position
                position = self._get_node_position(node)

                # Record only when subtype, label, and position all exist
                if not (subtype and label and position):
                    continue

                # Initialize a dictionary if this subtype appears for the first time
                if subtype not in results[category]:
                    results[category][subtype] = {}

                # Record this entity position, using label as the unique identifier
                results[category][subtype][label] = position

        return results

    def get_boundary_features(
        self,
        district: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract boundary features, considering only category in {area, trans_facility, building}.
        Returns: { label: { "kind": "area|line|point|rectangle|circle", ... } }
        - area / line / point / rectangle: use "coords": [[x,y], ...]
        - circle: use "center": [x,y], "radius": float
        """
        results: Dict[str, Dict[str, Any]] = {}
        allowed = {"area", "trans_facility", "building"}

        for node in getattr(self, "_nodes", []):
            props = node.get("properties") or {}
            if not isinstance(props, dict):
                continue
            if props.get("category") not in allowed:
                continue
            if district is not None and props.get("district") != district:
                continue

            label = str(
                props.get("label")
                or node.get("label")
                or node.get("id")
                or props.get("id")
            )
            shape = node.get("shape") or {}
            t = isinstance(shape, dict) and shape.get("type")
            if not t:
                continue

            feature: Dict[str, Any] = {}
            try:
                if t == "polygon":
                    verts = shape.get("vertices") or []
                    coords = [
                        [float(v[0]), float(v[1])] for v in verts if v and len(v) >= 2
                    ]
                    if coords:
                        feature = {"kind": "area", "coords": coords}

                elif t == "rectangle":
                    a = shape.get("min_corner") or []
                    b = shape.get("max_corner") or []
                    x1, y1 = float(a[0]), float(a[1])
                    x2, y2 = float(b[0]), float(b[1])
                    feature = {
                        "kind": "rectangle",
                        "coords": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    }

                elif t == "circle":
                    c = shape.get("center") or []
                    r = shape.get("radius")
                    cx, cy, rr = float(c[0]), float(c[1]), float(r)
                    feature = {"kind": "circle", "center": [cx, cy], "radius": rr}

                elif t == "point":
                    c = shape.get("center") or []
                    cx, cy = float(c[0]), float(c[1])
                    feature = {"kind": "point", "coords": [[cx, cy]]}

                elif t in ("linestring", "polyline"):
                    pts = shape.get("points") or []
                    coords = [
                        [float(p[0]), float(p[1])] for p in pts if p and len(p) >= 2
                    ]
                    if coords:
                        feature = {"kind": "line", "coords": coords}
            except Exception:
                feature = {}

            if feature:
                results[label] = feature

        return results

    # ==================== Path and Position Query API ====================
    def find_path(self, src_id: str, dst_id: str) -> List[str]:
        return _nav_find_path(self, src_id, dst_id)

    def find_side_edges_near_path(
        self, src_id: str, dst_id: str
    ) -> List[Dict[str, Any]]:
        return _nav_find_side_edges_near_path(self, src_id, dst_id)

    def find_candidate_locations(
        self, object_or_carrier_id: str, prefer_parking: bool = True
    ) -> List[Dict[str, Any]]:
        return _nav_find_candidate_locations(self, object_or_carrier_id, prefer_parking)

    def has_path(self, src_id: str, dst_id: str) -> bool:
        path = _nav_find_path(self, src_id, dst_id)
        return len(path) > 0

    # ==================== Other Query API ====================

    def _get_node_location_label(
        self, node: Dict[str, Any]
    ) -> Optional[Union[str, int]]:
        """Get the node location ID."""
        if not node:
            return None

        node_category = node.get("properties", {}).get("category")

        # A building's location is itself
        if node_category in ("building", "trans_facility", "area"):
            return node.get("properties", {}).get("label")

        # Other node locations come from their location property
        location_label = node.get("properties", {}).get("location", {}).get("label")
        if location_label:
            return location_label

        return None

    def get_available_robots(
        self, types: Optional[List[str]] = None, concise: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        def canonicalize(t: Optional[str]) -> Optional[str]:
            if not t:
                return None
            t_norm = t.strip().lower().replace(" ", "").replace("-", "_")
            aliases = {
                # core
                "uav": "UAV",
                "fw_uav": "FW_UAV",
                "fwuav": "FW_UAV",
                "fixedwing": "FW_UAV",
                "fixed_wing": "FW_UAV",
                "ugv": "UGV",
                "quadruped": "Quadruped",
                "quadrupte": "Quadruped",  # common typo
                "quad": "Quadruped",
                "humanoid": "Humanoid",
            }
            return aliases.get(t_norm)

        id_tail_re = re.compile(r"(\d+)$")

        def extract_numeric_tail(label: str) -> Optional[int]:
            m = id_tail_re.search(label.strip())
            return int(m.group(1)) if m else None

        # types filter
        include_set: Optional[Set[str]] = None
        if types is not None:
            include_set = {c for c in (canonicalize(t) for t in types) if c is not None}

        # storage
        detailed: Dict[str, Dict[str, Any]] = {}

        # iterate graph nodes
        for node in self._nodes:
            props = node.get("properties", {})
            if props.get("category") != "robot":
                continue
            # must be healthy & with sufficient battery
            if (
                props.get("status") == "error"
                or props.get("battery_level") < 20.0
                or props.get("comm") == "jammed"
            ):
                continue

            robot_type_raw = props.get("type")
            robot_label = props.get("label")
            canonical_type = canonicalize(robot_type_raw)
            if not canonical_type or not robot_label:
                continue

            # apply type filter if provided
            if include_set is not None and canonical_type not in include_set:
                continue

            # bucket
            if canonical_type not in detailed:
                detailed[canonical_type] = {"labels": [], "num": 0}
            detailed[canonical_type]["labels"].append(robot_label)
            detailed[canonical_type]["num"] += 1

        if not concise:
            return detailed

        # concise mapping: type -> list of numeric suffixes
        concise_map: Dict[str, List[int]] = {}
        for rtype, info in detailed.items():
            nums: List[int] = []
            for label in info.get("labels", []):
                n = extract_numeric_tail(label)
                if n is not None:
                    nums.append(n)
            if nums:  # Return this type only when numeric suffixes were parsed
                concise_map[rtype] = nums

        return concise_map

    def get_robot_states(self) -> dict:
        """
        Extract robot state information from the node list.

        Args:
            nodes (list): List containing robot nodes.

        Returns:
            dict: Robot state dictionary keyed by robot_label, with state information as values.
        """
        robot_states = {}

        # Get all nodes
        all_nodes = self._nodes

        for node in all_nodes:
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

    def get_node_map(
        self, map_type: Literal["id_to_label", "label_to_id"] = "id_to_label"
    ) -> Dict[Union[str, int], Union[str, int]]:
        """
        Generate an ID/Label mapping dictionary for all nodes.

        Args:
            map_type (str): Mapping type to generate.
                            - 'id_to_label' (default): map node ID to label.
                            - 'label_to_id': map node label to ID.

        Returns:
            Dict: Requested mapping dictionary.

        Raises:
            ValueError: If an invalid map_type is provided.
        """
        if map_type not in ["id_to_label", "label_to_id"]:
            raise ValueError("Invalid map_type. Must be 'id_to_label' or 'label_to_id'.")

        node_map = {}
        for node in self._nodes:
            # Safely get ID and Label from the node structure
            # .get('properties', {}) prevents errors when the 'properties' key is missing
            node_id = node.get("id")
            label = node.get("properties", {}).get("label")

            # Create mapping entries only when both ID and Label exist
            if node_id is not None and label is not None:
                if map_type == "id_to_label":
                    node_map[node_id] = label
                else:  # map_type must be 'label_to_id' here
                    node_map[label] = node_id

        return node_map

    def get_all_node_categories(self) -> Dict[str, str]:
        """
        Generate a Label-to-Category mapping dictionary for all nodes.

        This function iterates through all nodes, extracts their 'label' and 'category'
        properties, and returns a dictionary mapping each label to its category.

        Returns:
            Dict[str, str]: Dictionary representing the "node label -> category" mapping.
                            Nodes missing label or category are not included.

        Example output:
            {
                'Parking Lot-1': 'building',
                'vehicle-F-3803': 'prop'
            }
        """
        category_map: Dict[str, str] = {}
        for node in self._nodes:
            # Safely get the properties dictionary from the node structure
            properties = node.get("properties", {})

            # Get label and category from properties
            label = properties.get("label")
            category = properties.get("category")

            # Create a mapping entry only when both label and category exist
            if label is not None and category is not None:
                category_map[label] = category

        return category_map

    # ==================== Event-Driven CRUD API ====================

    def handle_node_event(self, event_type: str, node_data: Dict[str, Any]) -> bool:
        """Handle a node event.

        Args:
            event_type: Event type ('add', 'update', 'remove').
            node_data: Node data; remove events only require id.

        Returns:
            Whether the operation succeeded.
        """
        try:
            if event_type == "add":
                return self._add_node(node_data)
            elif event_type == "update":
                return self._update_node(node_data)
            elif event_type == "remove":
                node_id = node_data.get("id")
                if node_id:
                    return self._remove_node(str(node_id))
                return False
            else:
                self.logger.warning(f"Unknown node event type: {event_type}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to handle node event: {e}")
            return False

    def handle_edge_event(self, operation: str, edge_data: Dict[str, Any]) -> bool:
        """Handle an edge event.

        This method handles edge operations passed through DataModificationEvent.
        All edge add/update/remove operations should call this method through the event mechanism.

        Args:
            operation: Operation type ('add', 'update', 'remove').
            edge_data: Edge data; must contain source and target fields.

        Returns:
            bool: Whether the operation succeeded.

        Note:
            This method should not be called directly; trigger it by publishing a DataModificationEvent.
        """
        try:
            self.logger.debug(
                f"Processing edge event: {operation} for edge {edge_data.get('id')}"
            )

            if operation == "add":
                return self._add_edge(edge_data)
            elif operation == "update":
                return self._update_edge(edge_data)
            elif operation == "remove":
                edge_id = edge_data.get("id") or edge_data.get("edge_id")
                if not edge_id:
                    # Try to build edge_id from source and target
                    source = edge_data.get("source")
                    target = edge_data.get("target")
                    if source and target:
                        edge_id = f"{source}->{target}"
                    else:
                        self.logger.error(
                            "Remove operation requires edge id or source/target"
                        )
                        return False
                return self._remove_edge(edge_id)
            else:
                self.logger.warning(f"Unknown edge operation: {operation}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to handle edge event: {e}")
            return False

    # ==================== Internal Implementation Methods ====================

    def _add_node(self, node_data: Dict[str, Any]) -> bool:
        """Add a node.

        Args:
            node_data: Node data.

        Returns:
            Whether the add succeeded.
        """
        # Ensure ID exists
        if "id" not in node_data:
            node_data["id"] = str(uuid.uuid4())

        node_id = str(node_data["id"])

        # Check whether ID already exists
        if self.get_node_by_id(node_id):
            self.logger.warning(f"Node ID {node_id} already exists")
            return False

        # Add node
        self._nodes.append(node_data.copy())
        self.logger.info(f"Added node: {node_id}")
        return True

    def _update_node(self, node_data: Dict[str, Any]) -> bool:
        """Update a node.

        Args:
            node_data: Node data; must contain id.

        Returns:
            Whether the update succeeded.
        """
        node_id = node_data.get("id")
        if not node_id:
            self.logger.warning("Missing ID when updating node")
            return False

        node_id = str(node_id)

        # Find and update node
        for i, node in enumerate(self._nodes):
            if str(node.get("id")) == node_id:
                # Deep merge update
                updated_node = node.copy()
                self._deep_update(updated_node, node_data)
                self._nodes[i] = updated_node
                self.logger.info(f"Updated node: {node_id}")
                return True

        self.logger.warning(f"Node not found for update: {node_id}")
        return False

    def _remove_node(self, node_id: str) -> bool:
        """Remove a node.

        Args:
            node_id: Node ID.

        Returns:
            Whether the removal succeeded.
        """
        # Remove node
        for i, node in enumerate(self._nodes):
            if str(node.get("id")) == str(node_id):
                self._nodes.pop(i)

                # Remove related edges
                self._edges = [
                    edge
                    for edge in self._edges
                    if (
                        str(edge.get("source")) != node_id
                        and str(edge.get("target")) != node_id
                    )
                ]

                self.logger.info(f"Removed node: {node_id}")
                return True

        self.logger.warning(f"Node not found for removal: {node_id}")
        return False

    def _add_edge(self, edge_data: Dict[str, Any]) -> bool:
        """Add an edge.

        Args:
            edge_data: Edge data; must contain source and target.

        Returns:
            Whether the add succeeded.
        """
        source_id = edge_data.get("source")
        target_id = edge_data.get("target")

        if not source_id or not target_id:
            self.logger.warning("Missing source or target when adding edge")
            return False

        # Verify nodes exist
        if not self.get_node_by_id(source_id) or not self.get_node_by_id(target_id):
            self.logger.warning(f"Source node ({source_id}) or target node ({target_id}) does not exist")
            return False

        # Check whether edge already exists
        if self.get_edge(source_id, target_id):
            self.logger.warning(f"Edge {source_id} -> {target_id} already exists")
            return False

        # Add edge
        self._edges.append(edge_data.copy())
        self.logger.info(f"Added edge: {source_id} -> {target_id}")
        return True

    def _update_edge(self, edge_data: Dict[str, Any]) -> bool:
        """Update an edge.

        Args:
            edge_data: Edge data; must contain source and target.

        Returns:
            Whether the update succeeded.
        """
        # source_id = edge_data.get('source')
        # target_id = edge_data.get('target')

        # if not source_id or not target_id:
        #     self.logger.warning("Missing source or target when updating edge")
        #     return False

        # source_id = str(source_id)
        # target_id = str(target_id)

        # # Find and update edge
        # for i, edge in enumerate(self._edges):
        #     if (str(edge.get('source')) == source_id and
        #             str(edge.get('target')) == target_id):
        #         # Deep merge update
        #         updated_edge = edge.copy()
        #         self._deep_update(updated_edge, edge_data)
        #         self._edges[i] = updated_edge
        #         self.logger.info(f"Updated edge: {source_id} -> {target_id}")
        #         return True

        # self.logger.warning(f"Edge not found for update: {source_id} -> {target_id}")
        # return False

        source_id = edge_data.get("source")
        new_target_id = edge_data.get("target")
        edge_type = edge_data.get("type")

        if not all([source_id, new_target_id, edge_type]):
            self.logger.warning(
                f"Missing required keys when updating edge: 'source', 'target', or 'type'. Provided data: {edge_data}"
            )
            return False

        for i, edge in enumerate(self._edges):
            if (
                str(edge.get("source")) == str(source_id)
                and str(edge.get("type")) == edge_type
            ):
                self._edges[i]["target"] = new_target_id
                return True

        return False

    def _remove_edge(self, edge_id: str) -> bool:
        """Remove an edge.

        Args:
            edge_id: Edge ID, either in 'source_id->target_id' format or as a direct edge ID.

        Returns:
            Whether the removal succeeded.
        """
        # Try to parse edge_id
        if "->" in edge_id:
            source_id, target_id = edge_id.split("->", 1)
            source_id = source_id.strip()
            target_id = target_id.strip()

            # Search by source and target
            for i, edge in enumerate(self._edges):
                if str(edge.get("source")) == str(source_id) and str(
                    edge.get("target")
                ) == str(target_id):
                    self._edges.pop(i)
                    self.logger.info(f"Removed edge: {source_id} -> {target_id}")
                    return True
        else:
            # Search by edge ID
            for i, edge in enumerate(self._edges):
                if str(edge.get("id", "")) == str(edge_id):
                    self._edges.pop(i)
                    self.logger.info(f"Removed edge: {edge_id}")
                    return True

        self.logger.warning(f"Edge not found for removal: {edge_id}")
        return False

    # def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
    #     """Deep-update a dictionary.

    #     Args:
    #         target: Target dictionary.
    #         source: Source dictionary.
    #     """
    #     for key, value in source.items():
    #         if key in target and isinstance(target[key], dict) and isinstance(value, dict):
    #             self._deep_update(target[key], value)
    #         else:
    #             target[key] = value

    def _deep_update(self, target_dict: Dict[str, Any], source: Dict[str, Any]) -> bool:
        """
        Find keys with the same names as source in target_dict in place and overwrite their values.
        If the current level does not match and the value is a dict, recurse downward.
        On match, replace the whole value directly whether the old/new value is a dict or not; do not merge.
        """
        updated = False

        for key, value in list(target_dict.items()):
            # Case 1: current level matches -> overwrite directly, including whole-dict replacement
            if key in source:
                target_dict[key] = source[key]
                updated = True
            # Case 2: no match, but current value is a dict -> search/overwrite deeper
            elif isinstance(value, dict):
                if self._deep_update(value, source):
                    updated = True

        return updated

    # ==================== Utility Methods ====================

    def get_id(self) -> str:
        """Get context ID."""
        return f"simplified_context_{id(self)}"

    def get_scene_config(self) -> Dict[str, Any]:
        """Get scene config."""
        return {"nodes": self.get_all_nodes(), "edges": self.get_all_edges()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
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
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def clear(self) -> None:
        """Clear all data."""
        self._nodes.clear()
        self._edges.clear()
        self.logger.info("Cleared all nodes and edges")

    def load_from_dict(self, data: Dict[str, Any]) -> bool:
        """Load data from a dictionary.

        Args:
            data: Dictionary containing nodes and edges.

        Returns:
            Whether loading succeeded.
        """
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

    def __str__(self) -> str:
        """String representation."""
        return (
            f"SemanticSceneGraph(nodes={len(self._nodes)}, edges={len(self._edges)})"
        )

    def __repr__(self) -> str:
        """Detailed string representation."""
        stats = self._get_node_statistics()
        return (
            f"SemanticSceneGraph(id={self.get_id()}, "
            f"nodes={len(self._nodes)}, edges={len(self._edges)}, "
            f"categories={stats})"
        )

    def __del__(self):
        """Destructor that cleans up event subscriptions."""
        try:
            for subscription_id in self._event_subscriptions:
                unsubscribe_event(subscription_id)
            self.logger.info("Event subscriptions cleaned up")
        except Exception as e:
            self.logger.error(f"Error cleaning up event subscriptions: {e}")

    async def cleanup(self):
        """Async cleanup method.

        Cleans up event subscriptions and releases resources.
        Recommended when the context is no longer used.
        """
        try:
            for subscription_id in self._event_subscriptions:
                unsubscribe_event(subscription_id)
            self._event_subscriptions.clear()
            self._is_initialized = False
            self.logger.info("SemanticSceneGraph cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
            raise

    # ==================== Event-Driven Data Modification API ====================

    async def add_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Add a node asynchronously through the event mechanism.

        Args:
            node_data: Node data.
            request_id: Request ID used to track the operation result.

        Returns:
            bool: Whether the operation was published successfully.
        """
        try:
            event = DataModificationEvent(
                operation="add",
                entity_type="node",
                entity_id=node_data.get("id", ""),
                data=node_data,
                request_id=request_id or str(uuid.uuid4()),
            )
            await publish_event_sync(event)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish add node event: {e}")
            return False

    async def update_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Update a node asynchronously through the event mechanism.

        Args:
            node_data: Node data.
            request_id: Request ID used to track the operation result.

        Returns:
            bool: Whether the operation was published successfully.
        """
        try:
            event = DataModificationEvent(
                operation="update",
                entity_type="node",
                entity_id=node_data.get("id", ""),
                data=node_data,
                request_id=request_id or str(uuid.uuid4()),
            )
            await publish_event_sync(event)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish update node event: {e}")
            return False

    async def remove_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Remove a node asynchronously through the event mechanism.

        Args:
            node_id: Node ID.
            request_id: Request ID used to track the operation result.

        Returns:
            bool: Whether the operation was published successfully.
        """
        try:
            node_id = node_data.get("id")
            if not node_id:
                self.logger.warning("Remove node requires node ID")
                return False

            event = DataModificationEvent(
                operation="remove",
                entity_type="node",
                entity_id=node_id,
                data={"id": node_id},
                request_id=request_id or str(uuid.uuid4()),
            )
            await publish_event_sync(event)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish remove node event: {e}")
            return False

    async def add_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Add an edge asynchronously through the event mechanism.

        Args:
            edge_data: Edge data.
            request_id: Request ID used to track the operation result.

        Returns:
            bool: Whether the operation was published successfully.
        """
        try:
            edge_id = (
                edge_data.get("id")
                or f"{edge_data.get('source')}->{edge_data.get('target')}"
            )
            event = DataModificationEvent(
                operation="add",
                entity_type="edge",
                entity_id=edge_id,
                data=edge_data,
                request_id=request_id or str(uuid.uuid4()),
            )
            await publish_event_sync(event)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish add edge event: {e}")
            return False

    async def update_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Update an edge asynchronously through the event mechanism.

        Args:
            edge_data: Edge data.
            request_id: Request ID used to track the operation result.

        Returns:
            bool: Whether the operation was published successfully.
        """
        try:
            edge_id = (
                edge_data.get("id")
                or f"{edge_data.get('source')}->{edge_data.get('target')}"
            )
            event = DataModificationEvent(
                operation="update",
                entity_type="edge",
                entity_id=edge_id,
                data=edge_data,
                request_id=request_id or str(uuid.uuid4()),
            )
            await publish_event_sync(event)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish update edge event: {e}")
            return False

    async def remove_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Remove an edge asynchronously through the event mechanism.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            request_id: Request ID used to track the operation result.

        Returns:
            bool: Whether the operation was published successfully.
        """
        try:
            source_id = edge_data.get("source")
            target_id = edge_data.get("target")
            if not source_id or not target_id:
                self.logger.warning("Remove edge requires source and target IDs")
                return False

            edge_id = f"{source_id}->{target_id}"
            event = DataModificationEvent(
                operation="remove",
                entity_type="edge",
                entity_id=edge_id,
                data={"source": source_id, "target": target_id, "id": edge_id},
                request_id=request_id or str(uuid.uuid4()),
            )
            await publish_event_sync(event)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish remove edge event: {e}")
            return False

    async def remove_edge_by_id_async(
        self, edge_id: str, request_id: Optional[str] = None
    ) -> bool:
        """Remove an edge asynchronously by edge ID through the event mechanism.

        Args:
            edge_id: Edge ID.
            request_id: Request ID used to track the operation result.

        Returns:
            bool: Whether the operation was published successfully.
        """
        try:
            event = DataModificationEvent(
                operation="remove",
                entity_type="edge",
                entity_id=edge_id,
                data={"id": edge_id},
                request_id=request_id or str(uuid.uuid4()),
            )
            await publish_event_sync(event)
            return True
        except Exception as e:
            self.logger.error(f"Failed to publish remove edge by id event: {e}")
            return False


# ==================== Global Singleton Management ====================

# Global SemanticSceneGraph instance management
_global_scene_graph: Optional[SemanticSceneGraph] = None
_context_lock = asyncio.Lock()


def get_global_scene_graph() -> AbstractSceneGraph:
    """Get the global SemanticSceneGraph instance.

    Returns:
        AbstractSceneGraph: Global singleton instance.
    """
    global _global_scene_graph
    if _global_scene_graph is None:
        _global_scene_graph = SemanticSceneGraph()
    return _global_scene_graph


async def initialize_global_scene_graph(
    initial_nodes: Optional[List[Dict[str, Any]]] = None,
    initial_edges: Optional[List[Dict[str, Any]]] = None,
    initial_goal: Optional[str] = None,
) -> AbstractSceneGraph:
    """Initialize the global SemanticSceneGraph instance.

    Args:
        initial_nodes: Initial node list.
        initial_edges: Initial edge list.
        initial_goal: Initial goal.
        new_case: New case.

    Returns:
        AbstractSceneGraph: Global singleton instance.
    """
    async with _context_lock:
        global _global_scene_graph
        if _global_scene_graph is None:
            _global_scene_graph = SemanticSceneGraph(
                initial_nodes=initial_nodes,
                initial_edges=initial_edges,
                initial_goal=initial_goal,
            )
        else:
            print(f"SemanticSceneGraph already initialized: {_global_scene_graph}")
        return _global_scene_graph


async def cleanup_global_scene_graph():
    """Clean up the global SemanticSceneGraph instance.

    Cleans up event subscriptions and releases resources.
    """
    async with _context_lock:
        global _global_scene_graph
        if _global_scene_graph:
            await _global_scene_graph.cleanup()
            _global_scene_graph = None


def reset_global_scene_graph():
    """Reset the global SemanticSceneGraph instance.

    Forces instance recreation, mainly for test scenarios.
    Note: make sure old instance resources have been cleaned up before using this.
    """
    global _global_scene_graph
    SemanticSceneGraph._instance = None
    _global_scene_graph = None
