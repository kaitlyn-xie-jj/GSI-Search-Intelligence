# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple, Optional, Set, Any, Union
import logging
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.utils.system.logging_utils import dlog

logger = logging.getLogger(__name__)

class WorldModelManager:
    """
    Local world model.
    """
    
    def __init__(self, scene_graph: Optional[AbstractSceneGraph] = None, logger=None):
        """
        Initialize the world model manager.
        
        Args:
            scene_graph: Scene graph, using AbstractSceneGraph.
            logger: Logger.
        """
        self.scene_graph: Optional[AbstractSceneGraph] = scene_graph
        self.logger = logger or logging.getLogger(__name__)
        
        # Knowledge scope setting.
        self.knowledge_scope: str = 'global'  # 'global' or 'local'
        
        # Local knowledge storage.
        self.known_nodes: List[Dict] = []
        self.known_edges: List[Dict] = []
        
        # Explored areas and search records.
        self.explored_locations: Set[str] = set()
        self.searched_areas: Dict[str, Set[str]] = {}  # location -> set of searched targets

        # Environment constraint records, such as blocked critical paths to destinations.
        self.blocked_destinations: Set[str] = set()
    
    def set_knowledge_scope(self, scope: str) -> None:
        """
        Set the knowledge scope.
        
        Args:
            scope: 'global' or 'local'.
        """
        if scope not in ['global', 'local']:
            raise ValueError("Knowledge scope must be either 'global' or 'local'")
        
        self.knowledge_scope = scope
        dlog(f"Knowledge scope set to: {scope}", logger=self.logger)
        
        # Initialize local knowledge when switching to local scope.
        if scope == 'local' and not self.known_nodes:
            self.initialize_local_knowledge()

    # ===================== Node CRUD interface =====================

    def add_node(self, node_data: Dict[str, Any]) -> bool:
        """
        Add a node to the world graph.
        
        Args:
            node_data: Node data dictionary. Must contain an 'id' field.
            
        Returns:
            bool: Whether the add succeeded.
        """
        if 'id' not in node_data:
            dlog("Cannot add node without 'id' field", logger=self.logger, level='warning')
            return False
        
        node_id = node_data['id']
        
        # Check whether the node already exists.
        if any(str(n.get('id')) == str(node_id) for n in self.known_nodes):
            dlog(f"Node with id {node_id} already exists", logger=self.logger, level='warning')
            return False
        
        # Add a node copy.
        self.known_nodes.append(node_data.copy())
        dlog(f"Added node {node_id} to world graph", logger=self.logger)
        return True

    def remove_node(self, node_id: Union[int, str]) -> bool:
        """
        Remove a node from the world graph.
        
        Args:
            node_id: Node ID to remove.
            
        Returns:
            bool: Whether the removal succeeded.
        """
        node_id_str = str(node_id)
        
        # Find and remove the node.
        for i, node in enumerate(self.known_nodes):
            if str(node.get('id')) == node_id_str:
                self.known_nodes.pop(i)
                
                # Also remove edges related to this node.
                self.known_edges = [
                    edge for edge in self.known_edges
                    if str(edge.get('source')) != node_id_str and str(edge.get('target')) != node_id_str
                ]
                
                dlog(f"Removed node {node_id} from world graph", logger=self.logger)
                return True
        
        dlog(f"Node {node_id} not found in world graph", logger=self.logger, level='warning')
        return False

    def update_node(self, node_id: Union[int, str], updates: Dict[str, Any]) -> bool:
        """
        Update node properties in the world graph.
        
        Args:
            node_id: Node ID to update.
            updates: Properties to update.
            
        Returns:
            bool: Whether the update succeeded.
        """
        node_id_str = str(node_id)
        
        for i, node in enumerate(self.known_nodes):
            if str(node.get('id')) == node_id_str:
                # Deep-merge updates.
                updated_node = node.copy()
                self._deep_update(updated_node, updates)
                self.known_nodes[i] = updated_node
                dlog(f"Updated node {node_id} in world graph", logger=self.logger)
                return True
        
        dlog(f"Node {node_id} not found for update", logger=self.logger, level='warning')
        return False

    def _deep_update(self, target: Dict[str, Any], source: Dict[str, Any]) -> None:
        """
        Deep-update a dictionary by recursively merging values.
        
        Args:
            target: Target dictionary.
            source: Source dictionary.
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

    # ===================== Node query interface =====================

    def get_node_by_id(self, node_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Query a node by ID.
        
        Args:
            node_id: Node ID.
            
        Returns:
            A copy of the node data, or None if it does not exist.
        """
        node_id_str = str(node_id)
        
        for node in self.known_nodes:
            if str(node.get('id')) == node_id_str:
                return node.copy()
        
        return None

    def get_node_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """
        Query a node by label.
        
        Args:
            label: The node's label property.
            
        Returns:
            A copy of the node data, or None if it does not exist.
        """
        for node in self.known_nodes:
            if node.get('properties', {}).get('label') == label:
                return node.copy()
        
        return None

    # ===================== Edge CRUD interface =====================

    def add_edge(self, edge_data: Dict[str, Any]) -> bool:
        """
        Add an edge to the world graph.
        
        Args:
            edge_data: Edge data dictionary. Must contain 'source' and 'target' fields.
            
        Returns:
            bool: Whether the add succeeded.
        """
        source_id = edge_data.get('source')
        target_id = edge_data.get('target')
        
        if source_id is None or target_id is None:
            dlog("Cannot add edge without 'source' and 'target' fields", logger=self.logger, level='warning')
            return False
        
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        # Verify that the source and target nodes exist.
        source_exists = any(str(n.get('id')) == source_id_str for n in self.known_nodes)
        target_exists = any(str(n.get('id')) == target_id_str for n in self.known_nodes)
        
        if not source_exists:
            dlog(f"Source node {source_id} does not exist in world graph", logger=self.logger, level='error')
            return False
        
        if not target_exists:
            dlog(f"Target node {target_id} does not exist in world graph", logger=self.logger, level='error')
            return False
        
        # Check whether the edge already exists.
        for edge in self.known_edges:
            if str(edge.get('source')) == source_id_str and str(edge.get('target')) == target_id_str:
                dlog(f"Edge {source_id} -> {target_id} already exists", logger=self.logger, level='warning')
                return False
        
        # Add an edge copy.
        self.known_edges.append(edge_data.copy())
        dlog(f"Added edge {source_id} -> {target_id} to world graph", logger=self.logger)
        return True

    def remove_edge(self, source_id: Union[int, str], target_id: Union[int, str]) -> bool:
        """
        Remove an edge from the world graph.
        
        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            
        Returns:
            bool: Whether the removal succeeded.
        """
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        for i, edge in enumerate(self.known_edges):
            if str(edge.get('source')) == source_id_str and str(edge.get('target')) == target_id_str:
                self.known_edges.pop(i)
                dlog(f"Removed edge {source_id} -> {target_id} from world graph", logger=self.logger)
                return True
        
        dlog(f"Edge {source_id} -> {target_id} not found in world graph", logger=self.logger, level='warning')
        return False

    def update_edge(self, source_id: Union[int, str], target_id: Union[int, str], 
                    updates: Dict[str, Any]) -> bool:
        """
        Update edge properties in the world graph.
        
        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            updates: Properties to update.
            
        Returns:
            bool: Whether the update succeeded.
        """
        source_id_str = str(source_id)
        target_id_str = str(target_id)
        
        for i, edge in enumerate(self.known_edges):
            if str(edge.get('source')) == source_id_str and str(edge.get('target')) == target_id_str:
                # Deep-merge updates.
                updated_edge = edge.copy()
                self._deep_update(updated_edge, updates)
                self.known_edges[i] = updated_edge
                dlog(f"Updated edge {source_id} -> {target_id} in world graph", logger=self.logger)
                return True
        
        dlog(f"Edge {source_id} -> {target_id} not found for update", logger=self.logger, level='warning')
        return False

    # ===================== Edge query interface =====================

    def get_edges_by_source(self, source_id: Union[int, str]) -> List[Dict[str, Any]]:
        """
        Query edges by source node ID.
        
        Args:
            source_id: Source node ID.
            
        Returns:
            Edges starting from the specified node, returned as copies.
        """
        source_id_str = str(source_id)
        
        return [
            edge.copy() for edge in self.known_edges
            if str(edge.get('source')) == source_id_str
        ]

    def get_edges_by_target(self, target_id: Union[int, str]) -> List[Dict[str, Any]]:
        """
        Query edges by target node ID.
        
        Args:
            target_id: Target node ID.
            
        Returns:
            Edges pointing to the specified node, returned as copies.
        """
        target_id_str = str(target_id)
        
        return [
            edge.copy() for edge in self.known_edges
            if str(edge.get('target')) == target_id_str
        ]

    # ===================== Initialization interface =====================

    def initialize_from_scene_graph(self, scene_graph) -> None:
        """
        Initialize the world graph from the scene graph, extracting robots and related nodes.
        
        This clearer initialization interface extracts robot nodes and their
        adjacent nodes from the scene graph as the initial world graph.
        
        Args:
            scene_graph: Scene graph instance, using SemanticSceneGraph.
        """
        if scene_graph is None:
            dlog("Cannot initialize world graph without scene graph", logger=self.logger, level='error')
            return
        
        # Save the scene graph reference.
        self.scene_graph = scene_graph
        
        # Clear existing data.
        self.known_nodes = []
        self.known_edges = []
        self.explored_locations = set()
        
        known_node_ids = set()
        known_nodes_map = {}
        
        # 1) Extract all robot nodes.
        all_robots = [
            n for n in scene_graph._nodes 
            if n.get('properties', {}).get('category') == 'robot'
        ]
        dlog(f"Found {len(all_robots)} robots for world graph initialization", logger=self.logger)
        
        # 2) Robots and adjacent nodes, such as bases and locations.
        for robot in all_robots:
            robot_id = robot['id']
            known_node_ids.add(robot_id)
            
            # Get the robot's neighbor nodes.
            neighbor_ids = scene_graph.get_neighbors(str(robot_id))
            if neighbor_ids:
                for neighbor_id in neighbor_ids:
                    try:
                        neighbor_id_int = int(neighbor_id)
                    except (ValueError, TypeError):
                        neighbor_id_int = neighbor_id
                    
                    if neighbor_id_int not in known_node_ids:
                        known_node_ids.add(neighbor_id_int)
                        
                        # Get neighbor node information.
                        neighbor_node = scene_graph.get_node_by_id(neighbor_id_int)
                        if neighbor_node:
                            neighbor_label = neighbor_node.get('properties', {}).get('label')
                            if neighbor_label:
                                self.explored_locations.add(neighbor_label)
                            
                            # Get the neighbor's neighbors through the located_in relation.
                            second_neighbors = scene_graph.get_neighbors_by_relation(neighbor_id, 'located_in')
                            if second_neighbors:
                                for second_id in second_neighbors:
                                    try:
                                        second_id_int = int(second_id)
                                    except (ValueError, TypeError):
                                        second_id_int = second_id
                                    
                                    if second_id_int not in known_node_ids:
                                        known_node_ids.add(second_id_int)
        
        # 3) Include district nodes as known.
        all_districts = [
            n for n in scene_graph._nodes 
            if n.get('properties', {}).get('category') == 'district'
        ]
        for district in all_districts:
            district_label = district.get('properties', {}).get('label')
            if district_label:
                self.explored_locations.add(district_label)
                known_node_ids.add(district['id'])
        
        # 3.5) Pre-stored assembly components in bases such as warehouses are initially known.
        self._collect_pre_known_assembly_components(known_node_ids)

        # 4) Build the known node list.
        for node_id in known_node_ids:
            node = scene_graph.get_node_by_id(node_id)
            if node:
                known_nodes_map[node['id']] = node.copy()
        self.known_nodes = list(known_nodes_map.values())
        
        # 5) Extract edges between known nodes.
        seen_edges = set()
        for node_id in known_node_ids:
            out_edges = scene_graph.get_edges_by_source(str(node_id))
            for edge in out_edges:
                # Check whether the target node is known.
                target_id = edge.get('target')
                try:
                    target_id_check = int(target_id)
                except (ValueError, TypeError):
                    target_id_check = target_id
                
                if target_id_check in known_node_ids or target_id in known_node_ids:
                    edge_key = (str(edge.get('source')), str(edge.get('target')))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        self.known_edges.append(edge.copy())
        
        dlog(
            f"World graph initialized: {len(self.known_nodes)} nodes, "
            f"{len(self.known_edges)} edges, {len(self.explored_locations)} explored locations",
            logger=self.logger
        )
    
    def get_current_world_state(self) -> Tuple[List[Dict], List[Dict]]:
        """
        Get the current world state based on knowledge scope.
        
        Returns:
            (nodes, edges) tuple.
        """
        if self.knowledge_scope == 'global':
            if self.scene_graph:
                nodes = self.scene_graph._nodes
                edges = self.scene_graph._edges
                dlog(f"Returning global knowledge: {len(nodes)} nodes, {len(edges)} edges", logger=self.logger)
                return nodes, edges
            else:
                dlog("No task context available for global knowledge", logger=self.logger, level='error')
                return [], []
        else:
            dlog(f"Returning local knowledge: {len(self.known_nodes)} nodes, {len(self.known_edges)} edges", logger=self.logger)
            return self.known_nodes, self.known_edges
    
    def initialize_local_knowledge(self) -> None:
        """
        Initialize local knowledge, starting from robot bases and adjacent areas.
        """
        if not self.scene_graph:
            dlog("Cannot initialize local knowledge without task context", logger=self.logger, level='error')
            return
        
        known_node_ids = set()
        known_nodes_map = {}
        
        # 1) All robots.
        all_robots = [
            n for n in self.scene_graph._nodes 
            if n.get('properties', {}).get('category') == 'robot'
        ]
        dlog(f"Found {len(all_robots)} robots for initial local knowledge", logger=self.logger)
        
        # 2) Robots, their bases, and adjacent nodes.
        for robot in all_robots:
            known_node_ids.add(robot['id'])
            base_ids = self.scene_graph.get_neighbors(str(robot['id']))
            if base_ids:
                for base_id in base_ids:
                    if int(base_id) not in known_node_ids:
                        known_node_ids.add(int(base_id))
                        base_node = self.scene_graph.get_node_by_id(int(base_id))
                        if base_node:
                            base_label = base_node.get('properties', {}).get('label')
                            if base_label:
                                self.explored_locations.add(base_label)
                        neighbor_ids = self.scene_graph.get_neighbors_by_relation(base_id, 'located_in')
                        if neighbor_ids:
                            for neighbor_id in neighbor_ids:
                                if int(neighbor_id) not in known_node_ids:
                                    known_node_ids.add(int(neighbor_id))

        # 2.5) Pre-stored assembly components in bases such as warehouses are initially known.
        self._collect_pre_known_assembly_components(known_node_ids)

        # 3) Include district nodes as known.
        all_districts = [
            n for n in self.scene_graph._nodes 
            if n.get('properties', {}).get('category') == 'district'
        ]
        for district in all_districts:
            district_label = district.get('properties', {}).get('label')
            if district_label and district_label not in self.explored_locations:
                self.explored_locations.add(district_label)
                known_node_ids.add(district['id'])
        
        # 4) Build the known node list.
        for node_id in known_node_ids:
            node = self.scene_graph.get_node_by_id(node_id)
            if node:
                known_nodes_map[node['id']] = node
        self.known_nodes = list(known_nodes_map.values())
        
        # 5) Known edges.
        self.known_edges = []
        seen_edges = set()
        for node_id in known_node_ids:
            out_edges = self.scene_graph.get_edges_by_source(str(node_id))
            for edge in out_edges:
                target_id = edge.get('target')
                try:
                    target_id_check = int(target_id)
                except (ValueError, TypeError):
                    target_id_check = target_id
                if target_id_check in known_node_ids or target_id in known_node_ids:
                    edge_key = (str(edge.get('source')), str(edge.get('target')))
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        self.known_edges.append(edge)
        
        dlog(
            f"Initial local knowledge: {len(self.known_nodes)} nodes, "
            f"{len(self.known_edges)} edges, {len(self.explored_locations)} explored locations",
            logger=self.logger
        )

    def _collect_pre_known_assembly_components(self, known_node_ids: Set) -> None:
        """
        Include pre-stored assembly_component nodes in the initial known nodes.

        These components are stored in warehouses, which have already entered
        known_node_ids through bases, so robots know them at task start.
        """
        if not self.scene_graph:
            return

        for node in self.scene_graph._nodes:
            props = node.get('properties', {}) or {}
            if (
                props.get('category') == 'prop'
                and props.get('type') == 'assembly_component'
                and node['id'] not in known_node_ids
            ):
                known_node_ids.add(node['id'])

    # ===================== New situation event updates =====================

    def update_from_events(self, event_type: str, details: Dict) -> None:
        """
        Update local knowledge from an event.
        """
        if self.knowledge_scope != 'local':
            return

        update_methods = {
            # Environment and path.
            'CRITICAL_PATH_BROKEN': self._upd_critical_path_broken,
            'AREA_RESTRICTED': self._upd_area_restricted,  # Process AREA_RESTRICTED
            'AREA_LOW_VISIBILITY': self._upd_area_low_visibility,  # Process AREA_LOW_VISIBILITY
            'AREA_IS_DARK': self._upd_area_is_dark,  # Process AREA_IS_DARK
            'AREA_STRONG_WIND': self._upd_area_strong_wind,  # Process AREA_STRONG_WIND
            'PATH_CONGESTED_VEHICLE': self._upd_path_congested_vehicle,  # Process PATH_CONGESTED_VEHICLE
            'PATH_CONGESTED_CROWD': self._upd_path_congested_crowd,  # Process PATH_CONGESTED_CROWD
            'TARGET_OBSTRUCTED': self._upd_target_obstructed,  # Process TARGET_OBSTRUCTED

            # Target and carrier.
            'TARGET_NOT_FOUND': self._upd_target_not_found,
            'CARRIER_NOT_FOUND': self._upd_carrier_not_found,
            'CARRIER_LOCATION_MISMATCH': self._upd_carrier_location_mismatch,
            'SURFACE_OBJECT_MISSING': self._upd_surface_object_missing,  # SURFACE_OBJECT_MISSING
            'SURFACE_OBJECT_LOCATION_MISMATCH': self._upd_surface_object_location_mismatch,  # SURFACE_OBJECT_LOCATION_MISMATCH
            'TARGET_DISAPPEARED': self._upd_target_disappeared,

            # Robot.
            'ROBOT_FAULT': self._upd_robot_fault,
            'ROBOT_BATTERY_LOW': self._upd_robot_battery_low,
            'ROBOT_COMM_JAMMED': self._upd_robot_comm_jammed,

            # System and definitions.
            'SKILL_NOT_FOUND': self._upd_noop,
            'ROBOT_NOT_APPLICABLE': self._upd_noop,

            # Compatibility events.
            'DISCOVERY': self._add_discovered_entity,
            'LOCATION_EXPLORED': self._mark_location_explored,
        }
        
        updater = update_methods.get(event_type)
        if updater:
            updater(details or {})
            dlog(
                f"Updated local knowledge for event '{event_type}'. "
                f"Now have {len(self.known_nodes)} nodes, {len(self.known_edges)} edges",
                logger=self.logger
            )

    # ===================== Specific update methods =====================

    def _upd_critical_path_broken(self, details: Dict) -> None:
        pass

    def _upd_area_restricted(self, details: Dict) -> None:
        pass

    def _upd_area_low_visibility(self, details: Dict) -> None:
        pass

    def _upd_area_is_dark(self, details: Dict) -> None:
        pass

    def _upd_area_strong_wind(self, details: Dict) -> None:
        pass

    def _upd_path_congested_vehicle(self, details: Dict) -> None:
        pass

    def _upd_path_congested_crowd(self, details: Dict) -> None:
        pass

    def _upd_target_obstructed(self, details: Dict) -> None:
        pass

    def _upd_target_not_found(self, details: Dict) -> None:
        pass

    def _upd_target_disappeared(self, details: Dict) -> None:
        pass

    def _upd_carrier_not_found(self, details: Dict) -> None:
        pass

    def _upd_carrier_location_mismatch(self, details: Dict) -> None:
        pass

    def _upd_surface_object_missing(self, details: Dict) -> None:
        pass

    def _upd_surface_object_location_mismatch(self, details: Dict) -> None:
        pass

    def _upd_robot_fault(self, details: Dict) -> None:
        """Robot fault: set robot status to error."""
        rnode = self._resolve_node_via_details(details, prefer='robot')
        if rnode:
            props = rnode.setdefault('properties', {})
            props['status'] = 'error'

    def _upd_robot_battery_low(self, details: Dict) -> None:
        """Robot battery low: record battery level and degraded status."""
        rnode = self._resolve_node_via_details(details, prefer='robot')
        if rnode:
            props = rnode.setdefault('properties', {})
            props['status'] = props.get('status') or 'degraded'
            level = props.get('battery_level') or details.get('battery_level') or details.get('new_battery_level') or 5
            props['battery_level'] = level

    def _upd_robot_comm_jammed(self, details: Dict) -> None:
        """Robot communication jammed: record communication status without changing availability."""
        rnode = self._resolve_node_via_details(details, prefer='robot')
        if rnode:
            props = rnode.setdefault('properties', {})
            if props.get('comm') != 'jammed':
                props['comm'] = 'jammed'

    def _upd_noop(self, details: Dict) -> None:
        """Placeholder implementation for events that do not update local knowledge."""
        pass

    def _add_discovered_entity(self, details: Dict) -> None:
        """Add a newly discovered entity."""
        new_entity = details.get('discovered_entity')
        if new_entity:
            entity_id = new_entity.get('id')
            if not any(n.get('id') == entity_id for n in self.known_nodes):
                self.known_nodes.append(new_entity)

    def _mark_location_explored(self, details: Dict) -> None:
        """Mark a location as explored."""
        location_label = details.get('location_label')
        if location_label:
            self.explored_locations.add(location_label)

    # ===================== Execution result updates =====================

    def update_from_outcomes(self, outcomes: List[Dict[str, Any]]) -> None:
        """Update the world model from outcomes.
        
        Iterate through outcomes and dispatch by type:
        - KNOWLEDGE_ACQUIRED: process entities found by search.
        - Other types: process entity state changes.
        
        Args:
            outcomes: Outcome list. Each outcome contains type, data, and meta fields.
        """
        if not outcomes:
            return
        
        for outcome in outcomes:
            if not isinstance(outcome, dict):
                continue
            
            outcome_type = outcome.get("type", "")
            data = outcome.get("data", {})
            meta = outcome.get("meta", {})
            skill = (meta.get("skill") or "").lower().strip()
            
            # Dispatch by type.
            if outcome_type == "KNOWLEDGE_ACQUIRED":
                if skill == "search":
                    self._process_search_outcome(outcome)
                elif skill in ["place", "drop"]:
                    self._process_entity_state_change(outcome)
                elif skill == "follow":
                    self._process_follow_outcome(outcome)
            
            # Process position updates, if any.
            if "entities" in data:
                self._update_entities_from_outcome(data["entities"])
        
        dlog(
            f"Updated world model from {len(outcomes)} outcomes. "
            f"Now have {len(self.known_nodes)} nodes",
            logger=self.logger
        )

    def _process_search_outcome(self, outcome: Dict[str, Any]) -> None:
        """Process a search skill outcome.
        
        Add discovered entities to known_nodes and add edges from location properties.
        
        Args:
            outcome: Search skill outcome.
        """
        data = outcome.get("data", {})
        meta = outcome.get("meta", {})
        
        # Check whether the target was found successfully.
        found = data.get("found", False)
        if not found:
            return
        
        # Extract discovered entities.
        entities = data.get("entities", [])
        if not entities:
            return
        
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            
            entity_id = entity.get("id")
            if entity_id is None:
                continue
            
            # Check whether the node already exists.
            existing = self._find_local_node_by_id(entity_id)
            if existing:
                # Update the existing node's properties.
                self._deep_update(existing, entity)
                dlog(f"Updated existing node {entity_id} from search outcome", logger=self.logger)
            else:
                # Add a new node.
                self.known_nodes.append(entity.copy())
                dlog(f"Added new node {entity_id} from search outcome", logger=self.logger)
            
            # Add edges from the location property.
            self._ensure_location_edge(entity)
        
        # Mark the searched area as explored.
        area_token = data.get("area_token") or data.get("area_searched") or data.get("area")
        if area_token:
            self.explored_locations.add(str(area_token))

    def _ensure_location_edge(self, entity: Dict[str, Any]) -> None:
        """
        If an entity has location info, such as {'label': 'Hotel-1', ...},
        ensure known_edges contains a located_at edge from the entity to that
        location node. Remove any old location edge first.
        """
        entity_id = entity.get("id")
        props = entity.get("properties", {}) or {}
        location_info = props.get("location")
        if not isinstance(location_info, dict) or not location_info.get("label"):
            return

        location_label = location_info["label"]

        # Find the location node.
        location_node = self._find_local_node_by_label(location_label)
        if not location_node:
            return

        location_id = location_node.get("id")
        entity_id_str = str(entity_id)
        location_id_str = str(location_id)

        # Remove existing location edges for this entity, then add the new edge.
        self.known_edges = [
            edge for edge in self.known_edges
            if not (
                str(edge.get("source")) == entity_id_str
                and edge.get("type") in ("located_at", "stored_at")
            )
        ]

        # Add the located_at edge.
        self.known_edges.append({
            "source": entity_id,
            "target": location_id,
            "type": "located_at",
        })
        dlog(f"Added edge {entity_id} -> {location_id} (located_at) from search outcome", logger=self.logger)

    def _process_entity_state_change(self, outcome: Dict[str, Any]) -> None:
        """Process an entity state change.
        
        Update node position or properties, such as is_carried.
        
        Args:
            outcome: Outcome containing a state change.
        """
        data = outcome.get("data", {})
        meta = outcome.get("meta", {})
        skill = (meta.get("skill") or "").lower().strip()
        
        # Get the target entity ID.
        target_id = data.get("object_id") or data.get("target_id")
        if target_id is None:
            return
        
        # Find the node.
        node = self._find_local_node_by_id(target_id)
        if not node:
            # If the node does not exist, try getting it from the scene graph.
            if self.scene_graph:
                node = self.scene_graph.get_node_by_id(target_id)
                if node:
                    self.known_nodes.append(node.copy())
                    node = self._find_local_node_by_id(target_id)
        
        if not node:
            return
        
        props = node.setdefault("properties", {})
        
        # Update state by skill type.
        if skill == "place":
            props["is_carried"] = True
            # Update carrier information.
            robot_label = meta.get("robot_label")
            if robot_label:
                props["carried_by"] = robot_label
        elif skill in ["place", "drop"]:
            props["is_carried"] = False
            props.pop("carried_by", None)
            # Update position, if any.
            if "entities" in data:
                for entity in data["entities"]:
                    if str(entity.get("id")) == str(target_id):
                        if "shape" in entity:
                            node["shape"] = entity["shape"]
                        break
        
        # Update location_label, if any.
        if "location_label" in data:
            props["location_label"] = data["location_label"]
        
        dlog(f"Updated entity {target_id} state: skill={skill}", logger=self.logger)

    def _process_follow_outcome(self, outcome: Dict[str, Any]) -> None:
        """Process a follow skill outcome.
        
        Update the target entity's position.
        
        Args:
            outcome: Follow skill outcome.
        """
        data = outcome.get("data", {})
        
        # Get the target entity ID.
        target_id = data.get("target_id") or data.get("object_id")
        if target_id is None:
            return
        
        # Update entity position, if any.
        entities = data.get("entities", [])
        for entity in entities:
            if str(entity.get("id")) == str(target_id):
                node = self._find_local_node_by_id(target_id)
                if node and "shape" in entity:
                    node["shape"] = entity["shape"]
                    dlog(f"Updated position for followed target {target_id}", logger=self.logger)
                break

    def _update_entities_from_outcome(self, entities: List[Dict[str, Any]]) -> None:
        """Update nodes from an outcome's entities field.
        
        Args:
            entities: Entity list.
        """
        for entity in entities:
            if not isinstance(entity, dict):
                continue
            
            entity_id = entity.get("id")
            if entity_id is None:
                continue
            
            node = self._find_local_node_by_id(entity_id)
            if node:
                # Update the existing node.
                if "shape" in entity:
                    node["shape"] = entity["shape"]
                if "properties" in entity:
                    props = node.setdefault("properties", {})
                    self._deep_update(props, entity["properties"])

    # ===================== Shared utility methods =====================

    def is_location_explored(self, location_label: str) -> bool:
        """Check whether a location has been explored."""
        return location_label in self.explored_locations
    
    def was_target_searched_at(self, target_label: str, location_label: str) -> bool:
        """Check whether a target was searched at a specific location."""
        return (location_label in self.searched_areas and 
                target_label in self.searched_areas[location_label])
    
    def get_unexplored_locations(self) -> List[str]:
        """
        Get unexplored locations. This is only meaningful in local mode.
        """
        if self.knowledge_scope == 'global':
            return []
        
        all_location_labels = set()
        for node in self.known_nodes:
            if node.get('properties', {}).get('category') == 'building':
                label = node.get('properties', {}).get('label')
                if label:
                    all_location_labels.add(label)
        
        unexplored = list(all_location_labels - self.explored_locations)
        return unexplored
    
    def merge_discovered_knowledge(self, new_nodes: List[Dict], new_edges: List[Dict]) -> None:
        """
        Merge newly discovered knowledge into the local knowledge base.
        """
        if self.knowledge_scope != 'local':
            return
        
        # Merge nodes.
        existing_ids = {n['id'] for n in self.known_nodes}
        new_nodes_to_add = [node for node in new_nodes if node['id'] not in existing_ids]
        if new_nodes_to_add:
            self.known_nodes.extend(new_nodes_to_add)

        # Merge edges.
        existing_edge_keys = {(e['source'], e['target']) for e in self.known_edges}
        new_edges_to_add = [edge for edge in new_edges if (edge['source'], edge['target']) not in existing_edge_keys]
        if new_edges_to_add:
            self.known_edges.extend(new_edges_to_add)

        if new_nodes_to_add or new_edges_to_add:
            dlog(
                f"Merged knowledge: added {len(new_nodes_to_add)} nodes and {len(new_edges_to_add)} edges",
                logger=self.logger
            )

    # ===================== Internal parsing and helpers =====================

    def _resolve_node_via_details(self, details: Dict[str, Any], prefer: str) -> Optional[Dict]:
        """
        Parse a node reference in local knowledge from details.
        prefer: 'robot' | 'target' | 'carrier'
        """
        # 1) Prefer ID.
        nid = None
        if prefer == 'robot':
            nid = (details.get('robot') or {}).get('id') or details.get('robot_id')
        elif prefer == 'target':
            oid = (details.get('object') or {}).get('id') or details.get('target_id')
            nid = oid
        elif prefer == 'carrier':
            nid = (details.get('carrier') or {}).get('id') or details.get('carrier_id')
        if nid is not None:
            try:
                nid_int = int(nid)
            except Exception:
                nid_int = nid
            node = self._find_local_node_by_id(nid_int)
            if node:
                return node

        # 2) Then try label.
        label = None
        if prefer == 'robot':
            label = (details.get('robot') or {}).get('label') or details.get('robot_label')
        elif prefer == 'target':
            label = details.get('target_label') or (details.get('object') or {}).get('label')
        elif prefer == 'carrier':
            label = (details.get('carrier') or {}).get('label') or details.get('carrier_label')
        if label:
            node = self._find_local_node_by_label(label)
            if node:
                return node

        # 3) Fallback: if not found locally and scene_graph exists, try global nodes and add them locally.
        if self.scene_graph:
            # Try again by id.
            if nid is not None:
                try:
                    nid_int = int(nid)
                except Exception:
                    nid_int = nid
                node = self.scene_graph.get_node_by_id(nid_int)
                if node:
                    self.known_nodes.append(node)
                    return node
            # Backfill by label.
            if label:
                for n in self.scene_graph._nodes:
                    if n.get('properties', {}).get('label') == label:
                        self.known_nodes.append(n)
                        return n
        return None

    def _find_local_node_by_id(self, nid: Any) -> Optional[Dict]:
        # Convert nid to string because graph database IDs are usually strings.
        try:
            nid_str = str(nid)
        except:
            return None
            
        for n in self.known_nodes:
            if str(n.get('id')) == nid_str:
                return n
        return None

    def _find_local_node_by_label(self, label: str) -> Optional[Dict]:
        for n in self.known_nodes:
            if n.get('properties', {}).get('label') == label:
                return n
        return None
