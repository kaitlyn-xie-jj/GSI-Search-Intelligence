# -*- coding: utf-8 -*-
"""
Abstract Scene Graph - abstract base class for scene graphs

Defines the unified scene graph interface that all platforms must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Literal


class AbstractSceneGraph(ABC):
    """Abstract base class for scene graphs.

    This abstract class defines the standard scene graph interface, including:
    - Goal query interface.
    - Node query interface.
    - Edge query interface.
    - Entity query interface.
    - Position query interface.
    - Mapping interface.
    - Path query interface.
    - Data modification interface, both sync and async.
    - Utility methods.

    All concrete platform implementations, such as SemanticSceneGraph and
    UnrealSceneGraph, must inherit from this class and implement all abstract methods.
    """
    
    # ==================== Goal Query Interface ====================
    
    @abstractmethod
    def get_goal(self) -> Optional[Dict[str, Any]]:
        """Get goal data.
        
        Returns:
            Goal data dictionary, or None if no goal is set.
        """
        ...
    
    @abstractmethod
    def get_goal_description(self) -> Optional[str]:
        """Get the goal description.
        
        Returns:
            Goal description string.
        """
        ...
    
    @abstractmethod
    def evaluate_goal(self) -> bool:
        """Evaluate whether the goal is achieved.
        
        Returns:
            bool: Whether the goal is achieved.
        """
        ...

    # ==================== Node Query Interface ====================
    
    @abstractmethod
    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes.
        
        Returns:
            List of copies of all nodes.
        """
        ...
    
    @abstractmethod
    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node by ID.
        
        Args:
            node_id: Node ID.
            
        Returns:
            Copy of node data, or None if it does not exist.
        """
        ...
    
    @abstractmethod
    def get_node_by_label(self, label: str) -> Optional[Dict[str, Any]]:
        """Get a node by label.
        
        Args:
            label: Node label.
            
        Returns:
            Copy of node data, or None if it does not exist.
        """
        ...
    
    @abstractmethod
    def get_nodes_by_category(self, category: str) -> List[Dict[str, Any]]:
        """Get nodes by category.
        
        Args:
            category: Node category, such as robot, prop, building, or goal.
            
        Returns:
            List of nodes matching the category.
        """
        ...
    
    @abstractmethod
    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get nodes by type.
        
        Args:
            node_type: Node type.
            
        Returns:
            List of nodes matching the type.
        """
        ...
    
    @abstractmethod
    def search_nodes(self, **criteria) -> List[Dict[str, Any]]:
        """Search nodes by criteria.
        
        Args:
            **criteria: Search criteria, supporting nested property queries.
            
        Returns:
            List of nodes matching the criteria.
        """
        ...
    
    @abstractmethod
    def get_node_info_by_label(self, label: str) -> Optional[Dict[str, str]]:
        """Return a node's category and type for the given label.
        
        Args:
            label: Node label name.
            
        Returns:
            Dictionary containing the node category and type, or None if not found.
        """
        ...

    # ==================== Edge Query Interface ====================
    
    @abstractmethod
    def get_all_edges(self) -> List[Dict[str, Any]]:
        """Get all edges.
        
        Returns:
            List of copies of all edges.
        """
        ...
    
    @abstractmethod
    def get_edges_by_source(self, source_id: str) -> List[Dict[str, Any]]:
        """Get all edges from the specified source node.
        
        Args:
            source_id: Source node ID.
            
        Returns:
            List of edges from the specified node.
        """
        ...
    
    @abstractmethod
    def get_edges_by_target(self, target_id: str) -> List[Dict[str, Any]]:
        """Get all edges to the specified target node.
        
        Args:
            target_id: Target node ID.
            
        Returns:
            List of edges pointing to the specified node.
        """
        ...
    
    @abstractmethod
    def get_edge(self, source_id: str, target_id: str) -> Optional[Dict[str, Any]]:
        """Get the specified edge.
        
        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            
        Returns:
            Copy of edge data, or None if it does not exist.
        """
        ...
    
    @abstractmethod
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[str]:
        """Get a node's neighbor node ID list.
        
        Args:
            node_id: Node ID.
            direction: Direction: 'in', 'out', or 'both'.
            
        Returns:
            Neighbor node ID list.
        """
        ...
    
    @abstractmethod
    def get_neighbors_by_relation(
        self, node_id: Union[int, str], relation: str
    ) -> List[Union[int, str]]:
        """Find directly related neighbor node IDs by node ID and relation type.
        
        Args:
            node_id: Unique ID of the node to query.
            relation: Edge relation type to filter by.
            
        Returns:
            List of all matching neighbor node IDs.
        """
        ...
    
    @abstractmethod
    def get_edges_from_node(
        self, node_id: Union[int, str], edge_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get edges from the specified node, optionally filtered by type.
        
        Args:
            node_id: Source node ID.
            edge_type: Edge type. If specified, only edges of this type are returned.
            
        Returns:
            List of edges from the specified node.
        """
        ...
    
    @abstractmethod
    def get_edges_by_type(self, edge_type: str) -> List[Dict[str, Any]]:
        """Get all edges of the specified type.
        
        Args:
            edge_type: Edge type.
            
        Returns:
            List of edges of the specified type.
        """
        ...
    
    @abstractmethod
    def has_edge_of_type(
        self, source_id: Union[int, str], target_id: Union[int, str], edge_type: str
    ) -> bool:
        """Check whether an edge of the specified type exists between two nodes.
        
        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            edge_type: Edge type.
            
        Returns:
            Whether an edge of this type exists.
        """
        ...

    # ==================== Entity Query Interface ====================
    
    @abstractmethod
    def get_robot(self, robot_id: str) -> Optional[Dict[str, Any]]:
        """Get a robot node by ID.
        
        Args:
            robot_id: Robot ID.
            
        Returns:
            Copy of robot node data, or None if it does not exist.
        """
        ...
    
    @abstractmethod
    def get_prop(self, prop_id: str = None) -> Optional[Dict[str, Any]]:
        """Get a prop node by ID.
        
        Args:
            prop_id: Prop ID. If None, all props are returned.
            
        Returns:
            Copy of prop node data, or None if it does not exist.
        """
        ...
    
    @abstractmethod
    def get_building(self, building_id: str) -> Optional[Dict[str, Any]]:
        """Get a building node by ID.
        
        Args:
            building_id: Building ID.
            
        Returns:
            Copy of building node data, or None if it does not exist.
        """
        ...
    
    @abstractmethod
    def get_all_robots(self) -> List[Dict[str, Any]]:
        """Get all robot nodes.
        
        Returns:
            List of copies of robot node data.
        """
        ...
    
    @abstractmethod
    def get_all_props(self) -> List[Dict[str, Any]]:
        """Get all prop nodes.
        
        Returns:
            List of copies of prop node data.
        """
        ...
    
    @abstractmethod
    def get_all_buildings(self) -> List[Dict[str, Any]]:
        """Get all building nodes.
        
        Returns:
            List of copies of building node data.
        """
        ...
    
    @abstractmethod
    def get_all_trans_facilitys(self) -> List[Dict[str, Any]]:
        """Get all infrastructure nodes.
        
        Returns:
            List of copies of infrastructure node data.
        """
        ...
    
    @abstractmethod
    def get_goal_node(self, goal_id: str) -> Optional[Dict[str, Any]]:
        """Get a goal node by ID.
        
        Args:
            goal_id: Goal ID.
            
        Returns:
            Copy of goal node data, or None if it does not exist.
        """
        ...
    
    @abstractmethod
    def get_available_robots(
        self, types: Optional[List[str]] = None, concise: bool = False
    ) -> Dict[str, Dict[str, Any]]:
        """Get available robots.
        
        Args:
            types: List of robot types to filter by.
            concise: Whether to return concise format.
            
        Returns:
            Dictionary of available robots.
        """
        ...
    
    @abstractmethod
    def get_robot_states(self) -> Dict[str, Dict[str, Any]]:
        """Extract robot state information from the node list.
        
        Returns:
            Robot state dictionary keyed by robot_label, with state information as values.
        """
        ...

    # ==================== Position Query Interface ====================
    
    @abstractmethod
    def get_position_by_id(self, node_id: int) -> Optional[List[float]]:
        """Get a single entity position by node ID.
        
        Args:
            node_id: Unique ID of the node to query.
            
        Returns:
            Entity position coordinates [x, y], or None if not found.
        """
        ...
    
    @abstractmethod
    def get_position_by_label(self, node_label: str) -> Optional[List[float]]:
        """Get a single entity position by node label.
        
        Args:
            node_label: Label of the node to query.
            
        Returns:
            Entity position coordinates [x, y], or None if not found.
        """
        ...
    
    @abstractmethod
    def get_positions_by_entity_type(
        self, entity_categories: List[str] = ["building", "robot"]
    ) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
        """Batch-get positions for specified entity categories.
        
        Args:
            entity_categories: Entity category list to query.
            
        Returns:
            Nested dictionary containing position information.
        """
        ...
    
    @abstractmethod
    def get_boundary_features(
        self, district: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Extract boundary features.
        
        Args:
            district: Area name, optional.
            
        Returns:
            Boundary feature dictionary.
        """
        ...

    # ==================== Mapping Interface ====================
    
    @abstractmethod
    def get_node_map(
        self, map_type: Literal["id_to_label", "label_to_id"] = "id_to_label"
    ) -> Dict:
        """Generate a mapping dictionary between IDs and labels for all nodes.
        
        Args:
            map_type: Mapping type to generate.
            
        Returns:
            Dictionary representing the requested mapping.
        """
        ...
    
    @abstractmethod
    def get_all_node_categories(self) -> Dict[str, str]:
        """Generate a label-to-category mapping dictionary for all nodes.
        
        Returns:
            Dictionary representing the "node label -> category" mapping.
        """
        ...
    
    # ==================== Path Query Interface ====================
    
    @abstractmethod
    def find_path(self, src_id: str, dst_id: str) -> List[str]:
        """Find a path between two nodes.
        
        Args:
            src_id: Source node ID.
            dst_id: Target node ID.
            
        Returns:
            Path node ID list.
        """
        ...
    
    @abstractmethod
    def has_path(self, src_id: str, dst_id: str) -> bool:
        """Check whether a path exists between two nodes.
        
        Args:
            src_id: Source node ID.
            dst_id: Target node ID.
            
        Returns:
            Whether a path exists.
        """
        ...
    
    @abstractmethod
    def find_side_edges_near_path(
        self, src_id: str, dst_id: str
    ) -> List[Dict[str, Any]]:
        """Find side edges near a path.
        
        Args:
            src_id: Source node ID.
            dst_id: Target node ID.
            
        Returns:
            Side edge list.
        """
        ...
    
    @abstractmethod
    def find_candidate_locations(
        self, object_or_carrier_id: str, prefer_parking: bool = True
    ) -> List[Dict[str, Any]]:
        """Find candidate locations.
        
        Args:
            object_or_carrier_id: Object or carrier ID.
            prefer_parking: Whether to prefer parking spots.
            
        Returns:
            Candidate location list.
        """
        ...

    # ==================== Data Modification Interface (Sync) ====================
    
    @abstractmethod
    def update_edge_target(
        self,
        source_id: Union[int, str],
        old_target_id: Union[int, str],
        new_target_id: Union[int, str],
        edge_type: Optional[str] = None,
    ) -> bool:
        """Update an edge's target node.
        
        Args:
            source_id: Source node ID.
            old_target_id: Old target node ID.
            new_target_id: New target node ID.
            edge_type: Edge type, optional.
            
        Returns:
            Whether the update succeeded.
        """
        ...
    
    @abstractmethod
    def remove_edges_from_node(
        self, node_id: Union[int, str], edge_type: Optional[str] = None
    ) -> int:
        """Remove all edges from the specified node.
        
        Args:
            node_id: Source node ID.
            edge_type: Edge type. If specified, only edges of this type are removed.
            
        Returns:
            Number of removed edges.
        """
        ...
    
    # ==================== Data Modification Interface (Async) ====================
    
    @abstractmethod
    async def add_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Add a node asynchronously.
        
        Args:
            node_data: Node data.
            request_id: Request ID used to track operation results.
            
        Returns:
            Whether the operation succeeded.
        """
        ...
    
    @abstractmethod
    async def update_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Update a node asynchronously.
        
        Args:
            node_data: Node data.
            request_id: Request ID used to track operation results.
            
        Returns:
            Whether the operation succeeded.
        """
        ...
    
    @abstractmethod
    async def remove_node_async(
        self, node_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Remove a node asynchronously.
        
        Args:
            node_data: Data containing the node ID.
            request_id: Request ID used to track operation results.
            
        Returns:
            Whether the operation succeeded.
        """
        ...
    
    @abstractmethod
    async def add_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Add an edge asynchronously.
        
        Args:
            edge_data: Edge data.
            request_id: Request ID used to track operation results.
            
        Returns:
            Whether the operation succeeded.
        """
        ...
    
    @abstractmethod
    async def update_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Update an edge asynchronously.
        
        Args:
            edge_data: Edge data.
            request_id: Request ID used to track operation results.
            
        Returns:
            Whether the operation succeeded.
        """
        ...
    
    @abstractmethod
    async def remove_edge_async(
        self, edge_data: Dict[str, Any], request_id: Optional[str] = None
    ) -> bool:
        """Remove an edge asynchronously.
        
        Args:
            edge_data: Edge data containing source and target.
            request_id: Request ID used to track operation results.
            
        Returns:
            Whether the operation succeeded.
        """
        ...

    # ==================== Utility Methods ====================
    
    @abstractmethod
    def get_id(self) -> str:
        """Get the scene graph ID.
        
        Returns:
            Unique scene graph identifier.
        """
        ...
    
    @abstractmethod
    def get_scene_config(self) -> Dict[str, Any]:
        """Get the scene configuration.
        
        Returns:
            Scene configuration dictionary containing nodes and edges.
        """
        ...
    
    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary.
        
        Returns:
            Dictionary representation of the scene graph.
        """
        ...
    
    @abstractmethod
    def to_json(self) -> str:
        """Convert to a JSON string.
        
        Returns:
            JSON string representation of the scene graph.
        """
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all data."""
        ...
    
    @abstractmethod
    def load_from_dict(self, data: Dict[str, Any]) -> bool:
        """Load data from a dictionary.
        
        Args:
            data: Dictionary containing nodes and edges.
            
        Returns:
            Whether loading succeeded.
        """
        ...
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Asynchronous cleanup method.
        
        Cleans up event subscriptions and releases resources.
        """
        ...
