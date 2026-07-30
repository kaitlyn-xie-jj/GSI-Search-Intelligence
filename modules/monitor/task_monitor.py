#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Monitor - Event-driven scene monitoring system based on graph structure

Graph structure design:
┌─────────────────────────────────────────────────────────────────┐
│                    TaskMonitor Core Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    subscribe    ┌──────────────────┐           │
│  │ ReplyEvent  │ ────────→  │   TaskMonitor    │           │
│  └─────────────┘            │                  │           │
│                              │  ┌─────────────┐ │           │
│  ┌─────────────┐    monitor     │  │SceneGraph  │ │           │
│  │ TaskEvent   │ ←──────────  │  │(Singleton)  │ │           │
│  └─────────────┘            │  └─────────────┘ │           │
│                              │                  │           │
│  ┌─────────────┐    generate    │  ┌─────────────┐ │           │
│  │NL Feedback  │ ←──────────  │  │Graph Desc   │ │           │
│  └─────────────┘            │  │Engine        │ │           │
│                              │  └─────────────┘ │           │
│                              └──────────────────┘           │
│                                                                 │
│  Data flow:                                                     │
│  ReplyEvent → NL conversion → TaskEvent → scene graph update   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Scene graph description structure:
┌─────────────────────────────────────────────────────────────────┐
│                     Scene Graph Node Relationship Diagram                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│     Robot Node ──── executes ───→ Skill Node                    │
│        │                        │                              │
│        │                        │                              │
│      located_at                     affects                    │
│        │                        │                              │
│        ↓                        ↓                              │
│   Building Node ←── contains ──── Prop Node                    │
│        │                        │                              │
│        │                        │                              │
│      leads_to                     serves                       │
│        │                        │                              │
│        ↓                        ↓                              │
│     Goal Node ←──── implements ──── Task Node                  │
│                                                                 │
│  Edge type descriptions:                                        │
│  • executes: Robot → Skill (action relationship)               │
│  • located_at: Robot → Building (spatial relationship)         │
│  • contains: Building → Prop (containment relationship)        │
│  • affects: Skill → Prop (operation relationship)              │
│  • leads_to: Building → Goal (goal relationship)               │
│  • serves: Prop → Goal (functional relationship)               │
│  • implements: Task → Goal (task relationship)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

Event processing flow:
┌─────────────────────────────────────────────────────────────────┐
│                      Event Processing Flow                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Event                                                    │
│      │                                                          │
│      ▼                                                          │
│  ┌─────────┐                                                    │
│  │Event    │ ──→ ReplyEvent ──→ ┌─────────────────┐            │
│  │Classifier│                    │NL Conversion    │            │
│  └─────────┘                    │Engine            │            │
│      │                          └─────────────────┘            │
│      ▼                                   │                      │
│  TaskEvent ──────────────────────────────┼──→ ┌─────────────┐   │
│      │                                   │    │Graph Structure│  │
│      ▼                                   ▼    │Updater       │  │
│  ┌─────────┐                        ┌─────────────┐└───────────┘│
│  │Event    │                        │Scene Desc   │            │
│  │Monitor  │                        │Generator    │            │
│  └─────────┘                        └─────────────┘            │
│      │                                   │                      │
│      ▼                                   ▼                      │
│  Statistics                            NL Description           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import uuid
import random
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict, Counter

# Import event system
from modules.events.event_bus import (
    get_global_event_bus, subscribe_event, unsubscribe_event,
    publish_event, publish_high_priority_event
)
from modules.platform.platform_factory import get_scene_graph as get_platform_scene_graph
from modules.config.events import EventType, TaskEvent, DataModificationReplyEvent

# Import scene graph abstract types
from modules.platform.abstract_scene_graph import AbstractSceneGraph
from modules.platform.semantic_platform.utils.scene_graph_utils import find_path, _have_traversable
from modules.utils.location_utils import (
    get_entity_position,
    extract_object_position,
    shape_center_point,
)

from modules.task_solver.world_model.goal_progress_monitor import GoalProgressMonitor

logger = logging.getLogger(__name__)


@dataclass
class GraphNode:
    """Graph node description
    
    Represents a node in the scene graph, containing basic node information and graph structure attributes.
    
    Attributes:
        node_id: Node unique identifier
        node_type: Node type (robot, prop, building, goal, skill, task)
        category: Node category
        properties: Node properties dict
        connections: List of edges connected to this node
        degree: Node degree (number of connected edges)
        centrality: Centrality measure
    """
    node_id: str
    node_type: str
    category: str
    properties: Dict[str, Any] = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)
    degree: int = 0
    centrality: float = 0.0


@dataclass
class GraphEdge:
    """Graph edge description
    
    Represents an edge in the scene graph, describing relationships between nodes.
    
    Attributes:
        edge_id: Edge unique identifier
        source_id: Source node ID
        target_id: Target node ID
        edge_type: Edge type (executes, located_at, contains, affects, leads_to, serves, implements)
        relationship: Relationship description
        weight: Edge weight
        properties: Edge properties dict
    """
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SceneGraphTopology:
    """Scene graph topology
    
    Describes the topological features and structural information of the scene graph.
    
    Attributes:
        total_nodes: Total node count
        total_edges: Total edge count
        node_type_distribution: Node type distribution
        edge_type_distribution: Edge type distribution
        connectivity_matrix: Connectivity matrix
        clustering_coefficient: Clustering coefficient
        average_path_length: Average path length
        central_nodes: Central nodes list
        isolated_nodes: Isolated nodes list
    """
    total_nodes: int
    total_edges: int
    node_type_distribution: Dict[str, int] = field(default_factory=dict)
    edge_type_distribution: Dict[str, int] = field(default_factory=dict)
    connectivity_matrix: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    clustering_coefficient: float = 0.0
    average_path_length: float = 0.0
    central_nodes: List[str] = field(default_factory=list)
    isolated_nodes: List[str] = field(default_factory=list)


@dataclass
class SceneGraphDescription:
    """Scene graph description data structure
    
    Provides a complete scene graph description, containing graph structure, topology information, and natural language description.
    
    Scene graph description structure:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Scene Graph Description Hierarchy          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌─────────────┐    contains    ┌─────────────┐            │
    │  │  Graph       │ ────────→  │  Node Set   │            │
    │  │  Metadata    │            └─────────────┘            │
    │  └─────────────┘                   │                       │
    │         │                          │                       │
    │       description                  connection              │
    │         │                          │                       │
    │         ▼                          ▼                       │
    │  ┌─────────────┐    association    ┌─────────────┐         │
    │  │NL Description│ ←──────────  │  Edge Set   │         │
    │  └─────────────┘            └─────────────┘         │
    │         │                          │                       │
    │         │                          │                       │
    │       based on                     analysis                │
    │         │                          │                       │
    │         ▼                          ▼                       │
    │  ┌─────────────┐    generate    ┌─────────────┐            │
    │  │Formal       │ ←──────────  │Topology Info│            │
    │  │Description  │            └─────────────┘            │
    │  └─────────────┘                                           │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    
    Attributes:
        graph_id: Graph unique identifier
        timestamp: Description generation timestamp
        nodes: Graph node set
        edges: Graph edge set
        topology: Topology information
        natural_description: Natural language description
        formal_description: Formal description
        statistics: Statistics
        metadata: Metadata
    """
    graph_id: str
    timestamp: datetime
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    topology: SceneGraphTopology = field(default_factory=SceneGraphTopology)
    natural_description: str = ""
    formal_description: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class GraphDescriptionEngine:
    """Graph Description Engine
    
    Responsible for generating natural language descriptions and formal descriptions of the scene graph.
    
    Description generation flow:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Description Generation Flow               │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  Input: Scene graph data                                    │
    │         │                                                   │
    │         ▼                                                   │
    │  ┌─────────────┐                                            │
    │  │Graph Structure│ ──→ Node analysis ──→ ┌─────────────┐   │
    │  │Analyzer      │                  │Node Description│   │
    │  └─────────────┘                  │Generator       │   │
    │         │                         └─────────────┘   │
    │         ▼                                │                   │
    │  ┌─────────────┐                        │                   │
    │  │Relationship │ ──→ Edge analysis ─────┼──→ NL Synthesis   │
    │  │Analyzer     │                        │                   │
    │  └─────────────┘                        │                   │
    │         │                               │                   │
    │         ▼                               ▼                   │
    │  ┌─────────────┐                  ┌─────────────┐           │
    │  │Topology     │ ──→ Structure ──→│Formal       │           │
    │  │Analyzer     │     analysis     │Description  │           │
    │  └─────────────┘                  │Generator    │           │
    │                                   └─────────────┘           │
    │                                          │                   │
    │                                          ▼                   │
    │                                   Output: Complete description│
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self):
        """Initialize Graph Description Engine"""
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Node type to natural language mapping
        self.node_type_descriptions = {
            "robot": "robot",
            "prop": "prop",
            "building": "building",
            "goal": "goal",
            "skill": "skill",
            "task": "task",
            "intersection": "intersection",
            "street_segment": "street segment",
            "bridge": "bridge",
            "garden": "garden",
            "water_body": "water body",
        }
        self.edge_type_descriptions = {
            "execute": "executes",
            "located_at": "located at",
            "contains": "contains",
            "affects": "affects",
            "leads_to": "leads to",
            "serves": "serves",
            "implements": "implements",
            "connects_to": "connects to",
            "traversable": "is traversable to",
            "adjacent_to": "is adjacent to",
            "located_in": "is located in",
            "has_entrance": "has entrance",
        }
        
        # Mapping from edge types to English relationship descriptions
        self.edge_type_english_descriptions = {
            "stationed_at": "stationed at",
            "parked_at": "parked at", 
            "stored_at": "stored at",
            "located_at": "located at",
            "reachable_from": "reachable from",
            "contains": "contains",
            "execute": "executes",
            "assigned_to": "assigned to",
            "depends_on": "depends on",
            "requires": "requires",
            "can_perform": "can perform",
            "has_skill": "has skill",
            "working_on": "working on",
            "affects": "affects",
            "leads_to": "leads to",
            "serves": "serves",
            "implements": "implements",
            "inside": "inside",
            "executed_by": "executed by",
            "connects_to": "connects to",
            "traversable": "is traversable to",
            "adjacent_to": "adjacent to",
            "located_in": "located in",
            "has_entrance": "has entrance",
        }
    
    def analyze_graph_structure(self, nodes: List[Dict[str, Any]], 
                              edges: List[Dict[str, Any]]) -> SceneGraphTopology:
        """Analyze graph structure
        
        Analyze the topological structure features of the scene graph.
        
        Args:
            nodes: List of node data
            edges: List of edge data
            
        Returns:
            Topology structure information
        """
        topology = SceneGraphTopology(total_nodes=len(nodes), total_edges=len(edges))
        
        # Count node type distribution
        node_types = [n.get('properties', {}).get('type', 'unknown') for n in nodes]
        topology.node_type_distribution = dict(Counter(node_types))
        
        # Count edge type distribution
        edge_types = [(e.get('type') or e.get('properties', {}).get('type', 'unknown')) for e in edges]
        topology.edge_type_distribution = dict(Counter(edge_types))
        
        # Build connectivity matrix
        node_ids = [str(node.get('id', '')) for node in nodes]
        topology.connectivity_matrix = {nid: {} for nid in node_ids}
        
        for edge in edges:
            source = str(edge.get('source', ''))
            target = str(edge.get('target', ''))
            if source in topology.connectivity_matrix:
                topology.connectivity_matrix[source][target] = True
        
        # Calculate node degrees and centrality
        degree_map = defaultdict(int)
        for edge in edges:
            source = str(edge.get('source', ''))
            target = str(edge.get('target', ''))
            degree_map[source] += 1
            degree_map[target] += 1
        
        # Find central nodes (nodes with highest degree)
        if degree_map:
            max_degree = max(degree_map.values())
            topology.central_nodes = [nid for nid, degree in degree_map.items() 
                                    if degree == max_degree]
        
        # Find isolated nodes (nodes with degree 0)
        topology.isolated_nodes = [nid for nid in node_ids if degree_map[nid] == 0]
        
        # Calculate average path length (simplified version)
        if topology.total_nodes > 0:
            topology.average_path_length = topology.total_edges / topology.total_nodes
        
        return topology
    
    def generate_node_description(self, node: Dict[str, Any]) -> str:
        """Generate natural language description of a node
        
        Args:
            node: Node data
            
        Returns:
            Natural language description of the node
        """
        node_id = str(node.get('id', 'unknown'))
        properties = node.get('properties', {})
        node_type = properties.get('type', 'unknown')
        category = properties.get('category', 'uncategorized')
        
        type_desc = self.node_type_descriptions.get(node_type, node_type)
        
        # Basic description
        description = f"{type_desc} node [{node_id}]"
        
        # Add category information
        if category != 'uncategorized':
            description += f" (category: {category})"
        
        # Add position information
        if 'position' in properties:
            pos = properties['position']
            if isinstance(pos, dict) and 'x' in pos and 'y' in pos:
                description += f", located at coordinates ({pos['x']:.1f}, {pos['y']:.1f})"
        
        # Add status information
        if 'status' in properties:
            description += f", status: {properties['status']}"
        
        return description
    
    def get_edge_description(self, edge: Dict[str, Any], 
                               nodes_map: Dict[str, Dict[str, Any]], 
                               source_perspective: bool = True) -> str:
        """Generate accurate English relationship description based on edge type
        
        Args:
            edge: Edge data
            nodes_map: Mapping from node ID to node data
            source_perspective: Whether to describe the relationship from source node's perspective
            
        Returns:
            English relationship description string
        """
        source_id = str(edge.get('source', ''))
        target_id = str(edge.get('target', ''))
        edge_type = edge.get('type') or edge.get('properties', {}).get('type', 'unknown')
        
        # Get node names
        source_node = nodes_map.get(source_id, {})
        target_node = nodes_map.get(target_id, {})
        
        source_name = source_node.get('properties', {}).get('label', 
                     source_node.get('properties', {}).get('name', source_id))
        target_name = target_node.get('properties', {}).get('label',
                     target_node.get('properties', {}).get('name', target_id))
        
        # Generate relationship description based on edge type
        if source_perspective:
            # Describe from source node's perspective
            if edge_type in self.edge_type_english_descriptions:
                relation = self.edge_type_english_descriptions[edge_type]
                return f"{relation} {target_name}"
            else:
                return f"connected to {target_name} ({edge_type})"
        else:
            # Describe from target node's perspective (passive relationship)
            reverse_relations = {
                "stationed_at": "has stationed",
                "parked_at": "has parked", 
                "stored_at": "has stored",
                "located_at": "contains",
                "reachable_from": "can reach",
                "contains": "contained by",
                "execute": "executed by",
                "assigned_to": "assigned",
                "depends_on": "depended on by",
                "requires": "required by",
                "executed_by": "executes"
            }
            
            if edge_type in reverse_relations:
                relation = reverse_relations[edge_type]
                return f"{relation} {source_name}"
            else:
                return f"connected by {source_name} ({edge_type})"
    
    def generate_edge_description(self, edge: Dict[str, Any], 
                                nodes_map: Dict[str, Dict[str, Any]]) -> str:
        """Generate natural language description of an edge
        
        Args:
            edge: Edge data
            nodes_map: Mapping from node ID to node data
            
        Returns:
            Natural language description of the edge
        """
        source_id = str(edge.get('source', ''))
        target_id = str(edge.get('target', ''))
        properties = edge.get('properties', {})
        edge_type = edge.get('type') or edge.get('properties', {}).get('type', 'unknown')
        
        # Get source and target node information
        source_node = nodes_map.get(source_id, {})
        target_node = nodes_map.get(target_id, {})
        
        source_type = source_node.get('properties', {}).get('type', 'unknown')
        target_type = target_node.get('properties', {}).get('type', 'unknown')
        
        source_desc = self.node_type_descriptions.get(source_type, source_type)
        target_desc = self.node_type_descriptions.get(target_type, target_type)
        relation_desc = self.edge_type_descriptions.get(edge_type, edge_type)
        
        description = f"{source_desc}[{source_id}] {relation_desc} {target_desc}[{target_id}]"
        
        # Add weight information
        if 'weight' in properties:
            description += f" (weight: {properties['weight']})"
        
        return description
    
    def generate_natural_description(self, nodes: List[Dict[str, Any]], 
                                   edges: List[Dict[str, Any]], 
                                   topology: SceneGraphTopology) -> str:
        """Generate natural language description of scene graph
        
        Improved description structure - ordered by semantic hierarchy and entity importance:
        ┌─────────────────────────────────────────────────────────┐
        │            Improved Natural Language Description         │
        ├─────────────────────────────────────────────────────────┤
        │                                                         │
        │  1. Scene Overview                                      │
        │     ├─ Overall scale statistics (nodes, edges)         │
        │     └─ Entity type distribution overview               │
        │                                                         │
        │  2. TransFacility Layer (Building nodes)              │
        │     ├─ Building entity detailed information            │
        │     ├─ Building functions and properties               │
        │     └─ Spatial relationships between buildings         │
        │                                                         │
        │  3. Props Layer (Prop nodes)                           │
        │     ├─ Prop entity detailed information                │
        │     ├─ Prop functions and states                       │
        │     └─ Associations between props and buildings        │
        │                                                         │
        │  4. Agent Layer (Robot nodes)                          │
        │     ├─ Robot entity detailed information               │
        │     ├─ Robot capabilities and states                   │
        │     └─ Robot-environment interaction relationships     │
        │                                                         │
        │  5. Goal Layer (Goal nodes)                            │
        │     ├─ Goal entity detailed information                │
        │     ├─ Goal requirements and conditions                │
        │     └─ Goal associations with other entities           │
        │                                                         │
        │  6. Relationship Network Detailed Analysis             │
        │     ├─ Core connection relationship descriptions       │
        │     ├─ Key node identification                         │
        │     └─ Network topology characteristics                │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
        
        Args:
            nodes: List of node data
            edges: List of edge data
            topology: Topology structure information
            
        Returns:
            Natural language description string
        """
        description_parts = []
        
        # 1. Scene Overview
        overview = f"Current scene contains {topology.total_nodes} nodes and {topology.total_edges} edges."
        
        if topology.node_type_distribution:
            # Display node type distribution ordered by importance
            type_order = ['building', 'prop', 'robot', 'goal', 'skill', 'task']
            ordered_types = []
            for node_type in type_order:
                if node_type in topology.node_type_distribution:
                    count = topology.node_type_distribution[node_type]
                    type_name = {
                        'building': 'buildings',
                        'prop': 'props', 
                        'robot': 'robots',
                        'goal': 'goals',
                        'skill': 'skills',
                        'task': 'tasks'
                    }.get(node_type, node_type)
                    ordered_types.append(f"{count} {type_name}")
            
            # Add other undefined types
            for node_type, count in topology.node_type_distribution.items():
                if node_type not in type_order:
                    ordered_types.append(f"{count} {node_type}")
            
            if ordered_types:
                overview += f" Entity distribution: {', '.join(ordered_types)}."
        
        description_parts.append(overview)
        
        # Group nodes by category (for hierarchical display, but retain detailed type information)
        nodes_by_category = defaultdict(list)
        nodes_by_type = defaultdict(list)  # Retain detailed type grouping for statistics
        
        for node in nodes:
            node_props = node.get('properties', {})
            node_type = node_props.get('type', 'unknown')
            node_category = node_props.get('category', '')
            
            # Group by detailed type
            nodes_by_type[node_type].append(node)
            
            # Group by category for hierarchical display
            if node_category:
                category = node_category
            else:
                # If no category field, map based on type
                type_to_category = {
                    'library': 'building',
                    'hospital': 'building', 
                    'power_station': 'building',
                    'parking': 'building',
                    'mall': 'building',
                    'garden': 'building',
                    'robot_base': 'building',
                    'UAV': 'robot',
                    'UGV': 'robot',
                    'vehicle': 'prop',
                    'cargo': 'prop',
                    'equipment_failure': 'prop',
                }
                category = type_to_category.get(node_type, 'other')
            nodes_by_category[category].append(node)
        
        # Create node mapping for edge relationship lookup
        nodes_map = {str(node.get('id', '')): node for node in nodes}
        
        # 2. TransFacility Layer Description (Building nodes)
        nodes_by_cat = defaultdict(list)
        for n in nodes:
            cat = n.get('properties', {}).get('category', '')
            nodes_by_cat[cat].append(n)
        nodes_map = {str(n.get('id', '')): n for n in nodes}

        if nodes_by_cat.get('trans_facility'):
            tf = nodes_by_cat['trans_facility']
            s = f"\n\nTransport Layer: {len(tf)} trans_facility entities."
            by_type = defaultdict(list)
            for x in tf: by_type[x.get('properties', {}).get('type','unknown')].append(x)
            for t, lst in by_type.items():
                s += f"\n   {t} ({len(lst)}):"
                for x in lst[:8]:  # Limit display count per type to avoid excessive length
                    xid = str(x.get('id','')); name = x.get('properties',{}).get('label', xid)
                    s += f" '{name}';"
            description_parts.append(s)

        if nodes_by_cat.get('area'):
            ars = nodes_by_cat['area']
            s = f"\n\nArea Layer: {len(ars)} areas."
            by_type = defaultdict(list)
            for a in ars: by_type[a.get('properties', {}).get('type','unknown')].append(a)
            for t, lst in by_type.items():
                s += f"\n   {t} ({len(lst)}):"
                for a in lst[:8]:
                    aid = str(a.get('id','')); name = a.get('properties',{}).get('label', aid)
                    s += f" '{name}';"
            description_parts.append(s)

        if 'building' in nodes_by_category:
            buildings = nodes_by_category['building']
            building_desc = f"\n\nTransFacility Layer: The scene contains {len(buildings)} building entities."
            
            # Group buildings by detailed type for display
            building_types = defaultdict(list)
            for building in buildings:
                building_type = building.get('properties', {}).get('type', 'unknown')
                building_types[building_type].append(building)
            
            for building_type, type_buildings in building_types.items():
                building_desc += f"\n   {building_type} ({len(type_buildings)} units):"
                for building in type_buildings:
                    building_id = str(building.get('id', ''))
                    building_props = building.get('properties', {})
                    building_name = building_props.get('label', building_id)
                    
                    building_detail = f"'{building_name}'" # (ID: {building_id})"
                    
                    # Find edges related to this building
                    related_edges = [e for e in edges if str(e.get('source', '')) == building_id or str(e.get('target', '')) == building_id]
                    if related_edges:
                        connections = []
                        for edge in related_edges:
                            if str(edge.get('source', '')) == building_id:
                                # Relationships originating from building
                                connection_desc = self.get_edge_description(edge, nodes_map, source_perspective=True)
                                connections.append(connection_desc)
                            else:
                                # Relationships pointing to building
                                connection_desc = self.get_edge_description(edge, nodes_map, source_perspective=False)
                                connections.append(connection_desc)
                        
                        if connections:
                            building_detail += f", {', '.join(connections[:2])}"  # Show at most 2 connections
                    
                    building_desc += f" {building_detail};"
            
            description_parts.append(building_desc)
        
        # 3. Props Layer Description (Prop nodes)
        if 'prop' in nodes_by_category:
            props = nodes_by_category['prop']
            prop_desc = f"\n\nProps Layer: The scene contains {len(props)} prop entities."
            
            # Group props by detailed type for display
            prop_types = defaultdict(list)
            for prop in props:
                prop_type = prop.get('properties', {}).get('type', 'unknown')
                prop_types[prop_type].append(prop)
            
            for prop_type, type_props in prop_types.items():
                prop_desc += f"\n   {prop_type} ({len(type_props)} units):"
                for prop in type_props:
                    prop_id = str(prop.get('id', ''))
                    prop_props = prop.get('properties', {})
                    prop_name = prop_props.get('label', prop_id)
                    prop_state = prop_props.get('state', prop_props.get('status', 'unknown state'))
                    
                    prop_detail = f"'{prop_name}'" # (ID: {prop_id})
                    
                    # Find edges related to this prop
                    related_edges = [e for e in edges if str(e.get('source', '')) == prop_id or str(e.get('target', '')) == prop_id]
                    if related_edges:
                        connections = []
                        for edge in related_edges:
                            if str(edge.get('source', '')) == prop_id:
                                # Relationships originating from prop
                                connection_desc = self.get_edge_description(edge, nodes_map, source_perspective=True)
                                connections.append(connection_desc)
                            else:
                                # Relationships pointing to prop
                                connection_desc = self.get_edge_description(edge, nodes_map, source_perspective=False)
                                connections.append(connection_desc)
                        
                        if connections:
                            prop_detail += f", {', '.join(connections[:2])}"  # Show at most 2 connection info
                    
                    prop_desc += f" {prop_detail};"
            
            description_parts.append(prop_desc)
        
        # 4. Agent Layer Description (Robot nodes)
        if 'robot' in nodes_by_category:
            robots = nodes_by_category['robot']
            robot_desc = f"\n\nAgent Layer: The scene contains {len(robots)} robot entities."
            
            # Group robots by detailed type for display
            robot_types = defaultdict(list)
            for robot in robots:
                robot_type = robot.get('properties', {}).get('type', 'unknown')
                robot_types[robot_type].append(robot)
            
            for robot_type, type_robots in robot_types.items():
                robot_desc += f"\n   {robot_type} ({len(type_robots)} units):"
                for robot in type_robots:
                    robot_id = str(robot.get('id', ''))
                    robot_props = robot.get('properties', {})
                    robot_name = robot_props.get('label', robot_id)
                    robot_status = robot_props.get('status', robot_props.get('state', 'unknown state'))
                    
                    robot_detail = f"'{robot_name}', status: {robot_status}" # (ID: {robot_id})
                    
                    # Find robot's related relationships
                    related_edges = [e for e in edges if str(e.get('source', '')) == robot_id or str(e.get('target', '')) == robot_id]
                    if related_edges:
                        connections = []
                        for edge in related_edges:
                            if str(edge.get('source', '')) == robot_id:
                                # Relationships originating from robot
                                connection_desc = self.get_edge_description(edge, nodes_map, source_perspective=True)
                                connections.append(connection_desc)
                            else:
                                # Relationships pointing to robot
                                connection_desc = self.get_edge_description(edge, nodes_map, source_perspective=False)
                                connections.append(connection_desc)
                        
                        if connections:
                            robot_detail += f", {', '.join(connections[:2])}"  # Show at most 2 connection info
                    
                    robot_desc += f" {robot_detail};"
            
            description_parts.append(robot_desc)
        
        # 5. Goal Layer Description (Goal nodes)
        if 'goal' in nodes_by_category:
            goals = nodes_by_category['goal']
            goal_desc = f"\n\nGoal Layer: The scene contains {len(goals)} goal entities."
            
            # Group goals by detailed type for display
            goal_types = defaultdict(list)
            for goal in goals:
                goal_type = goal.get('properties', {}).get('type', 'unknown')
                goal_types[goal_type].append(goal)
            
            for goal_type, type_goals in goal_types.items():
                goal_desc += f"\n   {goal_type} ({len(type_goals)} units):"
                for goal in type_goals:
                    goal_id = str(goal.get('id', ''))
                    goal_props = goal.get('properties', {})
                    goal_name = goal_props.get('name', goal_id)
                    goal_status = goal_props.get('status', goal_props.get('state', 'incomplete'))
                    
                    goal_detail = f"'{goal_name}', status: {goal_status}" # (ID: {goal_id})
                    
                    # Find goal's related relationships
                    related_edges = [e for e in edges if str(e.get('source', '')) == goal_id or str(e.get('target', '')) == goal_id]
                    if related_edges:
                        connections = []
                        for edge in related_edges:
                            if str(edge.get('source', '')) == goal_id:
                                # Relationships originating from goal
                                connection_desc = self.get_edge_description(edge, nodes_map, source_perspective=True)
                                connections.append(connection_desc)
                            else:
                                # Relationships pointing to goal
                                connection_desc = self.get_edge_description(edge, nodes_map, source_perspective=False)
                                connections.append(connection_desc)
                        
                        if connections:
                            goal_detail += f", {', '.join(connections[:2])}"  # Show at most 2 connection info
                    
                    goal_desc += f" {goal_detail};"
            
            description_parts.append(goal_desc)
        
        # 6. Relationship Network Detailed Analysis
        if edges:
            network_desc = f"\n\nRelationship Network: The scene contains a total of {len(edges)} connection relationships."
            
            # Group by edge type
            edges_by_type = defaultdict(int)
            for edge in edges:
                edge_type = edge.get('type', 'unknown')
                edges_by_type[edge_type] += 1
            
            if edges_by_type:
                edge_type_names = {
                    'execute': 'execution relationships',
                    'located_at': 'location relationships',
                    'contains': 'containment relationships',
                    'affects': 'influence relationships',
                    'leads_to': 'leading relationships',
                    'serves': 'service relationships',
                    'implements': 'implementation relationships',
                    'depends_on': 'dependency relationships',
                    'assigned_to': 'assignment relationships'
                }
                
                edge_dist = []
                for edge_type, count in sorted(edges_by_type.items(), key=lambda x: x[1], reverse=True):
                    type_name = edge_type_names.get(edge_type, edge_type)
                    edge_dist.append(f"{count} {type_name}")
                
                network_desc += f" Relationship type distribution: {', '.join(edge_dist[:5])}."  # Show at most 5 relationship types
            
            # Network topology characteristics
            if topology.total_nodes > 0:
                connectivity = topology.total_edges / topology.total_nodes
                if connectivity > 2.0:
                    complexity = "highly connected"
                elif connectivity > 1.0:
                    complexity = "moderately connected"
                else:
                    complexity = "sparsely connected"
                
                network_desc += f" Network characteristics: {complexity} graph structure with average connectivity of {connectivity:.2f}."
            
            # Key node analysis
            if topology.central_nodes:
                central_names = []
                for node_id in topology.central_nodes[:3]:  # Show at most 3
                    node = nodes_map.get(node_id)
                    if node:
                        node_name = node.get('properties', {}).get('label', node_id)
                        central_names.append(node_name)
                
                if central_names:
                    network_desc += f" Network central nodes: {', '.join(central_names)}."
            
            # Isolated nodes
            if topology.isolated_nodes:
                isolated_names = []
                for node_id in topology.isolated_nodes[:3]:  # Show at most 3
                    node = nodes_map.get(node_id)
                    if node:
                        node_name = node.get('properties', {}).get('label', node_id)
                        isolated_names.append(node_name)
                
                if isolated_names:
                    network_desc += f" Isolated nodes: {', '.join(isolated_names)}."
            
            description_parts.append(network_desc)
        
        return "".join(description_parts)
    
    def generate_formal_description(self, nodes: List[Dict[str, Any]], 
                                  edges: List[Dict[str, Any]], 
                                  topology: SceneGraphTopology) -> Dict[str, Any]:
        """Generate formal description
        
        Args:
            nodes: List of node data
            edges: List of edge data
            topology: Topology structure information
            
        Returns:
            Formal description dictionary
        """
        return {
            "graph_type": "directed_labeled_graph",
            "node_set": {
                "cardinality": topology.total_nodes,
                "types": topology.node_type_distribution,
                "identifiers": [str(node.get('id', '')) for node in nodes]
            },
            "edge_set": {
                "cardinality": topology.total_edges,
                "types": topology.edge_type_distribution,
                "relations": [
                    {
                        "source": str(edge.get('source', '')),
                        "target": str(edge.get('target', '')),
                        "type": edge.get('properties', {}).get('type', 'unknown')
                    }
                    for edge in edges
                ]
            },
            "topology_metrics": {
                "connectivity_ratio": topology.total_edges / max(topology.total_nodes, 1),
                "central_nodes": topology.central_nodes,
                "isolated_nodes": topology.isolated_nodes,
                "clustering_coefficient": topology.clustering_coefficient,
                "average_path_length": topology.average_path_length
            },
            "structural_properties": {
                "is_connected": len(topology.isolated_nodes) == 0,
                "is_acyclic": False,  # simplified assumption
                "max_degree": max([len(connections) for connections in topology.connectivity_matrix.values()]) if topology.connectivity_matrix else 0
            }
        }


class TaskMonitor:
    """Task Monitor
    
    Event-driven scene monitoring system based on graph structure.
    
    System Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                   TaskMonitor System Architecture           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  ┌─────────────────┐    manages   ┌─────────────────┐       │
    │  │Event Subscription│ ←──────────  │  TaskMonitor   │       │
    │  │    Manager      │             │  (Singleton)    │       │
    │  └─────────────────┘             └─────────────────┘       │
    │           │                              │                  │
    │       subscribes                      controls              │
    │           │                              │                  │
    │           ▼                              ▼                  │
    │  ┌─────────────────┐            ┌─────────────────┐         │
    │  │   EventBus     │            │GoalProgress    │         │
    │  │ (Global Event  │            │Monitor         │         │
    │  │     Bus)       │            │(Integrated)    │         │
    │  └─────────────────┘            └─────────────────┘         │
    │           │                              │                  │
    │           │                              │                  │
    │      Event Flow                     Data Flow               │
    │           │                              │                  │
    │           ▼                              ▼                  │
    │  ┌─────────────────┐   analyzes  ┌─────────────────┐        │
    │  │Event Processing │ ──────────→ │Graph Description│        │
    │  │    Engine       │             │    Engine       │        │
    │  └─────────────────┘             └─────────────────┘        │
    │                                          │                  │
    │                                          ▼                  │
    │                              Natural Language Output        │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    
    Attributes:
        _instance: Singleton instance
        _description_engine: Graph description engine
        _event_subscriptions: Event subscription list
        _monitored_events: Set of monitored events
        _event_statistics: Event statistics information
        _goal_progress_monitor: Goal progress monitor instance
    """
    
    _instance: Optional['TaskMonitor'] = None
    _lock = asyncio.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation"""
        if cls._instance is None:
            cls._instance = super(TaskMonitor, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize task monitor
        """
        # Avoid duplicate initialization
        if hasattr(self, '_is_initialized') and self._is_initialized:
            return
            
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Create graph description engine
        self._description_engine = GraphDescriptionEngine()
        
        # Event subscription management
        self._event_subscriptions: List[str] = []
        self._monitored_events: Set[str] = set()
        
        # Event statistics information
        self._event_statistics = {
            'total_events_processed': 0,
            'reply_events_converted': 0,
            'task_events_monitored': 0,
            'scene_descriptions_generated': 0,
            'last_activity_timestamp': None
        }
        
        # Goal progress monitor
        self._goal_progress_monitor: Optional[GoalProgressMonitor] = None
        
        # Initialization status
        self._is_initialized = False
        
        # Automatically setup event subscriptions
        self._setup_event_subscriptions()
        
        self._is_initialized = True
        self.logger.info("TaskMonitor initialized with graph description engine")
    
    def set_goal_progress_monitor(self, monitor: GoalProgressMonitor) -> None:
        """Set the goal progress monitor
        
        Args:
            monitor: GoalProgressMonitor instance to integrate
        """
        self._goal_progress_monitor = monitor
        self.logger.info("GoalProgressMonitor integrated with TaskMonitor")
    
    def get_goal_progress_monitor(self) -> Optional[GoalProgressMonitor]:
        """Get the goal progress monitor
        
        Returns:
            GoalProgressMonitor instance if set, None otherwise
        """
        return self._goal_progress_monitor
    
    @property
    def goal_progress_monitor(self) -> Optional[GoalProgressMonitor]:
        """Goal progress monitor property

        Returns:
            GoalProgressMonitor instance if set, None otherwise
        """
        return self._goal_progress_monitor
    
    def _setup_event_subscriptions(self):
        """Set up event subscriptions
        
        Subscribe to DataModificationReplyEvent and convert it to TaskEvent for publishing.
        """
        try:
            # Subscribe to DataModificationReplyEvent
            reply_subscription_id = subscribe_event(
                event_type=EventType.TASK.value,  # DataModificationReplyEvent belongs to task type
                handler=self._handle_reply_event,
                subscriber_id="task_monitor_reply_handler",
                filter_func=lambda event: isinstance(event, DataModificationReplyEvent),
                priority=10  # High priority processing
            )
            self._event_subscriptions.append(reply_subscription_id)
            
            # Subscribe to all TaskEvents for monitoring
            task_subscription_id = subscribe_event(
                event_type=EventType.TASK.value,
                handler=self._monitor_task_event,
                subscriber_id="task_monitor_observer",
                filter_func=lambda event: isinstance(event, TaskEvent),
                priority=5  # Medium priority
            )
            self._event_subscriptions.append(task_subscription_id)
            
            self.logger.info(f"Event subscriptions setup: {len(self._event_subscriptions)} subscriptions")
            
        except Exception as e:
            self.logger.error(f"Failed to setup event subscriptions: {e}")
            raise
    
    async def _handle_reply_event(self, event: DataModificationReplyEvent):
        """Handle DataModificationReplyEvent and convert to TaskEvent
        
        Convert DataModificationReplyEvent to a natural language form TaskEvent for publishing.
        
        Conversion flow:
        ┌─────────────────────────────────────────────────────────┐
        │               Reply Event Conversion Flow               │
        ├─────────────────────────────────────────────────────────┤
        │                                                         │
        │  Input: DataModificationReplyEvent                      │
        │           │                                             │
        │           ▼                                             │
        │  ┌─────────────────┐                                    │
        │  │  Event Content  │ ──→ Extract operation type         │
        │  │  Analysis       │     │                             │
        │  └─────────────────┘     │                             │
        │           │               ▼                             │
        │           │      ┌─────────────────┐                    │
        │           │      │ Success/Failure │                    │
        │           │      │ Determination   │                    │
        │           │      └─────────────────┘                    │
        │           │               │                             │
        │           ▼               ▼                             │
        │  ┌─────────────────┐ ┌─────────────────┐                │
        │  │NL Generation    │ │Graph Structure  │                │
        │  │Engine           │ │Change Detection │                │
        │  └─────────────────┘ └─────────────────┘                │
        │           │               │                             │
        │           │               │                             │
        │           ▼               ▼                             │
        │  ┌─────────────────┐ ┌─────────────────┐                │
        │  │Feedback Text    │ │Scene Change     │                │
        │  │Generation       │ │Description      │                │
        │  └─────────────────┘ └─────────────────┘                │
        │           │               │                             │
        │           └───────┬───────┘                             │
        │                   ▼                                     │
        │           ┌─────────────────┐                           │
        │           │   TaskEvent     │                           │
        │           │ (Natural Lang.) │                           │
        │           └─────────────────┘                           │
        │                   │                                     │
        │                   ▼                                     │
        │           Publish to EventBus                           │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
        
        Args:
            event: DataModificationReplyEvent instance
        """
        try:
            # Generate natural language feedback
            natural_feedback = self._generate_natural_feedback_from_reply(event)
            
            # Detect scene graph changes
            scene_change_description = await self._detect_scene_changes(event)
            
            # Create TaskEvent
            task_event = TaskEvent(
                task_id=event.request_id,
                action="data_modification_completed",
                data={
                    "original_reply": {
                        "request_id": event.request_id,
                        "reply_to": event.reply_to,
                        "success": event.success,
                        "result": event.result,
                        "error_message": event.error_message
                    },
                    "natural_feedback": natural_feedback,
                    "scene_change_description": scene_change_description,
                    "processed_by": "task_monitor",
                    "processing_timestamp": datetime.now().isoformat()
                },
                entity_type="scene_modification",
                entity_id=event.event_id,
                source="task_monitor",
                priority=1  # High priority
            )
            
            # Publish TaskEvent
            await publish_high_priority_event(task_event)
            
            # Update statistics
            self._event_statistics['reply_events_converted'] += 1
            self._event_statistics['total_events_processed'] += 1
            self._event_statistics['last_activity_timestamp'] = datetime.now().isoformat()
            
            # Record monitored event
            self._monitored_events.add(event.event_id)
            
            self.logger.debug(f"Converted DataModificationReplyEvent {event.event_id} to TaskEvent with natural feedback")
            
        except Exception as e:
            self.logger.error(f"Failed to handle DataModificationReplyEvent {event.event_id}: {e}")
    
    async def _monitor_task_event(self, event: TaskEvent):
        """Monitor TaskEvent
        
        Record and analyze TaskEvent for statistics and monitoring.
        
        Args:
            event: TaskEvent instance
        """
        try:
            # Update statistics
            self._event_statistics['task_events_monitored'] += 1
            self._event_statistics['total_events_processed'] += 1
            self._event_statistics['last_activity_timestamp'] = datetime.now().isoformat()
            
            # Record event
            self._monitored_events.add(event.event_id)
            
            self.logger.debug(f"Monitored TaskEvent: {event.action} - {event.task_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to monitor TaskEvent {event.event_id}: {e}")
    
    def _generate_natural_feedback_from_reply(self, event: DataModificationReplyEvent) -> str:
        """Generate natural language feedback from a reply event.
        
        Feedback generation rules:
        ┌─────────────────────────────────────────────────────────┐
        │           Natural Language Feedback Rules               │
        ├─────────────────────────────────────────────────────────┤
        │                                                         │
        │  Success feedback templates:                            │
        │  ├─ Add: "Successfully added {entity_type} {entity_id}"│
        │  ├─ Update: "Successfully updated {entity_type} info"  │
        │  └─ Remove: "Successfully removed {entity_type}"       │
        │                                                         │
        │  Failure feedback templates:                            │
        │  ├─ Add failed: "Failed to add {entity_type}: {reason}"│
        │  ├─ Update failed: "Failed to update: {reason}"        │
        │  └─ Remove failed: "Failed to remove: {reason}"        │
        │                                                         │
        │  Special cases:                                         │
        │  ├─ Unknown op: "Performed unknown operation"           │
        │  ├─ Empty result: "Operation completed, no details"    │
        │  └─ Exception: "Exception occurred during operation"   │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
        
        Args:
            event: DataModificationReplyEvent instance
            
        Returns:
            Natural language feedback string
        """
        if event.success:
            result = event.result
            operation = result.get('operation', 'unknown_operation')
            entity_type = result.get('entity_type', 'entity')
            
            # Entity type mapping
            entity_type_map = {
                'node': 'node',
                'edge': 'edge',
                'robot': 'robot',
                'prop': 'prop',
                'building': 'building',
                'goal': 'goal',
                'skill': 'skill',
                'task': 'task'
            }
            
            entity_desc = entity_type_map.get(entity_type, entity_type)
            
            if operation == 'add':
                return f"✅ Successfully added {entity_desc} to the scene graph"
            elif operation == 'update':
                return f"✅ Successfully updated {entity_desc} info"
            elif operation == 'remove':
                return f"✅ Successfully removed {entity_desc} from the scene graph"
            else:
                return f"✅ Successfully performed {operation} on {entity_desc}"
        else:
            error_msg = event.error_message or "unknown error"
            return f"❌ Operation failed: {error_msg}"
    
    async def _detect_scene_changes(self, event: DataModificationReplyEvent) -> str:
        """Detect scene graph changes.
        
        Analyze the impact of data modification events on the scene graph structure.
        
        Args:
            event: DataModificationReplyEvent instance
            
        Returns:
            Scene change description string
        """
        try:
            if not event.success:
                return "Scene graph unchanged (operation failed)"
            
            result = event.result
            operation = result.get('operation', '')
            entity_type = result.get('entity_type', '')
            
            current_description = await self.get_scene_description()
            
            if operation == 'add':
                if entity_type == 'node':
                    return f"Added 1 node to scene graph, total now {current_description.topology.total_nodes} nodes"
                elif entity_type == 'edge':
                    return f"Added 1 edge to scene graph, total now {current_description.topology.total_edges} edges"
            elif operation == 'remove':
                if entity_type == 'node':
                    return f"Removed 1 node from scene graph, total now {current_description.topology.total_nodes} nodes"
                elif entity_type == 'edge':
                    return f"Removed 1 edge from scene graph, total now {current_description.topology.total_edges} edges"
            elif operation == 'update':
                return f"Scene graph {entity_type} info updated"
            
            return "Scene graph structure changed"
            
        except Exception as e:
            self.logger.error(f"Failed to detect scene changes: {e}")
            return "Unable to detect scene graph changes"
    

    async def get_scene_description(self, scene_graph=None, include_details: bool = True) -> SceneGraphDescription:
        """Get scene graph description
        
        Generate a complete description of the current scene graph, including natural language and formal descriptions.
        
        Scene description generation flow:
        ┌─────────────────────────────────────────────────────────┐
        │                Scene Description Generation Flow        │
        ├─────────────────────────────────────────────────────────┤
        │                                                         │
        │  ┌─────────────────┐                                    │
        │  │Get scene graph  │ ──→ Node data                      │
        │  │data             │     │                             │
        │  └─────────────────┘     │                             │
        │           │               ▼                             │
        │           │      ┌─────────────────┐                    │
        │           │      │   Edge data     │                    │
        │           │      └─────────────────┘                    │
        │           │               │                             │
        │           ▼               ▼                             │
        │  ┌─────────────────┐ ┌─────────────────┐                │
        │  │Graph structure  │ │Topology         │                │
        │  │analysis         │ │computation      │                │
        │  └─────────────────┘ └─────────────────┘                │
        │           │               │                             │
        │           │               │                             │
        │           ▼               ▼                             │
        │  ┌─────────────────┐ ┌─────────────────┐                │
        │  │NL description   │ │Formal description│               │
        │  │generation       │ │generation        │               │
        │  └─────────────────┘ └─────────────────┘                │
        │           │               │                             │
        │           └───────┬───────┘                             │
        │                   ▼                                     │
        │           ┌─────────────────┐                           │
        │           │Complete scene   │                           │
        │           │graph description│                           │
        │           └─────────────────┘                           │
        │                                                         │
        └─────────────────────────────────────────────────────────┘
        
        Args:
            scene_graph: AbstractSceneGraph instance
            include_details: Whether to include detailed information
            
        Returns:
            Scene graph description object
        """
        try:
            if scene_graph is None:
                scene_graph = get_platform_scene_graph()
            
            if scene_graph is None:
                raise ValueError("scene_graph is required but not provided")
            
            # Get scene graph data
            nodes = scene_graph.get_all_nodes()
            edges = scene_graph.get_all_edges()
            
            # Analyze graph structure
            topology = self._description_engine.analyze_graph_structure(nodes, edges)
            
            # Generate natural language description
            natural_desc = self._description_engine.generate_natural_description(
                nodes, edges, topology
            )
            
            # Generate formal description
            formal_desc = self._description_engine.generate_formal_description(
                nodes, edges, topology
            )
            
            # Build graph node and edge objects
            graph_nodes = []
            graph_edges = []
            
            if include_details:
                # Build detailed graph node objects
                for node in nodes:
                    graph_node = GraphNode(
                        node_id=str(node.get('id', '')),
                        node_type=node.get('properties', {}).get('type', 'unknown'),
                        category=node.get('properties', {}).get('category', 'uncategorized'),
                        properties=node.get('properties', {})
                    )
                    graph_nodes.append(graph_node)
                
                # Build detailed graph edge objects
                for edge in edges:
                    graph_edge = GraphEdge(
                        edge_id=f"{edge.get('source', '')}->{edge.get('target', '')}",
                        source_id=str(edge.get('source', '')),
                        target_id=str(edge.get('target', '')),
                        edge_type=str(edge.get('type', 'unknown')),
                        properties=edge.get('properties', {})
                    )
                    graph_edges.append(graph_edge)
            
            # Generate statistics
            statistics = {
                'generation_timestamp': datetime.now().isoformat(),
                'total_processing_time': 0.0,  # Can add actual timing
                'description_length': len(natural_desc),
                'complexity_score': topology.total_edges / max(topology.total_nodes, 1)
            }
            
            # Create scene graph description object
            description = SceneGraphDescription(
                graph_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                nodes=graph_nodes,
                edges=graph_edges,
                topology=topology,
                natural_description=natural_desc,
                formal_description=formal_desc,
                statistics=statistics,
                metadata={
                    'generator': 'TaskMonitor',
                    'version': '1.0',
                    'include_details': include_details
                }
            )
            
            # Update statistics
            self._event_statistics['scene_descriptions_generated'] += 1
            
            return description
            
        except Exception as e:
            self.logger.error(f"Failed to generate scene description: {e}")
            # Return empty description
            return SceneGraphDescription(
                graph_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                natural_description=f"Error generating scene description: {e}",
                formal_description={'error': str(e)}
            )

    def get_naturel_language_description(self, scene_graph=None) -> str:
        """Get natural language description

        Args:
            scene_graph: AbstractSceneGraph instance
            
        Returns:
            Natural language description string of the scene graph
        """
        try:
            if scene_graph is None:
                # Try to get global scene graph as fallback
                scene_graph = get_platform_scene_graph()
            
            if scene_graph is None:
                return "Unable to generate natural language description: no scene graph provided"
            
            description = self._description_engine.generate_natural_description(
                scene_graph.get_all_nodes(),
                scene_graph.get_all_edges(),
                self._description_engine.analyze_graph_structure(
                    scene_graph.get_all_nodes(),
                    scene_graph.get_all_edges()
                )
            )
            return description
        except Exception as e:
            self.logger.error(f"Failed to get natural language description: {e}")
            return "Unable to generate natural language description"

    def get_monitoring_statistics(self, scene_graph=None) -> Dict[str, Any]:
        """Get monitoring statistics
        
        Args:
            scene_graph: AbstractSceneGraph instance
        
        Returns:
            Monitoring statistics dictionary
        """
        stats = {
            **self._event_statistics,
            'monitored_events_count': len(self._monitored_events),
            'active_subscriptions': len(self._event_subscriptions),
        }
        
        # Add scene graph status if provided
        if scene_graph is None:
            scene_graph = get_platform_scene_graph()
        
        if scene_graph is not None:
            stats['scene_graph_status'] = {
                'total_nodes': len(scene_graph.get_all_nodes()),
                'total_edges': len(scene_graph.get_all_edges())
            }
        else:
            stats['scene_graph_status'] = {
                'total_nodes': 0,
                'total_edges': 0,
                'note': 'scene_graph not provided'
            }
        
        return stats
    
    def get_robot_statistics(self, scene_graph=None) -> Dict[str, Any]:
        """Get robot statistics
        
        Args:
            scene_graph: AbstractSceneGraph instance
        
        Returns:
            Robot statistics dictionary
        """
        try:
            if scene_graph is None:
                scene_graph = get_platform_scene_graph()
            
            if scene_graph is None:
                return {
                    'total_robots': 0,
                    'robot_types': {},
                    'active_robots': 0,
                    'idle_robots': 0,
                    'error': 'scene_graph not provided'
                }
            
            nodes = scene_graph.get_all_nodes()
            robot_nodes = [node for node in nodes if node.get('properties', {}).get('type') == 'robot']
            
            robot_stats = {
                'total_robots': len(robot_nodes),
                'robot_types': {},
                'active_robots': 0,
                'idle_robots': 0
            }
            
            for robot in robot_nodes:
                props = robot.get('properties', {})
                robot_type = props.get('robot_type', 'unknown')
                status = props.get('status', 'unknown')
                
                # Count robot types
                if robot_type not in robot_stats['robot_types']:
                    robot_stats['robot_types'][robot_type] = 0
                robot_stats['robot_types'][robot_type] += 1
                
                # Count robot status
                if status == 'active':
                    robot_stats['active_robots'] += 1
                elif status == 'idle':
                    robot_stats['idle_robots'] += 1
            
            return robot_stats
            
        except Exception as e:
            self.logger.error(f"Failed to get robot statistics: {e}")
            return {
                'total_robots': 0,
                'robot_types': {},
                'active_robots': 0,
                'idle_robots': 0,
                'error': str(e)
            }
    
    def get_task_goals(self, scene_graph=None) -> List[Dict[str, Any]]:
        """Get task goal list
        
        Args:
            scene_graph: AbstractSceneGraph instance
        
        Returns:
            Task goal list
        """
        try:
            if scene_graph is None:
                scene_graph = get_platform_scene_graph()
            
            if scene_graph is None:
                return []
            
            nodes = scene_graph.get_all_nodes()
            goal_nodes = [node for node in nodes if node.get('properties', {}).get('type') == 'goal']
            
            goals = []
            for goal in goal_nodes:
                props = goal.get('properties', {})
                goal_info = {
                    'goal_id': goal.get('id'),
                    'description': props.get('description', 'Unknown target'),
                    'status': props.get('status', 'pending'),
                    'priority': props.get('priority', 'normal'),
                    'target_location': props.get('target_location', 'Unspecified'),
                    'assigned_robot': props.get('assigned_robot', None)
                }
                goals.append(goal_info)
            
            return goals
            
        except Exception as e:
            self.logger.error(f"Failed to get task goals: {e}")
            return []
    
    def get_graph_statistics(self, scene_graph=None) -> Dict[str, Any]:
        """Get graph statistics
        
        Args:
            scene_graph: AbstractSceneGraph instance
        
        Returns:
            Graph statistics dictionary
        """
        try:
            if scene_graph is None:
                scene_graph = get_platform_scene_graph()
            
            if scene_graph is None:
                return {
                    'total_nodes': 0,
                    'total_edges': 0,
                    'node_types': {},
                    'edge_types': {},
                    'graph_density': 0,
                    'average_degree': 0,
                    'error': 'scene_graph not provided'
                }
            
            nodes = scene_graph.get_all_nodes()
            edges = scene_graph.get_all_edges()
            
            # Node type statistics
            node_types = {}
            for node in nodes:
                node_type = node.get('properties', {}).get('type', 'unknown')
                if node_type not in node_types:
                    node_types[node_type] = 0
                node_types[node_type] += 1
            
            # Edge type statistics
            edge_types = {}
            for edge in edges:
                edge_type = edge.get('properties', {}).get('type', 'unknown')
                if edge_type not in edge_types:
                    edge_types[edge_type] = 0
                edge_types[edge_type] += 1
            
            # Calculate graph density
            total_nodes = len(nodes)
            total_edges = len(edges)
            max_possible_edges = total_nodes * (total_nodes - 1) if total_nodes > 1 else 0
            density = total_edges / max_possible_edges if max_possible_edges > 0 else 0
            
            return {
                'total_nodes': total_nodes,
                'total_edges': total_edges,
                'node_types': node_types,
                'edge_types': edge_types,
                'graph_density': density,
                'average_degree': (2 * total_edges) / total_nodes if total_nodes > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get graph statistics: {e}")
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'node_types': {},
                'edge_types': {},
                'graph_density': 0,
                'average_degree': 0,
                'error': str(e)
            }
    
    async def cleanup(self):
        """Clean up resources
        
        Cancel all event subscriptions and clean up monitoring data.
        """
        try:
            # Cancel all event subscriptions
            for subscription_id in self._event_subscriptions:
                unsubscribe_event(subscription_id)
            
            self._event_subscriptions.clear()
            self._monitored_events.clear()
            
            self.logger.info("TaskMonitor cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup TaskMonitor: {e}")


# Global TaskMonitor instance management
_global_task_monitor: Optional[TaskMonitor] = None
_monitor_lock = asyncio.Lock()


def get_global_task_monitor() -> TaskMonitor:
    """Get the global TaskMonitor instance
    
    Returns the global singleton TaskMonitor instance, creating it if necessary.
    
    Returns:
        Global TaskMonitor instance
    """
    global _global_task_monitor
    if _global_task_monitor is None:
        _global_task_monitor = TaskMonitor()
    return _global_task_monitor


async def start_task_monitoring() -> TaskMonitor:
    """Start task monitoring
    
    Returns:
        TaskMonitor instance
    """
    async with _monitor_lock:
        monitor = get_global_task_monitor()
        logger.info("Task monitoring started successfully")
        return monitor


async def stop_task_monitoring():
    """Stop task monitoring"""
    async with _monitor_lock:
        global _global_task_monitor
        if _global_task_monitor:
            await _global_task_monitor.cleanup()
            _global_task_monitor = None
        logger.info("Task monitoring stopped successfully")

def reset_task_monitoring():
    """Reset task monitoring"""
    global _global_task_monitor
    TaskMonitor._instance = None
    _global_task_monitor = None


async def get_current_scene_description(scene_graph=None, include_details: bool = True) -> SceneGraphDescription:
    """Get current scene description
    
    Args:
        scene_graph: AbstractSceneGraph instance (optional, will use global if not provided)
        include_details: Whether to include detailed information
        
    Returns:
        Current scene graph description
    """
    monitor = get_global_task_monitor()
    return await monitor.get_scene_description(scene_graph=scene_graph, include_details=include_details)


def get_scene_graph() -> AbstractSceneGraph:
    """Get the global scene_graph instance
    
    Returns:
        AbstractSceneGraph instance
    """
    return get_platform_scene_graph()