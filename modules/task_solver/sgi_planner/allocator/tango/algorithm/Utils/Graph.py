from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
import sys
import yaml

# -------------------------------
# GraphEdge Class
# -------------------------------
@dataclass
class GraphEdge:
    """Represents an edge in the graph, containing two nodes and edge type"""
    node1: int = -1
    node2: int = -1
    type: int = 0

    def set(self, node1: int, node2: int, type: int) -> None:
        """Set edge properties"""
        self.node1 = node1
        self.node2 = node2
        self.type = type

    def __str__(self) -> str:
        """String representation in format 'node1,node2,type'"""
        return f"{self.node1},{self.node2},{self.type}"
    
    def __hash__(self):
        # Compute hash based on immutable properties
        return hash((self.node1, self.node2, self.type))

    def __eq__(self, other: object) -> bool:
        """Compare if two edges are identical"""
        if not isinstance(other, GraphEdge):
            return NotImplemented
        return (self.node1 == other.node1 and 
                self.node2 == other.node2 and 
                self.type == other.type)

# -------------------------------
# GraphEdgeParam Class
# -------------------------------
@dataclass
class GraphEdgeParam:
    """Represents edge parameters including edge information and associated costs"""
    edge: GraphEdge = field(default_factory=GraphEdge)
    eng_cost: float = 0.0
    time_cost: float = 0.0

    def remove(self) -> None:
        """Remove edge parameters (set edge to invalid state)"""
        self.edge.set(-1, -1, 0)

    def exist(self) -> bool:
        """Check if edge exists (node1 >= 0)"""
        return self.edge.node1 >= 0

    def __str__(self) -> str:
        """String representation in format 'GraphEdgeParam(edge, eng_cost=x, time_cost=y)'"""
        return (f"GraphEdgeParam({self.edge}, eng_cost={self.eng_cost}, time_cost={self.time_cost})")

# -------------------------------
# GraphNodeParam Class
# -------------------------------
@dataclass
class GraphNodeParam:
    """Represents node parameters including node information and sets of incoming/outgoing edges"""
    node: int = -1
    time_cost: float = 0.0
    out_edge_ids: Set[int] = field(default_factory=set)
    in_edge_ids: Set[int] = field(default_factory=set)

    def remove(self) -> None:
        """Remove node parameters (set node to invalid state)"""
        self.node = -1
        self.out_edge_ids.clear()
        self.in_edge_ids.clear()

    def exist(self) -> bool:
        """Check if node exists (node >= 0)"""
        return self.node >= 0

    def __str__(self) -> str:
        """String representation in format 'GraphNodeParam(node=x, time_cost=y)'"""
        return f"GraphNodeParam(node={self.node}, time_cost={self.time_cost})"

# -------------------------------
# Graph Class
# -------------------------------
class Graph:
    def __init__(self):
        self.edge2edge_id: Dict[GraphEdge, int] = {}
        self.node2node_id: Dict[int, int] = {}
        self.edgeId2edgeParam: List[GraphEdgeParam] = []
        self.nodeId2nodeParam: List[GraphNodeParam] = []

    def clear(self) -> None:
        """Clear all graph data"""
        self.edge2edge_id.clear()
        self.node2node_id.clear()
        self.edgeId2edgeParam.clear()
        self.nodeId2nodeParam.clear()

    def edge_num(self) -> int:
        """Get number of edges in graph"""
        return len(self.edge2edge_id)

    def node_num(self) -> int:
        """Get number of nodes in graph"""
        return len(self.node2node_id)

    def has_node(self, node: int) -> bool:
        """Check if node exists in graph"""
        return node in self.node2node_id

    def has_edge(self, edge: GraphEdge) -> bool:
        """Check if edge exists in graph"""
        return edge in self.edge2edge_id

    def add_node(self, node: int, time_cost: float = 1.0) -> bool:
        """Add a node to the graph"""
        if self.has_node(node):
            return False
        temp_node_param = GraphNodeParam(node=node, time_cost=time_cost)
        temp_node_id = len(self.nodeId2nodeParam)
        self.nodeId2nodeParam.append(temp_node_param)
        self.node2node_id[node] = temp_node_id
        return True

    def add_edge(self, node1: int, node2: int, edge_type: int, eng_cost: float, time_cost: float,
                 flag_directed: bool = False) -> bool:
        """Add an edge to the graph (optionally directed)"""
        if not self.has_node(node1) or not self.has_node(node2):
            sys.stderr.write("ERROR: No such graph node.\n")
            return False

        temp_edge = GraphEdge(node1=node1, node2=node2, type=edge_type)
        if self.has_edge(temp_edge):
            return False
        
        # Add forward edge
        temp_edge_param = GraphEdgeParam(edge=temp_edge, eng_cost=eng_cost, time_cost=time_cost)
        temp_edge_id = len(self.edgeId2edgeParam)
        self.edgeId2edgeParam.append(temp_edge_param)
        self.edge2edge_id[temp_edge] = temp_edge_id

        node1_id = self.node2node_id[node1]
        node2_id = self.node2node_id[node2]
        self.nodeId2nodeParam[node1_id].out_edge_ids.add(temp_edge_id)
        self.nodeId2nodeParam[node2_id].in_edge_ids.add(temp_edge_id)

        if flag_directed:
            return True

        # Add reverse edge (for undirected graphs)
        temp_edge_rev = GraphEdge(node1=node2, node2=node1, type=edge_type)
        temp_edge_param_rev = GraphEdgeParam(edge=temp_edge_rev, eng_cost=eng_cost, time_cost=time_cost)
        temp_edge_id = len(self.edgeId2edgeParam)
        self.edgeId2edgeParam.append(temp_edge_param_rev)
        self.edge2edge_id[temp_edge_rev] = temp_edge_id

        self.nodeId2nodeParam[node2_id].out_edge_ids.add(temp_edge_id)
        self.nodeId2nodeParam[node1_id].in_edge_ids.add(temp_edge_id)
        return True

    def print(self) -> None:
        """Print basic graph information"""
        print("Node:")
        for node_param in self.nodeId2nodeParam:
            print(node_param.node, end=";   ")
        print("\nEdge:")
        for edge_param in self.edgeId2edgeParam:
            print(edge_param.edge, end=";   ")
        print("")

    def edge2id(self, node1: int, node2: int, type: int) -> int:
        """Get edge ID from node pair and edge type"""
        temp_edge = GraphEdge(node1=node1, node2=node2, type=type)
        return self.edge2edge_id.get(temp_edge, -1)

    def id2edge(self, id: int) -> Optional[GraphEdgeParam]:
        """Get edge parameters from edge ID"""
        if 0 <= id < len(self.edgeId2edgeParam) and self.edgeId2edgeParam[id].exist():
            return self.edgeId2edgeParam[id]
        return None

    def node2id(self, node: int) -> int:
        """Get node ID from node number"""
        return self.node2node_id.get(node, -1)

    def id2node(self, id: int) -> Optional[GraphNodeParam]:
        """Get node parameters from node ID"""
        if 0 <= id < len(self.nodeId2nodeParam) and self.nodeId2nodeParam[id].exist():
            return self.nodeId2nodeParam[id]
        return None

    def node(self, id: int) -> GraphNodeParam:
        """Get node parameters (with error checking)"""
        if id < 0 or id >= len(self.nodeId2nodeParam):
            sys.stderr.write("ERROR: Graph node.\n")
        return self.nodeId2nodeParam[id]

    def edge(self, id: int) -> GraphEdgeParam:
        """Get edge parameters (with error checking)"""
        if id < 0 or id >= len(self.edgeId2edgeParam):
            sys.stderr.write("ERROR: Graph edge.\n")
        return self.edgeId2edgeParam[id]