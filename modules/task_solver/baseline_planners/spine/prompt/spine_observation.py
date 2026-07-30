# -*- coding: utf-8 -*-
"""
SPINE Observation Formatting

Output example:
    Scene graph:
    objects: [Person-1, Vehicle-3, Cargo-2]
    regions: [Street Segment-10, campus-1, Hotel-1, Intersection-5]
    object_connections: [[Person-1, Street Segment-10], [Vehicle-3, campus-1]]
    region_connections: [[Street Segment-10, Intersection-5], [Intersection-5, campus-1]]
    robot_locations: {UAV-1: campus-1, Quadruped-1: Street Segment-10}
"""
from typing import Dict, List


# Categories that map to SPINE "regions"
_REGION_CATEGORIES = {"area", "building", "trans_facility", "district", "poi"}

# Edge types that represent object-to-region spatial relationships
_OBJECT_EDGE_TYPES = {"located_at", "stored_at", "located_in"}

# Edge types that represent region-to-region connectivity
_REGION_EDGE_TYPES = {"connects_to", "adjacent_to", "traversable"}

# Edge types that indicate a robot's location
_ROBOT_LOCATION_EDGE_TYPES = {"stationed_at", "located_at"}


def _display_name(node: Dict) -> str:
    """Pick the most informative display name for a node."""
    props = node.get("properties", {}) or {}
    if props.get("type") == "assembly_component":
        return props.get("subtype") or props.get("label") or str(node.get("id", "unknown"))
    return props.get("label", str(node.get("id", "unknown")))


def format_spine_observation(
    known_nodes: List[Dict],
    known_edges: List[Dict],
    robot_labels: List[str],
) -> str:
    """Format WorldModelManager's known_nodes/known_edges into SPINE scene graph format.

    Args:
        known_nodes: World model node list, each containing 'id' and 'properties'.
        known_edges: World model edge list, each containing 'source', 'target', 'type'.
        robot_labels: List of currently available robot labels.

    Returns:
        Formatted scene graph observation string prefixed with "Scene graph:".
    """
    # Build lookup maps
    id_to_label: Dict[str, str] = {}
    id_to_category: Dict[str, str] = {}

    for node in known_nodes:
        node_id = str(node.get("id", ""))
        props = node.get("properties", {}) or {}
        label = _display_name(node)
        category = props.get("category", "")
        id_to_label[node_id] = label
        id_to_category[node_id] = category

    # Classify nodes into objects and regions
    objects: List[str] = []
    regions: List[str] = []

    for node in known_nodes:
        node_id = str(node.get("id", ""))
        props = node.get("properties", {}) or {}
        category = props.get("category", "")
        label = id_to_label[node_id]

        # Skip robot nodes (represented via robot_locations)
        if category == "robot":
            continue

        if category == "prop":
            objects.append(label)
        elif category in _REGION_CATEGORIES:
            regions.append(label)
        else:
            # Unknown category — treat as region by default
            regions.append(label)

    # Classify edges
    object_connections: List[List[str]] = []
    region_connections: List[List[str]] = []
    robot_locations: Dict[str, str] = {}

    seen_obj_conns = set()
    seen_region_conns = set()

    for edge in known_edges:
        src_id = str(edge.get("source", ""))
        tgt_id = str(edge.get("target", ""))
        edge_type = edge.get("type", edge.get("relation", ""))

        src_label = id_to_label.get(src_id, src_id)
        tgt_label = id_to_label.get(tgt_id, tgt_id)
        src_category = id_to_category.get(src_id, "")
        tgt_category = id_to_category.get(tgt_id, "")

        # Robot location edges
        if src_category == "robot" and edge_type in _ROBOT_LOCATION_EDGE_TYPES:
            robot_locations[src_label] = tgt_label
            continue

        # Object-to-region connections (prop → region)
        if edge_type in _OBJECT_EDGE_TYPES:
            if src_category == "prop" and tgt_category in _REGION_CATEGORIES:
                key = (src_label, tgt_label)
                if key not in seen_obj_conns:
                    object_connections.append([src_label, tgt_label])
                    seen_obj_conns.add(key)
            elif tgt_category == "prop" and src_category in _REGION_CATEGORIES:
                key = (tgt_label, src_label)
                if key not in seen_obj_conns:
                    object_connections.append([tgt_label, src_label])
                    seen_obj_conns.add(key)
            continue

        # Region-to-region connections
        if edge_type in _REGION_EDGE_TYPES:
            if (src_category in _REGION_CATEGORIES or src_category == "") and \
               (tgt_category in _REGION_CATEGORIES or tgt_category == ""):
                key = tuple(sorted([src_label, tgt_label]))
                if key not in seen_region_conns:
                    region_connections.append(sorted([src_label, tgt_label]))
                    seen_region_conns.add(key)
            continue

    # Build output
    parts = ["Scene graph:"]
    parts.append(f"objects: [{', '.join(objects)}]")
    parts.append(f"regions: [{', '.join(regions)}]")

    # Format connections as [[a, b], [c, d], ...]
    obj_conn_strs = [f"[{c[0]}, {c[1]}]" for c in object_connections]
    parts.append(f"object_connections: [{', '.join(obj_conn_strs)}]")

    reg_conn_strs = [f"[{c[0]}, {c[1]}]" for c in region_connections]
    parts.append(f"region_connections: [{', '.join(reg_conn_strs)}]")

    # Robot locations
    if robot_labels:
        active_locs = {rl: robot_locations[rl] for rl in robot_labels if rl in robot_locations}
        if len(active_locs) == 1:
            rl, loc = next(iter(active_locs.items()))
            parts.append(f"robot_location: {loc}")
        elif len(active_locs) > 1:
            loc_strs = [f"{rl}: {loc}" for rl, loc in active_locs.items()]
            parts.append(f"robot_locations: {{{', '.join(loc_strs)}}}")
        else:
            parts.append(f"available_robots: [{', '.join(robot_labels)}]")

    return "\n".join(parts)
