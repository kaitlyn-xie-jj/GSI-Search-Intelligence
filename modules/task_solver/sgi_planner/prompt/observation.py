# -*- coding: utf-8 -*-
"""
Observation Formatting
"""
from typing import Dict, List


def _display_name(node: Dict) -> str:
    """Pick the most informative display name for a node."""
    props = node.get("properties", {}) or {}
    if props.get("type") == "assembly_component":
        return props.get("subtype") or props.get("label") or str(node.get("id", "unknown"))
    return props.get("label", str(node.get("id", "unknown")))


def format_observation(
    known_nodes: List[Dict],
    known_edges: List[Dict],
    robot_labels: List[str],
) -> str:
    """Format WorldModelManager's known_nodes and known_edges into an observation string.

    Notes:
        - Robots appear in Nodes with their status, e.g. `UAV-1(idle)`.
          Their location is conveyed via edges (e.g., `stationed_at`).
        - Assembly components are identified by their `subtype` (e.g., `wall_panel`)
          rather than their generic label (e.g., `Assembly Component-2`).

    Args:
        known_nodes: World model node list, each node containing 'id' and 'properties' fields.
        known_edges: World model edge list, each edge containing 'source' and 'target' fields.
        robot_labels: List of currently available robot labels, e.g., ['UAV-1', 'UGV-1'].

    Returns:
        Formatted observation string.
    """
    parts = ["Environment observation:"]

    # -- Nodes (robots show status: e.g. `UAV-1(idle)`) --
    node_strs = []
    for node in known_nodes:
        props = node.get("properties", {}) or {}
        name = _display_name(node)
        if props.get("category") == "robot":
            node_strs.append(f"{name}({props.get('status', 'active')})")
        else:
            node_strs.append(name)
    parts.append(f"Known Nodes: [{', '.join(node_strs)}]" if node_strs else "Nodes: []")

    # -- Edges (compact: `src relation tgt`; merge same `(relation, tgt)` group) --
    id_to_display = {str(n.get("id", "")): _display_name(n) for n in known_nodes}

    # Group sources by (relation, target) while preserving first-seen order.
    grouped: "Dict[tuple, List[str]]" = {}
    for edge in known_edges:
        src = id_to_display.get(str(edge.get("source", "")), str(edge.get("source", "")))
        tgt = id_to_display.get(str(edge.get("target", "")), str(edge.get("target", "")))
        relation = edge.get("relation", edge.get("type", "related_to"))
        if relation in ("stationed_at", "located_at", "stored_at"):
            grouped.setdefault((relation, tgt), []).append(src)

    edge_strs = []
    for (relation, tgt), srcs in grouped.items():
        if len(srcs) == 1:
            edge_strs.append(f"{srcs[0]} {relation} {tgt}")
        else:
            edge_strs.append(f"{', '.join(srcs)} all {relation} {tgt}")

    parts.append(f"Known Edges: [{', '.join(edge_strs)}]" if edge_strs else "Edges: []")

    return "\n".join(parts)
