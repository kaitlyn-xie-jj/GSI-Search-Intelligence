
from typing import Any, Dict, List, Optional, Tuple
import math
from collections import deque

def _is_intersection(node: Dict[str, Any]) -> bool:
    p = (node or {}).get('properties', {})
    return p.get('category') == 'trans_facility' and p.get('type') == 'intersection'

def _is_road_segment(node: Dict[str, Any]) -> bool:
    p = (node or {}).get('properties', {})
    return p.get('category') == 'trans_facility' and p.get('type') in ('street_segment', 'bridge')

def _nearest_intersection_id(ctx, ref: Any) -> Optional[str]:
    # Get reference point coordinates
    pos = None
    node = None

    # Coordinate form
    if isinstance(ref, dict) and "x" in ref and "y" in ref:
        pos = (float(ref["x"]), float(ref["y"]))
    elif isinstance(ref, (list, tuple)) and len(ref) == 2:
        pos = (float(ref[0]), float(ref[1]))
    else:
        # Node or node ID
        node = ref if isinstance(ref, dict) else ctx.get_node_by_id(str(ref))
        if not node:
            return None

        # If ref itself is an intersection, return it directly.
        if _is_intersection(node):
            return str(node.get("id"))

        pos = ctx._get_node_position(node)
        if not pos:
            return None

    # Find nearest among all intersections
    best_id, best_d = None, float("inf")
    for cand in ctx.get_nodes_by_type("intersection"):
        cpos = ctx._get_node_position(cand)
        if not cpos:
            continue
        d = math.hypot(cpos[0]-pos[0], cpos[1]-pos[1])
        if d < best_d:
            best_d, best_id = d, str(cand.get("id"))
    return best_id

def _have_traversable(ctx, a: str, b: str) -> bool:
    """
    Check if traversable between intersections a and b:
    Considers traversable if edge with {'type': 'traversable'} exists (any direction).
    """
    try:
        return (
            ctx.has_edge_of_type(a, b, 'traversable') or
            ctx.has_edge_of_type(b, a, 'traversable')
        )
    except Exception:
        # Fallback when has_edge_of_type unavailable (full table scan)
        for e in ctx.get_all_edges():
            if e.get('type') == 'traversable':
                s, t = str(e.get('source')), str(e.get('target'))
                if (s == str(a) and t == str(b)) or (s == str(b) and t == str(a)):
                    return True
        return False

def _intersection_graph(ctx, require_traversable: bool = True) -> Dict[str, set]:
    """
    Compress (street_segment/bridge) and intersection connects_to into
    an undirected intersection-to-intersection graph; also requires a traversable edge between the intersection pair.
    """
    connects = ctx.get_edges_by_type('connects_to')
    seg_to_inters: Dict[str, set] = {}
    for e in connects:
        src, tgt = str(e.get('source')), str(e.get('target'))
        src_node = ctx.get_node_by_id(src)
        tgt_node = ctx.get_node_by_id(tgt)
        if src_node and tgt_node and _is_road_segment(src_node) and _is_intersection(tgt_node):
            seg_to_inters.setdefault(src, set()).add(tgt)

    graph: Dict[str, set] = {}

    def add_edge(a: str, b: str):
        if a == b:
            return
        graph.setdefault(a, set()).add(b)
        graph.setdefault(b, set()).add(a)

    # Only connect a-b if they belong to same road segment and have traversable edge
    for _, inters in seg_to_inters.items():
        L = list(inters)
        for i in range(len(L)):
            for j in range(i + 1, len(L)):
                a, b = L[i], L[j]
                if not require_traversable or _have_traversable(ctx, a, b):
                    add_edge(a, b)

    # Ensure isolated intersections are in graph
    for inter in ctx.get_nodes_by_type('intersection'):
        iid = str(inter.get('id'))
        graph.setdefault(iid, graph.get(iid, set()))
    return graph

def find_path(ctx, src_id: str, dst_id: str,  require_traversable: bool = True) -> List[str]:
    """
    Return the shortest traversable path consisting only of intersections.
    Maps src/dst to nearest intersection if non-intersection.
    """
    s = _nearest_intersection_id(ctx, str(src_id))
    t = _nearest_intersection_id(ctx, str(dst_id))
    if not s or not t:
        return []
    if s == t:
        return [s]

    graph = _intersection_graph(ctx, require_traversable=require_traversable)
    q = deque([s])
    prev = {s: None}
    while q:
        u = q.popleft()
        if u == t:
            break
        for v in graph.get(u, ()):
            if v not in prev:
                prev[v] = u
                q.append(v)

    if t not in prev:
        return []

    # Backtrack
    path = []
    cur = t
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path

def find_side_edges_near_path(ctx, src_id: str, dst_id: str) -> List[Dict[str, Any]]:
    """
    Along the shortest path, find "nearby but non-critical" edges:
      - For each segment (I_k, I_{k+1}), find road segment nodes (street_segment/bridge) connecting the two intersections.
      - Collect `adjacent_to` edges from these road segments to areas/buildings as disturbable candidates.
    Returns list of edge dictionaries (adapts to upper-level DELETE_EDGE edge_data).
    """
    path = find_path(ctx, src_id, dst_id)
    if len(path) < 2:
        return []

    connects = ctx.get_edges_by_type('connects_to')
    seg_to_inters: Dict[str, set] = {}
    for e in connects:
        s, t = str(e.get('source')), str(e.get('target'))
        s_node, t_node = ctx.get_node_by_id(s), ctx.get_node_by_id(t)
        if _is_road_segment(s_node) and _is_intersection(t_node):
            seg_to_inters.setdefault(s, set()).add(t)

    carrier_segments: set = set()
    for i in range(len(path) - 1):
        a, b = path[i], path[i + 1]
        for seg, inters in seg_to_inters.items():
            if a in inters and b in inters:
                carrier_segments.add(seg)

    results: List[Dict[str, Any]] = []
    for seg in carrier_segments:
        for e in ctx.get_edges_from_node(seg, edge_type='adjacent_to'):
            tgt_node = ctx.get_node_by_id(str(e.get('target')))
            p = (tgt_node or {}).get('properties', {})
            if p.get('category') in ('area', 'building'):
                results.append({
                    'source': e.get('source'),
                    'target': e.get('target'),
                    'type': 'adjacent_to'
                })
    return results

def find_candidate_locations(ctx, object_or_carrier_id: str, prefer_parking: bool = True) -> List[Dict[str, Any]]:
    """
    Return candidate location dictionaries writable as `properties.location`:
      {'category':'building','type':'parking','label':'Parking-1'}
    Prioritizes parking/staging/base buildings; sorted by geometric distance from current object location.
    """
    obj = ctx.get_node_by_id(str(object_or_carrier_id))
    if not obj:
        return []

    preferred_types = ['parking', 'staging', 'robot_base'] if prefer_parking else ['robot_base', 'staging', 'parking']

    buildings = ctx.get_all_buildings()
    buckets = {t: [] for t in preferred_types}
    others: List[Dict[str, Any]] = []
    for b in buildings:
        t = (b.get('properties') or {}).get('type')
        if t in buckets:
            buckets[t].append(b)
        else:
            others.append(b)

    ordered: List[Dict[str, Any]] = []
    for t in preferred_types:
        ordered.extend(buckets[t])
    ordered.extend(others)

    obj_pos = ctx._get_node_position(obj)

    def dist(n: Dict[str, Any]) -> float:
        p = ctx._get_node_position(n)
        return math.hypot(p[0] - obj_pos[0], p[1] - obj_pos[1])

    ordered.sort(key=dist)

    out: List[Dict[str, Any]] = []
    for b in ordered:
        bp = b.get('properties', {})
        label = bp.get('label'); btype = bp.get('type')
        if label and btype:
            out.append({'category': 'building', 'type': btype, 'label': label})
    return out
