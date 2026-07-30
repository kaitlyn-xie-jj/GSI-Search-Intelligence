from typing import Dict, List, Tuple, Optional
from random import Random

from modules.config import Category, TransFacilityType, EdgeType
from modules.dataset_builder.scene_utils.scene_graph_helpers import center_of_shape

# ----- Utilities -----
def _build_node_index(nodes: List[Dict]) -> Dict[str, Dict]:
    return {n["id"]: n for n in nodes}

def _parent_of(node_id: str, edges: List[Dict], kinds=(
    EdgeType.LOCATED_AT.value, EdgeType.LOCATED_IN.value
)) -> Optional[str]:
    for e in edges:
        if e.get("type") in kinds and e.get("source") == node_id:
            return e.get("target")
    return None

def _xy(node: Dict) -> Tuple[float, float]:
    cx, cy = center_of_shape(node.get("shape", {}))
    return float(cx), float(cy)

def _is_road(node: Dict) -> bool:
    if node is None: return False
    p = node.get("properties", {})
    return p.get("category") == Category.TRANS_Facility.value and \
           p.get("type") == TransFacilityType.STREET_SEGMENT.value

def _is_intersection(node: Dict) -> bool:
    if node is None: return False
    p = node.get("properties", {})
    return p.get("category") == Category.TRANS_Facility.value and \
           p.get("type") == TransFacilityType.INTERSECTION.value

def _is_bridge(node: Dict) -> bool:
    if node is None: return False
    p = node.get("properties", {})
    return p.get("category") == Category.TRANS_Facility.value and \
           p.get("type") == TransFacilityType.BRIDGE.value

def _is_parking_building(node: Dict) -> bool:
    if node is None: return False
    p = node.get("properties", {})
    return p.get("category") == Category.BUILDING.value and p.get("type") == "parking"

def _is_power_station(node: Dict) -> bool:
    if node is None: return False
    p = node.get("properties", {})
    return p.get("category") == Category.BUILDING.value and p.get("type") == "power_station"


# ============================
#        Unified annotation entry
# ============================
def annotate_attributes(
    nodes: List[Dict],
    edges: List[Dict],
    rng: Random,
    group_radius: float = 6.0,
    p_suspicious: float = 0.5,
    p_injured: float = 0.5,
    p_hazard: float = 0.3,
) -> None:
    """
    Annotate attribute labels on nodes:
      - Non-prop (building/infrastructure): is_fire / is_spill (robot_base always False)
      - Prop:
          * person: suspicious / injured / crowd (based on location clustering)
          * vehicle: illegal_parking / traffic_violation; random is_fire / is_spill
          * boat:    random is_fire / is_spill
          * assembly_component: is_installed=False, parent_component=None
          * fire / equipment_failure / hazmat: is_handled=False
    """

    idx = _build_node_index(nodes)

    # ========= Pass 1: per-node attributes by category and type =========
    # Also record person/vehicle for secondary computation (crowd clustering / vehicle context)
    persons: List[Tuple[Dict, Optional[Dict], Tuple[float, float]]] = []
    vehicles: List[Tuple[Dict, Optional[Dict]]] = []

    for n in nodes:
        props = n.get("properties", {})
        cat = props.get("category")
        ntype = props.get("type")

        # --- Non-prop: environmental hazards ---
        if cat in (Category.BUILDING.value, Category.TRANS_Facility.value):
            if cat == Category.BUILDING.value and ntype == "robot_base":
                props["is_fire"] = False
                props["is_spill"] = False
            else:
                props["is_fire"]  = (rng.random() < p_hazard)
                props["is_spill"] = (rng.random() < p_hazard)
            continue

        # --- Prop: dispatch by type ---
        if cat != Category.PROP.value:
            continue

        # person: set individual attributes first; crowd computed later
        if ntype == "person":
            props["suspicious"] = (rng.random() < p_suspicious)
            props["injured"]    = (rng.random() < p_injured)
            props["crowd"]      = False  # Initially False; clustering may set True later
            pid = _parent_of(n["id"], edges)
            parent = idx.get(pid) if pid else None
            persons.append((n, parent, _xy(n)))
            continue

        # vehicle: is_fire/is_spill set immediately; parking/violation determined later by context
        if ntype == "vehicle":
            props["is_fire"]  = (rng.random() < p_hazard)
            props["is_spill"] = (rng.random() < p_hazard)
            pid = _parent_of(n["id"], edges)
            parent = idx.get(pid) if pid else None
            vehicles.append((n, parent))
            continue

        # boat: random hazards
        if ntype == "boat":
            props["is_fire"]  = (rng.random() < p_hazard)
            props["is_spill"] = (rng.random() < p_hazard)
            continue

        # Component: default not installed, no parent component
        if ntype == "assembly_component":
            props.setdefault("is_installed", False)
            props.setdefault("parent_component", None)
            continue

        # Event-type props: default not handled
        if ntype in ("fire", "equipment_failure", "hazmat"):
            props.setdefault("is_handled", False)
            continue

    # ========= Pass 2: crowd annotation for persons (only under specific parent nodes) =========
    def _allowed_parent(p: Optional[Dict]) -> bool:
        return _is_intersection(p) or _is_road(p) or _is_bridge(p) or _is_power_station(p)

    # Bucket by same parent node (location)
    buckets: Dict[str, List[Tuple[Dict, Tuple[float, float]]]] = {}
    for n, parent, xy in persons:
        if _allowed_parent(parent):
            key = parent["id"] if parent else "none"
            buckets.setdefault(key, []).append((n, xy))

    # Radius clustering within each bucket: connected components of size >=5 get crowd=True
    for _, items in buckets.items():
        m = len(items)
        if m < 5:
            # Small bucket: keep as False
            for n, _ in items:
                n["properties"]["crowd"] = False
            continue

        # O(k^2) adjacency
        adj = [[] for _ in range(m)]
        for i in range(m):
            xi, yi = items[i][1]
            for j in range(i + 1, m):
                xj, yj = items[j][1]
                dx, dy = xi - xj, yi - yj
                if (dx*dx + dy*dy) <= (group_radius * group_radius):
                    adj[i].append(j)
                    adj[j].append(i)

        # Connected components
        seen = [False] * m
        comps: List[List[int]] = []
        for i in range(m):
            if seen[i]: continue
            stack = [i]; seen[i] = True; comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
            comps.append(comp)

        big = set(u for comp in comps if len(comp) >= 5 for u in comp)
        for i, (n, _) in enumerate(items):
            n["properties"]["crowd"] = (i in big)

    # If no crowd=True at all, fallback: randomly set some persons to True (helps trigger downstream tasks)
    if persons and not any(n["properties"].get("crowd", False) for n, _, _ in persons):
        k = max(1, min(5, len(persons)))
        for i in rng.sample(range(len(persons)), k):
            persons[i][0]["properties"]["crowd"] = True

    # ========= Pass 3: vehicle illegal parking / traffic violation (depends on parent node context) =========
    for n, parent in vehicles:
        on_road        = _is_road(parent)
        in_parking     = _is_parking_building(parent)
        at_intersection= _is_intersection(parent)
        illegal_parking = not (on_road or in_parking)
        traffic_violation = bool(at_intersection)
        n["properties"]["illegal_parking"]  = illegal_parking
        n["properties"]["traffic_violation"] = traffic_violation