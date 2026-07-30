# scene_graph_builder.py
from typing import Dict, List, Tuple, Optional, Any
from random import Random
import math
from shapely.geometry import box as shp_box, Polygon as ShpPolygon, Point as ShpPoint, LineString
from shapely.ops import unary_union
from shapely.strtree import STRtree

from modules.config import (
    Category, EdgeType,
    TransFacilityType, POIType,
)
from modules.config.managers import UnifiedTemplateManager
from modules.dataset_builder.scene_utils.pcg_generators import generate_zones_voronoi, skeleton_roads_from_polygons
from modules.dataset_builder.scene_utils.geometry_utils import (
    rect_from_center, segment_normal, point_along_segment, offset_point
)

from .scene_graph_helpers import (
    bounds_from_rect,
    center_of_shape,
    footprint_from_size_at_center,
    footprint_from_template,
    half_extent_normal,
    child_shape_within_parent,
    choose_attr_value,
    decorate_human,
    decorate_car,
    decorate_boat,
    decorate_cargo,
    decorate_assembly_component,
    snap_on_road_center,
    place_in_building_any,
    place_in_open_area,
    nearest_district_id,
    compute_boat_center,
    location_dict,
    find_containing_or_nearest_area,
)
from modules.dataset_builder.scene_utils.annotate_attributes import annotate_attributes

class SceneGraphBuilder:
    """
    End-to-end procedural Cybertown builder:
      1) generate districts
      2) generate major areas inside districts (water, parks, zones)
      3) generate primary road network
      4) populate zones (secondary roads + buildings)
      5) add POIs
      6) connect relationships (edges)
    """
    def __init__(self, unified_template_manager: UnifiedTemplateManager, rng_seed: Optional[int] = None):
        self.tmpl = unified_template_manager
        self.nodes: List[Dict] = []
        self.edges: List[Dict] = []
        self._placed_areas_bboxes: List[Dict] = []  # for overlap checks (rect only)
        self._id_counter = 0
        self.rng = Random(rng_seed)
        self.world_bounds = None  # Global bounding box

        # caches
        self._districts: Dict[str, Dict] = {}
        self._nodes_by_category: Dict[str, List[Dict]] = {}

        self.non_buildable_area_types = {"water_body", "garden", "square", "greenbelt"}
        self._non_buildable_polys = []  # list[List[(x,y)]]

        # area caches
        self._areas_by_type: Dict[str, List[ShpPolygon]] = {}
        self._area_unions: Dict[str, ShpPolygon] = {}   # e.g. "garden" -> union polygon
        self._adjacency_tol = 12.0   # meters, used for "adjacent to garden" softness

    # --------- utilities ---------
    def _uid(self) -> str:
        self._id_counter += 1
        return f"{self._id_counter:0d}"

    def _push_node(self, node: Dict):
        self.nodes.append(node)
        cat = node["properties"].get("category")
        self._nodes_by_category.setdefault(cat, []).append(node)

    def _add_edge(self, src_id: str, dst_id: str, etype: str, props: Optional[Dict]=None):
        e = {"source": src_id, "target": dst_id, "type": etype}
        if props: e["properties"] = props
        self.edges.append(e)

    # --------- phases ---------
    def generate_districts_from_layout(self, district_layout: Dict[str, Dict]):
        """Use your existing layout (bounds + description) to create district nodes."""
        for district_id, info in district_layout.items():
            bounds = info["bounds"]
            center = [(bounds['x_min'] + bounds['x_max'])/2, (bounds['y_min'] + bounds['y_max'])/2]
            shape = {"type": "rectangle",
                     "min_corner": [bounds['x_min'], bounds['y_min']],
                     "max_corner": [bounds['x_max'], bounds['y_max']]}
            node = {
                "id": self._uid(),
                "properties": {
                    "category": Category.DISTRICT.value,
                    "type": "district",
                    "label": district_id,
                    "illumination": "bright",
                    "description": info.get("description", f"District {district_id}")
                },
                "shape": shape
            }
            self._districts[district_id] = node
            self._push_node(node)

        bxs = []
        for d in district_layout.values():
            b = d["bounds"]
            bxs.append(shp_box(b["x_min"], b["y_min"], b["x_max"], b["y_max"]))
        self.world_bounds = unary_union(bxs).envelope  # Degenerate to a single bbox

    def generate_major_areas(self, areas_per_district: Dict[str, Dict[str, int]]):
        """
        For each district, generate target counts of areas by type.
        areas_per_district example:
           {"center_district": {"water_body": 1, "garden": 2}}
        """
        for d_id, counts in areas_per_district.items():
            district_node = self._districts[d_id]
            db = bounds_from_rect(district_node["shape"])  
            # Step 1: union all requested area polygons via Voronoi (fallback-ready)
            total_cells = max(1, sum(counts.values()))
            polygons = generate_zones_voronoi(db, total_cells, rng=self.rng) 
            # assign types
            types = []
            for t, c in counts.items():
                types += [t] * c
            self.rng.shuffle(types)
            for poly, area_type in zip(polygons, types):
                if len(poly) < 3:
                    continue
                node = {
                    "id": self._uid(),
                    "properties": {
                        "category": Category.AREA.value,
                        "type": area_type,
                        "label": f"{area_type}-{self._seq_for_type(area_type)}",
                        "district": d_id,
                        "status": self.tmpl.get_default_status(Category.AREA.value, area_type),
                        "passability": "traversable",   # traversable / restricted
                        "visibility": "high",      # high / low
                        "wind_condition": "weak",  # weak / strong
                    },
                    "shape": {"type": "polygon", "vertices": [(float(x), float(y)) for x, y in poly]}
                }
                self._push_node(node)
                self._add_edge(node["id"], district_node["id"], EdgeType.LOCATED_IN.value)

                self._areas_by_type.setdefault(area_type, []).append(ShpPolygon(node["shape"]["vertices"]))

                if area_type in self.non_buildable_area_types:
                    self._non_buildable_polys.append(node["shape"]["vertices"])

        self._recompute_buildable_union()

    def generate_primary_roads(self):
        """Use area polygons as skeleton to form primary streets and intersections."""
        area_nodes = [n for n in self._nodes_by_category.get(Category.AREA.value, [])
                      if n["shape"]["type"] == "polygon"]
        area_polys = [ShpPolygon(n["shape"]["vertices"]) for n in area_nodes]

        area_index = STRtree(area_polys)
        idx_to_node_id = [n["id"] for n in area_nodes]
        geom_id_to_node_id = {id(g): n["id"] for g, n in zip(area_polys, area_nodes)}

        adj_tol = getattr(self, "_road_area_adj_tol", 1.5)

        segments, intersections = skeleton_roads_from_polygons(area_polys)

        # intersections
        id_map = {}
        for p in intersections:
            node = {
                "id": self._uid(),
                "properties": {"category": Category.TRANS_Facility.value,
                               "type": TransFacilityType.INTERSECTION.value,
                               "label": f"Intersection-{self._seq_for_type('intersection')}",
                               "status": self.tmpl.get_default_status(Category.TRANS_Facility.value, TransFacilityType.INTERSECTION.value),
                               "visibility": "high",     # high / low
                               "wind_condition": "weak",   # weak / strong
                               "congestion": "none",     # vehicle / crowd / none
                },
                "shape": {"type": "point", "center": [float(p[0]), float(p[1])]}
            }
            self._push_node(node)
            id_map[(round(p[0], 3), round(p[1], 3))] = node["id"]

        # street segments
        for a, b in segments:
            seg_node = {
                "id": self._uid(),
                "properties": {"category": Category.TRANS_Facility.value,
                               "type": TransFacilityType.STREET_SEGMENT.value,
                               "label": f"Street Segment-{self._seq_for_type('street_segment')}",
                               "status": self.tmpl.get_default_status(Category.TRANS_Facility.value, TransFacilityType.STREET_SEGMENT.value),
                               "visibility": "high",     # high / low
                               "wind_condition": "weak",   # weak / strong
                               "congestion": "none",     # vehicle / crowd / none
                },
                "shape": {"type": "linestring", "points": [[float(a[0]), float(a[1])],
                                                          [float(b[0]), float(b[1])]]}
            }
            self._push_node(seg_node)

            a_id = id_map.get((round(a[0], 3), round(a[1], 3)))
            b_id = id_map.get((round(b[0], 3), round(b[1], 3)))
            if a_id:
                self._add_edge(seg_node["id"], a_id, EdgeType.CONNECTS_TO.value)
            if b_id:
                self._add_edge(seg_node["id"], b_id, EdgeType.CONNECTS_TO.value)

            # If the segment connects two intersections, add a “traversable” edge between them
            if a_id and b_id and a_id != b_id:
                lo, hi = sorted((a_id, b_id), key=int)
                key = (lo, hi)   # dedup key

                traversable_cache = getattr(self, "_traversable_pairs", None)
                if traversable_cache is None:
                    self._traversable_pairs = set()
                    traversable_cache = self._traversable_pairs

                if key not in traversable_cache:
                    # Convention: store only one undirected edge — smaller ID as source, larger as target
                    self._add_edge(lo, hi, EdgeType.TRAVERSABLE.value)
                    traversable_cache.add(key)

            ls = LineString([(float(a[0]), float(a[1])), (float(b[0]), float(b[1]))])
            ls_band = ls.buffer(adj_tol, cap_style=2)

            candidates = area_index.query(ls_band)

            touched_ids = set()
            for cand in candidates:
                try:
                    if not cand.intersects(ls_band):
                        continue
                    nid = geom_id_to_node_id.get(id(cand))
                    if nid:
                        touched_ids.add(nid)
                except AttributeError:
                    idx = int(cand)
                    poly = area_polys[idx]
                    if not poly.intersects(ls_band):
                        continue
                    touched_ids.add(idx_to_node_id[idx])

            for aid in touched_ids:
                self._add_edge(seg_node["id"], aid, EdgeType.ADJACENT_TO.value)

    def generate_bridges(self):
        """
        Conditions:
        - The water body polygon must have more than 3 vertices
        - Randomly pick a pair of “diagonal” vertices as bridge endpoints (only use existing intersections; do not create new ones)
        - A bridge is built only if both vertices already have existing intersections
        """
        # Get all water_body polygons
        water_nodes = [
            n for n in self._nodes_by_category.get(Category.AREA.value, [])
            if n.get("shape", {}).get("type") == "polygon"
            and n.get("properties", {}).get("type") == "water_body"
        ]

        for wn in water_nodes:
            verts = list(wn["shape"]["vertices"])
            if len(verts) <= 3:
                continue

            # Remove duplicate if first and last vertices are the same
            if verts[0] == verts[-1]:
                verts = verts[:-1]
            n = len(verts)
            if n <= 3:
                continue

            # Choose a pair of "diagonal" vertices:
            i = self.rng.randrange(n)
            candidates = [k for k in range(n) if k != i and (k - i) % n not in (1, n - 1)] # Candidate set: all non-self, non-adjacent vertices
            if not candidates:
                continue  # Degenerate case, skip this water body

            # Randomly pick one from the candidate set
            j = self.rng.choice(candidates)

            a = (float(verts[i][0]), float(verts[i][1]))
            b = (float(verts[j][0]), float(verts[j][1]))

            # Only use existing intersections; vertices must already have an Intersection
            a_id = self._find_intersection_id_by_point(*a)
            b_id = self._find_intersection_id_by_point(*b)
            if not (a_id and b_id) or a_id == b_id:
                continue  # Skip this water body if two distinct intersections are not found

            # Create bridge node
            bridge = {
                "id": self._uid(),
                "properties": {
                    "category": Category.TRANS_Facility.value,
                    "type": TransFacilityType.BRIDGE.value,
                    "label": f"Bridge-{self._seq_for_type('bridge')}",
                    "status": self.tmpl.get_default_status(Category.TRANS_Facility.value, TransFacilityType.BRIDGE.value),
                    "visibility": "high",     # high / low
                    "wind_condition": "weak",   # weak / strong
                    "congestion": "none",     # vehicle / crowd / none
                },
                "shape": {"type": "linestring", "points": [[a[0], a[1]], [b[0], b[1]]]}
            }
            self._push_node(bridge)

            # Establish connects_to with both endpoint intersections
            self._add_edge(bridge["id"], a_id, EdgeType.CONNECTS_TO.value)
            self._add_edge(bridge["id"], b_id, EdgeType.CONNECTS_TO.value)

            # Create a unique undirected traversable edge between the two intersections (smaller ID as source, larger as target)
            lo, hi = sorted((a_id, b_id), key=int)
            key = (lo, hi)
            traversable_cache = getattr(self, "_traversable_pairs", None)
            if traversable_cache is None:
                self._traversable_pairs = set()
                traversable_cache = self._traversable_pairs
            if key not in traversable_cache:
                self._add_edge(lo, hi, EdgeType.TRAVERSABLE.value)
                traversable_cache.add(key)

            # Establish adjacent_to between the bridge and the water body (preserve semantics)
            self._add_edge(bridge["id"], wn["id"], EdgeType.ADJACENT_TO.value)
    

    def populate_buildings_along_streets(self, building_plan: Dict[str, int],
                                         align_offset: float = 10.0,
                                         road_clearance: float = 4.0) -> bool:
        streets = [n for n in self._nodes_by_category.get(Category.TRANS_Facility.value, [])
                   if n["properties"]["type"] == TransFacilityType.STREET_SEGMENT.value]

        order = []
        if "robot_base" in building_plan: order.append(("robot_base", building_plan["robot_base"]))
        for k, v in building_plan.items():
            if k != "robot_base": order.append((k, v))

        for btype, total in order:
            def _try_place_one_candidate(seg_choice=True, internal_choice=True, require_adjacent_park=False):
                nonlocal placed

                # Phase 1: along street
                attempts = 0
                if seg_choice:
                    max_attempts_street = max(50, (total - placed) * 30)
                    while placed < total and attempts < max_attempts_street and streets:
                        attempts += 1
                        seg = self.rng.choice(streets)
                        p1, p2 = tuple(seg["shape"]["points"][0]), tuple(seg["shape"]["points"][1])
                        t = 0.1 + 0.8 * self.rng.random()
                        base = point_along_segment(p1, p2, t)
                        nvec = segment_normal(p1, p2)

                        center = offset_point(base, nvec, align_offset + half_norm)
                        shape_dict, geom = footprint_from_template(center, template)

                        if not self._geom_inside_world(geom):             continue
                        if self._overlaps_non_buildable_geom(geom):       continue
                        if self._overlaps_existing_buildings(geom):       continue
                        if self._intersects_any_road(geom, clearance=road_clearance): continue

                        if btype in ("power_station", "parking") and self._intersects_area_type(geom, "campus"):
                            continue

                        if btype == "parking" and require_adjacent_park:
                            if not self._is_adjacent_to_area(geom, "garden", self._adjacency_tol):
                                continue

                        label = f"{btype.replace('_',' ').title()}-{self._seq_for_type(btype)}"
                        node = {
                            "id": self._uid(),
                            "properties": {
                                "category": Category.BUILDING.value,
                                "type": btype,
                                "label": label,
                                "status": self.tmpl.get_default_status(Category.BUILDING.value, btype),
                                "visibility": "high",     # high / low
                                "wind_condition": "weak",   # weak / strong
                            },
                            "shape": shape_dict,
                        }
                        self._push_node(node)
                        placed += 1

                # Phase 2: internal (buildable union)
                attempts = 0
                if internal_choice:
                    max_attempts_internal = max(50, (total - placed) * 30)
                    while placed < total and attempts < max_attempts_internal:
                        attempts += 1
                        c = self._sample_point_in_buildable(margin=half_norm)
                        if c is None:
                            break
                        shape_dict, geom = footprint_from_template(c, template)

                        if not self._geom_inside_world(geom):             continue
                        if self._overlaps_non_buildable_geom(geom):       continue
                        if self._overlaps_existing_buildings(geom):       continue
                        if self._intersects_any_road(geom, clearance=road_clearance): continue

                        if btype in ("power_station", "parking") and self._intersects_area_type(geom, "campus"):
                            continue

                        if btype == "parking" and require_adjacent_park:
                            if not self._is_adjacent_to_area(geom, "garden", self._adjacency_tol):
                                continue

                        label = f"{btype.replace('_',' ').title()}-{self._seq_for_type(btype)}"
                        node = {
                            "id": self._uid(),
                            "properties": {
                                "category": Category.BUILDING.value,
                                "type": btype,
                                "label": label,
                                "status": self.tmpl.get_default_status(Category.BUILDING.value, btype),
                                "visibility": "high",     # high / low
                                "wind_condition": "weak",   # weak / strong
                            },
                            "shape": shape_dict,
                        }
                        self._push_node(node)
                        placed += 1

            placed = 0
            template = self.tmpl.get_template(Category.BUILDING.value, btype)
            half_norm = half_extent_normal(template) 

            if btype == "parking":
                _try_place_one_candidate(require_adjacent_park=True)
                if placed < total:
                    _try_place_one_candidate(require_adjacent_park=False)
            else:
                _try_place_one_candidate(require_adjacent_park=False)

            if btype == "robot_base" and placed <= 0:
                print("[Warn] Required building 'robot_base' could not be placed in the scene.")
                return False

            if placed < total:
                print(f"[Info] Capacity limited: placed {placed}/{total} for '{btype}'.")

        return True 

    def generate_props_and_robots(
        self,
        robot_plan: Optional[Dict[str, int]] = None,
        prop_plan: Optional[Dict[str, int]] = None,
        **prop_kwargs
    ) -> None:
        self.spawn_robots(robot_plan)
        self.populate_props(prop_plan, **prop_kwargs)
        annotate_attributes(self.nodes, self.edges, self.rng, group_radius=120, p_suspicious=0.5, p_injured=0.1) # Annotate event attributes for persons/vehicles: illegal parking/violation/gathering/suspicious

    def spawn_robots(
        self,
        plan: Optional[Dict[str, int]] = None,
        base_type: str = "robot_base",
        per_robot_max_trials: int = 80,
        padding: float = 0.6
    ) -> None:
        buildings = self._nodes_by_category.get(Category.BUILDING.value, [])
        streets = [
            n for n in self._nodes_by_category.get(Category.TRANS_Facility.value, [])
            if n["properties"].get("type") in (TransFacilityType.STREET_SEGMENT.value, TransFacilityType.BRIDGE.value)
        ]
        intersections = [
            n for n in self._nodes_by_category.get(Category.TRANS_Facility.value, [])
            if n["properties"].get("type") == TransFacilityType.INTERSECTION.value
        ]

        if not buildings and not streets and not intersections:
            print("[Robots] No placement parents available; skip.")
            return

        if plan is None:
            num_bases = max(1, len([b for b in buildings if b["properties"].get("type") == base_type]))
            plan = {"UAV": 2 * num_bases, "UGV": 2 * num_bases, "Quadruped": 1 * num_bases, "Humanoid": 1 * num_bases}

        for rtype, count in plan.items():
            tpl = self.tmpl.get_template(Category.ROBOT.value, rtype)
            if not tpl:
                print(f"[Robots] Missing robot template: {rtype}")
                continue
            size = tpl.get("size", {"width": 1.0, "length": 1.0})
            default_status = self.tmpl.get_default_status(Category.ROBOT.value, rtype)

            for _ in range(count):
                placed = False
                for _try in range(per_robot_max_trials):
                    pick_trans = (not buildings) or (streets or intersections) and (self.rng.random() < 0.5)
                    if not pick_trans and buildings:
                        # Place inside a building
                        parent, shape, _ = place_in_building_any(buildings, size, self.rng, padding=padding)
                        if not parent:
                            continue
                    else:
                        # Place on a road or intersection
                        parent, center = snap_on_road_center(streets, intersections, self.rng, seg_prob=0.6)
                        if parent is None or center is None:
                            continue
                        shape, _ = footprint_from_size_at_center(center, size)

                    label = f"{rtype}-{self._seq_for_type(rtype)}"
                    node = {
                        "id": self._uid(),
                        "properties": {
                            "category": Category.ROBOT.value,
                            "type": rtype,
                            "label": label,
                            "status": default_status,
                            "battery_level": 100.0,
                            "comm": "clear",
                            "location": self._infer_location(shape, parent),
                        },
                        "shape": shape,
                    }
                    self._push_node(node)
                    p_cat = parent["properties"]["category"]
                    p_type = parent["properties"].get("type")
                    edge_type = EdgeType.STATIONED_AT.value if (p_cat == Category.BUILDING.value and p_type == base_type) else EdgeType.LOCATED_AT.value
                    self._add_edge(node["id"], parent["id"], edge_type)
                    placed = True
                    break

                if not placed:
                    print(f"[Robots] Failed to place one {rtype} after trials.")

    def populate_props(
        self,
        plan: Optional[Dict[str, int]] = None,
        p_building_human: float = 0.5,
        p_road_human: float = 0.3,
        p_road_car: float = 0.8,
        open_margin: float = 0.6,
    ) -> None:
        # ---------- Base collections ----------
        buildings_all = self._nodes_by_category.get(Category.BUILDING.value, [])
        buildings_no_robotbase = [b for b in buildings_all if b["properties"].get("type") != "robot_base"]
        hotels = [b for b in buildings_all if b["properties"].get("type") == "hotel"]
        parkings = [b for b in buildings_all if b["properties"].get("type") == "parking"]
        streets = [n for n in self._nodes_by_category.get(Category.TRANS_Facility.value, [])
                if n["properties"].get("type") == TransFacilityType.STREET_SEGMENT.value]
        intersections = [n for n in self._nodes_by_category.get(Category.TRANS_Facility.value, [])
                        if n["properties"].get("type") == TransFacilityType.INTERSECTION.value]

        # ---------- Default quotas ----------
        if plan is None:
            plan = {
                "person": 12,
                "vehicle": 10,
                "cargo": 6,
                "boat": len(self._areas_by_type.get("water_body", [])),  # 1 boat per water body
                "equipment_failure": 2,
                "fire": 2,
                "hazmat": 4,                 
                "assembly_component": 0,     
            }

        def _new_prop_node(ptype: str, label: str, shape: Dict, status: str, extra: Optional[Dict] = None):
            props = {
                "category": Category.PROP.value,
                "type": ptype,
                "label": label,
                "status": status,
            }
            if extra:
                props.update(extra)
            node = {"id": self._uid(), "properties": props, "shape": shape}
            self._push_node(node)
            return node

        # ================ PERSON ================
        n_human = plan.get("person", 0)
        if n_human > 0:
            tpl = self.tmpl.get_template(Category.PROP.value, "person")
            if tpl:
                size = tpl.get("size", {"width": 0.6, "length": 0.6})
                default_status = self.tmpl.get_default_status(Category.PROP.value, "person")
                for _ in range(n_human):
                    dice = self.rng.random()

                    # A) Building (excluding robot_base)
                    if dice < p_building_human and buildings_no_robotbase:
                        parent, shape, _ = place_in_building_any(buildings_no_robotbase, size, self.rng, padding=0.5)
                        if parent:
                            extras = decorate_human(self.tmpl, self.rng)
                            extras["location"] = self._infer_location(shape, parent)
                            node = _new_prop_node("person", f"Person-{self._seq_for_type('person')}", shape, default_status, extras)
                            self._add_edge(node["id"], parent["id"], EdgeType.LOCATED_AT.value)
                            continue

                    # B) Road / intersection
                    if dice < p_building_human + p_road_human:
                        parent, center = snap_on_road_center(streets, intersections, self.rng, seg_prob=0.6)
                        if parent is not None:
                            shape, _ = footprint_from_size_at_center(center, size)
                            extras = decorate_human(self.tmpl, self.rng)
                            extras["location"] = self._infer_location(shape, parent)
                            node = _new_prop_node("person", f"Person-{self._seq_for_type('person')}", shape, default_status, extras)
                            self._add_edge(node["id"], parent["id"], EdgeType.LOCATED_AT.value)
                            continue

                    # C) Open area
                    shape, center = place_in_open_area(getattr(self, "_buildable_union", None), size, self.rng, margin_scale=open_margin)
                    if shape is not None:
                        area_node = find_containing_or_nearest_area(self._nodes_by_category.get(Category.AREA.value, []), center)
                        extras = decorate_human(self.tmpl, self.rng)
                        if area_node:
                            extras["location"] = self._infer_location(shape, area_node)
                        node = _new_prop_node("person", f"Person-{self._seq_for_type('person')}", shape, default_status, extras)
                        if area_node:
                            self._add_edge(node["id"], area_node["id"], EdgeType.LOCATED_IN.value)

        # ================ VEHICLE (road/intersection/open area/parking; no other buildings) ================
        n_car = plan.get("vehicle", 0)
        if n_car > 0:
            tpl = self.tmpl.get_template(Category.PROP.value, "vehicle")
            if tpl:
                size = tpl.get("size", {"width": 2.0, "length": 5.0})
                default_status = self.tmpl.get_default_status(Category.PROP.value, "vehicle")
                for _ in range(n_car):
                    # 1) Road first
                    if self.rng.random() < p_road_car:
                        parent, center = snap_on_road_center(streets, intersections, self.rng, seg_prob=0.6)
                        if parent is not None:
                            shape, _ = footprint_from_size_at_center(center, size)
                            extras = decorate_car(self.tmpl, self.rng)
                            extras["location"] = self._infer_location(shape, parent)
                            node = _new_prop_node("vehicle", f"Vehicle-{self._seq_for_type('vehicle')}", shape, default_status, extras)
                            self._add_edge(node["id"], parent["id"], EdgeType.LOCATED_AT.value)
                            continue
                    # 2) Parking lot (the only allowed building type)
                    if parkings and self.rng.random() < 0.9:
                        parent, shape, _ = place_in_building_any(parkings, size, self.rng, padding=0.5)
                        if parent:
                            extras = decorate_car(self.tmpl, self.rng)
                            extras["location"] = self._infer_location(shape, parent)
                            node = _new_prop_node("vehicle", f"Vehicle-{self._seq_for_type('vehicle')}", shape, default_status, extras)
                            self._add_edge(node["id"], parent["id"], EdgeType.LOCATED_AT.value)
                            continue
                    # 3) Open area fallback
                    shape, center = place_in_open_area(getattr(self, "_buildable_union", None), size, self.rng, margin_scale=open_margin)
                    if shape is not None:
                        area_node = find_containing_or_nearest_area(self._nodes_by_category.get(Category.AREA.value, []), center)
                        extras = decorate_car(self.tmpl, self.rng)
                        if area_node:
                            extras["location"] = self._infer_location(shape, area_node)
                        node = _new_prop_node("vehicle", f"Vehicle-{self._seq_for_type('vehicle')}", shape, default_status, extras)
                        if area_node:
                            self._add_edge(node["id"], area_node["id"], EdgeType.LOCATED_IN.value)

        # ================ CARGO (building or open area) ================
        n_cargo = plan.get("cargo", 0)
        if n_cargo > 0:
            tpl = self.tmpl.get_template(Category.PROP.value, "cargo")
            if tpl:
                size = tpl.get("size", {"width": 1.0, "length": 1.0})
                default_status = self.tmpl.get_default_status(Category.PROP.value, "cargo")
                for _ in range(n_cargo):
                    use_building = (self.rng.random() < 0.5) and bool(buildings_no_robotbase)
                    if use_building:
                        parent, shape, _ = place_in_building_any(buildings_no_robotbase, size, self.rng, padding=0.5)
                        if parent:
                            extras = decorate_cargo(self.tmpl, self.rng)
                            extras["location"] = self._infer_location(shape, parent)
                            node = _new_prop_node("cargo", f"Cargo-{self._seq_for_type('cargo')}", shape, default_status, extras)
                            self._add_edge(node["id"], parent["id"], EdgeType.STORED_AT.value)
                            continue
                    shape, center = place_in_open_area(getattr(self, "_buildable_union", None), size, self.rng, margin_scale=open_margin)
                    if shape is not None:
                        area_node = find_containing_or_nearest_area(self._nodes_by_category.get(Category.AREA.value, []), center)
                        extras = decorate_cargo(self.tmpl, self.rng)
                        if area_node:
                            extras["location"] = self._infer_location(shape, area_node)   
                        node = _new_prop_node("cargo", f"Cargo-{self._seq_for_type('cargo')}", shape, default_status, extras if extras else None)
                        if area_node:
                            self._add_edge(node["id"], area_node["id"], EdgeType.LOCATED_IN.value)

        # ================ BOAT (water body only) ================
        n_boat = plan.get("boat", 0)
        if n_boat > 0:
            tpl = self.tmpl.get_template(Category.PROP.value, "boat")
            if tpl:
                size = tpl.get("size", {"width": 2.0, "length": 6.0})
                default_status = self.tmpl.get_default_status(Category.PROP.value, "boat")
                water = self._union_of("water_body")
                if not water:
                    print("[Props] No water_body; cannot place boats.")
                else:
                    for _ in range(n_boat):
                        center = compute_boat_center(water, self.rng, size, max_trials=200)
                        if center is None:
                            print("[Props] Failed to place a boat after trials.")
                            continue
                        shape, _ = footprint_from_size_at_center(center, size)
                        water_area = find_containing_or_nearest_area(
                            self._nodes_by_category.get(Category.AREA.value, []), center, type_filter="water_body"
                        )
                        extras = decorate_boat(self.tmpl, self.rng)
                        if water_area:
                            extras["location"] = self._infer_location(shape, water_area)
                        node = _new_prop_node("boat", f"Boat-{self._seq_for_type('boat')}", shape, default_status, extras)
                        if water_area:
                            self._add_edge(node["id"], water_area["id"], EdgeType.LOCATED_IN.value)

        # ================ EQUIPMENT_FAILURE (buildings only, excluding robot_base) ================
        n_ef = plan.get("equipment_failure", 0)
        if n_ef > 0:
            tpl = self.tmpl.get_template(Category.PROP.value, "equipment_failure")
            if tpl:
                size = tpl.get("size", {"width": 1.0, "length": 1.0})
                default_status = self.tmpl.get_default_status(Category.PROP.value, "equipment_failure")
                for _ in range(n_ef):
                    parent, shape, _ = place_in_building_any(buildings_no_robotbase, size, self.rng, padding=0.5)
                    if parent:
                        extras = {}
                        extras["location"] = self._infer_location(shape, parent)
                        node = _new_prop_node("equipment_failure",
                                            f"Equipment Failure-{self._seq_for_type('equipment_failure')}",
                                            shape, default_status, extras)
                        self._add_edge(node["id"], parent["id"], EdgeType.LOCATED_AT.value)
        
        # ================ FIRE (buildings only, excluding robot_base) ================
        n_fire = plan.get("fire", 0)
        if n_fire > 0:
            tpl = self.tmpl.get_template(Category.PROP.value, "fire")
            if tpl:
                size = tpl.get("size", {"width": 1.0, "length": 1.0})
                default_status = self.tmpl.get_default_status(Category.PROP.value, "fire")
                for _ in range(n_fire):
                    parent, shape, _ = place_in_building_any(buildings_no_robotbase, size, self.rng, padding=0.5)
                    if parent:
                        extras = {"location": self._infer_location(shape, parent)}
                        node = _new_prop_node(
                            "fire",
                            f"Fire-{self._seq_for_type('fire')}",
                            shape,
                            default_status,
                            extras
                        )
                        self._add_edge(node["id"], parent["id"], EdgeType.LOCATED_AT.value)

        # ================ HAZMAT (hazardous materials, non-robot_base buildings only) ================
        n_haz = plan.get("hazmat", 0)
        if n_haz > 0:
            tpl = self.tmpl.get_template(Category.PROP.value, "hazmat")
            if tpl and buildings_no_robotbase:
                size = tpl.get("size", {"width": 1.0, "length": 1.0})
                default_status = self.tmpl.get_default_status(Category.PROP.value, "hazmat")
                for _ in range(n_haz):
                    parent, shape, _ = place_in_building_any(buildings_no_robotbase, size, self.rng, padding=0.5)
                    if not parent:
                        continue
                    extras = {"location": self._infer_location(shape, parent)}
                    node = _new_prop_node("hazmat", f"Hazmat-{self._seq_for_type('hazmat')}", shape, default_status, extras)
                    self._add_edge(node["id"], parent["id"], EdgeType.STORED_AT.value)


        # ================ ASSEMBLY_COMPONENT (any building; at least one per subtype) ================
        comp_tpl = self.tmpl.get_template(Category.PROP.value, "assembly_component") or {}
        subtype_opts = (((comp_tpl.get("attributes") or {}).get("subtype") or {}).get("options") or [])
        n_extra_comp = max(0, int(plan.get("assembly_component", 0)))
        if buildings_no_robotbase and subtype_opts:
            size = (comp_tpl.get("size") or {"width": 1.0, "length": 1.0})
            default_status = self.tmpl.get_default_status(Category.PROP.value, "assembly_component")
            # First ensure at least one of each subtype
            for st in subtype_opts:
                parent, shape, _ = place_in_building_any(buildings_no_robotbase, size, self.rng, padding=0.5)
                if not parent:
                    continue
                extras = decorate_assembly_component(self.tmpl, self.rng, subtype=st)
                extras["location"] = self._infer_location(shape, parent)
                node = _new_prop_node("assembly_component", f"Assembly Component-{self._seq_for_type('assembly_component')}", shape, default_status, extras)
                self._add_edge(node["id"], parent["id"], EdgeType.STORED_AT.value)
        else:
            if not buildings_no_robotbase:
                print("[Props] No building found; cannot place assembly_component.")


    def add_pois(self, per_mall: int = 2, per_intersection: int = 1):
        malls = [n for n in self._nodes_by_category.get(Category.BUILDING.value, [])
                if n["properties"].get("type") == "mall"]
        inters = [n for n in self._nodes_by_category.get(Category.TRANS_Facility.value, [])
                if n["properties"]["type"] == "intersection"]

        # Entrances for malls
        for m in malls:
            for _ in range(per_mall):
                cx = (m["shape"]["min_corner"][0] + m["shape"]["max_corner"][0]) / 2.0
                cy = m["shape"]["min_corner"][1]
                poi = {
                    "id": self._uid(),
                    "properties": {
                        "category": Category.POI.value,
                        "type": POIType.ENTRANCE.value,
                        "label": f"Entrance-{self._seq_for_type('entrance')}",
                        "status": self.tmpl.get_default_status(Category.POI.value, POIType.ENTRANCE.value)
                    },
                    "shape": {"type": "point", "center": [cx, cy]},
                }
                self._push_node(poi)
                self._add_edge(m["id"], poi["id"], EdgeType.HAS_ENTRANCE.value)

        # Charging stations at intersections
        for x in inters:
            cx, cy = x["shape"]["center"]
            for _ in range(per_intersection):
                poi = {
                    "id": self._uid(),
                    "properties": {
                        "category": Category.POI.value,
                        "type": POIType.CHARGING_STATION.value,
                        "label": f"Charging Station-{self._seq_for_type('charging_station')}",
                        "status": self.tmpl.get_default_status(Category.POI.value, POIType.CHARGING_STATION.value)
                    },
                    "shape": {"type": "point", "center": [cx, cy]},
                }
                self._push_node(poi)
                self._add_edge(poi["id"], x["id"], EdgeType.LOCATED_IN.value)


    # --------- helpers that still rely on builder state ---------
    def _seq_for_type(self, t: str) -> int:
        key = f"seq::{t}"
        if not hasattr(self, "_seq"):
            self._seq = {}
        self._seq[key] = self._seq.get(key, 0) + 1
        return self._seq[key]

    def _geom_inside_world(self, geom) -> bool:
        if not self.world_bounds:
            return True
        return self.world_bounds.contains(geom)

    def _overlaps_existing_buildings(self, geom) -> bool:
        for n2 in self._nodes_by_category.get(Category.BUILDING.value, []):
            s2 = n2.get("shape", {})
            if s2.get("type") == "rectangle":
                g2 = shp_box(s2["min_corner"][0], s2["min_corner"][1], s2["max_corner"][0], s2["max_corner"][1])
            elif s2.get("type") == "circle":
                g2 = ShpPoint(s2["center"][0], s2["center"][1]).buffer(float(s2["radius"]), resolution=32)
            else:
                continue
            if geom.intersects(g2):
                return True
        return False

    def _overlaps_non_buildable_geom(self, geom) -> bool:
        for verts in self._non_buildable_polys:
            if ShpPolygon(verts).intersects(geom):
                return True
        return False

    def _intersects_any_road(self, geom, clearance: float = 0.0) -> bool:
        roads = [n for n in self._nodes_by_category.get(Category.TRANS_Facility.value, [])
                 if n["properties"]["type"] == TransFacilityType.STREET_SEGMENT.value]
        if not roads:
            return False
        for r in roads:
            (x1, y1), (x2, y2) = r["shape"]["points"]
            ls = LineString([(x1, y1), (x2, y2)])
            g = ls.buffer(clearance, cap_style=2) if clearance > 0 else ls
            if geom.intersects(g):
                return True
        return False

    def _union_of(self, area_type: str):
        u = self._area_unions.get(area_type)
        return u if u and (not hasattr(u, "is_empty") or not u.is_empty) else None

    def _intersects_area_type(self, geom, area_type: str) -> bool:
        u = self._union_of(area_type)
        return bool(u and geom.intersects(u))

    def _is_adjacent_to_area(self, geom, area_type: str, max_dist: float) -> bool:
        u = self._union_of(area_type)
        if not u:
            return False
        return geom.distance(u) <= max_dist

    def _recompute_buildable_union(self):
        if not self.world_bounds:
            self._buildable_union = None
            self._area_unions.clear()
            return

        forbidden_polys = [ShpPolygon(vs) for vs in self._non_buildable_polys] if self._non_buildable_polys else []
        forb_union = unary_union(forbidden_polys) if forbidden_polys else None
        self._buildable_union = self.world_bounds.difference(forb_union) if forb_union else self.world_bounds

        self._area_unions = {}
        for area_type, plist in self._areas_by_type.items():
            self._area_unions[area_type] = unary_union(plist) if plist else None

    def _sample_point_in_buildable(self, margin: float, max_tries: int = 200) -> Optional[Tuple[float, float]]:
        if self._buildable_union is None or self._buildable_union.is_empty:
            return None
        poly = self._buildable_union
        if margin > 0:
            shrunk = poly.buffer(-margin)
            if shrunk.is_empty:
                return None
            poly = shrunk
        minx, miny, maxx, maxy = poly.bounds
        for _ in range(max_tries):
            x = self.rng.uniform(minx, maxx)
            y = self.rng.uniform(miny, maxy)
            # tiny rectangle to avoid point-on-boundary issues
            if poly.contains(ShpPolygon([(x-1e-6,y-1e-6),(x+1e-6,y-1e-6),(x+1e-6,y+1e-6),(x-1e-6,y+1e-6)])):
                return (x, y)
        return None

    def _nearest_district_id(self, point_xy: Tuple[float, float]) -> Optional[str]:
        if not self._districts:
            return None
        px, py = point_xy
        best = None
        best_d = 1e30
        for did, node in self._districts.items():
            cx, cy = center_of_shape(node["shape"])
            d2 = (px - cx) ** 2 + (py - cy) ** 2
            if d2 < best_d:
                best_d = d2
                best = did
        return best
    
    def _find_intersection_id_by_point(self, x: float, y: float, eps: float = 1e-2) -> Optional[str]:
        inter_nodes = [
            n for n in self._nodes_by_category.get(Category.TRANS_Facility.value, [])
            if n.get("properties", {}).get("type") == TransFacilityType.INTERSECTION.value
            and n.get("shape", {}).get("type") == "point"
        ]
        for n in inter_nodes:
            cx, cy = n["shape"]["center"]
            if abs(float(cx) - float(x)) <= eps and abs(float(cy) - float(y)) <= eps:
                return n["id"]
        return None

    @property
    def _location_ref_nodes(self) -> List[Dict[str, Any]]:
        """All nodes that can serve as location references (building / trans_facility / area)."""
        refs: List[Dict[str, Any]] = []
        for cat in (Category.BUILDING.value, Category.TRANS_Facility.value, Category.AREA.value):
            refs.extend(self._nodes_by_category.get(cat, []))
        return refs

    def _infer_location(self, child_shape: Dict[str, Any], fallback_node: Dict[str, Any]) -> Dict[str, Any]:
        """Infer location precisely based on child node position, falling back to fallback_node."""
        return location_dict(
            fallback_node,
            child_pos=center_of_shape(child_shape),
            location_candidates=self._location_ref_nodes,
        )
