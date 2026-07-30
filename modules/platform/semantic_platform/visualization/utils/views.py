# views.py
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
import matplotlib.patheffects as path_effects
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon, Circle as MplCircle, Rectangle as MplRect
from modules.dataset_builder.scene_utils.perlin_utils import _edge_key

# Edge types that carry geographic meaning in the spatial view
SPATIALLY_RELEVANT_EDGES = {"stationed_at", "parked_at", "located_at", "reachable_from", "carrying"}
label_fontsize = 12


class Views:
    """
    View layer responsible for static rendering (spatial map + relation graph + legend).
    Does not include realtime animation or dynamic patch management (those live in realtime.py).
    """

    def __init__(self,
                 fig: plt.Figure,
                 ax_spatial: plt.Axes,
                 ax_relation: Optional[plt.Axes],
                 color_map: Dict[str, str],
                 rel_color_map: Dict[str, str],
                 bounds: Dict[str, float]):
        """
        Args:
            fig: Matplotlib Figure
            ax_spatial: left-side spatial view Axes
            ax_relation: right-side relation graph Axes
            color_map: type -> color mapping
            rel_color_map: relation type -> color mapping
            bounds: world bounds {'x_min','x_max','y_min','y_max'}
        """
        self.fig = fig
        self.ax_spatial = ax_spatial
        self.ax_relation = ax_relation
        self.color_map = color_map
        self.rel_color_map = rel_color_map
        self.bounds = bounds

        # Internal state for the relation graph
        self.semantic_graph_elements: Dict[str, Dict] = {'nodes': {}, 'edges': {}}
        self.highlighted_edges: Set[Tuple[int, int, str]] = set()
        self.legend_artists: List[Any] = []

    def _abbr_label(self, s: str, max_letters: int = 4) -> str:
        """
        Abbreviate a label to at most max_letters letters; digits and other
        characters do not count toward the letter limit and are kept as-is.
        E.g.: 'Parking-12' -> 'Park-12', 'Vehicle-11' -> 'Vehi-11', 'UAV-1' -> 'UAV-1'
        """
        out = []
        letters = 0
        for ch in s:
            if ch.isalpha():
                if letters < max_letters:
                    out.append(ch)
                letters += 1
            else:
                # Digits, hyphens, spaces, etc. do not count and are kept as-is
                out.append(ch)
        return ''.join(out)

    # =========================
    #      Spatial view (left)
    # =========================
    def draw_spatial(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        moving_entities: Set[int],
        carried_prop_ids: Optional[Set[int]] = None,
        perlin_edge_curves: Optional[Dict[Tuple, List[Tuple[float, float]]]] = None,
        perlin_edge_curves_loose: Optional[Dict[Tuple, List[Tuple[float, float]]]] = None,
        street_match_quant: float = 0.5,
        draw_labels: bool = True,
        prop_label_ids: Optional[Set[int]] = None, 
    ) -> None:
        """
        Draw the static base of the spatial map:
        - District borders
        - Area polygons (filled)
        - Streets (preferentially matched to Perlin curves)
        - Bridges (dashed)
        - Buildings (filled)
        - POIs (scatter)
        - Keep the existing static base for props/robots; prop labels follow
          prop_label_ids / the random-2 strategy
        """
        if carried_prop_ids is None:
            carried_prop_ids = {e['target'] for e in edges if e.get('type') == 'carrying'}

        ax = self.ax_spatial

        # Axes and background
        ax.set_xlim(self.bounds['x_min'], self.bounds['x_max'])
        ax.set_ylim(self.bounds['y_min'], self.bounds['y_max'])
        ax.set_aspect('auto', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#F5F5F5')
        ax.grid(True, linestyle='--', alpha=0.2, color='#CCCCCC', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333'); spine.set_linewidth(2)

        # ===== Districts (thin border, no fill) =====
        for n in nodes:
            props = n.get("properties", {})
            shp = n.get("shape", {})
            if props.get("category") == "district" and shp.get("type") == "rectangle":
                minx, miny = shp["min_corner"]
                maxx, maxy = shp["max_corner"]
                rect = Rectangle(
                    (minx, miny),
                    maxx - minx,
                    maxy - miny,
                    fill=False,
                    ec="#9E9E9E",
                    lw=1.2,
                    zorder=0,
                )
                ax.add_patch(rect)

        # ===== Areas (polygons, using color_map[area_type]) =====
        area_patches: List[MplPolygon] = []
        area_colors: List[str] = []
        for n in nodes:
            props = n.get("properties", {})
            if props.get("category") == "area":
                shp = n.get("shape", {})
                if shp.get("type") == "polygon":
                    verts = shp.get("vertices", [])
                    if len(verts) >= 3:
                        area_patches.append(MplPolygon(verts, closed=True))
                        a_type = props.get("type", "area")
                        area_colors.append(self.color_map.get(a_type, "#D0D0D0"))
        if area_patches:
            pc_areas = PatchCollection(
                area_patches,
                facecolor=area_colors,
                edgecolor="#424242",
                linewidth=1.0,
                alpha=0.65,
                zorder=1,
            )
            ax.add_collection(pc_areas)

        # ===== Streets (line segments; preferentially matched to Perlin curves) =====
        street_color = self.color_map.get("street_segment", "#616161")
        for n in nodes:
            props = n.get("properties", {})
            if props.get("category") == "trans_facility" and props.get("type") == "street_segment":
                pts = n.get("shape", {}).get("points", [])
                if len(pts) == 2:
                    a = (pts[0][0], pts[0][1])
                    b = (pts[1][0], pts[1][1])

                    poly = None
                    if perlin_edge_curves is not None:
                        # Exact match
                        k_exact = _edge_key(a, b, quant=1e-6)
                        poly = perlin_edge_curves.get(k_exact)
                    if poly is None and perlin_edge_curves_loose is not None:
                        # Loose match
                        aa = (round(a[0] / street_match_quant) * street_match_quant,
                            round(a[1] / street_match_quant) * street_match_quant)
                        bb = (round(b[0] / street_match_quant) * street_match_quant,
                            round(b[1] / street_match_quant) * street_match_quant)
                        k_loose = (aa, bb) if aa <= bb else (bb, aa)
                        poly = perlin_edge_curves_loose.get(k_loose)

                    if poly:
                        ax.plot(
                            [p[0] for p in poly],
                            [p[1] for p in poly],
                            linewidth=2.0,
                            alpha=0.9,
                            zorder=2,
                            solid_capstyle="round",
                            color=street_color,
                        )
                    else:
                        ax.plot(
                            [a[0], b[0]],
                            [a[1], b[1]],
                            linewidth=2.0,
                            alpha=0.85,
                            zorder=2,
                            solid_capstyle="round",
                            color=street_color,
                        )

        # ===== Bridges (thick dashed lines) =====
        bridge_color = self.color_map.get("bridge", "#455A64")
        for n in nodes:
            props = n.get("properties", {})
            if props.get("category") == "trans_facility" and props.get("type") == "bridge":
                pts = n.get("shape", {}).get("points", [])
                if len(pts) == 2:
                    a = (pts[0][0], pts[0][1])
                    b = (pts[1][0], pts[1][1])
                    ax.plot(
                        [a[0], b[0]],
                        [a[1], b[1]],
                        linewidth=3.0,
                        alpha=0.95,
                        zorder=2.5,
                        linestyle="--",
                        color=bridge_color,
                        solid_capstyle="round",
                    )

        # ===== Buildings (filled, reusing the stylish-effect helper) =====
        for n in nodes:
            props = n.get("properties", {})
            if props.get("category") == "building":
                shp = n.get("shape", {})
                b_type = props.get("type", "building")
                base_color = self.color_map.get(b_type, "#C0C0C0")
                self._draw_stylish_building(ax, shp, base_color, props)

        # ===== POIs (scatter) =====
        poi_x, poi_y, poi_c = [], [], []
        for n in nodes:
            props = n.get("properties", {})
            if props.get("category") == "poi" and n.get("shape", {}).get("type") == "point":
                c = n["shape"]["center"]
                poi_x.append(c[0]); poi_y.append(c[1])
                poi_c.append(self.color_map.get(props.get("type", "poi"), "#757575"))
        if poi_x:
            ax.scatter(poi_x, poi_y, s=30, marker="o", zorder=4, c=poi_c, edgecolors="black", linewidths=0.6)

        # ===== Props / Robots static base (prop labels are not decided here; only draw shapes first) =====
        layers = [
            lambda n: n["properties"]["category"] == "building",            # buildings already drawn, not repeated here
            lambda n: n["properties"]["category"] in ["prop", "anomaly"],
            lambda n: n["properties"]["category"] == "robot",
        ]

        for layer_fn in layers:
            for n in filter(layer_fn, nodes):
                if n["properties"]["category"] == "building":
                    continue  # building already drawn

                shape = n.get("shape")
                if not shape:
                    continue

                props = n["properties"]

                # Visibility (by status)
                is_visible = True
                if props.get("type") in ['equipment_failure', 'security_breach']:
                    status = props.get("status")
                    if isinstance(status, dict):
                        if status.get("default") in ['undiscovered', 'unresolved']:
                            is_visible = False
                    elif isinstance(status, str):
                        if status in ['undiscovered', 'unresolved']:
                            is_visible = False

                category = props.get("category", "")
                t = props.get("type") or category
                color = self.color_map.get(t, "#C0C0C0")

                if shape['type'] == 'rectangle':
                    bl = shape["min_corner"]
                    w = shape["max_corner"][0] - bl[0]
                    h = shape["max_corner"][1] - bl[1]
                    shadow = Rectangle((bl[0] + 0.5, bl[1] - 0.5), w, h,
                                    facecolor='gray', alpha=0.15, zorder=4, visible=is_visible)
                    ax.add_patch(shadow)
                    rect = Rectangle(bl, w, h,
                                    edgecolor="black", facecolor=color,
                                    alpha=0.9, linewidth=1.5, label=t,
                                    zorder=9 if category == "prop" else 5,
                                    visible=is_visible)
                    ax.add_patch(rect)

                elif shape['type'] == 'circle':
                    center = shape["center"]
                    radius = shape["radius"]
                    shadow = Circle((center[0] + 0.5, center[1] - 0.5), radius,
                                    facecolor='gray', alpha=0.15, zorder=4, visible=is_visible)
                    ax.add_patch(shadow)
                    circle = Circle(center, radius,
                                    edgecolor="black", facecolor=color,
                                    alpha=0.9, linewidth=1.5, label=t,
                                    zorder=9 if category == "prop" else 5,
                                    visible=is_visible)
                    ax.add_patch(circle)

        # ===== Prop label strategy =====
        # 1) First collect candidates (not carried, not moving, visible, has label)
        label_candidates_by_type: Dict[str, List[Dict[str, Any]]] = {}
        for n in nodes:
            props = n.get("properties", {})
            if props.get("category") != "prop":
                continue

            shape = n.get("shape", {})
            if not shape:
                continue

            # Visibility
            is_visible = True
            if props.get("type") in ['equipment_failure', 'security_breach']:
                status = props.get("status")
                if isinstance(status, dict):
                    if status.get("default") in ['undiscovered', 'unresolved']:
                        is_visible = False
                elif isinstance(status, str):
                    if status in ['undiscovered', 'unresolved']:
                        is_visible = False
            if not is_visible:
                continue

            pid = n['id']
            if pid in carried_prop_ids or pid in moving_entities:
                continue

            label = props.get("label", "")
            if not label:
                continue

            # Compute label coordinates
            if shape.get('type') == 'rectangle':
                bl = shape["min_corner"]
                x = bl[0] + (shape["max_corner"][0] - bl[0]) / 2
                y = bl[1] - 3
            elif shape.get('type') == 'circle':
                center = shape["center"]; radius = shape["radius"]
                x = center[0]; y = center[1] - radius - 3
            else:
                continue

            t = props.get("type", "")
            label_candidates_by_type.setdefault(t, []).append({
                "id": pid,
                "x": x, "y": y,
                "text": self._abbr_label(label),
                "facecolor": self.color_map.get(t, "#C0C0C0"),
            })

        # 2) Select which prop labels to draw:
        #   - If prop_label_ids is passed (non-empty), draw only those ids; otherwise, randomly pick 2 per type.
        selected_labels: List[Dict[str, Any]] = []
        if prop_label_ids:
            idset = set(prop_label_ids)
            for lst in label_candidates_by_type.values():
                for it in lst:
                    if it["id"] in idset:
                        selected_labels.append(it)
        else:
            for t, lst in label_candidates_by_type.items():
                if len(lst) <= 2:
                    selected_labels.extend(lst)
                else:
                    selected_labels.extend(lst[:2])

        # 3) Batch-draw prop labels
        # for it in selected_labels:
        #     text = ax.text(
        #         it["x"], it["y"], it["text"],
        #         ha="center", va="top",
        #         fontsize=label_fontsize, weight="bold", color="#333333",
        #         bbox=dict(facecolor=it["facecolor"], alpha=0.8,
        #                 edgecolor='black', linewidth=1.5,
        #                 pad=2, boxstyle="round,pad=0.3"),
        #         zorder=20
        #     )
        #     text.set_path_effects([
        #         path_effects.Stroke(linewidth=2, foreground='white'),
        #         path_effects.Normal()
        #     ])

        # ===== Optional: full-map annotations (buildings/areas, etc.) =====
        if draw_labels:
            for n in nodes:
                shp = n.get("shape", {})
                label = n.get("properties", {}).get("label")
                type = n.get("properties", {}).get("type")
                if not label:
                    continue
                label = self._abbr_label(label)
                if shp.get("type") == "rectangle" and (type == "person" or type == "cargo" or type == "vehicle"):
                    x = (shp["min_corner"][0] + shp["max_corner"][0]) / 2.0
                    y = shp["max_corner"][1] + 4.0
                # elif shp.get("type") == "circle":
                #     x, y = shp["center"][0], shp["center"][1] + shp["radius"] + 4.0
                # elif shp.get("type") == "polygon":
                #     xs = [vx for vx, _ in shp.get("vertices", [])]
                #     ys = [vy for _, vy in shp.get("vertices", [])]
                #     if not xs or not ys:
                #         continue
                #     x, y = (min(xs) + max(xs)) / 2.0, max(ys) + 4.0
                # elif shp.get("type") == "point":
                #     x, y = shp["center"][0], shp["center"][1] + 4.0
                else:
                    continue

                ax.text(x, y, label,
                        ha='center', va='bottom',
                        fontsize=0.9*label_fontsize,
                        color='#1A1A1A',
                        bbox=dict(boxstyle='round,pad=0.2',
                                facecolor='white', alpha=0.8, edgecolor='#BDBDBD'),
                        zorder=50)
                
    def _draw_stylish_building(self, ax: plt.Axes, shape: Dict[str, Any], base_color: str, props: Dict[str, Any]) -> None:
        """Stylish building rendering: glow + body + label"""
        if shape['type'] == 'rectangle':
            bl = shape["min_corner"]
            w = shape["max_corner"][0] - bl[0]
            h = shape["max_corner"][1] - bl[1]

            # Glow (reduced to two layers to avoid excessive clutter)
            for i in range(2):
                glow = Rectangle((bl[0] - i * 1.5, bl[1] - i * 1.5), w + i * 3, h + i * 3,
                                 facecolor='none', edgecolor=base_color,
                                 alpha=0.1 * (2 - i), linewidth=1.5, zorder=3 - i)
                ax.add_patch(glow)

            main_rect = Rectangle(bl, w, h,
                                  edgecolor='#2C3E50', facecolor=base_color,
                                  alpha=0.7, linewidth=2, zorder=6)
            ax.add_patch(main_rect)

            label = props.get("label", "")
            if label:
                label_x = bl[0] + w / 2
                label_y = bl[1] - 5
                text = ax.text(label_x, label_y, label,
                               ha="center", va="top",
                               fontsize=label_fontsize, weight="bold",
                               color="#2C3E50", zorder=10)
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),
                    path_effects.Normal()
                ])

        elif shape['type'] == 'circle':
            center = shape["center"]
            radius = shape["radius"]

            for i in range(2):
                glow = Circle(center, radius + i * 1.5,
                              facecolor='none', edgecolor=base_color,
                              alpha=0.1 * (2 - i), linewidth=1.5, zorder=3 - i)
                ax.add_patch(glow)

            main_circle = Circle(center, radius,
                                 edgecolor='#2C3E50', facecolor=base_color,
                                 alpha=0.7, linewidth=2, zorder=6)
            ax.add_patch(main_circle)

            label = props.get("label", "")
            if label:
                label_y = center[1] - radius - 5
                text = ax.text(center[0], label_y, label,
                               ha="center", va="top",
                               fontsize=label_fontsize, weight="bold",
                               color="#2C3E50", zorder=10)
                text.set_path_effects([
                    path_effects.Stroke(linewidth=3, foreground='white'),
                    path_effects.Normal()
                ])

    def draw_spatial_edges(self, nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]], style: Dict[str, Any]) -> None:
        """
        Optional: draw spatially meaningful edges (stationed_at/parked_at/...) on the spatial map.
        Can be called after draw_spatial.
        """
        node_map = {n['id']: n for n in nodes}
        plottable = [e for e in edges if e.get('type') in SPATIALLY_RELEVANT_EDGES]

        for edge in plottable:
            s = node_map.get(edge['source'])
            t = node_map.get(edge['target'])
            if not s or not t or not s.get('shape') or not t.get('shape'):
                continue

            s_shape, t_shape = s['shape'], t['shape']
            x_s = (s_shape['min_corner'][0] + s_shape['max_corner'][0]) / 2
            y_s = (s_shape['min_corner'][1] + s_shape['max_corner'][1]) / 2
            x_t = (t_shape['min_corner'][0] + t_shape['max_corner'][0]) / 2
            y_t = (t_shape['min_corner'][1] + t_shape['max_corner'][1]) / 2

            self.ax_spatial.plot([x_s, x_t], [y_s, y_t], **style)
            self.ax_spatial.text((x_s + x_t) / 2, (y_s + y_t) / 2 + 0.03 * self.bounds['y_max'],
                                 edge.get('type', ''),
                                 color=style.get('color', 'black'),
                                 fontsize=label_fontsize, weight='bold',
                                 ha='center', va='center',
                                 bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # =========================
    #      Relation graph (right)
    # =========================
    def draw_graph(self,
               nodes: List[Dict[str, Any]],
               edges: List[Dict[str, Any]],
               added_edges: List[Dict[str, Any]],
               deleted_edges: List[Dict[str, Any]]) -> None:
        """
        Draw the semantic relation graph (only robot / building / prop are shown).
        Dynamic highlighting is controlled by set_edge_highlight / highlighted_edges.
        """
        if not self.ax_relation:
            return
        
        ax = self.ax_relation
        fig = self.fig

        ax.set_facecolor('white')
        ax.axis("off")

        G = nx.DiGraph()
        labels: Dict[int, str] = {}
        layer_map: Dict[int, List[int]] = {}

        nodes_by_id: Dict[int, Dict] = {n["id"]: n for n in nodes}

        # -- Keep only three categories: robot / building / prop --
        ALLOWED_CATS = {"robot", "building", "prop"}

        # Limit each prop type to at most the first 2
        prop_keep_ids: Set[int] = set()
        prop_buckets: Dict[str, List[Dict]] = {}
        for n in nodes:
            if n["properties"].get("category") == "prop":
                t = n["properties"].get("type", "")
                prop_buckets.setdefault(t, []).append(n)
        for t, lst in prop_buckets.items():
            lst.sort(key=lambda x: (x["properties"].get("label", ""), x["id"]))  # stable sort
            for keep in lst[:2]:
                prop_keep_ids.add(keep["id"])

        # Build nodes: add only the allowed three categories, and apply the "keep first 2" strategy to props
        for n in nodes:
            cat = n["properties"].get("category")
            if cat not in ALLOWED_CATS:
                continue                      # skip area / trans_facility / poi / district, etc.
            if cat == "prop" and n["id"] not in prop_keep_ids:
                continue                      # skip extra props

            nid = n["id"]
            layer = 0 if cat == "robot" else 1 if cat == "building" else 2
            G.add_node(nid, layer=layer)
            layer_map.setdefault(layer, []).append(nid)
            labels[nid] = n["properties"].get("label", str(nid))

        # Filter and add edges: only add if both endpoints are in G
        edge_labels = {}
        for e in edges:
            u, v = e["source"], e["target"]
            if (u not in G.nodes) or (v not in G.nodes):
                continue
            rel = e.get("type", "")
            G.add_edge(u, v)
            if rel:
                edge_labels[(u, v)] = rel

        # Three-layer horizontal layout + even vertical spacing
        xs = {0: 0.12, 1: 0.5, 2: 0.88}
        pos: Dict[int, tuple] = {}
        for layer, ids in layer_map.items():
            sorted_ids = sorted(ids, key=lambda i: labels[i])
            ys = np.linspace(0.94, 0.04, len(sorted_ids) if len(sorted_ids) > 1 else 1)
            if len(sorted_ids) == 1:
                ys = [0.58]
            for nid, y in zip(sorted_ids, ys):
                pos[nid] = (xs[layer], y)

        # Clear old cache
        self.semantic_graph_elements = {'nodes': {}, 'edges': {}}

        # Draw nodes
        for n in G.nodes():
            x, y = pos[n]
            node = nodes_by_id[n]
            category = node['properties'].get('category')
            node_type = node['properties'].get('type', category)
            color = self.color_map.get(node_type, "#C0C0C0")

            if category == 'robot':
                circle_radius = 0.04
                circle = Circle((x, y), circle_radius,
                                facecolor=color, edgecolor='black',
                                linewidth=2.5, alpha=0.9,
                                transform=ax.transAxes, zorder=12, clip_on=False)
                ax.add_patch(circle)
            elif category == 'building':
                rect_width = 0.08; rect_height = 0.05
                rect = Rectangle((x - rect_width / 2, y - rect_height / 2),
                                rect_width, rect_height,
                                facecolor=color, edgecolor='black',
                                linewidth=2, alpha=0.8,
                                transform=ax.transAxes, zorder=10)
                ax.add_patch(rect)
            else:  # prop
                triangle_size = 0.03
                tri = plt.Polygon([(x, y + triangle_size),
                                (x - triangle_size * 0.866, y - triangle_size * 0.5),
                                (x + triangle_size * 0.866, y - triangle_size * 0.5)],
                                facecolor=color, edgecolor='black',
                                linewidth=2, alpha=0.8,
                                transform=ax.transAxes, zorder=10)
                ax.add_patch(tri)

        # Node labels
        for n, (x, y) in pos.items():
            ax.text(x, y, labels[n],
                    ha='center', va='center',
                    fontsize=label_fontsize, weight='bold',
                    color='black', transform=ax.transAxes, zorder=15)

        # Set of "newly added edges" to highlight
        added_edge_ids = {(e["source"], e["target"], e.get("type", "")) for e in added_edges
                        if (e["source"] in G.nodes and e["target"] in G.nodes)}

        # Draw edges (curved + highlight)
        for u, v in G.edges():
            edge_id = (u, v, edge_labels.get((u, v), ''))
            x0, y0 = pos[u]; x1, y1 = pos[v]
            rel = edge_labels.get((u, v), "")
            base_color = self.rel_color_map.get(rel, "#888888")
            style = {'lw': 2.0, 'color': base_color}

            is_highlighted = edge_id in added_edge_ids or edge_id in self.highlighted_edges
            if is_highlighted:
                style['lw'] = 4.0
                style['color'] = '#00FF00'
                style['zorder'] = 20

            dy, dx = y1 - y0, x1 - x0
            if abs(dx) > 0.3:
                rad = 0.3 * np.sign(dy) if abs(dy) > 0.01 else 0.2
            else:
                rad = 0.15 * np.sign(dy) if abs(dy) > 0.01 else 0.1

            arrow = FancyArrowPatch((x0, y0), (x1, y1),
                                    connectionstyle=f"arc3,rad={rad}",
                                    arrowstyle="-|>",
                                    mutation_scale=12,
                                    transform=ax.transAxes,
                                    **style)
            ax.add_patch(arrow)

            # Label position
            t = 0.5
            if rad != 0:
                ctrl_x = (x0 + x1) / 2 + rad * (y1 - y0) * 0.3
                ctrl_y = (y0 + y1) / 2 - rad * (x1 - x0) * 0.3
                mid_x = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * ctrl_x + t ** 2 * x1
                mid_y = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * ctrl_y + t ** 2 * y1
            else:
                mid_x = (x0 + x1) / 2; mid_y = (y0 + y1) / 2

            label_text = ax.text(mid_x, mid_y, rel.replace('_', ' ').title(),
                                ha='center', va='center',
                                fontsize=1.2*label_fontsize if is_highlighted else label_fontsize,
                                weight='bold' if is_highlighted else 'normal',
                                transform=ax.transAxes,
                                bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor='#00FF00' if is_highlighted else 'white',
                                        alpha=0.8,
                                        edgecolor='black' if is_highlighted else 'gray'),
                                visible=is_highlighted,
                                zorder=21)

            self.semantic_graph_elements['edges'][edge_id] = {
                'arrow': arrow,
                'label': label_text,
                'original_color': base_color,
                'type': rel
            }

        # Bottom legend
        self._draw_edge_legend_in_graph(edges)

    def set_edge_highlight(self, edge_id: Tuple[int, int, str], on: bool = True) -> None:
        """Called by the realtime layer: set the highlight state of an edge in the relation graph"""
        if not self.ax_relation:
            return
        
        if on:
            self.highlighted_edges.add(edge_id)
        else:
            self.highlighted_edges.discard(edge_id)

        # Sync cached elements (if already drawn this frame)
        info = self.semantic_graph_elements.get('edges', {}).get(edge_id)
        if info:
            arrow = info['arrow']
            label = info.get('label')
            if on:
                arrow.set_linewidth(4.0)
                arrow.set_color('#00FF00')
                arrow.set_zorder(20)
                if label:
                    label.set_visible(True)
                    label.set_fontsize(10)
                    label.set_weight('bold')
                    label.set_bbox(dict(
                        boxstyle='round,pad=0.3',
                        facecolor='#00FF00',
                        alpha=0.8,
                        edgecolor='black'
                    ))
                    label.set_zorder(21)
            else:
                arrow.set_linewidth(2.0)
                arrow.set_color(info['original_color'])
                arrow.set_zorder(10)
                if label:
                    label.set_visible(False)

    # =========================
    #        Legend
    # =========================
    def _draw_edge_legend_in_graph(self, edges: List[Dict[str, Any]]) -> None:
        """Draw the edge-type legend below the relation graph (Figure coordinate system)"""
        edge_types = sorted({e.get("type") for e in edges if e.get("type")})
        if not edge_types:
            # Clear old legend
            for artist in self.legend_artists:
                artist.remove()
            self.legend_artists.clear()
            return

        # Clear previous legend elements
        for artist in self.legend_artists:
            artist.remove()
        self.legend_artists.clear()

        num_types = len(edge_types)
        num_cols = min(3, num_types)
        num_rows = (num_types + num_cols - 1) // num_cols

        legend_x = 0.64
        legend_y = 0.02
        col_width = 0.11
        row_height = 0.04
        legend_width = col_width * num_cols + 0.01
        legend_height = row_height * num_rows + 0.06

        if legend_x + legend_width > 1:
            legend_width = 1 - legend_x
            col_width = (legend_width - 0.01) / num_cols

        # Background
        bg = Rectangle((legend_x - 0.01, legend_y),
                       legend_width + 0.02, legend_height,
                       facecolor='white', edgecolor='gray',
                       linewidth=1.5, alpha=0.95,
                       transform=self.fig.transFigure, zorder=100)
        self.fig.add_artist(bg)
        self.legend_artists.append(bg)

        # Title
        title = self.fig.text(legend_x + legend_width / 2, legend_y + legend_height - 0.02,
                              'Edge Types',
                              transform=self.fig.transFigure,
                              fontsize=label_fontsize, weight='bold',
                              ha='center', va='top', zorder=101)
        self.legend_artists.append(title)

        # Each type
        for i, edge_type in enumerate(edge_types):
            col = i % num_cols
            row = i // num_cols

            x_pos = legend_x + col * col_width
            y_pos = legend_y + legend_height - 0.05 - (row + 0.5) * row_height

            color = self.rel_color_map.get(edge_type, "#888888")

            # Line segment
            line_start = x_pos
            line_end = x_pos + 0.035
            line = plt.Line2D([line_start, line_end], [y_pos, y_pos],
                              color=color, linewidth=7.5,
                              transform=self.fig.transFigure, zorder=101)
            self.fig.add_artist(line)
            self.legend_artists.append(line)

            # Arrow
            arrow = FancyArrowPatch((line_end - 0.008, y_pos),
                                    (line_end, y_pos),
                                    arrowstyle="-|>", mutation_scale=30,
                                    color=color, linewidth=3.5,
                                    transform=self.fig.transFigure, zorder=101)
            self.fig.add_artist(arrow)
            self.legend_artists.append(arrow)

            # Text
            text = self.fig.text(line_end + 0.005, y_pos,
                                 edge_type.replace("_", " ").title(),
                                 transform=self.fig.transFigure,
                                 fontsize=label_fontsize, va='center', zorder=101)
            self.legend_artists.append(text)

    def draw_goal_area(
        self,
        goal: Dict[str, Any],
        *,
        edge_color: str = "#1976D2",
        fill_alpha: float = 0.08,
        line_width: float = 2.4,
        show_label: bool = True,
        label: Optional[str] = None,
    ) -> None:
        """
        Draw the search-area boundary of the task on the spatial view (Point Radius / Boundary Selection).
        - Only responsible for drawing; does not modify other layers.
        """
        ax = self.ax_spatial

        def _find_area_in_success_condition(sc: Any) -> Optional[Dict[str, Any]]:
            if not isinstance(sc, dict):
                return None
            args = sc.get("args")
            if isinstance(args, dict):
                area = args.get("area")
                if isinstance(area, dict):
                    return area
            elif isinstance(args, list):
                for node in args:
                    if isinstance(node, dict):
                        # Child nodes may also be {op: ..., args: {...}} structures
                        hit = _find_area_in_success_condition(node)
                        if hit is not None:
                            return hit
            return None

        def _get_area_dict(g: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            # 1) context.search_area (context may be a dict or a list)
            ctx = (g or {}).get("context")
            if isinstance(ctx, dict):
                area = ctx.get("search_area")
                if isinstance(area, dict):
                    return area
            elif isinstance(ctx, list):
                for block in ctx:
                    if isinstance(block, dict):
                        area = block.get("search_area")
                        if isinstance(area, dict):
                            return area

            # 2) any args.area inside success_condition
            sc = (g or {}).get("success_condition")
            area = _find_area_in_success_condition(sc)
            if isinstance(area, dict):
                return area

            return None

        area = _get_area_dict(goal)
        if not isinstance(area, dict):
            return

        atype = (area.get("area_type") or "").strip().lower()

        if atype == "point radius":
            c = area.get("center_point", {}) or {}
            try:
                x, y = float(c.get("x", 0.0)), float(c.get("y", 0.0))
                r = float(area.get("radius_m", 0.0))
            except (TypeError, ValueError):
                return
            if r <= 0:
                return

            patch = MplCircle((x, y), r,
                            facecolor=edge_color, edgecolor=edge_color,
                            linewidth=line_width, alpha=fill_alpha, zorder=70)
            ax.add_patch(patch)
            # Draw a clearer outline ring (on top of the fill layer)
            outline = MplCircle((x, y), r,
                                facecolor="none", edgecolor=edge_color, linestyle='--',
                                linewidth=line_width, alpha=0.95, zorder=71)
            ax.add_patch(outline)

            if show_label:
                text = label or f"SEARCH AREA (r={int(r)}m)"
                ax.text(x, y, text,
                        ha="center", va="center",
                        fontsize=label_fontsize, weight="bold",
                        color=edge_color, zorder=72,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=edge_color))

        elif atype == "boundary selection":
            pts = area.get("boundary_points") or []
            verts = []
            for p in pts:
                if isinstance(p, dict) and "x" in p and "y" in p:
                    try:
                        verts.append((float(p["x"]), float(p["y"])))
                    except (TypeError, ValueError):
                        continue
            if len(verts) < 3:
                return

            poly = MplPolygon(verts, closed=True,
                            facecolor=edge_color, edgecolor=edge_color,
                            linewidth=line_width, alpha=fill_alpha, zorder=70, joinstyle="round")
            ax.add_patch(poly)
            # Reinforce the outline
            ax.plot([v[0] for v in verts] + [verts[0][0]],
                    [v[1] for v in verts] + [verts[0][1]],
                    linestyle="--", linewidth=line_width, color=edge_color, alpha=0.95, zorder=71)

            if show_label:
                cx = sum(v[0] for v in verts) / len(verts)
                cy = sum(v[1] for v in verts) / len(verts)
                text = label or "SEARCH AREA"
                ax.text(cx, cy, text,
                        ha="center", va="center",
                        fontsize=label_fontsize, weight="bold",
                        color=edge_color, zorder=72,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85, edgecolor=edge_color))
        else:
            # Other types (e.g., Named Area) are not drawn for now
            return