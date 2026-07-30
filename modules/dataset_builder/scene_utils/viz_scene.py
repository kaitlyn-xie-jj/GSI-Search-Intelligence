# viz_scene.py
from typing import Dict, List, Tuple, Optional
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon, Circle
from matplotlib.collections import PatchCollection
from modules.config.base.enums import Category
from .perlin_utils import _edge_key, _auto_bounds_from_nodes

# -----------------------------
# Color maps
# -----------------------------
def _color_for_area(area_type: str) -> str:
    cmap = {
        "water_body": "#00BCD4",
        "garden": "#4CAF50",
        "neighborhood": "#90CAF9",
        "industrial_park": "#FF9800",
        "greenbelt": "#81C784",
        "square": "#9E9E9E",
        "campus": "#3F51B5",
    }
    return cmap.get(area_type, "#BDBDBD")

def _color_for_building(btype: str) -> str:
    cmap = {
        "mall": "#9C27B0",
        "hospital": "#E53935",
        "power_station": "#FB8C00",
        "library": "#42A5F5",
        "parking": "#757575",
        "robot_base": "#00ACC1",
    }
    return cmap.get(btype, "#607D8B")


# -----------------------------
# Label helpers
# -----------------------------
def _node_label(n: dict) -> str:
    props = n.get("properties", {})
    label = props.get("label")
    if label:
        return str(label)
    t = props.get("type", "node")
    nid = n.get("id", "")
    suffix = nid[-4:] if isinstance(nid, str) else "id"
    return f"{t}-{suffix}"

def _polygon_centroid(verts: List[Tuple[float, float]]) -> Tuple[float, float]:
    area = 0.0
    cx = 0.0
    cy = 0.0
    n = len(verts)
    if n < 3:
        return verts[0] if n else (0.0, 0.0)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area *= 0.5
    if abs(area) < 1e-8:
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    cx /= (6.0 * area)
    cy /= (6.0 * area)
    return (cx, cy)

def _rect_label_pos(rect: dict) -> Tuple[float, float]:
    minx, miny = rect["min_corner"]
    maxx, maxy = rect["max_corner"]
    return ((minx + maxx) / 2.0, maxy + 2.0)

def _circle_label_pos(center: Tuple[float, float], radius: float) -> Tuple[float, float]:
    return (center[0], center[1] + radius + 2.0)

def _point_label_pos(center: Tuple[float, float]) -> Tuple[float, float]:
    return (center[0] + 2.0, center[1] + 2.0)

def _annotate_all_nodes(ax, nodes: List[dict], fontsize: int = 15):
    bbox_style = dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7)
    for n in nodes:
        shp = n.get("shape", {})
        st = shp.get("type")
        label = _node_label(n)
        if st == "rectangle":
            x, y = _rect_label_pos(shp)
            ax.text(x, y, label, fontsize=fontsize, ha="center", va="bottom", bbox=bbox_style, zorder=10)
        elif st == "polygon":
            verts = shp.get("vertices", [])
            if verts:
                cx, cy = _polygon_centroid(verts)
                ax.text(cx, cy, label, fontsize=fontsize, ha="center", va="center", bbox=bbox_style, zorder=10)
        elif st == "circle":
            x, y = _circle_label_pos(shp.get("center", [0.0, 0.0]), shp.get("radius", 1.0))
            ax.text(x, y, label, fontsize=fontsize, ha="center", va="bottom", bbox=bbox_style, zorder=10)
        elif st == "point":
            x, y = _point_label_pos(shp.get("center", [0.0, 0.0]))
            ax.text(x, y, label, fontsize=fontsize, ha="left", va="bottom", bbox=bbox_style, zorder=10)


# -----------------------------
# Legend
# -----------------------------
def _legend_proxy(ax):
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    street_proxy = mlines.Line2D([0], [0], lw=2.0, color="#000000", label="Street")
    bridge_proxy = mlines.Line2D([0], [0], lw=3.0, linestyle="--", color="#1E88E5", label="Bridge")
    poi_proxy = mlines.Line2D([0], [0], lw=0, marker="o", color="#000000", label="POI")
    area_proxy = mpatches.Patch(facecolor="#4CAF50", edgecolor="#424242", label="Area (example)")
    bld_proxy = mpatches.Patch(facecolor="#42A5F5", edgecolor="#212121", label="Building (example)")
    ax.legend(handles=[street_proxy, bridge_proxy, poi_proxy, area_proxy, bld_proxy], loc="upper right")


# -----------------------------
# Main drawing
# -----------------------------
def draw_scene(
    nodes: List[dict],
    edges: List[dict],
    out_png: str = "cybertown_preview.png",
    show: bool = False,
    figsize: Tuple[float, float] = (16, 16),
    draw_labels: bool = True,
    label_fontsize: int = 15,
    perlin_edge_curves: Optional[Dict[Tuple, List[Tuple[float, float]]]] = None,
    perlin_edge_curves_loose: Optional[Dict[Tuple, List[Tuple[float, float]]]] = None,
    street_match_quant: float = 0.5,
):
    fig, ax = plt.subplots(figsize=figsize)

    # Districts
    for n in nodes:
        props = n.get("properties", {})
        if props.get("category") == Category.DISTRICT.value and n.get("shape", {}).get("type") == "rectangle":
            r = n["shape"]
            minx, miny = r["min_corner"]
            maxx, maxy = r["max_corner"]
            rect = Rectangle(
                (minx, miny), maxx - minx, maxy - miny, fill=False, ec="#9E9E9E", lw=1.2, zorder=0
            )
            ax.add_patch(rect)

    # Areas (filled polygons)
    area_patches: List[MplPolygon] = []
    area_colors: List[str] = []
    for n in nodes:
        props = n.get("properties", {})
        if props.get("category") == Category.AREA.value:
            shp = n.get("shape", {})
            if shp.get("type") == "polygon":
                verts = shp.get("vertices", [])
                if len(verts) >= 3:
                    area_patches.append(MplPolygon(verts, closed=True))
                    area_colors.append(_color_for_area(props.get("type", "")))
    if area_patches:
        pc_areas = PatchCollection(
            area_patches, facecolor=area_colors, edgecolor="#424242", linewidth=1.0, alpha=0.65, zorder=1
        )
        ax.add_collection(pc_areas)

    # Streets (curve-match with Perlin edges if provided)
    for n in nodes:
        props = n.get("properties", {})
        if props.get("category") == Category.TRANS_Facility.value and props.get("type") == "street_segment":
            pts = n.get("shape", {}).get("points", [])
            if len(pts) == 2:
                a = (pts[0][0], pts[0][1])
                b = (pts[1][0], pts[1][1])

                poly = None
                if perlin_edge_curves is not None:
                    # exact match
                    k_exact = _edge_key(a, b, quant=1e-6)
                    poly = perlin_edge_curves.get(k_exact)
                if poly is None and perlin_edge_curves_loose is not None:
                    # loose match
                    def kq(p, q):
                        return (round(p[0] / q) * q, round(p[1] / q) * q)
                    aa, bb = kq(a, street_match_quant), kq(b, street_match_quant)
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
                    )
                else:
                    ax.plot(
                        [a[0], b[0]],
                        [a[1], b[1]],
                        linewidth=2.0,
                        alpha=0.8,
                        zorder=2,
                        solid_capstyle="round",
                    )

    # Bridges (different style)
    for n in nodes:
        props = n.get("properties", {})
        if props.get("category") == Category.TRANS_Facility.value and props.get("type") == "bridge":
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
                )

    # Buildings
    building_patches: List = []
    building_colors: List[str] = []
    for n in nodes:
        props = n.get("properties", {})
        if props.get("category") == Category.BUILDING.value:
            shp = n.get("shape", {})
            st = shp.get("type")
            if st == "rectangle":
                minx, miny = shp["min_corner"]
                maxx, maxy = shp["max_corner"]
                building_patches.append(Rectangle((minx, miny), maxx - minx, maxy - miny, zorder=3))
                building_colors.append(_color_for_building(props.get("type", "")))
            elif st == "circle":
                cx, cy = shp["center"]
                radius = shp["radius"]
                building_patches.append(Circle((cx, cy), radius=radius, zorder=3))
                building_colors.append(_color_for_building(props.get("type", "")))
    if building_patches:
        pc_blds = PatchCollection(
            building_patches, facecolor=building_colors, edgecolor="#212121", linewidth=1.0, alpha=0.9, zorder=3
        )
        ax.add_collection(pc_blds)

    # POIs
    poi_x, poi_y = [], []
    for n in nodes:
        props = n.get("properties", {})
        if props.get("category") == Category.POI.value and n.get("shape", {}).get("type") == "point":
            c = n["shape"]["center"]
            poi_x.append(c[0])
            poi_y.append(c[1])
    if poi_x:
        ax.scatter(poi_x, poi_y, s=30, marker="o", zorder=4)

    # Axes & bounds
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.set_title("Cybertown Scene Preview")

    xmin, ymin, xmax, ymax = _auto_bounds_from_nodes(nodes)
    pad = max(20.0, 0.02 * max(xmax - xmin, ymax - ymin))
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)

    _legend_proxy(ax)

    if draw_labels:
        _annotate_all_nodes(ax, nodes, fontsize=label_fontsize)

    ax.set_axis_off()   # Hide axes (including ticks/labels)
    ax.set_title("")    # Remove title
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    if show:
        plt.show()
    plt.close(fig)
