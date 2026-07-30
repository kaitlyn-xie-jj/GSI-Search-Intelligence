# geometry_utils.py
from typing import List, Tuple, Dict, Optional
import math
from random import Random

# Optional high-fidelity libs
try:
    import shapely.geometry as shg
    import shapely.ops as shop
    HAS_SHAPELY = True
except Exception:
    HAS_SHAPELY = False

Point = Tuple[float, float]
Polygon = List[Point]

def rect_from_center(center: Point, size: Dict[str, float]) -> Dict:
    w, h = size.get("width", 10.0), size.get("length", 10.0)
    cx, cy = center
    half_w, half_h = w / 2.0, h / 2.0
    return {
        "type": "rectangle",
        "min_corner": [cx - half_w, cy - half_h],
        "max_corner": [cx + half_w, cy + half_h]
    }

def bbox_overlap(a: Dict, b: Dict) -> bool:
    if a["type"] != "rectangle" or b["type"] != "rectangle":
        return False  # simple fallback; polygon checks require shapely
    ax1, ay1 = a["min_corner"]; ax2, ay2 = a["max_corner"]
    bx1, by1 = b["min_corner"]; bx2, by2 = b["max_corner"]
    return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

def poly_area(poly: Polygon) -> float:
    # Shoelace formula
    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i+1) % n]
        area += (x1 * y2 - x2 * y1)
    return abs(area) / 2.0

def segment_normal(p1: Point, p2: Point) -> Point:
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length, dx / length
    return (nx, ny)

def point_along_segment(p1: Point, p2: Point, t: float) -> Point:
    return (p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t)

def offset_point(p: Point, n: Point, d: float) -> Point:
    return (p[0] + n[0]*d, p[1] + n[1]*d)

def polygon_from_bbox(bounds: Dict[str, float]) -> Polygon:
    return [
        (bounds['x_min'], bounds['y_min']),
        (bounds['x_max'], bounds['y_min']),
        (bounds['x_max'], bounds['y_max']),
        (bounds['x_min'], bounds['y_max'])
    ]

def clip_poly_to_bbox(poly: Polygon, bounds: Dict[str, float]) -> Polygon:
    """If shapely exists, clip; otherwise return input."""
    if HAS_SHAPELY:
        poly_obj = shg.Polygon(poly)
        bbox = shg.box(bounds['x_min'], bounds['y_min'], bounds['x_max'], bounds['y_max'])
        clipped = poly_obj.intersection(bbox)
        if clipped.is_empty:
            return []
        if hasattr(clipped, "exterior"):
            return list(clipped.exterior.coords)[:-1]
    return poly

def random_point_in_bounds(bounds: Dict[str, float], rng: Optional[Random] = None) -> Point:
    rng = rng or Random(0)
    return (rng.uniform(bounds['x_min'], bounds['x_max']),
            rng.uniform(bounds['y_min'], bounds['y_max']))


def point_in_polygon(p: Point, poly: List[Point]) -> bool:
    """Ray casting. True if point inside polygon (strict), False otherwise."""
    x, y = p
    inside = False
    n = len(poly)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        intersect = ((yi > y) != (yj > y)) and \
                    (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
        if intersect:
            inside = not inside
        j = i
    return inside

def rect_corners(rect: Dict) -> List[Point]:
    x1, y1 = rect["min_corner"]
    x2, y2 = rect["max_corner"]
    return [(x1,y1),(x2,y1),(x2,y2),(x1,y2)]

def point_in_rect(p: Point, rect: Dict) -> bool:
    x1, y1 = rect["min_corner"]
    x2, y2 = rect["max_corner"]
    return (x1 <= p[0] <= x2) and (y1 <= p[1] <= y2)

def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])

def _on_segment(a: Point, b: Point, c: Point) -> bool:
    return min(a[0], b[0]) - 1e-9 <= c[0] <= max(a[0], b[0]) + 1e-9 and \
           min(a[1], b[1]) - 1e-9 <= c[1] <= max(a[1], b[1]) + 1e-9

def segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    """Proper + colinear overlap."""
    o1 = _orient(p1, p2, q1)
    o2 = _orient(p1, p2, q2)
    o3 = _orient(q1, q2, p1)
    o4 = _orient(q1, q2, p2)

    if (o1*o2 < 0) and (o3*o4 < 0):
        return True
    # Colinear cases
    if abs(o1) < 1e-12 and _on_segment(p1, p2, q1): return True
    if abs(o2) < 1e-12 and _on_segment(p1, p2, q2): return True
    if abs(o3) < 1e-12 and _on_segment(q1, q2, p1): return True
    if abs(o4) < 1e-12 and _on_segment(q1, q2, p2): return True
    return False

def rect_polygon_intersect(rect: Dict, poly: List[Point]) -> bool:
    """
    Conservative test without shapely:
    - any rect corner inside poly
    - any poly vertex inside rect
    - any rect edge intersects any poly edge
    """
    rc = rect_corners(rect)
    # 1) rect corner inside polygon
    for c in rc:
        if point_in_polygon(c, poly):
            return True
    # 2) polygon vertex inside rect
    for v in poly:
        if point_in_rect(v, rect):
            return True
    # 3) edge intersection
    rect_edges = [(rc[i], rc[(i+1)%4]) for i in range(4)]
    poly_edges = [(poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly))]
    for e1 in rect_edges:
        for e2 in poly_edges:
            if segments_intersect(e1[0], e1[1], e2[0], e2[1]):
                return True
    return False
    
