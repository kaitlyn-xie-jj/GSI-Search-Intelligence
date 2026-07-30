# -*- coding: utf-8 -*-
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

# ---------- Basic geometry ----------
def euclid(a: Optional[List[float]], b: Optional[List[float]]) -> float:
    if not a or not b: return 0.0
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def point_on_segment(p, a, b, eps=1e-9) -> bool:
    ax, ay = a; bx, by = b; px, py = p
    cross = abs((bx-ax)*(py-ay) - (by-ay)*(px-ax))
    if cross > eps: return False
    dot = (px-ax)*(bx-ax) + (py-ay)*(by-ay)
    if dot < -eps: return False
    sq_len = (bx-ax)**2 + (by-ay)**2
    if dot - sq_len > eps: return False
    return True

def point_in_polygon(p: List[float], poly: List[List[float]]) -> bool:
    if not poly: return False
    x, y = p; inside = False; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        if point_on_segment([x,y], [x1,y1], [x2,y2]): return True
        if ((y1 <= y < y2) or (y2 <= y < y1)):
            xinters = x1 + (y-y1) * (x2-x1) / (y2-y1 + 1e-12)
            if xinters >= x - 1e-12: inside = not inside
    return inside

def polygon_area(poly: List[List[float]]) -> float:
    if not poly: return 0.0
    s = 0.0; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        s += x1*y2 - x2*y1
    return 0.5 * s

def polygon_centroid(poly: List[List[float]]) -> Optional[List[float]]:
    if not poly: return None
    A = polygon_area(poly)
    if abs(A) < 1e-12:
        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
        return [sum(xs)/len(xs), sum(ys)/len(ys)]
    cx = cy = 0.0; n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]; x2, y2 = poly[(i+1) % n]
        cross = x1*y2 - x2*y1
        cx += (x1+x2)*cross; cy += (y1+y2)*cross
    cx /= (6.0*A); cy /= (6.0*A); return [cx, cy]

def point_to_polyline_distance(p: List[float], line: List[List[float]]) -> Tuple[Optional[float], Optional[List[float]]]:
    if not line: return None, None
    px, py = p; dmin = None; qmin = None
    for i in range(len(line)-1):
        ax, ay = line[i]; bx, by = line[i+1]
        vx, vy = bx-ax, by-ay; wx, wy = px-ax, py-ay
        denom = vx*vx + vy*vy
        t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, (wx*vx + wy*vy)/denom))
        qx, qy = ax + t*vx, ay + t*vy
        d = ((px-qx)**2 + (py-qy)**2)**0.5
        if (dmin is None) or (d < dmin): dmin, qmin = d, [qx, qy]
    return dmin, qmin

def polyline_length(line: List[List[float]]) -> float:
    if not line or len(line) < 2: return 0.0
    return sum(euclid(line[i], line[i+1]) for i in range(len(line)-1))

def point_in_area_geometry(pt: List[float], geom: Dict[str, Any]) -> bool:
    kind = geom.get('kind')
    if kind in ('area', 'rectangle'):
        return point_in_polygon(pt, geom.get('coords', []))
    if kind == 'circle':
        c = geom.get('center'); r = float(geom.get('radius', 0.0) or 0.0)
        return c is not None and euclid(pt, c) <= r + 1e-9
    if kind == 'line':
        line = geom.get('coords', []); buf = 20.0
        d, _ = point_to_polyline_distance(pt, line)
        return (d is not None) and (d <= buf + 1e-9)
    if kind == 'point':
        pts = geom.get('coords', []); buf = 20.0
        p = pts[0] if pts else None
        return p is not None and euclid(pt, p) <= buf + 1e-9
    return False

def area_centroid(area_geom: Dict[str, Any]) -> Optional[List[float]]:
    """Get area center: point, circle center, polygon vertex-average centroid, line midpoint, or rectangle center."""
    if not isinstance(area_geom, dict):
        return None
    k = area_geom.get("kind")
    if k == "point":
        coords = area_geom.get("coords") or []
        if coords and len(coords[0]) >= 2:
            return [float(coords[0][0]), float(coords[0][1])]
    elif k == "circle":
        center = area_geom.get("center")
        if isinstance(center, (list, tuple)) and len(center) >= 2:
            return [float(center[0]), float(center[1])]
    elif k == "area":
        coords = area_geom.get("coords") or []
        if coords:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            return [float(sum(xs) / len(xs)), float(sum(ys) / len(ys))]
    elif k == "line":
        coords = area_geom.get("coords") or []
        if len(coords) >= 2:
            p1, p2 = coords[0], coords[-1]
            return [float((p1[0] + p2[0]) / 2.0), float((p1[1] + p2[1]) / 2.0)]
    elif k == "rectangle":
        coords = area_geom.get("coords") or []
        if len(coords) >= 2:
            if len(coords) == 4:
                xs = [c[0] for c in coords]
                ys = [c[1] for c in coords]
                return [float((min(xs) + max(xs)) / 2.0), float((min(ys) + max(ys)) / 2.0)]
            else:
                minc, maxc = coords[0], coords[-1]
                return [float((minc[0] + maxc[0]) / 2.0), float((minc[1] + maxc[1]) / 2.0)]
            
    return None

def distance(p1: Optional[List[float]], p2: Optional[List[float]]) -> Optional[float]:
    """Distance between two points, or None if either point is None."""
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def point_to_segment_distance(p: List[float], a: List[float], b: List[float]) -> Optional[float]:
    """Shortest distance from point p to segment a-b."""
    if not p or not a or not b:
        return None
    # Reuse the polyline implementation.
    d, _ = point_to_polyline_distance(p, [a, b])
    return d

def segment_intersects_area(p1: List[float],
                            p2: List[float],
                            geom: Dict[str, Any],
                            max_samples: int = 40,
                            step_length: float = 40.0) -> bool:
    """
    Check whether segment p1->p2 intersects or crosses the given area geometry.

    For area, rectangle, circle, line, and point geometries:
      - Sample evenly spaced points along the segment.
      - Use point_in_area_geometry to test whether sampled points are inside.
    If any sampled point is inside, the segment is considered to cross the area.

    max_samples: Maximum number of sampled points.
    step_length: Desired sampling interval in coordinate-system units.
    """
    if p1 is None or p2 is None or not geom:
        return False

    L = euclid(p1, p2)
    if L <= 0.0:
        # Degenerate case: check whether the point is inside the area.
        return point_in_area_geometry(p1, geom)

    # Adapt sample count by length without exceeding max_samples.
    n = max(1, min(max_samples, int(L / max(step_length, 1.0))))
    x1, y1 = float(p1[0]), float(p1[1])
    x2, y2 = float(p2[0]), float(p2[1])

    for i in range(n + 1):
        t = i / float(n)
        px = x1 + (x2 - x1) * t
        py = y1 + (y2 - y1) * t
        if point_in_area_geometry([px, py], geom):
            return True
    return False
