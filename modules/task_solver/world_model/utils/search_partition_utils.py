# -*- coding: utf-8 -*-
from typing import List, Dict, Any, Tuple, Optional
import math

EPS = 1e-9


def _interpolate_point(a: List[float], b: List[float], t: float) -> List[float]:
    """Return the point at fraction t between a and b."""
    ax, ay = a
    bx, by = b
    return [ax + t * (bx - ax), ay + t * (by - ay)]


def split_rectangle(rect_coords: List[List[float]], n: int) -> List[Dict[str, Any]]:
    # rect_coords: four rectangle vertices; any order is accepted and split by bbox.
    xs = [p[0] for p in rect_coords]
    ys = [p[1] for p in rect_coords]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    width = (x1 - x0) / max(n, 1)
    parts = []
    for i in range(n):
        xa = x0 + i*width; xb = x0 + (i+1)*width
        parts.append({
            "kind": "rectangle",
            "coords": [[xa,y0],[xb,y0],[xb,y1],[xa,y1]],
        })
    return parts

def split_polygon_stripes_strict(
    poly_coords: List[List[float]],
    n: int,
    axis: str = "x",           # "x": vertical stripes; "y": horizontal stripes.
    expand: float = 0.0        # Optional padding around each stripe to avoid numeric boundary misses.
) -> List[Dict[str, Any]]:
    """
    Strict trim: split a polygon's bounding box into n equal stripes, then use
    Sutherland-Hodgman clipping against each stripe rectangle to return
    sub-polygons fully inside the original polygon.

    Returns: [{"kind":"area","coords":[[x,y],...]} , ...], with empty results removed.
    """
    if not poly_coords or n <= 0:
        return []

    xs = [p[0] for p in poly_coords]
    ys = [p[1] for p in poly_coords]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)

    parts: List[Dict[str, Any]] = []

    if axis == "x":
        # Vertical stripes.
        width = (x1 - x0) / float(n)
        for i in range(n):
            xa = x0 + i * width
            xb = x0 + (i + 1) * width
            # Clipping rectangle for each stripe, with optional padding for float edges.
            clip_rect = (xa, xb, y0 - expand, y1 + expand)  # (xmin, xmax, ymin, ymax)
            clipped = _clip_polygon_with_rect(poly_coords, clip_rect)
            if len(clipped) >= 3:
                parts.append({"kind": "area", "coords": clipped})
    else:
        # Horizontal stripes.
        height = (y1 - y0) / float(n)
        for i in range(n):
            ya = y0 + i * height
            yb = y0 + (i + 1) * height
            clip_rect = (x0 - expand, x1 + expand, ya, yb)  # (xmin, xmax, ymin, ymax)
            clipped = _clip_polygon_with_rect(poly_coords, clip_rect)
            if len(clipped) >= 3:
                parts.append({"kind": "area", "coords": clipped})

    return parts

def split_polygon_bbox_stripes(poly_coords: List[List[float]], n: int) -> List[Dict[str, Any]]:
    # Simplified: split the polygon bbox into vertical stripes for approximate assignment.
    xs = [p[0] for p in poly_coords]; ys = [p[1] for p in poly_coords]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    width = (x1 - x0) / max(n, 1)
    parts = []
    for i in range(n):
        xa = x0 + i*width; xb = x0 + (i+1)*width
        parts.append({
            "kind": "rectangle",   # Approximate with rectangle stripes; can upgrade to strict trim later.
            "coords": [[xa,y0],[xb,y0],[xb,y1],[xa,y1]],
        })
    return parts

def split_circle_sectors(
    center: List[float],
    radius: float,
    n: int,
    max_arc_step_deg: float = 12.0,   # Max arc sampling step in degrees; smaller is smoother.
    start_angle: float = 0.0          # Optional global start angle in radians.
) -> List[Dict[str, Any]]:
    """
    Split a circle into n sectors, approximating each sector as a polygon:
      coords = [center, arc_point_0, arc_point_1, ..., arc_point_m]
    Arc segments are subdivided by max_arc_step_deg to avoid degenerate triangles when n=2.
    """
    parts: List[Dict[str, Any]] = []
    if radius <= 0 or n <= 0:
        return parts

    cx, cy = center
    two_pi = 2.0 * math.pi
    sector_angle = two_pi / float(n)
    step = max(math.radians(1e-3), math.radians(max_arc_step_deg))  # Lower bound prevents zero.

    for i in range(n):
        a0 = start_angle + i * sector_angle
        a1 = a0 + sector_angle

        # Compute the number of arc segments for this sector.
        segs = max(2, int(math.ceil((a1 - a0) / step)))

        # Generate arc sample points, including endpoints.
        coords = [[cx, cy]]
        for k in range(segs + 1):
            a = a0 + (a1 - a0) * (k / segs)
            coords.append([cx + radius * math.cos(a), cy + radius * math.sin(a)])

        parts.append({"kind": "area", "coords": coords})

    return parts

def split_polyline_by_length(line: List[List[float]], n: int) -> List[Dict[str, Any]]:
    # Split a polyline evenly by arc length; buffer is handled by upstream params['line_buffer'].
    if not line or n <= 1:
        return [{"kind": "line", "coords": line}]
    # Compute cumulative lengths.
    lens = [0.0]
    for i in range(len(line)-1):
        ax, ay = line[i]; bx, by = line[i+1]
        seg = ((bx-ax)**2 + (by-ay)**2)**0.5
        lens.append(lens[-1] + seg)
    total = lens[-1] if lens else 0.0
    targets = [total * k / n for k in range(1, n)]
    parts, seg_i, acc = [], 0, 0.0
    cur = [line[0]]
    for tlen in targets + [total]:
        while seg_i < len(line)-1 and lens[seg_i+1] < tlen - 1e-9:
            cur.append(line[seg_i+1]); seg_i += 1
        if seg_i < len(line)-1:
            seg_len = lens[seg_i+1] - lens[seg_i]
            t = 0.0 if seg_len < 1e-12 else (tlen - lens[seg_i]) / seg_len
            cut_pt = _interpolate_point(line[seg_i], line[seg_i+1], t)
            cur.append(cut_pt)
            parts.append({"kind": "line", "coords": cur})
            cur = [cut_pt]
    return parts

def _clip_polygon_with_rect(poly: List[List[float]], rect: Tuple[float, float, float, float]) -> List[List[float]]:
    xmin, xmax, ymin, ymax = rect
    # Clip against the four half-planes: x>=xmin, x<=xmax, y>=ymin, y<=ymax.
    out = poly
    out = _clip_against_halfplane(out, lambda P: P[0] >= xmin - EPS, _intersect_x_eq, xmin)
    out = _clip_against_halfplane(out, lambda P: P[0] <= xmax + EPS, _intersect_x_eq, xmax)
    out = _clip_against_halfplane(out, lambda P: P[1] >= ymin - EPS, _intersect_y_eq, ymin)
    out = _clip_against_halfplane(out, lambda P: P[1] <= ymax + EPS, _intersect_y_eq, ymax)
    return out

def _clip_against_halfplane(poly: List[List[float]], inside, intersect_func, k) -> List[List[float]]:
    if not poly:
        return []
    res: List[List[float]] = []
    n = len(poly)
    for i in range(n):
        A = poly[i - 1]
        B = poly[i]
        Ain = inside(A)
        Bin = inside(B)
        if Ain and Bin:
            # A and B are inside: keep B.
            res.append(B)
        elif Ain and not Bin:
            # Exiting: add the intersection.
            I = intersect_func(A, B, k)
            if I is not None:
                res.append(I)
        elif (not Ain) and Bin:
            # Entering: add the intersection and B.
            I = intersect_func(A, B, k)
            if I is not None:
                res.append(I)
            res.append(B)
        # else: both are outside, add nothing.
    # Deduplicate or merge adjacent duplicate points.
    if len(res) > 1 and _dist2(res[0], res[-1]) < EPS*EPS:
        res.pop()  # Remove duplicate closing point.
    return res

def _intersect_x_eq(A: List[float], B: List[float], xk: float) -> List[float]:
    # Intersection of segment AB with x = xk.
    ax, ay = A; bx, by = B
    dx = bx - ax
    if abs(dx) < EPS:
        return [xk, ay]  # Parallel to x=constant; degenerate case keeps y=Ay.
    t = (xk - ax) / dx
    t = max(0.0, min(1.0, t))
    y = ay + t * (by - ay)
    return [xk, y]

def _intersect_y_eq(A: List[float], B: List[float], yk: float) -> List[float]:
    # Intersection of segment AB with y = yk.
    ax, ay = A; bx, by = B
    dy = by - ay
    if abs(dy) < EPS:
        return [ax, yk]  # Parallel to y=constant; degenerate case keeps x=Ax.
    t = (yk - ay) / dy
    t = max(0.0, min(1.0, t))
    x = ax + t * (bx - ax)
    return [x, yk]

def _dist2(a: List[float], b: List[float]) -> float:
    dx = a[0] - b[0]; dy = a[1] - b[1]
    return dx*dx + dy*dy
