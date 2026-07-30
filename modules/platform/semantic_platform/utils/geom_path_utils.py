# geom_path_utils.py
# Pure function utilities for geometry and path
# Notes:
# - About obstacles: assumes obstacle.shape contains at least
#     - For rectangle: {'type': 'rectangle', 'min_corner': [x_min, y_min], 'max_corner': [x_max, y_max]}
#     - For circle: {'type': 'circle', 'center': [cx, cy], 'radius': r}
#   For circles, converts to bounding rectangle when needed.

from typing import List, Tuple, Any
import numpy as np


# ===================== Basic geometry =====================

def _rect_from_obstacle_shape(shape: dict, margin: float = 0.0) -> Tuple[float, float, float, float]:
    """
    Convert obstacle.shape to rectangle (xmin, ymin, xmax, ymax), uses bounding rect for circles if needed.
    """
    t = shape.get("type", "rectangle")
    if t == "circle":
        cx, cy = shape["center"]
        r = float(shape["radius"]) + float(margin)
        return (cx - r, cy - r, cx + r, cy + r)
    else:
        # rectangle (default):
        xmin, ymin = shape["min_corner"]
        xmax, ymax = shape["max_corner"]
        return (xmin - margin, ymin - margin, xmax + margin, ymax + margin)


def point_in_any_obstacle(point: np.ndarray, obstacles: List[Any], margin: float = 0.0) -> bool:
    """
    Check if point is inside any obstacle (with margin), including boundaries.
    - For circular obstacles, uses bounding rectangle for fast check (simple and robust).
    """
    px, py = float(point[0]), float(point[1])
    for obs in obstacles:
        xmin, ymin, xmax, ymax = _rect_from_obstacle_shape(obs.shape, margin=margin)
        if (xmin <= px <= xmax) and (ymin <= py <= ymax):
            return True
    return False


def line_intersects_obstacle(start: List[float], end: List[float], obstacle: Any) -> bool:
    """
    Check if line segment intersects obstacle (rectangle/circle).
    - Converts circle to bounding rectangle first, performs AABB line segment intersection test (simple and robust).
    Reference approach: StackOverflow#99353 (line segment and AABB intersection)
    """
    x1, y1 = float(start[0]), float(start[1])
    x2, y2 = float(end[0]), float(end[1])

    xmin, ymin, xmax, ymax = _rect_from_obstacle_shape(obstacle.shape, margin=0.0)

    # Fast rejection
    if max(x1, x2) < xmin or min(x1, x2) > xmax:
        return False
    if max(y1, y2) < ymin or min(y1, y2) > ymax:
        return False

    # Endpoint inside
    if (xmin <= x1 <= xmax and ymin <= y1 <= ymax) or (xmin <= x2 <= xmax and ymin <= y2 <= ymax):
        return True

    dx, dy = x2 - x1, y2 - y1

    # Intersect with four edges
    if dx != 0.0:
        # x = xmin
        t = (xmin - x1) / dx
        if 0.0 <= t <= 1.0:
            y = y1 + t * dy
            if ymin <= y <= ymax:
                return True
        # x = xmax
        t = (xmax - x1) / dx
        if 0.0 <= t <= 1.0:
            y = y1 + t * dy
            if ymin <= y <= ymax:
                return True

    if dy != 0.0:
        # y = ymin
        t = (ymin - y1) / dy
        if 0.0 <= t <= 1.0:
            x = x1 + t * dx
            if xmin <= x <= xmax:
                return True
        # y = ymax
        t = (ymax - y1) / dy
        if 0.0 <= t <= 1.0:
            x = x1 + t * dx
            if xmin <= x <= xmax:
                return True

    return False


# ===================== Path generation/processing =====================

def generate_straight_path(start: np.ndarray, end: np.ndarray, num_points: int) -> List[np.ndarray]:
    """
    Generate straight line path (num_points points, including start and end).
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    n = max(2, int(num_points))
    path: List[np.ndarray] = []
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        point = start * (1.0 - t) + end * t
        path.append(point)
    return path


def generate_bezier_path(p0: List[float], p1: List[float], p2: List[float], num_points: int) -> List[np.ndarray]:
    """
    Generate quadratic Bezier curve (p0, p1, p2) path.
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    n = max(2, int(num_points))
    path: List[np.ndarray] = []
    for i in range(n):
        t = 0.0 if n == 1 else i / (n - 1)
        point = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
        path.append(point)
    return path


def generate_smooth_path(waypoints: List[np.ndarray], num_points: int) -> List[np.ndarray]:
    """
    Generate smooth path using cubic Bezier segment splicing (consistent with original class implementation).
    """
    if not waypoints:
        return []
    if len(waypoints) == 1:
        return [np.asarray(waypoints[0], dtype=float)]
    if len(waypoints) == 2:
        return generate_straight_path(waypoints[0], waypoints[1], num_points)

    way = [np.asarray(w, dtype=float) for w in waypoints]
    smooth_path: List[np.ndarray] = []
    segments = len(way) - 1
    points_per_segment = max(3, int(num_points // segments))

    for i in range(segments):
        p0 = way[i]
        p3 = way[i + 1]

        # Control point p1
        if i == 0:
            direction = way[i + 1] - way[i]
            p1 = p0 + direction * 0.3
        else:
            tangent = (way[i + 1] - way[i - 1]) / 2.0
            p1 = p0 + tangent * 0.3

        # Control point p2
        if i == segments - 1:
            direction = way[i + 1] - way[i]
            p2 = p3 - direction * 0.3
        else:
            tangent = (way[i + 2] - way[i]) / 2.0
            p2 = p3 - tangent * 0.3

        seg_pts = points_per_segment if i < segments - 1 else max(2, num_points - len(smooth_path))
        for j in range(seg_pts):
            t = 0.0 if seg_pts <= 1 else j / (seg_pts - 1)
            point = (
                (1 - t) ** 3 * p0
                + 3 * (1 - t) ** 2 * t * p1
                + 3 * (1 - t) * (t ** 2) * p2
                + (t ** 3) * p3
            )
            if not smooth_path or np.linalg.norm(point - smooth_path[-1]) > 0.1:
                smooth_path.append(point)

    if len(smooth_path) > 0 and np.linalg.norm(smooth_path[-1] - way[-1]) > 0.1:
        smooth_path.append(way[-1])

    return smooth_path


# ===================== Graph shortest path =====================

def dijkstra(n: int, edges: List[Tuple[int, int, float]], start_idx: int, goal_idx: int) -> List[int]:
    """
    Dijkstra shortest path (undirected graph).
    Parameters
    - n: number of nodes
    - edges: list of (u, v, weight)
    - start_idx, goal_idx: start and goal node indices
    Returns
    - path of node indices, returns [] if unreachable
    """
    import heapq

    graph: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        graph[u].append((v, float(w)))
        graph[v].append((u, float(w)))

    dist = [float("inf")] * n
    parent = [-1] * n
    dist[start_idx] = 0.0

    pq: List[Tuple[float, int]] = [(0.0, start_idx)]
    visited = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == goal_idx:
            break

        for v, w in graph[u]:
            if v in visited:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    if parent[goal_idx] == -1 and start_idx != goal_idx:
        return []

    path = []
    cur = goal_idx
    while cur != -1:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path
