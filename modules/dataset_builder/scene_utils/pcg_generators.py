# pcg_generators.py
from typing import List, Dict, Tuple, Optional
import random
from random import Random

from .geometry_utils import (
    Point, Polygon, polygon_from_bbox, clip_poly_to_bbox
)

# Optional libs
try:
    from scipy.spatial import Voronoi
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _random_seeds(bounds: Dict[str, float], n: int, rng: Random) -> List[Tuple[float, float]]:
    return [(rng.uniform(bounds['x_min'], bounds['x_max']),
             rng.uniform(bounds['y_min'], bounds['y_max'])) for _ in range(n)]

def _subdivide_rect(bounds: Dict[str, float], target_n: int) -> List[Polygon]:
    """Axis-aligned recursive subdivision until reaching target_n cells."""
    polys = [polygon_from_bbox(bounds)]
    while len(polys) < target_n:
        p = polys.pop(0)
        # split along longer axis
        xs = [pt[0] for pt in p]; ys = [pt[1] for pt in p]
        minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
        if (maxx - minx) >= (maxy - miny):
            mid = (minx + maxx) / 2.0
            p1 = [(minx, miny), (mid, miny), (mid, maxy), (minx, maxy)]
            p2 = [(mid, miny), (maxx, miny), (maxx, maxy), (mid, maxy)]
        else:
            mid = (miny + maxy) / 2.0
            p1 = [(minx, miny), (maxx, miny), (maxx, mid), (minx, mid)]
            p2 = [(minx, mid), (maxx, mid), (maxx, maxy), (minx, maxy)]
        polys.extend([p1, p2])
    return polys[:target_n]

def generate_zones_voronoi(bounds, n_cells, rng=None):
    if rng is None:
        from random import Random
        rng = Random(0)

    xmin, ymin, xmax, ymax = bounds["x_min"], bounds["y_min"], bounds["x_max"], bounds["y_max"]

    try:
        from shapely.geometry import Point, Polygon, MultiPoint, box
        from shapely.ops import voronoi_diagram
        # 1) Sample (over-sample then take first n_cells for more natural shapes)
        seeds = _random_seeds(bounds, max(n_cells, 6), rng)
        mp = MultiPoint([Point(x, y) for x, y in seeds])
        env = box(xmin, ymin, xmax, ymax)

        # 2) Generate clipped Voronoi; polygons are already within the envelope
        vd = voronoi_diagram(mp, envelope=env, tolerance=0.0)  # MultiPolygon
        polys = [p.intersection(env) for p in vd.geoms if not p.is_empty]

        # 3) Keep only the first n_cells; export as vertex lists
        out = []
        for p in polys[:n_cells]:
            xs, ys = p.exterior.xy
            out.append([(float(x), float(y)) for x, y in zip(xs, ys)][:-1])  # Drop closing duplicate point
        if out:
            return out
    except Exception:
        pass  # Shapely not installed or version too old; fall back to alternative

    return _subdivide_rect(bounds, n_cells)


def skeleton_roads_from_polygons(polys: List[Polygon]) -> Tuple[List[Tuple[Point, Point]], List[Point]]:
    """
    Build primary road segments along polygon borders (simple heuristic):
    - For each shapely Polygon, add its exterior edges as candidate street segments
    - Intersections are polygon vertices shared by >= 2 polygons (approximate)
    """
    from collections import defaultdict
    edge_set = set()
    vertex_count = defaultdict(int)
    segments: List[Tuple[Point, Point]] = []

    def norm_edge(a: Point, b: Point) -> Tuple[Point, Point]:
        # Undirected edge: smaller point first
        return (a, b) if a < b else (b, a)

    for poly in polys:
        # shapely Polygon.exterior auto-closes; first point == last point
        coords = list(poly.exterior.coords)
        n = len(coords) - 1
        for i in range(n):
            a = (float(coords[i][0]), float(coords[i][1]))
            b = (float(coords[i+1][0]), float(coords[i+1][1]))
            edge = norm_edge(a, b)
            if edge not in edge_set:
                edge_set.add(edge)
                segments.append((a, b))
            vertex_count[a] += 1
            vertex_count[b] += 1

    # "Intersection" defined as: a vertex shared by edges of at least two polygons
    intersections = [v for v, cnt in vertex_count.items() if cnt >= 2]
    return segments, intersections
