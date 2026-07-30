# perlin_utils.py
from typing import Dict, List, Tuple, Optional
from functools import lru_cache
import math, hashlib, os, copy
from modules.config.base.enums import Category

# ---------- Perlin backend selection ----------
try:
    import noise as _perlin_lib
    _HAS_PERLIN = True
except Exception:
    _HAS_PERLIN = False

_NOISE_BACKEND = os.environ.get("CYBERTOWN_NOISE_BACKEND", "pure").lower()

# ---------- Utilities ----------
def _dist2(p, q):
    dx, dy = p[0]-q[0], p[1]-q[1]
    return dx*dx + dy*dy

def _pt_close(p, q, tol=1e-6):
    return _dist2(p, q) <= tol*tol

def _snap(p, q):
    return (q[0], q[1])

def _edge_key(a, b, quant=1e-6):
    def rr(p, q=quant):
        return (round(p[0]/q)*q, round(p[1]/q)*q)
    aa, bb = rr(a), rr(b)
    return (aa, bb) if aa <= bb else (bb, aa)

def _align_polyline_direction(a, b, base_poly):
    da = _dist2(base_poly[0], a) + _dist2(base_poly[-1], b)
    db = _dist2(base_poly[-1], a) + _dist2(base_poly[0], b)
    return base_poly if da <= db else list(reversed(base_poly))

def _densify_segment(a, b, max_seg_len: float) -> List[Tuple[float,float]]:
    ax, ay = a; bx, by = b
    dx, dy = bx-ax, by-ay
    dist = math.hypot(dx, dy)
    if dist <= max_seg_len:
        return [a, b]
    k = max(1, int(math.floor(dist/max_seg_len)))
    out = [(ax + dx*t/(k+1), ay + dy*t/(k+1)) for t in range(1, k+1)]
    return [a, *out, b]

# ---------- Pure Python Perlin / fBM ----------
def _fade(t: float) -> float:
    return t*t*t*(t*(t*6 - 15) + 10)

def _lerp(a: float, b: float, t: float) -> float:
    return a + t*(b - a)

@lru_cache(maxsize=1_000_000)
def _grad_vec(ix: int, iy: int, seed: int) -> Tuple[float,float]:
    data = f"{ix},{iy},{seed}".encode("utf-8")
    h = hashlib.blake2b(data, digest_size=8).digest()
    u = int.from_bytes(h, "little") / 2**64
    ang = 2.0*math.pi*u
    return (math.cos(ang), math.sin(ang))

def _dot_grid_grad(ix: int, iy: int, x: float, y: float, seed: int) -> float:
    gx, gy = _grad_vec(ix, iy, seed)
    dx = x - ix; dy = y - iy
    return gx*dx + gy*dy

def _perlin_raw2(x: float, y: float, seed: int) -> float:
    x0 = math.floor(x); y0 = math.floor(y)
    x1 = x0 + 1;       y1 = y0 + 1
    sx = _fade(x - x0); sy = _fade(y - y0)
    n00 = _dot_grid_grad(x0, y0, x, y, seed)
    n10 = _dot_grid_grad(x1, y0, x, y, seed)
    n01 = _dot_grid_grad(x0, y1, x, y, seed)
    n11 = _dot_grid_grad(x1, y1, x, y, seed)
    ix0 = _lerp(n00, n10, sx)
    ix1 = _lerp(n01, n11, sx)
    val = _lerp(ix0, ix1, sy)
    return max(-1.0, min(1.0, val * 1.4142135623730951))  # Normalize to [-1,1]

def _perlin2_pure(x: float, y: float, scale: float, octaves: int, seed: int,
                  persistence: float = 0.5, lacunarity: float = 2.0) -> float:
    x *= scale; y *= scale
    amp = 1.0; freq = 1.0
    total = 0.0; norm = 0.0
    oc = max(1, min(int(octaves), 8))
    for _ in range(oc):
        total += amp * _perlin_raw2(x*freq, y*freq, seed)
        norm  += amp
        amp *= persistence
        freq *= lacunarity
    return total / (norm if norm else 1.0)

def _perlin2(x, y, scale, octaves, seed, backend: Optional[str]=None):
    backend = (backend or _NOISE_BACKEND)
    oc = int(max(1, min(int(octaves), 8)))
    if backend in ("pure", "auto") or not _HAS_PERLIN:
        return _perlin2_pure(float(x), float(y), float(scale), oc, int(seed) & 0xffffffff)
    try:
        return float(_perlin_lib.pnoise2(float(x)*float(scale),
                                         float(y)*float(scale),
                                         octaves=oc,
                                         base=(int(seed) & 0xffffffff)))
    except TypeError:
        try:
            return float(_perlin_lib.pnoise2(float(x)*float(scale),
                                             float(y)*float(scale),
                                             oc, 0.5, 2.0, 1024, 1024,
                                             int(seed) & 0xffffffff))
        except Exception:
            return _perlin2_pure(float(x), float(y), float(scale), oc, int(seed) & 0xffffffff)
    except Exception:
        return _perlin2_pure(float(x), float(y), float(scale), oc, int(seed) & 0xffffffff)

# ---------- Geometry ----------
def _poly_centroid(verts: List[Tuple[float,float]]) -> Tuple[float,float]:
    A = 0.0; Cx = 0.0; Cy = 0.0
    n = len(verts)
    for i in range(n):
        x1,y1 = verts[i]; x2,y2 = verts[(i+1)%n]
        cross = x1*y2 - x2*y1
        A  += cross
        Cx += (x1 + x2) * cross
        Cy += (y1 + y2) * cross
    A *= 0.5
    if abs(A) < 1e-12:
        xs=[v[0] for v in verts]; ys=[v[1] for v in verts]
        return (sum(xs)/len(xs), sum(ys)/len(ys))
    return (Cx/(6*A), Cy/(6*A))

def _stitch_ring(original: List[Tuple[float,float]],
                 edge_polylines: Dict[int, List[Tuple[float,float]]],
                 tol=1e-6) -> List[Tuple[float,float]]:
    m = len(original); assert m >= 3
    ring: List[Tuple[float,float]] = []
    for i in range(m):
        a = original[i]; b = original[(i+1)%m]
        poly = edge_polylines.get(i) or [a, b]
        if not _pt_close(poly[0], a, tol):
            poly = [_snap(poly[0], a), *poly[1:]]
        if not _pt_close(poly[-1], b, tol):
            poly = [*poly[:-1], _snap(poly[-1], b)]
        if not ring:
            ring.extend(poly)
        else:
            if _pt_close(ring[-1], poly[0], tol):
                seg = poly[1:]
            elif _pt_close(ring[-1], poly[-1], tol):
                seg = list(reversed(poly[:-1]))
            else:
                poly = [_snap(poly[0], ring[-1]), *poly[1:]]
                seg = poly[1:]
            ring.extend(seg)
    if not _pt_close(ring[-1], ring[0], tol):
        ring[-1] = _snap(ring[-1], ring[0])
    cleaned = [ring[0]]
    for p in ring[1:]:
        if not _pt_close(p, cleaned[-1], tol):
            cleaned.append(p)
    return cleaned

def _world_bbox_from_nodes(nodes) -> Tuple[float,float,float,float]:
    xs, ys = [], []
    for n in nodes:
        p = n.get("properties", {})
        if p.get("category") == Category.DISTRICT.value:
            r = n.get("shape", {})
            if r.get("type") == "rectangle":
                (x1,y1) = r["min_corner"]; (x2,y2) = r["max_corner"]
                xs += [x1,x2]; ys += [y1,y2]
    if not xs:
        xmin,ymin,xmax,ymax = _auto_bounds_from_nodes(nodes)
        return xmin,ymin,xmax,ymax
    return min(xs), min(ys), max(xs), max(ys)

def _is_on_world_edge(p: Tuple[float,float], bbox, eps=1e-6) -> bool:
    x,y = p; xmin,ymin,xmax,ymax = bbox
    return (abs(x-xmin)<eps or abs(x-xmax)<eps or abs(y-ymin)<eps or abs(y-ymax)<eps)

def _edge_on_world_boundary(a: Tuple[float,float], b: Tuple[float,float], bbox, eps=1e-6) -> bool:
    return _is_on_world_edge(a,bbox,eps) and _is_on_world_edge(b,bbox,eps)

def _auto_bounds_from_nodes(nodes):
    xmin = ymin = float("inf"); xmax = ymax = float("-inf")
    def upd(x, y):
        nonlocal xmin, ymin, xmax, ymax
        xmin = min(xmin, x); ymin = min(ymin, y)
        xmax = max(xmax, x); ymax = max(ymax, y)
    for n in nodes:
        shp = n.get("shape", {})
        if not shp: continue
        t = shp.get("type")
        if t == "rectangle":
            (x1, y1) = shp["min_corner"]; (x2, y2) = shp["max_corner"]
            for x, y in [(x1,y1),(x2,y2)]: upd(x, y)
        elif t == "polygon":
            for (x, y) in shp.get("vertices", []): upd(x, y)
        elif t == "point":
            x, y = shp["center"]; upd(x, y)
        elif t == "circle":
            cx, cy = shp["center"]; r = shp["radius"]
            upd(cx - r, cy - r); upd(cx + r, cy + r)
    if not math.isfinite(xmin):
        xmin = ymin = 0.0; xmax = ymax = 1.0
    return xmin, ymin, xmax, ymax

# ---------- Normal jitter ----------
def _jitter_polyline_along_normal(points: List[Tuple[float,float]],
                                  amplitude: float, scale: float, octaves: int,
                                  seed: int, outward_dir: Optional[Tuple[float,float]]=None
                                 ) -> List[Tuple[float,float]]:
    if len(points) < 2:
        return points[:]
    Ns = []
    for i in range(len(points)-1):
        x1,y1=points[i]; x2,y2=points[i+1]
        tx,ty = x2-x1, y2-y1
        L = math.hypot(tx,ty) or 1e-9
        Nx,Ny = -ty/L, tx/L
        Ns.append((Nx,Ny))
    out=[]
    for i,(x,y) in enumerate(points):
        if i==0: nx,ny = Ns[0]
        elif i==len(points)-1: nx,ny = Ns[-1]
        else:
            nx,ny = (Ns[i-1][0]+Ns[i][0], Ns[i-1][1]+Ns[i][1])
            nL = math.hypot(nx,ny) or 1e-9
            nx,ny = nx/nL, ny/nL
        if outward_dir is not None and (nx*outward_dir[0] + ny*outward_dir[1] < 0):
            nx,ny = -nx,-ny
        nval = _perlin2(x,y,scale,octaves,seed)
        d = amplitude * float(nval)
        out.append((x + nx*d, y + ny*d))
    return out

# ---------- Main entry ----------
def apply_perlin_to_areas_consistent(
    nodes,
    seed: int = 42,
    types=("water_body","garden"),
    amplitude: float = 8.0,
    scale: float = 0.02,
    octaves: int = 2,
    max_seg_len: float = 18.0,
    edge_quant: float = 1e-6,
    world_eps: float = 1e-3,
    tol_join: float = 1e-6,
    street_match_quant: float = 0.5,
):
    nodes_in = copy.deepcopy(nodes)
    world_bbox = _world_bbox_from_nodes(nodes_in)
    edge_curves: Dict[Tuple, List[Tuple[float,float]]] = {}

    # Collect all AREA polygons
    area_nodes = []
    for n in nodes_in:
        props = n.get("properties", {})
        if props.get("category") == Category.AREA.value and n.get("shape",{}).get("type") == "polygon":
            V = n["shape"]["vertices"]
            if len(V) >= 3:
                area_nodes.append(n)

    # Edge index
    edge_map: Dict[Tuple, List[Tuple[int,int,Tuple[float,float],Tuple[float,float]]]] = {}
    for idx, n in enumerate(area_nodes):
        V = n["shape"]["vertices"]; m = len(V)
        for i in range(m):
            a = V[i]; b = V[(i+1)%m]
            if _pt_close(a, b, tol=1e-12):
                continue
            key = _edge_key(a, b, quant=edge_quant)
            edge_map.setdefault(key, []).append((idx, i, a, b))

    replace_map: Dict[Tuple[int,int], List[Tuple[float,float]]] = {}
    shared_cache: Dict[Tuple, List[Tuple[float,float]]] = {}

    for key, refs in edge_map.items():
        idx0, i0, a0, b0 = refs[0]
        base = _densify_segment(a0, b0, max_seg_len)

        # At least one side is a target type?
        touches_target = any(area_nodes[idx]["properties"].get("type") in types for (idx, _, _, _) in refs)

        if _edge_on_world_boundary(a0, b0, world_bbox, eps=world_eps):
            for (idx, i, _, _) in refs:
                replace_map[(idx, i)] = base
            continue

        if touches_target:
            if key not in shared_cache:
                # Stable hash for cross-platform reproducibility
                h = hashlib.blake2b(repr(key).encode("utf-8"), digest_size=8).digest()
                seed_key = (int.from_bytes(h, "little") ^ (seed & 0x7fffffff)) & 0x7fffffff
                jit = _jitter_polyline_along_normal(base, amplitude, scale, octaves, seed=seed_key)
                shared_cache[key] = jit
            jit = shared_cache[key]
            edge_curves[key] = jit
            for (idx, i, a, b) in refs:
                replace_map[(idx, i)] = _align_polyline_direction(a, b, jit)
        else:
            for (idx, i, _, _) in refs:
                replace_map[(idx, i)] = base

    # Write back vertices
    for idx, n in enumerate(area_nodes):
        V = n["shape"]["vertices"]; m = len(V)
        edge_dict = {}
        for i in range(m):
            repl = replace_map.get((idx, i))
            if repl and len(repl) >= 2:
                edge_dict[i] = repl
        n["shape"]["vertices"] = _stitch_ring(V, edge_dict, tol=tol_join)

    # Loose matching table for road lookup
    def _edge_key_q(a, b, q):
        aa = (round(a[0]/q)*q, round(a[1]/q)*q)
        bb = (round(b[0]/q)*q, round(b[1]/q)*q)
        return (aa, bb) if aa <= bb else (bb, aa)

    edge_curves_loose = {}
    for key, poly in edge_curves.items():
        a, b = key
        edge_curves_loose[_edge_key_q(a, b, street_match_quant)] = poly

    return {
        "edge_curves": edge_curves,
        "edge_curves_loose": edge_curves_loose,
        "street_match_quant": street_match_quant,
        "nodes": nodes_in,  # Noise-applied copy for rendering
    }
