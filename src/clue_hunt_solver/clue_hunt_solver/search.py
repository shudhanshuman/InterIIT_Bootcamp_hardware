"""Viewing poses, ring viewpoints, arena tour, goal snapping (final_plan.md
§4.2, §4.6, §4.8) plus the map-grid IO they all share.

Pure Python + numpy/cv2 (no ROS imports) so every piece is unit-testable.
Grid convention matches the ROS map_server image convention:
row 0 = TOP = maximum y; cell (r, c) centre is
x = ox + (c + 0.5) * res,  y = oy + (H - r - 0.5) * res.
ROS OccupancyGrid (row 0 = bottom) must be np.flipud'ed before wrapping.
"""
import math
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml

from .vision import DEFAULT_VIEW_DISTANCE

SEARCH_RADIUS = 1.8        # m around a command target where the next board may sit
RING_RADIUS = 2.0          # m ring of L1 viewpoints around the target
RING_N = 6
CLEARANCE_M = 0.45         # tour viewpoint clearance from obstacles (robot 0.28 + margin)
TOUR_N = 10                # 8-12 per final_plan §4.6
TREASURE_SNAP_M = 0.25     # final_plan §4.8: stay inside the 0.3 m tolerance
VIEW_WIDEN_DEG = 30        # max sideways angle when the direct viewing pose is blocked


@dataclass
class Grid:
    occ: np.ndarray                 # bool (H, W), True = occupied
    free: np.ndarray                # bool (H, W), True = free (known)
    ox: float
    oy: float
    res: float

    @property
    def h(self) -> int:
        return self.occ.shape[0]

    @property
    def w(self) -> int:
        return self.occ.shape[1]

    def cell_to_xy(self, r: int, c: int) -> Tuple[float, float]:
        return (self.ox + (c + 0.5) * self.res,
                self.oy + (self.h - r - 0.5) * self.res)

    def xy_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        c = int(math.floor((x - self.ox) / self.res))
        r = int(math.floor(self.h - (y - self.oy) / self.res))
        return r, c

    def in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.h and 0 <= c < self.w


def load_grid(yaml_path: str) -> Grid:
    """Read a map yaml + PGM into a Grid (occupied / free / unknown)."""
    meta = {}
    with open(yaml_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, _, val = line.partition(':')
            meta[key.strip()] = val.strip()
    image = meta['image']
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image)
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'cannot read map image {image}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    occ_t = float(meta.get('occupied_thresh', 0.65))
    free_t = float(meta.get('free_thresh', 0.25))
    negate = int(meta.get('negate', 0))
    p = (255.0 - img) / 255.0 if not negate else img / 255.0
    ox, oy, _ = (float(v) for v in meta['origin'].strip('[]').split(','))
    return Grid(occ=p > occ_t, free=p < free_t, ox=ox, oy=oy,
                res=float(meta['resolution']))


def grid_from_occupancy(data: Sequence[int], width: int, height: int,
                        ox: float, oy: float, res: float) -> Grid:
    """Wrap a ROS OccupancyGrid payload (row 0 = bottom) into a Grid."""
    arr = np.asarray(data, dtype=np.int16).reshape(height, width)
    arr = np.flipud(arr)                       # -> row 0 = top
    return Grid(occ=arr > 50, free=arr == 0, ox=ox, oy=oy, res=res)


# ---------------------------------------------------------------------------
# Viewing pose (final_plan §4.2)
# ---------------------------------------------------------------------------
def viewing_pose(board_xy: Sequence[float], normal_yaw: float,
                 dist: float = DEFAULT_VIEW_DISTANCE) -> Tuple[float, float, float]:
    """Stand `dist` along the board's outward normal, facing the board."""
    bx, by = board_xy
    gx = bx + dist * math.cos(normal_yaw)
    gy = by + dist * math.sin(normal_yaw)
    yaw = math.atan2(by - gy, bx - gx)
    return (gx, gy, yaw)


def viewing_pose_candidates(board_xy: Sequence[float], normal_yaw: float,
                            max_deg: float = VIEW_WIDEN_DEG, step_deg: float = 10.0,
                            dist: float = DEFAULT_VIEW_DISTANCE
                            ) -> List[Tuple[float, float, float]]:
    """Direct pose first, then sidesteps of +/-step up to max_deg (§4.2)."""
    out = [viewing_pose(board_xy, normal_yaw, dist)]
    n = int(max_deg // step_deg)
    for k in range(1, n + 1):
        for sign in (+1.0, -1.0):
            out.append(viewing_pose(board_xy,
                                    normal_yaw + math.radians(sign * k * step_deg),
                                    dist))
    return out


def ring_viewpoints(cx: float, cy: float, radius: float = RING_RADIUS,
                    n: int = RING_N, start_yaw: float = 0.0
                    ) -> List[Tuple[float, float, float]]:
    """n viewpoints on a ring around (cx, cy), each facing the centre (§4.6 L1)."""
    out = []
    for i in range(n):
        a = start_yaw + 2.0 * math.pi * i / n
        x = cx + radius * math.cos(a)
        y = cy + radius * math.sin(a)
        out.append((x, y, math.atan2(cy - y, cx - x)))
    return out


def pose_sees_point(x: float, y: float, yaw: float,
                    tx: float, ty: float,
                    max_dist: float = 2.5, min_dist: float = 0.4,
                    half_fov: float = math.radians(30.0)) -> bool:
    """True if a camera at (x, y, yaw) can plausibly decode a board at (tx, ty):
    inside QR decode range and inside the horizontal field of view."""
    dx, dy = tx - x, ty - y
    d = math.hypot(dx, dy)
    if not min_dist <= d <= max_dist:
        return False
    want = math.atan2(dy, dx)
    return abs((yaw - want + math.pi) % (2.0 * math.pi) - math.pi) <= half_fov


def angular_diff(a: float, b: float) -> float:
    """Absolute wrapped difference of two yaws in [0, pi]."""
    return abs((a - b + math.pi) % (2.0 * math.pi) - math.pi)


def same_side_as_observers(bx: float, by: float,
                           observers: Sequence[Sequence[float]],
                           cx: float, cy: float,
                           max_deg: float = 60.0) -> bool:
    """True if candidate (cx, cy) lies on the board's observed (front) side.

    Observer positions saw the marker, so they are front-side by construction -
    this holds even when the PnP normal is flipped or garbage. No observers
    recorded -> no opinion (True).
    """
    if not observers:
        return True
    want = math.atan2(cy - by, cx - bx)
    lim = math.radians(max_deg)
    for ox, oy in observers:
        if angular_diff(want, math.atan2(oy - by, ox - bx)) <= lim:
            return True
    return False


def incidence_ok(bx: float, by: float, normal: float,
                 cx: float, cy: float,
                 max_deg: float = 40.0) -> bool:
    """True if a camera at (cx, cy) views the board face within max_deg of
    head-on (QR decode needs a fairly frontal view)."""
    return angular_diff(math.atan2(cy - by, cx - bx), normal) \
        <= math.radians(max_deg)


def clearance_dist(grid: Grid) -> np.ndarray:
    """Metres to the nearest non-free cell per cell (static-grid cache)."""
    free = grid.free.astype(np.uint8)
    if not (~grid.free).any():
        return np.full(grid.free.shape, 1e9, dtype=np.float32)
    return (cv2.distanceTransform(free, cv2.DIST_L2, 3)
            * grid.res).astype(np.float32)


def clearance_at(dist: np.ndarray, grid: Grid, x: float, y: float) -> float:
    """Look up a precomputed clearance map (0.0 outside the grid)."""
    r, c = grid.xy_to_cell(float(x), float(y))
    if not grid.in_bounds(r, c):
        return 0.0
    return float(dist[r, c])


# ---------------------------------------------------------------------------
# Goal snapping (final_plan §4.4, §4.8)
# ---------------------------------------------------------------------------
def clearance_mask(grid: Grid, radius_m: float) -> np.ndarray:
    """Free cells at least `radius_m` from any occupied/unknown cell."""
    solid = ~(grid.free)                       # occupied OR unknown blocks
    free = grid.free.astype(np.uint8)
    if solid.any():
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 3) * grid.res
    else:
        dist = np.full(grid.free.shape, 1e9, dtype=np.float32)
    return grid.free & (dist >= radius_m)


def reachable_region(grid: Grid, mask: np.ndarray, seed: Sequence[float],
                     seed_r_m: float = 1.0) -> np.ndarray:
    """`mask` restricted to the connected part containing `seed` (x, y).

    The cleaned map has no unknown cells - the space beyond the walls is
    FREE - but those pockets are unreachable in the arena. Walls seal them
    off, so a connectivity test excludes them without hard-coding geometry.
    """
    if not mask.any():
        return mask
    _nlab, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    r0, c0 = grid.xy_to_cell(float(seed[0]), float(seed[1]))
    steps = int(seed_r_m / grid.res) + 1
    best = None
    for dr in range(-steps, steps + 1):
        for dc in range(-steps, steps + 1):
            r, c = r0 + dr, c0 + dc
            if grid.in_bounds(r, c) and mask[r, c]:
                d = dr * dr + dc * dc
                if best is None or d < best[0]:
                    best = (d, r, c)
    if best is None:                        # seed far from any valid cell
        return mask
    return mask & (lab == lab[best[1], best[2]])


def snap_to_free(grid: Optional[Grid], x: float, y: float,
                 max_r: float = 1.5,
                 clearance_m: float = 0.0,
                 seed: Optional[Sequence[float]] = None
                 ) -> Optional[Tuple[float, float]]:
    """Nearest truly-free (and optionally clearance-checked) cell to (x, y).

    Returns None if nothing free within max_r; grid None -> identity.
    With `seed` (usually the robot) only the seed's connected region is
    considered, so goals never land in a walled-off pocket.
    """
    if grid is None:
        return (x, y)
    mask = clearance_mask(grid, clearance_m) if clearance_m > 0.0 else grid.free
    if seed is not None:
        mask = reachable_region(grid, mask, seed)
    r0, c0 = grid.xy_to_cell(x, y)
    if grid.in_bounds(r0, c0) and mask[r0, c0]:
        return grid.cell_to_xy(r0, c0)
    steps = int(max_r / grid.res) + 1
    best, best_d = None, None
    r_max, c_max = steps, steps
    for dr in range(-r_max, r_max + 1):
        for dc in range(-c_max, c_max + 1):
            r, c = r0 + dr, c0 + dc
            if not grid.in_bounds(r, c) or not mask[r, c]:
                continue
            d = math.hypot(r - r0, c - c0)
            if d > steps:
                continue
            if best_d is None or d < best_d:
                best, best_d = (r, c), d
    if best is None:
        return None
    return grid.cell_to_xy(*best)


# ---------------------------------------------------------------------------
# Arena tour (final_plan §4.6 L2)
# ---------------------------------------------------------------------------
def order_nearest(points: List[Tuple[float, float, float]],
                  start: Tuple[float, float]) -> List[Tuple[float, float, float]]:
    """Greedy nearest-neighbour ordering from `start`."""
    remaining = list(points)
    out = []
    cur = start
    while remaining:
        i = min(range(len(remaining)),
                key=lambda k: math.hypot(remaining[k][0] - cur[0],
                                         remaining[k][1] - cur[1]))
        out.append(remaining.pop(i))
        cur = (out[-1][0], out[-1][1])
    return out


def build_tour_viewpoints(grid: Grid, n: int = TOUR_N,
                          start: Tuple[float, float] = (0.0, 0.0),
                          clearance_m: float = CLEARANCE_M
                          ) -> List[Tuple[float, float, float]]:
    """Spread free, well-cleared viewpoints for L2 (built from OUR map).

    First point is nearest the start; the rest are farthest-point samples so
    the tour covers the arena; ordered nearest-first from start (§4.6).
    Yaw faces the mean of all candidates (roughly the arena centre).
    """
    mask = clearance_mask(grid, clearance_m)
    mask = reachable_region(grid, mask, start)     # skip walled-off pockets
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError('no free cells with clearance for the tour')
    pts = [grid.cell_to_xy(int(r), int(c))
           for r, c in zip(ys[::4], xs[::4])]          # subsample ~0.2 m
    if len(pts) < n:
        pts = [grid.cell_to_xy(int(r), int(c)) for r, c in zip(ys, xs)]
    n = min(n, len(pts))
    cx = float(np.mean([p[0] for p in pts]))
    cy = float(np.mean([p[1] for p in pts]))
    first = min(pts, key=lambda p: math.hypot(p[0] - start[0], p[1] - start[1]))
    chosen = [first]
    while len(chosen) < n:
        rest = (p for p in pts if p not in chosen)
        nxt = max(rest, key=lambda p: min(math.hypot(p[0] - q[0], p[1] - q[1])
                                          for q in chosen))
        chosen.append(nxt)
    out = [(x, y, math.atan2(cy - y, cx - x)) for x, y in chosen]
    return order_nearest(out, start)


def save_tour(path: str, viewpoints: List[Tuple[float, float, float]],
              source: str = '') -> None:
    doc = {'viewpoints': [{'x': round(x, 3), 'y': round(y, 3),
                           'yaw': round(yaw, 3)} for x, y, yaw in viewpoints]}
    with open(path, 'w') as fh:
        fh.write('# generated by scripts/build_tour.py')
        fh.write(f' from {source}\n' if source else '\n')
        yaml.safe_dump(doc, fh, default_flow_style=False, sort_keys=False)


def load_tour(path: str) -> List[Tuple[float, float, float]]:
    with open(path) as fh:
        doc = yaml.safe_load(fh) or {}
    out = []
    for p in doc.get('viewpoints', []):
        out.append((float(p['x']), float(p['y']), float(p.get('yaw', 0.0))))
    return out
