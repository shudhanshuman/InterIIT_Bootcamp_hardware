"""Pillar centres from OUR cleaned map + runtime colour assignment
(final_plan.md §4.5).

Centres come from perception of our own map (connected components + circle
fit, same gates as scripts/clean_map.py) - pillar coordinates never live in
code. Colours are perceived at runtime: project the centre into the camera,
gate with /scan, take a median chromaticity patch, then assign RED/GREEN/BLUE
to the three centres (brute-force Hungarian - only 3! permutations). Practice
world colours may swap (PS), so nothing here assumes a colour order.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .clue import COLOURS
from .search import Grid

PILLAR_R_NOMINAL = 0.2      # m (map redraw radius, clean_map.py)
AREA_MIN, AREA_MAX = 12, 140        # component area in cells (0.05 m cells)
R_MIN, R_MAX = 0.12, 0.27           # fitted radius gate (treasure disc is 0.30)
CIRCLE_RESID_MAX = 0.04             # mean |dist - r|
CIRCLE_BEATS_LINE = 0.6             # circle_resid < 0.6 * line_resid
SAMPLE_HALF = 5                     # patch half-size, px
CHROMA_MAX_DIST = 0.55              # gate: patch must resemble some pillar colour
# (0.40 proved too strict for rendered/sim lighting: zero patches ever passed.
# The assignment gap, not this gate, is the real guard - keep it tight.)

# Chromaticity references: saturated sim colours land near the axis vectors.
REF_CHROMA = {'RED': (1.0, 0.0, 0.0),
              'GREEN': (0.0, 1.0, 0.0),
              'BLUE': (0.0, 0.0, 1.0)}


# ---------------------------------------------------------------------------
# Map -> centres
# ---------------------------------------------------------------------------
def fit_circle(pts: np.ndarray):
    """Kasa least-squares fit -> (cx, cy, r, resid) or None (clean_map.py)."""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim == 2 and pts.shape[0] == 2 and pts.shape[1] != 2:
        pts = pts.T                        # tolerate column-vector layout
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 8:
        return None
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x))])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, c = sol
    r2 = c + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return None
    r = math.sqrt(r2)
    resid = float(np.mean(np.abs(np.hypot(x - cx, y - cy) - r)))
    return float(cx), float(cy), float(r), resid


def line_residual(pts: np.ndarray) -> float:
    """Mean residual of the best axis-aligned line (thin shapes score low)."""
    x, y = pts[:, 0], pts[:, 1]
    if np.ptp(x) >= np.ptp(y):
        k, b = np.polyfit(x, y, 1)
        return float(np.mean(np.abs(y - (k * x + b))))
    k, b = np.polyfit(y, x, 1)
    return float(np.mean(np.abs(x - (k * y + b))))


def pillar_centres(grid: Grid, expected_n: int = 3) -> List[Tuple[float, float, float]]:
    """Circle-fitted pillar centres from the occupied mask -> [(x, y, r)]."""
    import cv2
    mask = grid.occ.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    found = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not AREA_MIN <= area <= AREA_MAX:
            continue
        ys, xs = np.nonzero(labels == i)
        if len(ys) < 12:
            continue
        comp = labels == i
        # Fit the component OUTLINE, not its area: our cleaned map draws
        # filled discs (area fit -> resid ~0.19R), raw SLAM gives thin rings
        # (rim fit works there too, both rims average to the same centre).
        edge = np.zeros_like(comp)
        edge[:-1, :] |= comp[:-1, :] & ~comp[1:, :]
        edge[1:, :] |= comp[1:, :] & ~comp[:-1, :]
        edge[:, :-1] |= comp[:, :-1] & ~comp[:, 1:]
        edge[:, 1:] |= comp[:, 1:] & ~comp[:, :-1]
        eys, exs = np.nonzero(edge)
        if len(eys) < 8:
            continue
        pts = np.array([grid.cell_to_xy(int(r), int(c))
                        for r, c in zip(eys, exs)])      # (N, 2)
        fit = fit_circle(pts)
        if fit is None:
            continue
        cx, cy, r, resid = fit
        if not R_MIN <= r <= R_MAX or resid > CIRCLE_RESID_MAX:
            continue
        if resid > CIRCLE_BEATS_LINE * line_residual(pts):
            continue                               # straight line, not a disc
        found.append((cx, cy, r))
    found.sort()
    if expected_n and len(found) != expected_n:
        import warnings
        warnings.warn(f'pillar_centres: found {len(found)}, expected {expected_n}')
    return found


# ---------------------------------------------------------------------------
# Visibility & projection
# ---------------------------------------------------------------------------
def line_of_sight(grid: Grid, a: Sequence[float], b: Sequence[float],
                  end_clear: float = 0.25) -> bool:
    """True if the straight segment a->b crosses no occupied/unknown cell.

    Cells within `end_clear` of the endpoint are allowed (a pillar centre is
    itself inside its own occupied disc).
    """
    r0, c0 = grid.xy_to_cell(float(a[0]), float(a[1]))
    r1, c1 = grid.xy_to_cell(float(b[0]), float(b[1]))
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    steps = max(dr, dc, 1)
    for k in range(steps + 1):
        r = int(round(r0 + (r1 - r0) * k / steps))
        c = int(round(c0 + (c1 - c0) * k / steps))
        if not grid.in_bounds(r, c):
            return False
        if k == 0:
            continue                                # start cell (our viewpoint)
        if grid.occ[r, c] or not grid.free[r, c]:
            cx, cy = grid.cell_to_xy(r, c)
            if math.hypot(cx - float(b[0]), cy - float(b[1])) <= end_clear:
                continue                            # the target's own disc
            return False
    return True


def project_point(p_map: Sequence[float], cam_pos: Sequence[float],
                  cam_R: np.ndarray, camera_matrix) -> Optional[Tuple[float, float, float]]:
    """Map point -> (u, v, depth) in the optical frame, or None if behind."""
    p = np.asarray(p_map, dtype=np.float64)
    t = np.asarray(cam_pos, dtype=np.float64)
    R = np.asarray(cam_R, dtype=np.float64)          # optical -> map rotation
    pc = R.T @ (p - t)
    if pc[2] <= 0.05:
        return None
    K = np.asarray(camera_matrix, dtype=np.float64)
    u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
    v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
    return float(u), float(v), float(pc[2])


def pillar_visible(ranges: Sequence[float], angle_min: float, angle_inc: float,
                   robot_xy: Sequence[float], robot_yaw: float,
                   pillar_xy: Sequence[float],
                   radius: float = PILLAR_R_NOMINAL, tol: float = 0.5,
                   window_deg: float = 2.0) -> bool:
    """Scan gate: a beam near the pillar bearing must hit its surface (§4.5).

    Assumes the lidar is aligned with the base (yaw offset 0). Occluders read
    much shorter than the expected surface distance; missing pillars let the
    beam travel on to a far wall.
    """
    if not len(ranges):
        return False
    bearing = math.atan2(pillar_xy[1] - robot_xy[1],
                         pillar_xy[0] - robot_xy[0]) - robot_yaw
    expected = math.hypot(pillar_xy[0] - robot_xy[0],
                          pillar_xy[1] - robot_xy[1]) - radius
    win = math.radians(window_deg)
    best = None
    for i, rng in enumerate(ranges):
        if not math.isfinite(rng) or rng <= 0.0:
            continue
        a = angle_min + i * angle_inc
        if abs((a - bearing + math.pi) % (2 * math.pi) - math.pi) <= win:
            if best is None or rng < best:
                best = rng
    if best is None:
        return False
    return abs(best - expected) <= tol


# ---------------------------------------------------------------------------
# Colour patches + assignment
# ---------------------------------------------------------------------------
def chromaticity(bgr: Sequence[float]) -> Optional[Tuple[float, float, float]]:
    """(r, g, b) / (r + g + b) - lighting-invariant colour signature (§4.5)."""
    b, g, r = (float(v) for v in bgr)
    s = r + g + b
    if s < 30.0:                       # near-black patch: not a colour sample
        return None
    return (r / s, g / s, b / s)


def sample_patch(img: np.ndarray, u: float, v: float,
                 half: int = SAMPLE_HALF) -> Optional[Tuple[float, float, float]]:
    """Median chromaticity of a small patch centred at pixel (u, v)."""
    h, w = img.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if not (half <= ui < w - half and half <= vi < h - half):
        return None
    patch = img[vi - half:vi + half + 1, ui - half:ui + half + 1]
    med = np.median(patch.reshape(-1, 3), axis=0)   # BGR
    return chromaticity(med)


def chroma_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


# ---------------------------------------------------------------------------
# Whole-image colour blobs + bearing association (HSV path)
# ---------------------------------------------------------------------------
# Why: the single-patch projector looks at ONE pixel derived from TF+AMCL. A
# 0.2 m AMCL drift or a rim projection lands the 11x11 patch on background ->
# patch=0/chroma=0 even with the pillar big in the image (the log pattern:
# vis=1 proj=3 patch=0). HSV finds every saturated blob first, then the
# robot yaw + xy (relative bearing to the known map centres) decides WHICH
# pillar it is. Runs every frame, cheap (3x inRange on 640x480).
HSV_MIN_AREA = 200                # px, pillar blobs are large (see module doc)
HSV_BEARING_GATE_DEG = 10.0       # observed vs expected pillar bearing


def detect_colour_blobs(img) -> list:
    """Whole-image RED/GREEN/BLUE blobs -> [(colour, u, v, area)].

    HSV (lighting-tolerant hue + S/V gates), 5x5 open, connected components.
    No position prior: association happens via relative bearing in hunt_node.
    """
    import cv2
    if img is None or getattr(img, 'size', 0) == 0:
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 80, 60), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 80, 60), (179, 255, 255))
    masks = {
        'RED': cv2.bitwise_or(red1, red2),
        'GREEN': cv2.inRange(hsv, (35, 70, 60), (85, 255, 255)),
        'BLUE': cv2.inRange(hsv, (90, 70, 60), (130, 255, 255)),
    }
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    out = []
    for colour, m in masks.items():
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
        n, _labels, stats, cent = cv2.connectedComponentsWithStats(m, 8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < HSV_MIN_AREA:
                continue
            u, v = float(cent[i][0]), float(cent[i][1])
            out.append((colour, u, v, area))
    out.sort(key=lambda e: -e[3])
    return out


def bearing_for_pixel(u: float, v: float, camera_matrix,
                      cam_R) -> Optional[float]:
    """Image pixel -> compass yaw of that ray in the map frame."""
    import math as _math
    import numpy as _np
    try:
        K = _np.asarray(camera_matrix, dtype=float)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        ray_cam = _np.array([(float(u) - cx) / fx,
                             (float(v) - cy) / fy, 1.0])
        ray_map = _np.asarray(cam_R, dtype=float) @ ray_cam
        return _math.atan2(float(ray_map[1]), float(ray_map[0]))
    except Exception:                          # noqa: BLE001
        return None


def nearest_ref(chroma: Sequence[float],
                refs: Optional[Dict[str, Sequence[float]]] = None
                ) -> Tuple[Optional[str], float]:
    refs = REF_CHROMA if refs is None else refs
    best, best_d = None, None
    for name, ref in refs.items():
        d = chroma_distance(chroma, ref)
        if best_d is None or d < best_d:
            best, best_d = name, d
    return best, best_d


def assign_colours(observations: Dict[int, Sequence[float]],
                   ref_map: Optional[Dict[str, Sequence[float]]] = None
                   ) -> Tuple[Optional[Dict[str, int]], float]:
    """Assign RED/GREEN/BLUE to observed centre indices.

    Brute-force Hungarian over <= 3! permutations. Returns
    ({colour: centre_index}, gap) where gap = second-best minus best cost
    over DISTINCT assignments of the observed indices (inf for a single
    observation). With two observations the third colour follows by
    elimination (final_plan section 4.5 "two known => third").

    ref_map overrides the reference chroma (calibrated refs from live data);
    defaults to the ideal axis REF_CHROMA.
    """
    from itertools import permutations
    refs = REF_CHROMA if ref_map is None else ref_map
    obs = {idx: tuple(ch) for idx, ch in observations.items()
           if 0 <= idx <= 2
           and chroma_distance(ch, refs[nearest_ref(ch, refs)[0]]) <= CHROMA_MAX_DIST}
    if not obs:
        return None, 0.0
    idxs = sorted(obs)
    # cost per DISTINCT assignment of the observed indices only: permutations
    # that differ only in unused trailing colours are not "second best".
    costs = {}
    for perm in permutations(COLOURS):
        key = perm[:len(idxs)]
        if key in costs:
            continue
        costs[key] = sum(chroma_distance(obs[idx], refs[key[k]])
                         for k, idx in enumerate(idxs))
    best_map = min(costs, key=costs.get)
    ranked = sorted(costs.values())
    gap = float('inf') if len(ranked) < 2 else float(ranked[1] - ranked[0])
    mapping = {best_map[k]: idx for k, idx in enumerate(idxs)}
    leftover_cols = [c for c in COLOURS if c not in mapping]
    for idx in range(3):                 # unseen centre gets the leftover colours
        if idx in mapping.values():
            continue
        if not leftover_cols:
            break
        mapping[leftover_cols.pop(0)] = idx
    return mapping, gap


def choose_colour_viewpoints(candidates: Sequence[Tuple[float, float, float]],
                             pillar_xys: Sequence[Sequence[float]],
                             grid: Grid, need: int = 2
                             ) -> List[Tuple[float, float, float]]:
    """Viewpoints (from the tour / open cells) from which ALL pillars are
    visible by line of sight (§4.5 fallback viewports)."""
    out = []
    for vp in candidates:
        if all(line_of_sight(grid, vp, p) for p in pillar_xys):
            out.append(vp)
            if len(out) >= need:
                break
    return out


def pillar_ring_viewpoints(pillar_xy: Sequence[float], grid: Grid,
                           radius: float = 2.5, n: int = 3
                           ) -> List[Tuple[float, float, float]]:
    """Viewpoints around ONE pillar, each facing it (colour sampling).

    Used by the pillar task: close, facing, LOS-checked poses beat
    far tour viewpoints whose patches are background-dominated.
    """
    px, py = float(pillar_xy[0]), float(pillar_xy[1])
    out = []
    for i in range(n):
        a = 2.0 * math.pi * i / n + math.pi / n
        x, y = px + radius * math.cos(a), py + radius * math.sin(a)
        if line_of_sight(grid, (x, y), (px, py)):
            out.append((x, y, math.atan2(py - y, px - x)))
    return out
