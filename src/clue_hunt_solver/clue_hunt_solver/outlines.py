"""Board-outline memory from live lidar (never hard-coded positions).

Boards are static *within* a run but move across worlds, so their positions
must be mapped live. A board panel is a 0.64 x 0.37 m box from the ground up;
the lidar plane (0.135 m) cuts it, so boards appear in /scan as short FLAT
segments - unlike walls (long), pillars (curved, and known from our map) and
the treasure disc (curved). Camera detections then supply facing + id + token.

An outline pins the board to ~5 cm (lidar), so approach poses stop depending
on noisy single-frame PnP anchors - the root cause of the duplicate-cluster
bounce (anchors scattered >0.6 m, one approach burned per duplicate).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .search import Grid

BOARD_MIN_LEN = 0.15             # m; edge-on boards are thinner (seen flat later)
BOARD_MAX_LEN = 0.85            # m; face-on panel is 0.64
WALL_MIN_LEN = 1.2              # longer segments are walls, not boards
FLAT_RESID_MAX = 0.035          # mean |perp distance| to the PCA line
PILLAR_MASK_R = 0.50            # outline centres nearer a known pillar are pillars
MAP_STANDOFF_CELLS = 1          # centre neighbourhood must be map-free (boards stand alone)
CLUSTER_GAP = 0.30              # m between consecutive endpoints: new cluster
RANGE_JUMP = 0.40               # m range jump between adjacent beams: new cluster
MIN_BEAMS = 4
MAX_SCAN_RANGE = 8.0            # ignore far returns for outlines
REAR_BLIND_R = 2.0              # ignore close-behind clusters (own trailer later)
REAR_BLIND_DEG = 120.0          # |bearing - yaw| beyond this is "behind"
ASSOC_DIST = 0.45               # lidar hyp <-> outline association gate
ASSOC_AXIS_DEG = 25.0
ASSOC_CLOSE_DIST = 0.35         # ...but below this, lidar decides alone: two
                                # observations 0.35 m apart are one object even
                                # when a partial view garbles the axis
CAM_ASSOC_DIST = 1.0            # noisy PnP <-> outline association gate
CAM_ASSOC_SAME_ID = 1.5         # ...widened when the ids already agree
FRESH_WINDOW_S = 1.5            # sighting freshness for arrival gates
FRESH_RADIUS_M = 1.5
CONFIRM_HITS = 2                # supporting scans before an outline is actionable
VIEW_DIST = 1.5                 # viewing-pose distance (matches vision measurement)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def mean_yaw(ys: Sequence[float]) -> Optional[float]:
    """Circular mean (None if empty)."""
    ys = list(ys)
    if not ys:
        return None
    s = sum(math.sin(a) for a in ys) / len(ys)
    c = sum(math.cos(a) for a in ys) / len(ys)
    if not (s or c):
        return None
    return math.atan2(s, c)


def confident_yaw(ys: Sequence[float], min_n: int = 3,
                  min_r: float = 0.906) -> Optional[float]:
    """Circular mean only if >= min_n samples agree (~25 deg spread)."""
    ys = list(ys)[-20:]
    if len(ys) < min_n:
        return None
    s = sum(math.sin(a) for a in ys) / len(ys)
    c = sum(math.cos(a) for a in ys) / len(ys)
    if math.hypot(s, c) < min_r:
        return None
    return math.atan2(s, c)


@dataclass
class ScanHyp:
    """One board-like segment from a single scan."""
    x: float
    y: float                     # centre, map frame
    axis: float                  # panel direction yaw (sign-ambiguous)
    length: float
    kind: str                    # 'board' | 'pillar' | 'wall' | 'other'
    n: int = 0                   # supporting beams


@dataclass
class Outline:
    """One physical board candidate, refined over scans."""
    oid: int
    x: float
    y: float
    axis: Optional[float]        # None until a lidar hyp supplies it
    length: float = 0.0
    facing: int = 0              # 0 unknown, +1/-1 side of normal_vec()
    tried: int = 0               # bitmask of attempted sides (1: +1, 2: -1)
    hits: int = 0
    source: str = 'lidar'        # 'lidar' | 'pnp'
    last_seen: float = 0.0
    marker_seen: float = -1e9          # last CAMERA marker sighting here
    observers: List[Tuple[float, float]] = field(default_factory=list)
    n_yaws: List[float] = field(default_factory=list)
    board_id: Optional[int] = None     # marker id seen here (camera)
    verdict: str = 'fresh'       # fresh|valid|parked|rejected|unread|exhausted
    parked_id: Optional[int] = None    # chain id this outline is parked for
    cool_until: float = 0.0
    fails: int = 0                     # check attempts burned here

    @property
    def xy(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def confirmed(self) -> bool:
        return self.hits >= CONFIRM_HITS

    def actionable(self, t: float) -> bool:
        return self.confirmed() and t >= self.cool_until \
            and self.verdict in ('fresh', 'parked', 'unread')


def normal_vec(axis: float, side: int) -> Tuple[float, float]:
    """Outward normal for one side of a panel axis (axis sign-ambiguous, so
    side selects the hemisphere)."""
    a = axis + math.pi / 2.0
    if side < 0:
        a += math.pi
    return (math.cos(a), math.sin(a))


def side_of(ox: float, oy: float, o: Outline) -> int:
    """Which side of the outline an observer at (ox, oy) stands on."""
    if o.axis is None:
        return +1
    nx, ny = normal_vec(o.axis, +1)
    return +1 if (ox - o.x) * nx + (oy - o.y) * ny >= 0.0 else -1


def view_pose(o: Outline, side: int,
              dist: float = VIEW_DIST) -> Optional[Tuple[float, float, float]]:
    """Viewing pose on one side: (x, y, yaw facing the board centre)."""
    if o.axis is None:
        return None
    nx, ny = normal_vec(o.axis, side)
    gx, gy = o.x + nx * dist, o.y + ny * dist
    return (gx, gy, math.atan2(o.y - gy, o.x - gx))


def _pca_axis(pts: np.ndarray):
    c = pts.mean(axis=0)
    d = pts - c
    cov = (d.T @ d) / len(pts)
    vals, vecs = np.linalg.eigh(cov)
    ax = vecs[:, int(np.argmax(vals))]
    along = d @ ax
    perp = d @ np.array([-ax[1], ax[0]])
    length = float(np.ptp(along))
    resid = float(np.mean(np.abs(perp)))
    return math.atan2(float(ax[1]), float(ax[0])), length, resid


def extract_hyps(ranges: Sequence[float], angle_min: float, angle_inc: float,
                 robot_xy: Sequence[float], robot_yaw: float,
                 grid: Optional[Grid] = None,
                 pillar_xys: Sequence[Sequence[float]] = ()
                 ) -> List[ScanHyp]:
    """One scan -> board-like segment hypotheses (map frame).

    Clusters consecutive endpoints (0.30 m gap / 0.40 m range jump); classifies
    by extent + flatness + known map geometry. Close-behind clusters are
    skipped (own trailer in later phases lives there).
    """
    rx, ry = float(robot_xy[0]), float(robot_xy[1])
    pts, valid = [], []
    for i, r in enumerate(ranges):
        if not math.isfinite(r) or r <= 0.05 or r > MAX_SCAN_RANGE:
            valid.append(False)
            pts.append((0.0, 0.0))
            continue
        a = angle_min + i * angle_inc
        wx = rx + r * math.cos(robot_yaw + a)
        wy = ry + r * math.sin(robot_yaw + a)
        pts.append((wx, wy))
        valid.append(True)
    # beam-order clustering
    clusters: List[List[Tuple[float, float]]] = []
    cur: List[Tuple[float, float]] = []
    prev = None
    prev_r = 0.0
    for (x, y), ok, r in zip(pts, valid,
                             [rr if math.isfinite(rr) else 0.0 for rr in ranges]):
        if not ok:
            if len(cur) >= MIN_BEAMS:
                clusters.append(cur)
            cur, prev = [], None
            continue
        if prev is not None:
            if math.hypot(x - prev[0], y - prev[1]) > CLUSTER_GAP \
                    or abs(r - prev_r) > RANGE_JUMP:
                if len(cur) >= MIN_BEAMS:
                    clusters.append(cur)
                cur = []
        cur.append((x, y))
        prev, prev_r = (x, y), r
    if len(cur) >= MIN_BEAMS:
        clusters.append(cur)

    out: List[ScanHyp] = []
    for cl in clusters:
        parr = np.array(cl)
        axis, length, resid = _pca_axis(parr)
        cx, cy = float(parr[:, 0].mean()), float(parr[:, 1].mean())
        # own trailer / close-behind blind cone
        bd = math.hypot(cx - rx, cy - ry)
        ba = math.atan2(cy - ry, cx - rx)
        if bd < REAR_BLIND_R and abs(_wrap_pi(ba - robot_yaw)) > \
                math.radians(REAR_BLIND_DEG):
            continue
        kind = 'other'
        if length > WALL_MIN_LEN:
            kind = 'wall'
        elif any(math.hypot(cx - float(p[0]), cy - float(p[1])) < PILLAR_MASK_R
                 for p in pillar_xys):
            kind = 'pillar'
        elif BOARD_MIN_LEN <= length <= BOARD_MAX_LEN \
                and resid <= FLAT_RESID_MAX and _stands_alone(grid, cx, cy):
            kind = 'board'
        out.append(ScanHyp(x=cx, y=cy, axis=axis, length=length, kind=kind,
                           n=len(cl)))
    return out


def _stands_alone(grid: Optional[Grid], x: float, y: float) -> bool:
    """Centre neighbourhood is map-free (boards stand away from walls)."""
    if grid is None:
        return True
    r0, c0 = grid.xy_to_cell(x, y)
    for dr in range(-MAP_STANDOFF_CELLS, MAP_STANDOFF_CELLS + 1):
        for dc in range(-MAP_STANDOFF_CELLS, MAP_STANDOFF_CELLS + 1):
            r, c = r0 + dr, c0 + dc
            if not grid.in_bounds(r, c) or grid.occ[r, c]:
                return False
    return True


def fresh_near(dets, mid: Optional[int], x: float, y: float, t: float,
             window: float = FRESH_WINDOW_S,
             radius: float = FRESH_RADIUS_M) -> bool:
    """Any recent detection of this board near this point?

    dets: iterable of (marker_id, x, y, t). Same-id detections stick to the
    board even when PnP drift re-associated them elsewhere - this is what
    stops duplicate outlines from starving each other (each arrival looked
    'blind' while the sibling got the sightings).
    """
    for did, dx, dy, dt in dets:
        if mid is not None and did != mid:
            continue
        if t - dt <= window and math.hypot(dx - x, dy - y) <= radius:
            return True
    return False


class BoardMemory:
    """Live map of board outlines for the current run (positions vary across
    worlds, so this is built from scratch every run - never persisted)."""

    def __init__(self):
        self.outlines: List[Outline] = []
        self.next_id = 0

    def update_from_scan(self, hyps: Sequence[ScanHyp],
                         robot_xy: Sequence[float], t: float) -> List[Outline]:
        """Fuse board hyps into outlines (nearest within 0.45 m + axis within
        25 deg). Returns the outlines touched this scan."""
        touched = []
        for h in hyps:
            if h.kind != 'board':
                continue
            best, best_d = None, ASSOC_DIST
            for o in self.outlines:
                d = math.hypot(h.x - o.x, h.y - o.y)
                if d > best_d:
                    continue
                if d >= ASSOC_CLOSE_DIST and o.axis is not None:
                    dd = abs((h.axis - o.axis) % math.pi)
                    if min(dd, math.pi - dd) > math.radians(ASSOC_AXIS_DEG):
                        continue
                best, best_d = o, d
            if best is None:
                best = Outline(oid=self.next_id, x=h.x, y=h.y, axis=h.axis,
                               length=h.length, source='lidar', last_seen=t)
                self.next_id += 1
                self.outlines.append(best)
            elif best.axis is None:
                best.axis, best.source = h.axis, 'lidar'   # pnp upgraded
                best.length = max(best.length, h.length)
                w = min(best.hits + 1, 50)
                best.x = (best.x * best.hits + h.x) / w
                best.y = (best.y * best.hits + h.y) / w
            else:
                w = min(best.hits + 1, 50)
                best.x = (best.x * best.hits + h.x) / w
                best.y = (best.y * best.hits + h.y) / w
                # doubled-angle average (axis sign-ambiguous)
                s = math.sin(2.0 * best.axis) * best.hits + math.sin(2.0 * h.axis)
                c = math.cos(2.0 * best.axis) * best.hits + math.cos(2.0 * h.axis)
                best.axis = math.atan2(s, c) / 2.0
                best.length = max(best.length, h.length)
            best.hits = min(best.hits + 1, 50)
            best.last_seen = t
            best.observers.append((float(robot_xy[0]), float(robot_xy[1])))
            if len(best.observers) > 6:
                del best.observers[:-6]
            touched.append(best)
        # prune: keep the map bounded (drop stale, never-actionable outlines)
        if len(self.outlines) > 40:
            self.outlines.sort(key=lambda o: (o.verdict in ('valid', 'rejected'),
                                               o.last_seen))
            del self.outlines[:-40]
        return touched

    def associate(self, x: float, y: float,
                  max_d: float = CAM_ASSOC_DIST) -> Optional[Outline]:
        """Nearest outline to a (noisy PnP) camera position."""
        best, best_d = None, max_d
        for o in self.outlines:
            d = math.hypot(x - o.x, y - o.y)
            if d <= best_d:
                best, best_d = o, d
        return best

    def associate_same_id(self, marker_id: int, x: float, y: float,
                          max_d: float = CAM_ASSOC_SAME_ID
                          ) -> Optional[Outline]:
        """Nearest outline already known as this marker id (wider gate: PnP
        drifts as the viewpoint changes, but ids don't lie - except the
        look-alike pair, which sits 2.2 m apart, safely outside this gate)."""
        best, best_d = None, max_d
        for o in self.outlines:
            if o.board_id != marker_id:
                continue
            d = math.hypot(x - o.x, y - o.y)
            if d <= best_d:
                best, best_d = o, d
        return best

    def add_provisional(self, x: float, y: float, t: float,
                        normal_yaw: Optional[float],
                        observer_xy: Optional[Tuple[float, float]]
                        ) -> Outline:
        """Camera-only board (lidar hasn't confirmed it yet): PnP position,
        facing from the observer side once known."""
        o = Outline(oid=self.next_id, x=x, y=y, axis=None, source='pnp',
                    last_seen=t)
        self.next_id += 1
        if observer_xy is not None:
            o.observers.append((float(observer_xy[0]), float(observer_xy[1])))
        if normal_yaw is not None:
            o.n_yaws.append(normal_yaw)
        self.outlines.append(o)
        return o

    def note_detection(self, o: Outline, observer_xy: Sequence[float],
                       marker_id: int, normal_yaw: Optional[float], t: float
                       ) -> None:
        """A marker was seen at/near this outline: record id + facing."""
        if o.board_id is None:
            o.board_id = marker_id
        o.last_seen = t
        o.marker_seen = t
        o.observers.append((float(observer_xy[0]), float(observer_xy[1])))
        if len(o.observers) > 6:
            del o.observers[:-6]
        if normal_yaw is not None:
            o.n_yaws.append(normal_yaw)
            if len(o.n_yaws) > 12:
                del o.n_yaws[:-12]
        if o.axis is not None:
            o.facing = side_of(float(observer_xy[0]), float(observer_xy[1]), o)

    def clear_cooldowns(self) -> int:
        n = 0
        for o in self.outlines:
            if o.cool_until > 0.0:
                o.cool_until = 0.0
                n += 1
        return n

    def hygiene(self, t: float, provisional_ttl: float = 60.0) -> int:
        """Drop stale unconfirmed provisionals (camera ghosts and far boards
        lidar never confirmed - e.g. detections 15 m outside the arena).
        Lidar-confirmed outlines persist (real geometry). Returns drop count.
        """
        before = len(self.outlines)
        self.outlines = [
            o for o in self.outlines
            if not (o.source == 'pnp' and o.hits == 0
                    and t - o.last_seen > provisional_ttl)]
        return before - len(self.outlines)

    def candidates(self, t: float,
                   board_id: Optional[int] = None) -> List[Outline]:
        """Actionable outlines, optionally filtered by known marker id."""
        out = [o for o in self.outlines if o.actionable(t)]
        if board_id is not None:
            out = [o for o in out
                   if o.board_id is None or o.board_id == board_id]
        return out


def eligible(o: Outline, expected: Optional[int]) -> bool:
    """May this outline become a CHECK task?

    Camera-corroborated outlines (a real marker was seen: board_id known)
    always qualify when the id fits. Pure lidar outlines qualify only when
    they could BE the expected board - pre-anchor the root must be seen by
    the camera first (this kills ghost-chasing: wall corners, the follower,
    anything the camera never confirmed).
    """
    if o.board_id is not None:
        return expected is None or o.board_id == expected
    return expected is not None


def target_key(o: Outline, expected: Optional[int],
               robot_xy: Tuple[float, float]) -> Tuple[int, float]:
    """Order: expected-id boards first, then nearest."""
    d = math.hypot(o.x - robot_xy[0], o.y - robot_xy[1])
    if expected is not None and o.board_id == expected:
        return (0, d)
    return (1, d)


def pre_anchor_key(o: Outline,
                   robot_xy: Tuple[float, float]) -> Tuple[int, float]:
    """Root-hunt order: boards never successfully decoded first, then nearest.

    A parked outline was already decoded and is provably NOT the root (root
    token is fixed); an unread/fresh camera-known outline might be. Without
    this the robot re-reads known decoys nearby forever instead of going back
    to the one board it never managed to decode.
    """
    d = math.hypot(o.x - robot_xy[0], o.y - robot_xy[1])
    decoded = 1 if o.verdict == 'parked' else 0
    return (decoded, d)
