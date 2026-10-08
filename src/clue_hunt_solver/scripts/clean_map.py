#!/usr/bin/env python3
"""
Phase 1 - clean the raw SLAM map into a reusable arena map.

WHY (final_plan.md, Phase 1):
  * The raw teleop map contains all 7 boards as thin black segments.
    Boards MOVE in hidden worlds -> phantom obstacles and bad AMCL matching.
  * Walls come out as hatched/double lines; the interior has grey unknown blobs.
  * The raw map frame is NOT guaranteed to be world-aligned (the map can come
    back rotated, e.g. ~45 deg, if the robot/odom yaw was offset at SLAM start).
    This script DETECTS that rotation from the raw walls and RECTIFIES:
    everything is drawn on a fresh, world-aligned canvas so that
    AMCL's set_initial_pose (0,0,0) stays valid.

WHAT IT DOES (perception of OUR OWN map only, no Gazebo ground truth):
  1. Read the raw map produced by mapping.launch.py (arena.yaml + arena.pgm).
  2. Measure the raw map's orientation vs the world (Hough line families ->
     the two wall families -> their angles and separations; separations are
     checked against the static arena structure 12.15 m / 9.15 m).
  3. Disambiguate the 180 deg ambiguity by scoring the analytic wall template
     (outer walls are symmetric, the two INNER walls are not) against the raw
     occupancy.
  4. Detect the three pillars FROM THE MAP (connected components + least-
     squares circle fit, gated by "circle beats line" so board segments are
     rejected). Pixel positions are transformed into world coordinates first.
  5. Draw a clean world-aligned canvas:
         outside arena                 -> unknown (205)
         arena interior                -> free    (254)
         wall bands / inner walls      -> occupied (0)
         detected pillar discs r=0.2 m -> occupied (0)
     Wall geometry IS written down (outer rect x -1.5..10.5, y -4.5..4.5,
     wall thickness 0.15 m, two inner walls) because the PS guarantees walls
     are identical in every world; this is map building, not answer
     hard-coding. Pillar positions are never written down - they are detected.
  6. Save arena_raw.* (copy of the input, for the report) and arena.* .

USAGE (commands live in ~/hunt_ws/phase_1.md):
  python3 clean_map.py
  python3 clean_map.py --raw ~/hunt_ws/maps/arena.yaml --out-dir <pkg>/maps
  python3 clean_map.py --pillars "5.5,2.5 5.0,-2.0 2.2,2.8"   # manual fallback

EXIT CODES: 0 = success, 1 = detection failed (nothing written).
"""
import argparse
import math
import os
import shutil
import sys

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Arena static structure (identical in ALL worlds per the PS README).
# Units are metres, map output frame == world frame (rectified by this script).
# ---------------------------------------------------------------------------
INNER = (-1.5, 10.5, -4.5, 4.5)      # xmin, xmax, ymin, ymax (free space)
WALL_T = 0.15                        # wall thickness (from the world SDF)
# Inner walls: (xmin, xmax, ymin, ymax) from practice.sdf wall4 / wall5
INNER_WALLS = (
    (3.425, 3.575, 1.0, 4.5),
    (6.925, 7.075, -4.5, -1.0),
)
PILLAR_R = 0.2                       # pillar cylinder radius (world SDF)
CELL_PAD = 0.025                     # half a cell: guarantee the disc renders

# Sanity values for the OUTER wall centre lines (static arena structure)
SEP_X_EXPECT = 12.15                 # distance between the x = -1.575 / 10.575 walls
SEP_Y_EXPECT = 9.15                  # distance between the y = -4.575 / 4.575 walls
CENTER_WORLD = (4.5, 0.0)            # intersection of the outer centre lines

# Output canvas (world frame), 0.35 m beyond the outer walls
CANVAS = (-2.0, 11.0, -5.0, 5.0)     # x0, x1, y0, y1
RES = 0.05

FREE, OCC, UNK = 254, 0, 205

# ---- detection gates (calibrated on raw practice maps) --------------------
AREA_MIN, AREA_MAX = 10, 200         # component area in cells
R_MIN, R_MAX = 0.12, 0.27            # fitted circle radius (treasure disc is 0.30)
CIRCLE_RESID_MAX = 0.04              # mean |dist - r|
CIRCLE_BEATS_LINE = 0.6              # circle_resid must be < 0.6 * line_resid
HOUGH_MIN_LEN = 15                   # px, wall segments for orientation
HOUGH_SEGS_MIN = 4                   # min segments per wall family
SEP_TOL = 0.8                        # m, sanity tolerance on separations
SCORE_AMBIG = 0.03                   # score delta below this -> warn


# ---------------------------------------------------------------------------
# Map IO / grids
# ---------------------------------------------------------------------------
def load_raw(yaml_path):
    """Return (meta dict, grayscale uint8 image, resolved image path)."""
    meta = {}
    with open(yaml_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            key, _, val = line.partition(':')
            meta[key.strip()] = val.strip()
    image = meta.get('image')
    if not image:
        sys.exit(f'ERROR: no "image:" key in {yaml_path}')
    if not os.path.isabs(image):
        image = os.path.join(os.path.dirname(os.path.abspath(yaml_path)), image)
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f'ERROR: cannot read map image {image}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for key in ('resolution', 'origin'):
        if key not in meta:
            sys.exit(f'ERROR: no "{key}:" key in {yaml_path}')
    return meta, img, image


def raw_grid(meta, shape):
    """Raw map grid info: cell-centre axes in MAP frame metres."""
    ox, oy, _ = (float(v) for v in meta['origin'].strip('[]').split(','))
    res = float(meta['resolution'])
    h, w = shape
    xs = ox + (np.arange(w) + 0.5) * res
    ys = oy + (h - np.arange(h) - 0.5) * res   # row 0 = top = max y
    return {'ox': ox, 'oy': oy, 'res': res, 'h': h, 'w': w, 'xs': xs, 'ys': ys}


def canvas_grid():
    """Fresh world-aligned output canvas (X, Y meshes).

    Row order follows the ROS map convention: row 0 = TOP = maximum y
    (cv2.imwrite writes array row 0 at the top of the image, and map_server
    interprets the top row as max y).
    """
    x0, x1, y0, y1 = CANVAS
    xs = x0 + (np.arange(int(round((x1 - x0) / RES))) + 0.5) * RES
    ys = y1 - (np.arange(int(round((y1 - y0) / RES))) + 0.5) * RES
    return np.meshgrid(xs, ys)


def world_masks(X, Y):
    """Arena predicates on world-coordinate arrays."""
    xmin, xmax, ymin, ymax = INNER
    t = WALL_T
    inside_i = (X >= xmin) & (X <= xmax) & (Y >= ymin) & (Y <= ymax)
    inside_o = (X >= xmin - t) & (X <= xmax + t) & (Y >= ymin - t) & (Y <= ymax + t)
    band = inside_o & ~inside_i
    walls = band.copy()
    for wx0, wx1, wy0, wy1 in INNER_WALLS:
        walls |= (X >= wx0) & (X <= wx1) & (Y >= wy0) & (Y <= wy1)
    return inside_i, walls


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def fit_circle(pts):
    """Kasa least-squares circle fit. Returns (cx, cy, r, mean_residual) or None."""
    if len(pts) < 8:
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


def line_residual(pts):
    """Mean perpendicular distance to the best-fit line (PCA normal)."""
    if len(pts) < 3:
        return float('inf')
    q = pts - pts.mean(axis=0)
    cov = np.cov(q.T)
    if not np.all(np.isfinite(cov)):
        return float('inf')
    _, vec = np.linalg.eigh(cov)
    normal = vec[:, 0]              # min-variance direction = line normal
    return float(np.mean(np.abs(q @ normal)))


def rot(theta_deg):
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])


# ---------------------------------------------------------------------------
# Step 1: orientation of the raw map vs the world
# ---------------------------------------------------------------------------
def _family_segs(segs, angle_center, tol=6.0):
    return [s for s in segs if abs((s[0] - angle_center + 90) % 180 - 90) <= tol]


def _family_midpoints(group, normal, g):
    """Centre of a wall family's two lines, along `normal`, in MAP frame metres."""
    proj = []
    for s in group:
        for (px, py) in ((s[4][0], s[4][1]), (s[4][2], s[4][3])):
            wx = g['ox'] + (px + 0.5) * g['res']
            wy = g['oy'] + (g['h'] - py - 0.5) * g['res']
            proj.append(wx * normal[0] + wy * normal[1])
    proj = np.asarray(proj)
    return 0.5 * (proj.min() + proj.max())


def detect_arena_pose(raw, g):
    """Return (theta_deg, t(2,), report dict): p_map = R(theta) p_world + t."""
    occ_u = (raw < 100).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(occ_u, 1, np.pi / 360, threshold=30,
                            minLineLength=HOUGH_MIN_LEN, maxLineGap=3)
    if lines is None:
        sys.exit('ERROR: no wall segments found in the raw map - re-map the arena.')
    segs = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = x2 - x1, y2 - y1
        ang = math.degrees(math.atan2(dy, dx)) % 180.0
        segs.append((ang, math.hypot(dx, dy), (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                     (x1, y1, x2, y2)))
    hist = np.zeros(180)
    for s in segs:
        hist[int(s[0]) % 180] += 1
    peak = int(np.argmax(hist))
    g1 = _family_segs(segs, peak)
    g2 = _family_segs(segs, (peak + 90) % 180)
    if len(g1) < HOUGH_SEGS_MIN or len(g2) < HOUGH_SEGS_MIN:
        sys.exit(f'ERROR: wall families not clearly visible '
                 f'(G1={len(g1)}, G2={len(g2)}) - re-map the arena, drive slower.')

    def fam(group):
        # circular mean of line angles (mod 180) - safe across the 0/180 wrap
        angs = np.radians([s[0] for s in group])
        phi = math.degrees(0.5 * math.atan2(np.mean(np.sin(2 * angs)),
                                            np.mean(np.cos(2 * angs)))) % 180.0
        n = (-math.sin(math.radians(phi)), math.cos(math.radians(phi)))
        proj = []
        for s in group:
            for (px, py) in ((s[4][0], s[4][1]), (s[4][2], s[4][3])):
                wx = g['ox'] + (px + 0.5) * g['res']
                wy = g['oy'] + (g['h'] - py - 0.5) * g['res']
                proj.append(wx * n[0] + wy * n[1])
        proj = np.asarray(proj)
        return phi, float(proj.max() - proj.min())

    phi1, sep1 = fam(g1)
    phi2, sep2 = fam(g2)

    # X-family = the pair of walls separated by ~12.15 m (world x walls)
    if abs(sep1 - SEP_X_EXPECT) <= abs(sep2 - SEP_X_EXPECT):
        xfam, yfam = g1, g2
        phi_x, sep_x, sep_y = phi1, sep1, sep2
    else:
        xfam, yfam = g2, g1
        phi_x, sep_x, sep_y = phi2, sep2, sep1

    warnings = []
    if abs(sep_x - SEP_X_EXPECT) > SEP_TOL or abs(sep_y - SEP_Y_EXPECT) > SEP_TOL:
        warnings.append(f'wall separations {sep_x:.2f}/{sep_y:.2f} m differ from the '
                        f'expected {SEP_X_EXPECT}/{SEP_Y_EXPECT} m - the raw map may be '
                        f'warped (driving too fast?). Consider re-mapping.')

    # 180-deg disambiguation by template agreement (inner walls are asymmetric)
    occ_dil = cv2.dilate(occ_u, np.ones((3, 3), np.uint8)) > 0
    Xt, Yt = canvas_grid()
    _, walls_tmpl = world_masks(Xt, Yt)
    tmpl = np.column_stack([Xt[walls_tmpl], Yt[walls_tmpl]])

    def score(theta):
        R = rot(theta)
        ux = R @ np.array([1.0, 0.0])          # X-wall normal in map
        uy = R @ np.array([0.0, 1.0])          # Y-wall normal in map
        cx = _family_midpoints(xfam, ux, g)    # X-family measured along ux
        cy = _family_midpoints(yfam, uy, g)    # Y-family measured along uy
        C = cx * ux + cy * uy                  # = R @ (cx, cy)
        t = C - R @ np.array(CENTER_WORLD)
        pm = (R @ tmpl.T).T + t
        cols = np.round((pm[:, 0] - g['ox']) / g['res'] - 0.5).astype(int)
        rows = np.round(g['h'] - 0.5 - (pm[:, 1] - g['oy']) / g['res']).astype(int)
        ok = (cols >= 0) & (cols < g['w']) & (rows >= 0) & (rows < g['h'])
        hits = float(occ_dil[rows[ok], cols[ok]].sum())
        return hits / len(tmpl), t

    # The X-family (world direction 90 deg) sits at phi_x = 90 + theta (mod 180),
    # hence theta candidates are phi_x +/- 90; the two differ by 180 deg and are
    # disambiguated by the asymmetric inner-wall template below.
    s0, t0 = score(phi_x + 90.0)
    s1, t1 = score(phi_x + 270.0)
    if s0 >= s1:
        theta, t, s_win, s_lose = phi_x + 90.0, t0, s0, s1
    else:
        theta, t, s_win, s_lose = phi_x + 270.0, t1, s1, s0
    if s_win - s_lose < SCORE_AMBIG:
        warnings.append(f'template scores too close ({s0:.1%} vs {s1:.1%}) - '
                        f'orientation may be wrong.')

    report = {
        'theta': theta % 360.0, 'sep_x': sep_x, 'sep_y': sep_y,
        'n_x': len(xfam), 'n_y': len(yfam),
        'score_a': s0, 'score_b': s1, 't': t, 'warnings': warnings,
    }
    return theta, t, report


# ---------------------------------------------------------------------------
# Step 2: pillar detection (world coordinates)
# ---------------------------------------------------------------------------
def center_in_free_world(x, y, pad=CELL_PAD):
    """True if (x, y) is in arena free space (world frame): inside inner rect,
    not inside an inner wall band."""
    xmin, xmax, ymin, ymax = INNER
    if not (xmin + pad <= x <= xmax - pad and ymin + pad <= y <= ymax - pad):
        return False
    for wx0, wx1, wy0, wy1 in INNER_WALLS:
        if wx0 - pad <= x <= wx1 + pad and wy0 - pad <= y <= wy1 + pad:
            return False
    return True


def detect_pillars(raw, g, theta, t):
    """Detect pillar rings; returns fits [(cx, cy, r, resid, area)] in world.

    Component pixels are given in raw-map metres; they are rectified to world
    coordinates with p_world = R(theta)^T (p_map - t) before gating.
    """
    keep = raw < 100
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        keep.astype(np.uint8), connectivity=8)
    R = rot(theta)

    fits, rejected = [], []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (AREA_MIN <= area <= AREA_MAX):
            rejected.append((area, 'area gate'))
            continue
        rows, cols = np.nonzero(labels == i)
        pm = np.column_stack([g['ox'] + (cols + 0.5) * g['res'],
                              g['oy'] + (g['h'] - rows - 0.5) * g['res']])
        pw = (R.T @ (pm - t).T).T              # -> world frame
        fit = fit_circle(pw)
        if fit is None:
            rejected.append((area, 'circle fit failed'))
            continue
        cx, cy, r, resid = fit
        if not (R_MIN <= r <= R_MAX):
            rejected.append((area, f'radius {r:.2f} m outside [{R_MIN}, {R_MAX}]'))
            continue
        if resid > CIRCLE_RESID_MAX:
            rejected.append((area, f'circle residual {resid:.3f} > {CIRCLE_RESID_MAX}'))
            continue
        if resid >= CIRCLE_BEATS_LINE * line_residual(pw):
            rejected.append((area, 'fits a straight line better (board segment)'))
            continue
        if not center_in_free_world(cx, cy):
            rejected.append((area, f'center ({cx:.2f},{cy:.2f}) not in arena free space'))
            continue
        fits.append((cx, cy, r, resid, area))

    print(f'  components: {n - 1} kept-candidates, {len(rejected)} rejected, '
          f'{len(fits)} passed the pillar gates')
    for area, why in rejected:
        if AREA_MIN <= area <= AREA_MAX:
            print(f'    - rejected area={area:4d}: {why}')
    return fits


# ---------------------------------------------------------------------------
# Step 3: build the clean world-aligned canvas
# ---------------------------------------------------------------------------
def build_clean(pillars):
    X, Y = canvas_grid()
    inside_i, walls = world_masks(X, Y)
    out = np.full(X.shape, UNK, np.uint8)
    out[inside_i] = FREE
    out[walls] = OCC
    for cx, cy in pillars:
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        out[d2 <= (PILLAR_R + CELL_PAD) ** 2] = OCC
    return out


def parse_pillars_arg(text):
    """Parse the manual override format: "x1,y1 x2,y2 x3,y3"."""
    out = []
    for pair in text.split():
        x, y = pair.split(',')
        out.append((float(x), float(y)))
    return out


def dump_yaml(meta, image_name, origin):
    lines = [
        f'image: {image_name}',
        f'mode: {meta.get("mode", "trinary")}',
        f'resolution: {RES}',
        f'origin: [{origin[0]}, {origin[1]}, 0.0]',
        f'negate: {meta.get("negate", "0")}',
        f'occupied_thresh: {meta.get("occupied_thresh", "0.65")}',
        f'free_thresh: {meta.get("free_thresh", "0.25")}',
    ]
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------------------
def main():
    here = os.path.dirname(os.path.abspath(__file__))
    pkg_root = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default=os.path.join(pkg_root, '..', '..', 'maps', 'arena.yaml'),
                    help='raw map yaml produced by mapping.launch.py')
    ap.add_argument('--out-dir', default=os.path.join(pkg_root, 'maps'),
                    help='output directory (default: <pkg>/maps)')
    ap.add_argument('--pillars', default=None,
                    help='manual fallback: "x1,y1 x2,y2 x3,y3" (world coords, confirm first)')
    args = ap.parse_args()

    raw_yaml = os.path.abspath(os.path.expanduser(args.raw))
    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    if not os.path.isfile(raw_yaml):
        sys.exit(f'ERROR: raw map not found: {raw_yaml}')

    print(f'raw map : {raw_yaml}')
    meta, img, raw_img = load_raw(raw_yaml)
    g = raw_grid(meta, img.shape)
    occ = img < 100
    print(f'  size {g["w"]}x{g["h"]}  origin=({g["ox"]}, {g["oy"]})  '
          f'occupied={int(occ.sum())} '
          f'unknown={int(((img > 100) & (img < 250)).sum())} free={int((img >= 250).sum())}')

    print('orientation: measuring raw map vs world frame...')
    theta, t, rep = detect_arena_pose(img, g)
    print(f'  raw map is rotated by {rep["theta"]:.1f} deg -> rectifying to world frame')
    print(f'  wall families: {rep["n_x"]}+{rep["n_y"]} segs, '
          f'separations {rep["sep_x"]:.2f} m / {rep["sep_y"]:.2f} m '
          f'(expected {SEP_X_EXPECT} / {SEP_Y_EXPECT})')
    print(f'  template agreement: {rep["score_a"]:.1%} vs {rep["score_b"]:.1%} '
          f'(180-deg disambiguation)')
    for w in rep['warnings']:
        print(f'  WARNING: {w}')

    if args.pillars:
        pillars = parse_pillars_arg(args.pillars)
        if len(pillars) != 3:
            sys.exit('ERROR: --pillars needs exactly three "x,y" pairs')
        print(f'pillars : manual override -> {pillars}')
    else:
        print('pillars : detecting from the raw map (world coords, circle fit)...')
        fits = detect_pillars(img, g, theta, t)
        if len(fits) > 3:
            print(f'  WARNING: {len(fits)} candidates passed the gates; keeping '
                  f'the 3 best (lowest residual, radius closest to {PILLAR_R} m)')
            fits.sort(key=lambda f: (f[3], abs(f[2] - PILLAR_R)))
            fits = fits[:3]
        if len(fits) < 3:
            sys.exit('ERROR: only ' + str(len(fits)) + ' pillar(s) detected (expected 3). '
                     'Found: ' + ', '.join(f'({f[0]:.2f},{f[1]:.2f})' for f in fits) +
                     '\nRe-map the arena, or pass --pillars "x1,y1 x2,y2 x3,y3" '
                     'after confirming the positions.')
        fits.sort(key=lambda f: (f[1], f[0]))
        pillars = [(f[0], f[1]) for f in fits]
        for cx, cy, r, resid, area in fits:
            print(f'    world=({cx:6.2f}, {cy:6.2f})  r={r:.2f} m  '
                  f'resid={resid:.3f} m  area={area} px')

    clean = build_clean(pillars)

    os.makedirs(out_dir, exist_ok=True)
    # keep the raw map next to the cleaned one (report evidence)
    shutil.copy2(raw_img, os.path.join(out_dir, 'arena_raw.pgm'))
    with open(os.path.join(out_dir, 'arena_raw.yaml'), 'w') as fh:
        fh.write(dump_yaml(meta, 'arena_raw.pgm', (g['ox'], g['oy'])))
    out_pgm = os.path.join(out_dir, 'arena.pgm')
    if not cv2.imwrite(out_pgm, clean):
        sys.exit(f'ERROR: failed to write {out_pgm}')
    with open(os.path.join(out_dir, 'arena.yaml'), 'w') as fh:
        fh.write(dump_yaml(meta, 'arena.pgm', (CANVAS[0], CANVAS[2])))

    print(f'  cleaned: occupied={int((clean == OCC).sum())} '
          f'unknown={int((clean == UNK).sum())} free={int((clean == FREE).sum())}')
    print(f'written : {out_pgm}')
    print(f'          {os.path.join(out_dir, "arena.yaml")}  '
          f'(origin {CANVAS[0]}, {CANVAS[2]} - world aligned)')
    print(f'          {os.path.join(out_dir, "arena_raw.pgm")} (+ .yaml)')


if __name__ == '__main__':
    main()
