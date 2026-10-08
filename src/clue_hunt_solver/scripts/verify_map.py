#!/usr/bin/env python3
"""
Phase 1 - numeric acceptance check for the cleaned arena map.

Verifies (all in world/map coordinates, ROS map convention: image row 0 = top
= max y):

  * grid geometry: size 260x200 @ 0.05 m, origin (-2.0, -5.0)
  * outer walls occupied, space beyond them unknown, interior free
  * inner walls present on the correct sides (detects a vertical flip)
  * AMCL initial pose (0, 0) is free
  * --expected-pillars: each given point is occupied (pillar discs drawn)
  * --practice: board sites are free again (boards must be removed)

USAGE (see ~/hunt_ws/phase_1.md, Step 1):
  python3 verify_map.py
  python3 verify_map.py --expected-pillars "5.0,-2.0 5.5,2.5 2.2,2.8" --practice

EXIT CODES: 0 = all checks passed, 1 = at least one failure.
"""
import argparse
import os
import sys

import cv2
import numpy as np

FREE, OCC, UNK = 254, 0, 205
X0, Y0, RES = -2.0, -5.0, 0.05
W, H = 260, 200

GEOMETRY_CHECKS = [
    ("outer wall x=-1.575", -1.575, 0.0, OCC),
    ("outer wall x=+10.575", 10.575, 0.0, OCC),
    ("outer wall y=-4.575", 4.5, -4.575, OCC),
    ("outer wall y=+4.575", 4.5, 4.575, OCC),
    ("outside west unknown", -1.9, 0.0, UNK),
    ("outside north unknown", 4.5, 4.9, UNK),
    ("AMCL init pose free", 0.0, 0.0, FREE),
    ("open area free", 8.0, 3.0, FREE),
    # asymmetric inner walls - these four also catch a vertical flip
    ("inner wall4 top (3.5, 2.0)", 3.5, 2.0, OCC),
    ("wall4 gap below (3.5, -2.0)", 3.5, -2.0, FREE),
    ("inner wall5 bottom (7.0, -3.0)", 7.0, -3.0, OCC),
    ("wall5 gap above (7.0, 3.0)", 7.0, 3.0, FREE),
]

PRACTICE_BOARD_SITES = [
    ("board b2 site (1.5, -3.9)", 1.5, -3.9, FREE),
    ("board x4 site (6.62, -0.83)", 6.62, -0.83, FREE),
    ("treasure site (8.5, -3.5)", 8.5, -3.5, FREE),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_yaml = os.path.join(os.path.dirname(here), 'maps', 'arena.yaml')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--map', default=default_yaml, help='cleaned map yaml to verify')
    ap.add_argument('--expected-pillars', default=None,
                    help='"x,y x,y x,y" world points that must be occupied')
    ap.add_argument('--practice', action='store_true',
                    help='also check that practice-world board sites are free')
    args = ap.parse_args()

    if not os.path.isfile(args.map):
        sys.exit(f'ERROR: map not found: {args.map}')
    try:
        text = open(args.map).read()
    except OSError as exc:
        sys.exit(f'ERROR: cannot read {args.map}: {exc}')
    image_rel = next((ln.split(':', 1)[1].strip() for ln in text.splitlines()
                      if ln.strip().startswith('image:')), None)
    if not image_rel:
        sys.exit(f'ERROR: no "image:" key in {args.map}')
    image_path = image_rel if os.path.isabs(image_rel) else os.path.join(
        os.path.dirname(args.map), image_rel)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        sys.exit(f'ERROR: cannot read {image_path}')
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    checks = list(GEOMETRY_CHECKS)
    if args.expected_pillars:
        for pair in args.expected_pillars.split():
            x, y = pair.split(',')
            checks.append((f'pillar disc ({x}, {y})', float(x), float(y), OCC))
    if args.practice:
        checks += PRACTICE_BOARD_SITES

    fails = 0

    def report(ok, label, expected=None, got=None):
        nonlocal fails
        fails += not ok
        extra = '' if expected is None else f' expected={expected} got={got}'
        print(f'{"PASS" if ok else "FAIL"}  {label}{extra}')

    report(img.shape == (H, W), f'grid size {W}x{H} @ {RES} m',
           f'({H}, {W})', img.shape)

    h, w = img.shape
    for label, x, y, exp in checks:
        col = int(round((x - X0) / RES - 0.5))
        row = int(round(h - 0.5 - (y - Y0) / RES))
        if not (0 <= col < w and 0 <= row < h):
            report(False, label, f'inside grid', f'col={col} row={row}')
            continue
        got = int(img[row, col])
        report(got == exp, label, exp, got)

    report('origin: [-2.0, -5.0, 0.0]' in text, 'yaml origin is world-aligned')
    report('resolution: 0.05' in text, 'yaml resolution 0.05')
    report('image: arena.pgm' in text, 'yaml points at arena.pgm')

    raw_pgm = os.path.join(os.path.dirname(args.map), 'arena_raw.pgm')
    report(os.path.isfile(raw_pgm), 'arena_raw.pgm kept as evidence')

    print(f'\n{"ALL CHECKS PASSED" if fails == 0 else str(fails) + " CHECKS FAILED"}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
