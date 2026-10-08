#!/usr/bin/env python3
"""Build config/search_viewpoints.yaml (L2 arena tour) from OUR cleaned map.

final_plan.md section 4.6: 8-12 well-cleared viewpoints, nearest-first from
the spawn. Run from the workspace root (phase_3.md Step 3), then rebuild so
the yaml is installed into the package share:

  python3 src/clue_hunt_solver/scripts/build_tour.py
  colcon build --symlink-install --packages-select clue_hunt_solver
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from clue_hunt_solver import search          # noqa: E402


def default_map() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = os.path.join(get_package_share_directory('clue_hunt_solver'),
                         'maps', 'arena.yaml')
        if os.path.isfile(p):
            return p
    except Exception:                         # noqa: BLE001
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, '..', 'maps', 'arena.yaml')


def default_out() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, '..', 'config', 'search_viewpoints.yaml')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--map', default=default_map(), help='map yaml')
    ap.add_argument('--out', default=default_out(), help='output yaml')
    ap.add_argument('-n', type=int, default=search.TOUR_N, help='viewpoint count')
    ap.add_argument('--clearance', type=float, default=search.CLEARANCE_M)
    ap.add_argument('--start', nargs=2, type=float, default=[0.0, 0.0],
                    metavar=('X', 'Y'), help='tour ordering origin')
    args = ap.parse_args()

    grid = search.load_grid(args.map)
    pts = search.build_tour_viewpoints(grid, n=args.n,
                                       start=tuple(args.start),
                                       clearance_m=args.clearance)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    search.save_tour(out, pts, source=os.path.basename(args.map))

    print(f'map      : {os.path.abspath(args.map)}')
    print(f'clearance: {args.clearance} m   viewpoints: {len(pts)}')
    print(f'written  : {out}')
    mask = search.clearance_mask(grid, args.clearance)
    for i, (x, y, yaw) in enumerate(pts):
        r, c = grid.xy_to_cell(x, y)
        ok = bool(mask[r, c]) if grid.in_bounds(r, c) else False
        print(f'  {i:2d}: ({x:6.2f}, {y:6.2f}) yaw={yaw:5.2f}  clear={ok}')
        if not ok:
            print('ERROR: viewpoint outside the clearance mask', file=sys.stderr)
            return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
