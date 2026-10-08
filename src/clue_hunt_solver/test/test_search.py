"""Unit tests for search.py (viewing poses, ring, snapping, tour - §4.2/§4.6)."""
import math
import os
import tempfile
import unittest

import numpy as np

from clue_hunt_solver import search


def _box_grid(w=100, h=80, border=2):
    """Free rectangle with an occupied border (image convention, row0 = top)."""
    occ = np.zeros((h, w), dtype=bool)
    occ[:border, :] = occ[-border:, :] = True
    occ[:, :border] = occ[:, -border:] = True
    return search.Grid(occ=occ, free=~occ, ox=-2.0, oy=-5.0, res=0.05)


class TestViewingPose(unittest.TestCase):
    def test_board1_face_on(self):
        """Board at (1.5, 0) facing spawn: stand at origin, look +x (§2.2)."""
        x, y, yaw = search.viewing_pose((1.5, 0.0), math.pi)
        self.assertAlmostEqual(x, 0.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_distance_and_facing_all_candidates(self):
        board = (6.6, 1.6)
        normal = 2.1
        cands = search.viewing_pose_candidates(board, normal)
        self.assertEqual(len(cands), 7)                 # direct + 3 widens x2
        for x, y, yaw in cands:
            self.assertAlmostEqual(math.hypot(x - board[0], y - board[1]),
                                   search.DEFAULT_VIEW_DISTANCE, places=6)
            want = math.atan2(board[1] - y, board[0] - x)
            self.assertAlmostEqual(yaw, want, places=6)
        # widen offsets up to +/-30 deg, direct pose first
        off = [math.atan2(math.sin(a - normal), math.cos(a - normal))
               for a in (math.atan2(y - board[1], x - board[0])
                         for x, y, _ in cands)]
        self.assertAlmostEqual(off[0], 0.0, places=6)
        self.assertLessEqual(max(abs(o) for o in off),
                             math.radians(search.VIEW_WIDEN_DEG) + 1e-9)


class TestRing(unittest.TestCase):
    def test_ring_viewpoints(self):
        pts = search.ring_viewpoints(4.0, -1.0, radius=2.0, n=6)
        self.assertEqual(len(pts), 6)
        for x, y, yaw in pts:
            self.assertAlmostEqual(math.hypot(x - 4.0, y + 1.0), 2.0, places=6)
            self.assertAlmostEqual(yaw, math.atan2(-1.0 - y, 4.0 - x), places=6)
        # evenly spaced
        angs = sorted(math.atan2(y + 1.0, x - 4.0) for x, y, _ in pts)
        gaps = [(angs[(i + 1) % 6] - angs[i]) % (2 * math.pi) for i in range(6)]
        for g in gaps:
            self.assertAlmostEqual(g, math.pi / 3, places=6)


class TestSnap(unittest.TestCase):
    def test_free_cell_unchanged(self):
        g = _box_grid()
        x, y = g.cell_to_xy(40, 50)
        sx, sy = search.snap_to_free(g, x, y, max_r=1.0)
        self.assertAlmostEqual(sx, x, places=6)
        self.assertAlmostEqual(sy, y, places=6)

    def test_obstacle_cell_moves_to_free(self):
        g = _box_grid()
        x, y = g.cell_to_xy(0, 50)                  # on the top border
        sx, sy = search.snap_to_free(g, x, y, max_r=1.0)
        r, c = g.xy_to_cell(sx, sy)
        self.assertTrue(g.free[r, c])
        self.assertGreater(r, 0)                    # moved off the border

    def test_too_far_returns_none(self):
        g = _box_grid(w=6, h=6, border=4)           # almost everything blocked
        sx, sy = g.cell_to_xy(2, 2)
        self.assertIsNone(search.snap_to_free(g, sx, sy, max_r=0.1))

    def test_none_grid_is_identity(self):
        self.assertEqual(search.snap_to_free(None, 1.0, 2.0), (1.0, 2.0))

    def test_clearance_mask(self):
        g = _box_grid()
        mask = search.clearance_mask(g, 0.20)       # 4 cells at 0.05
        self.assertTrue(mask[40, 50])
        self.assertFalse(mask[3, 50])               # too close to the border
        self.assertFalse(mask[0, 0])


class TestTour(unittest.TestCase):
    def test_build_and_roundtrip(self):
        g = _box_grid()
        pts = search.build_tour_viewpoints(g, n=8, start=(0.0, 0.0))
        self.assertEqual(len(pts), 8)
        mask = search.clearance_mask(g, search.CLEARANCE_M)
        for x, y, yaw in pts:
            r, c = g.xy_to_cell(x, y)
            self.assertTrue(g.in_bounds(r, c) and mask[r, c])
        # nearest-first: first point is the closest candidate to the start
        d0 = math.hypot(pts[0][0] - 0.0, pts[0][1] - 0.0)
        best = min(math.hypot(x - 0.0, y - 0.0) for x, y, _ in pts)
        self.assertAlmostEqual(d0, best, places=6)

        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, 'tour.yaml')
            search.save_tour(path, pts, source='test')
            back = search.load_tour(path)
        self.assertEqual(len(back), len(pts))
        for a, b in zip(pts, back):
            for k in range(3):
                self.assertAlmostEqual(a[k], b[k], places=3)

    def test_order_nearest(self):
        pts = [(5.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        ordered = search.order_nearest(pts, (0.0, 0.0))
        self.assertEqual([p[0] for p in ordered], [1.0, 3.0, 5.0])


class TestReachability(unittest.TestCase):
    """Our map paints the space beyond the walls FREE; goals must stay
    inside the room reachable from the robot (final_plan §4.6/§4.8)."""

    @staticmethod
    def _ring_grid():
        w, h = 160, 140
        occ = np.zeros((h, w), dtype=bool)
        occ[20:121, 20] = occ[20:121, 140] = True      # side walls
        occ[20, 20:141] = occ[120, 20:141] = True      # top/bottom walls
        return search.Grid(occ=occ, free=~occ, ox=-4.0, oy=-7.0, res=0.05)

    def test_snap_stays_in_seed_region(self):
        g = self._ring_grid()
        inside = g.cell_to_xy(70, 80)
        outside = g.cell_to_xy(5, 5)
        sx, sy = search.snap_to_free(g, *outside, max_r=1.5,
                                     clearance_m=0.0, seed=inside)
        r, c = g.xy_to_cell(sx, sy)
        self.assertTrue(21 <= r <= 119 and 21 <= c <= 139, (r, c))
        # without a seed the sealed-off pocket is (wrongly) acceptable
        sx2, sy2 = search.snap_to_free(g, *outside, max_r=1.5)
        r2, c2 = g.xy_to_cell(sx2, sy2)
        self.assertFalse(21 <= r2 <= 119 and 21 <= c2 <= 139, (r2, c2))

    def test_tour_stays_inside_room(self):
        g = self._ring_grid()
        start = g.cell_to_xy(70, 80)
        pts = search.build_tour_viewpoints(g, n=6, start=start)
        self.assertEqual(len(pts), 6)
        for x, y, _yaw in pts:
            r, c = g.xy_to_cell(x, y)
            self.assertTrue(21 <= r <= 119 and 21 <= c <= 139, (r, c))


class TestGrid(unittest.TestCase):
    def test_cell_xy_roundtrip(self):
        g = _box_grid()
        for r, c in ((5, 7), (79, 99), (40, 50)):
            x, y = g.cell_to_xy(r, c)
            r2, c2 = g.xy_to_cell(x, y)
            self.assertEqual((r2, c2), (r, c))

    def test_from_occupancy_flips(self):
        # ROS layout: row 0 = bottom. Occupied in the TOP row of the image
        # must come from the LAST row of the payload.
        w, h = 4, 3
        data = [0] * (w * h)
        for c in range(w):
            data[(h - 1) * w + c] = 100             # last payload row = image top
        g = search.grid_from_occupancy(data, w, h, 0.0, 0.0, 0.05)
        self.assertTrue(g.occ[0, :].all())          # image top row occupied
        self.assertTrue(g.free[2, :].all())


class TestPoseSeesPoint(unittest.TestCase):
    def test_face_on_in_range(self):
        self.assertTrue(search.pose_sees_point(0.0, 0.0, 0.0, 1.5, 0.0))

    def test_too_far_too_close(self):
        self.assertFalse(search.pose_sees_point(0.0, 0.0, 0.0, 3.0, 0.0))
        self.assertFalse(search.pose_sees_point(0.0, 0.0, 0.0, 0.2, 0.0))

    def test_outside_fov(self):
        # 90 deg off-axis with a 60 deg (half 30) frustum
        self.assertFalse(search.pose_sees_point(0.0, 0.0, 0.0, 0.0, 1.5))
        # 20 deg off-axis is inside
        a = math.radians(20.0)
        self.assertTrue(search.pose_sees_point(
            0.0, 0.0, 0.0, 1.5 * math.cos(a), 1.5 * math.sin(a)))


class TestFrontSide(unittest.TestCase):
    """Wrong-side parking (staring at a board's back) must be rejectable."""

    def test_same_side_as_observers(self):
        # board at origin, seen from +x before
        self.assertTrue(search.same_side_as_observers(
            0.0, 0.0, [(1.5, 0.1)], 1.5, 0.0))
        # candidate on the opposite side -> False
        self.assertFalse(search.same_side_as_observers(
            0.0, 0.0, [(1.5, 0.1)], -1.5, 0.0))
        # no observers recorded -> no opinion
        self.assertTrue(search.same_side_as_observers(0.0, 0.0, [], -1.5, 0.0))

    def test_incidence(self):
        # head-on (board normal +x, camera at +x) passes
        self.assertTrue(search.incidence_ok(0.0, 0.0, 0.0, 1.5, 0.0))
        # edge-on fails
        self.assertFalse(search.incidence_ok(0.0, 0.0, 0.0, 0.0, 1.5))
        # behind the board fails
        self.assertFalse(search.incidence_ok(0.0, 0.0, 0.0, -1.5, 0.0))

    def test_angular_diff_wraps(self):
        self.assertAlmostEqual(search.angular_diff(3.1, -3.1), 0.083185307179586, places=6)
        self.assertAlmostEqual(search.angular_diff(0.0, math.pi), math.pi)


class TestClearance(unittest.TestCase):
    def test_clearance_at(self):
        occ = np.zeros((40, 40), dtype=bool)
        occ[20, 20] = True
        g = search.Grid(occ=occ, free=~occ, ox=0.0, oy=0.0, res=0.05)
        dist = search.clearance_dist(g)
        x, y = g.cell_to_xy(20, 25)          # 5 cells from the obstacle
        self.assertAlmostEqual(search.clearance_at(dist, g, x, y), 0.25,
                               delta=0.02)   # chamfer transform ~= euclidean
        x0, y0 = g.cell_to_xy(20, 20)        # on the obstacle
        self.assertAlmostEqual(search.clearance_at(dist, g, x0, y0), 0.0, places=2)
        self.assertEqual(search.clearance_at(dist, g, 999.0, 999.0), 0.0)


if __name__ == '__main__':
    unittest.main()
