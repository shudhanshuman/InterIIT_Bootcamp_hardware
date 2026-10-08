"""Unit tests for outlines.py - lidar board-outline memory (pure, no ROS)."""
import math
import unittest

import numpy as np

from clue_hunt_solver import outlines
from clue_hunt_solver.outlines import BoardMemory


def _scan(panel=None, wall_r=6.0, n=360):
    """Synthetic 360x1deg scan, robot at origin facing +x.

    panel: (cx, cy, nx, ny, half_w) wall segment; beams hitting it first win.
    """
    ranges = np.full(n, wall_r)
    for i in range(n):
        a = -math.pi + i * math.pi / 180.0
        dx, dy = math.cos(a), math.sin(a)
        if panel is not None:
            cx, cy, nx, ny, hw = panel
            denom = dx * nx + dy * ny
            if abs(denom) > 1e-9:
                t = ((cx) * nx + (cy) * ny) / denom
                if t > 0.05:
                    px, py = dx * t, dy * t
                    along = (px - cx) * (-ny) + (py - cy) * nx
                    if abs(along) <= hw and t < ranges[i]:
                        ranges[i] = t
    return ranges, -math.pi, math.pi / 180.0


class TestExtract(unittest.TestCase):
    def test_face_on_panel_is_board(self):
        # 0.64 m panel at (3, 0), normal -x (facing the robot)
        ranges, amin, ainc = _scan(panel=(3.0, 0.0, -1.0, 0.0, 0.32))
        hyps = outlines.extract_hyps(ranges, amin, ainc, (0.0, 0.0), 0.0)
        boards = [h for h in hyps if h.kind == 'board']
        self.assertEqual(len(boards), 1, [(h.kind, h.length) for h in hyps])
        b = boards[0]
        self.assertLess(abs(b.x - 3.0) + abs(b.y), 0.1)
        self.assertLess(abs(b.length - 0.64), 0.1)
        # axis along the panel (+-y); sign ambiguous
        self.assertLess(min(abs((b.axis - math.pi / 2 + math.pi) % math.pi),
                            abs((b.axis + math.pi / 2 + math.pi) % math.pi)), 0.1)

    def test_long_wall_is_wall(self):
        ranges, amin, ainc = _scan()
        hyps = outlines.extract_hyps(ranges, amin, ainc, (0.0, 0.0), 0.0)
        self.assertTrue(hyps)
        self.assertTrue(all(h.kind == 'wall' for h in hyps))

    def test_known_pillar_is_pillar(self):
        ranges, amin, ainc = _scan()
        # carve a disc-like bump: beams near 26.6 deg read short
        for i in range(n := 360):
            a = -math.pi + i * math.pi / 180.0
            d = abs((a - math.atan2(1.0, 2.0) + math.pi) % (2 * math.pi) - math.pi)
            if d < math.radians(5.0):
                ranges[i] = math.hypot(2.0, 1.0) - 0.2 * math.cos(d / math.radians(5.0) * 1.2)
        hyps = outlines.extract_hyps(ranges, amin, ainc, (0.0, 0.0), 0.0,
                                     pillar_xys=[(2.0, 1.0)])
        kinds = [h.kind for h in hyps
                 if math.hypot(h.x - 2.0, h.y - 1.0) < 0.6]
        self.assertIn('pillar', kinds)
        self.assertNotIn('board', kinds)

    def test_curved_disc_is_not_board(self):
        # treasure-like r=0.30 disc at (4, 0): right size, wrong shape
        ranges, amin, ainc = _scan()
        for i in range(360):
            a = -math.pi + i * math.pi / 180.0
            d = abs((a + math.pi) % (2 * math.pi) - math.pi)
            if d < math.asin(0.3 / 4.0):
                ranges[i] = 4.0 * math.cos(d) - math.sqrt(
                    max(0.09 - (4.0 * math.sin(d)) ** 2, 0.0))
        hyps = outlines.extract_hyps(ranges, amin, ainc, (0.0, 0.0), 0.0)
        near = [h for h in hyps if math.hypot(h.x - 4.0, h.y) < 0.8]
        self.assertTrue(near)
        self.assertTrue(all(h.kind != 'board' for h in near), [h.kind for h in near])

    def test_rear_cone_skipped(self):
        # panel directly behind the robot at 1.5 m -> ignored (own trailer zone)
        ranges, amin, ainc = _scan(panel=(-1.5, 0.0, 1.0, 0.0, 0.32))
        hyps = outlines.extract_hyps(ranges, amin, ainc, (0.0, 0.0), 0.0)
        self.assertFalse([h for h in hyps if h.kind == 'board'])


class TestMemory(unittest.TestCase):
    def test_associate_and_refine(self):
        mem = BoardMemory()
        t1 = mem.update_from_scan(
            [outlines.ScanHyp(x=3.0, y=0.02, axis=1.57, length=0.63,
                              kind='board', n=12)], (0.0, 0.0), 1.0)
        t2 = mem.update_from_scan(
            [outlines.ScanHyp(x=3.04, y=-0.02, axis=-1.57, length=0.64,
                              kind='board', n=12)], (0.5, 0.0), 2.0)
        self.assertEqual(len(mem.outlines), 1)   # same board, axis sign-agnostic
        o = mem.outlines[0]
        self.assertEqual(o.hits, 2)
        self.assertTrue(o.confirmed())
        self.assertEqual(t1, t2)
        self.assertLess(abs(o.x - 3.02) + abs(o.y), 0.05)

    def test_separate_boards_stay_separate(self):
        mem = BoardMemory()
        mem.update_from_scan(
            [outlines.ScanHyp(x=3.0, y=0.0, axis=1.57, length=0.64,
                              kind='board', n=12)], (0.0, 0.0), 1.0)
        mem.update_from_scan(
            [outlines.ScanHyp(x=6.3, y=-0.8, axis=1.0, length=0.60,
                              kind='board', n=10)], (4.0, 0.0), 2.0)
        self.assertEqual(len(mem.outlines), 2)

    def test_facing_and_view_pose(self):
        mem = BoardMemory()
        mem.update_from_scan(
            [outlines.ScanHyp(x=3.0, y=0.0, axis=math.pi / 2, length=0.64,
                              kind='board', n=12)], (0.0, 0.0), 1.0)
        o = mem.outlines[0]
        mem.note_detection(o, (0.0, 0.0), marker_id=3, normal_yaw=None, t=1.5)
        self.assertEqual(o.facing, +1)     # observer at -x side
        self.assertEqual(o.board_id, 3)
        vp = outlines.view_pose(o, +1)
        self.assertIsNotNone(vp)
        self.assertAlmostEqual(vp[0], 1.5, places=6)
        self.assertAlmostEqual(vp[1], 0.0, places=6)
        self.assertAlmostEqual(vp[2], 0.0, places=6)

    def test_provisional_upgrade(self):
        mem = BoardMemory()
        o = mem.add_provisional(4.4, 0.3, 1.0, normal_yaw=0.0,
                                observer_xy=(2.0, 0.3))
        self.assertEqual(o.source, 'pnp')
        self.assertIsNone(outlines.view_pose(o, +1))
        mem.update_from_scan(
            [outlines.ScanHyp(x=4.41, y=0.29, axis=0.1, length=0.62,
                              kind='board', n=12)], (2.0, 0.3), 2.0)
        self.assertEqual(len(mem.outlines), 1)   # associated, not duplicated
        self.assertEqual(o.source, 'lidar')
        self.assertIsNotNone(o.axis)

    def test_hygiene_expires_ghosts(self):
        mem = BoardMemory()
        g = mem.add_provisional(0.46, -15.26, 1.0, normal_yaw=None,
                                observer_xy=None)   # far PnP ghost
        mem.update_from_scan(
            [outlines.ScanHyp(x=3.0, y=0.0, axis=1.57, length=0.63,
                              kind='board', n=12)], (0.0, 0.0), 2.0)
        self.assertEqual(mem.hygiene(3.0), 0)        # ghost still young
        self.assertEqual(mem.hygiene(62.0), 1)       # ghost expired...
        self.assertNotIn(g, mem.outlines)
        self.assertEqual(len(mem.outlines), 1)       # ...lidar outline kept


class TestTargeting(unittest.TestCase):
    """Task selection: camera-corroborated or expected-id outlines only."""

    def _outline(self, oid, x, y, board_id=None, verdict='fresh'):
        o = outlines.Outline(oid=oid, x=x, y=y, axis=0.0, hits=3)
        o.board_id = board_id
        o.verdict = verdict
        return o

    def test_pure_lidar_ghost_never_eligible(self):
        g = self._outline(0, 1.0, 0.0)          # never camera-seen
        self.assertFalse(outlines.eligible(g, None))    # root hunt: tour
        self.assertTrue(outlines.eligible(g, 3))        # may BE board 3

    def test_known_wrong_id_never_eligible(self):
        o = self._outline(1, 1.0, 0.0, board_id=7)
        self.assertFalse(outlines.eligible(o, 3))   # post-anchor: skip
        # pre-anchor it could still be the root - eligible, validator decides
        self.assertTrue(outlines.eligible(o, None))

    def test_expected_id_always_eligible(self):
        o = self._outline(2, 9.0, 0.0, board_id=3)
        self.assertTrue(outlines.eligible(o, 3))

    def test_target_key_expected_first_then_nearest(self):
        near_unknown = self._outline(0, 1.0, 0.0)
        far_expected = self._outline(1, 8.0, 0.0, board_id=3)
        self.assertLess(
            outlines.target_key(far_expected, 3, (0.0, 0.0)),
            outlines.target_key(near_unknown, 3, (0.0, 0.0)))

    def test_pre_anchor_key_undecoded_first(self):
        # parked = already decoded and provably not the root (root token is
        # fixed); unread/fresh camera-known boards might be - check them first
        parked = self._outline(0, 1.0, 0.0, board_id=7)
        parked.verdict = 'parked'
        unread = self._outline(1, 5.0, 0.0, board_id=2)
        unread.verdict = 'unread'
        fresh = self._outline(2, 9.0, 0.0, board_id=4)
        order = sorted([parked, unread, fresh],
                       key=lambda o: outlines.pre_anchor_key(o, (0.0, 0.0)))
        self.assertEqual([o.oid for o in order], [1, 2, 0])


if __name__ == '__main__':
    unittest.main()
