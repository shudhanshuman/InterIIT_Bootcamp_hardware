"""Unit tests for frames.py - reproduces the final_plan.md §2.2 table exactly
using the practice-world SDF poses (offline fixtures, never code inputs)."""
import math
import unittest

from clue_hunt_solver.frames import (BoardPose, axes, between_target,
                                     board_to_marker, distance, goto_target,
                                     marker_to_board, quaternion_from_yaw,
                                     quaternion_to_matrix, rel_target,
                                     treasure_target, yaw_from_quaternion)

# practice.sdf poses (offline ground truth for tests only)
B2 = BoardPose(1.5, -3.9, math.pi / 2)
B4 = BoardPose(4.4, 0.3, 0.0)
B5 = BoardPose(9.5, -0.5, math.pi)
TREASURE = (8.5, -3.5)
PILLAR_RED = (5.5, 2.5)      # colour->centre mapping is pillars.py (Phase 3)
PILLAR_GREEN = (5.0, -2.0)
PILLAR_BLUE = (2.2, 2.8)


class TestAxes(unittest.TestCase):
    def test_yaw_zero(self):
        self.assertEqual(axes(0.0), ((1.0, 0.0), (0.0, 1.0)))

    def test_yaw_pi(self):
        ex, ey = axes(math.pi)
        self.assertAlmostEqual(ex[0], -1.0)
        self.assertAlmostEqual(ex[1], 0.0, places=9)
        self.assertAlmostEqual(ey[0], 0.0, places=9)
        self.assertAlmostEqual(ey[1], -1.0)

    def test_yaw_half_pi(self):
        ex, ey = axes(math.pi / 2)
        self.assertAlmostEqual(ex[0], 0.0, places=9)
        self.assertAlmostEqual(ex[1], 1.0)
        self.assertAlmostEqual(ey[0], -1.0)
        self.assertAlmostEqual(ey[1], 0.0, places=9)

    def test_axes_orthonormal(self):
        for yaw in (0.0, 0.7, 2.3562, math.pi):
            (ex, ey), (rx, ry) = axes(yaw)
            self.assertAlmostEqual(ex * rx + ey * ry, 0.0, places=9)
            self.assertAlmostEqual(ex * ex + ey * ey, 1.0, places=9)


class TestTable22(unittest.TestCase):
    """§2.2 computed-target column, to the centimetre."""

    def test_goto_b1(self):
        self.assertEqual(goto_target(1.5, -3.9), (1.5, -3.9))   # == b2 pose

    def test_rel_b4(self):
        got = rel_target(B4.xy, B4.yaw, 5.08, -0.65)
        self.assertAlmostEqual(got[0], 9.48, places=6)
        self.assertAlmostEqual(got[1], -0.35, places=6)
        self.assertLess(distance(got, B5.xy), 0.16)              # table: 0.15 m

    def test_treasure_rel_b5(self):
        got = treasure_target(B5.xy, B5.yaw, 0.98, 3.15)
        self.assertAlmostEqual(got[0], 8.52, places=6)
        self.assertAlmostEqual(got[1], -3.65, places=6)
        self.assertLess(distance(got, TREASURE), 0.16)           # table: 0.15 m

    def test_between_blue_green(self):
        got = between_target(PILLAR_BLUE, PILLAR_GREEN, 0.59)
        self.assertAlmostEqual(got[0], 3.85, places=2)
        self.assertAlmostEqual(got[1], -0.03, places=2)

    def test_search_region_covers_next_board(self):
        # §2.2: PILLAR RED target (5.5, 2.5), b3 at (6.6, 1.6) -> 1.42 m < 1.8 m
        self.assertLess(distance(PILLAR_RED, (6.6, 1.6)), 1.8)


class TestAxisPermutation(unittest.TestCase):
    def test_qr_sits_at_board_point(self):
        # marker-frame QR offset (+0.30, 0, 0) -> board point (0, +0.30, 0)
        self.assertEqual(marker_to_board((0.30, 0.0, 0.0)), (0.0, 0.30, 0.0))

    def test_round_trip(self):
        p = (0.1, 0.2, 0.3)
        self.assertEqual(board_to_marker(marker_to_board(p)), p)


class TestYawQuaternion(unittest.TestCase):
    def test_round_trip(self):
        for yaw in (-2.0, -0.3, 0.0, 1.5708, 3.1416, 2.3562):
            x, y, z, w = quaternion_from_yaw(yaw)
            got = yaw_from_quaternion(x, y, z, w)
            # atan2 range is (-pi, pi]; inputs slightly beyond pi wrap by 2*pi
            delta = (got - yaw + math.pi) % (2 * math.pi) - math.pi
            self.assertAlmostEqual(delta, 0.0, places=9)

    def test_quaternion_to_matrix_matches_yaw(self):
        for yaw in (0.0, 1.2, -2.5, 3.1416):
            x, y, z, w = quaternion_from_yaw(yaw)
            R = quaternion_to_matrix(x, y, z, w)
            vx = [sum(R[i][j] * (1.0, 0.0, 0.0)[j] for j in range(3))
                  for i in range(3)]
            self.assertAlmostEqual(vx[0], math.cos(yaw), places=9)
            self.assertAlmostEqual(vx[1], math.sin(yaw), places=9)
            self.assertAlmostEqual(vx[2], 0.0, places=9)


if __name__ == '__main__':
    unittest.main()
