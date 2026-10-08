"""Unit tests for pillars.py (centres from our map, colours, visibility - §4.5)."""
import math
import os
import unittest

import numpy as np

from clue_hunt_solver import pillars, search

PRACTICE_PILLARS = [(5.5, 2.5), (5.0, -2.0), (2.2, 2.8)]   # offline validation only
MAP_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'maps', 'arena.yaml')
TWO_PI = 2.0 * math.pi


def _canvas(res=0.05, ox=-1.0, oy=-3.0, w_m=12.0, h_m=9.0):
    w, h = int(w_m / res), int(h_m / res)
    return np.zeros((h, w), dtype=bool), res, ox, oy


def _paint_disc(occ, res, ox, oy, cx, cy, r):
    h, w = occ.shape
    rr = oy + (h - np.arange(h) - 0.5) * res         # row 0 = top = max y
    cc = ox + (np.arange(w) + 0.5) * res
    occ |= ((rr[:, None] - cy) ** 2 + (cc[None, :] - cx) ** 2) <= r * r


def _paint_wall(occ, res, ox, oy, x0, y0, x1, y1):
    h, w = occ.shape
    rr = oy + (h - np.arange(h) - 0.5) * res
    cc = ox + (np.arange(w) + 0.5) * res
    occ |= ((rr[:, None] >= y0) & (rr[:, None] <= y1)
            & (cc[None, :] >= x0) & (cc[None, :] <= x1))


def _grid(occ, res, ox, oy):
    return search.Grid(occ=occ, free=~occ, ox=ox, oy=oy, res=res)


class TestPillarCentres(unittest.TestCase):
    def test_synthetic_discs_wall_and_treasure(self):
        occ, res, ox, oy = _canvas()
        truth = [(2.0, 1.0), (6.0, -1.5), (-0.4, -2.0)]
        for cx, cy in truth:
            _paint_disc(occ, res, ox, oy, cx, cy, 0.225)
        _paint_wall(occ, res, ox, oy, 8.0, -2.5, 8.15, 1.0)   # long thin wall
        _paint_disc(occ, res, ox, oy, 4.0, 2.5, 0.325)         # treasure-size disc
        found = pillars.pillar_centres(_grid(occ, res, ox, oy), expected_n=0)
        self.assertEqual(len(found), 3, found)
        for cx, cy, r in found:
            nearest = min(truth, key=lambda t: math.hypot(t[0] - cx, t[1] - cy))
            self.assertLess(math.hypot(nearest[0] - cx, nearest[1] - cy), 0.1)
            self.assertLess(abs(r - 0.225), 0.05)

    def test_real_cleaned_map(self):
        if not os.path.isfile(MAP_YAML):
            self.skipTest('maps/arena.yaml not present')
        grid = search.load_grid(MAP_YAML)
        found = pillars.pillar_centres(grid, expected_n=0)
        self.assertEqual(len(found), 3, f'found {found}')
        for cx, cy, r in found:
            nearest = min(PRACTICE_PILLARS,
                          key=lambda t: math.hypot(t[0] - cx, t[1] - cy))
            self.assertLess(math.hypot(nearest[0] - cx, nearest[1] - cy), 0.15,
                            f'({cx:.2f}, {cy:.2f}) far from {nearest}')
            self.assertTrue(0.15 <= r <= 0.30, r)


class TestColours(unittest.TestCase):
    def test_full_assignment(self):
        obs = {0: (1.0, 0.0, 0.0), 1: (0.0, 1.0, 0.0), 2: (0.0, 0.0, 1.0)}
        mapping, gap = pillars.assign_colours(obs)
        self.assertEqual(mapping, {'RED': 0, 'GREEN': 1, 'BLUE': 2})
        self.assertGreaterEqual(gap, 0.15)

    def test_swapped_assignment(self):
        obs = {0: (0.0, 0.1, 0.9), 1: (0.9, 0.1, 0.0)}       # blue, red
        mapping, gap = pillars.assign_colours(obs)
        self.assertEqual(mapping['BLUE'], 0)
        self.assertEqual(mapping['RED'], 1)
        self.assertEqual(mapping['GREEN'], 2)                 # elimination
        self.assertGreaterEqual(gap, 0.15)

    def test_ambiguous_pair_has_small_gap(self):
        # Two red-ish patches: leftover colours are equally far -> tiny gap,
        # hunt_node must refuse (gap >= COLOUR_GAP_MIN).
        obs = {0: (1.0, 0.0, 0.0), 1: (0.9, 0.1, 0.1)}
        mapping, gap = pillars.assign_colours(obs)
        self.assertIsNotNone(mapping)
        self.assertLess(gap, 0.15)

    def test_grey_and_empty(self):
        # Grey patch is far from every reference -> filtered out entirely.
        mapping, gap = pillars.assign_colours({0: (0.34, 0.33, 0.33)})
        self.assertIsNone(mapping)
        # One good patch + one grey -> single observation, gap is unambiguous
        mapping, gap = pillars.assign_colours({0: (1.0, 0.0, 0.0),
                                               1: (0.34, 0.33, 0.33)})
        self.assertIsNotNone(mapping)
        self.assertEqual(mapping['RED'], 0)
        self.assertGreaterEqual(gap, 0.15)

    def test_desaturated_passes_grey_filtered(self):
        # Rendered/sim colours are desaturated vs the axis refs: a reddish
        # patch at ~0.3 from RED must pass the gate, real grey must not.
        redish = (0.75, 0.15, 0.10)
        ref, d = pillars.nearest_ref(redish)
        self.assertEqual(ref, 'RED')
        self.assertLessEqual(d, pillars.CHROMA_MAX_DIST)
        grey = (0.34, 0.33, 0.33)
        _ref, dg = pillars.nearest_ref(grey)
        self.assertGreater(dg, pillars.CHROMA_MAX_DIST)

    def test_assign_custom_ref_map(self):
        # calibrated (desaturated) RED ref: assigns with a confident gap,
        # default axes would leave this ambiguous
        personal = dict(pillars.REF_CHROMA)
        personal['RED'] = (0.61, 0.19, 0.19)
        m, gap = pillars.assign_colours({0: (0.62, 0.18, 0.20),
                                         1: (0.15, 0.70, 0.15)},
                                        ref_map=personal)
        self.assertEqual(m['RED'], 0)
        self.assertEqual(m['GREEN'], 1)
        self.assertGreaterEqual(gap, 0.25)

    def test_chromaticity(self):
        self.assertEqual(pillars.chromaticity((0, 0, 255)), (1.0, 0.0, 0.0))  # BGR
        self.assertEqual(pillars.chromaticity((255, 0, 0)), (0.0, 0.0, 1.0))
        self.assertIsNone(pillars.chromaticity((1, 1, 1)))    # near-black

    def test_sample_patch(self):
        img = np.zeros((40, 40, 3), dtype=np.uint8)
        img[15:25, 15:25] = (0, 0, 255)                       # red block
        self.assertEqual(pillars.sample_patch(img, 20, 20), (1.0, 0.0, 0.0))
        self.assertIsNone(pillars.sample_patch(img, 2, 2))    # inside margin
        self.assertIsNone(pillars.sample_patch(img, -5, 20))
        black = np.zeros((40, 40, 3), dtype=np.uint8)
        self.assertIsNone(pillars.sample_patch(black, 20, 20))


class TestVisibility(unittest.TestCase):
    def test_project_point(self):
        K = np.array([[554.25, 0, 320], [0, 554.25, 240], [0, 0, 1.0]])
        cam_R = np.eye(3)                   # optical axes aligned with map
        out = pillars.project_point((0.11, -0.05, 2.0), (0.0, 0.0, 0.0),
                                    cam_R, K)
        self.assertIsNotNone(out)
        u, v, depth = out
        self.assertAlmostEqual(u, 554.25 * 0.11 / 2.0 + 320, places=4)
        self.assertAlmostEqual(v, 554.25 * -0.05 / 2.0 + 240, places=4)
        self.assertAlmostEqual(depth, 2.0, places=6)
        # behind the camera -> None
        self.assertIsNone(pillars.project_point((0.0, 0.0, -1.0),
                                                (0.0, 0.0, 0.0), cam_R, K))

    def _scan_hits_pillar(self, ranges, angle_min=-math.pi,
                          angle_inc=TWO_PI / 360.0):
        return pillars.pillar_visible(ranges, angle_min, angle_inc,
                                      (0.0, 0.0), 0.0, (2.0, 0.0), 0.2)

    def test_pillar_visible_scan_gate(self):
        # bearing 0 -> beam index 180 for angle_min=-pi, inc=1 deg
        ranges = np.full(360, 1.8)          # every beam at the pillar surface
        self.assertTrue(self._scan_hits_pillar(ranges))
        ranges = np.full(360, 5.0)          # wall beyond, pillar absent
        self.assertFalse(self._scan_hits_pillar(ranges))
        ranges = np.full(360, 5.0)
        ranges[178:183] = 1.0               # occluder well in front
        self.assertFalse(self._scan_hits_pillar(ranges))

    def test_line_of_sight(self):
        occ, res, ox, oy = _canvas()
        _paint_wall(occ, res, ox, oy, 4.0, -3.0, 4.15, 3.0)   # wall at x=4
        g = _grid(occ, res, ox, oy)
        self.assertTrue(pillars.line_of_sight(g, (1.0, 0.0), (3.0, 0.0)))
        self.assertFalse(pillars.line_of_sight(g, (1.0, 0.0), (6.0, 0.0)))
        # a pillar's own disc does not block its own LOS
        _paint_disc(occ, res, ox, oy, 2.0, 1.0, 0.225)
        g2 = _grid(occ, res, ox, oy)
        self.assertTrue(pillars.line_of_sight(g2, (1.0, 1.0), (2.0, 1.0)))

    def test_choose_colour_viewpoints(self):
        occ, res, ox, oy = _canvas()
        _paint_disc(occ, res, ox, oy, 6.0, 1.0, 0.225)
        g = _grid(occ, res, ox, oy)
        cands = [(3.0, 1.0, 0.0), (6.0, -2.0, 1.57), (0.0, -2.0, 0.0)]
        pillar_xys = [(6.0, 1.0), (0.0, 0.0), (1.0, -1.0)]
        out = pillars.choose_colour_viewpoints(cands, pillar_xys, g, need=2)
        self.assertEqual(len(out), 2)
        for x, y, _yaw in out:
            self.assertTrue(all(pillars.line_of_sight(g, (x, y), p)
                                for p in pillar_xys))


if __name__ == '__main__':
    unittest.main()
