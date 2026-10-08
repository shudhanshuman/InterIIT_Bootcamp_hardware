"""Unit tests for the nav-stack health detector (reject_burst).

Action servers EXIST while Nav2 lifecycle is still activating, so
server_is_ready() cannot see that outage - but a burst of instant
rejections can. See phase_3.md troubleshooting (02:49 run: whole tour
burned in ~1 s on rejects, ladder idle at +9 s).
"""
import unittest

from clue_hunt_solver.hunt_node import (REJECT_TRIP_N, REJECT_WINDOW_S,
                                         reject_burst)


class TestRejectBurst(unittest.TestCase):
    def test_trips_on_three_fast_rejects(self):
        self.assertTrue(reject_burst([10.0, 10.1, 10.2], 10.2))
        self.assertTrue(reject_burst([8.0, 9.5, 9.9, 10.0], 10.0))

    def test_scattered_rejects_do_not_trip(self):
        # healthy runs essentially never reject; singles/pairs must not hold
        self.assertFalse(reject_burst([], 10.0))
        self.assertFalse(reject_burst([10.0], 10.0))
        self.assertFalse(reject_burst([9.0, 10.0], 10.0))
        self.assertFalse(reject_burst([0.0, 5.0, 10.0], 10.0))

    def test_old_rejects_age_out_of_window(self):
        self.assertFalse(reject_burst([0.0, 0.1, 10.0], 10.0))
        self.assertTrue(reject_burst([9.9, 10.0, 10.1], 10.1,
                                     n=3, window=0.5))

    def test_constants_sane(self):
        # window must exceed the approach resend cadence (~1 s) or slow
        # rejects never trip; N=3 keeps single glitches harmless
        self.assertEqual(REJECT_TRIP_N, 3)
        self.assertGreaterEqual(REJECT_WINDOW_S, 2.0)


class TestSelfCalls(unittest.TestCase):
    """Every self._x() call in hunt_node must resolve (a missing method
    crashes the node on first use - py_compile and imports can't see it)."""

    def test_all_private_self_calls_defined(self):
        import ast
        import os
        from rclpy.node import Node
        from clue_hunt_solver import hunt_node as hn
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'clue_hunt_solver', 'hunt_node.py')
        tree = ast.parse(open(path).read())
        cls = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.ClassDef) and n.name == 'HuntNode')
        called = set()
        for n in ast.walk(cls):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and isinstance(n.func.value, ast.Name) \
                    and n.func.value.id == 'self' \
                    and n.func.attr.startswith('_') \
                    and not (n.func.attr.startswith('__')
                             and n.func.attr.endswith('__')):
                called.add(n.func.attr)
        missing = [c for c in sorted(called)
                   if not hasattr(hn.HuntNode, c) and not hasattr(Node, c)]
        self.assertEqual(missing, [], f'undefined self calls: {missing}')

    def test_no_stale_marker_id_attribute(self):
        # Outlines carry board_id (not the old Cluster.marker_id): any other
        # `*.marker_id` access in hunt_node is a crash bug (killed a live run
        # at the first read-adjust). det.marker_id / reading_id are fine.
        import os
        import re
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'clue_hunt_solver', 'hunt_node.py')
        bad = [ln for ln in open(path).read().splitlines()
               if re.search(r'(?<!det\.)(?<!reading_)marker_id', ln)
               and 'marker_id: int' not in ln and 'marker_id,' not in ln]
        self.assertEqual(bad, [], f'stale marker_id uses: {bad}')

    def test_no_undefined_names_pyflakes(self):
        # A NameError (e.g. a module imported by-item but used by-module)
        # kills the node at runtime where no unit test reaches: pyflakes sees
        # undefined names statically. Skipped if pyflakes is unavailable.
        try:
            from pyflakes import api as pf_api
            from pyflakes import reporter as pf_reporter
        except ImportError:
            self.skipTest('pyflakes not installed')
        import glob
        import io
        import os
        pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'clue_hunt_solver')
        bad = []
        for path in sorted(glob.glob(os.path.join(pkg, '*.py'))):
            out, err = io.StringIO(), io.StringIO()
            pf_api.checkPath(path, pf_reporter.Reporter(out, err))
            bad.extend(ln for ln in out.getvalue().splitlines()
                       if 'undefined name' in ln)
        self.assertEqual(bad, [], f'pyflakes undefined names: {bad}')


if __name__ == '__main__':
    unittest.main()
