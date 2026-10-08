"""Unit tests for vision.py against the actual board model textures.

The textures are offline stand-ins for what the camera renders: same ArUco
pattern, same QR, same relative geometry. Skipped automatically when the
gazebo models are not present (e.g. evaluator checkout without textures).
"""
import os
import unittest

import cv2
import numpy as np

from clue_hunt_solver import vision
from clue_hunt_solver.clue import parse_clue

TEXTURE = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'clue_hunt_gazebo', 'models', 'board_practice_b1', 'meshes', 'board_practice_b1.png'))

# 640x480 / 60 deg HFOV camera model (only used to give PnP plausible numbers)
K = np.array([[554.25, 0.0, 320.0],
              [0.0, 554.25, 240.0],
              [0.0, 0.0, 1.0]], dtype=np.float64)
DIST = np.zeros(5)


def _load_texture():
    if not os.path.isfile(TEXTURE):
        raise unittest.SkipTest(f'texture not found: {TEXTURE}')
    img = cv2.imread(TEXTURE)
    if img is None:
        raise unittest.SkipTest(f'cannot read {TEXTURE}')
    return img


class TestConstants(unittest.TestCase):
    def test_measured_constants(self):
        self.assertAlmostEqual(vision.MARKER_SIZE, 0.24, places=6)
        self.assertEqual(tuple(vision.QR_OFFSET), (0.30, 0.0))
        self.assertAlmostEqual(vision.QR_SIZE, 0.234, places=3)
        self.assertTrue(0.5 <= vision.DEFAULT_VIEW_DISTANCE <= 2.0)
        self.assertIn(49, vision.IGNORE_MARKER_IDS)


class TestSyntheticProjection(unittest.TestCase):
    def test_qr_projects_right_of_marker_centre(self):
        centre = vision.project_points([[0.0, 0.0, 0.0]], (0, 0, 0), (0, 0, 2), K, DIST)[0]
        qr = vision.project_points([(*vision.QR_OFFSET, 0.0)],
                                   (0, 0, 0), (0, 0, 2), K, DIST)[0]
        self.assertAlmostEqual(centre[0], 320.0, places=6)
        self.assertAlmostEqual(centre[1], 240.0, places=6)
        self.assertGreater(qr[0], centre[0] + 50.0)
        self.assertAlmostEqual(qr[1], centre[1], places=6)


class TestSyntheticMarkers(unittest.TestCase):
    def _canvas_with(self, ids):
        canvas = np.full((480, 640, 3), 255, np.uint8)
        x = 60
        for mid in ids:
            side = 180
            marker = cv2.aruco.generateImageMarker(vision.get_dictionary(), mid, side)
            y = 150
            canvas[y:y + side, x:x + side] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            x += side + 60
        return canvas

    def test_follower_tag_49_ignored(self):
        canvas = self._canvas_with([48, 49])
        found = [d.marker_id for d in vision.detect_markers(canvas, K, DIST, max_reproj=8.0)]
        self.assertEqual(found, [48])

    def test_plain_marker_detected(self):
        canvas = self._canvas_with([48])
        dets = vision.detect_markers(canvas, K, DIST, max_reproj=8.0)
        self.assertEqual(len(dets), 1)
        self.assertGreater(dets[0].distance, 0.0)


class TestTextureEndToEnd(unittest.TestCase):
    def test_detect_and_decode_b1(self):
        img = _load_texture()
        dets = vision.detect_markers(img, K, DIST)
        self.assertTrue(dets, 'no marker detected on b1 texture')
        det = dets[0]
        self.assertEqual(det.marker_id, 1)
        self.assertLess(det.reproj_err, vision.REPROJ_MAX_PX)

        text = vision.read_qr(img, det, K, DIST)
        self.assertTrue(text.startswith('HUNT:1:'), f'unexpected QR text: {text!r}')
        clue = parse_clue(text)
        self.assertIsNotNone(clue)
        self.assertEqual((clue.id, clue.token), (1, '7196'))

    def test_analyse_frame_row(self):
        img = _load_texture()
        rows = vision.analyse_frame(img, K, DIST)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['marker_id'], 1)
        self.assertTrue(rows[0]['qr_text'].startswith('HUNT:1:'))

    def test_full_frame_fallback_without_marker(self):
        img = _load_texture()
        dets = vision.detect_markers(img, K, DIST)
        self.assertTrue(dets)
        det = dets[0]
        cx, cy = det.centre_px
        side = float(np.linalg.norm(det.corners[1] - det.corners[0]))
        dx = vision.QR_OFFSET[0] / vision.MARKER_SIZE * side
        half = vision.QR_SIZE / vision.MARKER_SIZE * side / 2 * 1.6
        x0 = int(cx + 0.55 * side)                       # right of the marker
        x1 = min(img.shape[1], int(cx + dx + half))
        y0 = max(0, int(cy - half))
        y1 = min(img.shape[0], int(cy + half))
        crop = img[y0:y1, x0:x1]
        scale = 300.0 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        canvas = np.full((480, 640, 3), 255, np.uint8)
        h, w = crop.shape[:2]
        oy, ox = (480 - h) // 2, (640 - w) // 2
        canvas[oy:oy + h, ox:ox + w] = crop

        rows = vision.analyse_frame(canvas, K, DIST)
        self.assertIsNone(rows[0]['marker_id'])
        self.assertTrue(rows[0]['qr_text'].startswith('HUNT:1:'),
                        f'full-frame fallback failed: {rows[0]["qr_text"]!r}')

    def test_no_cross_board_qr_read(self):
        """Marker visible + foreign QR elsewhere: crop-only, no full-frame read.

        Regression for phase_2 Step 3: at range the crop failed and the
        full-frame fallback returned another board's QR (2264 cross-reads).
        """
        b1 = _load_texture()
        b3_path = TEXTURE.replace('_b1', '_b3')
        if not os.path.isfile(b3_path):
            raise unittest.SkipTest(f'texture not found: {b3_path}')
        b3 = cv2.imread(b3_path)
        if b3 is None:
            raise unittest.SkipTest(f'cannot read {b3_path}')

        dets = vision.detect_markers(b1, K, DIST)
        self.assertEqual([d.marker_id for d in dets], [1])
        det = dets[0]
        cx, cy = det.centre_px
        side = float(np.linalg.norm(det.corners[1] - det.corners[0]))

        d3 = vision.detect_markers(b3, K, DIST)
        self.assertEqual([d.marker_id for d in d3], [3])
        c3x, c3y = d3[0].centre_px
        s3 = float(np.linalg.norm(d3[0].corners[1] - d3[0].corners[0]))
        half = vision.QR_SIZE / vision.MARKER_SIZE * s3 / 2 * 1.6
        fx0, fx1 = int(c3x + 0.55 * s3), min(b3.shape[1], int(c3x + s3 * 2.02))
        fy0, fy1 = max(0, int(c3y - half)), min(b3.shape[0], int(c3y + half))
        foreign = b3[fy0:fy1, fx0:fx1]

        m = int(0.05 * side)
        patch = b1[int(cy - side / 2 - m):int(cy + side / 2 + m),
                   int(cx - side / 2 - m):int(cx + side / 2 + m)]
        scale = 0.7
        patch = cv2.resize(patch, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        fscale = 200.0 / max(foreign.shape[:2])
        foreign = cv2.resize(foreign, None, fx=fscale, fy=fscale, interpolation=cv2.INTER_AREA)

        canvas = np.full((700, 800, 3), 255, np.uint8)
        ph, pw = patch.shape[:2]
        poy, pox = 250 - ph // 2, 200 - pw // 2
        canvas[poy:poy + ph, pox:pox + pw] = patch
        fh, fw = foreign.shape[:2]
        foy, fox = 540 - fh // 2, 640 - fw // 2
        canvas[foy:foy + fh, fox:fox + fw] = foreign

        # foreign QR is present and decodable full-frame...
        full, _ = vision.decode_text(canvas)
        self.assertTrue(full.startswith('HUNT:3:'),
                        f'foreign QR not decodable full-frame: {full!r}')
        # ...but a marker is present, so only its own crop may answer
        dets = vision.detect_markers(canvas, K, DIST)
        self.assertEqual([d.marker_id for d in dets], [1])
        text = vision.read_qr(canvas, dets[0], K, DIST)
        self.assertEqual(text, '', f'cross-board read: {text!r}')


if __name__ == '__main__':
    unittest.main()
