#!/usr/bin/env python3
"""Phase 2 measurement harness (final_plan.md §6): drive away from a board and
log, per distance bin, whether the QR still decodes; also cross-check ArUco id
vs QR clue id. Run instructions live in ~/hunt_ws/phase_2.md (do not guess the
viewing distance - set vision.DEFAULT_VIEW_DISTANCE from this measurement).

Outputs:
  /tmp/qr_range.csv          per-detection rows
  stdout summary on Ctrl-C   distance bins, decode rates, id agreement, and a
                             suggested DEFAULT_VIEW_DISTANCE value
"""
import csv
import sys
import time

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from clue_hunt_solver import vision
from clue_hunt_solver.clue import parse_clue

BIN_M = 0.5
CSV_PATH = '/tmp/qr_range.csv'
STATUS_EVERY_S = 5.0


class QrRangeProbe(Node):
    def __init__(self):
        super().__init__('qr_range_probe')
        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.frames = 0
        self.rows_seen = 0
        self.qr_ok = 0
        self.qr_only = 0
        self.no_marker = 0
        self.max_decode = 0.0
        self.bins = {}            # bin index -> {'aruco': n, 'qr': n}
        self.pairs = {}           # (aruco_id, qr_id) -> n  (qr_id None = failed)
        self._t0 = time.time()
        self._last_status = self._t0
        self._csv = open(CSV_PATH, 'w', newline='')
        self._writer = csv.writer(self._csv)
        self._writer.writerow(['t_s', 'marker_id', 'dist_m', 'reproj_px',
                               'qr_text', 'qr_id'])
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '/camera/image_raw', self.on_image,
                                 qos_profile_sensor_data)
        print(f'probing /camera/image_raw - writing {CSV_PATH}', flush=True)

    def on_info(self, msg):
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def on_image(self, msg):
        if self.K is None:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:                              # noqa: BLE001
            print(f'cv_bridge error: {exc}', flush=True)
            return
        self.frames += 1
        rows = vision.analyse_frame(img, self.K, self.D)
        t = time.time() - self._t0
        marked = False
        for row in rows:
            self.rows_seen += 1
            dist = row['dist']
            text = row['qr_text'] or ''
            clue = parse_clue(text) if text else None
            qr_id = clue.id if clue else None
            if dist is not None:
                marked = True
                b = int(dist // BIN_M)
                slot = self.bins.setdefault(b, {'aruco': 0, 'qr': 0})
                slot['aruco'] += 1
                if text:
                    slot['qr'] += 1
                    self.max_decode = max(self.max_decode, dist)
            elif text:
                self.qr_only += 1
            if text:
                self.qr_ok += 1
            key = (row['marker_id'], qr_id)
            self.pairs[key] = self.pairs.get(key, 0) + 1
            self._writer.writerow([f'{t:.2f}', row['marker_id'],
                                   '' if dist is None else f'{dist:.3f}',
                                   '' if row['reproj'] is None else f'{row["reproj"]:.2f}',
                                   text, qr_id if qr_id is not None else ''])
        if not marked:
            self.no_marker += 1
        now = time.time()
        if now - self._last_status >= STATUS_EVERY_S:
            self._last_status = now
            print(f'[{now - self._t0:6.1f}s] frames={self.frames} '
                  f'rows={self.rows_seen} qr_ok={self.qr_ok} '
                  f'best_dist={self.max_decode:.2f} m', flush=True)

    def summary(self):
        self._csv.flush()
        print('\n=== QR range summary ===', flush=True)
        print(f'frames: {self.frames}   without marker: '
              f'{self.no_marker}   detection rows: {self.rows_seen}   '
              f'QR decodes: {self.qr_ok}   QR-only (no aruco): {self.qr_only}', flush=True)
        print(f'\n{"dist bin":>12}  {"aruco":>6} {"qr ok":>6} {"rate":>6}')
        reliable_top = 0.0
        for b in sorted(self.bins):
            s = self.bins[b]
            rate = s['qr'] / s['aruco'] if s['aruco'] else 0.0
            print(f'{b * BIN_M:4.1f}-{(b + 1) * BIN_M:4.1f} m  {s["aruco"]:6d} '
                  f'{s["qr"]:6d} {rate:5.0%}')
            if rate >= 0.5 and s['aruco'] >= 3:
                reliable_top = (b + 1) * BIN_M
        print(f'\nfurthest successful decode: {self.max_decode:.2f} m', flush=True)
        if reliable_top > 0:
            suggest = min(1.5, max(1.0, 0.8 * reliable_top))
            print(f'furthest >=50% bin: {reliable_top:.1f} m', flush=True)
            print(f'=> set vision.DEFAULT_VIEW_DISTANCE = {suggest:.1f} '
                  f'(0.8 x reliable range, clamped 1.0..1.5)', flush=True)
        else:
            print('=> no reliable bin yet - get closer to the board and re-run', flush=True)
        print('\nArUco id vs QR clue id:', flush=True)
        mismatch = 0
        for (aid, qid), n in sorted(
                self.pairs.items(),
                key=lambda kv: (kv[0][0] is None, kv[0][0] or 0,
                                kv[0][1] is None, kv[0][1] or 0)):
            if aid is None:
                label = 'no aruco'
            elif qid is None:
                label = 'QR failed'
            elif aid == qid:
                label = 'MATCH'
            else:
                label = 'MISMATCH'
                mismatch += 1
            print(f'  aruco={aid} qr_id={qid}: {n}  {label}', flush=True)
        print(f'mismatches: {mismatch} (practice expectation: 0; still ranking-only '
              f'in code - §2.1)', flush=True)
        self._csv.close()


def main():
    rclpy.init()
    node = QrRangeProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.summary()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == '__main__':
    sys.exit(main())
