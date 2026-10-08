#!/usr/bin/env python3
"""Collect follower-camera training images for the leader YOLO model.

Easiest way (no rqt save dialog): one command auto-saves frames while you
drive the leader with RViz Nav2 goals and the follower trails it.

Run (sim + Nav2 + both nodes already up, clean DDS env first - last_phase §8):
    python3 src/clue_hunt_solver/scripts/collect_training_data.py \
        --out ~/leader_dataset --rate 2.0 --auto-label

Output:
    <out>/images/frame_000001.jpg ...   (640x480 BGR)
    <out>/labels/frame_000001.txt ...   (YOLO txt, class 0, only --auto-label
                                         frames where tag 49 decoded)
    <out>/collect.csv                   (frame, dist_m, source)

Upload <out>/images (+ labels as pre-annotations) to Roboflow - see
train_data.md. Needs only rclpy + cv2 (no torch).
"""
import argparse
import csv
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..'))

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from cv_bridge import CvBridge


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--out', default=os.path.expanduser('~/leader_dataset'))
    p.add_argument('--rate', type=float, default=2.0,
                   help='save Hz (2 = ~120 shots/min while driving)')
    p.add_argument('--auto-label', action='store_true',
                   help='write YOLO txt from tag-49 leader box when decoded')
    p.add_argument('--max', type=int, default=0,
                   help='stop after N images (0 = unlimited, Ctrl-C to stop)')
    return p.parse_args()


class Collector(Node):
    def __init__(self, args):
        super().__init__('training_data_collector')
        self.args = args
        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.n = 0
        self.last_save = 0.0
        imgd = os.path.join(args.out, 'images')
        os.makedirs(imgd, exist_ok=True)
        if args.auto_label:
            os.makedirs(os.path.join(args.out, 'labels'), exist_ok=True)
        self.csv = open(os.path.join(args.out, 'collect.csv'), 'a', newline='')
        self.w = csv.writer(self.csv)
        if os.path.getsize(os.path.join(args.out, 'collect.csv')) == 0:
            self.w.writerow(['frame', 'dist_m', 'label'])
        self.create_subscription(Image, '/follower/camera/image_raw',
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/follower/camera/camera_info',
                                 self.on_info, qos_profile_sensor_data)
        self.get_logger().info(
            f'collecting -> {args.out} @ {args.rate} Hz '
            f'(auto-label={args.auto_label}). Drive the leader now!')

    def on_info(self, msg):
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def leader_box(self, img):
        """Tag-49 leader box (same math as follower overlay) -> (x1,y1,x2,y2,d)."""
        from clue_hunt_solver import vision
        if self.K is None:
            return None
        corners, ids, _ = vision.get_detector().detectMarkers(img)
        if ids is None or not len(ids):
            return None
        for quad, mid in zip(corners, ids.reshape(-1)):
            if int(mid) != 49:
                continue
            pose = vision.solve_marker_pose(
                np.asarray(quad, dtype=np.float64).reshape(4, 2),
                0.12, self.K, self.D)
            if pose is None:
                continue
            rvec, tvec, _err = pose
            pts = np.array([[x, y, z]
                            for x in (-0.20, 0.20)
                            for y in (-0.30, 0.15)
                            for z in (-0.40, 0.05)], dtype=np.float64)
            px = vision.project_points(pts, rvec, tvec, self.K, self.D)
            if not np.all(np.isfinite(px)):
                return None
            h, w = img.shape[:2]
            box = (max(float(np.min(px[:, 0])), 0.0),
                   max(float(np.min(px[:, 1])), 0.0),
                   min(float(np.max(px[:, 0])), float(w - 1)),
                   min(float(np.max(px[:, 1])), float(h - 1)))
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                return None
            return box + (float(np.linalg.norm(tvec)),)
        return None

    def on_image(self, msg):
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self.last_save < 1.0 / max(self.args.rate, 0.1):
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:                              # noqa: BLE001
            return
        self.last_save = now
        self.n += 1
        name = f'frame_{self.n:06d}'
        cv2.imwrite(os.path.join(self.args.out, 'images', name + '.jpg'), img)
        label, dist = '', ''
        if self.args.auto_label:
            got = self.leader_box(img)
            if got is not None:
                x1, y1, x2, y2, d = got
                h, w = img.shape[:2]
                cx, cy = ((x1 + x2) / 2.0 / w), ((y1 + y2) / 2.0 / h)
                bw, bh = ((x2 - x1) / w), ((y2 - y1) / h)
                with open(os.path.join(self.args.out, 'labels', name + '.txt'),
                          'w') as fh:
                    fh.write(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n')
                label, dist = 'auto', f'{d:.2f}'
        self.w.writerow([name, dist, label])
        self.csv.flush()
        self.get_logger().info(f'saved {name} ({dist}m {label}) [{self.n}]')
        if self.args.max and self.n >= self.args.max:
            self.get_logger().info('max reached - exiting')
            raise KeyboardInterrupt


def main():
    args = parse_args()
    rclpy.init()
    node = Collector(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        print(f'\nDone: {node.n} images in {args.out}/images')
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
