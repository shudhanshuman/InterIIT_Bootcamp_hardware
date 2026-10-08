#!/usr/bin/env python3
"""
Clue Chain Hunt - FOLLOWER node (last_phase.md).

Legal inputs ONLY: /follower/camera/*, /follower/odom, follower/* TF,
/leader/status. Output: /follower/cmd_vel @ 20 Hz + /follower/debug/image
(overlay for visual verification). Never touches leader odom/TF, Nav2, map.

Perception priority: tag 49 PnP (metric, authoritative) -> YOLO box
(verify/fallback when tag occluded) -> recovery spin. Control: trail
pure-pursuit, stand-off 1.3 m, band 0.6-2.0 m (final_plan.md section 5).

YOLO is OPTIONAL and lazy: no torch/ultralytics import at module load (torch
core-dumps in some VMs), loads only if param use_yolo:=true AND the weights
file exists. Tag-only is the default and is fully compliant.
"""
import csv
import math
import os
import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener, TransformException

from cv_bridge import CvBridge

from . import vision
from .frames import quaternion_to_matrix

ODOM_FRAME = 'follower/odom'
BASE_FRAME = 'follower/base_footprint'
CAM_FRAME = 'follower/cam_optical_link'

TAG_ID = 49
TAG_SIZE = 0.12                  # m, leader back tag
TAG_BEHIND_M = 0.21              # tag sits this far behind leader centre
STAND_OFF = 1.3                  # m, band midpoint (0.6-2.0)
TRAIL_MAX = 300
LOST_COAST_S = 0.7
KP_D = 0.8
KP_W = 1.2
V_MAX = 0.40
W_MAX = 1.0
YOLO_WIDTH_AT_STANDOFF = 110.0   # px, box width at 1.3 m (tune after verify)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class FollowerNode(Node):
    def __init__(self):
        super().__init__('follower_node')
        self.declare_parameter('use_yolo', False)
        self.declare_parameter('yolo_weights', '')
        self.declare_parameter('yolo_conf', 0.40)
        self.declare_parameter('debug_image', True)

        self.bridge = CvBridge()
        self.K: Optional[np.ndarray] = None
        self.D: Optional[np.ndarray] = None
        self.status = 'SEARCHING'
        self.last_tag_t = -1e9
        self.last_leader_xy: Optional[Tuple[float, float]] = None
        self.last_bearing = 0.0
        self.last_dist: Optional[float] = None
        self.last_source = 'none'          # tag | yolo | coast | lost
        self.trail: deque = deque(maxlen=TRAIL_MAX)
        self.robot_xy: Optional[Tuple[float, float]] = None
        self.robot_yaw = 0.0
        self._yolo = None
        self._yolo_tried = False
        self._last_frame = 0.0
        self._overlay_boxes: List[tuple] = []   # (x1,y1,x2,y2,label,src)

        self.create_subscription(Image, '/follower/camera/image_raw',
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/follower/camera/camera_info',
                                 self.on_info, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/follower/odom',
                                 self.on_odom, qos_profile_sensor_data)
        self.create_subscription(String, '/leader/status',
                                 self.on_status, 10)
        self.cmd_pub = self.create_publisher(Twist, '/follower/cmd_vel', 10)
        self.dbg_pub = self.create_publisher(Image, '/follower/debug/image', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_timer(0.05, self.control)

        self.log_path = '/tmp/follower_run_{}.csv'.format(
            time.strftime('%Y%m%d_%H%M%S'))
        self.log = open(self.log_path, 'w', newline='')
        self.csv = csv.writer(self.log)
        self.csv.writerow(['t', 'dist', 'bearing_deg', 'source', 'status'])
        self.get_logger().info(
            f'follower_node started (tag 49 PnP + YOLO verify) log {self.log_path}')

    # ------------------------------------------------------------- inputs
    def on_info(self, msg):
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def on_odom(self, msg):
        pass                               # trail lives in odom via TF lookups

    def on_status(self, msg):
        self.status = str(msg.data) if msg.data else 'SEARCHING'

    def _tf_odom_cam(self, stamp) -> Optional[tuple]:
        try:
            tr = self.tf_buffer.lookup_transform(ODOM_FRAME, CAM_FRAME, stamp,
                                                 Duration(seconds=0.15))
        except TransformException:
            return None
        t = tr.transform.translation
        q = tr.transform.rotation
        R = np.asarray(quaternion_to_matrix(q.x, q.y, q.z, q.w))
        return np.array([t.x, t.y, t.z]), R

    def _tf_robot(self) -> bool:
        try:
            tr = self.tf_buffer.lookup_transform(ODOM_FRAME, BASE_FRAME,
                                                 Time(), Duration(seconds=0.2))
        except TransformException:
            return False
        t = tr.transform.translation
        q = tr.transform.rotation
        R = np.asarray(quaternion_to_matrix(q.x, q.y, q.z, q.w))
        self.robot_xy = (float(t.x), float(t.y))
        self.robot_yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return True

    # ---------------------------------------------------------- perception
    def _detect_tag(self, img) -> Optional[dict]:
        """Tag 49 only, 0.12 m PnP. Returns dict with tvec/corners/reproj."""
        if self.K is None:
            return None
        corners, ids, _ = vision.get_detector().detectMarkers(img)
        if ids is None or not len(ids):
            return None
        for quad, mid in zip(corners, ids.reshape(-1)):
            if int(mid) != TAG_ID:
                continue
            pose = vision.solve_marker_pose(
                np.asarray(quad, dtype=np.float64).reshape(4, 2),
                TAG_SIZE, self.K, self.D)
            if pose is None:
                continue
            rvec, tvec, err = pose
            if err > vision.REPROJ_MAX_PX:
                continue
            return {'corners': np.asarray(quad, dtype=np.float32).reshape(4, 2),
                    'rvec': rvec, 'tvec': tvec, 'reproj': err}
        return None

    def _maybe_load_yolo(self):
        """Lazy YOLO: only on opt-in, never crashes tag-only mode."""
        if self._yolo_tried:
            return
        self._yolo_tried = True
        if not bool(self.get_parameter('use_yolo').value):
            return
        weights = str(self.get_parameter('yolo_weights').value or '')
        if not weights or not os.path.isfile(weights):
            self.get_logger().warn(
                f'use_yolo:=true but weights missing ({weights!r}) - tag-only')
            return
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(weights)
            self.get_logger().info(f'YOLO verify loaded: {weights}')
        except Exception as exc:                       # noqa: BLE001
            self._yolo = None
            self.get_logger().warn(f'YOLO load failed ({exc}) - tag-only')

    def _yolo_boxes(self, img) -> List[tuple]:
        """Leader boxes -> [(x1,y1,x2,y2,conf)] or []. Rectangles are YOLO's
        native output (axis-aligned x1,y1,x2,y2) - see last_phase.md."""
        self._maybe_load_yolo()
        if self._yolo is None:
            return []
        try:
            conf = float(self.get_parameter('yolo_conf').value)
            res = self._yolo.predict(img, conf=conf, verbose=False)[0]
            out = []
            for b in res.boxes:
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                out.append((x1, y1, x2, y2, float(b.conf[0])))
            out.sort(key=lambda e: -((e[2] - e[0]) * (e[3] - e[1])))
            return out
        except Exception:                              # noqa: BLE001
            return []

    def on_image(self, msg):
        if self.K is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if now - self._last_frame < 0.05:
            return                                     # ~20 Hz max
        self._last_frame = now
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:                              # noqa: BLE001
            return
        stamp = Time.from_msg(msg.header.stamp)
        tf = self._tf_odom_cam(stamp)
        if tf is None:
            return
        cam_pos, cam_R = tf
        self._overlay_boxes = []
        tag = self._detect_tag(img)
        if tag is not None:
            p_cam = tag['tvec'].reshape(3)
            p_odom = cam_R @ p_cam + cam_pos
            R_tag, _ = cv2.Rodrigues(tag['rvec'].reshape(3, 1))
            n_odom = cam_R @ (R_tag @ np.array([0.0, 0.0, 1.0]))
            nl = float(np.linalg.norm(n_odom[:2]))
            if nl > 1e-6:
                n_odom = n_odom / max(float(np.linalg.norm(n_odom)), 1e-9)
                lx = float(p_odom[0] - TAG_BEHIND_M * n_odom[0])
                ly = float(p_odom[1] - TAG_BEHIND_M * n_odom[1])
            else:
                lx, ly = float(p_odom[0]), float(p_odom[1])
            self.last_leader_xy = (lx, ly)
            self.last_tag_t = now
            self.last_source = 'tag'
            self.trail.append((lx, ly, now))
            c = tag['corners'].astype(int)
            cv2.polylines(img, [c], True, (0, 255, 0), 2)
            d = float(np.linalg.norm(p_cam))
            self.last_dist = math.hypot(lx - (self.robot_xy[0] if self.robot_xy else 0.0),
                                        ly - (self.robot_xy[1] if self.robot_xy else 0.0))
            cv2.putText(img, f'tag49 {d:.2f}m', tuple(c[0].tolist()),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            self._draw_leader_box(img, tag)
        else:
            for (x1, y1, x2, y2, cf) in self._yolo_boxes(img)[:1]:
                self._overlay_boxes.append((x1, y1, x2, y2, f'yolo {cf:.2f}', 'yolo'))
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                              (255, 0, 0), 2)
                self.last_source = 'yolo'
                w = max(x2 - x1, 1.0)
                # width ~ 1/dist: estimate dist from stand-off calibration
                self.last_dist = STAND_OFF * YOLO_WIDTH_AT_STANDOFF / w
                cx = (x1 + x2) / 2.0
                fx = float(self.K[0, 0])
                self.last_bearing = math.atan2(
                    (cx - float(self.K[0, 2])) / fx, 1.0)
        for (x1, y1, x2, y2, label, _s) in self._overlay_boxes:
            cv2.putText(img, label, (int(x1), int(y1) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        if bool(self.get_parameter('debug_image').value):
            try:
                self.dbg_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
            except Exception:                          # noqa: BLE001
                pass

    def _draw_leader_box(self, img, tag):
        """Rectangular box around the whole leader buggy, derived from the tag
        pose (chassis 0.40 x 0.30 m, tag 0.30 m up on the back plate). Drawn
        every tag frame, no YOLO needed - this is the 'box around my leader'
        for T2 verification. YOLO boxes (when enabled) use this same rectangle
        format: axis-aligned (x1,y1,x2,y2)."""
        try:
            xs = (-0.20, 0.20)
            ys = (-0.30, 0.15)          # floor .. roof relative tag centre
            zs = (-0.40, 0.05)          # body extends forward from the tag
            pts = np.array([[x, y, z] for x in xs for y in ys for z in zs],
                           dtype=np.float64)
            px = vision.project_points(pts, tag['rvec'], tag['tvec'],
                                       self.K, self.D)
            if not np.all(np.isfinite(px)):
                return
            h, w = img.shape[:2]
            x1 = max(float(np.min(px[:, 0])), 0.0)
            y1 = max(float(np.min(px[:, 1])), 0.0)
            x2 = min(float(np.max(px[:, 0])), float(w - 1))
            y2 = min(float(np.max(px[:, 1])), float(h - 1))
            if x2 - x1 < 8 or y2 - y1 < 8:
                return
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)),
                          (0, 255, 0), 2)
            cv2.putText(img, 'leader', (int(x1), max(int(y1) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception:                              # noqa: BLE001
            pass

    # ------------------------------------------------------------- control
    def _carrot(self) -> Optional[Tuple[float, float]]:
        """Trail point STAND_OFF behind the leader (collision-free path)."""
        if not self.trail or self.last_leader_xy is None:
            return None
        tx, ty = self.last_leader_xy
        want = STAND_OFF
        acc = 0.0
        px, py = tx, ty
        for (hx, hy, _t) in reversed(self.trail):
            acc += math.hypot(px - hx, py - hy)
            px, py = hx, hy
            if acc >= want:
                return (hx, hy)
        return (self.trail[0][0], self.trail[0][1])

    def control(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        cmd = Twist()
        if self.status == 'DONE':
            pass                                   # stopped, in band hopefully
        elif not self._tf_robot() or self.last_leader_xy is None:
            pass                                   # no fix yet: hold still
        else:
            rx, ry = self.robot_xy
            lx, ly = self.last_leader_xy
            d = math.hypot(lx - rx, ly - ry)
            self.last_dist = d
            carrot = self._carrot()
            cx, cy = carrot if carrot is not None else (lx, ly)
            bearing = _wrap_pi(math.atan2(cy - ry, cx - rx) - self.robot_yaw)
            self.last_bearing = bearing
            age = now - self.last_tag_t
            if self.last_source == 'tag' and age > LOST_COAST_S \
                    and self.last_source != 'yolo':
                self.last_source = 'lost'
            v = _clamp(KP_D * (d - STAND_OFF), -0.10, V_MAX)
            v = v * max(math.cos(bearing), 0.3)
            w = _clamp(KP_W * bearing, -W_MAX, W_MAX)
            if d < 0.6:
                v = -0.10 if d < 0.55 else 0.0    # too close: back off/stop
            elif d > 1.8:
                v = V_MAX                          # catch up
            if d < 1.0 and abs(bearing) > math.radians(20.0):
                v = 0.0                            # turn first, no corner cut
            if self.status in ('SEARCHING', 'READING') and 0.6 <= d <= 2.0:
                v, w = 0.0, 0.0                    # hold while tag spins away
                if d > 1.8:
                    v = V_MAX
            cmd.linear.x, cmd.angular.z = float(v), float(w)
        self.cmd_pub.publish(cmd)
        try:
            self.csv.writerow([f'{now:.2f}',
                               f'{self.last_dist:.2f}' if self.last_dist else '',
                               f'{math.degrees(self.last_bearing):.1f}',
                               self.last_source, self.status])
            self.log.flush()
        except Exception:                              # noqa: BLE001
            pass


def main():
    rclpy.init()
    node = FollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
