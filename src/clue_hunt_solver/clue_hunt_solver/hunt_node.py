#!/usr/bin/env python3
"""
Clue Chain Hunt - LEADER node (final_plan.md section 4).

Run (with the simulation + Nav2 already running):
  ros2 run clue_hunt_solver hunt_node
  ros2 launch clue_hunt_solver hunt.launch.py     (both nodes, start_nav:=false)

Topics the referee listens to (published here):
  /hunt/clues     std_msgs/String            full QR text of every VALID clue, in chain order
  /hunt/boards    std_msgs/String            "<id> <x> <y>" board ArUco centre estimate, map frame
  /hunt/treasure  geometry_msgs/PoseStamped  (frame "map") computed treasure point, once, after
                                             the leader has driven onto it
  /leader/status  std_msgs/String            MOVING / SEARCHING / READING / DONE  (from t = 0)

Design (final_plan.md section 4):
  * board outlines from LIVE lidar (flat 0.15-0.85 m segments - walls are long,
    pillars are known from our map): the "where to look" memory, ~5 cm, never
    hard-coded, rebuilt every run. Camera supplies facing + id + token.
  * one committed task at a time (EXPLORE coverage / CHECK a board both sides /
    PILLAR colours / GOTO target / TREASURE); no preemption of a driving goal,
    new tasks only when idle - this is what kills the bounce loop.
  * clue.py chain rule is the only filter (decoy / look-alike never publish)
  * wide spaces only (pose + follower-slot clearance, spins gated on clearance)
  * Nav2 actions are the only motion source (NavigateToPose / Spin)
Not allowed: /model ground truth, hard-coded board / pillar / treasure XY.
"""
import csv
import math
import os
import time
from collections import Counter, deque
from typing import List, Optional, Tuple

import numpy as np

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from action_msgs.msg import GoalStatus
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from cv_bridge import CvBridge

from . import outlines, pillars, search, vision
from .clue import ChainValidator, Clue, Verdict
from .frames import quaternion_from_yaw, quaternion_to_matrix, rel_target
from .outlines import Outline

MAP_FRAME = 'map'
BASE_FRAME = 'base_footprint'
CAM_FRAME = 'cam_optical_link'

STEP_YAW = math.radians(45.0)      # step-and-stare: ~FOV / 1.33
STARE_STEPS = 8                    # 8 x 45 deg = 360
STARE_PAUSE_S = 0.4
READING_SETTLE_S = 0.3             # discard samples while still settling
READING_DONE_S = 2.0               # early finish once enough good frames exist
READING_HARD_TIMEOUT_S = 4.5
APPROACH_TIMEOUT_S = 45.0
TRANSIT_TIMEOUT_S = 90.0
VIEWPOINT_TIMEOUT_S = 60.0
SPIN_TIMEOUT_S = 8.0
RING_BUDGET_S = 150.0
TOUR_BUDGET_S = 600.0
READ_ATTEMPTS_MAX = 2
APPROACH_FAILS_MAX = 2
SNAP_MAX_R = 1.0
COLOUR_GAP_MIN = 0.25              # assignment confidence (chromaticity gap)
COLOUR_OK_N = 2                    # >= 2 valid patches to trust colours
COLOUR_ROUNDS_MAX = 3             # colour-mission attempts while a clue is deferred
COLOUR_LOCK_N = 4                 # samples before a single pillar colour locks
COLOUR_LOCK_SPREAD = 0.20         # max mean chroma spread for a lock
COLOUR_LOCK_DIST = 0.60           # max dist to nearest ref for a lock
QR_HINT_MAX_D = 2.5               # opportunistic QR pre-read range (m)
QR_HINT_DT = 0.5                  # min gap between hint decodes (s)
STARTUP_NAV_WAIT_S = 30.0         # STARTUP waits this long for Nav2 before proceeding anyway
STARTUP_L0_WAIT_S = 8.0           # stare at spawn for the visible board-1 QR first
STARTUP_CREEP_M = 1.0             # then creep this far straight ahead for a closer read
STARTUP_CREEP_SETTLE_S = 3.0      # wait after creep for decode before main loop
WATCHDOG_S = 240.0                # no new clue for this long -> escalate (requeue + fresh tour)
APPROACH_RESEND_S = 1.0           # min gap between approach sends after a rejection
VIEWPOINT_RESEND_S = 1.0          # same for tour/ring/colour viewpoint sends
ARRIVAL_FRESH_S = 1.5             # sighting must be this fresh to enter READING on arrival
READING_EARLY_S = 2.5             # give up early if N samples decoded nothing
READING_EARLY_N = 8
READING_ADJ_MAX = 2               # in-place yaw nudges per reading when visible but undecodable
READING_ADJ_YAW = math.radians(15.0)
READING_ADJ_HOLD_S = 2.5          # ignore motion-blurred samples while nudging
READING_BASE_N = 10               # stationary frames for a normal finalize
READING_MAX_N = 24                # ...scaled up when localization is uncertain
AMCL_WARN_STD = 0.15              # log a warning above this localization std (m)
FOLLOWER_SLOT_M = 1.3             # keep-out behind the robot for the trailer
SLOT_CLEAR_MIN = 0.35             # min clearance required at pose and follower slot
SPIN_CLEAR_MIN = 1.0              # stare spins only with this clearance (follower-safe)
OUTLINE_COOL_S = 180.0            # exhausted outline revisit cooldown
SIBLING_COOL_R = 0.70             # same physical board: duplicates within this cool together
VIEWPOINT_VISIT_R = 0.60          # same viewpoint position radius (m)
VIEWPOINT_REVISIT_S = 180.0       # revisit a viewpoint after this (2nd sweep "just in case")
SCAN_OUTLINE_DT = 0.5             # lidar outline extraction throttle (s)
FRAME_MIN_DT = 0.10               # perception at most ~10 Hz (was 0.15/~7 Hz: colours/QR starved)
REJECT_TRIP_N = 3                 # rapid rejects in the window => stack not active
REJECT_WINDOW_S = 5.0
STACK_HOLD_S = 5.0                # hold sends this long per trip/probe cycle
IDLE_RETRY_S = 60.0               # idle re-tours this often (idle is never terminal)


def _now_sec(node: Node) -> float:
    return node.get_clock().now().nanoseconds * 1e-9


def reject_burst(times, now: float, n: int = REJECT_TRIP_N,
                 window: float = REJECT_WINDOW_S) -> bool:
    """True if >= n rejection timestamps fall in (now-window, now].

    Pure/testable core of the stack-health detector: action servers EXIST
    while Nav2 lifecycle is still activating, so server_is_ready() cannot see
    that outage - but a burst of instant rejections can.
    """
    cut = now - window
    k = 0
    for ts in times:
        if ts > cut:
            k += 1
            if k >= n:
                return True
    return False


def rel_from_board(board, a: float, b: float) -> Tuple[float, float]:
    (cx, cy), yaw = board
    return rel_target((cx, cy), yaw, a, b)


class HuntNode(Node):
    def __init__(self):
        super().__init__('hunt_node')
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('tour_yaml', '')

        # --- io ------------------------------------------------------------
        self.bridge = CvBridge()
        self.K = None
        self.D = None
        self.scan = None                     # (ranges, angle_min, angle_inc)
        self.amcl_std: Optional[float] = None   # localization std (m), None unknown
        self.robot_xy: Optional[Tuple[float, float]] = None
        self.robot_yaw = 0.0

        cam_group = ReentrantCallbackGroup()
        nav_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Image, '/camera/image_raw', self.on_image,
                                 qos_profile_sensor_data, callback_group=cam_group)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info,
                                 qos_profile_sensor_data, callback_group=cam_group)
        self.create_subscription(LaserScan, '/scan', self.on_scan,
                                 qos_profile_sensor_data, callback_group=cam_group)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose',
                                 self.on_amcl, 5, callback_group=cam_group)
        costmap_qos = QoSProfile(depth=1,
                                 durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                 reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap',
                                 self.on_costmap, costmap_qos,
                                 callback_group=cam_group)
        self.clue_pub = self.create_publisher(String, '/hunt/clues', 10)
        self.board_pub = self.create_publisher(String, '/hunt/boards', 10)
        self.treasure_pub = self.create_publisher(PoseStamped, '/hunt/treasure', 10)
        self.status_pub = self.create_publisher(String, '/leader/status', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                       callback_group=nav_group)
        self.spin_client = ActionClient(self, Spin, 'spin',
                                        callback_group=nav_group)

        # --- map, pillars, tour -------------------------------------------
        share = get_package_share_directory('clue_hunt_solver')
        map_yaml = self.get_parameter('map_yaml').value or \
            os.path.join(share, 'maps', 'arena.yaml')
        self.grid: Optional[search.Grid] = None
        self.clear_dist = None                   # static clearance map cache
        self.pillar_xys: List[Tuple[float, float, float]] = []
        self.tour: List[Tuple[float, float, float]] = []
        self.costmap_grid: Optional[search.Grid] = None
        try:
            self.grid = search.load_grid(map_yaml)
            self.pillar_xys = pillars.pillar_centres(self.grid)
            self.clear_dist = search.clearance_dist(self.grid)
            self.get_logger().info(
                f'map {map_yaml}: {len(self.pillar_xys)} pillar centre(s)')
        except Exception as exc:                       # noqa: BLE001
            self.grid = None
            self.clear_dist = None
            self.get_logger().error(f'map/pillar load failed: {exc}')
        tour_yaml = self.get_parameter('tour_yaml').value or \
            os.path.join(share, 'config', 'search_viewpoints.yaml')
        try:
            if os.path.isfile(tour_yaml):
                self.tour = search.load_tour(tour_yaml)
                self.get_logger().info(
                    f'tour: {len(self.tour)} viewpoints from {tour_yaml}')
            elif self.grid is not None:
                self.tour = search.build_tour_viewpoints(self.grid)
                self.get_logger().warn(
                    f'{tour_yaml} missing - built {len(self.tour)} viewpoints from the map')
        except Exception as exc:                       # noqa: BLE001
            self.get_logger().error(f'tour load failed: {exc}')

        # --- chain / board memory -----------------------------------------
        # BoardMemory (live lidar outlines) is the ONLY "where" memory: static
        # within a run, rebuilt every run. ChainValidator is the only filter.
        self.validator = ChainValidator()
        self.board_mem = outlines.BoardMemory()
        self.last_board: Optional[Tuple[Tuple[float, float], float]] = None
        self._last_scan_outline = 0.0

        # --- state ----------------------------------------------------------
        self.state = 'STARTUP'
        self.mode: Optional[str] = None            # ring | tour | colour | idle
        self._step2_cool = 0.0
        self.current_target: Optional[Tuple[float, float]] = None
        self.deferred_clue: Optional[Clue] = None
        self.t0 = _now_sec(self)
        self.last_progress_t = self.t0    # last accepted clue (watchdog)
        self.status = 'SEARCHING'

        self.reading_cluster: Optional[Outline] = None
        self.reading_id: Optional[int] = None
        self.reading_t0 = 0.0
        self.reading_samples: List[Tuple[float, float, float, str]] = []
        self.reading_attempts = 0
        self.reading_adj = 0
        self.reading_hold_until = 0.0

        self.approach_cluster: Optional[Outline] = None
        self.approach_cands: List[Tuple[float, float, float]] = []
        self.approach_idx = 0
        self.approach_normal: Optional[float] = None
        self.approach_board_xy: Tuple[float, float] = (0.0, 0.0)
        self.approach_backoff_until = 0.0

        self.viewpoints: List[Tuple[float, float, float]] = []
        self.vp_idx = 0
        self.at_vp = False
        self.stare_step = 0
        self.vp_backoff_until = 0.0
        self.pause_until = 0.0
        self.plan_t = 0.0
        self.tour_restarts = 0

        self.goal: Optional[dict] = None
        self.check_side = +1                       # side being checked (+1/-1)
        self.tour_pass = 0
        self._recent_dets = deque(maxlen=30)  # (id, x, y, t)
        self.pending_treasure: Optional[Tuple[float, float]] = None
        self.idle_since = 0.0
        self._reject_times: List[float] = []
        self.stack_ready = True                    # optimistic; trips on bursts
        self.stack_hold_until = 0.0
        self.last_treasure: Optional[Tuple[float, float]] = None

        self.colour_obs: dict = {}                 # centre idx -> [chroma...]
        self.chroma_personal: dict = {}          # colour -> calibrated ref chroma
        self.colour_map: Optional[dict] = None     # colour -> centre idx
        self.colour_locked: dict = {}            # centre idx -> colour (per-pillar lock, stops sampling)
        self.colour_rounds = 0
        self._last_colour_dbg = 0.0
        self._last_qr_hint = 0.0
        self._qr_hint: dict = {}                 # outline oid -> last hint text
        self.visited_views: list = []            # [(x, y, t)] viewpoint positions stood at
        self._startup_creep_sent = False
        self._startup_creep_done = False
        self._startup_creep_settle_until = 0.0
        self.frame_n = 0

        self.log_path = '/tmp/hunt_run_{}.csv'.format(
            time.strftime('%Y%m%d_%H%M%S'))
        self.log = open(self.log_path, 'w', newline='')
        self.csv = csv.writer(self.log)
        self.csv.writerow(['t', 'state', 'event', 'detail'])
        self.last_status_pub = 0.0
        self._last_health = 0.0
        self._last_stream_t: Optional[float] = None
        self._last_img_n = 0
        self._last_scan_n = 0
        self._t0_wall: Optional[float] = None
        self._t0_sim: Optional[float] = None
        self._img_n = 0
        self._last_frame = 0.0
        self._tf_ok = 0
        self._tf_fail = 0
        self._scan_n = 0
        self.create_timer(0.1, self.tick)
        self.create_timer(1.0, self._pose_tick)
        self.get_logger().info('hunt_node ready (state STARTUP)')
        self.get_logger().info(f'hunt log {self.log_path}')
        try:
            import datetime
            mtime = os.path.getmtime(__file__)
            stamp = datetime.datetime.fromtimestamp(mtime).strftime(
                '%Y-%m-%d %H:%M:%S')
        except Exception:                            # noqa: BLE001
            stamp = 'unknown'
        self.get_logger().info(f'hunt code mtime {stamp} ({__file__})')

    # ------------------------------------------------------------------ io
    def on_info(self, msg):
        self.K = np.asarray(msg.k, dtype=np.float64).reshape(3, 3)
        self.D = np.asarray(msg.d, dtype=np.float64) if len(msg.d) else np.zeros(5)

    def on_scan(self, msg):
        self.scan = (tuple(msg.ranges), float(msg.angle_min),
                     float(msg.angle_increment))
        self._scan_n += 1
        t = _now_sec(self)
        if t - self._last_scan_outline < SCAN_OUTLINE_DT:
            return
        self._last_scan_outline = t
        if self.robot_xy is None:
            return
        try:
            hyps = outlines.extract_hyps(
                msg.ranges, float(msg.angle_min), float(msg.angle_increment),
                self.robot_xy, self.robot_yaw, grid=self.grid,
                pillar_xys=[(p[0], p[1]) for p in self.pillar_xys])
        except Exception:                            # noqa: BLE001
            return
        n0 = len(self.board_mem.outlines)
        touched = self.board_mem.update_from_scan(hyps, self.robot_xy, t)
        if len(self.board_mem.outlines) > n0:
            self._log('outlines',
                      f'{len(self.board_mem.outlines) - n0} new outline(s), '
                      f'{len(self.board_mem.outlines)} total')

    def on_amcl(self, msg):
        try:
            cov = msg.pose.covariance
            self.amcl_std = float(max(math.sqrt(max(cov[0], 0.0)),
                                      math.sqrt(max(cov[7], 0.0))))
        except Exception:                            # noqa: BLE001
            self.amcl_std = None

    def on_costmap(self, msg):
        try:
            self.costmap_grid = search.grid_from_occupancy(
                msg.data, msg.info.width, msg.info.height,
                msg.info.origin.position.x, msg.info.origin.position.y,
                msg.info.resolution)
        except Exception as exc:                     # noqa: BLE001
            self.get_logger().warn(f'costmap convert failed: {exc}',
                                   throttle_duration_sec=30.0)

    def _log(self, event: str, detail: str = ''):
        self.csv.writerow([f'{_now_sec(self):.2f}', self.state, event, detail])
        self.log.flush()

    def _nav_ready(self) -> bool:
        """Both action servers accepting goals (else sends would be instant
        rejections that burn approach candidates for nothing)."""
        try:
            return bool(self.nav_client.server_is_ready()
                        and self.spin_client.server_is_ready())
        except Exception:                            # noqa: BLE001
            return False

    def _stack_held(self, t: float) -> bool:
        """True while a rejection burst says the stack is not really up.

        server_is_ready() is TRUE during lifecycle activation, so this (burst
        detector + hold window with probes) is the real readiness signal.
        When healthy this is always False and every send proceeds as before.
        """
        return (not self.stack_ready) and t < self.stack_hold_until

    def _los_clear(self, x: float, y: float, tx: float, ty: float) -> bool:
        """Static-map line of sight from a viewpoint to a board position."""
        if self.grid is None:
            return True
        try:
            return pillars.line_of_sight(self.grid, (x, y), (tx, ty))
        except Exception:                            # noqa: BLE001
            return True

    # ----------------------------------------------------------------- tf
    def _tf(self, source: str, stamp) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        try:
            tr = self.tf_buffer.lookup_transform(MAP_FRAME, source, stamp,
                                                 Duration(seconds=0.15))
        except TransformException:
            self._tf_fail += 1
            return None
        t = tr.transform.translation
        q = tr.transform.rotation
        R = np.asarray(quaternion_to_matrix(q.x, q.y, q.z, q.w), dtype=np.float64)
        self._tf_ok += 1
        return np.array([t.x, t.y, t.z]), R

    def _pose_tick(self):
        """1 Hz pose refresh decoupled from the camera: a camera stall must
        blind only perception - navigation, outlines and tasking keep their
        pose. Uses latest-time TF (not image-stamp)."""
        if self.state == 'DONE':
            return
        try:
            tr = self.tf_buffer.lookup_transform(
                MAP_FRAME, BASE_FRAME, Time(), Duration(seconds=0.2))
        except TransformException:
            self._tf_fail += 1
            return
        self._tf_ok += 1
        t = tr.transform.translation
        q = tr.transform.rotation
        self.robot_xy = (float(t.x), float(t.y))
        R = np.asarray(quaternion_to_matrix(q.x, q.y, q.z, q.w), dtype=np.float64)
        self.robot_yaw = float(np.arctan2(R[1, 0], R[0, 0]))

    # ------------------------------------------------------------ perception
    def on_image(self, msg):
        if self.state == 'DONE' or self.K is None:
            return
        now = _now_sec(self)
        if now - self._last_frame < FRAME_MIN_DT:
            return                             # drop frames: sim RTF matters more
        self._last_frame = now
        self.frame_n += 1
        self._img_n += 1
        stamp = Time.from_msg(msg.header.stamp)
        cam = self._tf(CAM_FRAME, stamp)
        if cam is None:
            return
        cam_pos, cam_R = cam
        base = self._tf(BASE_FRAME, stamp)
        if base is not None:
            self.robot_xy = (float(base[0][0]), float(base[0][1]))
            self.robot_yaw = float(np.arctan2(base[1][1, 0], base[1][0, 0]))
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:                            # noqa: BLE001
            return
        dets = vision.detect_markers(img, self.K, self.D)
        t = _now_sec(self)

        for det in dets:
            p_cam = det.tvec.reshape(3)
            p_map = cam_R @ p_cam + cam_pos
            n_map = cam_R @ det.normal_in_camera()
            n_xy = math.hypot(float(n_map[0]), float(n_map[1]))
            normal_yaw = math.atan2(float(n_map[1]), float(n_map[0])) \
                if n_xy > 0.3 else None
            # Camera -> outline memory: associate (lidar position wins), else a
            # provisional PnP outline. Facing comes from the observer side.
            # Detections NEVER steer motion directly - task selection does that
            # when idle, so a committed drive cannot be hijacked (no bounce).
            o = self.board_mem.associate(float(p_map[0]), float(p_map[1]))
            if o is None:
                # same id, drifted PnP: stick to the known board instead of
                # spawning/starving duplicate outlines
                o = self.board_mem.associate_same_id(
                    det.marker_id, float(p_map[0]), float(p_map[1]))
            self._recent_dets.append(
                (det.marker_id, float(p_map[0]), float(p_map[1]), t))
            if o is None:
                o = self.board_mem.add_provisional(
                    float(p_map[0]), float(p_map[1]), t, normal_yaw,
                    self.robot_xy)
                self._log('outline_new',
                          f'pnp id={det.marker_id} xy={p_map[0]:.2f},{p_map[1]:.2f}')
            if self.robot_xy is not None:
                self.board_mem.note_detection(o, self.robot_xy, det.marker_id,
                                              normal_yaw, t)
            self._log('sighting',
                      f'id={det.marker_id} xy={p_map[0]:.2f},{p_map[1]:.2f} '
                      f'reproj={det.reproj_err:.2f} outline={o.oid} '
                      f'facing={o.facing} verdict={o.verdict}')
            if self.state == 'READING' and \
                    (self.reading_id is None or det.marker_id == self.reading_id) \
                    and self._sample_near_outline(o, p_map):
                text = vision.read_qr(img, det, self.K, self.D)
                if t - self.reading_t0 >= READING_SETTLE_S and \
                        t >= self.reading_hold_until:
                    yaw = float(math.atan2(n_map[1], n_map[0])) \
                        if normal_yaw is not None else float('nan')
                    self.reading_samples.append(
                        (float(p_map[0]), float(p_map[1]), yaw, text))
        # Opportunistic QR hint outside READING: decode the closest marker
        # each QR_HINT_DT so the chain pre-reads while approaching/touring.
        # Capped to 1 decode/frame to protect RTF; READING still does the
        # authoritative 10-frame median.
        if self.state != 'READING' and dets and \
                t - self._last_qr_hint >= QR_HINT_DT:
            try:
                nearest_det = min(dets, key=lambda d: d.distance)
            except ValueError:                   # noqa: BLE001
                nearest_det = None
            if nearest_det is not None and nearest_det.distance <= QR_HINT_MAX_D \
                    and nearest_det.reproj_err <= 3.0:
                hint = vision.read_qr(img, nearest_det, self.K, self.D)
                self._last_qr_hint = t
                if hint:
                    self._qr_hint[nearest_det.marker_id] = hint
                    self._log('qr_hint',
                              f'id={nearest_det.marker_id} d={nearest_det.distance:.2f} '
                              f'{hint}')
        self._sample_colours(img, cam_pos, cam_R)

    def _sample_near_outline(self, o: Outline, p_map) -> bool:
        rc = self.reading_cluster
        if rc is None:
            return False
        if o is rc:
            return True
        # same id, different outline (look-alike): only sample if the DETECTED
        # point is actually near the outline we are reading
        return math.hypot(p_map[0] - rc.x, p_map[1] - rc.y) <= 0.9

    def _sample_colours(self, img, cam_pos, cam_R):
        if self.colour_map is not None or not self.pillar_xys or self.scan is None:
            return
        if self.robot_xy is None:
            return
        # Every processed frame (was every 5th: ~1.3 Hz -> ~10 Hz). Locked
        # pillars are skipped entirely; once 2 lock, the 3rd is elimination.
        ranges, a_min, a_inc = self.scan
        n_vis = n_proj = n_patch = n_chroma = 0
        for idx, (px, py, _r) in enumerate(self.pillar_xys):
            if idx in self.colour_locked:
                continue                               # found: stop sampling it
            if not pillars.pillar_visible(ranges, a_min, a_inc, self.robot_xy,
                                          self.robot_yaw, (px, py)):
                continue
            n_vis += 1
            # sample several heights: the centre projection can land on the
            # pillar rim/background while a higher/lower patch is clean
            best, best_d = None, None
            for dz in (0.3, 0.5, 0.7):
                proj = pillars.project_point((px, py, dz), cam_pos, cam_R,
                                              self.K)
                if proj is None:
                    continue
                n_proj += 1
                u, v, _d = proj
                ch = pillars.sample_patch(img, u, v)
                if ch is None:
                    continue
                n_patch += 1
                _ref, d = pillars.nearest_ref(ch)
                if best_d is None or d < best_d:
                    best, best_d = ch, d
            if best is None or best_d > pillars.CHROMA_MAX_DIST:
                continue
            n_chroma += 1
            self.colour_obs.setdefault(idx, []).append(best)
        self._sample_colours_hsv(img, cam_pos, cam_R)
        self._update_colour_locks()
        t = _now_sec(self)
        if t - self._last_colour_dbg >= 5.0:
            self._last_colour_dbg = t
            parts = []
            for k, chs in sorted(self.colour_obs.items()):
                med = tuple(float(np.median([c[j] for c in chs])) for j in range(3))
                ref, d = pillars.nearest_ref(med)
                lock = self.colour_locked.get(k, '-')
                parts.append(f'{k}:n={len(chs)} med=({med[0]:.2f},{med[1]:.2f},'
                             f'{med[2]:.2f}) {ref}~{d:.2f} lock={lock}')
            self._log('colour_dbg',
                      f'vis={n_vis} proj={n_proj} patch={n_patch} chroma={n_chroma} '
                      f'obs={{{"; ".join(parts)}}} resolved={self.colour_map is not None}')
        self._try_resolve_colours()

    def _sample_colours_hsv(self, img, cam_pos, cam_R):
        """Second colour path: 3 HSV filters over the whole image every frame,
        associated by relative bearing (robot yaw + xy vs known map centres).

        The projector path looks at ONE pixel and dies on any TF/AMCL offset
        (log: vis=1 proj=3 patch=0). This path finds the saturated blob first
        (you can see it in the feed) and asks WHICH pillar sits on that
        bearing. Votes feed the same colour_obs/locks, so 2 votes still imply
        the 3rd by elimination."""
        if self.colour_map is not None or self.robot_xy is None:
            return
        try:
            blobs = pillars.detect_colour_blobs(img)
        except Exception:                          # noqa: BLE001
            return
        if not blobs:
            return
        gate = math.radians(getattr(pillars, 'HSV_BEARING_GATE_DEG', 10.0))
        for colour, u, v, area in blobs[:3]:      # biggest blobs only
            obs_yaw = pillars.bearing_for_pixel(u, v, self.K, cam_R)
            if obs_yaw is None:
                continue
            best_idx, best_d = None, gate
            for idx, (px, py, _r) in enumerate(self.pillar_xys):
                if idx in self.colour_locked:
                    continue                       # found: stop voting for it
                exp_yaw = math.atan2(py - self.robot_xy[1],
                                     px - self.robot_xy[0])
                d = abs((obs_yaw - exp_yaw + math.pi) % (2.0 * math.pi) - math.pi)
                if d < best_d:
                    best_idx, best_d = idx, d
            if best_idx is None:
                continue
            # vote the canonical ref chroma: lock logic needs tight agreement,
            # and HSV already classified the hue (no grey-patch ambiguity)
            vote = tuple(pillars.REF_CHROMA[colour])
            self.colour_obs.setdefault(best_idx, []).append(vote)
            self._log('hsv_vote',
                      f'{colour} blob u={u:.0f} v={v:.0f} area={area} '
                      f'-> idx {best_idx} off={math.degrees(best_d):.1f}deg')

    def _update_colour_locks(self):
        """Lock a pillar's colour once its samples are tight + stable.

        A locked pillar is never sampled again. Two distinct locks imply
        the third by elimination (relative-vector targets PILLAR/BETWEEN
        then resolve without waiting for the far pillar).
        """
        if self.colour_map is not None:
            return
        for idx, chs in list(self.colour_obs.items()):
            if idx in self.colour_locked or len(chs) < COLOUR_LOCK_N:
                continue
            med = tuple(float(np.median([c[k] for c in chs])) for k in range(3))
            spread = float(np.mean(
                [pillars.chroma_distance(c, med) for c in chs]))
            name, d = pillars.nearest_ref(med)
            if spread <= COLOUR_LOCK_SPREAD and d <= COLOUR_LOCK_DIST:
                # require the last LOCK_N samples to agree on the colour
                tail = [pillars.nearest_ref(c)[0] for c in chs[-COLOUR_LOCK_N:]]
                if all(nm == name for nm in tail) and \
                        name not in self.colour_locked.values():
                    self.colour_locked[idx] = name
                    self._log('colour_lock',
                              f'idx {idx} -> {name} n={len(chs)} '
                              f'spread={spread:.2f} d={d:.2f}')
                    self.get_logger().info(
                        f'pillar idx {idx} locked as {name} (stop sampling it)')
        # Two distinct locks -> third by elimination, no Hungarian needed.
        if len(self.colour_locked) == 2 and self.colour_map is None:
            from .clue import COLOURS
            locked_vals = set(self.colour_locked.values())
            if len(locked_vals) == 2:
                missing = [c for c in COLOURS if c not in locked_vals][0]
                missing_idx = [i for i in range(len(self.pillar_xys))
                               if i not in self.colour_locked][0]
                mapping = dict(self.colour_locked)
                # invert: colour -> idx
                inv = {v: k for k, v in self.colour_locked.items()}
                inv[missing] = missing_idx
                self.colour_map = inv
                self.colour_rounds = 0
                # relative vectors now known: log pillar geometry for report
                vecs = {}
                for c, j in inv.items():
                    px, py, _r = self.pillar_xys[j]
                    vecs[c] = (round(px, 2), round(py, 2))
                self._log('colour', f'{inv} via 2-lock elimination {vecs}')
                self.get_logger().info(
                    f'pillar colours assigned via elimination: {inv}')

    def _try_resolve_colours(self):
        if self.colour_map is not None or not self.colour_obs:
            return
        medians = {}
        for idx, chs in self.colour_obs.items():
            medians[idx] = tuple(float(np.median([c[k] for c in chs]))
                                 for k in range(3))
        # Calibrate: a pillar with >=4 tight samples adopts its median as the
        # personal reference for its nearest axis colour (rendered colours are
        # desaturated vs the ideal axes - measured RED~(0.61,0.19,0.19)).
        # Identity stays anchored to the axes; the gap test still guards.
        for idx, chs in self.colour_obs.items():
            if len(chs) < 4:
                continue
            med = medians[idx]
            spread = float(np.mean(
                [pillars.chroma_distance(c, med) for c in chs]))
            name, d = pillars.nearest_ref(med)
            if spread < 0.20 and d <= 0.60 and name not in self.chroma_personal:
                self.chroma_personal[name] = med
                self._log('chroma_calib',
                          f'{name} <- idx {idx} med=({med[0]:.2f},{med[1]:.2f},'
                          f'{med[2]:.2f}) spread={spread:.2f}')
        ref_map = dict(pillars.REF_CHROMA)
        ref_map.update(self.chroma_personal)
        valid_n = 0
        for idx, med in medians.items():
            best = min(pillars.chroma_distance(med, ref) for ref in ref_map.values())
            if best <= pillars.CHROMA_MAX_DIST:
                valid_n += 1
        mapping, gap = pillars.assign_colours(medians, ref_map=ref_map)
        if mapping is not None and valid_n >= COLOUR_OK_N and gap >= COLOUR_GAP_MIN:
            self.colour_map = mapping
            self.colour_rounds = 0
            self._log('colour', f'{mapping} gap={gap:.3f} n={valid_n}')
            self.get_logger().info(f'pillar colours assigned: {mapping}')

    # -------------------------------------------------------- goal plumbing
    def _send_nav(self, x: float, y: float, yaw: float, purpose: str,
                  meta=None, timeout: float = TRANSIT_TIMEOUT_S) -> bool:
        if self.goal is not None:
            self._cancel_goal()
        t = _now_sec(self)
        if self._stack_held(t):
            # outage hold: refuse without consuming anything; callers wait
            # (their backoff was set here) and retry - nothing burns.
            self.vp_backoff_until = t + VIEWPOINT_RESEND_S
            return False
        if not self.nav_client.server_is_ready():
            self.get_logger().warn('navigate_to_pose server not ready',
                                   throttle_duration_sec=5.0)
        pose = PoseStamped()
        pose.header.frame_id = MAP_FRAME
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x, pose.pose.position.y = float(x), float(y)
        q = quaternion_from_yaw(float(yaw))
        pose.pose.orientation.z, pose.pose.orientation.w = q[2], q[3]
        goal = NavigateToPose.Goal()
        goal.pose = pose
        self.goal = {'kind': 'nav', 'purpose': purpose, 'meta': meta,
                     'send': self.nav_client.send_goal_async(goal),
                     'handle': None, 'result': None,
                     't0': _now_sec(self), 'timeout': timeout}
        self._log('goal_send', f'{purpose} {x:.2f},{y:.2f},{yaw:.2f}')
        return True

    def _send_spin(self, step: float = STEP_YAW) -> bool:
        if self.goal is not None:
            self._cancel_goal()
        if self._stack_held(_now_sec(self)):
            self.vp_backoff_until = _now_sec(self) + VIEWPOINT_RESEND_S
            return False
        goal = Spin.Goal()
        goal.target_yaw = step
        self.goal = {'kind': 'spin', 'purpose': 'spin', 'meta': None,
                     'send': self.spin_client.send_goal_async(goal),
                     'handle': None, 'result': None,
                     't0': _now_sec(self), 'timeout': SPIN_TIMEOUT_S}
        self._log('goal_send', f'spin {math.degrees(step):.0f}deg')
        return True

    def _cancel_goal(self):
        g = self.goal
        if g is None:
            return
        try:
            if g['handle'] is None:
                if g['send'].done() and g['send'].result() is not None:
                    g['send'].result().cancel_goal_async()
            elif g['handle'].done() and g['handle'].result() is not None:
                g['handle'].result().cancel_goal_async()
        except Exception:                            # noqa: BLE001
            pass
        self.goal = None
        self._log('goal_cancel', '')

    def _poll_goal(self) -> Optional[Tuple[str, str]]:
        """Advance the current goal; return (purpose, outcome) when finished."""
        g = self.goal
        if g is None:
            return None
        t = _now_sec(self)
        if g['handle'] is None:
            if not g['send'].done():
                if t - g['t0'] > 10.0:
                    self.goal = None
                    self._log('goal_fail', 'send timeout')
                    return (g['purpose'], 'failed')
                return None
            handle = g['send'].result()
            if handle is None or not handle.accepted:
                self.goal = None
                self._log('goal_fail', 'rejected')
                # stack-health detector: a burst of instant rejections means
                # the stack is not really up (lifecycle activating), no matter
                # what server_is_ready() claims. Trip -> hold sends, probe on.
                self._reject_times.append(t)
                self._reject_times = [
                    ts for ts in self._reject_times if ts > t - REJECT_WINDOW_S]
                self.stack_hold_until = t + STACK_HOLD_S
                if self.stack_ready and reject_burst(self._reject_times, t):
                    self.stack_ready = False
                    self._log('stack_hold',
                              f'{len(self._reject_times)} rejects in '
                              f'{REJECT_WINDOW_S:.0f}s - holding sends')
                    self.get_logger().warn(
                        'nav stack rejecting goals - holding sends (no burn)')
                return (g['purpose'], 'rejected')
            if not self.stack_ready:
                self.stack_ready = True
                self._log('stack_ready', 'goal accepted - resuming sends')
                self.get_logger().info('nav stack accepting goals - resumed')
            g['handle'] = handle
            g['result'] = handle.get_result_async()
        if g['result'] is not None and g['result'].done():
            status = g['result'].result().status
            self.goal = None
            if status == GoalStatus.STATUS_SUCCEEDED:
                self._log('goal_ok', g['purpose'])
                return (g['purpose'], 'ok')
            self._log('goal_fail', f"{g['purpose']} status={status}")
            return (g['purpose'], 'failed')
        if t - g['t0'] > g['timeout']:
            self._cancel_goal()
            return (g['purpose'], 'failed')
        return None

    # ----------------------------------------------------------- geometry
    def _snap(self, x: float, y: float,
              max_r: float = SNAP_MAX_R) -> Tuple[float, float]:
        seed = self.robot_xy
        if self.costmap_grid is not None:
            out = search.snap_to_free(self.costmap_grid, x, y, max_r=max_r,
                                      seed=seed)
        elif self.grid is not None:
            out = search.snap_to_free(self.grid, x, y, max_r=max_r,
                                      clearance_m=0.35, seed=seed)
        else:
            out = None
        if out is None:
            self.get_logger().warn(f'no free cell near {x:.2f},{y:.2f}',
                                   throttle_duration_sec=10.0)
            return (x, y)
        return out

    def _resolve(self, clue: Clue) -> Optional[Tuple[float, float]]:
        """Command -> target XY (final_plan 4.4); None if colours missing."""
        if clue.verb == 'GOTO':
            return (clue.args[0], clue.args[1])
        if clue.verb == 'REL':
            if self.last_board is None:
                self.get_logger().error('REL without a measured board')
                return None
            return rel_from_board(self.last_board, clue.args[0], clue.args[1])
        if clue.verb == 'PILLAR':
            return self._pillar_centre(clue.args[0])
        if clue.verb == 'BETWEEN':
            a = self._pillar_centre(clue.args[0])
            b = self._pillar_centre(clue.args[1])
            if a is None or b is None:
                return None
            f = clue.args[2]
            return (a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        return None

    def _pillar_centre(self, colour: str) -> Optional[Tuple[float, float]]:
        if self.colour_map is None or colour not in self.colour_map:
            return None
        idx = self.colour_map[colour]
        if not 0 <= idx < len(self.pillar_xys):
            return None
        px, py, _r = self.pillar_xys[idx]
        return (px, py)

    # --------------------------------------------------------- state enter
    def _enter(self, state: str, why: str = ''):
        self.get_logger().info(f'state {self.state} -> {state} {why}')
        self._log('state', f'{self.state}->{state} {why}')
        self.state = state

    def _build_approach_cands(self, o: Outline, side: int,
                              normal: Optional[float]) -> List[Tuple[float, float, float]]:
        """Viewing poses for one side, then observer-bearing fallback poses.

        With a lidar axis the side is exact; the fallback covers provisional
        (camera-only) outlines whose normal is untrusted.
        """
        bx, by = o.x, o.y
        out: List[Tuple[float, float, float]] = []
        if o.axis is not None:
            nx, ny = outlines.normal_vec(o.axis, side)
            out.extend(search.viewing_pose_candidates(
                (bx, by), math.atan2(ny, nx)))
        if normal is not None:
            for c in search.viewing_pose_candidates((bx, by), normal):
                if c not in out:
                    out.append(c)
        if o.observers:
            mox = sorted(p[0] for p in o.observers)[len(o.observers) // 2]
            moy = sorted(p[1] for p in o.observers)[len(o.observers) // 2]
            back = math.atan2(moy - by, mox - bx)
            for c in search.viewing_pose_candidates((bx, by), back):
                if c not in out:
                    out.append(c)
        return out

    def _try_other_side(self, o: Outline) -> bool:
        """Blind arrival with unknown facing: check the outline's other face.
        Returns True if a goal was sent. Known facing never tries the back."""
        if o.axis is None or o.facing != 0:
            return False
        other = -self.check_side
        if o.tried & (1 if other > 0 else 2):
            return False
        self.check_side = other
        o.tried |= (1 if other > 0 else 2)
        self.approach_cands = self._build_approach_cands(
            o, other, self.approach_normal)
        self.approach_idx = 0
        self._enter('APPROACH', f'outline={o.oid} other side {other:+d}')
        return self._send_approach_goal()

    def _begin_approach(self, o: Outline, side: int = 0):
        """Commit to checking one outline from one side (CHECK task)."""
        self._cancel_goal()
        self.approach_cluster = o
        if side == 0:
            side = o.facing if o.facing != 0 else +1
        self.check_side = side
        o.tried |= (1 if side > 0 else 2)
        self.reading_attempts = 0
        bx, by = o.x, o.y
        self.approach_board_xy = (bx, by)
        normal = outlines.mean_yaw(o.n_yaws)
        if normal is None and self.robot_xy is not None:
            normal = math.atan2(self.robot_xy[1] - by, self.robot_xy[0] - bx)
        if normal is None:
            normal = 0.0
        self.approach_normal = normal
        self.approach_cands = self._build_approach_cands(o, side, normal)
        self.approach_idx = 0
        self._enter('APPROACH',
                    f'outline={o.oid} id={o.board_id} side={side:+d} '
                    f'at {bx:.2f},{by:.2f}')
        if self._send_approach_goal():
            return
        if self._stack_held(_now_sec(self)) or not self._nav_ready():
            return                                     # wait in APPROACH
        self._approach_failed()

    def _cand_score(self, sx: float, sy: float, yaw: float) -> float:
        """min(pose clearance, follower-slot clearance): room to park AND room
        for the 1.3 m trailer behind without reversing into a wall."""
        if self.clear_dist is None or self.grid is None:
            return 1e9
        cp = search.clearance_at(self.clear_dist, self.grid, sx, sy)
        fx = sx - FOLLOWER_SLOT_M * math.cos(yaw)
        fy = sy - FOLLOWER_SLOT_M * math.sin(yaw)
        cf = search.clearance_at(self.clear_dist, self.grid, fx, fy)
        return min(cp, cf)

    def _send_approach_goal(self) -> bool:
        """Send the best visible, front-side candidate. Returns True if sent."""
        if not self._nav_ready():
            self.approach_backoff_until = _now_sec(self) + APPROACH_RESEND_S
            self.get_logger().warn(
                'nav not ready - deferring approach (no candidate consumed)',
                throttle_duration_sec=5.0)
            return False
        o = self.approach_cluster
        if o is None:
            return False
        bx, by = self.approach_board_xy
        observers = list(o.observers[-6:])
        normal = outlines.confident_yaw(o.n_yaws)
        if normal is None:
            normal = getattr(self, 'approach_normal', None)
        # head-on preference: incidence from the lidar axis (accurate to ~2
        # deg, independent of the camera) when the facing side is known
        axis_nyaw = None
        if o.axis is not None:
            _ax, _ay = outlines.normal_vec(
                o.axis, o.facing if o.facing != 0 else self.check_side)
            axis_nyaw = math.atan2(_ay, _ax)
        scored = []
        for i in range(self.approach_idx, len(self.approach_cands)):
            gx, gy, gyaw = self.approach_cands[i]
            sx, sy = self._snap(gx, gy, max_r=0.8)
            why = None
            if not search.pose_sees_point(sx, sy, gyaw, bx, by):
                why = 'frustum/range'
            elif not self._los_clear(sx, sy, bx, by):
                why = 'los-blocked'
            elif not search.same_side_as_observers(bx, by, observers, sx, sy):
                want = math.atan2(sy - by, sx - bx)
                near = min((search.angular_diff(
                    want, math.atan2(oy - by, ox - bx)) for ox, oy in observers),
                    default=float('nan'))
                why = (f'back-side pose_bear={math.degrees(want):.0f} '
                       f'nearest_obs_off={math.degrees(near):.0f}deg')
            elif normal is not None and not search.incidence_ok(
                    bx, by, normal, sx, sy):
                want = math.atan2(sy - by, sx - bx)
                diff = abs((want - normal + math.pi) % (2.0 * math.pi) - math.pi)
                why = (f'incidence pose_bear={math.degrees(want):.0f} '
                       f'normal={math.degrees(normal):.0f} '
                       f'off={math.degrees(diff):.0f}deg')
            if why is not None:
                self._log('approach_skip',
                          f'idx={i} {why} pose={sx:.2f},{sy:.2f}')
                continue
            inc = 0.0
            if axis_nyaw is not None:
                inc = math.degrees(search.angular_diff(
                    math.atan2(sy - by, sx - bx), axis_nyaw))
            scored.append((self._cand_score(sx, sy, gyaw), i, sx, sy, gyaw,
                           inc))
        if not scored:
            self.approach_idx = len(self.approach_cands)
            return False
        # safe pockets first, then most head-on, then roomier (follower safety
        # outranks a few degrees of incidence)
        scored.sort(key=lambda e: (0.0 if e[0] >= SLOT_CLEAR_MIN else 1.0,
                                   e[5], -e[0], e[1]))
        score, i, sx, sy, gyaw, inc = scored[0]
        self._log('approach_send',
                  f'idx={i} score={score:.2f} inc={inc:.0f}deg '
                  f'pose={sx:.2f},{sy:.2f}')
        if score < SLOT_CLEAR_MIN:
            self._log('approach_tight',
                      f'idx={i} score={score:.2f} (tight pocket, sending anyway)')
        self.approach_idx = i
        sent = self._send_nav(sx, sy, gyaw, 'approach',
                              timeout=APPROACH_TIMEOUT_S)
        if not sent:
            # refused (stack hold): back off so the tick waits instead of
            # re-scanning at 10 Hz claiming sends that never happened
            self.approach_backoff_until = _now_sec(self) + APPROACH_RESEND_S
        return sent

    def _cool_siblings(self, o: Outline, t: float):
        """Persistent board memory: duplicates of the same physical board cool
        together. Lidar/camera splits one board into several oids (b2 was
        17/5/63/2/...); cooling only the checked oid lets the next tick chase
        its twin 0.3 m away forever. Siblings within SIBLING_COOL_R share the
        same cool_until so the chain moves forward instead of bouncing."""
        n = 0
        for s in self.board_mem.outlines:
            if s is o:
                continue
            if math.hypot(s.x - o.x, s.y - o.y) > SIBLING_COOL_R:
                continue
            if s.verdict in ('valid', 'rejected'):
                continue                       # definitive: leave alone
            if s.cool_until < o.cool_until:
                s.cool_until = o.cool_until
                n += 1
        if n:
            self._log('sibling_cool',
                      f'outline={o.oid} cooled {n} sibling(s) within '
                      f'{SIBLING_COOL_R:.1f}m')

    def _visited_recent(self, x: float, y: float, t: float) -> bool:
        """Was this viewpoint position already stood at (and not yet due for
        the 2nd-sweep revisit)? Persistent viewpoint memory."""
        for vx, vy, vt in self.visited_views:
            if math.hypot(vx - x, vy - y) <= VIEWPOINT_VISIT_R \
                    and t - vt < VIEWPOINT_REVISIT_S:
                return True
        return False

    def _mark_view_visited(self, x: float, y: float, t: float):
        self.visited_views.append((float(x), float(y), float(t)))
        if len(self.visited_views) > 200:
            del self.visited_views[:-200]

    def _clear_visit_memory(self):
        """2nd sweep 'just in case': fresh pass may re-try everything."""
        n = self.board_mem.clear_cooldowns()
        if self.visited_views:
            self._log('requeue',
                      f'{len(self.visited_views)} viewpoint visit(s) cleared')
            self.visited_views = []
        return n

    def _approach_failed(self):
        o = self.approach_cluster
        if o is not None:
            o.fails += 1
            if o.fails >= APPROACH_FAILS_MAX:
                o.verdict = 'unread'
                o.cool_until = _now_sec(self) + OUTLINE_COOL_S
                self._cool_siblings(o, _now_sec(self))
            self.get_logger().warn(
                f'approach failed for outline={o.oid} id={o.board_id} '
                f'(fails={o.fails})')
            self._log('approach_fail',
                      f'outline={o.oid} id={o.board_id} fails={o.fails}')
        self.approach_cluster = None
        self._enter('SEARCHING', 'approach failed')
        if self.mode is None:
            self._begin_ring() if self.current_target else self._begin_tour()

    def _begin_reading(self, o: Outline, mid: Optional[int]):
        self.reading_cluster = o
        self.reading_id = mid
        self.reading_t0 = _now_sec(self)
        self.reading_samples = []
        self.reading_adj = 0
        self.reading_hold_until = 0.0
        self._enter('READING', f'outline={o.oid} id={mid}')

    def _reset_vp(self):
        self.vp_idx = 0
        self.at_vp = False
        self.stare_step = 0
        self.pause_until = 0.0

    def _begin_ring(self):
        self._cancel_goal()
        if self.current_target is None:
            self._begin_tour()
            return
        cx, cy = self.current_target
        pts = []
        for x, y, yaw in search.ring_viewpoints(cx, cy):
            s = self._snap(x, y, max_r=1.2)
            pts.append((s[0], s[1], yaw))
        self.viewpoints = pts
        self.mode = 'ring'
        self.plan_t = _now_sec(self)
        self._reset_vp()
        self.get_logger().info(
            f'ring: {len(pts)} viewpoints around {cx:.1f},{cy:.1f}')

    def _begin_tour(self):
        self._cancel_goal()
        start = self.robot_xy or (0.0, 0.0)
        self.viewpoints = search.order_nearest(self.tour, start)
        self.mode = 'tour'
        self.plan_t = _now_sec(self)
        self._reset_vp()
        self.get_logger().info(
            f'tour: {len(self.viewpoints)} viewpoints (restarts={self.tour_restarts})')

    def _begin_colour(self):
        self._cancel_goal()
        self.mode = 'colour'
        self.colour_rounds += 1
        self._reset_vp()
        self.viewpoints = []
        if self.robot_xy is None or not self.pillar_xys or self.grid is None:
            self.get_logger().warn('colour mission impossible (map/robot missing)')
            self._advance_mode('colour impossible')
            return
        # One pillar at a time, nearest unconfirmed first, 3 facing poses
        # each: close + facing beats far tour viewpoints whose patches are
        # background-dominated. Only 3 pillars, so this is bounded.
        confirmed = set(self.colour_map.values()) if self.colour_map else set()
        todo = sorted(
            (i for i in range(len(self.pillar_xys)) if i not in confirmed),
            key=lambda i: math.hypot(self.pillar_xys[i][0] - self.robot_xy[0],
                                     self.pillar_xys[i][1] - self.robot_xy[1]))
        vps = []
        for idx in todo:
            vps.extend(pillars.pillar_ring_viewpoints(
                self.pillar_xys[idx], self.grid))
        if self.colour_rounds > 1 and len(vps) > 2:
            sh = ((self.colour_rounds - 1) * 2) % len(vps)
            vps = vps[sh:] + vps[:sh]
        if not vps:
            self.get_logger().warn('no viewpoint sees all pillars - skipping colour step')
            self._advance_mode('no colour viewpoint')
            return
        self.viewpoints = vps
        self.plan_t = _now_sec(self)
        self.get_logger().info(
            f'colour mission round {self.colour_rounds}: {len(vps)} viewpoint(s)')

    def _advance_mode(self, why: str):
        self.get_logger().info(f'search level done: {self.mode} ({why})')
        self._log('ladder', f'{self.mode} done: {why}')
        if self.mode == 'ring':
            self._begin_tour()
        elif self.mode == 'colour':
            if self.deferred_clue is not None and self.colour_map is None:
                self.get_logger().warn(
                    'colours still unresolved - arena tour; will resume if colours arrive')
            if self.current_target is not None:
                self._begin_ring()
            else:
                self._begin_tour()
        elif self.mode == 'tour':
            if self.deferred_clue is not None and self.colour_map is None \
                    and self.colour_rounds < COLOUR_ROUNDS_MAX:
                self._begin_colour()           # another colour round first
                return
            if self.tour_restarts < 1:
                self.tour_restarts += 1
                self.tour_pass += 1
                n = self._clear_visit_memory()
                nh = self.board_mem.hygiene(_now_sec(self))
                if n or nh:
                    self._log('requeue', f'{n} cooldown(s) cleared, '
                                         f'{nh} ghost(s) dropped')
                    self.get_logger().info(
                        f'cleared {n} cooldown(s), dropped {nh} ghost(s)')
                self._begin_tour()
            else:
                self.mode = 'idle'
                self.idle_since = _now_sec(self)
                self.viewpoints = []
                self.get_logger().error(
                    'search ladder exhausted - idling safely (never publish wrong clues)')
                self._log('ladder', 'exhausted -> idle')

    # ---------------------------------------------------------------- tick
    def tick(self):
        t = _now_sec(self)
        if t - self.last_status_pub >= 1.0:
            status = 'DONE' if self.state == 'DONE' else \
                'READING' if self.state == 'READING' else \
                'MOVING' if self.goal is not None or self.state in ('APPROACH', 'MOVING') \
                else 'SEARCHING'
            self.status = status
            msg = String()
            msg.data = self.status
            self.status_pub.publish(msg)
            self.last_status_pub = t

        if t - self._last_health >= 5.0:
            # STARTUP/IDLE observability: why-standing answers itself here
            self._last_health = t
            if self._t0_wall is None:
                self._t0_wall, self._t0_sim = time.time(), t
            rtf = (t - self._t0_sim) / max(time.time() - self._t0_wall, 1e-9)
            pose = ('%.2f,%.2f' % self.robot_xy) if self.robot_xy else 'None'
            self._log('health',
                      f'rtf={rtf:.2f} img={self._img_n} tf_ok={self._tf_ok} '
                      f'tf_fail={self._tf_fail} scan={self._scan_n} '
                      f'pose={pose} nav_ready={self._nav_ready()} '
                      f'stack_ready={self.stack_ready} '
                      f'outlines={len(self.board_mem.outlines)}')
            self.get_logger().info(
                f'health state={self.state} rtf={rtf:.2f} img={self._img_n} '
                f'tf_ok={self._tf_ok} tf_fail={self._tf_fail} '
                f'scan={self._scan_n} pose={pose} '
                f'nav_ready={self._nav_ready()} stack_ready={self.stack_ready} '
                f'outlines={len(self.board_mem.outlines)}',
                throttle_duration_sec=5.0)
            # stream stall: sim time flows but a sensor stopped arriving
            # (the 14:31 run: img frozen at 14 for 85 s = dead camera bridge)
            if self._last_stream_t is None:
                self._last_stream_t, self._last_img_n = t, self._img_n
                self._last_scan_n = self._scan_n
            elif t - self._last_stream_t >= 10.0:
                dt = t - self._last_stream_t
                for name, now, prev in (('camera', self._img_n, self._last_img_n),
                                        ('scan', self._scan_n, self._last_scan_n)):
                    if now == prev:
                        self._log('stream_stall', f'{name} frozen {dt:.0f}s')
                        self.get_logger().error(
                            f'{name} stream dead {dt:.0f}s - restart sim/bridge, '
                            'no perception is possible',
                            throttle_duration_sec=30.0)
                self._last_stream_t, self._last_img_n = t, self._img_n
                self._last_scan_n = self._scan_n
            if rtf < 0.5:
                self.get_logger().error(
                    f'sim RTF {rtf:.2f} - time itself is crawling, relaunch '
                    'headless (rviz:=false), kill duplicate stacks',
                    throttle_duration_sec=30.0)

        if self.state != 'DONE' and t - self.last_progress_t > WATCHDOG_S:
            # No accepted clue for ages: clear outline cooldowns and force a
            # fresh coverage tour (fresh ground finds what the memory missed).
            self.last_progress_t = t
            n = self.board_mem.clear_cooldowns()
            nh = self.board_mem.hygiene(t)
            self.tour_restarts = 0
            self.colour_rounds = 0
            self._log('watchdog',
                      f'no progress - cleared {n} cooldown(s), dropped {nh}, '
                      'tours/colour reset')
            self.get_logger().warn(
                f'watchdog: no clue for {WATCHDOG_S:.0f}s - cleared {n}, '
                f'dropped {nh}, tours/colour reset')
            if self.validator.accepted == 0 and self.state == 'SEARCHING':
                self._begin_tour()

        poll = self._poll_goal()
        # NOTE: no early-spot preemption by design. Camera detections only
        # update the outline memory (facing/id); a committed drive is never
        # cancelled for a sighting - new tasks start when idle. This is what
        # kills the bounce loop.

        if self.state == 'STARTUP':
            self._tick_startup(t, poll)
        elif self.state == 'APPROACH':
            self._tick_approach(t, poll)
        elif self.state == 'READING':
            self._tick_reading(t)
        elif self.state == 'MOVING':
            self._tick_moving(poll)
        elif self.state == 'SEARCHING':
            self._tick_searching(t, poll)

    def _startup_root_outline(self):
        """Outline for the spawn-visible root QR (peek only, no chain advance).

        Returns the outline holding a hint whose token == START_TOKEN, else
        None. Pure parse + token compare - never validator.validate()."""
        from .clue import START_TOKEN, parse_clue
        for _mid, text in list(self._qr_hint.items()):
            try:
                clue = parse_clue(text)
            except Exception:                      # noqa: BLE001
                continue
            if clue is None or clue.token != START_TOKEN:
                continue
            # outline already tagged with this id, else nearest recent
            # detection of it (no validator.validate: peek only, no advance)
            best, best_d = None, 1.5
            for cand in self.board_mem.outlines:
                if cand.board_id is not None and cand.board_id != clue.id:
                    continue
                for did, dx, dy, _dt in reversed(self._recent_dets):
                    if did != clue.id:
                        continue
                    d = math.hypot(cand.x - dx, cand.y - dy)
                    if d < best_d:
                        best, best_d = cand, d
                        break
                if best is not None and best_d < 0.45:
                    break
            if best is not None:
                return best
        return None

    def _tick_startup(self, t, poll=None):
        # L0: the root board is usually visible in the very first frame, so
        # stare for its QR first; creep 1 m straight ahead for a closer read;
        # only then start the main tour loop. Never tour blind (no TF / no
        # image / dead stack burns the ladder in seconds).
        if poll is not None and poll[0] == 'startup_creep':
            _purpose, outcome = poll
            self._startup_creep_done = True
            self._startup_creep_settle_until = t + STARTUP_CREEP_SETTLE_S
            self._log('startup', f'creep {outcome} - settle for closer read')
        need_pose = self.robot_xy is None
        if (not self._nav_ready() or need_pose) and t - self.t0 < STARTUP_NAV_WAIT_S:
            return
        if need_pose and t - self.t0 >= STARTUP_NAV_WAIT_S:
            self.get_logger().warn(
                'no TF pose after 30 s - proceeding anyway (degraded)',
                throttle_duration_sec=30.0)
        if self._img_n == 0:
            self.get_logger().warn('startup: waiting for first camera frame',
                                    throttle_duration_sec=5.0)
            return                             # dead bridge: don't tour blind
        # 1. root already decoded in place? go CHECK it now.
        o = self._startup_root_outline()
        if o is not None:
            self._log('startup', f'root hint id={o.board_id} outline={o.oid} - L0 CHECK')
            self._begin_approach(o)
            return
        # 2. stare L0 window for the spawn view to decode.
        if t - self.t0 < STARTUP_L0_WAIT_S:
            return
        # 3. creep 1 m straight ahead (spawn faces +X) for a closer read.
        if not self._startup_creep_sent:
            if self.robot_xy is None:
                return
            if self.goal is not None:
                return                         # stack busy, wait
            cx = self.robot_xy[0] + math.cos(self.robot_yaw) * STARTUP_CREEP_M
            cy = self.robot_xy[1] + math.sin(self.robot_yaw) * STARTUP_CREEP_M
            sx, sy = self._snap(cx, cy, max_r=1.0)
            if self._send_nav(sx, sy, self.robot_yaw, 'startup_creep',
                              timeout=TRANSIT_TIMEOUT_S):
                self._startup_creep_sent = True
                self._log('startup', f'creep to {sx:.2f},{sy:.2f} for closer read')
            return
        if not self._startup_creep_done:
            return                             # creep driving...
        if t < self._startup_creep_settle_until:
            # settle window: a fresh hint may still arrive; re-check it.
            o = self._startup_root_outline()
            if o is not None:
                self._log('startup', f'root hint after creep outline={o.oid} - L0 CHECK')
                self._begin_approach(o)
            return
        # 4. still nothing: main loop.
        self._enter('SEARCHING', 'startup coverage tour')
        self._begin_tour()

    def _tick_approach(self, t: float, poll):
        if self.approach_cluster is None:
            self._enter('SEARCHING', 'no approach target')
            return
        if t < self.approach_backoff_until:
            return                                     # waiting out a rejection/backoff
        if poll is not None and poll[0] == 'approach':
            _purpose, outcome = poll
            if outcome == 'ok':
                o = self.approach_cluster
                # Enter READING when the board is readable NOW: a fresh
                # sighting, or geometric visibility from the arrived pose.
                # (Freshness alone fails systematically: the drive itself yaws
                # the camera off the board, so arrivals are always "stale"
                # even when parked face-on 1.4 m away.)
                seen_id = o.board_id
                fresh = outlines.fresh_near(self._recent_dets, seen_id,
                                            o.x, o.y, t)
                sees_now = self.robot_xy is not None and \
                    search.pose_sees_point(self.robot_xy[0], self.robot_xy[1],
                                           self.robot_yaw, o.x, o.y) and \
                    self._los_clear(self.robot_xy[0], self.robot_xy[1],
                                    o.x, o.y)
                if fresh or sees_now:
                    self._begin_reading(o, seen_id)
                    return
                self.get_logger().warn(
                    f'arrived but outline={o.oid} marker not seen - other side')
                self._log('approach_blind',
                          f'outline={o.oid} id={o.board_id} '
                          f'marker_seen={t - o.marker_seen:.1f}s ago')
                if not self._try_other_side(o):
                    self._approach_failed()
                return
            elif outcome == 'rejected':
                # Never burn a candidate on an instant rejection: either Nav2
                # is not ready (wait, keep idx) or the pose was refused (widen).
                self.approach_backoff_until = t + APPROACH_RESEND_S
                if not self._nav_ready() or self._stack_held(t):
                    return
                self.approach_idx += 1
            else:                                      # genuine nav failure
                self.approach_idx += 1
        elif poll is not None:
            return                                     # unrelated goal; wait
        if self.goal is not None:
            return                                     # still driving to a candidate
        if self.approach_idx >= len(self.approach_cands):
            self._approach_failed()
            return
        if self._send_approach_goal():
            return
        if self._stack_held(t) or not self._nav_ready():
            self.approach_backoff_until = t + APPROACH_RESEND_S
            return                                     # wait, keep candidate
        self._approach_failed()      # ready, but no candidate can see the board

    def _frames_required(self) -> int:
        """Stationary frames for finalize: more when localization is shaky
        (sitting still lets AMCL converge, and the median needs the samples)."""
        if self.amcl_std is None:
            return READING_BASE_N
        return min(READING_BASE_N + int(self.amcl_std * 40.0), READING_MAX_N)

    def _tick_reading(self, t):
        cl = self.reading_cluster
        if cl is None:
            self._enter('SEARCHING', 'reading without cluster')
            return
        texts = [s[3] for s in self.reading_samples if s[3]]
        n = len(self.reading_samples)
        n_req = self._frames_required()
        enough = (len(texts) >= 1 and n >= n_req) or \
                 (len(texts) >= 3 and n >= 5 and t - self.reading_t0 >= READING_DONE_S)
        if enough:
            self._finalize_reading()
            return
        if t - self.reading_t0 <= READING_HARD_TIMEOUT_S:
            # Closed loop: board IS visible (fresh sightings) but nothing
            # decodes - nudge yaw in place instead of staring at a bad angle
            # for the full 4.5 s. Samples taken while turning are ignored.
            if (not texts and n >= 5
                    and t - self.reading_t0 >= READING_SETTLE_S + 1.2
                    and self.reading_adj < READING_ADJ_MAX
                    and t >= self.reading_hold_until
                    and outlines.fresh_near(self._recent_dets, cl.board_id,
                                           cl.x, cl.y, t)):
                self.reading_adj += 1
                step = READING_ADJ_YAW if self.reading_adj == 1 \
                    else -READING_ADJ_YAW
                self.reading_hold_until = t + READING_ADJ_HOLD_S
                self._send_spin(step)
                self._log('read_adjust',
                          f'outline={cl.oid} id={cl.board_id} '
                          f'nudge={self.reading_adj} '
                          f'{math.degrees(step):+.0f}deg')
                return
            # Early give-up: plenty of frames, zero decodes - retry now
            # instead of burning the whole hard timeout.
            if not (not texts and n >= READING_EARLY_N
                    and t - self.reading_t0 >= READING_EARLY_S):
                return
        if texts:
            self._finalize_reading()
            return
        self.reading_attempts += 1
        if self.reading_attempts < READ_ATTEMPTS_MAX:
            self.get_logger().warn('no QR decode - re-approaching once')
            self._log('read_retry',
                      f'outline={cl.oid} id={cl.board_id} '
                      f'attempt={self.reading_attempts}')
            self.reading_samples = []
            self.reading_t0 = t
            # Retry the SAME side with the latest facing info (a blind arrival
            # already tried the other side, if there is one).
            self.approach_board_xy = (cl.x, cl.y)
            normal = outlines.mean_yaw(cl.n_yaws)
            if normal is None:
                normal = self.approach_normal if self.approach_normal is not None else 0.0
            self.approach_normal = normal
            self.approach_cluster = cl
            self.approach_cands = self._build_approach_cands(
                cl, self.check_side, normal)
            self.approach_idx = 0
            self._enter('APPROACH', 'read retry')
            if self._send_approach_goal() or not self._nav_ready() \
                    or self._stack_held(t):
                return
        cl.verdict = 'unread'
        cl.cool_until = t + OUTLINE_COOL_S
        self._cool_siblings(cl, t)
        self.get_logger().warn(
            f'no QR after {READ_ATTEMPTS_MAX} attempts - outline={cl.oid} '
            f'cooling down')
        self._log('read_fail', f'outline={cl.oid} id={cl.board_id}')
        self.reading_cluster = None
        self.reading_attempts = 0
        self._enter('SEARCHING', 'unread')

    def _finalize_reading(self):
        o = self.reading_cluster
        samples = self.reading_samples
        xs = [s[0] for s in samples]
        ys = [s[1] for s in samples]
        yaws = [s[2] for s in samples if not math.isnan(s[2])]
        texts = [s[3] for s in samples if s[3]]
        text = Counter(texts).most_common(1)[0][0]
        mx, my = float(np.median(xs)), float(np.median(ys))
        if yaws:
            myaw = math.atan2(float(np.mean([math.sin(a) for a in yaws])),
                              float(np.mean([math.cos(a) for a in yaws])))
        else:
            myaw = 0.0
        verdict, clue = self.validator.validate(text)
        # measurement geometry for the report: robot pose, incidence of the
        # viewing ray vs the board face, yaw spread, localization health
        inc_deg: float = float('nan')
        if o.axis is not None and self.robot_xy is not None:
            _nx, _ny = outlines.normal_vec(
                o.axis, o.facing if o.facing != 0 else self.check_side)
            _nyaw = math.atan2(_ny, _nx)
            inc_deg = math.degrees(search.angular_diff(
                math.atan2(self.robot_xy[1] - o.y, self.robot_xy[0] - o.x),
                _nyaw))
        spread_deg = float('nan')
        if yaws:
            _rs = sum(math.sin(a) for a in yaws) / len(yaws)
            _rc = sum(math.cos(a) for a in yaws) / len(yaws)
            _r = math.hypot(_rs, _rc)
            if _r > 1e-9:
                spread_deg = math.degrees(math.sqrt(max(-2.0 * math.log(_r), 0.0)))
        amcl_s = f'{self.amcl_std:.3f}' if self.amcl_std is not None else '?'
        if self.amcl_std is not None and self.amcl_std > AMCL_WARN_STD:
            self.get_logger().warn(
                f'localization std {amcl_s} m at finalize (outline={o.oid})',
                throttle_duration_sec=10.0)
        self._log('reading', f'outline={o.oid} id={clue.id if clue else "?"} '
                             f'n={len(samples)} verdict={verdict.value} text={text!r} '
                             f'inc={inc_deg:.0f}deg yawspread={spread_deg:.0f}deg '
                             f'amcl={amcl_s} n_req={self._frames_required()}')
        self.get_logger().info(
            f'READING outline={o.oid} -> {verdict.value}  {text!r} '
            f'({len(samples)} frames, med {mx:.2f},{my:.2f})')
        self.reading_cluster = None
        self.reading_attempts = 0

        if verdict is not Verdict.VALID:
            if clue is None:
                o.verdict = 'unread'
            elif verdict is Verdict.LOOKALIKE:
                o.verdict = 'rejected'       # right id, wrong token: definitive
                o.board_id = clue.id
            elif verdict is Verdict.DECOY:
                o.board_id = clue.id
                exp = self.validator.expected_id()
                if self.validator.accepted == 0 or \
                        (exp is not None and clue.id > exp):
                    o.verdict = 'parked'     # future clue: revisit when expected
                    o.parked_id = clue.id
                else:
                    o.verdict = 'rejected'   # already-passed id: done
            else:
                o.verdict = 'unread'
            o.cool_until = _now_sec(self) + OUTLINE_COOL_S
            self._cool_siblings(o, _now_sec(self))
            self.get_logger().warn(f'ignored {verdict.value}: {text!r}')
            self._enter('SEARCHING', f'rejected {verdict.value}')
            if self.mode is None:
                self._begin_ring() if self.current_target else self._begin_tour()
            return

        o.verdict = 'valid'
        o.board_id = clue.id
        self.deferred_clue = None       # any accepted clue supersedes an old one
        self.last_board = ((mx, my), myaw)
        self.last_progress_t = _now_sec(self)
        self._clear_visit_memory()   # parked-with-expected is due NOW + 2nd sweep reset
        nh = self.board_mem.hygiene(_now_sec(self))
        if nh:
            self._log('hygiene', f'dropped {nh} ghost outline(s)')
        assert clue is not None
        self.clue_pub.publish(String(data=clue.raw))
        self.board_pub.publish(String(data=f'{clue.id} {mx:.3f} {my:.3f}'))
        self._log('publish_clue', clue.raw)
        self.get_logger().info(f'PUBLISHED clue #{self.validator.accepted}: {clue.raw}')

        if clue.treasure:
            tx, ty = rel_from_board(self.last_board, clue.args[0], clue.args[1])
            sx, sy = self._snap(tx, ty, max_r=search.TREASURE_SNAP_M)
            self.last_treasure = (tx, ty)
            if self._send_nav(sx, sy, 0.0, 'treasure', meta=(tx, ty),
                              timeout=TRANSIT_TIMEOUT_S):
                self._enter('MOVING', f'treasure {tx:.2f},{ty:.2f}')
                return
            # stack held: never lose the treasure task - retry from SEARCHING
            self.pending_treasure = (tx, ty)
            self._enter('SEARCHING', 'treasure held - stack down')
            if self.mode is None:
                self._begin_ring() if self.current_target else self._begin_tour()
            return
        if clue.verb == 'PILLAR' and self.colour_map is None:
            self.deferred_clue = clue
            self.current_target = None
            self._enter('SEARCHING', 'need pillar colours first')
            self._begin_colour()
            return
        if clue.verb == 'BETWEEN' and self.colour_map is None:
            self.deferred_clue = clue
            self.current_target = None
            self._enter('SEARCHING', 'need pillar colours first')
            self._begin_colour()
            return
        target = self._resolve(clue)
        if target is None:
            self.get_logger().error('cannot resolve command - searching arena')
            self.deferred_clue = clue
            self.current_target = None
            self._enter('SEARCHING', 'unresolvable target')
            self._begin_colour()
            return
        if clue.verb == 'PILLAR':
            # pillar centre is a search REGION, never a goal (final_plan 4.4)
            self.current_target = target
            self._enter('SEARCHING', f'PILLAR region {target[0]:.2f},{target[1]:.2f}')
            self._begin_ring()
            return
        sx, sy = self._snap(target[0], target[1])
        self.current_target = (sx, sy)
        if self._send_nav(sx, sy, 0.0, 'transit', timeout=TRANSIT_TIMEOUT_S):
            self._enter('MOVING', f'{clue.verb} -> {sx:.2f},{sy:.2f}')
        # else held: current_target kept so the ladder stages toward it

    def _tick_moving(self, poll):
        if poll is None:
            return
        purpose, outcome = poll
        if purpose == 'treasure':
            if outcome != 'ok':
                self.get_logger().warn(
                    'treasure approach did not succeed - publishing computed point anyway')
            assert self.last_treasure is not None
            self._publish_treasure(*self.last_treasure)
            self._enter('DONE', f'treasure {outcome}')
            return
        if purpose == 'transit':
            if outcome == 'ok':
                self._enter('SEARCHING', f'arrived {self.current_target}')
            else:
                self.get_logger().warn(
                    'transit failed - searching around the target anyway')
                self._enter('SEARCHING', 'transit failed')
            self._begin_ring() if self.current_target else self._begin_tour()

    def _publish_treasure(self, x: float, y: float):
        msg = PoseStamped()
        msg.header.frame_id = MAP_FRAME
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.position.x, msg.pose.position.y = float(x), float(y)
        msg.pose.orientation.w = 1.0
        self.treasure_pub.publish(msg)
        self._log('publish_treasure', f'{x:.3f},{y:.3f}')
        self.get_logger().info(f'treasure published at {x:.3f},{y:.3f}')

    def _select_board_task(self, t: float, expected: Optional[int]) -> bool:
        """Pick one board CHECK when idle. Returns True if a task started.

        The user's algorithm, exactly: explore (tour) until a hint; CHECK an
        outline the camera corroborated (a real marker was seen there) or the
        expected id; both sides with facing; go back to explored outlines when
        due (cooldowns); otherwise keep exploring. Pure lidar ghosts (wall
        corners, the follower - never camera-seen) are mapped, never chased.
        A committed task is never preempted.
        """
        if t < self._step2_cool or self.robot_xy is None:
            return False
        if self.pending_treasure is not None:
            return False                       # deliver treasure first
        rx, ry = self.robot_xy
        match = [o for o in self.board_mem.candidates(t)
                 if outlines.eligible(o, expected)]
        if not match:
            return False
        if expected is None:
            o = min(match,
                    key=lambda o: outlines.pre_anchor_key(o, (rx, ry)))
        else:
            o = min(match,
                    key=lambda o: outlines.target_key(o, expected, (rx, ry)))
        d = math.hypot(o.x - rx, o.y - ry)
        if d > search.SEARCH_RADIUS + 1.0:
            if self.current_target is None or math.hypot(
                    self.current_target[0] - o.x,
                    self.current_target[1] - o.y) > 0.5:
                sx, sy = self._snap(o.x, o.y)
                self.current_target = (sx, sy)
                if self._send_nav(sx, sy, 0.0, 'transit',
                                  timeout=TRANSIT_TIMEOUT_S):
                    self._enter('MOVING',
                                f'outline={o.oid} id={o.board_id} '
                                f'-> {sx:.2f},{sy:.2f}')
                    return True
                # refused (stack hold): fall through to the ladder, which
                # stages toward current_target via ring - never return here
                # claiming motion that isn't happening (that livelock froze a
                # whole run: returns with no goal, no sends, forever).
            elif self.goal is not None:
                return True                # genuinely heading there
            # else: target set but no goal in flight -> fall through to ladder
        self._begin_approach(o)
        if self.approach_cluster is None and self.goal is None:
            self._step2_cool = t + 5.0     # instant fail: don't hot-loop
        return True

    def _tick_searching(self, t, poll):
        if self.goal is None and t < self.vp_backoff_until:
            return                         # holding sends during stack outage
        # 0. undelivered treasure first: the chain is complete, just deliver
        if self.pending_treasure is not None and self.goal is None:
            tx, ty = self.pending_treasure
            sx, sy = self._snap(tx, ty, max_r=search.TREASURE_SNAP_M)
            if self._send_nav(sx, sy, 0.0, 'treasure', meta=(tx, ty),
                              timeout=TRANSIT_TIMEOUT_S):
                self.pending_treasure = None
                self._enter('MOVING', f'treasure {tx:.2f},{ty:.2f} (retry)')
            return
        # 1. deferred PILLAR/BETWEEN target once colours arrive
        if self.deferred_clue is not None and self.colour_map is not None:
            clue = self.deferred_clue
            target = self._resolve(clue)
            if target is None:
                self.deferred_clue = None      # unresolvable: drop (as before)
            else:
                if clue.verb == 'PILLAR':
                    self.deferred_clue = None
                    self.current_target = target
                    self._enter('SEARCHING', 'PILLAR region (colours ready)')
                    self._begin_ring()
                    return
                sx, sy = self._snap(target[0], target[1])
                self.current_target = (sx, sy)
                if self._send_nav(sx, sy, 0.0, 'transit',
                                  timeout=TRANSIT_TIMEOUT_S):
                    self.deferred_clue = None
                    self._enter('MOVING',
                                f'{clue.verb} (colours ready) -> {sx:.2f},{sy:.2f}')
                return                         # held: deferred kept, retry

        # 2. task selection (one committed task at a time; new tasks only
        # when idle - never preempt a driving goal).
        expected = self.validator.expected_id()
        if self._select_board_task(t, expected):
            return

        if self.mode == 'idle':
            # idle is a rest stop, not terminal: fresh coverage pass every
            # minute (the 02:49 run proved a boot-time idle otherwise parks
            # the robot until the 240 s watchdog).
            if t - self.idle_since > IDLE_RETRY_S:
                self._log('ladder', 'idle retry - fresh coverage pass')
                self.get_logger().info('idle retry - fresh coverage pass')
                self._clear_visit_memory()
                self._begin_tour()
            return
        if self.mode is None:
            self._begin_ring() if self.current_target else self._begin_tour()
            if self.mode is None:
                return

        # 3. level budgets
        if self.mode == 'ring' and t - self.plan_t > RING_BUDGET_S:
            self._advance_mode('ring budget exceeded')
            return
        if self.mode == 'tour' and t - self.plan_t > TOUR_BUDGET_S:
            self._advance_mode('tour budget exceeded')
            return

        # 4. goal outcomes for viewpoint motion / staring
        if poll is not None:
            purpose, outcome = poll
            if purpose == 'viewpoint':
                if outcome == 'ok':
                    self.at_vp = True
                    self.stare_step = 0
                    if self.vp_idx < len(self.viewpoints):
                        vx, vy, _vyaw = self.viewpoints[self.vp_idx]
                        self._mark_view_visited(vx, vy, t)
                elif outcome == 'rejected' and (
                        not self._nav_ready() or self._stack_held(t)):
                    # Nav2 down: keep the viewpoint, wait (else the whole tour
                    # burns in seconds on instant rejections).
                    self.vp_backoff_until = t + VIEWPOINT_RESEND_S
                else:
                    self.vp_idx += 1
            elif purpose == 'spin':
                self.stare_step += 1
                if outcome == 'ok':
                    self.pause_until = t + STARE_PAUSE_S
        if self.goal is not None:
            return                                     # still driving / spinning

        # 5. viewpoint stepping
        if self.vp_idx >= len(self.viewpoints):
            if self.mode == 'colour' and self.colour_map is None:
                self._try_resolve_colours()
                if self.colour_map is None:
                    if self.colour_rounds < COLOUR_ROUNDS_MAX:
                        self._begin_colour()   # next round, rotated viewpoints
                    else:
                        self._advance_mode('colour viewpoints exhausted')
                    return
                # resolved: step 1 picks up deferred_clue next tick
                return
            self._advance_mode('viewpoints exhausted')
            return
        if t < self.vp_backoff_until:
            return                                     # waiting out Nav2 outage
        # Skip viewpoints already stood at this sweep: persistent viewpoint
        # memory. Cleared on tour restart / valid clue, so the 2nd sweep still
        # goes back "just in case".
        while self.vp_idx < len(self.viewpoints) and not self.at_vp:
            wx, wy, _wyaw = self.viewpoints[self.vp_idx]
            if self._visited_recent(wx, wy, t):
                self._log('vp_skip_visited', f'vp={self.vp_idx} {wx:.2f},{wy:.2f}')
                self.vp_idx += 1
                continue
            break
        if self.vp_idx >= len(self.viewpoints):
            self._advance_mode('viewpoints exhausted (all visited)')
            return
        if not self.at_vp:
            x, y, yaw = self.viewpoints[self.vp_idx]
            sx, sy = self._snap(x, y, max_r=1.2)
            if self.mode == 'colour' and self.pillar_xys:
                # face the nearest pillar: tour yaws aim at the arena centre,
                # which can leave every pillar out of the camera frustum
                px, py, _r = min(self.pillar_xys,
                                 key=lambda p: (p[0] - sx) ** 2 + (p[1] - sy) ** 2)
                yaw = math.atan2(py - sy, px - sx)
            self._send_nav(sx, sy, yaw, 'viewpoint', timeout=VIEWPOINT_TIMEOUT_S)
            return
        if t < self.pause_until:
            return
        if self.stare_step < STARE_STEPS:
            # spins need swing room (the trailer follows 1.3 m behind): skip
            # staring in tight pockets instead of sweeping into walls/follower
            if self.robot_xy is not None and self.clear_dist is not None \
                    and self.grid is not None and search.clearance_at(
                        self.clear_dist, self.grid,
                        self.robot_xy[0], self.robot_xy[1]) < SPIN_CLEAR_MIN:
                self._log('spin_skip',
                          f'vp={self.vp_idx} tight pocket - staring skipped')
                self.vp_idx += 1
                self.at_vp = False
                return
            self._send_spin()
            return
        self.vp_idx += 1
        self.at_vp = False


def main():
    rclpy.init()
    node = HuntNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
