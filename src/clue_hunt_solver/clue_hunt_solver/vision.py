"""ArUco + QR perception for the leader (final_plan.md §4.2).

Pure OpenCV (no ROS imports) so it is unit-testable against the board model
textures. Physical constants were measured from
`clue_hunt_gazebo/models/board_practice_*/meshes/*.png` + the face mesh:

  * face mesh 0.60 x 0.30 m, texture 1024x512 -> marker side 409 px = 0.24 m
  * QR is 0.97 marker sides, centred at marker-frame (+0.30, 0.00) m
    (right of the marker; board point (0, +0.30, 0) after the axis
    permutation in frames.marker_to_board)
  * ArUco id == QR id on every practice texture, including the look-alike
    (both "4") -> ranking hint only, never a filter (§2.1)
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

DICT_ID = cv2.aruco.DICT_4X4_50
IGNORE_MARKER_IDS = frozenset({49})       # follower tag
MARKER_SIZE = 0.24                        # m, physical marker side
QR_OFFSET = (0.30, 0.0)                   # m, marker frame (x right, y up), z = 0
QR_SIZE = 0.234                           # m, measured from textures
DEFAULT_VIEW_DISTANCE = 1.5               # m; measured in phase_2 Step 3: 99% decode at 1.0-2.0 m
REPROJ_MAX_PX = 3.0                       # reprojection gate
QR_QUIET_MARGIN = 1.4                     # crop expansion around the projected QR
QR_PAD_PX = 24                            # white quiet-zone padding after warp


@dataclass
class Detection:
    marker_id: int
    corners: np.ndarray                   # (4, 2) image points, TL TR BR BL
    rvec: np.ndarray                      # (3, 1)
    tvec: np.ndarray                      # (3, 1) camera optical frame
    reproj_err: float                     # px

    @property
    def centre_px(self) -> Tuple[float, float]:
        m = self.corners.reshape(-1, 2).mean(axis=0)
        return (float(m[0]), float(m[1]))

    @property
    def distance(self) -> float:
        t = self.tvec.reshape(3)
        return float(np.linalg.norm(t))

    def normal_in_camera(self) -> np.ndarray:
        """Out-of-face normal in the camera optical frame. For a detection the
        face is visible, so its z component is negative (points at the camera)."""
        R, _ = cv2.Rodrigues(self.rvec.reshape(3, 1))
        return R @ np.array([0.0, 0.0, 1.0])


_detector_cache = {}


def get_dictionary():
    if 'dict' not in _detector_cache:
        _detector_cache['dict'] = cv2.aruco.getPredefinedDictionary(DICT_ID)
    return _detector_cache['dict']


def get_detector():
    if 'det' not in _detector_cache:
        params = cv2.aruco.DetectorParameters()
        for attr, value in (('cornerRefinementMethod', cv2.aruco.CORNER_REFINE_SUBPIX),
                            ('doCornerRefinement', True)):
            try:
                setattr(params, attr, value)
            except (AttributeError, TypeError):
                pass
        _detector_cache['det'] = cv2.aruco.ArucoDetector(get_dictionary(), params)
    return _detector_cache['det']


def marker_object_points(marker_size: float = MARKER_SIZE) -> np.ndarray:
    """IPPE_SQUARE object points, order TL TR BR BL (marker frame: x right, y up)."""
    h = marker_size / 2.0
    return np.array([[-h, h, 0.0], [h, h, 0.0],
                     [h, -h, 0.0], [-h, -h, 0.0]], dtype=np.float64)


def qr_object_points(qr_offset: Sequence[float] = QR_OFFSET,
                     qr_size: float = QR_SIZE,
                     margin: float = 1.0) -> np.ndarray:
    """QR square corners (TL TR BR BL) in the marker frame."""
    ox, oy = qr_offset
    h = qr_size * margin / 2.0
    return np.array([[ox - h, oy + h, 0.0], [ox + h, oy + h, 0.0],
                     [ox + h, oy - h, 0.0], [ox - h, oy - h, 0.0]], dtype=np.float64)


def _norm_dist(dist) -> Optional[np.ndarray]:
    if dist is None:
        return None
    d = np.asarray(dist, dtype=np.float64).reshape(-1)
    return d if d.size else None


def project_points(obj_pts: np.ndarray, rvec, tvec, camera_matrix, dist) -> np.ndarray:
    pts, _ = cv2.projectPoints(np.asarray(obj_pts, dtype=np.float64).reshape(-1, 1, 3),
                               np.asarray(rvec, dtype=np.float64).reshape(3, 1),
                               np.asarray(tvec, dtype=np.float64).reshape(3, 1),
                               np.asarray(camera_matrix, dtype=np.float64),
                               _norm_dist(dist))
    return pts.reshape(-1, 2)


def solve_marker_pose(corners: np.ndarray, marker_size: float,
                      camera_matrix, dist) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """PnP on the marker quad: IPPE_SQUARE + ITERATIVE, best mean reproj px.

    Both candidates must have the marker normal pointing at the camera
    (n_z < 0, i.e. we see the printed face) - this rejects behind-flip
    solutions. ITERATIVE is the fallback for axis-aligned quads, where
    IPPE_SQUARE can return a ~10 px-residual pose (measured on the board
    textures). Returns (rvec, tvec, mean reproj px) or None.
    """
    obj = marker_object_points(marker_size)
    img_pts = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    K = np.asarray(camera_matrix, dtype=np.float64)
    dist_coeffs = _norm_dist(dist)
    best = None
    for flags in (cv2.SOLVEPNP_IPPE_SQUARE, cv2.SOLVEPNP_ITERATIVE):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, img_pts, K, dist_coeffs, flags=flags)
        except cv2.error:
            continue
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        if float((R @ np.array([0.0, 0.0, 1.0]))[2]) > 0.0:
            continue                      # face away from camera: behind-flip
        err = float(np.mean(np.linalg.norm(
            project_points(obj, rvec, tvec, camera_matrix, dist) - img_pts, axis=1)))
        if best is None or err < best[2]:
            best = (rvec, tvec, err)
    return best


def detect_markers(img: np.ndarray, camera_matrix, dist,
                   max_reproj: float = REPROJ_MAX_PX,
                   ignore_ids: Sequence[int] = tuple(IGNORE_MARKER_IDS)) -> List[Detection]:
    corners, ids, _ = get_detector().detectMarkers(img)
    if ids is None or not len(ids):
        return []
    ignore = set(ignore_ids)
    out: List[Detection] = []
    for quad, mid in zip(corners, ids.reshape(-1)):
        if int(mid) in ignore:
            continue
        pose = solve_marker_pose(quad, MARKER_SIZE, camera_matrix, dist)
        if pose is None:
            continue
        rvec, tvec, err = pose
        if err > max_reproj:
            continue
        out.append(Detection(marker_id=int(mid),
                             corners=np.asarray(quad, dtype=np.float32).reshape(4, 2),
                             rvec=rvec.reshape(3, 1), tvec=tvec.reshape(3, 1),
                             reproj_err=err))
    return out


def rectify_region(img: np.ndarray, quad: np.ndarray, out_size: int = 320) -> np.ndarray:
    """Perspective-warp the image quad (TL TR BR BL) onto an out_size square."""
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    dst = np.array([[0, 0], [out_size, 0], [out_size, out_size], [0, out_size]],
                   dtype=np.float32)
    mat = cv2.getPerspectiveTransform(quad, dst)
    patch = cv2.warpPerspective(img, mat, (out_size, out_size),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return cv2.copyMakeBorder(patch, QR_PAD_PX, QR_PAD_PX, QR_PAD_PX, QR_PAD_PX,
                              cv2.BORDER_CONSTANT, value=(255, 255, 255))


def decode_text(img: np.ndarray) -> Tuple[str, str]:
    """QR decode with fallbacks. Returns (text, method); text '' = failure."""
    detector = cv2.QRCodeDetector()
    text, _, _ = detector.detectAndDecode(img)
    if text:
        return text.strip(), 'plain'

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    text, _, _ = detector.detectAndDecode(clahe)
    if text:
        return text.strip(), 'clahe'

    text, _, _ = detector.detectAndDecode(cv2.bitwise_not(gray))
    if text:
        return text.strip(), 'inverted'

    big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    text, _, _ = detector.detectAndDecode(big)
    if text:
        return text.strip(), 'upscale2'

    try:
        from pyzbar.pyzbar import decode as zbar_decode
        for candidate, tag in ((img, 'pyzbar'), (clahe, 'pyzbar-clahe')):
            found = zbar_decode(candidate)
            if found:
                return found[0].data.decode('utf-8', 'replace').strip(), tag
    except ImportError:
        pass
    return '', ''


def read_qr(img: np.ndarray, det: Optional[Detection],
            camera_matrix, dist) -> str:
    """Decode the QR belonging to `det` via its rectified crop.

    With a marker present the crop is authoritative: a full-frame decode can
    return a DIFFERENT board's QR (measured in phase_2 Step 3: 2264 cross-
    reads at range), so crop failure returns ''. Full-frame decode only when
    no marker was detected (QR-only frames).
    """
    if det is None:
        text, _ = decode_text(img)
        return text
    quad = project_points(qr_object_points(margin=QR_QUIET_MARGIN),
                          det.rvec, det.tvec, camera_matrix, dist)
    if not np.all(np.isfinite(quad)):
        return ''
    text, _ = decode_text(rectify_region(img, quad))
    return text


def analyse_frame(img: np.ndarray, camera_matrix, dist) -> List[dict]:
    """Per-frame report for the measurement harness: markers seen + QR text.

    Returns rows {'marker_id', 'dist', 'reproj', 'qr_text'}; if no marker is
    visible, one row with marker_id None and a full-frame QR attempt.
    """
    detections = detect_markers(img, camera_matrix, dist)
    rows = []
    for d in detections:
        rows.append({'marker_id': d.marker_id, 'dist': d.distance,
                     'reproj': d.reproj_err, 'qr_text': read_qr(img, d, camera_matrix, dist)})
    if not rows:
        text, _ = decode_text(img)
        rows.append({'marker_id': None, 'dist': None, 'reproj': None, 'qr_text': text})
    return rows
