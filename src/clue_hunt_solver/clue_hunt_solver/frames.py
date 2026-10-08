"""World/board geometry (final_plan.md §4.4, verified in §2.2).

Axis convention: X = (cos yaw, sin yaw) points out of the board face,
Y = (-sin yaw, cos yaw) is the reader's right; a REL target is
T = c + a*X + b*Y with c the board centre in `map`.

Axis permutation (§4.2): board X = marker z, board Y = marker x,
board Z = marker y - ties the PnP marker frame to the board frame, so the
QR (marker-frame offset +0.30 m along marker x) sits at board point (0, +0.30, 0).
"""
import math
from dataclasses import dataclass
from typing import Sequence, Tuple

Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class BoardPose:
    x: float
    y: float
    yaw: float

    @property
    def xy(self) -> Vec2:
        return (self.x, self.y)


def axes(yaw: float) -> Tuple[Vec2, Vec2]:
    """Board frame axes in map coordinates."""
    c, s = math.cos(yaw), math.sin(yaw)
    return (c, s), (-s, c)


def goto_target(x: float, y: float) -> Vec2:
    return (x, y)


def rel_target(centre: Sequence[float], yaw: float, a: float, b: float) -> Vec2:
    (ex, ey), (rx, ry) = axes(yaw)
    return (centre[0] + a * ex + b * rx, centre[1] + a * ey + b * ry)


def treasure_target(centre: Sequence[float], yaw: float, a: float, b: float) -> Vec2:
    return rel_target(centre, yaw, a, b)


def between_target(p_a: Sequence[float], p_b: Sequence[float], f: float) -> Vec2:
    return (p_a[0] + f * (p_b[0] - p_a[0]), p_a[1] + f * (p_b[1] - p_a[1]))


def distance(p: Sequence[float], q: Sequence[float]) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def marker_to_board(p_marker: Sequence[float]) -> Tuple[float, float, float]:
    """Permutation: board X = marker z, Y = marker x, Z = marker y."""
    return (p_marker[2], p_marker[0], p_marker[1])


def board_to_marker(p_board: Sequence[float]) -> Tuple[float, float, float]:
    return (p_board[1], p_board[2], p_board[0])


def quaternion_from_yaw(yaw: float) -> Tuple[float, float, float, float]:
    """(x, y, z, w) rotation about +z."""
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quaternion_to_matrix(x: float, y: float, z: float, w: float
                         ) -> Tuple[Tuple[float, float, float], ...]:
    """(x, y, z, w) -> 3x3 rotation matrix (rows), e.g. for tf2 quaternions."""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        return ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    x, y, z, w = x / n, y / n, z / n, w / n
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
        (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
        (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
    )
