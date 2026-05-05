import math
from collections.abc import Sequence


def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def shortest_angular_distance(current: float, target: float) -> float:
    return wrap_to_pi(target - current)


def base_target_error(
    *,
    current_xy: Sequence[float],
    current_yaw: float,
    target_xy: Sequence[float],
    target_yaw: float,
) -> tuple[float, float]:
    pos_err = math.hypot(float(target_xy[0]) - float(current_xy[0]), float(target_xy[1]) - float(current_xy[1]))
    yaw_err = abs(shortest_angular_distance(float(current_yaw), float(target_yaw)))
    return pos_err, yaw_err


def base_target_reached(
    *,
    current_xy: Sequence[float],
    current_yaw: float,
    target_xy: Sequence[float],
    target_yaw: float,
    pos_tol: float,
    yaw_tol: float,
) -> bool:
    pos_err, yaw_err = base_target_error(
        current_xy=current_xy,
        current_yaw=current_yaw,
        target_xy=target_xy,
        target_yaw=target_yaw,
    )
    return pos_err <= float(pos_tol) and yaw_err <= float(yaw_tol)


def target_progress_stalled(
    *,
    previous_pos_err: float,
    previous_yaw_err: float,
    current_pos_err: float,
    current_yaw_err: float,
    min_pos_improvement: float,
    min_yaw_improvement: float,
) -> bool:
    pos_improvement = float(previous_pos_err) - float(current_pos_err)
    yaw_improvement = float(previous_yaw_err) - float(current_yaw_err)
    return pos_improvement < float(min_pos_improvement) and yaw_improvement < float(min_yaw_improvement)


def estimate_target_duration(
    *,
    start_xy: Sequence[float],
    start_yaw: float,
    target_xy: Sequence[float],
    target_yaw: float,
    min_duration: float,
    linear_speed: float,
    angular_speed: float,
    settle_time: float,
) -> float:
    if linear_speed <= 0.0:
        raise ValueError("linear_speed must be positive")
    if angular_speed <= 0.0:
        raise ValueError("angular_speed must be positive")

    linear_time = math.hypot(float(target_xy[0]) - float(start_xy[0]), float(target_xy[1]) - float(start_xy[1])) / float(linear_speed)
    angular_time = abs(shortest_angular_distance(float(start_yaw), float(target_yaw))) / float(angular_speed)
    return max(float(min_duration), linear_time, angular_time) + float(settle_time)
