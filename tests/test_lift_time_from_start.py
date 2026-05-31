"""Tests for _build_velocity_limited_time_from_start.

Verifies the helper that enforces per-joint velocity limits when building
trajectory time_from_start from per-segment waypoints.

Run with:
    PYTHONPATH=src .venv/bin/python -m pytest tests/test_lift_time_from_start.py -v
"""

from __future__ import annotations

import torch

# Import the helper directly from the example module.
from examples.tutorials.rrt_path_planning_hsr import (
    _build_velocity_limited_time_from_start,
)


def test_monotonic_and_first_time() -> None:
    """Times are strictly monotonic and the first time equals dt."""
    # 6 waypoints, 2 segments of 3 each, no lift change.
    arm_positions = torch.zeros((6, 5), dtype=torch.float64)
    dt = 0.01
    segment_end_wps = [2, 5]
    per_segment_duration = 4.0
    lift_speed_limit = 0.1

    t = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration, velocity_limits={0: lift_speed_limit},
    )

    assert t.shape == (6,), f"Expected shape (6,), got {t.shape}"
    # Strictly increasing
    assert torch.all(t[1:] > t[:-1]), "Times must be strictly monotonic"
    # First time equals dt
    assert abs(float(t[0]) - dt) < 1e-10, f"First time should be {dt}, got {t[0]}"


def test_segment_duration_at_least_per_segment() -> None:
    """Each segment spans at least per_segment_duration."""
    arm_positions = torch.zeros((6, 5), dtype=torch.float64)
    dt = 0.01
    segment_end_wps = [2, 5]
    per_segment_duration = 4.0
    lift_speed_limit = 0.1

    t = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration, velocity_limits={0: lift_speed_limit},
    )

    # Segment 0: waypoints 0..2, duration = t[2] - t[0]
    seg0_dur = float(t[2] - t[0])
    assert seg0_dur >= per_segment_duration - 1e-10, (
        f"Segment 0 duration {seg0_dur:.4f}s < {per_segment_duration}s"
    )
    # Segment 1: waypoints 3..5, duration = t[5] - t[3]
    seg1_dur = float(t[5] - t[3])
    assert seg1_dur >= per_segment_duration - 1e-10, (
        f"Segment 1 duration {seg1_dur:.4f}s < {per_segment_duration}s"
    )


def test_lift_speed_limit_respected() -> None:
    """Arm lift speed never exceeds the limit, even with a sharp lift change."""
    # 6 waypoints.  Segment 0: lift=0.  Segment 1 jumps to 0.5, then stays.
    lift = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    arm_positions = torch.tensor(
        [[v, 0.0, 0.0, 0.0, 0.0] for v in lift],
        dtype=torch.float64,
    )
    dt = 0.01
    segment_end_wps = [2, 5]
    per_segment_duration = 4.0
    lift_speed_limit = 0.1  # m/s

    t = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration, velocity_limits={0: lift_speed_limit},
    )

    for i in range(len(lift) - 1):
        dl = abs(arm_positions[i + 1, 0] - arm_positions[i, 0])
        dt_interval = float(t[i + 1] - t[i])
        if dt_interval > 1e-12:
            speed = float(dl) / dt_interval
            assert speed <= lift_speed_limit + 1e-10, (
                f"Interval {i}→{i+1}: lift speed {speed:.6f} m/s "
                f"exceeds limit {lift_speed_limit} m/s"
            )


def test_sharp_lift_change_stretches_total_duration() -> None:
    """A sharp lift change at the segment boundary extends total duration."""
    lift = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]
    arm_positions = torch.tensor(
        [[v, 0.0, 0.0, 0.0, 0.0] for v in lift],
        dtype=torch.float64,
    )
    dt = 0.01
    segment_end_wps = [2, 5]
    per_segment_duration = 4.0
    lift_speed_limit = 0.1

    t = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration, velocity_limits={0: lift_speed_limit},
    )

    total = float(t[-1] - t[0])
    # The 0→0.5 jump needs 5 s minimum, so total >= 8 + 1.0 = 9.0
    # (2 segments × 4 s base + extra for lift)
    min_bound = 2 * per_segment_duration + (0.5 / lift_speed_limit) - 0.1
    assert total >= min_bound, (
        f"Total duration {total:.4f}s should be >= {min_bound:.4f}s "
        f"for a 0.5m lift change at {lift_speed_limit} m/s limit"
    )


def test_descend_stretched_more_than_ascend() -> None:
    """Descending lift changes are stretched more with a lower descend limit."""
    # 4 waypoints: start low, go up, stay, come down.
    lift = [0.15, 0.55, 0.55, 0.15]
    arm_positions = torch.tensor(
        [[v, 0.0, 0.0, 0.0, 0.0] for v in lift],
        dtype=torch.float64,
    )
    dt = 0.01
    segment_end_wps = [3]  # single segment
    per_segment_duration = 4.0
    lift_speed_limit = 0.1
    descend_lift_speed_limit = 0.03  # 3× slower descent

    t = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration, velocity_limits={0: lift_speed_limit},
        descend_velocity_limits={0: descend_lift_speed_limit},
    )

    lift_changes = [lift[i + 1] - lift[i] for i in range(len(lift) - 1)]
    for i in range(len(lift) - 1):
        dt_int = float(t[i + 1] - t[i])
        dl = abs(lift[i + 1] - lift[i])
        if dl < 1e-12:
            continue
        speed = dl / dt_int
        if lift_changes[i] > 0:  # ascending
            assert speed <= lift_speed_limit + 1e-10, (
                f"Ascending interval {i}→{i+1}: speed {speed:.6f} m/s "
                f"exceeds ascend limit {lift_speed_limit}"
            )
        elif lift_changes[i] < 0:  # descending
            assert speed <= descend_lift_speed_limit + 1e-10, (
                f"Descending interval {i}→{i+1}: speed {speed:.6f} m/s "
                f"exceeds descend limit {descend_lift_speed_limit}"
            )

    # The descending interval should be noticeably longer than ascending
    # because 0.03 < 0.1 for the same 0.4 m displacement.
    ascend_dt = float(t[1] - t[0])
    descend_dt = float(t[3] - t[2])
    assert descend_dt > ascend_dt + 0.5, (
        f"Descending dt {descend_dt:.4f}s should be substantially larger "
        f"than ascending dt {ascend_dt:.4f}s"
    )


def test_lift_speed_exact_boundary() -> None:
    """Lift speed exactly at the limit is not stretched further."""
    # 2 waypoints only: lift increases exactly at lift_speed_limit
    arm_positions = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, 0.0],
         [0.2, 0.0, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    dt = 0.01
    segment_end_wps = [1]  # single segment
    per_segment_duration = 4.0
    # lift_speed = 0.2/4.0 = 0.05 < 0.1, should not stretch
    lift_speed_limit = 0.1

    t = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration, velocity_limits={0: lift_speed_limit},
    )

    # No stretch needed: delta_t should be 4.0
    delta_t = float(t[1] - t[0])
    assert abs(delta_t - per_segment_duration) < 1e-10, (
        f"Delta_t {delta_t:.4f}s should equal "
        f"per_segment_duration {per_segment_duration}s when within limit"
    )


def test_flex_speed_limit_respected() -> None:
    """Arm flex speed never exceeds the limit, even with a sharp flex change.

    With zero lift delta and large flex delta, the time must be stretched
    to satisfy the flex speed limit, proving flex enforcement is active.
    """
    # 3 waypoints: no lift change, but large flex change at the boundary.
    lift = [0.0, 0.0, 0.0]
    flex = [0.0, 1.5, 1.5]  # 1.5 rad change at boundary → needs >= 1.5 s at 1.0 rad/s
    arm_positions = torch.tensor(
        [[lift[i], flex[i], 0.0, 0.0, 0.0] for i in range(3)],
        dtype=torch.float64,
    )
    dt = 0.01
    segment_end_wps = [2]  # single segment
    per_segment_duration = 1.0  # shorter than flex-limited minimum
    lift_speed_limit = 0.1
    flex_speed_limit = 1.0  # rad/s

    t = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration,
        velocity_limits={0: lift_speed_limit, 1: flex_speed_limit},
    )

    for i in range(len(lift) - 1):
        df = float(abs(arm_positions[i + 1, 1].item() - arm_positions[i, 1].item()))
        dt_int = float(t[i + 1] - t[i])
        if dt_int > 1e-12 and df > 0.0:
            speed = df / dt_int
            assert speed <= flex_speed_limit + 1e-10, (
                f"Interval {i}→{i+1}: flex speed {speed:.6f} rad/s "
                f"exceeds limit {flex_speed_limit} rad/s"
            )

    # The flex-limited interval (0→1) should be >= 1.5 s (1.5 rad / 1.0 rad/s)
    flex_dt = float(t[1] - t[0])
    assert flex_dt >= 1.5 - 1e-10, (
        f"Flex-limited dt {flex_dt:.4f}s should be >= 1.5 s "
        f"for 1.5 rad change at 1.0 rad/s"
    )


def test_all_limited_joints_respect_max_speed() -> None:
    """Multiple limited joints: each must respect its speed limit.

    Tests both lift (ascending/descending) and flex limits simultaneously,
    verifying the ``max`` over joint deltas/limits determines timing.
    """
    # 4 waypoints:
    #   Lift:  0.0 → 0.3 (ascend) → 0.3 → 0.0 (descend)
    #   Flex:  0.0 → 0.0 → 1.5 → 1.5
    lift = [0.0, 0.3, 0.3, 0.0]
    flex = [0.0, 0.0, 1.5, 1.5]
    arm_positions = torch.tensor(
        [[lift[i], flex[i], 0.0, 0.0, 0.0] for i in range(4)],
        dtype=torch.float64,
    )
    dt = 0.01
    segment_end_wps = [3]  # single segment
    per_segment_duration = 1.0  # short so velocity limits dominate
    velocity_limits = {0: 0.1, 1: 1.0}
    descend_velocity_limits = {0: 0.03}

    t = _build_velocity_limited_time_from_start(
        arm_positions, dt, segment_end_wps,
        per_segment_duration,
        velocity_limits=velocity_limits,
        descend_velocity_limits=descend_velocity_limits,
    )

    for i in range(len(lift) - 1):
        d_lift = float(abs(lift[i + 1] - lift[i]))
        d_flex = float(abs(flex[i + 1] - flex[i]))
        dt_int = float(t[i + 1] - t[i])
        if dt_int > 1e-12:
            if d_lift > 0:
                speed = d_lift / dt_int
                is_descend = (lift[i + 1] - lift[i]) < 0
                limit = descend_velocity_limits[0] if is_descend else velocity_limits[0]
                assert speed <= limit + 1e-10, (
                    f"Interval {i}→{i+1}: lift speed {speed:.6f} m/s "
                    f"exceeds limit {limit}"
                )
            if d_flex > 0:
                speed = d_flex / dt_int
                assert speed <= velocity_limits[1] + 1e-10, (
                    f"Interval {i}→{i+1}: flex speed {speed:.6f} rad/s "
                    f"exceeds limit {velocity_limits[1]}"
                )

    # Expected min durations:
    #   Interval 0 (0→1): lift ascend 0.3/0.1=3.0s, flex 0.0 → uniform=1.0/3≈0.33 → max=3.0s
    #   Interval 1 (1→2): lift 0.0, flex 1.5/1.0=1.5s → max=1.5s
    #   Interval 2 (2→3): lift descend 0.3/0.03=10.0s, flex 0.0 → max=10.0s
    # Total ≥ 3.0 + 1.5 + 10.0 = 14.5s (uniform adds negligible extra)
    min_expected = 3.0 + 1.5 + 10.0
    total = float(t[-1] - t[0])
    assert total >= min_expected - 0.01, (
        f"Total duration {total:.4f}s should be >= {min_expected:.4f}s "
        f"with both lift and flex limits"
    )

    # The descend interval (2→3) should be the longest (10s at 0.03 m/s for 0.3m)
    desc_dt = float(t[3] - t[2])
    assert desc_dt > 9.0, (
        f"Descending dt {desc_dt:.4f}s should be ~10s for 0.3m at 0.03 m/s"
    )
