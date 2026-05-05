import math

from hsr_genesis.trajectory_completion import (
    base_target_error,
    base_target_reached,
    estimate_target_duration,
    target_progress_stalled,
)


def test_base_target_reached_requires_position_and_yaw_within_tolerance():
    assert not base_target_reached(
        current_xy=(0.0, 0.0),
        current_yaw=0.0,
        target_xy=(0.20, 0.0),
        target_yaw=0.0,
        pos_tol=0.10,
        yaw_tol=math.radians(5.0),
    )
    assert not base_target_reached(
        current_xy=(0.0, 0.0),
        current_yaw=math.radians(12.0),
        target_xy=(0.0, 0.0),
        target_yaw=0.0,
        pos_tol=0.10,
        yaw_tol=math.radians(5.0),
    )
    assert base_target_reached(
        current_xy=(0.04, -0.03),
        current_yaw=math.radians(2.0),
        target_xy=(0.0, 0.0),
        target_yaw=0.0,
        pos_tol=0.10,
        yaw_tol=math.radians(5.0),
    )


def test_base_target_error_reports_distance_and_wrapped_yaw_error():
    pos_err, yaw_err = base_target_error(
        current_xy=(0.0, 0.0),
        current_yaw=math.radians(179.0),
        target_xy=(3.0, 4.0),
        target_yaw=math.radians(-179.0),
    )

    assert math.isclose(pos_err, 5.0, rel_tol=0.0, abs_tol=1.0e-9)
    assert math.isclose(yaw_err, math.radians(2.0), rel_tol=0.0, abs_tol=1.0e-9)


def test_target_progress_stalled_requires_meaningful_improvement():
    assert target_progress_stalled(
        previous_pos_err=0.45,
        previous_yaw_err=math.radians(10.0),
        current_pos_err=0.44,
        current_yaw_err=math.radians(9.5),
        min_pos_improvement=0.02,
        min_yaw_improvement=math.radians(2.0),
    )
    assert not target_progress_stalled(
        previous_pos_err=0.45,
        previous_yaw_err=math.radians(10.0),
        current_pos_err=0.39,
        current_yaw_err=math.radians(9.5),
        min_pos_improvement=0.02,
        min_yaw_improvement=math.radians(2.0),
    )


def test_estimate_target_duration_grows_for_farther_targets():
    near_duration = estimate_target_duration(
        start_xy=(0.0, 0.0),
        start_yaw=0.0,
        target_xy=(0.1, 0.0),
        target_yaw=0.0,
        min_duration=1.1,
        linear_speed=0.5,
        angular_speed=1.0,
        settle_time=0.3,
    )
    far_duration = estimate_target_duration(
        start_xy=(0.0, 0.0),
        start_yaw=0.0,
        target_xy=(2.0, 0.0),
        target_yaw=math.pi,
        min_duration=1.1,
        linear_speed=0.5,
        angular_speed=1.0,
        settle_time=0.3,
    )

    assert math.isclose(near_duration, 1.4, rel_tol=0.0, abs_tol=1.0e-9)
    assert far_duration > near_duration
    assert math.isclose(far_duration, 4.3, rel_tol=0.0, abs_tol=1.0e-9)
