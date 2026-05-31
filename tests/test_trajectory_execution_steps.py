import math

from examples.tutorials.rrt_path_planning_hsr import _trajectory_execution_steps


def test_trajectory_execution_steps_with_first_step_at_zero():
    """Verify that the returned step count yields max sampled relative time >= duration,
    given the controller samples relative time 0 on the first step (step 0 at t=0).
    Without the fix, ceil(duration/dt) gives one step too few."""
    duration = 4.0
    dt = 0.1
    steps = _trajectory_execution_steps(duration, dt)
    # Controller: step k samples at t = k*dt
    # After N steps the max sampled time is (N-1)*dt
    max_sampled_time = (steps - 1) * dt
    assert max_sampled_time >= duration, (
        f"With {steps} steps and dt={dt}, max sampled time = {max_sampled_time} < {duration}"
    )


def test_trajectory_execution_steps_exact_division():
    """When duration is exactly divisible by dt, ceil(duration/dt) + 1 steps needed."""
    steps = _trajectory_execution_steps(1.0, 0.1)
    assert steps == 11  # ceil(1.0/0.1) + 1 = 10 + 1 = 11


def test_trajectory_execution_steps_not_exact():
    """When duration is not divisible by dt, ceil+1 still covers."""
    steps = _trajectory_execution_steps(1.05, 0.1)
    assert steps == 12  # ceil(1.05/0.1) + 1 = 11 + 1 = 12
