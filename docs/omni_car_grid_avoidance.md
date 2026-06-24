# Omni Car Grid Avoidance

`OmniCarGridAvoidance` is a vectorized 2D task for a rectangular omnidirectional
vehicle under user teleoperation.

The user command has three velocity axes: body-frame `x`, body-frame `y`, and
yaw rate. Each axis is bounded to `[-2, 2]` by default. The policy outputs the
safe velocity command that the vehicle executes, with the same three dimensions.

Observation contract:

- `80 x 80` local occupancy grid, flattened to 6400 values.
- Each cell is `0.05 m`, so the grid covers a `4 m x 4 m` square.
- The grid origin is the vehicle body center and the grid axes are body-aligned.
- Low-dimensional state appends user command, current velocity, last action,
  nearest clearance, and collision flag.

Reward contract:

- Reward command tracking and projection along the user command.
- Reward yaw-rate tracking separately.
- Penalize low obstacle clearance, collisions, action rate, and speed-limit
  violations.

PPO smoke command:

```bash
uv run train --algo ppo --task omni_car_grid_avoidance --sim mujoco \
  algo.num_envs=8 algo.num_steps_per_env=4 algo.max_iterations=1 \
  training.no_play=true training.play_render_mode=none training.logger=tensorboard
```

This smoke run validates integration with the UniLab PPO pipeline. It is not a
convergence benchmark.
