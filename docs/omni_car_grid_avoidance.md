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
- Reward safe response progress when the executed velocity moves closer to the
  user command, gated down near obstacles so avoidance can override intent.
- Penalize normalized output velocity jumps to prefer smooth safe commands.
- Penalize low obstacle clearance and collisions.

Physical limit contract:

- The environment enforces velocity and acceleration limits before integrating
  body motion.
- Speed and acceleration limits are not represented as reward penalties.

Obstacle contract:

- Obstacles are sampled as a configurable mixture of circles, rotated boxes,
  and long thin wall segments.
- All obstacle types are rasterized into the same body-centered occupancy grid.
- Clearance uses circle distance for circles and signed distance to the rotated
  rectangle for boxes and walls.

Policy network:

- The default PPO owner config uses `OmniCarGridCNNModel`.
- The first 6400 observation values are reshaped into a `1 x 80 x 80`
  occupancy image and encoded by a CNN.
- User command, executed velocity, last action, clearance, and collision state
  are concatenated with CNN features before the MLP head.

PPO smoke command:

```bash
uv run train --algo ppo --task omni_car_grid_avoidance --sim mujoco \
  algo.num_envs=8 algo.num_steps_per_env=4 algo.max_iterations=1 \
  training.no_play=true training.play_render_mode=none training.logger=tensorboard
```

This smoke run validates integration with the UniLab PPO pipeline. It is not a
convergence benchmark.

Native GLFW/OpenGL 3D viewer playback:

```bash
uv run scripts/visualize_omni_car.py --policy reflex --steps 600
```

Checkpoint playback through the PPO eval path:

```bash
uv run eval --algo ppo --task omni_car_grid_avoidance --sim mujoco \
  --render-mode interactive --load-run -1 \
  training.log_root=logs/graphical training.play_steps=600 training.export_onnx=false
```

The viewer shows the rectangular vehicle, circle/box/wall obstacles, the
body-centered local grid footprint, a green user-command arrow, and a blue
executed-velocity arrow.
