# OmniCar Grid Avoidance

This task models a rectangular omnidirectional car that receives a user command
`[vx, vy, vyaw]` and must keep that intent when the local grid is safe, while
deviating smoothly when obstacles block the path.

## Observation and action

- Grid input: `80 x 80`, `0.05 m` per cell, centered on the robot body frame.
- Policy input: occupancy grid, current smoothed command, current velocity, last
  action, command history, velocity history, action history, nearest clearance,
  and collision flag.
- Action output: body-frame `vx`, `vy`, `vyaw`.
- Physical limits are enforced before integration. Tracking smoothness is shaped
  in reward, while absolute feasibility comes from velocity and acceleration
  caps.

## Current training profile

The MuJoCo owner config lives in
`conf/ppo/task/omni_car_grid_avoidance/mujoco.yaml`.

Current defaults in this branch:

- Episode horizon: `60 s`
- Observation history: `24` frames
- Command resample interval: `2.5 s`
- Command smoothing time constant: `0.55 s`
- PPO rollout: `128` envs, `32` steps per env
- Policy architecture: `OmniCarGridCNNModel` (`CNN + MLP`)
- Obstacle sampling: `18` mixed obstacles with a `0.90 m` keepout radius
- Physical acceleration caps: `3.5 / 3.5 / 4.5` for `vx / vy / vyaw`

Reward shaping emphasizes four things:

1. Track commanded planar and yaw intent.
2. Improve response speed when clearance allows.
3. Penalize per-axis action diff and jerk independently for `vx`, `vy`, `vyaw`.
4. Preserve clearance and heavily punish collision.

The current tuning pass intentionally shifted some burden from reward penalties
back into the physical envelope: acceleration caps were loosened so the policy
can respond to intent and obstacles faster, while collision and clearance terms
were strengthened to stop the extra agility from turning into reckless contact.

## Training

```bash
uv run train --algo ppo --task omni_car_grid_avoidance --sim mujoco \
  training.logger=tensorboard \
  algo.num_envs=128 \
  algo.num_steps_per_env=32
```

On Linux, keep the default `uv` torch source on `cu128`. The previous
`2.7.0+cu128` build is not sufficient for RTX 50-series (`sm_120`) GPUs; the
current repo baseline is `torch 2.11.0+cu128`.

If you want a longer run similar to the latest tuning pass:

```bash
uv run train --algo ppo --task omni_car_grid_avoidance --sim mujoco \
  training.logger=tensorboard \
  algo.num_envs=128 \
  algo.num_steps_per_env=32 \
  algo.max_iterations=94
```

## Native viewer

Open the native OpenGL viewer with a hand-authored policy:

```bash
uv run python scripts/visualize_omni_car.py --policy reflex --steps 600 --seed 7 --obstacles 20
```

Because the script now has a shebang, this also works after checkout:

```bash
uv run scripts/visualize_omni_car.py --policy intent --steps 600
```

## Checkpoint playback

For trained checkpoints, use the UniLab evaluation CLI:

```bash
uv run eval --algo ppo --task omni_car_grid_avoidance --sim mujoco \
  --render-mode interactive \
  --load-run /absolute/path/to/run_dir \
  --checkpoint model_40.pt
```

`--checkpoint` also accepts an iteration number such as `40` when the
corresponding `model_40.pt` exists under the run directory.

For a direct native 3D viewer entrypoint dedicated to this task:

```bash
uv run scripts/view_omni_car_checkpoint.py \
  --load-run /absolute/path/to/run_dir \
  --checkpoint 93
```

## Quantitative evaluation

For headless metric checks on a trained checkpoint:

```bash
uv run scripts/evaluate_omni_car_checkpoint.py \
  --load-run /absolute/path/to/run_dir \
  --checkpoint 93 \
  --num-envs 64 \
  --num-steps 512 \
  --json
```

This reports:

- episodic return / episode length
- collision fraction and worst observed clearance
- per-axis `vx`, `vy`, `vyaw` tracking MAE / RMSE
- logged reward terms for response progress and diff / jerk costs
