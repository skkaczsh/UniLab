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
3. Penalize per-axis safe-space tracking error independently for `vx`, `vy`, `vyaw`.
4. Penalize per-axis action diff and jerk independently for `vx`, `vy`, `vyaw`.
5. Preserve clearance and heavily punish collision.

The current tuning pass intentionally shifted some burden from reward penalties
back into the physical envelope: acceleration caps were loosened so the policy
can respond to intent and obstacles faster, while collision and clearance terms
were strengthened to stop the extra agility from turning into reckless contact.
The latest yaw-response pass adds explicit clearance-gated per-axis tracking
costs and raises yaw intent/tracking weight. This addresses a failure mode seen
in the `remote_5070_git_long_3000` run where later checkpoints became very
smooth by suppressing yaw output instead of following `vyaw` commands.

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

## Remote sync

When GitHub SSH/HTTPS is unreliable from the training host, sync the current
branch over LAN with a temporary local bundle server:

```bash
uv run python scripts/sync_remote_bundle.py \
  --remote zsh@skkac.top \
  --ssh-port 6010 \
  --local-host 192.168.0.3 \
  --bundle-source-url https://github.com/skkaczsh/UniLab.git \
  --clone-proxy http://127.0.0.1:7890 \
  --origin-url git@github.com:skkaczsh/UniLab.git \
  --venv-source /home/zsh/develop/worktrees/UniLab-omni-car/.venv
```

The script updates an existing remote repo/worktree in place by fetching the
bundle and resetting tracked files to the local HEAD. It does not delete the
remote worktree, so logs and other untracked training artifacts remain in place.
The `--bundle-source-url` path is important when the local checkout is a
partial/promisor clone; the script creates and verifies a temporary full clone
before serving the bundle.

## Environment benchmark

The OmniCar task follows UniLab's intended split: CPU-side environment stepping
and grid construction feed a GPU PPO learner. To measure the CPU side directly:

```bash
uv run python scripts/benchmark_omni_car_env.py \
  --num-envs 128 \
  --steps 120 \
  --warmup-steps 10 \
  --action-mode zero \
  --profile-top 15
```

The earlier attempt to parallelize this path with a Python `ThreadPoolExecutor`
regressed throughput. The bottleneck was not raw arithmetic; it was the amount
of memory scanned per obstacle plus the overhead of dispatching many small
Python tasks. The current fast path fixes that in two ways:

- occupancy grids are rasterized only inside each obstacle's local AABB instead
  of scanning the full `80 x 80` grid for every obstacle
- clearance is computed in one batched vectorized pass across all envs

With those changes, the local benchmark moved again:

- `128` envs: roughly `57.3 ms/step` -> `10.6 ms/step`
- `512` envs: roughly `102.1 ms/step` -> `34.6 ms/step`

That is a better lever than Python threading for this environment because it
cuts the CPU work itself instead of parallelizing avoidable work.

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
- logged reward terms for response progress and track / diff / jerk costs

Latest tuned run on the remote RTX 5070 Ti host:

- run dir:
  `logs/remote_5070_axis_track_3000/OmniCarGridAvoidance/2026-06-27_14-11-59_mujoco`
- checkpoint: `model_2999.pt`
- `collision_fraction`: `0.0268`
- `mean_episode_return`: `304.3877`
- `mean_episode_length`: `21.7563`
- `vx / vy / vyaw tracking_mae`: `0.3766 / 0.3909 / 0.4214`
- `omni_car/response_progress`: `0.0168`
- `omni_car/tracking_error`: `0.3840`
- `omni_car/vx / vy / vyaw jerk_cost`: `0.1322 / 0.1540 / 0.0476`

Multi-seed evaluation evidence is stored in
`artifacts/omni_car/axis_track_2999_multiseed_eval.json` (`64` envs, `512`
steps, seeds `7 / 17 / 23 / 31 / 43`). Across those seeds:

- `collision_fraction`: mean `0.0271`, range `0.0242 - 0.0304`
- `mean_episode_return`: mean `284.4116`, range `263.3304 - 304.3877`
- `omni_car/tracking_error`: mean `0.3831`, range `0.3730 - 0.3906`
- `vx / vy / vyaw tracking_mae`: mean `0.3850 / 0.3831 / 0.4184`
- `vx / vy / vyaw jerk_cost`: mean `0.1338 / 0.1530 / 0.0476`

The checkpoint itself is intentionally not committed. Its reproducibility
manifest is stored in
`artifacts/omni_car/axis_track_2999_checkpoint_manifest.json`, including the
remote run path, byte count, SHA-256, evaluation artifact, and copy/viewer
commands. To cache or verify it locally for the native viewer:

```bash
uv run scripts/fetch_omni_car_checkpoint.py
uv run scripts/fetch_omni_car_checkpoint.py --verify-only
uv run scripts/view_omni_car_checkpoint.py --load-run artifacts/omni_car/checkpoints/model_2999.pt --device cpu
```

Previous pre-axis-tracking comparison run:

- run dir:
  `logs/remote_5070_tuned/OmniCarGridAvoidance/2026-06-26_16-15-08_mujoco`
- checkpoint: `model_259.pt`
- `collision_fraction`: `0.1273`
- `mean_episode_return`: `3.7147`
- `mean_episode_length`: `7.2488`
- `vx / vy / vyaw tracking_mae`: `0.7240 / 0.6734 / 0.7304`
- `omni_car/response_progress`: `0.0251`
- `omni_car/tracking_error`: `0.6578`
