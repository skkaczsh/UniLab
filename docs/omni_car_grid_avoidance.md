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
  algo.max_iterations=3000
```

On the remote training host, prefer the tmux launcher so long runs survive SSH
disconnects and use the remote Clash proxy for any `uv` network work:

```bash
uv run scripts/remote_omni_car_tmux.py status
uv run scripts/remote_omni_car_tmux.py train \
  --session omni-car-axis-next \
  --run-name remote_tmux_axis_next \
  --max-iterations 3000
```

The `remote_tmux_axis_next` relaunch completed cleanly through this path
(`3000` iterations, `128` envs, `32` steps/env, `algo.save_interval=100`), but
it is not the selected checkpoint. Its final `model_2999.pt` single-seed eval
had `collision_fraction=0.0481`, `omni_car/tracking_error=0.5351`, and
`mean_episode_return=97.0819`; a checkpoint scan found no better tradeoff than
the existing `remote_5070_axis_track_3000` best. The evidence is stored in
`artifacts/omni_car/remote_tmux_axis_next_eval.json`, with the standardized
scanner output in
`artifacts/omni_car/remote_tmux_axis_next_checkpoint_scan.json`.

The `remote_tmux_seed2_axis` repeat run also completed cleanly on the remote
host (`2026-06-27 16:01:24 +08:00` to `16:18:29 +08:00`, commit
`f1858e5036235125df58c79644569f57721ad89e`). Its final `model_2999.pt` has
SHA-256 `0927aaa984046427fc13dd4f3e120a99d7cb5c540cfb459c95306c9bf03c7905`
and did not pass the current best gate: the best scanned checkpoint was
`2999` with `collision_fraction=0.0351` and
`omni_car/tracking_error=0.6714` versus the reference `0.0271` / `0.3831`.
The evidence is stored in
`artifacts/omni_car/remote_tmux_seed2_axis_checkpoint_scan.json`.

The first long run after the CPU step-path optimization,
`remote_tmux_cpuopt_seed11_axis`, also completed cleanly on the optimized commit
`c0a5d4901a8bddf3053222c1c411ddc2a3fad3d2` (`2026-06-27 16:26:33 +08:00` to
`16:43:48 +08:00`). Its final `model_2999.pt` has SHA-256
`9f7f7dd9a324285002fce73196fefc1bde6fdb01de544b803987d1d0ee62a2b1`. The best
scanned checkpoint was again `2999`, with `collision_fraction=0.0409` and
`omni_car/tracking_error=0.5023`, so it did not pass the current best gate. The
evidence is stored in
`artifacts/omni_car/remote_tmux_cpuopt_seed11_axis_checkpoint_scan.json`.

The matched-seed optimized rerun `remote_tmux_cpuopt_seed1_axis` used the same
effective training seed as the selected best run (`algo.seed=1`) and completed
on commit `000df9469c0dd46e410ea1b7635160bd1d600daa`. Its late checkpoint scan
found the best tradeoff at `model_2950.pt` (SHA-256
`5b28a1357b6986c9b5be2e4afa652c902217d2d9029cb55504abb1cd74558843`), with
single-seed `collision_fraction=0.0296` and
`omni_car/tracking_error=0.3818`. That improves tracking but misses the
collision gate against the current reference `0.0271` / `0.3831`. A five-seed
check of `model_2950.pt` was also below the selected checkpoint:
`collision_fraction` mean `0.0325` and `omni_car/tracking_error` mean `0.3839`.
The evidence is stored in
`artifacts/omni_car/remote_tmux_cpuopt_seed1_axis_checkpoint_scan.json`,
`artifacts/omni_car/remote_tmux_cpuopt_seed1_axis_late_checkpoint_scan.json`,
and `artifacts/omni_car/remote_tmux_cpuopt_seed1_axis_2950_multiseed_eval.json`.

## Remote sync

When GitHub SSH/HTTPS is unreliable from the training host, sync the current
branch over LAN with a temporary local bundle server. The OmniCar wrapper uses
the known remote host/worktree and local Clash proxy defaults:

```bash
uv run scripts/sync_omni_car_remote.py
```

By default the wrapper copies the bundle with `scp`, which avoids requiring the
macOS workstation to accept inbound HTTP connections. Pass `--transport http
--local-host <LAN-IP>` only when a temporary local HTTP server is reachable from
the training host. The wrapper delegates to `scripts/sync_remote_bundle.py`,
which updates an existing remote repo/worktree in place by fetching the bundle
and resetting tracked files to the local HEAD on the stable remote branch
`codex/omni-car-grid-ppo-wt-remote`. It does not delete the remote worktree, so
logs and other untracked training artifacts remain in place.

For normal repeat syncs, the wrapper first checks the remote worktree HEAD. If
that commit is already the local HEAD, sync is a no-op. If the remote commit is
an ancestor of local HEAD, the wrapper sends an incremental bundle from that
base and does not use GitHub or the local clone proxy. If the remote base is
missing or divergent, it falls back to `--bundle-source-url`, which creates and
verifies a temporary full clone before serving the bundle; this fallback is
important when the local checkout is a partial/promisor clone.

## Environment benchmark

The OmniCar task follows UniLab's intended split: CPU-side environment stepping
and grid construction feed a GPU PPO learner. To measure the CPU side directly:

```bash
uv run scripts/benchmark_omni_car_env.py \
  --num-envs 128 \
  --steps 120 \
  --warmup-steps 10 \
  --action-mode zero \
  --profile-top 15
```

Use `--grid-workers` only as a diagnostic switch. It monkeypatches the benchmark
process with a persistent Python `ThreadPoolExecutor` and splits occupancy-grid
row chunks across worker threads:

```bash
uv run scripts/benchmark_omni_car_env.py \
  --num-envs 128 \
  --steps 80 \
  --warmup-steps 10 \
  --obstacles 18 \
  --action-mode zero \
  --grid-workers 4 \
  --json
```

The attempt to parallelize this path with Python threads regresses throughput.
The bottleneck is not raw arithmetic; it is many small obstacle AABB raster
tasks plus large occupancy/observation buffer writes. Threading adds chunk
dispatch and synchronization without changing that memory-access pattern. A
profile of the threaded diagnostic path showed the concrete failure mode:
splitting by env chunk calls `_fill_occupancy_grid` once per worker, so the
process repeats body-frame obstacle transforms and grid clears, then the main
thread waits on `Future.result()`.

The production path therefore stays single-process/vectorized. It fixes CPU
cost in four ways:

- occupancy grids are rasterized only inside each obstacle's local AABB instead
  of scanning the full `80 x 80` grid for every obstacle
- clearance is computed in one batched vectorized pass across all envs
- observation, critic, grid, velocity-limit, and env-index arrays are reused
  instead of reallocated on every step
- large batches (`>=256` envs) precompute obstacle bounds in batched NumPy
  arrays before the AABB raster loop, while the `128` env training case keeps
  the lower-overhead scalar AABB path

With those changes, the local serial benchmark moved again. A same-machine
comparison against commit `4271cf84` on 2026-06-27 showed:

- default `14` obstacles, `128` envs: roughly `57.3 ms/step` -> `8.7 ms/step`
- default `14` obstacles, `512` envs: roughly `102.1 ms/step` -> `33.9 ms/step`
- training-like `18` obstacles, `128` envs: old serial `12.45 ms/step`,
  current serial `12.49 ms/step`, current `4` grid workers `15.60 ms/step`
- training-like `18` obstacles, `512` envs: old serial `44.85 ms/step`,
  current serial `42.58 ms/step`

That is a better lever than Python threading for this environment because it
cuts the CPU work itself instead of parallelizing avoidable work.

## Native viewer

Open the native OpenGL viewer with a hand-authored policy:

```bash
uv run scripts/visualize_omni_car.py --policy reflex --steps 600 --seed 7 --obstacles 20
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

For a new long run, scan candidate checkpoints with the same reference gates
before deciding whether it replaces the current best. The scanner suppresses
evaluator model/debug logs by default so stdout remains valid JSON; add
`--verbose` when inspecting evaluator startup:

```bash
uv run scripts/scan_omni_car_checkpoints.py \
  --load-run /absolute/path/to/run_dir \
  --checkpoints 500 900 1200 1500 1800 2200 2600 2900 2999 \
  --reference-manifest artifacts/omni_car/axis_track_2999_checkpoint_manifest.json \
  --output artifacts/omni_car/<run_name>_checkpoint_scan.json
```

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
