# OmniCar Grid Avoidance

This task models a rectangular omnidirectional car that receives a user command
`[vx, vy, vyaw]` and must keep that intent when the local grid is safe, while
deviating smoothly when obstacles block the path.

## Observation and action

- Grid input: `80 x 80`, `0.05 m` per cell, centered on the robot body frame.
- Policy input: deployable signals only: `10` occupancy-grid frames, current
  smoothed command, current velocity, last action, four command-aligned
  occupancy-grid risk features, command history, velocity history, and action
  history. Nearest clearance and collision are not actor inputs.
- Critic input: policy input plus train-time privileged state: nearest
  clearance, collision flag, and global pose.
- Action output: body-frame `vx`, `vy`, `vyaw`.
- Physical limits are enforced before integration. Tracking smoothness is shaped
  in reward, while absolute feasibility comes from velocity and acceleration
  caps.

## Large scene mode

The default MuJoCo task now runs in an IsaacLab-style shared scene instead of
giving every vectorized env an isolated local obstacle cloud.

- One `36 m x 36 m` world is shared by all vectorized cars.
- Static obstacles are sampled once into the world with mixed circles, boxes,
  and wall segments.
- Dense regions receive a higher fraction of obstacles, while the rest of the
  world remains sparsely populated.
- The outside boundary is represented by thick segmented wall obstacles and an
  explicit border-clearance collision check.
- Other cars are inserted into each agent's local `80 x 80` occupancy grid as
  dynamic circular obstacles.
- Agent-agent, static-obstacle, and border collisions are logged separately and
  all contribute to the terminal collision penalty.
- In large-scene mode, timeout reset is disabled by default. Episodes reset only
  on collision/death or when cumulative reward has not improved for the
  configured stagnation window.

The policy still sees a body-centered local grid; the large scene changes the
source of that grid, not the network input contract.

## Current training profile

The MuJoCo owner config lives in
`conf/ppo/task/omni_car_grid_avoidance/mujoco.yaml`.

Current defaults in this branch:

- Trainer episode horizon statistic: `300 s`
- Grid history: `10` frames
- Observation history: `24` frames
- Command hold curriculum: short holds from `2-8 s`, long holds from `8-35 s`
  for `40%` of samples, plus `30%` explicit zero-input samples
- Command smoothing time constant: `0.55 s`
- PPO rollout: `128` envs, `32` steps per env
- Policy architecture: `OmniCarGridCNNGRUModel` with command-conditioned grid
  channels (`occupancy + command longitudinal + |lateral| + command norm` per
  grid frame, then GRU over 10 CNN features + MLP)
- The four scalar grid-derived risk features summarize command-corridor
  occupancy, left-side occupancy, right-side occupancy, and right-minus-left
  imbalance. They are deterministic projections of the same `80 x 80`
  occupancy grid, so they are deployable perception features rather than
  privileged simulator clearance or collision flags.
- The actor uses a command-skip residual form,
  `action = command + residual`. In the current default config the planar
  residual is interpreted in command coordinates as
  `[parallel_to_command, lateral_to_command]`, then rotated back to body axes.
  This keeps "slow down along the requested direction" and "sidestep" separate
  outputs.
- Body footprint: `0.56 m x 0.32 m`; local collision/nearest-clearance uses the
  rectangular footprint, while the `0.05 m` safety margin is applied to
  command-corridor and grid safety reasoning.
- Physical velocity caps: `2.0 / 1.0 / 2.0` for `vx / vy / vyaw`
- Scene: shared `36 m` world, `260` mixed static obstacles, heterogeneous dense
  regions, thick closed perimeter, and car-to-car dynamic obstacles
- Physical acceleration caps: `3.5 / 3.5 / 4.5` for `vx / vy / vyaw`

The command sampler balances the active-axis combinations across `vx`, `vy`,
and `vyaw` and samples low, medium, and high amplitude bands. Planar commands
with both `vx` and `vy` active are sampled by direction in normalized velocity
space, then scaled by the physical `2.0 m/s` longitudinal and `1.0 m/s`
lateral limits. This gives PPO coverage over pure longitudinal, pure lateral,
pure yaw, planar, yaw-coupled, and full omnidirectional commands instead of
relying on independent uniform axis sampling. Each vectorized agent owns an
independent hold timer, so commands do not all change on the same global step.
The default non-zero mode weights are `[0.10, 0.10, 0.10, 0.22, 0.10, 0.10,
0.28]` for `vx`, `vy`, `vyaw`, `vx+vy`, `vx+vyaw`, `vy+vyaw`, and
`vx+vy+vyaw`; this deliberately gives more mass to arbitrary planar directions
while retaining pure-axis and yaw-coupled cases. Zero-command samples are kept
as true zero-input windows instead of only slow smoothing tails: when the raw
sampled command norm is within `zero_snap_norm`, the smoothed command snaps to
zero and the observation history can contain sustained all-zero commands.

The local obstacle curriculum reserves a controlled slice of resets for
command-relative geometry: `25%` clear command corridors, `45%` front blockers,
`20%` side-wall cases, and the rest unconstrained random obstacle fields. Clear
corridors keep obstacles outside the commanded swept path so long-hold commands
continue to provide strong follow-the-input samples. Front blockers are
intentionally mixed across circles, boxes, and wall-like rectangles so the
policy sees both single-object avoidance and true corridor closure under
sustained forward commands.

Before launching a long run, inspect the actual input distribution with:

```bash
uv run scripts/analyze_omni_car_input_coverage.py \
  --num-envs 128 \
  --num-steps 512 \
  --command-samples 8192 \
  --large-scene off
```

This reports standalone command balance, hold-duration percentiles, and rollout
fractions for blocked, partially blocked, and clear command directions. Treat it
as a data check for whether the training stream really contains enough zero
input, long same-direction commands, all active-axis modes, blocked forward
pushes, and near-obstacle command-grid pairs.

## Xbox human-command mode

Set `env.human_command.enabled=true` to replace the sampled user command for one
randomly selected vectorized agent with live Xbox input. This does not bypass
the policy: the controller only writes the desired command `[vx, vy, vyaw]`;
the actor still outputs the safe executed action.

Default mapping:

- Left stick vertical -> `vx` in `[-2.0, 2.0]`, inverted so pushing forward is
  positive `vx`.
- Left stick horizontal -> `vy` in `[-1.0, 1.0]`.
- Right stick horizontal -> `vyaw` in `[-2.0, 2.0]`.

Start a local interactive training run with the connected controller:

```bash
uv run train --algo ppo --task omni_car_grid_avoidance --sim mujoco \
  env.human_command.enabled=true \
  env.human_command.render_enabled=true \
  training.no_play=true \
  training.play_render_mode=none
```

The live training viewer is separate from PPO playback. It refreshes during
rollout collection and tracks the human-command agent directly. Closing the
viewer window hides visualization but lets training continue.

If the Xbox axis order differs on your machine, inspect it with:

```bash
uv run python scripts/check_xbox_controller.py
```

Then override the axis id, for example:

```bash
uv run train --algo ppo --task omni_car_grid_avoidance --sim mujoco \
  env.human_command.enabled=true \
  env.human_command.axis_vyaw=3
```

`env.human_command.replay_fanout=N` can copy the same human command stream into
`N` extra randomly selected agents. Keep it at `0` for a pure one-agent manual
signal, or raise it when you want the same human command waveform sampled
against more obstacle layouts.

To continue low-latency manual training from a selected checkpoint, place or
symlink that checkpoint as `artifacts/omni_car/checkpoints/best.pt`, then run:

```bash
uv run scripts/train_omni_car_human_from_best.py
```

The launcher resumes from `best.pt`, enables Xbox command input and the live
training viewer, and defaults to a light local profile: `8` envs, `8` rollout
steps, `1` learning epoch, and `1` minibatch. Override those flags when you want
more throughput and can tolerate longer viewer stalls.

Reward shaping emphasizes these signals:

1. Reward positive output-velocity projection along the commanded planar
   direction.
2. Penalize motion that reduces nearest clearance, plus a one-step prediction of
   whether the raw policy target would reduce clearance. This makes pushing into
   a wall expensive without giving the policy an action-level geometry gate, and
   without charging the same cost for wall-parallel motion that preserves
   clearance.
3. Reward increasing nearest clearance during active planar-command windows.
   This gives the policy positive credit for sliding away from a near obstacle
   when there is space, while zero-command windows still prefer idle output.
4. Reward lateral escape when the commanded corridor is blocked, the executed
   action keeps positive forward projection low, and the car is not actively
   reducing clearance. This supplies the first-step credit needed for sliding
   around a centered front obstacle without adding a geometry gate.
5. Penalize positive projection along the commanded direction when the swept
   command corridor is blocked. This is a continuous reward cost, not a hard
   action gate; side-wall forward motion remains learnable when the corridor
   ahead is open.
6. Penalize residual planar speed when the commanded corridor is tightly
   blocked. This targets centered front blockers where left/right escape is
   symmetric and the deterministic policy mean should learn to stop instead of
   creeping forward until collision.
7. Reward low positive forward projection when the commanded corridor is
   blocked. This gives PPO direct credit for stopping or reducing forward push
   before collision, while still leaving wall-parallel sliding to be learned
   from clearance and off-axis tradeoffs.
8. Penalize any output velocity during zero-command windows, including both the
   raw policy target and the physically limited executed velocity, so the
   learned optimum is idle when the operator is idle.
9. Penalize velocity far from the commanded planar direction with an explicit
   off-axis term.
10. Penalize reverse motion against the commanded planar direction.
11. Track commanded yaw intent independently, but only when the operator gives a
   non-zero yaw command so zero-yaw commands do not create a constant reward for
   standing still.
12. Penalize yaw output when the user did not command yaw, including during
   planar-only commands.
13. Improve response speed whenever the executed action reduces command-tracking
   error.
14. Penalize per-axis tracking, action diff, and jerk independently for `vx`,
   `vy`, and `vyaw`.
15. Preserve clearance and heavily punish collision.

Safety and idle penalties use bounded normalized costs for the one-step target
collision estimate, clearance-closing speed, and zero-command raw target
motion. This keeps a single stochastic policy sample from dominating an entire
PPO update while preserving the ordering that unsafe target motion is worse
than stopping and zero-input motion is worse than idle output.

The actor does not receive privileged nearest-clearance or collision flags;
those remain critic/logging signals only. The reward can still use privileged
training information, but the positive planar intent term is now tied to the
actual projection onto the user command, so pure side slip or reverse output
does not earn forward-intent reward.
The executed action is not clamped by obstacle geometry, and no
hard command-direction feasibility switch is used inside the reward. Obstacle
avoidance is learned through PPO from projection rewards, smoothness penalties,
clearance-closing costs, clearance-opening and blocked-lateral-escape rewards,
blocked-corridor projection/speed costs and stop rewards,
collision costs, and curriculum signals. The current
reward balance makes clear-command projection, response, and per-axis tracking
large enough to compete with smoothness penalties, while making off-axis drift,
clearance loss, and collision expensive enough that sliding into walls or
pushing through blockers is not a profitable local optimum. Entropy is kept low
enough that zero-command samples can converge to a stationary mean action.

Recent diagnostics showed that clear-scene command following and blocked-path
safety are both learnable, but the previous command-agnostic grid encoder could
let one behavior overwrite the other. The current actor keeps the same
non-privileged observation contract, but injects the operator's current planar
command into the grid encoder as directional coordinate channels. This gives the
CNN direct access to "obstacle along commanded corridor" versus "side wall while
forward path is open" without using nearest-clearance or collision flags.

The recommended training path is therefore an alternating curriculum: run short
safety, clear, and balanced phases, scan checkpoints after each phase, and
continue from the checkpoint with the best behavior coverage before
phase-specific probe cost. This prevents low-collision but low-speed
checkpoints from overwriting clear command-following. This is still an RL
curriculum; no geometry-level action gate is inserted.

The command-conditioned `CNN + GRU` model is intentionally incompatible with
older `CNN + MLP` checkpoints and earlier command-agnostic `CNN + GRU`
checkpoints. Historical checkpoint manifests remain useful for record keeping,
but this branch needs a fresh training run before a checkpoint can be loaded
with the current default config.

## Supervised obstacle repair

Recent command-frame experiments show a consistent failure mode: PPO from the
command-skip initialization learns clear-scene command following, zero input,
and yaw control quickly, but it does not reliably discover the large negative
parallel residual needed for front blockers or the lateral residual needed for
wall sliding. The supervised oracle scenarios are useful, but the optimizer
shape matters.

Do not use `--balanced-batch` as the default repair path for this task. It
accumulates all clear, front-blocked, and wall-slide scenario gradients into one
optimizer step. That can make mutually opposed residual targets cancel each
other. On the `v33b` SiLU capacity model, balanced BC from final PPO degraded
the full gate to clear `17/48`, front `1/16`, side-wall `2/16`, while the same
scenario set trained with ordinary per-scenario SGD from `model_0.pt` reached
clear `48/48`, front `9/16`, side-wall `15/16`, yaw `2/2`, and zero `1/1`.

The current recommended repair sequence is scripted so checkpoint lineage and
gate results stay in one manifest:

```bash
uv run scripts/run_omni_car_sgd_bc_curriculum.py \
  --load-run logs/rsl_rl_ppo/OmniCarGridAvoidance/<run>/model_0.pt \
  --name-prefix silu_capacity_vNN \
  --device cuda:0
```

The launcher runs full-oracle per-scenario SGD BC first, gates that checkpoint
with `scripts/evaluate_omni_car_robustness.py`, and only runs the second
front-focused repair stage if the first gate does not pass. The equivalent
manual commands are:

```bash
uv run scripts/train_omni_car_wall_slide_bc.py \
  --load-run logs/rsl_rl_ppo/OmniCarGridAvoidance/<run>/model_0.pt \
  --output artifacts/omni_car/checkpoints/silu_capacity_v33b_init_sgd_bc.pt \
  --num-envs 64 \
  --iterations 2500 \
  --rollout-steps 2 \
  --rollout-actions target \
  --learning-rate 1e-4 \
  --device cuda:0 \
  --progress-interval 250

uv run scripts/train_omni_car_wall_slide_bc.py \
  --load-run artifacts/omni_car/checkpoints/silu_capacity_v33b_init_sgd_bc.pt \
  --output artifacts/omni_car/checkpoints/silu_capacity_v33b_front_repair.pt \
  --num-envs 64 \
  --iterations 1400 \
  --rollout-steps 2 \
  --rollout-actions target \
  --learning-rate 5e-5 \
  --scenario-group base \
  --scenario-group directional_clear \
  --scenario-group directional_front \
  --device cuda:0 \
  --progress-interval 200
```

Use the broad robustness gate for promotion:

```bash
uv run scripts/evaluate_omni_car_robustness.py \
  --load-run artifacts/omni_car/checkpoints/silu_capacity_v33b_front_repair.pt \
  --num-envs 16 \
  --num-steps 128 \
  --directions 16 \
  --device cuda:0 \
  --json
```

The best current remote checkpoint from this path is
`artifacts/omni_car/checkpoints/silu_capacity_v34b_search_c07_full_it500_lr1em05_s372.pt`
on the RTX 5070 Ti host. Its `16` direction, `128` step gate result is clear
`48/48`, front-blocked `16/16`, side-wall `15/16`, yaw `2/2`, zero `1/1`, with
zero collision in every category. It still misses strict promotion because
`left_wall_dir_02` has projection `0.153`, below the `0.18` threshold. Treat
this as the current best candidate, not as proof of completion.

An exact low-LR repair on `directional_left_wall_04` from that checkpoint
(`silu_capacity_v34d_leftwall_exact.pt`) did not fix the gate and caused a
front-blocked regression with a small collision fraction. Do not promote that
checkpoint.

The wall-slide oracle now uses a projection-dominant target: move along the wall
at `0.65 m/s` and add only `0.25 m/s` away from the wall. This better matches
the desired "slide along the wall" behavior than the older equal
forward/lateral target. For narrow repairs, use `--scenario-weight` to boost the
failed scenario while keeping zero/front/clear rehearsal active, for example:

```bash
uv run scripts/train_omni_car_wall_slide_bc.py \
  --load-run artifacts/omni_car/checkpoints/silu_capacity_v34b_search_c07_full_it500_lr1em05_s372.pt \
  --output artifacts/omni_car/checkpoints/silu_capacity_v35_leftwall_rehearsal.pt \
  --num-envs 64 \
  --iterations 600 \
  --rollout-steps 2 \
  --rollout-actions target \
  --learning-rate 2e-5 \
  --seed 381 \
  --scenario-group base \
  --scenario-group directional_clear \
  --scenario-group directional_front \
  --scenario directional_left_wall_04 \
  --scenario-weight directional_left_wall_04=8 \
  --device cuda:0
```

The standard 16-direction broad gate can pass while denser arbitrary-direction
front blockers still fail. To train the intermediate directions exposed by a
32-direction gate, include the dense oracle groups:

```bash
uv run scripts/train_omni_car_wall_slide_bc.py \
  --load-run artifacts/omni_car/checkpoints/silu_capacity_v35d_front_balanced_rehearsal.pt \
  --output artifacts/omni_car/checkpoints/silu_capacity_vNN_dense_front_rehearsal.pt \
  --num-envs 64 \
  --iterations 16 \
  --rollout-steps 2 \
  --rollout-actions target \
  --balanced-batch \
  --learning-rate 1e-6 \
  --scenario-group base \
  --scenario-group directional_clear \
  --scenario-group directional32_clear \
  --scenario-group directional32_front \
  --scenario directional_left_wall_04 \
  --scenario-weight directional_left_wall_04=3 \
  --device cuda:0
```

When a two-stage repair run still misses the strict gate, run a bounded
candidate search instead of manually promoting the latest checkpoint:

```bash
uv run scripts/search_omni_car_repair_candidates.py \
  --load-run artifacts/omni_car/checkpoints/silu_capacity_vNN_init_sgd_bc.pt \
  --output logs/omni_car_repair_search/vNN_manifest.json \
  --artifact-dir artifacts/omni_car/checkpoints \
  --name-prefix silu_capacity_vNN_search \
  --profiles front,front_wall \
  --seeds 361,362,363 \
  --learning-rates 5e-5,3e-5 \
  --iterations 900,1400 \
  --device cuda:0 \
  --gate-device cuda:0
```

The search runner evaluates every candidate with the same broad robustness gate
and records a ranked `best` entry in the manifest. The ranking first prefers a
strict pass, then fewer failed scenarios, then better front-blocked, side-wall,
clear, zero, and yaw coverage, with collision used as a tie-breaker. A search
completion is not the same as model completion; only a strict gate pass should
be promoted to `best.pt`.

## Teacher-student distillation path

The current deployable policy should remain a compact non-privileged actor, but
the teacher used to generate smoother and more globally consistent labels can
be larger. The clean split is:

- Deployable teacher: uses only actor-available signals: occupancy-grid
  history, smoothed command and command history, velocity history, and last
  action/action history. It can use a heavier temporal model such as shared CNN
  grid encoding followed by a Transformer over grid, command, velocity, and
  action tokens.
- Privileged teacher: may also use simulator-only training signals such as
  nearest clearance, collision state, global pose, or future rollout risk. Its
  outputs can regularize labels or value targets, but those privileged signals
  must not become direct student inputs.
- Student: keeps the deployment contract of the current `CNN + GRU + MLP`
  actor and is evaluated only through non-privileged observation input plus the
  broad behavior gates.

A suitable teacher architecture is:

```text
grid history [T, 80, 80]
  -> shared CNN grid encoder
  -> grid feature tokens
  + command / velocity / action history tokens
  -> temporal Transformer
  -> MLP head for [vx, vy, vyaw]
```

This branch provides that deployable teacher as
`unilab.algos.torch.omni_car:OmniCarGridCNNTransformerModel`. It uses the same
actor observation contract and the same command-skip residual action head as
the default student, but replaces the GRU temporal fusion with a Transformer
encoder. Launch it through the checked wrapper rather than hand-writing the
Hydra override set:

```bash
uv run scripts/train_omni_car_transformer_teacher.py \
  --run-name transformer_teacher_v01 \
  --num-envs 128 \
  --num-steps-per-env 32 \
  --max-iterations 260 \
  --transformer-dim 256 \
  --transformer-heads 8 \
  --transformer-layers 3 \
  --transformer-ff-dim 768 \
  --device cuda:0
```

The wrapper expands to the required `class_name` overrides and `+`-prefixed
Transformer-specific Hydra keys because the base OmniCar PPO config is
structured around the GRU student. Existing `gru_hidden_dim` values are
accepted by the model as a compatibility alias for `transformer_dim` when no
explicit Transformer dimension is provided.

The distillation loss should be axis-aware rather than a single scalar MSE:
Huber or MSE action imitation per `vx`, `vy`, and `vyaw`; per-axis diff and
jerk imitation; zero-input stop loss; command-direction projection and
off-axis penalties; and optional KL if the teacher predicts a Gaussian action
distribution. The data mix still needs the same coverage guarantees as PPO:
true zero-input windows, long same-direction commands over arbitrary planar
directions and speeds, yaw-only and yaw-coupled commands, front blockers, side
walls, and mixed obstacle types.

Once a Transformer teacher checkpoint exists, distill it into the current
deployable CNN-GRU student with:

```bash
uv run scripts/distill_omni_car_transformer_teacher.py \
  --teacher-load-run logs/rsl_rl_ppo/OmniCarGridAvoidance/<teacher_run>/model_<N>.pt \
  --student-load-run artifacts/omni_car/checkpoints/silu_capacity_v34b_search_c07_full_it500_lr1em05_s372.pt \
  --output artifacts/omni_car/checkpoints/silu_capacity_vNN_distilled_student.pt \
  --num-envs 64 \
  --iterations 2000 \
  --learning-rate 3e-5 \
  --rollout-source teacher \
  --device cuda:0
```

The distiller loads both checkpoints with their own model configs, rolls out a
shared non-privileged OmniCar observation stream, and trains only the student
actor. The default loss combines axis-weighted action imitation, diff
imitation, jerk imitation, and an explicit zero-command stop term. The output
checkpoint must still pass the same broad robustness gate before promotion.

Distillation does not replace the safety gate. A distilled student is promoted
only after passing the broad robustness gate for zero input, clear following,
front-blocked stop, wall sliding, yaw, and collision behavior.

For checkpoint-level behavior gates, run:

```bash
uv run scripts/evaluate_omni_car_behaviors.py \
  --load-run artifacts/omni_car/checkpoints/best.pt \
  --num-envs 16 \
  --num-steps 96 \
  --strict
```

The behavior probe forces controlled scenarios for zero input, clear forward
and diagonal following, front-blocked push, and a near-wall forward command.
It reports planar speed, yaw drift, command projection, off-axis motion,
collision fraction, clearance risk, and reward components. A checkpoint should pass this
probe before being treated as a candidate for manual Xbox fine-tuning.

To automate the alternating PPO phases and checkpoint selection:

```bash
uv run scripts/run_omni_car_alternating_curriculum.py \
  --load-run /absolute/path/to/model.pt \
  --run-prefix remote_auto_curriculum \
  --output logs/remote_auto_curriculum/manifest.json \
  --rounds 2 \
  --phase-iterations 75 \
  --num-envs 128 \
  --num-steps-per-env 32
```

The orchestrator writes a manifest after every phase, including the run
directory, scanned checkpoints, selected checkpoint path, behavior cost, and
whether all strict behavior gates passed.

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

Use `--summary-window` with `status` to include a JSON summary from the tmux log:

```bash
uv run scripts/remote_omni_car_tmux.py status \
  --session omni-car-large-scene-c22 \
  --summary-window 20
```

Summarize a long tmux log into a small JSON artifact before committing progress
snapshots:

```bash
uv run scripts/summarize_omni_car_training_log.py \
  artifacts/omni_car/remote_large_scene_c22_tmux.log \
  --window 20 \
  --output artifacts/omni_car/remote_large_scene_c22_progress.json
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

The CPU/sync-path validation run `remote_tmux_cpu_path_fcbbd4` completed on
commit `fcbbd4c16cf1192b5e16c218452771d289690a59` (`2026-06-27 23:45:31
+08:00` to `2026-06-28 00:01:45 +08:00`). It used the same `128` envs,
`32` steps/env, `3000` iterations profile and mainly validates the current
remote sync / tmux / CPU environment path. Its final `model_2999.pt` has
SHA-256 `d5f34aefd71478255fdf2621b862ef49cdf70d06340da43c78da3bea028bfb9d`.
The scan selected `2999` with `collision_fraction=0.0138` but
`omni_car/tracking_error=0.4262`; the collision gate passed, the tracking gate
did not, so it does not replace the current selected checkpoint. Evidence is
stored in `artifacts/omni_car/remote_tmux_cpu_path_fcbbd4_checkpoint_scan.json`.

The shared large-scene run `remote_large_scene_c22_3000` completed on the
remote RTX 5070 Ti host (`2026-06-28 00:28:15 +08:00` to `01:06:36 +08:00`),
with `128` envs, `32` steps/env, and `3000` iterations. Its final training
window reached `omni_car/collision_rate=0.003725` and
`omni_car/tracking_error=0.426535`. A `64` env, `512` step checkpoint scan
selected `model_2900.pt` as the best tradeoff: `collision_fraction=0.003845`,
`omni_car/tracking_error=0.393691`, and selection score `0.476784`. The final
`model_2999.pt` has the lowest scanned collision rate (`0.002960`) but higher
tracking error (`0.408384`), so `model_2900.pt` is the current large-scene
viewer/eval default and `model_2999.pt` is the low-collision alternate.
Evidence is stored in
`artifacts/omni_car/remote_large_scene_c22_progress.json` and
`artifacts/omni_car/remote_large_scene_c22_final_checkpoint_scan.json`.

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

To profile the current shared large-scene mode instead of the legacy
per-env-local obstacle mode, pass `--large-scene`:

```bash
uv run scripts/benchmark_omni_car_env.py \
  --num-envs 128 \
  --steps 40 \
  --warmup-steps 5 \
  --action-mode zero \
  --large-scene \
  --profile-top 18
```

On the RTX 5070 Ti host, commit `a35f108d` measured the large-scene path at
about `14.98 ms/step` for `128` envs, `372` total scene obstacles, `72` local
static obstacles per agent, and `24` dynamic agents per agent. The matching
training run spent roughly `0.67-0.71 s` in rollout collection versus
`0.07-0.08 s` in PPO learning per iteration. That makes the CPU occupancy-grid
raster path the current bottleneck; the CUDA learner is not saturated by neural
network work.

GPU acceleration can help, but only if the observation path becomes
torch-native. A partial CUDA port that rasterizes the grid on GPU, copies it
back to NumPy for the env API, and then lets the runner copy it back to CUDA for
PPO will likely lose much of the gain to synchronization and PCIe copies. The
right performance direction is to keep scene obstacles, agent poses, occupancy
grids, and policy observations as torch tensors on the learner device, then
feed the PPO actor/critic without CPU round-trips.

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
- local clearance is computed in one batched vectorized pass across all envs
  against the rectangular body footprint
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

For the current large-scene checkpoint, first cache the default checkpoint from
the remote manifest, then open the native viewer:

```bash
uv run scripts/fetch_omni_car_checkpoint.py
uv run scripts/view_omni_car_checkpoint.py \
  --load-run artifacts/omni_car/checkpoints/remote_large_scene_c22_model_2900.pt \
  --device cpu
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
commands. To cache or verify this older axis-track checkpoint locally for the
native viewer, pass the manifest explicitly:

```bash
uv run scripts/fetch_omni_car_checkpoint.py \
  --manifest artifacts/omni_car/axis_track_2999_checkpoint_manifest.json
uv run scripts/fetch_omni_car_checkpoint.py \
  --manifest artifacts/omni_car/axis_track_2999_checkpoint_manifest.json \
  --verify-only
uv run scripts/view_omni_car_checkpoint.py --load-run artifacts/omni_car/checkpoints/model_2999.pt --device cpu
```

The current large-scene checkpoint is also not committed. Its manifest is stored
in `artifacts/omni_car/remote_large_scene_c22_2900_checkpoint_manifest.json`;
this is the default manifest for `scripts/fetch_omni_car_checkpoint.py`, and it
uses a distinct local cache name so it can coexist with other `model_*.pt`
files:

```bash
uv run scripts/fetch_omni_car_checkpoint.py
uv run scripts/fetch_omni_car_checkpoint.py --verify-only
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
