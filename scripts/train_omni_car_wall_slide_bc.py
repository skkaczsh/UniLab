#!/usr/bin/env python3
"""Supervised behavior correction for OmniCar PPO actor checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.evaluate_omni_car_checkpoint as checkpoint_eval
import scripts.train_rsl_rl as train_rsl_rl
from scripts.evaluate_omni_car_behaviors import BehaviorScenario, _apply_scenario
from unilab.training.experiment import patch_rsl_rl_resume_state

MAX_X_SPEED = 2.0
MAX_Y_SPEED = 1.0


@dataclass(frozen=True)
class OracleScenario:
    name: str
    behavior: BehaviorScenario
    target_action: tuple[float, float, float]
    weight: float = 1.0


def _unit(angle: float) -> np.ndarray:
    return np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float64)


def _lateral(direction: np.ndarray) -> np.ndarray:
    return np.asarray([-direction[1], direction[0]], dtype=np.float64)


def _directional_command(angle: float, fraction: float) -> tuple[float, float, float]:
    direction = _unit(angle)
    axis_scale = min(
        MAX_X_SPEED / max(abs(float(direction[0])), 1e-6),
        MAX_Y_SPEED / max(abs(float(direction[1])), 1e-6),
    )
    xy = direction * axis_scale * float(fraction)
    return (float(xy[0]), float(xy[1]), 0.0)


def _point(direction: np.ndarray, forward: float, lateral: float) -> tuple[float, float]:
    xy = direction * float(forward) + _lateral(direction) * float(lateral)
    return (float(xy[0]), float(xy[1]))


def _yaw_from_direction(direction: np.ndarray, lateral_axis: bool = False) -> float:
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    if lateral_axis:
        yaw += math.pi * 0.5
    return float(yaw)


def _clip_target_xy(xy: np.ndarray) -> tuple[float, float, float]:
    clipped = np.clip(
        np.asarray(xy, dtype=np.float64),
        np.asarray([-MAX_X_SPEED, -MAX_Y_SPEED], dtype=np.float64),
        np.asarray([MAX_X_SPEED, MAX_Y_SPEED], dtype=np.float64),
    )
    return (float(clipped[0]), float(clipped[1]), 0.0)


BASE_ORACLE_SCENARIOS: tuple[OracleScenario, ...] = (
    OracleScenario(
        name="zero_input_hold",
        behavior=BehaviorScenario(name="zero_input_hold", command=(0.0, 0.0, 0.0)),
        target_action=(0.0, 0.0, 0.0),
        weight=8.0,
    ),
    OracleScenario(
        name="clear_forward_follow",
        behavior=BehaviorScenario(name="clear_forward_follow", command=(1.0, 0.0, 0.0)),
        target_action=(1.0, 0.0, 0.0),
        weight=3.0,
    ),
    OracleScenario(
        name="clear_diagonal_follow",
        behavior=BehaviorScenario(name="clear_diagonal_follow", command=(0.8, 0.4, 0.0)),
        target_action=(0.8, 0.4, 0.0),
        weight=3.0,
    ),
    OracleScenario(
        name="clear_left_follow",
        behavior=BehaviorScenario(name="clear_left_follow", command=(0.0, 0.7, 0.0)),
        target_action=(0.0, 0.7, 0.0),
        weight=3.0,
    ),
    OracleScenario(
        name="clear_right_follow",
        behavior=BehaviorScenario(name="clear_right_follow", command=(0.0, -0.7, 0.0)),
        target_action=(0.0, -0.7, 0.0),
        weight=3.0,
    ),
    OracleScenario(
        name="clear_backward_follow",
        behavior=BehaviorScenario(name="clear_backward_follow", command=(-0.8, 0.0, 0.0)),
        target_action=(-0.8, 0.0, 0.0),
        weight=2.5,
    ),
    OracleScenario(
        name="clear_slow_diagonal_follow",
        behavior=BehaviorScenario(
            name="clear_slow_diagonal_follow",
            command=(0.35, -0.25, 0.0),
        ),
        target_action=(0.35, -0.25, 0.0),
        weight=2.0,
    ),
    OracleScenario(
        name="yaw_only_follow",
        behavior=BehaviorScenario(name="yaw_only_follow", command=(0.0, 0.0, 1.0)),
        target_action=(0.0, 0.0, 1.0),
        weight=5.0,
    ),
    OracleScenario(
        name="yaw_negative_follow",
        behavior=BehaviorScenario(name="yaw_negative_follow", command=(0.0, 0.0, -1.0)),
        target_action=(0.0, 0.0, -1.0),
        weight=5.0,
    ),
    OracleScenario(
        name="front_blocked_stop",
        behavior=BehaviorScenario(
            name="front_blocked_stop",
            command=(1.0, 0.0, 0.0),
            obstacle_xy=((0.70, 0.0),),
            obstacle_radius=(0.22,),
        ),
        target_action=(0.0, 0.0, 0.0),
        weight=12.0,
    ),
    OracleScenario(
        name="right_wall_slide",
        behavior=BehaviorScenario(
            name="right_wall_slide",
            command=(1.0, 0.0, 0.0),
            obstacle_xy=((0.60, -0.34), (1.05, -0.34), (1.50, -0.34)),
            obstacle_radius=(0.22, 0.22, 0.22),
        ),
        target_action=(0.45, 0.45, 0.0),
        weight=8.0,
    ),
    OracleScenario(
        name="left_wall_slide",
        behavior=BehaviorScenario(
            name="left_wall_slide",
            command=(1.0, 0.0, 0.0),
            obstacle_xy=((0.60, 0.34), (1.05, 0.34), (1.50, 0.34)),
            obstacle_radius=(0.22, 0.22, 0.22),
        ),
        target_action=(0.45, -0.45, 0.0),
        weight=8.0,
    ),
)


def _make_directional_oracle_scenarios(directions: int = 16) -> tuple[OracleScenario, ...]:
    scenarios: list[OracleScenario] = []
    angles = [2.0 * math.pi * i / int(directions) for i in range(int(directions))]
    for angle_index, angle in enumerate(angles):
        direction = _unit(angle)
        for speed_name, fraction, weight in (
            ("slow", 0.30, 1.6),
            ("mid", 0.60, 2.0),
            ("fast", 0.85, 2.2),
        ):
            command = _directional_command(angle, fraction)
            scenarios.append(
                OracleScenario(
                    name=f"directional_clear_{angle_index:02d}_{speed_name}",
                    behavior=BehaviorScenario(
                        name=f"directional_clear_{angle_index:02d}_{speed_name}",
                        command=command,
                    ),
                    target_action=command,
                    weight=weight,
                )
            )
        shape = ("circle", "box", "wall")[angle_index % 3]
        scenarios.append(
            OracleScenario(
                name=f"directional_front_stop_{angle_index:02d}_{shape}",
                behavior=BehaviorScenario(
                    name=f"directional_front_stop_{angle_index:02d}_{shape}",
                    command=_directional_command(angle, 0.65),
                    obstacle_xy=(_point(direction, 0.70, 0.0),),
                    obstacle_radius=(0.22,),
                    obstacle_type=(shape,),
                    obstacle_half_extents=((0.22, 0.18),),
                    obstacle_yaw=(_yaw_from_direction(direction, lateral_axis=True),),
                ),
                target_action=(0.0, 0.0, 0.0),
                weight=10.0,
            )
        )
        if angle_index % 2 == 0:
            for side_name, side_sign in (("right", -1.0), ("left", 1.0)):
                lateral_offset = 0.34 * side_sign
                away = -side_sign
                target_xy = direction * 0.45 + _lateral(direction) * (0.45 * away)
                scenarios.append(
                    OracleScenario(
                        name=f"directional_{side_name}_wall_{angle_index:02d}",
                        behavior=BehaviorScenario(
                            name=f"directional_{side_name}_wall_{angle_index:02d}",
                            command=_directional_command(angle, 0.65),
                            obstacle_xy=tuple(
                                _point(direction, forward, lateral_offset)
                                for forward in (0.60, 1.05, 1.50)
                            ),
                            obstacle_radius=(0.22, 0.22, 0.22),
                            obstacle_type=("circle", "wall", "circle"),
                            obstacle_half_extents=((0.22, 0.22), (0.36, 0.08), (0.22, 0.22)),
                            obstacle_yaw=(0.0, _yaw_from_direction(direction), 0.0),
                        ),
                        target_action=_clip_target_xy(target_xy),
                        weight=6.0,
                    )
                )
    return tuple(scenarios)


DIRECTIONAL_ORACLE_SCENARIOS = _make_directional_oracle_scenarios()
ORACLE_SCENARIOS: tuple[OracleScenario, ...] = (
    BASE_ORACLE_SCENARIOS + DIRECTIONAL_ORACLE_SCENARIOS
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Checkpoint path to correct.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint id/name.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument(
        "--rollout-actions",
        choices=("policy", "target"),
        default="policy",
        help=(
            "Action source used to advance the supervised rollout. "
            "The default trains on the policy's own closed-loop state distribution."
        ),
    )
    parser.add_argument(
        "--balanced-batch",
        action="store_true",
        help=(
            "Accumulate one optimizer update across every oracle scenario each iteration. "
            "Useful for diagnostics, but per-scenario SGD is usually better for the "
            "clear/front/wall oracle mix because full-batch gradients can cancel."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.name for scenario in ORACLE_SCENARIOS),
        help="Limit BC to one or more exact oracle scenarios. Defaults to all scenarios.",
    )
    parser.add_argument(
        "--scenario-group",
        action="append",
        choices=(
            "base",
            "directional",
            "directional_clear",
            "directional_front",
            "directional_wall",
        ),
        help="Add a named oracle scenario group to the selected BC set.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=0,
        help="Print progress every N optimizer iterations. Disabled by default.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _scenario_group_names(groups: Sequence[str] | None) -> set[str]:
    if not groups:
        return set()
    selected: set[str] = set()
    for group in groups:
        if group == "base":
            selected.update(scenario.name for scenario in BASE_ORACLE_SCENARIOS)
        elif group == "directional":
            selected.update(scenario.name for scenario in DIRECTIONAL_ORACLE_SCENARIOS)
        elif group == "directional_clear":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL_ORACLE_SCENARIOS
                if scenario.name.startswith("directional_clear_")
            )
        elif group == "directional_front":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL_ORACLE_SCENARIOS
                if scenario.name.startswith("directional_front_stop_")
            )
        elif group == "directional_wall":
            selected.update(
                scenario.name
                for scenario in DIRECTIONAL_ORACLE_SCENARIOS
                if "_wall_" in scenario.name
            )
        else:
            raise ValueError(f"Unknown scenario group: {group}")
    return selected


def _selected_scenarios(
    names: Sequence[str] | None, groups: Sequence[str] | None = None
) -> tuple[OracleScenario, ...]:
    selected_names = set(names or ())
    selected_names.update(_scenario_group_names(groups))
    if not selected_names:
        return ORACLE_SCENARIOS
    selected = tuple(scenario for scenario in ORACLE_SCENARIOS if scenario.name in selected_names)
    if len(selected) != len(selected_names):
        known = {scenario.name for scenario in ORACLE_SCENARIOS}
        missing = sorted(selected_names - known)
        raise ValueError(f"Unknown oracle scenario(s): {missing}")
    return selected


def _scenario_probabilities(scenarios: Sequence[OracleScenario]) -> np.ndarray:
    weights = np.asarray([scenario.weight for scenario in scenarios], dtype=np.float64)
    return weights / np.sum(weights)


def _shuffled_scenarios(
    rng: np.random.Generator, scenarios: Sequence[OracleScenario]
) -> list[OracleScenario]:
    order = rng.permutation(len(scenarios))
    return [scenarios[int(index)] for index in order]


def _target_tensor(
    scenario: OracleScenario, *, num_envs: int, device: str | torch.device
) -> torch.Tensor:
    return torch.tensor(scenario.target_action, dtype=torch.float32, device=device).repeat(
        int(num_envs), 1
    )


def _make_runner(args: argparse.Namespace) -> tuple[Any, Any, Any, Path, str]:
    train_rsl_rl.ensure_registries()
    cfg = checkpoint_eval._compose_cfg(args)
    device = checkpoint_eval._resolve_device(args.device)
    rl_cfg = train_rsl_rl._algo_config_dict(cfg)
    wrapper_cls = train_rsl_rl._resolve_ppo_wrapper_cls(rl_cfg)
    env_cfg_override = train_rsl_rl.build_ppo_play_env_cfg_override(cfg)
    env_cfg_override.update(
        {
            "seed": int(args.seed),
            "large_scene": {"enabled": False},
            "obstacles": {
                "count": 4,
                "circle_fraction": 1.0,
                "box_fraction": 0.0,
                "wall_fraction": 0.0,
            },
            "human_command": {"enabled": False, "render_enabled": False},
        }
    )
    env = train_rsl_rl.create_env(cfg, num_envs=int(args.num_envs), env_cfg_override=env_cfg_override)
    wrapped_env = wrapper_cls(env, device=device)
    train_cfg = train_rsl_rl.normalize_ppo_train_cfg(rl_cfg)
    train_rsl_rl.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    train_cfg.setdefault("runner", {})
    train_cfg["runner"]["logger"] = "none"
    train_cfg["logger"] = "none"
    patch_rsl_rl_resume_state()
    runner = train_rsl_rl.OnPolicyRunner(wrapped_env, train_cfg, log_dir=None, device=device)
    load_path, _ = train_rsl_rl.parse_checkpoint_path(cfg, root_dir=train_rsl_rl.ROOT_DIR)
    if load_path is None or not load_path.exists():
        raise FileNotFoundError(
            f"Could not resolve checkpoint from load_run={args.load_run!r} checkpoint={args.checkpoint!r}"
        )
    with train_rsl_rl.policy_load_dim_guard(
        env_obs_dim=getattr(wrapped_env, "num_obs", None),
        env_action_dim=getattr(wrapped_env, "num_actions", None),
        algo_name="ppo",
    ):
        runner.load(str(load_path), map_location=device)
    return runner, env, wrapped_env, load_path, device


def _save_corrected_checkpoint(runner: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    optimizer = getattr(runner.alg, "optimizer", None)
    if optimizer is not None:
        optimizer.state.clear()
    payload = runner.alg.save()
    payload["iter"] = int(getattr(runner, "current_learning_iteration", 0))
    payload["infos"] = None
    logger = getattr(runner, "logger", None)
    payload["unilab_logger_state"] = {
        "tot_time": float(getattr(logger, "tot_time", 0.0)),
        "tot_timesteps": int(getattr(logger, "tot_timesteps", 0)),
    }
    torch.save(payload, output)


def train_wall_slide_bc(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    runner, env, wrapped_env, load_path, device = _make_runner(args)
    policy = runner.alg.get_policy()
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.learning_rate))
    scenarios = _selected_scenarios(args.scenario, args.scenario_group)
    probabilities = _scenario_probabilities(scenarios)
    loss_history: list[float] = []
    scenario_counts = {scenario.name: 0 for scenario in scenarios}
    started_at = time.time()
    try:
        wrapped_env.reset()
        if bool(args.dry_run):
            selected = ORACLE_SCENARIOS[0]
            obs = _apply_scenario(env, wrapped_env, selected.behavior)
            output = policy(obs)
            target = _target_tensor(selected, num_envs=env.num_envs, device=device)
            loss = torch.nn.functional.mse_loss(output, target)
            return {
                "status": "dry_run",
                "load_path": str(load_path),
                "output": str(args.output),
                "loss": float(loss.detach().cpu().item()),
                "policy_output_shape": list(output.shape),
                "rollout_actions": str(args.rollout_actions),
            }
        for iteration in range(int(args.iterations)):
            if bool(args.balanced_batch):
                optimizer.zero_grad(set_to_none=True)
                iteration_losses: list[float] = []
                normalizer = float(sum(scenario.weight for scenario in scenarios))
                normalizer *= float(max(int(args.rollout_steps), 1))
                for scenario in _shuffled_scenarios(rng, scenarios):
                    scenario_counts[scenario.name] += 1
                    obs = _apply_scenario(env, wrapped_env, scenario.behavior)
                    target = _target_tensor(scenario, num_envs=env.num_envs, device=device)
                    for _step in range(int(args.rollout_steps)):
                        prediction = policy(obs)
                        raw_loss = torch.nn.functional.mse_loss(prediction, target)
                        loss = raw_loss * float(scenario.weight) / normalizer
                        loss.backward()
                        raw_loss_value = float(raw_loss.detach().cpu().item())
                        iteration_losses.append(raw_loss_value)
                        loss_history.append(raw_loss_value)
                        with torch.no_grad():
                            step_action = (
                                prediction.detach()
                                if args.rollout_actions == "policy"
                                else target
                            )
                            obs, _rewards, dones, _infos = wrapped_env.step(step_action)
                            if bool(torch.any(dones).item()):
                                obs = _apply_scenario(env, wrapped_env, scenario.behavior)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                if args.progress_interval > 0 and (
                    (iteration + 1) % int(args.progress_interval) == 0
                ):
                    mean_loss = float(np.mean(iteration_losses)) if iteration_losses else 0.0
                    max_loss = float(np.max(iteration_losses)) if iteration_losses else 0.0
                    print(
                        json.dumps(
                            {
                                "iteration": iteration + 1,
                                "iterations": int(args.iterations),
                                "loss_iteration_mean": mean_loss,
                                "loss_iteration_max": max_loss,
                                "scenario_counts": scenario_counts,
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                continue

            scenario = scenarios[int(rng.choice(len(scenarios), p=probabilities))]
            scenario_counts[scenario.name] += 1
            obs = _apply_scenario(env, wrapped_env, scenario.behavior)
            target = _target_tensor(scenario, num_envs=env.num_envs, device=device)
            for _step in range(int(args.rollout_steps)):
                prediction = policy(obs)
                loss = torch.nn.functional.mse_loss(prediction, target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                loss_history.append(float(loss.detach().cpu().item()))
                with torch.no_grad():
                    step_action = prediction.detach() if args.rollout_actions == "policy" else target
                    obs, _rewards, dones, _infos = wrapped_env.step(step_action)
                    if bool(torch.any(dones).item()):
                        obs = _apply_scenario(env, wrapped_env, scenario.behavior)
            if args.progress_interval > 0 and (
                (iteration + 1) % int(args.progress_interval) == 0
            ):
                print(
                    json.dumps(
                        {
                            "iteration": iteration + 1,
                            "iterations": int(args.iterations),
                            "loss_recent": float(loss_history[-1]),
                            "scenario_counts": scenario_counts,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        output = Path(args.output)
        _save_corrected_checkpoint(runner, output)
        return {
            "status": "completed",
            "load_path": str(load_path),
            "output": str(output),
            "iterations": int(args.iterations),
            "rollout_steps": int(args.rollout_steps),
            "num_envs": int(args.num_envs),
            "learning_rate": float(args.learning_rate),
            "rollout_actions": str(args.rollout_actions),
            "balanced_batch": bool(args.balanced_batch),
            "scenarios": [scenario.name for scenario in scenarios],
            "scenario_counts": scenario_counts,
            "loss_initial": loss_history[0] if loss_history else None,
            "loss_final": loss_history[-1] if loss_history else None,
            "loss_mean_last_20": (
                float(np.mean(loss_history[-20:])) if len(loss_history) >= 20 else None
            ),
            "wall_time_sec": time.time() - started_at,
        }
    finally:
        env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = train_wall_slide_bc(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
