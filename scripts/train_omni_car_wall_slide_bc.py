#!/usr/bin/env python3
"""Supervised wall-slide correction for OmniCar PPO actor checkpoints."""

from __future__ import annotations

import argparse
import json
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


@dataclass(frozen=True)
class OracleScenario:
    name: str
    behavior: BehaviorScenario
    target_action: tuple[float, float, float]
    weight: float = 1.0


ORACLE_SCENARIOS: tuple[OracleScenario, ...] = (
    OracleScenario(
        name="zero_input_hold",
        behavior=BehaviorScenario(name="zero_input_hold", command=(0.0, 0.0, 0.0)),
        target_action=(0.0, 0.0, 0.0),
        weight=2.0,
    ),
    OracleScenario(
        name="clear_forward_follow",
        behavior=BehaviorScenario(name="clear_forward_follow", command=(1.0, 0.0, 0.0)),
        target_action=(1.0, 0.0, 0.0),
        weight=1.5,
    ),
    OracleScenario(
        name="clear_diagonal_follow",
        behavior=BehaviorScenario(name="clear_diagonal_follow", command=(0.8, 0.4, 0.0)),
        target_action=(0.8, 0.4, 0.0),
        weight=1.5,
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
        weight=2.0,
    ),
    OracleScenario(
        name="right_wall_slide",
        behavior=BehaviorScenario(
            name="right_wall_slide",
            command=(1.0, 0.0, 0.0),
            obstacle_xy=((0.60, -0.34), (1.05, -0.34), (1.50, -0.34)),
            obstacle_radius=(0.22, 0.22, 0.22),
        ),
        target_action=(0.65, 0.28, 0.0),
        weight=4.0,
    ),
    OracleScenario(
        name="left_wall_slide",
        behavior=BehaviorScenario(
            name="left_wall_slide",
            command=(1.0, 0.0, 0.0),
            obstacle_xy=((0.60, 0.34), (1.05, 0.34), (1.50, 0.34)),
            obstacle_radius=(0.22, 0.22, 0.22),
        ),
        target_action=(0.65, -0.28, 0.0),
        weight=4.0,
    ),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Checkpoint path to correct.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint id/name.")
    parser.add_argument("--output", required=True, help="Output checkpoint path.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--rollout-steps", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _scenario_probabilities() -> np.ndarray:
    weights = np.asarray([scenario.weight for scenario in ORACLE_SCENARIOS], dtype=np.float64)
    return weights / np.sum(weights)


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
    probabilities = _scenario_probabilities()
    loss_history: list[float] = []
    scenario_counts = {scenario.name: 0 for scenario in ORACLE_SCENARIOS}
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
            }
        for _ in range(int(args.iterations)):
            scenario = ORACLE_SCENARIOS[int(rng.choice(len(ORACLE_SCENARIOS), p=probabilities))]
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
                    obs, _rewards, dones, _infos = wrapped_env.step(target)
                    if bool(torch.any(dones).item()):
                        obs = _apply_scenario(env, wrapped_env, scenario.behavior)
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
