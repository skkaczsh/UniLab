#!/usr/bin/env python3
"""Quantitatively evaluate an OmniCar PPO checkpoint without opening a viewer."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.train_rsl_rl as train_rsl_rl


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-run", required=True, help="Run dir name, run dir path, or checkpoint path."
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint id or filename, e.g. 93 or model_93.pt."
    )
    parser.add_argument(
        "--num-steps", type=int, default=512, help="Evaluation horizon in control steps."
    )
    parser.add_argument(
        "--num-envs", type=int, default=64, help="Parallel env count for evaluation."
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device. Defaults to the training helper auto-selection.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Evaluation seed override.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args(argv)


@dataclass
class OmniCarEvalAccumulator:
    num_envs: int
    total_steps: int = 0
    total_reward: float = 0.0
    total_collisions: float = 0.0
    total_abs_tracking_error: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    total_sq_tracking_error: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    total_abs_action: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    total_abs_command: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    current_returns: np.ndarray = field(init=False)
    current_lengths: np.ndarray = field(init=False)
    completed_returns: list[float] = field(default_factory=list)
    completed_lengths: list[int] = field(default_factory=list)
    min_clearance_observed: float = float("inf")
    summed_logs: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.current_returns = np.zeros((self.num_envs,), dtype=np.float64)
        self.current_lengths = np.zeros((self.num_envs,), dtype=np.int32)

    def update(
        self,
        *,
        rewards: np.ndarray,
        dones: np.ndarray,
        commands: np.ndarray,
        actions: np.ndarray,
        collisions: np.ndarray,
        nearest_clearance: np.ndarray,
        step_logs: dict[str, float],
    ) -> None:
        rewards = np.asarray(rewards, dtype=np.float64).reshape(self.num_envs)
        dones = np.asarray(dones, dtype=bool).reshape(self.num_envs)
        commands = np.asarray(commands, dtype=np.float64).reshape(self.num_envs, 3)
        actions = np.asarray(actions, dtype=np.float64).reshape(self.num_envs, 3)
        collisions = np.asarray(collisions, dtype=np.float64).reshape(self.num_envs)
        nearest_clearance = np.asarray(nearest_clearance, dtype=np.float64).reshape(self.num_envs)

        tracking_error = actions - commands
        self.total_steps += 1
        self.total_reward += float(np.sum(rewards))
        self.total_collisions += float(np.sum(collisions))
        self.total_abs_tracking_error += np.sum(np.abs(tracking_error), axis=0)
        self.total_sq_tracking_error += np.sum(np.square(tracking_error), axis=0)
        self.total_abs_action += np.sum(np.abs(actions), axis=0)
        self.total_abs_command += np.sum(np.abs(commands), axis=0)
        self.min_clearance_observed = min(
            self.min_clearance_observed, float(np.min(nearest_clearance))
        )

        self.current_returns += rewards
        self.current_lengths += 1
        done_ids = np.flatnonzero(dones)
        if done_ids.size:
            self.completed_returns.extend(self.current_returns[done_ids].tolist())
            self.completed_lengths.extend(self.current_lengths[done_ids].astype(int).tolist())
            self.current_returns[done_ids] = 0.0
            self.current_lengths[done_ids] = 0

        for key, value in step_logs.items():
            self.summed_logs[key] = self.summed_logs.get(key, 0.0) + float(value)

    def finalize(self) -> dict[str, float | int | list[float] | None]:
        sample_count = max(self.total_steps * self.num_envs, 1)
        summary: dict[str, float | int | list[float] | None] = {
            "num_envs": int(self.num_envs),
            "num_steps": int(self.total_steps),
            "env_steps": int(sample_count),
            "mean_step_reward": float(self.total_reward / sample_count),
            "collision_fraction": float(self.total_collisions / sample_count),
            "min_clearance_observed": (
                float(self.min_clearance_observed)
                if np.isfinite(self.min_clearance_observed)
                else None
            ),
            "episodes_completed": int(len(self.completed_returns)),
            "mean_episode_return": (
                float(np.mean(self.completed_returns)) if self.completed_returns else None
            ),
            "mean_episode_length": (
                float(np.mean(self.completed_lengths)) if self.completed_lengths else None
            ),
            "vx_tracking_mae": float(self.total_abs_tracking_error[0] / sample_count),
            "vy_tracking_mae": float(self.total_abs_tracking_error[1] / sample_count),
            "vyaw_tracking_mae": float(self.total_abs_tracking_error[2] / sample_count),
            "vx_tracking_rmse": float(np.sqrt(self.total_sq_tracking_error[0] / sample_count)),
            "vy_tracking_rmse": float(np.sqrt(self.total_sq_tracking_error[1] / sample_count)),
            "vyaw_tracking_rmse": float(np.sqrt(self.total_sq_tracking_error[2] / sample_count)),
            "vx_action_mean_abs": float(self.total_abs_action[0] / sample_count),
            "vy_action_mean_abs": float(self.total_abs_action[1] / sample_count),
            "vyaw_action_mean_abs": float(self.total_abs_action[2] / sample_count),
            "vx_command_mean_abs": float(self.total_abs_command[0] / sample_count),
            "vy_command_mean_abs": float(self.total_abs_command[1] / sample_count),
            "vyaw_command_mean_abs": float(self.total_abs_command[2] / sample_count),
        }
        for key, value in sorted(self.summed_logs.items()):
            summary[key] = float(value / max(self.total_steps, 1))
        return summary


def _compose_cfg(args: argparse.Namespace) -> DictConfig:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT_DIR / "conf" / "ppo")):
        overrides = [
            "task=omni_car_grid_avoidance/mujoco",
            "training.play_only=true",
            "training.play_render_mode=none",
            f"algo.load_run={args.load_run}",
        ]
        if args.checkpoint is not None:
            overrides.append(f"algo.checkpoint={args.checkpoint}")
        return compose(config_name="config", overrides=overrides)


def _resolve_device(device: str | None) -> str:
    return str(device) if device is not None else str(train_rsl_rl.get_default_device())


def _format_summary(summary: dict[str, float | int | list[float] | None]) -> str:
    ordered_keys = [
        "num_envs",
        "num_steps",
        "env_steps",
        "mean_step_reward",
        "mean_episode_return",
        "mean_episode_length",
        "collision_fraction",
        "min_clearance_observed",
        "vx_tracking_mae",
        "vy_tracking_mae",
        "vyaw_tracking_mae",
        "vx_tracking_rmse",
        "vy_tracking_rmse",
        "vyaw_tracking_rmse",
        "omni_car/tracking_error",
        "omni_car/response_progress",
        "omni_car/command_clearance",
        "omni_car/clearance_risk",
        "omni_car/vx_track_cost",
        "omni_car/vy_track_cost",
        "omni_car/vyaw_track_cost",
        "omni_car/vx_diff_cost",
        "omni_car/vy_diff_cost",
        "omni_car/vyaw_diff_cost",
        "omni_car/vx_jerk_cost",
        "omni_car/vy_jerk_cost",
        "omni_car/vyaw_jerk_cost",
        "omni_car/collision_rate",
        "omni_car/mean_clearance",
        "omni_car/command_norm",
        "omni_car/reward/intent",
        "omni_car/reward/intent_projection",
        "omni_car/reward/clearance_motion",
        "omni_car/reward/clearance_target_motion",
        "omni_car/reward/blocked_projection",
        "omni_car/reward/blocked_stop",
        "omni_car/reward/off_axis",
        "omni_car/reward/reverse",
        "omni_car/reward/total",
    ]
    lines = []
    for key in ordered_keys:
        if key not in summary:
            continue
        value = summary[key]
        if isinstance(value, float):
            lines.append(f"{key}: {value:.6f}")
        else:
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, float | int | list[float] | None]:
    train_rsl_rl.ensure_registries()
    cfg = _compose_cfg(args)
    device = _resolve_device(args.device)
    rl_cfg = train_rsl_rl._algo_config_dict(cfg)
    wrapper_cls = train_rsl_rl._resolve_ppo_wrapper_cls(rl_cfg)
    env_cfg_override = train_rsl_rl.build_ppo_play_env_cfg_override(cfg)
    env_cfg_override["seed"] = int(args.seed)
    env = train_rsl_rl.create_env(
        cfg, num_envs=int(args.num_envs), env_cfg_override=env_cfg_override
    )
    wrapped_env = wrapper_cls(env, device=device)
    train_cfg = train_rsl_rl.normalize_ppo_train_cfg(rl_cfg)
    train_rsl_rl.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    if "runner" not in train_cfg:
        train_cfg["runner"] = {}
    train_cfg["runner"]["logger"] = "none"
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
    policy = runner.get_inference_policy(device=device)

    accumulator = OmniCarEvalAccumulator(num_envs=int(args.num_envs))
    obs, _ = wrapped_env.reset()
    try:
        with torch.inference_mode():
            for _ in range(int(args.num_steps)):
                actions = policy(obs)
                obs, rewards, dones, _infos = wrapped_env.step(actions)
                state = env.state
                if state is None:
                    raise RuntimeError(
                        "Environment state is unexpectedly unavailable during evaluation."
                    )
                accumulator.update(
                    rewards=rewards.detach().cpu().numpy(),
                    dones=dones.detach().cpu().numpy(),
                    commands=state.info["commands"],
                    actions=env._last_action,
                    collisions=state.info["collision"].astype(np.float32),
                    nearest_clearance=state.info["nearest_clearance"],
                    step_logs=state.info["log"],
                )
    finally:
        env.close()
    summary = accumulator.finalize()
    summary["checkpoint_path"] = str(load_path)
    summary["device"] = device
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.json:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = evaluate_checkpoint(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        summary = evaluate_checkpoint(args)
        print(_format_summary(summary))
        print("\nJSON:")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
