#!/usr/bin/env python3
"""Measure OmniCar command and command-grid input coverage."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import scripts.train_rsl_rl as train_rsl_rl

MODE_NAMES = {
    0: "zero",
    1: "vx",
    2: "vy",
    3: "vx_vy",
    4: "vyaw",
    5: "vx_vyaw",
    6: "vy_vyaw",
    7: "vx_vy_vyaw",
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel env count.")
    parser.add_argument("--num-steps", type=int, default=512, help="Rollout steps.")
    parser.add_argument(
        "--command-samples",
        type=int,
        default=8192,
        help="Standalone command samples used to inspect sampler balance.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("command", "zero"),
        default="command",
        help="Action used during rollout coverage. 'command' exposes command transitions.",
    )
    parser.add_argument(
        "--large-scene",
        choices=("config", "on", "off"),
        default="config",
        help="Override env.large_scene.enabled for coverage diagnostics.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Environment seed override.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args(argv)


def _compose_cfg() -> Any:
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT_DIR / "conf" / "ppo")):
        return compose(
            config_name="config",
            overrides=[
                "task=omni_car_grid_avoidance/mujoco",
                "training.no_play=true",
                "training.play_render_mode=none",
            ],
        )


def _mode_indices(commands: np.ndarray, deadband: float) -> np.ndarray:
    active = np.abs(commands) > float(deadband)
    return (
        active[:, 0].astype(np.int8)
        + 2 * active[:, 1].astype(np.int8)
        + 4 * active[:, 2].astype(np.int8)
    )


def _fraction_by_mode(commands: np.ndarray, deadband: float) -> dict[str, float]:
    mode_ids = _mode_indices(commands, deadband)
    total = max(int(mode_ids.size), 1)
    return {
        MODE_NAMES[mode_id]: float(np.count_nonzero(mode_ids == mode_id) / total)
        for mode_id in range(8)
    }


def _hist_fraction(values: np.ndarray, bins: Sequence[float]) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return [0.0 for _ in range(len(bins) - 1)]
    counts, _ = np.histogram(values, bins=np.asarray(bins, dtype=np.float64))
    return (counts / max(values.size, 1)).astype(float).tolist()


def summarize_commands(commands: np.ndarray, *, limits: np.ndarray, deadband: float) -> dict[str, Any]:
    commands = np.asarray(commands, dtype=np.float64)
    planar = commands[:, :2]
    planar_norm = np.linalg.norm(planar, axis=1)
    planar_active = planar_norm > float(deadband)
    normed_planar = planar / np.maximum(limits[:2], 1e-6)
    normed_planar_norm = np.linalg.norm(normed_planar, axis=1)
    yaw_abs_norm = np.abs(commands[:, 2]) / max(float(limits[2]), 1e-6)
    angle_bins = np.linspace(-np.pi, np.pi, 9)
    angles = np.arctan2(normed_planar[planar_active, 1], normed_planar[planar_active, 0])
    return {
        "sample_count": int(commands.shape[0]),
        "zero_fraction": float(np.mean(np.linalg.norm(commands, axis=1) <= float(deadband))),
        "mode_fraction": _fraction_by_mode(commands, deadband),
        "planar_active_fraction": float(np.mean(planar_active)),
        "planar_speed_band_fraction": {
            "low_0_035": _hist_fraction(normed_planar_norm[planar_active], [0.0, 0.35])[0],
            "mid_035_065": _hist_fraction(normed_planar_norm[planar_active], [0.35, 0.65])[0],
            "high_065_1": _hist_fraction(normed_planar_norm[planar_active], [0.65, 1.000001])[0],
        },
        "planar_angle_bin_fraction": _hist_fraction(angles, angle_bins),
        "yaw_abs_band_fraction": {
            "low_0_035": _hist_fraction(yaw_abs_norm, [0.0, 0.35])[0],
            "mid_035_065": _hist_fraction(yaw_abs_norm, [0.35, 0.65])[0],
            "high_065_1": _hist_fraction(yaw_abs_norm, [0.65, 1.000001])[0],
        },
        "mean_abs_command": np.mean(np.abs(commands), axis=0).astype(float).tolist(),
    }


def summarize_hold_steps(hold_steps: np.ndarray, *, ctrl_dt: float) -> dict[str, float]:
    durations = np.asarray(hold_steps, dtype=np.float64) * float(ctrl_dt)
    return {
        "mean_s": float(np.mean(durations)),
        "p10_s": float(np.quantile(durations, 0.10)),
        "p50_s": float(np.quantile(durations, 0.50)),
        "p90_s": float(np.quantile(durations, 0.90)),
        "long_ge_8s_fraction": float(np.mean(durations >= 8.0)),
        "very_long_ge_20s_fraction": float(np.mean(durations >= 20.0)),
    }


def _rollout_action(env: Any, mode: str) -> np.ndarray:
    if mode == "zero":
        return np.zeros((env.num_envs, 3), dtype=env._dtype)
    return env._commands.copy()


def rollout_coverage(env: Any, *, num_steps: int, action_mode: str) -> dict[str, Any]:
    env.init_state()
    commands: list[np.ndarray] = []
    gates: list[np.ndarray] = []
    clearances: list[np.ndarray] = []
    executed: list[np.ndarray] = []
    collisions: list[np.ndarray] = []
    for _ in range(int(num_steps)):
        action = _rollout_action(env, action_mode)
        state = env.step(action)
        commands.append(state.info["commands"].copy())
        gates.append(env._command_safety_gate.copy())
        clearances.append(env._command_clearance.copy())
        executed.append(state.info["executed_action"].copy())
        collisions.append(state.info["collision"].astype(np.float32).copy())

    command_arr = np.concatenate(commands, axis=0)
    gate_arr = np.concatenate(gates, axis=0)
    clearance_arr = np.concatenate(clearances, axis=0)
    executed_arr = np.concatenate(executed, axis=0)
    collision_arr = np.concatenate(collisions, axis=0)
    planar_active = np.linalg.norm(command_arr[:, :2], axis=1) > env._cfg.command.deadband
    zero = np.linalg.norm(command_arr, axis=1) <= env._cfg.command.deadband
    moving = np.linalg.norm(executed_arr, axis=1) > 0.10
    return {
        "sample_count": int(command_arr.shape[0]),
        "action_mode": action_mode,
        "command": summarize_commands(
            command_arr, limits=env._velocity_limit, deadband=env._cfg.command.deadband
        ),
        "command_safety_gate_mean": float(np.mean(gate_arr)),
        "command_safety_gate_band_fraction": {
            "blocked_0_02": _hist_fraction(gate_arr, [0.0, 0.2])[0],
            "partial_02_08": _hist_fraction(gate_arr, [0.2, 0.8])[0],
            "clear_08_1": _hist_fraction(gate_arr, [0.8, 1.000001])[0],
        },
        "command_clearance_mean": float(np.mean(clearance_arr)),
        "blocked_planar_fraction": float(np.mean(planar_active & (gate_arr < 0.2))),
        "partial_blocked_planar_fraction": float(
            np.mean(planar_active & (gate_arr >= 0.2) & (gate_arr < 0.8))
        ),
        "zero_with_motion_fraction": float(np.mean(zero & moving)),
        "collision_fraction": float(np.mean(collision_arr)),
    }


def analyze_coverage(args: argparse.Namespace) -> dict[str, Any]:
    train_rsl_rl.ensure_registries()
    cfg = _compose_cfg()
    env_cfg_override = copy.deepcopy(train_rsl_rl.build_ppo_env_cfg_override(cfg))
    env_cfg_override["seed"] = int(args.seed)
    if args.large_scene != "config":
        large_scene_override = env_cfg_override.setdefault("large_scene", {})
        large_scene_override["enabled"] = args.large_scene == "on"
    env = train_rsl_rl.create_env(
        cfg,
        num_envs=int(args.num_envs),
        env_cfg_override=env_cfg_override,
    )
    try:
        command_samples = env._sample_commands(int(args.command_samples))
        hold_steps = env._sample_command_hold_steps(int(args.command_samples))
        return {
            "seed": int(args.seed),
            "large_scene": args.large_scene,
            "num_envs": int(args.num_envs),
            "num_steps": int(args.num_steps),
            "command_samples": summarize_commands(
                command_samples,
                limits=env._velocity_limit,
                deadband=env._cfg.command.deadband,
            ),
            "hold_duration": summarize_hold_steps(hold_steps, ctrl_dt=env._cfg.ctrl_dt),
            "rollout": rollout_coverage(
                env, num_steps=int(args.num_steps), action_mode=str(args.action_mode)
            ),
        }
    finally:
        env.close()


def _format_summary(summary: dict[str, Any]) -> str:
    command = summary["command_samples"]
    rollout = summary["rollout"]
    lines = [
        f"seed: {summary['seed']}",
        f"large_scene: {summary['large_scene']}",
        f"num_envs: {summary['num_envs']}",
        f"num_steps: {summary['num_steps']}",
        f"command_zero_fraction: {command['zero_fraction']:.4f}",
        f"hold_long_ge_8s_fraction: {summary['hold_duration']['long_ge_8s_fraction']:.4f}",
        f"rollout_blocked_planar_fraction: {rollout['blocked_planar_fraction']:.4f}",
        f"rollout_partial_blocked_planar_fraction: {rollout['partial_blocked_planar_fraction']:.4f}",
        f"rollout_zero_with_motion_fraction: {rollout['zero_with_motion_fraction']:.4f}",
        f"rollout_collision_fraction: {rollout['collision_fraction']:.4f}",
        "mode_fraction:",
    ]
    lines.extend(
        f"  {name}: {value:.4f}" for name, value in command["mode_fraction"].items()
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = analyze_coverage(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(_format_summary(summary))
        print("\nJSON:")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
