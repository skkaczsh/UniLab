#!/usr/bin/env python3
"""Probe OmniCar checkpoint behavior in controlled command/obstacle scenarios."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from collections.abc import Callable, Sequence
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


@dataclass(frozen=True)
class BehaviorScenario:
    name: str
    command: tuple[float, float, float]
    obstacle_xy: tuple[tuple[float, float], ...] = ()
    obstacle_radius: tuple[float, ...] = ()
    min_projection: float | None = None
    max_projection: float | None = None
    max_planar_speed: float | None = None
    max_yaw_abs: float | None = None
    max_collision_fraction: float = 0.0


SCENARIOS: tuple[BehaviorScenario, ...] = (
    BehaviorScenario(
        name="zero_input_hold",
        command=(0.0, 0.0, 0.0),
        max_planar_speed=0.08,
        max_yaw_abs=0.08,
    ),
    BehaviorScenario(
        name="clear_forward_follow",
        command=(1.0, 0.0, 0.0),
        min_projection=0.45,
    ),
    BehaviorScenario(
        name="clear_diagonal_follow",
        command=(0.8, 0.4, 0.0),
        min_projection=0.40,
    ),
    BehaviorScenario(
        name="front_blocked_stop",
        command=(1.0, 0.0, 0.0),
        obstacle_xy=((0.70, 0.0),),
        obstacle_radius=(0.22,),
        max_projection=0.20,
        max_planar_speed=0.35,
    ),
    BehaviorScenario(
        name="right_wall_forward",
        command=(1.0, 0.0, 0.0),
        obstacle_xy=((0.60, -0.34), (1.05, -0.34), (1.50, -0.34)),
        obstacle_radius=(0.22, 0.22, 0.22),
        min_projection=0.20,
        max_collision_fraction=0.02,
    ),
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--load-run", required=True, help="Run dir name, run dir path, or checkpoint path."
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Checkpoint id or filename, e.g. 93 or model_93.pt."
    )
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel envs per probe.")
    parser.add_argument("--num-steps", type=int, default=96, help="Steps per scenario.")
    parser.add_argument("--seed", type=int, default=11, help="Probe environment seed.")
    parser.add_argument("--device", default=None, help="Torch device override.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when gates fail.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    return parser.parse_args(argv)


def _make_local_probe_override(seed: int) -> dict[str, Any]:
    return {
        "seed": int(seed),
        "large_scene": {"enabled": False},
        "obstacles": {
            "count": 4,
            "circle_fraction": 1.0,
            "box_fraction": 0.0,
            "wall_fraction": 0.0,
        },
        "human_command": {"enabled": False, "render_enabled": False},
    }


def _load_policy_and_env(
    args: argparse.Namespace,
) -> tuple[Callable[[Any], torch.Tensor], Any, Any, str]:
    train_rsl_rl.ensure_registries()
    cfg = checkpoint_eval._compose_cfg(args)
    device = checkpoint_eval._resolve_device(args.device)
    rl_cfg = train_rsl_rl._algo_config_dict(cfg)
    wrapper_cls = train_rsl_rl._resolve_ppo_wrapper_cls(rl_cfg)
    env_cfg_override = train_rsl_rl.build_ppo_play_env_cfg_override(cfg)
    env_cfg_override.update(_make_local_probe_override(int(args.seed)))
    env = train_rsl_rl.create_env(
        cfg, num_envs=int(args.num_envs), env_cfg_override=env_cfg_override
    )
    wrapped_env = wrapper_cls(env, device=device)
    train_cfg = train_rsl_rl.normalize_ppo_train_cfg(rl_cfg)
    train_rsl_rl.apply_ppo_runtime_flags(train_cfg, cfg, training_enabled=False)
    train_cfg.setdefault("runner", {})
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
    return runner.get_inference_policy(device=device), env, wrapped_env, str(load_path)


def _apply_scenario(env: Any, wrapped_env: Any, scenario: BehaviorScenario) -> Any:
    env_ids = np.arange(env.num_envs, dtype=np.int32)
    command = np.asarray(scenario.command, dtype=env._dtype)
    env._pose[env_ids] = 0.0
    env._velocity[env_ids] = 0.0
    env._last_action[env_ids] = 0.0
    env._last_action_delta[env_ids] = 0.0
    env._raw_commands[env_ids] = command
    env._commands[env_ids] = command
    env._command_steps_remaining[env_ids] = int(1_000_000)
    env._seed_history(env_ids)

    env._obstacle_xy[env_ids] = np.asarray([10.0, 10.0], dtype=env._dtype)
    env._obstacle_radius[env_ids] = 0.10
    env._obstacle_type[env_ids] = env._OBSTACLE_CIRCLE
    env._obstacle_half_extents[env_ids] = 0.0
    env._obstacle_yaw[env_ids] = 0.0
    for obstacle_id, xy in enumerate(scenario.obstacle_xy):
        if obstacle_id >= env._cfg.obstacles.count:
            break
        env._obstacle_xy[:, obstacle_id] = np.asarray(xy, dtype=env._dtype)
        env._obstacle_radius[:, obstacle_id] = float(scenario.obstacle_radius[obstacle_id])
        env._obstacle_type[:, obstacle_id] = env._OBSTACLE_CIRCLE

    env._collision[env_ids] = False
    env._static_collision[env_ids] = False
    env._agent_collision[env_ids] = False
    env._border_collision[env_ids] = False
    env._stagnated[env_ids] = False
    env._nearest_clearance = env._compute_clearance(env_ids)
    env._grid_history_initialized[env_ids] = False
    obs = env._build_obs(env_ids)
    info = dict(env.state.info) if env.state is not None else {}
    info["commands"] = env._commands.copy()
    info["nearest_clearance"] = env._nearest_clearance.copy()
    info["collision"] = env._collision.copy()
    info["command_clearance"] = env._command_clearance.copy()
    info["clearance_risk"] = env._clearance_risk.copy()
    if env.state is not None:
        env._state = env.state.replace(obs=obs, info=info)
    return wrapped_env._obs_to_tensordict(obs, info)


def _empty_record() -> dict[str, list[float]]:
    return {
        "planar_speed": [],
        "action_vx": [],
        "action_vy": [],
        "action_vyaw": [],
        "yaw_abs": [],
        "projection": [],
        "off_axis_abs": [],
        "collision": [],
        "clearance_risk": [],
        "reward_total": [],
        "reward_clearance_motion": [],
        "reward_clearance_target_motion": [],
        "reward_clearance_opening": [],
        "reward_clearance_target_opening": [],
        "reward_blocked_lateral_escape": [],
        "reward_target_collision": [],
        "reward_blocked_projection": [],
        "reward_blocked_speed": [],
        "reward_blocked_stop": [],
        "reward_idle_stop": [],
        "reward_yaw_idle_stop": [],
    }


def _record_step(record: dict[str, list[float]], env: Any, step_info: dict[str, Any]) -> None:
    cmd = np.asarray(step_info["commands"], dtype=np.float64)
    action = np.asarray(step_info["executed_action"], dtype=np.float64)
    command_norm = np.linalg.norm(cmd[:, :2], axis=1)
    active = command_norm > env._cfg.command.deadband
    direction = np.zeros_like(cmd[:, :2])
    direction[active] = cmd[active, :2] / np.maximum(command_norm[active, None], 1e-6)
    projection = np.zeros((cmd.shape[0],), dtype=np.float64)
    projection[active] = np.sum(action[active, :2] * direction[active], axis=1)
    off_axis = np.abs(action[:, 0] * direction[:, 1] - action[:, 1] * direction[:, 0])

    record["planar_speed"].extend(np.linalg.norm(action[:, :2], axis=1).tolist())
    record["action_vx"].extend(action[:, 0].tolist())
    record["action_vy"].extend(action[:, 1].tolist())
    record["action_vyaw"].extend(action[:, 2].tolist())
    record["yaw_abs"].extend(np.abs(action[:, 2]).tolist())
    record["projection"].extend(projection.tolist())
    record["off_axis_abs"].extend(off_axis.tolist())
    record["collision"].extend(np.asarray(step_info["collision"], dtype=np.float32).tolist())
    record["clearance_risk"].extend(
        np.asarray(step_info["clearance_risk"], dtype=np.float64).tolist()
    )
    reward_components = step_info["reward_components"]
    record["reward_total"].extend(np.asarray(reward_components["total"]).tolist())
    record["reward_clearance_motion"].extend(
        np.asarray(reward_components["clearance_motion"]).tolist()
    )
    record["reward_clearance_target_motion"].extend(
        np.asarray(reward_components["clearance_target_motion"]).tolist()
    )
    clearance_opening = reward_components.get("clearance_opening")
    if clearance_opening is None:
        clearance_opening = np.zeros_like(reward_components["idle_stop"])
    record["reward_clearance_opening"].extend(np.asarray(clearance_opening).tolist())
    clearance_target_opening = reward_components.get("clearance_target_opening")
    if clearance_target_opening is None:
        clearance_target_opening = np.zeros_like(reward_components["idle_stop"])
    record["reward_clearance_target_opening"].extend(
        np.asarray(clearance_target_opening).tolist()
    )
    blocked_lateral_escape = reward_components.get("blocked_lateral_escape")
    if blocked_lateral_escape is None:
        blocked_lateral_escape = np.zeros_like(reward_components["idle_stop"])
    record["reward_blocked_lateral_escape"].extend(
        np.asarray(blocked_lateral_escape).tolist()
    )
    target_collision = reward_components.get("target_collision")
    if target_collision is None:
        target_collision = np.zeros_like(reward_components["idle_stop"])
    record["reward_target_collision"].extend(np.asarray(target_collision).tolist())
    blocked_projection = reward_components.get("blocked_projection")
    if blocked_projection is None:
        blocked_projection = np.zeros_like(reward_components["idle_stop"])
    record["reward_blocked_projection"].extend(np.asarray(blocked_projection).tolist())
    blocked_speed = reward_components.get("blocked_speed")
    if blocked_speed is None:
        blocked_speed = np.zeros_like(reward_components["idle_stop"])
    record["reward_blocked_speed"].extend(np.asarray(blocked_speed).tolist())
    blocked_stop = reward_components.get("blocked_stop")
    if blocked_stop is None:
        blocked_stop = np.zeros_like(reward_components["idle_stop"])
    record["reward_blocked_stop"].extend(np.asarray(blocked_stop).tolist())
    record["reward_idle_stop"].extend(np.asarray(reward_components["idle_stop"]).tolist())
    yaw_idle_stop = reward_components.get("yaw_idle_stop")
    if yaw_idle_stop is None:
        yaw_idle_stop = np.zeros_like(reward_components["idle_stop"])
    record["reward_yaw_idle_stop"].extend(np.asarray(yaw_idle_stop).tolist())


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _summarize_record(
    scenario: BehaviorScenario, record: dict[str, list[float]]
) -> dict[str, float | str | bool | list[str]]:
    summary: dict[str, float | str | bool | list[str]] = {
        "scenario": scenario.name,
        "planar_speed_mean": _mean(record["planar_speed"]),
        "action_vx_mean": _mean(record["action_vx"]),
        "action_vy_mean": _mean(record["action_vy"]),
        "action_vyaw_mean": _mean(record["action_vyaw"]),
        "yaw_abs_mean": _mean(record["yaw_abs"]),
        "projection_mean": _mean(record["projection"]),
        "off_axis_abs_mean": _mean(record["off_axis_abs"]),
        "collision_fraction": _mean(record["collision"]),
        "clearance_risk_mean": _mean(record["clearance_risk"]),
        "reward_total_mean": _mean(record["reward_total"]),
        "reward_clearance_motion_mean": _mean(record["reward_clearance_motion"]),
        "reward_clearance_target_motion_mean": _mean(
            record["reward_clearance_target_motion"]
        ),
        "reward_clearance_opening_mean": _mean(
            record.get("reward_clearance_opening", [])
        ),
        "reward_clearance_target_opening_mean": _mean(
            record.get("reward_clearance_target_opening", [])
        ),
        "reward_blocked_lateral_escape_mean": _mean(
            record.get("reward_blocked_lateral_escape", [])
        ),
        "reward_target_collision_mean": _mean(record.get("reward_target_collision", [])),
        "reward_blocked_projection_mean": _mean(record.get("reward_blocked_projection", [])),
        "reward_blocked_speed_mean": _mean(record.get("reward_blocked_speed", [])),
        "reward_blocked_stop_mean": _mean(record.get("reward_blocked_stop", [])),
        "reward_idle_stop_mean": _mean(record["reward_idle_stop"]),
        "reward_yaw_idle_stop_mean": _mean(record["reward_yaw_idle_stop"]),
    }
    failures: list[str] = []
    if scenario.min_projection is not None and summary["projection_mean"] < scenario.min_projection:
        failures.append(f"projection_mean < {scenario.min_projection}")
    if scenario.max_projection is not None and summary["projection_mean"] > scenario.max_projection:
        failures.append(f"projection_mean > {scenario.max_projection}")
    if scenario.max_planar_speed is not None and summary["planar_speed_mean"] > scenario.max_planar_speed:
        failures.append(f"planar_speed_mean > {scenario.max_planar_speed}")
    if scenario.max_yaw_abs is not None and summary["yaw_abs_mean"] > scenario.max_yaw_abs:
        failures.append(f"yaw_abs_mean > {scenario.max_yaw_abs}")
    if summary["collision_fraction"] > scenario.max_collision_fraction:
        failures.append(f"collision_fraction > {scenario.max_collision_fraction}")
    summary["passed"] = not failures
    summary["failures"] = failures
    return summary


def evaluate_behaviors(args: argparse.Namespace) -> dict[str, Any]:
    policy, env, wrapped_env, checkpoint_path = _load_policy_and_env(args)
    scenario_summaries: list[dict[str, float | str | bool | list[str]]] = []
    try:
        wrapped_env.reset()
        with torch.inference_mode():
            for scenario in SCENARIOS:
                obs = _apply_scenario(env, wrapped_env, scenario)
                record = _empty_record()
                for _ in range(int(args.num_steps)):
                    actions = policy(obs)
                    obs, _rewards, dones, _infos = wrapped_env.step(actions)
                    if env.state is None:
                        raise RuntimeError("Environment state is unavailable after step.")
                    _record_step(record, env, env.state.info)
                    if bool(torch.any(dones).item()):
                        obs = _apply_scenario(env, wrapped_env, scenario)
                scenario_summaries.append(_summarize_record(scenario, record))
    finally:
        env.close()
    return {
        "checkpoint_path": checkpoint_path,
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "strict_passed": all(bool(item["passed"]) for item in scenario_summaries),
        "scenarios": scenario_summaries,
    }


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"checkpoint_path: {summary['checkpoint_path']}",
        f"num_envs: {summary['num_envs']}",
        f"num_steps: {summary['num_steps']}",
        f"strict_passed: {summary['strict_passed']}",
    ]
    for item in summary["scenarios"]:
        status = "PASS" if item["passed"] else "FAIL"
        lines.append(
            f"{item['scenario']}: {status} "
            f"planar={item['planar_speed_mean']:.4f} "
            f"vx={item['action_vx_mean']:.4f} "
            f"vy={item['action_vy_mean']:.4f} "
            f"vyaw={item['action_vyaw_mean']:.4f} "
            f"yaw={item['yaw_abs_mean']:.4f} "
            f"proj={item['projection_mean']:.4f} "
            f"off_axis={item['off_axis_abs_mean']:.4f} "
            f"collision={item['collision_fraction']:.4f} "
            f"risk={item['clearance_risk_mean']:.4f} "
            f"r_clear_motion={item['reward_clearance_motion_mean']:.4f} "
            f"r_clear_target={item['reward_clearance_target_motion_mean']:.4f} "
            f"r_clear_open={item['reward_clearance_opening_mean']:.4f} "
            f"r_clear_target_open={item['reward_clearance_target_opening_mean']:.4f} "
            f"r_lat_escape={item['reward_blocked_lateral_escape_mean']:.4f} "
            f"r_target_collision={item['reward_target_collision_mean']:.4f} "
            f"r_blocked_proj={item['reward_blocked_projection_mean']:.4f} "
            f"r_blocked_speed={item['reward_blocked_speed_mean']:.4f} "
            f"r_blocked_stop={item['reward_blocked_stop_mean']:.4f} "
            f"r_idle={item['reward_idle_stop_mean']:.4f} "
            f"r_yaw_idle={item['reward_yaw_idle_stop_mean']:.4f}"
        )
        if item["failures"]:
            lines.append(f"  failures: {', '.join(item['failures'])}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.json:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = evaluate_behaviors(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        summary = evaluate_behaviors(args)
        print(_format_summary(summary))
        print("\nJSON:")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if args.strict and not bool(summary["strict_passed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
