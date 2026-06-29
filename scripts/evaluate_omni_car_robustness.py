#!/usr/bin/env python3
"""Broad OmniCar checkpoint gate over directions, speeds, yaw, and obstacle types."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import sys
from collections.abc import Sequence
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

from scripts.evaluate_omni_car_behaviors import (  # noqa: E402
    BehaviorScenario,
    _apply_scenario,
    _empty_record,
    _load_policy_and_env,
    _record_step,
    _summarize_record,
)

MAX_X_SPEED = 2.0
MAX_Y_SPEED = 1.0


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Run dir, checkpoint path, or run name.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint id/name.")
    parser.add_argument("--num-envs", type=int, default=16, help="Parallel envs per scenario.")
    parser.add_argument("--num-steps", type=int, default=192, help="Steps per scenario.")
    parser.add_argument("--directions", type=int, default=16, help="Planar command directions.")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default=None)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


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


def build_robustness_scenarios(directions: int = 16) -> tuple[BehaviorScenario, ...]:
    if directions < 4:
        raise ValueError("directions must be >= 4")
    scenarios: list[BehaviorScenario] = [
        BehaviorScenario(
            name="zero_input_hold_broad",
            command=(0.0, 0.0, 0.0),
            max_planar_speed=0.08,
            max_yaw_abs=0.08,
        ),
        BehaviorScenario(
            name="yaw_positive_hold_broad",
            command=(0.0, 0.0, 1.0),
            min_yaw_projection=0.35,
            max_planar_speed=0.10,
        ),
        BehaviorScenario(
            name="yaw_negative_hold_broad",
            command=(0.0, 0.0, -1.0),
            min_yaw_projection=0.35,
            max_planar_speed=0.10,
        ),
    ]
    angles = [2.0 * math.pi * i / directions for i in range(directions)]
    speed_fractions = (0.25, 0.55, 0.85)
    for angle_index, angle in enumerate(angles):
        for fraction in speed_fractions:
            command = _directional_command(angle, fraction)
            command_norm = float(np.linalg.norm(command[:2]))
            scenarios.append(
                BehaviorScenario(
                    name=f"clear_dir_{angle_index:02d}_speed_{fraction:.2f}",
                    command=command,
                    min_projection=max(0.10, 0.35 * command_norm),
                    min_projection_ratio=0.35,
                    max_off_axis_abs=0.30,
                )
            )
    for angle_index, angle in enumerate(angles):
        direction = _unit(angle)
        shape = ("circle", "box", "wall")[angle_index % 3]
        scenarios.append(
            BehaviorScenario(
                name=f"front_blocked_dir_{angle_index:02d}_{shape}",
                command=_directional_command(angle, 0.65),
                obstacle_xy=(_point(direction, 0.70, 0.0),),
                obstacle_radius=(0.22,),
                obstacle_type=(shape,),
                obstacle_half_extents=((0.22, 0.18),),
                obstacle_yaw=(_yaw_from_direction(direction, lateral_axis=True),),
                max_projection=0.20,
                max_planar_speed=0.35,
            )
        )
    wall_angles = angles[:: max(directions // 8, 1)]
    for angle_index, angle in enumerate(wall_angles):
        direction = _unit(angle)
        for side_name, lateral_offset in (("right", -0.34), ("left", 0.34)):
            obstacle_xy = tuple(
                _point(direction, forward, lateral_offset) for forward in (0.60, 1.05, 1.50)
            )
            scenarios.append(
                BehaviorScenario(
                    name=f"{side_name}_wall_dir_{angle_index:02d}",
                    command=_directional_command(angle, 0.65),
                    obstacle_xy=obstacle_xy,
                    obstacle_radius=(0.22, 0.22, 0.22),
                    obstacle_type=("circle", "wall", "circle"),
                    obstacle_half_extents=((0.22, 0.22), (0.36, 0.08), (0.22, 0.22)),
                    obstacle_yaw=(0.0, _yaw_from_direction(direction), 0.0),
                    min_projection=0.18,
                    max_collision_fraction=0.02,
                )
            )
    return tuple(scenarios)


def _category(name: str) -> str:
    if name.startswith("clear_"):
        return "clear"
    if name.startswith("front_blocked_"):
        return "front_blocked"
    if name.endswith("_wall_dir_00") or "_wall_dir_" in name:
        return "side_wall"
    if name.startswith("zero_"):
        return "zero"
    if name.startswith("yaw_"):
        return "yaw"
    return "other"


def _summarize_categories(scenarios: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    categories: dict[str, list[dict[str, Any]]] = {}
    for item in scenarios:
        categories.setdefault(_category(str(item["scenario"])), []).append(item)
    summary: dict[str, dict[str, float | int]] = {}
    for name, items in categories.items():
        summary[name] = {
            "count": len(items),
            "passed": sum(1 for item in items if bool(item["passed"])),
            "projection_mean_min": float(min(float(item["projection_mean"]) for item in items)),
            "projection_ratio_min": float(
                min(float(item["projection_ratio_mean"]) for item in items)
            ),
            "planar_speed_max": float(max(float(item["planar_speed_mean"]) for item in items)),
            "collision_fraction_max": float(
                max(float(item["collision_fraction"]) for item in items)
            ),
            "off_axis_abs_max": float(max(float(item["off_axis_abs_mean"]) for item in items)),
        }
    return summary


def evaluate_robustness(args: argparse.Namespace) -> dict[str, Any]:
    scenarios = build_robustness_scenarios(int(args.directions))
    policy, env, wrapped_env, checkpoint_path = _load_policy_and_env(args)
    scenario_summaries: list[dict[str, Any]] = []
    try:
        wrapped_env.reset()
        with torch.inference_mode():
            for scenario in scenarios:
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
    categories = _summarize_categories(scenario_summaries)
    return {
        "checkpoint_path": checkpoint_path,
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "directions": int(args.directions),
        "scenario_count": len(scenario_summaries),
        "strict_passed": all(bool(item["passed"]) for item in scenario_summaries),
        "category_summary": categories,
        "scenarios": scenario_summaries,
    }


def _format_summary(summary: dict[str, Any]) -> str:
    lines = [
        f"checkpoint_path: {summary['checkpoint_path']}",
        f"scenario_count: {summary['scenario_count']}",
        f"num_envs: {summary['num_envs']}",
        f"num_steps: {summary['num_steps']}",
        f"directions: {summary['directions']}",
        f"strict_passed: {summary['strict_passed']}",
    ]
    for category, item in sorted(summary["category_summary"].items()):
        lines.append(
            f"{category}: {item['passed']}/{item['count']} passed "
            f"min_proj={item['projection_mean_min']:.3f} "
            f"min_ratio={item['projection_ratio_min']:.3f} "
            f"max_planar={item['planar_speed_max']:.3f} "
            f"max_off_axis={item['off_axis_abs_max']:.3f} "
            f"max_collision={item['collision_fraction_max']:.3f}"
        )
    failures = [item for item in summary["scenarios"] if not bool(item["passed"])]
    if failures:
        lines.append("failures:")
        for item in failures:
            lines.append(
                f"  {item['scenario']}: proj={item['projection_mean']:.3f} "
                f"ratio={item['projection_ratio_mean']:.3f} "
                f"planar={item['planar_speed_mean']:.3f} "
                f"off_axis={item['off_axis_abs_mean']:.3f} "
                f"collision={item['collision_fraction']:.3f} "
                f"fail={', '.join(item['failures'])}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.json:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            summary = evaluate_robustness(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        summary = evaluate_robustness(args)
        print(_format_summary(summary))
        print("\nJSON:")
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if args.strict and not bool(summary["strict_passed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
