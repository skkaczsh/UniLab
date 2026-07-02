#!/usr/bin/env python3
"""Inspect gated OmniCar actor branch outputs on oracle scenarios."""

from __future__ import annotations

import argparse
import contextlib
import json
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

from scripts.train_omni_car_wall_slide_bc import (  # noqa: E402
    ORACLE_SCENARIOS,
    _apply_scenario,
    _make_runner,
    _selected_scenarios,
    _training_behavior,
)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-run", required=True, help="Checkpoint path or run directory.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint id/name.")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--actor-action-head-mode",
        choices=("single", "gated_two_head", "risk_gated_two_head"),
        default="gated_two_head",
    )
    parser.add_argument("--actor-branch-hidden-dims", default=None)
    parser.add_argument("--actor-action-gate-init-bias", type=float, default=None)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(scenario.name for scenario in ORACLE_SCENARIOS),
        help="Limit diagnostics to one or more exact oracle scenarios.",
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
            "directional32",
            "directional32_clear",
            "directional32_front",
            "directional32_wall",
            "max_stick",
            "max_stick_clear",
            "max_stick_front",
            "max_stick_wall",
        ),
        help="Add a named oracle scenario group to inspect.",
    )
    parser.add_argument("--scenario-weight", action="append", default=None)
    parser.add_argument("--scenario-target", action="append", default=None)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--output", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _mean_vector(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().mean(dim=0).cpu().tolist()]


def diagnose_branches(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    with contextlib.redirect_stdout(sys.stderr):
        runner, env, wrapped_env, load_path, _device = _make_runner(args)
    policy = runner.alg.get_policy()
    policy.eval()
    branch_outputs = getattr(policy, "branch_action_outputs", None)
    if not callable(branch_outputs):
        env.close()
        raise ValueError("Loaded policy does not expose gated branch outputs")
    scenarios = _selected_scenarios(
        args.scenario,
        args.scenario_group,
        getattr(args, "scenario_weight", None),
        getattr(args, "scenario_target", None),
    )
    results: list[dict[str, Any]] = []
    try:
        wrapped_env.reset()
        with torch.no_grad():
            for scenario in scenarios:
                obs = _apply_scenario(
                    env,
                    wrapped_env,
                    _training_behavior(
                        scenario,
                        rng=rng,
                        xy_std=0.0,
                        radius_std=0.0,
                        yaw_std=0.0,
                    ),
                )
                combined_values: list[torch.Tensor] = []
                stop_values: list[torch.Tensor] = []
                escape_values: list[torch.Tensor] = []
                gate_values: list[torch.Tensor] = []
                for _step in range(max(int(args.num_steps), 1)):
                    combined = policy(obs)
                    outputs = branch_outputs(obs)
                    if outputs is None:
                        raise ValueError("Loaded policy returned no branch outputs")
                    stop_action, escape_action, gate = outputs
                    combined_values.append(combined.detach())
                    stop_values.append(stop_action.detach())
                    escape_values.append(escape_action.detach())
                    gate_values.append(gate.detach())
                    obs, _rewards, dones, _infos = wrapped_env.step(combined)
                    if bool(torch.any(dones).item()):
                        break
                combined_stack = torch.cat(combined_values, dim=0)
                stop_stack = torch.cat(stop_values, dim=0)
                escape_stack = torch.cat(escape_values, dim=0)
                gate_stack = torch.cat(gate_values, dim=0)
                results.append(
                    {
                        "scenario": scenario.name,
                        "target": list(scenario.target_action),
                        "combined_mean": _mean_vector(combined_stack),
                        "stop_mean": _mean_vector(stop_stack),
                        "escape_mean": _mean_vector(escape_stack),
                        "gate_mean": float(gate_stack.detach().mean().cpu().item()),
                        "gate_min": float(gate_stack.detach().min().cpu().item()),
                        "gate_max": float(gate_stack.detach().max().cpu().item()),
                    }
                )
    finally:
        env.close()
    return {
        "load_path": str(load_path),
        "num_envs": int(args.num_envs),
        "num_steps": int(args.num_steps),
        "results": results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = diagnose_branches(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for item in summary["results"]:
            print(
                "{scenario}: gate={gate_mean:.4f} combined={combined_mean} "
                "stop={stop_mean} escape={escape_mean}".format(**item)
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
