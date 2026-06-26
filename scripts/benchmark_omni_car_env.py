#!/usr/bin/env python3
"""Benchmark the OmniCar grid-avoidance environment step loop."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab.base import registry


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=128, help="Parallel env count.")
    parser.add_argument("--steps", type=int, default=120, help="Timed control steps.")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps before timing.")
    parser.add_argument("--seed", type=int, default=7, help="Environment seed.")
    parser.add_argument(
        "--obstacles",
        type=int,
        default=None,
        help="Override obstacle count for the benchmark.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("zero", "command", "random"),
        default="zero",
        help="Action source used during the benchmark.",
    )
    parser.add_argument(
        "--profile-top",
        type=int,
        default=0,
        help="If > 0, print the top cumulative cProfile rows.",
    )
    parser.add_argument("--json", action="store_true", help="Print summary as JSON only.")
    return parser.parse_args(argv)


def _sample_actions(env, rng: np.random.Generator, action_mode: str) -> np.ndarray:  # type: ignore[no-untyped-def]
    if action_mode == "zero":
        return np.zeros((env.num_envs, 3), dtype=np.float32)
    if action_mode == "command":
        return env._commands.copy().astype(np.float32)
    high = np.asarray(env.action_space.high, dtype=np.float32)
    return rng.uniform(-high, high, size=(env.num_envs, 3)).astype(np.float32)


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, float | int | str], str | None]:
    registry.ensure_registries()
    env_cfg_override: dict[str, object] = {"seed": args.seed}
    if args.obstacles is not None:
        env_cfg_override["obstacles"] = {"count": args.obstacles}
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=env_cfg_override,
    )
    env.init_state()
    rng = np.random.default_rng(args.seed + 1)

    for _ in range(args.warmup_steps):
        env.step(_sample_actions(env, rng, args.action_mode))

    profiler: cProfile.Profile | None = None
    if args.profile_top > 0:
        profiler = cProfile.Profile()
        profiler.enable()
    start = time.perf_counter()
    for _ in range(args.steps):
        env.step(_sample_actions(env, rng, args.action_mode))
    elapsed = time.perf_counter() - start
    if profiler is not None:
        profiler.disable()
    env.close()

    env_steps = args.num_envs * args.steps
    summary: dict[str, float | int | str] = {
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "obstacles": int(args.obstacles) if args.obstacles is not None else int(env._cfg.obstacles.count),
        "action_mode": str(args.action_mode),
        "wall_time_s": float(elapsed),
        "per_step_ms": float(elapsed / max(args.steps, 1) * 1000.0),
        "env_steps_per_second": float(env_steps / max(elapsed, 1e-9)),
    }

    profile_text = None
    if profiler is not None:
        stream = io.StringIO()
        pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(args.profile_top)
        profile_text = stream.getvalue()
    return summary, profile_text


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    summary, profile_text = run_benchmark(args)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        if profile_text is not None:
            print()
            print(profile_text.rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
