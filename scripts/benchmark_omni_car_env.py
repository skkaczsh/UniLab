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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import MethodType
from typing import Any

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
        "--large-scene",
        action="store_true",
        help="Benchmark the shared large-scene mode used by the PPO owner config.",
    )
    parser.add_argument(
        "--scene-obstacles",
        type=int,
        default=260,
        help="Static obstacle count used with --large-scene.",
    )
    parser.add_argument(
        "--world-size",
        type=float,
        default=36.0,
        help="World size in meters used with --large-scene.",
    )
    parser.add_argument(
        "--max-local-static-obstacles",
        type=int,
        default=72,
        help="Nearest static obstacles rasterized per agent with --large-scene.",
    )
    parser.add_argument(
        "--max-dynamic-agents",
        type=int,
        default=24,
        help="Nearest dynamic agents rasterized per agent with --large-scene.",
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
    parser.add_argument(
        "--grid-workers",
        type=int,
        default=1,
        help=(
            "Use a ThreadPoolExecutor to split occupancy-grid fill across this "
            "many row chunks. 1 keeps the production serial/vectorized path."
        ),
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


def _install_threaded_grid_fill(env: Any, workers: int) -> ThreadPoolExecutor | None:
    if workers <= 1:
        return None

    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="omni-grid")
    serial_fill = env._fill_occupancy_grid

    def _threaded_fill(self, env_indices: np.ndarray, grid: np.ndarray) -> None:
        env_indices_arr = np.asarray(env_indices, dtype=np.int32)
        grid.fill(0.0)
        if env_indices_arr.size == 0:
            return
        if not self._large_scene_enabled and self._cfg.obstacles.count == 0:
            return
        row_chunks = [
            chunk
            for chunk in np.array_split(np.arange(env_indices_arr.size, dtype=np.int32), workers)
            if chunk.size > 0
        ]
        futures = [
            executor.submit(serial_fill, env_indices_arr[chunk], grid[chunk])
            for chunk in row_chunks
        ]
        for future in futures:
            future.result()

    env._fill_occupancy_grid = MethodType(_threaded_fill, env)
    return executor


def run_benchmark(args: argparse.Namespace) -> tuple[dict[str, float | int | str], str | None]:
    registry.ensure_registries()
    env_cfg_override: dict[str, object] = {"seed": args.seed}
    if args.obstacles is not None:
        env_cfg_override["obstacles"] = {"count": args.obstacles}
    if args.large_scene:
        env_cfg_override["large_scene"] = {
            "enabled": True,
            "world_size_m": args.world_size,
            "static_obstacle_count": args.scene_obstacles,
            "max_local_static_obstacles": args.max_local_static_obstacles,
            "max_dynamic_agents": args.max_dynamic_agents,
            "dense_region_count": 6,
            "dense_region_fraction": 0.58,
            "dense_region_radius_min_m": 2.0,
            "dense_region_radius_max_m": 5.2,
            "border_wall_segments_per_side": 28,
            "border_wall_thickness_m": 0.35,
            "agent_collision_radius_m": 0.34,
            "agent_spawn_keepout_m": 0.60,
            "reset_on_timeout": False,
            "stagnation_warmup_steps": 240,
            "stagnation_window_steps": 420,
            "stagnation_min_return_delta": 1.0,
            "resample_scene_on_full_reset": False,
        }
    env = registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=args.num_envs,
        env_cfg_override=env_cfg_override,
    )
    env.init_state()
    rng = np.random.default_rng(args.seed + 1)
    grid_workers = max(1, int(args.grid_workers))
    grid_executor = _install_threaded_grid_fill(env, grid_workers)

    try:
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
    finally:
        if grid_executor is not None:
            grid_executor.shutdown(wait=True)
        env.close()

    env_steps = args.num_envs * args.steps
    summary: dict[str, float | int | str] = {
        "num_envs": int(args.num_envs),
        "steps": int(args.steps),
        "warmup_steps": int(args.warmup_steps),
        "obstacles": int(args.obstacles) if args.obstacles is not None else int(env._cfg.obstacles.count),
        "large_scene": bool(args.large_scene),
        "scene_obstacles": int(env._scene_obstacle_count),
        "max_local_static_obstacles": int(env._cfg.large_scene.max_local_static_obstacles),
        "max_dynamic_agents": int(env._cfg.large_scene.max_dynamic_agents),
        "action_mode": str(args.action_mode),
        "grid_workers": grid_workers,
        "grid_mode": "threaded" if grid_workers > 1 else "serial",
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
