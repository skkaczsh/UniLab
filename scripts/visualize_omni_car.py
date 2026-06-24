"""Native 3D viewer for the OmniCarGridAvoidance task.

Usage:
    uv run scripts/visualize_omni_car.py --policy intent --steps 600
    uv run scripts/visualize_omni_car.py --policy reflex --seed 7
    uv run scripts/visualize_omni_car.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
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
from unilab.training import ensure_registries


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize OmniCarGridAvoidance in 3D.")
    parser.add_argument(
        "--policy",
        choices=["intent", "reflex", "random", "zero"],
        default="intent",
        help=(
            "Action source. 'intent' executes the user velocity command, 'reflex' "
            "slows/side-steps near obstacles, 'random' samples actions, and 'zero' holds still."
        ),
    )
    parser.add_argument("--steps", type=int, default=600, help="Viewer steps to run.")
    parser.add_argument("--seed", type=int, default=1, help="Environment RNG seed.")
    parser.add_argument("--obstacles", type=int, default=14, help="Number of circular obstacles.")
    parser.add_argument(
        "--max-speed", type=float, default=2.0, help="Per-axis command/action speed."
    )
    parser.add_argument(
        "--clearance-stop",
        type=float,
        default=0.25,
        help="Reflex policy starts slowing below this signed clearance in meters.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the env and print one action without opening the viewer.",
    )
    return parser.parse_args(argv)


def _make_env(args: argparse.Namespace):
    ensure_registries()
    return registry.make(
        "OmniCarGridAvoidance",
        sim_backend="mujoco",
        num_envs=1,
        env_cfg_override={
            "seed": int(args.seed),
            "command": {
                "max_x_speed": float(args.max_speed),
                "max_y_speed": float(args.max_speed),
                "max_yaw_rate": float(args.max_speed),
            },
            "obstacles": {"count": int(args.obstacles)},
        },
    )


def _select_action(env, args: argparse.Namespace) -> np.ndarray:
    if args.policy == "zero":
        return np.zeros((1, 3), dtype=np.float32)
    if args.policy == "random":
        return np.asarray([env.action_space.sample()], dtype=np.float32)

    assert env.state is not None
    command = np.asarray(env.state.info["commands"], dtype=np.float32).copy()
    if args.policy == "intent":
        return command

    clearance = float(np.asarray(env.state.info["nearest_clearance"])[0])
    scale = np.clip(clearance / max(float(args.clearance_stop), 1e-6), 0.0, 1.0)
    action = command.copy()
    action[:, :2] *= scale
    if scale < 0.8:
        # Side-step in body frame while preserving a small part of the yaw intent.
        action[:, 1] += np.sign(command[:, 1] + 1e-6) * float(args.max_speed) * (1.0 - scale)
        action[:, 2] *= 0.4 + 0.6 * scale
    return np.clip(action, env.action_space.low, env.action_space.high).astype(np.float32)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    env = _make_env(args)

    def initialize():
        return env.init_state()

    def step(_obs):
        action = _select_action(env, args)
        return env.step(action)

    obs = initialize()
    if args.dry_run:
        action = _select_action(env, args)
        print(f"obs={obs.obs['obs'].shape} action={action.tolist()} policy={args.policy}")
        env.close()
        return 0

    try:
        env.run_playback(
            initialize=lambda: obs,
            step=step,
            num_steps=max(1, int(args.steps)),
            camera_kwargs={"cam_distance": 5.0},
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
