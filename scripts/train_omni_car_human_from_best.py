#!/usr/bin/env python3
"""Continue OmniCar PPO training from best.pt with Xbox human-command input."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unilab import cli


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resume OmniCar CNN-GRU PPO training from best.pt with a live Xbox-controlled "
            "command agent."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/omni_car/checkpoints/best.pt",
        help="Checkpoint path to resume from. Defaults to artifacts/omni_car/checkpoints/best.pt.",
    )
    parser.add_argument("--run-name", default="local_xbox_human_resume_best")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps-per-env", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=260)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--learning-epochs", type=int, default=1)
    parser.add_argument("--mini-batches", type=int, default=1)
    parser.add_argument("--replay-fanout", type=int, default=0)
    parser.add_argument("--joystick-index", type=int, default=0)
    parser.add_argument("--render-every-steps", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides. Prefix with -- before the first override.",
    )
    return parser.parse_args(argv)


def _normalize_overrides(raw: Sequence[str]) -> list[str]:
    items = [str(item) for item in raw]
    if items and items[0] == "--":
        items = items[1:]
    return items


def _resolve_checkpoint(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return ROOT_DIR / path


def build_resume_command(args: argparse.Namespace) -> list[str]:
    checkpoint = _resolve_checkpoint(str(args.checkpoint))
    if not checkpoint.is_file() and not args.dry_run:
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")

    overrides = [
        "training.logger=tensorboard",
        "training.no_play=true",
        f"algo.load_run={checkpoint}",
        f"algo.run_name={args.run_name}",
        f"algo.num_envs={int(args.num_envs)}",
        f"algo.num_steps_per_env={int(args.num_steps_per_env)}",
        f"algo.max_iterations={int(args.max_iterations)}",
        f"algo.save_interval={int(args.save_interval)}",
        f"algo.algorithm.num_learning_epochs={int(args.learning_epochs)}",
        f"algo.algorithm.num_mini_batches={int(args.mini_batches)}",
        "env.human_command.enabled=true",
        "env.human_command.backend=pygame",
        f"env.human_command.joystick_index={int(args.joystick_index)}",
        f"env.human_command.replay_fanout={int(args.replay_fanout)}",
        "env.human_command.render_enabled=true",
        f"env.human_command.render_every_steps={int(args.render_every_steps)}",
        *_normalize_overrides(args.overrides),
    ]
    if args.device is not None:
        overrides.append(f"training.device={args.device}")

    return cli.build_command(
        mode="train",
        algo="ppo",
        task="omni_car_grid_avoidance",
        sim="mujoco",
        overrides=overrides,
        render_mode="none",
        root=ROOT_DIR,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    command = build_resume_command(args)
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
