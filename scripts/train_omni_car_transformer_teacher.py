#!/usr/bin/env python3
"""Train an OmniCar deployable Transformer teacher with PPO."""

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

TRANSFORMER_CLASS = "unilab.algos.torch.omni_car:OmniCarGridCNNTransformerModel"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="omni_car_transformer_teacher")
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--num-steps-per-env", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=260)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--learning-epochs", type=int, default=2)
    parser.add_argument("--mini-batches", type=int, default=8)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--transformer-heads", type=int, default=8)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-ff-dim", type=int, default=768)
    parser.add_argument("--transformer-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-activation", choices=("gelu", "relu"), default="gelu")
    parser.add_argument("--device", default=None)
    parser.add_argument("--load-run", default=None, help="Optional Transformer checkpoint/run to resume.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dry-run", action="store_true")
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


def _transformer_overrides(args: argparse.Namespace, *, prefix: str) -> list[str]:
    return [
        f"{prefix}.class_name={TRANSFORMER_CLASS}",
        f"+{prefix}.transformer_dim={int(args.transformer_dim)}",
        f"+{prefix}.transformer_heads={int(args.transformer_heads)}",
        f"+{prefix}.transformer_layers={int(args.transformer_layers)}",
        f"+{prefix}.transformer_ff_dim={int(args.transformer_ff_dim)}",
        f"+{prefix}.transformer_dropout={float(args.transformer_dropout)}",
        f"+{prefix}.transformer_activation={args.transformer_activation}",
    ]


def build_train_command(args: argparse.Namespace) -> list[str]:
    overrides = [
        "training.logger=tensorboard",
        "training.no_play=true",
        f"algo.run_name={args.run_name}",
        f"algo.num_envs={int(args.num_envs)}",
        f"algo.num_steps_per_env={int(args.num_steps_per_env)}",
        f"algo.max_iterations={int(args.max_iterations)}",
        f"algo.save_interval={int(args.save_interval)}",
        f"algo.algorithm.num_learning_epochs={int(args.learning_epochs)}",
        f"algo.algorithm.num_mini_batches={int(args.mini_batches)}",
        *_transformer_overrides(args, prefix="algo.actor"),
        *_transformer_overrides(args, prefix="algo.critic"),
        *_normalize_overrides(args.overrides),
    ]
    if args.device is not None:
        overrides.append(f"training.device={args.device}")
    if args.load_run is not None:
        overrides.append(f"algo.load_run={args.load_run}")
    if args.checkpoint is not None:
        overrides.append(f"algo.checkpoint={args.checkpoint}")

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
    command = build_train_command(args)
    if args.dry_run:
        print(" ".join(shlex.quote(part) for part in command))
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
