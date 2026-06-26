#!/usr/bin/env python3
"""Open the native MuJoCo viewer for a trained OmniCar checkpoint.

Usage:
    uv run scripts/view_omni_car_checkpoint.py \
      --load-run /absolute/path/to/run_dir \
      --checkpoint 93
"""

from __future__ import annotations

import argparse
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
        description="Open a native 3D viewer for OmniCar PPO checkpoints."
    )
    parser.add_argument(
        "--load-run",
        required=True,
        help="Run directory name, absolute run directory path, or checkpoint path.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint number like `93` or filename like `model_93.pt`. Ignored when --load-run points to a checkpoint file.",
    )
    parser.add_argument(
        "--device", default=None, help="Override playback device, e.g. cpu or cuda:0."
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides passed through to the eval/playback pipeline.",
    )
    return parser.parse_args(argv)


def _normalize_overrides(raw: Sequence[str]) -> list[str]:
    items = [str(item) for item in raw]
    if items and items[0] == "--":
        items = items[1:]
    return items


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    overrides = _normalize_overrides(args.overrides)
    if args.device is not None:
        overrides.append(f"training.device={args.device}")
    command = cli.build_command(
        mode="eval",
        algo="ppo",
        task="omni_car_grid_avoidance",
        sim="mujoco",
        overrides=overrides,
        load_run=str(args.load_run),
        checkpoint=args.checkpoint,
        render_mode="interactive",
        root=ROOT_DIR,
    )
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
