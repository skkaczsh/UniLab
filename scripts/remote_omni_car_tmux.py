#!/usr/bin/env python3
"""Launch and inspect remote OmniCar training sessions through tmux."""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE = "zsh@skkac.top"
DEFAULT_SSH_PORT = 6010
DEFAULT_WORKTREE = "/home/zsh/develop/worktrees/UniLab-omni-car-git"
DEFAULT_SESSION = "omni-car-train"
DEFAULT_PROXY = "http://127.0.0.1:7890"
DEFAULT_UV_BIN = "/home/zsh/.local/bin/uv"


@dataclass(frozen=True)
class RemoteTmuxPlan:
    remote: str
    ssh_port: int
    worktree: str
    session: str
    proxy: str | None


def _normalize_remainder(items: Sequence[str]) -> list[str]:
    values = [str(item) for item in items]
    if values and values[0] == "--":
        values = values[1:]
    return values


def _safe_log_stem(session: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", session).strip("._")
    return safe or "omni-car-train"


def _proxy_exports(proxy: str | None) -> list[str]:
    if not proxy:
        return []
    quoted = shlex.quote(proxy)
    return [
        f"export HTTP_PROXY={quoted}",
        f"export HTTPS_PROXY={quoted}",
        f"export http_proxy={quoted}",
        f"export https_proxy={quoted}",
    ]


def build_train_command(
    *,
    run_name: str,
    max_iterations: int,
    num_envs: int,
    num_steps_per_env: int,
    logger: str,
    uv_bin: str = DEFAULT_UV_BIN,
    extra_overrides: Sequence[str] = (),
) -> list[str]:
    return [
        uv_bin,
        "run",
        "train",
        "--algo",
        "ppo",
        "--task",
        "omni_car_grid_avoidance",
        "--sim",
        "mujoco",
        f"training.logger={logger}",
        f"training.log_root=logs/{run_name}",
        f"algo.num_envs={num_envs}",
        f"algo.num_steps_per_env={num_steps_per_env}",
        f"algo.max_iterations={max_iterations}",
        *_normalize_remainder(extra_overrides),
    ]


def build_train_remote_script(plan: RemoteTmuxPlan, train_command: Sequence[str]) -> str:
    session = shlex.quote(plan.session)
    log_file = f"logs/tmux/{_safe_log_stem(plan.session)}.log"
    status_file = f"logs/tmux/{_safe_log_stem(plan.session)}.status"
    inner_lines = [
        "set -euo pipefail",
        *_proxy_exports(plan.proxy),
        f"cd {shlex.quote(plan.worktree)}",
        "mkdir -p logs/tmux",
        f"log_file={shlex.quote(log_file)}",
        f"status_file={shlex.quote(status_file)}",
        "started_at=$(date -Is)",
        'echo "[remote_omni_car_tmux] started $(date -Is)"',
        'echo "[remote_omni_car_tmux] commit $(git rev-parse HEAD)"',
        "printf 'state=running\\nstarted_at=%s\\ncommit=%s\\nlog=%s\\n' "
        '"$started_at" "$(git rev-parse HEAD)" "$log_file" > "$status_file"',
        "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
        "--format=csv,noheader || true",
        "set +e",
        f"{shlex.join(train_command)} 2>&1 | tee -a \"$log_file\"",
        "exit_code=${PIPESTATUS[0]}",
        "set -e",
        "ended_at=$(date -Is)",
        "if [ \"$exit_code\" -eq 0 ]; then state=completed; else state=failed; fi",
        'echo "[remote_omni_car_tmux] ${state} exit_code=${exit_code} ended_at=${ended_at}" '
        '| tee -a "$log_file"',
        "printf 'state=%s\\nstarted_at=%s\\nended_at=%s\\nexit_code=%s\\ncommit=%s\\nlog=%s\\n' "
        '"$state" "$started_at" "$ended_at" "$exit_code" "$(git rev-parse HEAD)" "$log_file" '
        '> "$status_file"',
        "exit \"$exit_code\"",
    ]
    inner = "\n".join(inner_lines)
    tmux_command = f"bash -lc {shlex.quote(inner)}"
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(plan.worktree)}",
            f"if tmux has-session -t {session} 2>/dev/null; then",
            f"  echo 'tmux session already exists: {plan.session}'",
            "  exit 2",
            "fi",
            f"tmux new-session -d -s {session} {shlex.quote(tmux_command)}",
            f"echo 'started tmux session: {plan.session}'",
            f"echo 'capture: tmux capture-pane -pt {plan.session} -S -80'",
            f"echo 'log: {log_file}'",
            f"echo 'status: {status_file}'",
        ]
    )


def build_status_remote_script(plan: RemoteTmuxPlan, *, tail_lines: int) -> str:
    session = shlex.quote(plan.session)
    tail = max(int(tail_lines), 1)
    status_file = f"logs/tmux/{_safe_log_stem(plan.session)}.status"
    return "\n".join(
        [
            "set -euo pipefail",
            f"cd {shlex.quote(plan.worktree)}",
            "echo '--- git ---'",
            "git status --short --branch",
            "git rev-parse HEAD",
            "echo '--- gpu ---'",
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total "
            "--format=csv,noheader || true",
            "echo '--- tmux sessions ---'",
            "tmux list-sessions 2>/dev/null || true",
            f"if tmux has-session -t {session} 2>/dev/null; then",
            f"  echo '--- {plan.session} tail ---'",
            f"  tmux capture-pane -pt {session} -S -{tail} || true",
            "else",
            f"  echo 'tmux session not found: {plan.session}'",
            f"  if test -f {shlex.quote(status_file)}; then",
            f"    echo '--- {plan.session} last status ---'",
            f"    cat {shlex.quote(status_file)}",
            "  fi",
            "fi",
        ]
    )


def build_ssh_command(plan: RemoteTmuxPlan, remote_script: str) -> list[str]:
    return ["ssh", "-p", str(plan.ssh_port), plan.remote, remote_script]


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--remote", default=DEFAULT_REMOTE, help="SSH target.")
    parser.add_argument("--ssh-port", type=int, default=DEFAULT_SSH_PORT, help="SSH port.")
    parser.add_argument("--worktree", default=DEFAULT_WORKTREE, help="Remote UniLab worktree.")
    parser.add_argument("--session", default=DEFAULT_SESSION, help="Remote tmux session name.")
    parser.add_argument(
        "--proxy",
        default=DEFAULT_PROXY,
        help="Remote Clash HTTP proxy exported for uv/network operations.",
    )
    parser.add_argument("--no-proxy", action="store_true", help="Do not export proxy variables.")
    parser.add_argument("--dry-run", action="store_true", help="Print the SSH command and script.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status", help="Inspect remote git/GPU/tmux state.")
    _add_common_args(status)
    status.add_argument("--tail-lines", type=int, default=80, help="Pane lines to capture.")

    train = subparsers.add_parser("train", help="Start OmniCar PPO training in a remote tmux session.")
    _add_common_args(train)
    train.add_argument("--run-name", default="remote_tmux_omni_car", help="training.log_root suffix.")
    train.add_argument("--max-iterations", type=int, default=3000, help="PPO learning iterations.")
    train.add_argument("--num-envs", type=int, default=128, help="Parallel env count.")
    train.add_argument("--num-steps-per-env", type=int, default=32, help="PPO rollout length.")
    train.add_argument(
        "--uv-bin",
        default=DEFAULT_UV_BIN,
        help="Remote uv executable used inside tmux.",
    )
    train.add_argument(
        "--logger",
        default="tensorboard",
        choices=("tensorboard", "wandb"),
        help="UniLab training logger accepted by the RSL-RL training path.",
    )
    train.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="Extra Hydra overrides, optionally after --.",
    )
    return parser.parse_args(argv)


def _plan_from_args(args: argparse.Namespace) -> RemoteTmuxPlan:
    proxy = None if bool(args.no_proxy) else str(args.proxy)
    return RemoteTmuxPlan(
        remote=str(args.remote),
        ssh_port=int(args.ssh_port),
        worktree=str(args.worktree),
        session=str(args.session),
        proxy=proxy,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = _plan_from_args(args)
    if args.command == "status":
        remote_script = build_status_remote_script(plan, tail_lines=int(args.tail_lines))
    elif args.command == "train":
        train_command = build_train_command(
            run_name=str(args.run_name),
            max_iterations=int(args.max_iterations),
            num_envs=int(args.num_envs),
            num_steps_per_env=int(args.num_steps_per_env),
            logger=str(args.logger),
            uv_bin=str(args.uv_bin),
            extra_overrides=args.overrides,
        )
        remote_script = build_train_remote_script(plan, train_command)
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(f"unsupported command: {args.command}")

    ssh_command = build_ssh_command(plan, remote_script)
    if bool(args.dry_run):
        print(" ".join(shlex.quote(part) for part in ssh_command))
        print("--- remote script ---")
        print(remote_script)
        return 0

    return subprocess.run(ssh_command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
