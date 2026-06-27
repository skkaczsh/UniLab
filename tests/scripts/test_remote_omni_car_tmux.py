from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "remote_omni_car_tmux.py"
    spec = importlib.util.spec_from_file_location("remote_omni_car_tmux", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_train_command_uses_omni_car_defaults_and_extra_overrides() -> None:
    module = _load_module()

    command = module.build_train_command(
        run_name="remote_next",
        max_iterations=3000,
        num_envs=128,
        num_steps_per_env=32,
        logger="tensorboard",
        extra_overrides=["--", "algo.seed=7", "env.obstacles.count=20"],
    )

    assert command[:8] == [
        "uv",
        "run",
        "train",
        "--algo",
        "ppo",
        "--task",
        "omni_car_grid_avoidance",
        "--sim",
    ]
    assert "mujoco" in command
    assert "training.log_root=logs/remote_next" in command
    assert "algo.num_envs=128" in command
    assert "algo.num_steps_per_env=32" in command
    assert "algo.max_iterations=3000" in command
    assert "algo.seed=7" in command
    assert "env.obstacles.count=20" in command


def test_train_remote_script_exports_proxy_and_guards_existing_session() -> None:
    module = _load_module()
    plan = module.RemoteTmuxPlan(
        remote="zsh@skkac.top",
        ssh_port=6010,
        worktree="/home/zsh/develop/worktrees/UniLab-omni-car-git",
        session="omni car/train",
        proxy="http://127.0.0.1:7890",
    )

    script = module.build_train_remote_script(
        plan,
        ["uv", "run", "train", "algo.max_iterations=1"],
    )

    assert "tmux has-session" in script
    assert "tmux new-session -d -s" in script
    assert "HTTP_PROXY=http://127.0.0.1:7890" in script
    assert "logs/tmux/omni_car_train.log" in script
    assert "logs/tmux/omni_car_train.status" in script
    assert "state=running" in script
    assert "exit_code=${PIPESTATUS[0]}" in script
    assert "state=completed" in script
    assert "state=failed" in script
    assert "uv run train algo.max_iterations=1" in script


def test_status_remote_script_reports_git_gpu_and_tmux_tail() -> None:
    module = _load_module()
    plan = module.RemoteTmuxPlan(
        remote="zsh@skkac.top",
        ssh_port=6010,
        worktree="/remote/worktree",
        session="omni-car",
        proxy=None,
    )

    script = module.build_status_remote_script(plan, tail_lines=120)

    assert "git status --short --branch" in script
    assert "nvidia-smi --query-gpu" in script
    assert "tmux list-sessions" in script
    assert "tmux capture-pane -pt omni-car -S -120" in script
    assert "logs/tmux/omni-car.status" in script
    assert "last status" in script


def test_build_ssh_command_passes_remote_script_as_single_argument() -> None:
    module = _load_module()
    plan = module.RemoteTmuxPlan(
        remote="zsh@skkac.top",
        ssh_port=6010,
        worktree="/remote/worktree",
        session="omni-car",
        proxy=None,
    )

    command = module.build_ssh_command(plan, "echo ok")

    assert command == ["ssh", "-p", "6010", "zsh@skkac.top", "echo ok"]


def test_parser_rejects_rsl_rl_logger_values_that_fail_at_runtime() -> None:
    module = _load_module()

    with pytest.raises(SystemExit):
        module._parse_args(["train", "--logger", "none"])
