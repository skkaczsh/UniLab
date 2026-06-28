from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "train_omni_car_human_from_best.py"
    )
    spec = importlib.util.spec_from_file_location("train_omni_car_human_from_best", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_human_resume_script_builds_low_latency_resume_command(tmp_path, monkeypatch) -> None:
    module = _load_module()
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.chdir(tmp_path)

    args = module._parse_args(
        [
            "--checkpoint",
            "best.pt",
            "--run-name",
            "manual_resume",
            "--num-envs",
            "6",
            "--num-steps-per-env",
            "7",
            "--learning-epochs",
            "1",
            "--mini-batches",
            "1",
            "--",
            "env.human_command.axis_vyaw=3",
        ]
    )
    command = module.build_resume_command(args)

    assert command[1] == str(Path(module.ROOT_DIR) / "scripts" / "train_rsl_rl.py")
    assert "task=omni_car_grid_avoidance/mujoco" in command
    assert f"algo.load_run={checkpoint}" in command
    assert "algo.run_name=manual_resume" in command
    assert "algo.num_envs=6" in command
    assert "algo.num_steps_per_env=7" in command
    assert "algo.algorithm.num_learning_epochs=1" in command
    assert "algo.algorithm.num_mini_batches=1" in command
    assert "env.human_command.enabled=true" in command
    assert "env.human_command.render_enabled=true" in command
    assert "env.human_command.require_joystick=true" in command
    assert "env.human_command.axis_vyaw=3" in command
    assert "training.play_render_mode=none" in command
    assert "env.human_command.axis_vyaw=3" in command


def test_human_resume_script_defaults_to_repo_best_checkpoint() -> None:
    module = _load_module()
    checkpoint = module.ROOT_DIR / "artifacts" / "omni_car" / "checkpoints" / "best.pt"

    args = module._parse_args(["--dry-run"])
    command = module.build_resume_command(args)

    assert f"algo.load_run={checkpoint}" in command


def test_human_resume_script_can_allow_missing_joystick_for_debug_viewer() -> None:
    module = _load_module()

    args = module._parse_args(["--dry-run", "--allow-no-joystick"])
    command = module.build_resume_command(args)

    assert "env.human_command.require_joystick=false" in command
