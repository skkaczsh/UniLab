from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "train_omni_car_transformer_teacher.py"
    )
    spec = importlib.util.spec_from_file_location("train_omni_car_transformer_teacher", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transformer_teacher_script_builds_train_command() -> None:
    module = _load_module()

    args = module._parse_args(
        [
            "--run-name",
            "teacher_debug",
            "--num-envs",
            "32",
            "--num-steps-per-env",
            "16",
            "--max-iterations",
            "12",
            "--learning-epochs",
            "1",
            "--mini-batches",
            "4",
            "--transformer-dim",
            "128",
            "--transformer-heads",
            "4",
            "--transformer-layers",
            "2",
            "--transformer-ff-dim",
            "384",
            "--device",
            "cuda:0",
            "--",
            "algo.algorithm.learning_rate=3e-4",
        ]
    )
    command = module.build_train_command(args)

    assert command[1] == str(Path(module.ROOT_DIR) / "scripts" / "train_rsl_rl.py")
    assert "task=omni_car_grid_avoidance/mujoco" in command
    assert "training.play_render_mode=none" in command
    assert "algo.run_name=teacher_debug" in command
    assert "algo.num_envs=32" in command
    assert "algo.num_steps_per_env=16" in command
    assert "algo.max_iterations=12" in command
    assert "algo.algorithm.num_learning_epochs=1" in command
    assert "algo.algorithm.num_mini_batches=4" in command
    assert f"algo.actor.class_name={module.TRANSFORMER_CLASS}" in command
    assert f"algo.critic.class_name={module.TRANSFORMER_CLASS}" in command
    assert "+algo.actor.transformer_dim=128" in command
    assert "+algo.actor.transformer_heads=4" in command
    assert "+algo.actor.transformer_layers=2" in command
    assert "+algo.actor.transformer_ff_dim=384" in command
    assert "+algo.critic.transformer_dim=128" in command
    assert "training.device=cuda:0" in command
    assert "algo.algorithm.learning_rate=3e-4" in command


def test_transformer_teacher_script_can_resume_transformer_checkpoint() -> None:
    module = _load_module()

    args = module._parse_args(
        [
            "--load-run",
            "/tmp/teacher_run/model_10.pt",
            "--checkpoint",
            "10",
            "--dry-run",
        ]
    )
    command = module.build_train_command(args)

    assert "algo.load_run=/tmp/teacher_run/model_10.pt" in command
    assert "algo.checkpoint=10" in command
