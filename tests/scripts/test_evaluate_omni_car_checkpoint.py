from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_omni_car_checkpoint.py"
    spec = importlib.util.spec_from_file_location("evaluate_omni_car_checkpoint", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_accumulator_tracks_axis_metrics_and_episode_stats() -> None:
    module = _load_module()
    acc = module.OmniCarEvalAccumulator(num_envs=2)

    acc.update(
        rewards=np.asarray([1.0, 2.0], dtype=np.float32),
        dones=np.asarray([False, True]),
        commands=np.asarray([[1.0, 0.0, 0.5], [0.0, -1.0, -0.5]], dtype=np.float32),
        actions=np.asarray([[0.5, 0.0, 0.0], [0.0, -0.5, -1.0]], dtype=np.float32),
        collisions=np.asarray([0.0, 1.0], dtype=np.float32),
        nearest_clearance=np.asarray([0.3, -0.1], dtype=np.float32),
        step_logs={
            "omni_car/tracking_error": 0.8,
            "omni_car/response_progress": 0.05,
            "omni_car/vx_diff_cost": 0.2,
        },
    )
    acc.update(
        rewards=np.asarray([3.0, 4.0], dtype=np.float32),
        dones=np.asarray([True, False]),
        commands=np.asarray([[1.0, 1.0, 0.0], [0.0, 0.5, 0.5]], dtype=np.float32),
        actions=np.asarray([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]], dtype=np.float32),
        collisions=np.asarray([0.0, 0.0], dtype=np.float32),
        nearest_clearance=np.asarray([0.2, 0.4], dtype=np.float32),
        step_logs={
            "omni_car/tracking_error": 0.4,
            "omni_car/response_progress": 0.10,
            "omni_car/vx_diff_cost": 0.1,
        },
    )

    summary = acc.finalize()
    assert summary["num_steps"] == 2
    assert summary["env_steps"] == 4
    assert summary["episodes_completed"] == 2
    assert summary["mean_step_reward"] == 2.5
    assert summary["collision_fraction"] == 0.25
    assert summary["min_clearance_observed"] == pytest.approx(-0.1)
    assert summary["mean_episode_return"] == 3.0
    assert summary["mean_episode_length"] == 1.5
    assert summary["vx_tracking_mae"] == 0.125
    assert summary["vy_tracking_mae"] == 0.375
    assert summary["vyaw_tracking_mae"] == 0.375
    assert summary["omni_car/tracking_error"] == pytest.approx(0.6)
    assert summary["omni_car/response_progress"] == pytest.approx(0.075)
    assert summary["omni_car/vx_diff_cost"] == pytest.approx(0.15)


def test_json_cli_suppresses_evaluator_noise(monkeypatch, capsys) -> None:
    module = _load_module()

    def _noisy_evaluate(_args):  # type: ignore[no-untyped-def]
        print("model debug noise")
        return {"checkpoint_path": "model.pt", "collision_fraction": 0.0}

    monkeypatch.setattr(module, "evaluate_checkpoint", _noisy_evaluate)

    rc = module.main(["--load-run", "model.pt", "--json"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "model debug noise" not in captured.out
    assert json.loads(captured.out) == {
        "checkpoint_path": "model.pt",
        "collision_fraction": 0.0,
    }


def test_compose_cfg_restores_run_config_actor_overrides(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "config": {
                    "algo": {
                        "actor": {
                            "command_skip_scale": 1.0,
                            "residual_action_scale": 0.6,
                            "residual_action_mode": "tanh",
                            "residual_action_limit": [0.45, 0.35, 0.6],
                            "transformer_activation": "gelu",
                            "transformer_dim": 192,
                            "transformer_ff_dim": 576,
                            "transformer_num_heads": 6,
                            "transformer_num_layers": 3,
                            "zero_residual_head": True,
                        },
                        "critic": {
                            "transformer_dim": 192,
                            "transformer_num_heads": 6,
                        },
                        "load_run": "-1",
                    },
                    "training": {
                        "play_only": False,
                        "play_render_mode": "human",
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    cfg = module._compose_cfg(
        module.argparse.Namespace(
            load_run=str(run_dir),
            checkpoint="60",
            actor_action_head_mode=None,
            actor_branch_hidden_dims=None,
            actor_action_gate_init_bias=None,
        )
    )

    assert OmegaConf.select(cfg, "algo.actor.command_skip_scale") == 1.0
    assert OmegaConf.select(cfg, "algo.actor.residual_action_scale") == 0.6
    assert OmegaConf.select(cfg, "algo.actor.residual_action_mode") == "tanh"
    assert OmegaConf.select(cfg, "algo.actor.residual_action_limit") == [0.45, 0.35, 0.6]
    assert OmegaConf.select(cfg, "algo.actor.transformer_activation") == "gelu"
    assert OmegaConf.select(cfg, "algo.actor.transformer_dim") == 192
    assert OmegaConf.select(cfg, "algo.actor.transformer_ff_dim") == 576
    assert OmegaConf.select(cfg, "algo.actor.transformer_num_heads") == 6
    assert OmegaConf.select(cfg, "algo.actor.transformer_num_layers") == 3
    assert OmegaConf.select(cfg, "algo.critic.transformer_dim") == 192
    assert OmegaConf.select(cfg, "algo.critic.transformer_num_heads") == 6
    assert OmegaConf.select(cfg, "algo.actor.zero_residual_head") is True
    assert OmegaConf.select(cfg, "algo.load_run") == str(run_dir)
    assert OmegaConf.select(cfg, "algo.checkpoint") == "60"
    assert OmegaConf.select(cfg, "training.play_only") is True
    assert OmegaConf.select(cfg, "training.play_render_mode") == "none"


def test_compose_cfg_can_override_actor_head_mode(tmp_path: Path) -> None:
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cfg = module._compose_cfg(
        module.argparse.Namespace(
            load_run=str(run_dir),
            checkpoint=None,
            actor_action_head_mode="gated_two_head",
            actor_branch_hidden_dims="32,16",
            actor_action_gate_init_bias=-0.5,
        )
    )

    assert OmegaConf.select(cfg, "algo.actor.action_head_mode") == "gated_two_head"
    assert OmegaConf.select(cfg, "algo.actor.branch_hidden_dims") == [32, 16]
    assert OmegaConf.select(cfg, "algo.actor.action_gate_init_bias") == -0.5
