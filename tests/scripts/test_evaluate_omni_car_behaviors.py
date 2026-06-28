from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_omni_car_behaviors.py"
    spec = importlib.util.spec_from_file_location("evaluate_omni_car_behaviors", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_behavior_summary_reports_threshold_failures() -> None:
    module = _load_module()
    scenario = module.BehaviorScenario(
        name="zero",
        command=(0.0, 0.0, 0.0),
        max_planar_speed=0.08,
        max_yaw_abs=0.08,
    )
    summary = module._summarize_record(
        scenario,
        {
            "planar_speed": [0.10, 0.12],
            "yaw_abs": [0.02, 0.04],
            "projection": [0.0, 0.0],
            "off_axis_abs": [0.0, 0.0],
            "collision": [0.0, 0.0],
            "clearance_risk": [0.0, 0.0],
            "reward_total": [-1.0, -2.0],
            "reward_clearance_motion": [-0.5, -1.5],
            "reward_clearance_target_motion": [-0.25, -0.75],
            "reward_clearance_opening": [0.5, 1.5],
            "reward_idle_stop": [-2.0, -4.0],
            "reward_yaw_idle_stop": [-6.0, -8.0],
        },
    )

    assert summary["passed"] is False
    assert summary["planar_speed_mean"] == 0.11
    assert summary["reward_clearance_motion_mean"] == -1.0
    assert summary["reward_clearance_target_motion_mean"] == -0.5
    assert summary["reward_clearance_opening_mean"] == 1.0
    assert summary["reward_idle_stop_mean"] == -3.0
    assert summary["reward_yaw_idle_stop_mean"] == -7.0
    assert summary["failures"] == ["planar_speed_mean > 0.08"]


def test_record_step_uses_step_info_snapshot() -> None:
    module = _load_module()
    env = SimpleNamespace(_cfg=SimpleNamespace(command=SimpleNamespace(deadband=0.1)))
    record = module._empty_record()
    step_info = {
        "commands": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        "executed_action": np.asarray([[0.25, 0.50, 0.10]], dtype=np.float32),
        "collision": np.asarray([True]),
        "clearance_risk": np.asarray([0.25], dtype=np.float32),
        "reward_components": {
            "total": np.asarray([-2.0], dtype=np.float32),
            "clearance_motion": np.asarray([-3.0], dtype=np.float32),
            "clearance_target_motion": np.asarray([-1.5], dtype=np.float32),
            "clearance_opening": np.asarray([1.25], dtype=np.float32),
            "idle_stop": np.asarray([-4.0], dtype=np.float32),
            "yaw_idle_stop": np.asarray([-5.0], dtype=np.float32),
        },
    }

    module._record_step(record, env, step_info)

    assert record["planar_speed"] == [np.hypot(0.25, 0.50)]
    assert record["yaw_abs"] == [0.10000000149011612]
    assert record["projection"] == [0.25]
    assert record["off_axis_abs"] == [0.5]
    assert record["collision"] == [1.0]
    assert record["clearance_risk"] == [0.25]
    assert record["reward_total"] == [-2.0]
    assert record["reward_clearance_motion"] == [-3.0]
    assert record["reward_clearance_target_motion"] == [-1.5]
    assert record["reward_clearance_opening"] == [1.25]
    assert record["reward_idle_stop"] == [-4.0]
    assert record["reward_yaw_idle_stop"] == [-5.0]


def test_json_cli_suppresses_behavior_evaluator_noise(monkeypatch, capsys) -> None:
    module = _load_module()

    def _noisy_evaluate(_args):  # type: ignore[no-untyped-def]
        print("model debug noise")
        return {
            "checkpoint_path": "model.pt",
            "num_envs": 1,
            "num_steps": 1,
            "strict_passed": True,
            "scenarios": [],
        }

    monkeypatch.setattr(module, "evaluate_behaviors", _noisy_evaluate)

    rc = module.main(["--load-run", "model.pt", "--json", "--strict"])

    assert rc == 0
    captured = capsys.readouterr()
    assert "model debug noise" not in captured.out
    assert json.loads(captured.out) == {
        "checkpoint_path": "model.pt",
        "num_envs": 1,
        "num_steps": 1,
        "strict_passed": True,
        "scenarios": [],
    }
