from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script = Path(__file__).resolve().parents[2] / "scripts" / "evaluate_omni_car_behaviors.py"
    spec = importlib.util.spec_from_file_location("evaluate_omni_car_behaviors", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_behavior_summary_reports_gate_failures() -> None:
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
            "command_safety_gate": [1.0, 1.0],
            "reward_total": [-1.0, -2.0],
            "reward_blocked_stop": [0.0, 0.0],
        },
    )

    assert summary["passed"] is False
    assert summary["planar_speed_mean"] == 0.11
    assert summary["failures"] == ["planar_speed_mean > 0.08"]


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
