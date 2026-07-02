from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_omni_car_sgd_bc_curriculum.py"
    )
    spec = importlib.util.spec_from_file_location("run_omni_car_sgd_bc_curriculum", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_bc_command_uses_per_scenario_sgd_defaults() -> None:
    module = _load_module()

    command = module.build_bc_command(
        uv_bin="uv",
        load_run="/tmp/model_0.pt",
        checkpoint=None,
        output="/tmp/init.pt",
        num_envs=64,
        iterations=2500,
        rollout_steps=2,
        rollout_actions="target",
        learning_rate=1.0e-4,
        seed=351,
        device="cuda:0",
        progress_interval=250,
    )

    assert command[:3] == ["uv", "run", "scripts/train_omni_car_wall_slide_bc.py"]
    assert "--balanced-batch" not in command
    assert "--load-run" in command
    assert "/tmp/model_0.pt" in command
    assert "--rollout-actions" in command
    assert "target" in command
    assert "--learning-rate" in command
    assert "0.0001" in command
    assert "--device" in command
    assert "cuda:0" in command


def test_repair_command_includes_front_focused_rehearsal_groups() -> None:
    module = _load_module()

    command = module.build_bc_command(
        uv_bin="uv",
        load_run="/tmp/init.pt",
        output="/tmp/repair.pt",
        num_envs=64,
        iterations=1400,
        rollout_steps=2,
        rollout_actions="target",
        learning_rate=5.0e-5,
        seed=361,
        scenario_groups=("base", "directional_clear", "directional_front"),
    )

    assert "--scenario-group" in command
    assert command.count("--scenario-group") == 3
    assert "base" in command
    assert "directional_clear" in command
    assert "directional_front" in command
    assert "directional_wall" not in command


def test_build_gate_command_uses_broad_json_gate() -> None:
    module = _load_module()

    command = module.build_gate_command(
        uv_bin="uv",
        load_run="/tmp/repair.pt",
        num_envs=16,
        num_steps=128,
        directions=16,
        seed=101,
        device="cuda:0",
    )

    assert command[:3] == ["uv", "run", "scripts/evaluate_omni_car_robustness.py"]
    assert "--json" in command
    assert "--strict" not in command
    assert "--directions" in command
    assert "16" in command
    assert "--suite" in command
    assert "broad" in command
    assert "--device" in command
    assert "cuda:0" in command


def test_build_gate_command_can_select_max_stick_suite() -> None:
    module = _load_module()

    command = module.build_gate_command(
        uv_bin="uv",
        load_run="/tmp/repair.pt",
        num_envs=4,
        num_steps=96,
        directions=8,
        seed=73,
        suite="max_stick",
    )

    assert "--suite" in command
    assert "max_stick" in command


def test_curriculum_writes_manifest_and_stops_after_passing_init_gate(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    commands: list[list[str]] = []

    def _fake_run(command, check=True, capture_output=False, text=False):  # type: ignore[no-untyped-def]
        commands.append(list(command))
        if capture_output:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "checkpoint_path": "/tmp/init.pt",
                        "strict_passed": True,
                        "scenario_count": 3,
                        "category_summary": {},
                        "scenarios": [
                            {"scenario": "zero_input_hold_broad", "passed": True},
                            {"scenario": "yaw_positive_hold_broad", "passed": True},
                            {"scenario": "clear_dir_00_speed_0.25", "passed": True},
                        ],
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    manifest_path = tmp_path / "manifest.json"

    result = module.run_curriculum(
        module._parse_args(
            [
                "--load-run",
                "/tmp/model_0.pt",
                "--output",
                str(manifest_path),
                "--artifact-dir",
                str(tmp_path),
                "--name-prefix",
                "unit",
                "--device",
                "cpu",
            ]
        )
    )

    assert result["completed"] is True
    assert result["completed_early"] is True
    assert len(result["stages"]) == 1
    assert len(result["gates"]) == 1
    assert commands[0][:3] == ["uv", "run", "scripts/train_omni_car_wall_slide_bc.py"]
    assert commands[1][:3] == ["uv", "run", "scripts/evaluate_omni_car_robustness.py"]
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["completed"] is True


def test_curriculum_runs_repair_when_init_gate_fails(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    commands: list[list[str]] = []
    gate_calls = 0

    def _fake_run(command, check=True, capture_output=False, text=False):  # type: ignore[no-untyped-def]
        nonlocal gate_calls
        commands.append(list(command))
        if capture_output:
            gate_calls += 1
            passed = gate_calls == 2
            scenarios = [{"scenario": "front_blocked_dir_00_circle", "passed": passed}]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "checkpoint_path": str(command[command.index("--load-run") + 1]),
                        "strict_passed": passed,
                        "scenario_count": 1,
                        "category_summary": {},
                        "scenarios": scenarios,
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, "run", _fake_run)

    result = module.run_curriculum(
        module._parse_args(
            [
                "--load-run",
                "/tmp/model_0.pt",
                "--output",
                str(tmp_path / "manifest.json"),
                "--artifact-dir",
                str(tmp_path),
                "--name-prefix",
                "unit",
            ]
        )
    )

    assert result["completed"] is True
    assert len(result["stages"]) == 2
    assert len(result["gates"]) == 2
    assert result["gates"][0]["failed_scenarios"] == ["front_blocked_dir_00_circle"]
    assert commands[2].count("--scenario-group") == 3
