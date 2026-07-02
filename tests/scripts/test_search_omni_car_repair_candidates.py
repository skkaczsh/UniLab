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
        / "search_omni_car_repair_candidates.py"
    )
    spec = importlib.util.spec_from_file_location("search_omni_car_repair_candidates", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _gate(
    *,
    strict: bool,
    failed: int,
    clear: int = 47,
    front: int = 13,
    side: int = 15,
    collision: float = 0.0,
) -> dict[str, object]:
    return {
        "strict_passed": strict,
        "failed_count": failed,
        "category_summary": {
            "clear": {
                "passed": clear,
                "count": 48,
                "collision_fraction_max": collision,
                "projection_mean_min": 0.16,
            },
            "front_blocked": {
                "passed": front,
                "count": 16,
                "collision_fraction_max": collision,
                "projection_mean_min": -0.003,
            },
            "side_wall": {
                "passed": side,
                "count": 16,
                "collision_fraction_max": collision,
                "projection_mean_min": 0.11,
            },
            "yaw": {
                "passed": 2,
                "count": 2,
                "collision_fraction_max": 0.0,
                "projection_mean_min": 0.0,
            },
            "zero": {
                "passed": 1,
                "count": 1,
                "collision_fraction_max": 0.0,
                "projection_mean_min": 0.0,
            },
        },
    }


def test_build_candidates_orders_profiles_iterations_lrs_and_seeds() -> None:
    module = _load_module()

    candidates = module.build_candidates(
        profiles=["front", "front_wall"],
        seeds=[361, 362],
        learning_rates=[5.0e-5],
        iterations=[900],
        max_candidates=None,
        start_index=0,
    )

    assert [candidate["index"] for candidate in candidates] == [0, 1, 2, 3]
    assert [candidate["profile"] for candidate in candidates] == [
        "front",
        "front",
        "front_wall",
        "front_wall",
    ]
    assert candidates[0]["scenario_groups"] == [
        "base",
        "directional_clear",
        "directional_front",
    ]
    assert candidates[2]["scenario_groups"][-1] == "directional_wall"


def test_build_candidates_can_target_exact_max_stick_failures() -> None:
    module = _load_module()

    candidates = module.build_candidates(
        profiles=["max_stick_failures"],
        seeds=[401],
        learning_rates=[3.0e-5],
        iterations=[600],
    )

    assert candidates[0]["scenario_groups"] == ["base"]
    assert candidates[0]["scenarios"] == [
        "max_stick_front_blocked_dir_00_circle",
        "max_stick_front_blocked_dir_04_box",
        "max_stick_left_wall_dir_04",
    ]


def test_build_candidates_can_attach_profile_scenario_weights() -> None:
    module = _load_module()

    candidates = module.build_candidates(
        profiles=["max_stick_long_balance"],
        seeds=[541],
        learning_rates=[4.0e-6],
        iterations=[260],
    )

    assert "max_stick_right_wall_dir_07" in candidates[0]["scenarios"]
    assert "max_stick_front_blocked_dir_00_circle=2.0" in candidates[0][
        "scenario_weights"
    ]


def test_build_candidates_can_expand_target_sweeps() -> None:
    module = _load_module()

    target_sweeps = module._parse_target_sweeps(
        scenario="max_stick_right_wall_dir_07",
        vx_values="0.24,0.30",
        vy_values="-0.20,-0.16",
        vyaw_values="0.0",
    )
    candidates = module.build_candidates(
        profiles=["right_wall07_target_sweep"],
        seeds=[564],
        learning_rates=[5.0e-7],
        iterations=[4],
        target_sweeps=target_sweeps,
    )

    assert len(candidates) == 4
    assert candidates[0]["scenarios"] == [
        "max_stick_clear_dir_07",
        "max_stick_right_wall_dir_07",
        "max_stick_left_wall_dir_04",
    ]
    assert candidates[0]["scenario_targets"] == [
        "max_stick_right_wall_dir_07=0.24,-0.2,0"
    ]
    assert candidates[-1]["scenario_targets"] == [
        "max_stick_right_wall_dir_07=0.3,-0.16,0"
    ]


def test_build_candidates_can_resume_slice() -> None:
    module = _load_module()

    candidates = module.build_candidates(
        profiles=["front", "front_wall"],
        seeds=[361, 362],
        learning_rates=[5.0e-5],
        iterations=[900],
        max_candidates=2,
        start_index=1,
    )

    assert [candidate["index"] for candidate in candidates] == [1, 2]


def test_gate_score_prefers_fewer_failures_then_front_and_side_coverage() -> None:
    module = _load_module()

    worse_front = _gate(strict=False, failed=5, front=13, side=15)
    better_front = _gate(strict=False, failed=5, front=14, side=14)
    fewer_failures = _gate(strict=False, failed=4, front=13, side=13)

    assert module.gate_score(better_front) > module.gate_score(worse_front)
    assert module.gate_score(fewer_failures) > module.gate_score(better_front)


def test_gate_score_penalizes_collision_when_pass_counts_tie() -> None:
    module = _load_module()

    clean = _gate(strict=False, failed=5, front=13, side=15, collision=0.0)
    colliding = _gate(strict=False, failed=5, front=13, side=15, collision=0.01)

    assert module.gate_score(clean) > module.gate_score(colliding)


def test_gate_score_understands_max_stick_categories() -> None:
    module = _load_module()

    worse = {
        "strict_passed": False,
        "failed_count": 4,
        "category_summary": {
            "max_stick_clear": {"passed": 8, "count": 8, "collision_fraction_max": 0.0, "projection_mean_min": 0.7},
            "max_stick_front_blocked": {"passed": 5, "count": 8, "collision_fraction_max": 0.02, "projection_mean_min": -0.05},
            "max_stick_side_wall": {"passed": 15, "count": 16, "collision_fraction_max": 0.08, "projection_mean_min": 0.2},
        },
    }
    better = {
        "strict_passed": False,
        "failed_count": 3,
        "category_summary": {
            "max_stick_clear": {"passed": 8, "count": 8, "collision_fraction_max": 0.0, "projection_mean_min": 0.7},
            "max_stick_front_blocked": {"passed": 6, "count": 8, "collision_fraction_max": 0.02, "projection_mean_min": -0.05},
            "max_stick_side_wall": {"passed": 15, "count": 16, "collision_fraction_max": 0.08, "projection_mean_min": 0.2},
        },
    }

    assert module.gate_score(better) > module.gate_score(worse)


def test_search_dry_run_writes_candidate_manifest(tmp_path: Path) -> None:
    module = _load_module()

    result = module.run_search(
        module._parse_args(
            [
                "--load-run",
                "/tmp/init.pt",
                "--output",
                str(tmp_path / "manifest.json"),
                "--artifact-dir",
                str(tmp_path / "checkpoints"),
                "--name-prefix",
                "unit",
                "--profiles",
                "front",
                "--seeds",
                "361",
                "--learning-rates",
                "5e-5",
                "--iterations",
                "900",
                "--dry-run",
            ]
        )
    )

    assert result["candidate_count"] == 1
    assert result["best"]["name"].startswith("unit_c00_front")
    assert result["candidates"][0]["gate"]["strict_passed"] is False
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))[
        "candidate_count"
    ] == 1


def test_search_keeps_best_candidate_by_gate_score(monkeypatch, tmp_path: Path) -> None:
    module = _load_module()
    gate_calls = 0

    def _fake_run(command, check=True, capture_output=False, text=False):  # type: ignore[no-untyped-def]
        nonlocal gate_calls
        if capture_output:
            gate_calls += 1
            passed_front = 13 + gate_calls
            scenarios = [
                {"scenario": "front_blocked_dir_00_circle", "passed": False}
                for _ in range(16 - passed_front)
            ]
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=json.dumps(
                    {
                        "strict_passed": False,
                        "scenario_count": 83,
                        "category_summary": _gate(
                            strict=False,
                            failed=len(scenarios),
                            front=passed_front,
                        )["category_summary"],
                        "scenarios": scenarios,
                    }
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.sys.modules["scripts.run_omni_car_sgd_bc_curriculum"].subprocess, "run", _fake_run)

    result = module.run_search(
        module._parse_args(
            [
                "--load-run",
                "/tmp/init.pt",
                "--output",
                str(tmp_path / "manifest.json"),
                "--artifact-dir",
                str(tmp_path / "checkpoints"),
                "--name-prefix",
                "unit",
                "--profiles",
                "front",
                "--seeds",
                "361,362",
                "--learning-rates",
                "5e-5",
                "--iterations",
                "900",
            ]
        )
    )

    assert len(result["candidates"]) == 2
    assert result["best"]["name"].endswith("s362")


def test_search_forwards_target_sweep_and_closed_loop_options(
    monkeypatch, tmp_path: Path
) -> None:
    module = _load_module()
    train_commands: list[list[str]] = []

    def _fake_run_checked(command, *, dry_run):  # type: ignore[no-untyped-def]
        train_commands.append(list(command))

    def _fake_run_gate(command, *, output, dry_run):  # type: ignore[no-untyped-def]
        return {
            "strict_passed": False,
            "scenario_count": 1,
            "category_summary": {},
            "scenarios": [
                {"scenario": "max_stick_right_wall_dir_07", "passed": False}
            ],
        }

    monkeypatch.setattr(module, "_run_checked", _fake_run_checked)
    monkeypatch.setattr(module, "_run_gate", _fake_run_gate)

    module.run_search(
        module._parse_args(
            [
                "--load-run",
                "/tmp/base.pt",
                "--output",
                str(tmp_path / "manifest.json"),
                "--artifact-dir",
                str(tmp_path / "checkpoints"),
                "--name-prefix",
                "unit",
                "--profiles",
                "right_wall07_target_sweep",
                "--seeds",
                "564",
                "--learning-rates",
                "5e-7",
                "--iterations",
                "4",
                "--target-sweep-scenario",
                "max_stick_right_wall_dir_07",
                "--target-sweep-vx",
                "0.24",
                "--target-sweep-vy",
                "-0.2",
                "--rollout-actions",
                "policy",
                "--balanced-batch",
                "--dagger-replay-epochs",
                "1",
                "--scenario-jitter-xy-std",
                "0.03",
            ]
        )
    )

    command = train_commands[0]
    assert "--scenario-target" in command
    assert "max_stick_right_wall_dir_07=0.24,-0.2,0" in command
    assert "--balanced-batch" in command
    assert "--dagger-replay-epochs" in command
    assert "--scenario-jitter-xy-std" in command
