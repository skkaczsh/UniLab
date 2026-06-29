from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_omni_car_alternating_curriculum.py"
    )
    spec = importlib.util.spec_from_file_location("run_omni_car_alternating_curriculum", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _scenario(
    name: str,
    *,
    projection: float = 0.0,
    planar: float = 0.0,
    yaw: float = 0.0,
    collision: float = 0.0,
    passed: bool = False,
) -> dict[str, object]:
    return {
        "scenario": name,
        "projection_mean": projection,
        "planar_speed_mean": planar,
        "yaw_abs_mean": yaw,
        "collision_fraction": collision,
        "passed": passed,
    }


def _row(
    checkpoint: int,
    scenarios: list[dict[str, object]],
    *,
    passed: bool = False,
) -> dict[str, object]:
    return {
        "checkpoint": checkpoint,
        "checkpoint_path": f"/tmp/model_{checkpoint}.pt",
        "selection_score": float(checkpoint) / 1000.0,
        "behavior_gate": {
            "passed": bool(passed),
            "scenarios": scenarios,
        },
    }


def test_build_train_command_includes_phase_and_resume_overrides() -> None:
    module = _load_module()

    command = module.build_train_command(
        uv_bin="/opt/uv",
        run_name="auto_r00_safety",
        load_run="/tmp/model_100.pt",
        max_iterations=75,
        num_envs=128,
        num_steps_per_env=32,
        save_interval=25,
        phase=module.PHASES["safety"],
        extra_overrides=["algo.seed=7"],
    )

    assert command[:8] == [
        "/opt/uv",
        "run",
        "train",
        "--algo",
        "ppo",
        "--task",
        "omni_car_grid_avoidance",
        "--sim",
    ]
    assert "mujoco" in command
    assert "training.log_root=logs/auto_r00_safety" in command
    assert "algo.load_run=/tmp/model_100.pt" in command
    assert "algo.max_iterations=75" in command
    assert "algo.save_interval=25" in command
    assert "env.obstacles.front_blocker_fraction=0.55" in command
    assert "env.reward.intent_projection=65.0" in command
    assert "env.reward.blocked_lateral_escape=95.0" in command
    assert "algo.seed=7" in command


def test_build_scan_command_enables_behavior_gate() -> None:
    module = _load_module()

    command = module.build_scan_command(
        uv_bin="uv",
        run_dir=Path("/tmp/run"),
        checkpoints=[100, 125],
        output=Path("/tmp/scan.json"),
        num_envs=8,
        num_steps=96,
        seed=17,
        device="cpu",
        behavior_num_envs=16,
        behavior_num_steps=96,
        behavior_seed=19,
    )

    assert command[:3] == ["uv", "run", "scripts/scan_omni_car_checkpoints.py"]
    assert "--behavior-gate" in command
    assert "--checkpoints" in command
    assert "100" in command
    assert "125" in command
    assert "--device" in command
    assert "cpu" in command
    assert "/tmp/scan.json" in command


def test_select_checkpoint_prefers_global_behavior_coverage() -> None:
    module = _load_module()
    clear_good_front_bad = _row(
        100,
        [
            _scenario("zero_input_hold", planar=0.02, passed=True),
            _scenario("clear_forward_follow", projection=0.60, passed=True),
            _scenario("clear_diagonal_follow", projection=0.50, passed=True),
            _scenario("front_blocked_stop", projection=0.34, planar=0.34, collision=0.02),
            _scenario("right_wall_forward", projection=0.55, passed=True),
        ],
    )
    front_good_clear_bad = _row(
        200,
        [
            _scenario("zero_input_hold", planar=0.02, passed=True),
            _scenario("clear_forward_follow", projection=0.08),
            _scenario("clear_diagonal_follow", projection=0.06),
            _scenario("front_blocked_stop", projection=0.04, planar=0.05, collision=0.0, passed=True),
            _scenario("right_wall_forward", projection=0.08),
        ],
    )
    scan = {"evaluations": [clear_good_front_bad, front_good_clear_bad]}

    safety = module.select_checkpoint(scan, module.PHASES["safety"])
    clear = module.select_checkpoint(scan, module.PHASES["clear"])

    assert safety["checkpoint"] == 100
    assert clear["checkpoint"] == 100
    assert safety["behavior_pass_count"] == 4


def test_select_checkpoint_uses_phase_specific_tradeoffs_after_coverage() -> None:
    module = _load_module()
    clear_weighted = _row(
        100,
        [
            _scenario("zero_input_hold", planar=0.02, passed=True),
            _scenario("clear_forward_follow", projection=0.60, passed=True),
            _scenario("clear_diagonal_follow", projection=0.50, passed=True),
            _scenario("front_blocked_stop", projection=0.34, planar=0.34, collision=0.02),
            _scenario("right_wall_forward", projection=0.08),
        ],
    )
    safety_weighted = _row(
        200,
        [
            _scenario("zero_input_hold", planar=0.02, passed=True),
            _scenario("clear_forward_follow", projection=0.08),
            _scenario("clear_diagonal_follow", projection=0.06),
            _scenario("front_blocked_stop", projection=0.04, planar=0.05, collision=0.0, passed=True),
            _scenario("right_wall_forward", projection=0.25, passed=True),
        ],
    )
    scan = {"evaluations": [clear_weighted, safety_weighted]}

    safety = module.select_checkpoint(scan, module.PHASES["safety"])
    clear = module.select_checkpoint(scan, module.PHASES["clear"])

    assert safety["checkpoint"] == 200
    assert clear["checkpoint"] == 100
    assert safety["behavior_pass_count"] == 3
    assert clear["behavior_pass_count"] == 3


def test_select_checkpoint_prefers_all_gate_pass_when_available() -> None:
    module = _load_module()
    almost = _row(
        100,
        [
            _scenario("zero_input_hold", planar=0.01, passed=True),
            _scenario("clear_forward_follow", projection=0.60, passed=True),
            _scenario("clear_diagonal_follow", projection=0.50, passed=True),
            _scenario("front_blocked_stop", projection=0.10, planar=0.10, collision=0.01),
            _scenario("right_wall_forward", projection=0.50, passed=True),
        ],
    )
    passed = _row(
        200,
        [
            _scenario("zero_input_hold", planar=0.01, passed=True),
            _scenario("clear_forward_follow", projection=0.46, passed=True),
            _scenario("clear_diagonal_follow", projection=0.41, passed=True),
            _scenario("front_blocked_stop", projection=0.19, planar=0.20, collision=0.0, passed=True),
            _scenario("right_wall_forward", projection=0.21, passed=True),
        ],
        passed=True,
    )

    selected = module.select_checkpoint(
        {"evaluations": [almost, passed]},
        module.PHASES["balanced"],
    )

    assert selected["checkpoint"] == 200
    assert selected["passed_all_behavior_gates"] is True


def test_select_checkpoint_prefers_weighted_partial_passes_before_low_speed_collapse() -> None:
    module = _load_module()
    useful_but_front_bad = _row(
        2075,
        [
            _scenario("zero_input_hold", planar=0.07, passed=True),
            _scenario("clear_forward_follow", projection=0.70, planar=0.72, passed=True),
            _scenario("clear_diagonal_follow", projection=0.62, planar=0.65, passed=True),
            _scenario("front_blocked_stop", projection=0.31, planar=0.34, collision=0.07),
            _scenario("right_wall_forward", projection=0.34, planar=0.40, passed=True),
        ],
    )
    conservative_collapse = _row(
        2124,
        [
            _scenario("zero_input_hold", planar=0.05, passed=True),
            _scenario("clear_forward_follow", projection=0.07, planar=0.16),
            _scenario("clear_diagonal_follow", projection=0.16, planar=0.17),
            _scenario("front_blocked_stop", projection=0.07, planar=0.16, collision=0.0, passed=True),
            _scenario("right_wall_forward", projection=0.07, planar=0.16),
        ],
    )

    selected = module.select_checkpoint(
        {"evaluations": [useful_but_front_bad, conservative_collapse]},
        module.PHASES["balanced"],
    )

    assert selected["checkpoint"] == 2075
    assert selected["behavior_pass_count"] == 4
    assert selected["behavior_pass_score"] > 0
